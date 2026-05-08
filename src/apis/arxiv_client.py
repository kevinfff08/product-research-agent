"""arXiv API client for preprint paper search."""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import httpx

from src.logging_config import get_logger

logger = get_logger("apis.arxiv")

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

# arXiv asks API clients to avoid parallel requests and to wait at least
# 3 seconds between repeated calls. In practice a research run can share an
# IP with other tooling, so we use a wider gap and a run-level cooldown after
# 429/timeout responses instead of letting every coroutine retry independently.
_MIN_INTERVAL = 10.0
_MAX_JITTER = 3.0
_REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=10.0)
_MAX_ATTEMPTS = 3
_RATE_LIMIT_COOLDOWN = 60.0
_TIMEOUT_COOLDOWN = 30.0
_CIRCUIT_BREAKER_SECONDS = 600.0
_MAX_CONSECUTIVE_RATE_LIMITS = 2
_MAX_CONSECUTIVE_TIMEOUTS = 3


class ArxivClient:
    """Client for arXiv API - preprint paper search.

    Uses an asyncio.Lock + mandatory delay to serialize all requests, plus
    run-level cooldown/circuit breaking when arXiv starts returning 429 or
    timing out. This is essential because multiple research paths launch
    concurrent queries that would otherwise cause a retry storm.
    """

    BASE_URL = "https://export.arxiv.org/api/query"
    USER_AGENT = "ProductResearch/0.2 (research agent; contact: local)"

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0
        self._rate_lock = asyncio.Lock()
        self._cooldown_until: float = 0.0
        self._circuit_open_until: float = 0.0
        self._consecutive_rate_limits = 0
        self._consecutive_timeouts = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=_REQUEST_TIMEOUT,
                follow_redirects=True,
                trust_env=False,
                headers={"User-Agent": self.USER_AGENT},
            )
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 20,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[dict]:
        """Search arXiv papers.  Serialises all callers through a lock."""
        params = {
            "search_query": f"all:{query}",
            "max_results": min(max_results, 50),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        response: httpx.Response | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            async with self._rate_lock:
                self._raise_if_circuit_open(query)
                await self._wait_for_slot()

                logger.info("arXiv search: query=%s, max=%d", query, max_results)
                client = await self._get_client()
                try:
                    response = await client.get(self.BASE_URL, params=params)
                except httpx.TimeoutException as exc:
                    self._last_request_time = time.monotonic()
                    wait = self._register_timeout(attempt, query)
                    logger.warning(
                        "arXiv timeout on attempt %d/%d for '%s', cooling down %.0fs: %r",
                        attempt, _MAX_ATTEMPTS, query[:60], wait, exc,
                    )
                    if attempt >= _MAX_ATTEMPTS:
                        raise
                    continue
                self._last_request_time = time.monotonic()

                if response.status_code == 429:
                    wait = self._register_rate_limit(response, attempt, query)
                    logger.warning(
                        "arXiv 429 on attempt %d/%d for '%s', cooling down %.0fs...",
                        attempt, _MAX_ATTEMPTS, query[:60], wait,
                    )
                    if attempt >= _MAX_ATTEMPTS:
                        break
                    continue
                if response.status_code >= 400:
                    response.raise_for_status()
                self._register_success()
                break

        if response is None:
            raise RuntimeError(f"arXiv search did not receive a response for: {query}")
        if response.status_code == 429:
            raise RuntimeError(f"arXiv rate limited after 3 attempts for: {query}")

        papers = self._parse_response(response.text)
        logger.info("arXiv returned %d papers for: %s", len(papers), query)
        return papers

    async def _wait_for_slot(self) -> None:
        now = time.monotonic()
        next_interval_time = self._last_request_time + _MIN_INTERVAL + random.uniform(0, _MAX_JITTER)
        next_allowed = max(next_interval_time, self._cooldown_until)
        wait = next_allowed - now
        if wait > 0:
            logger.info("arXiv client waiting %.0fs before next request", wait)
            await asyncio.sleep(wait)
        self._raise_if_circuit_open("pending arXiv query")

    def _register_rate_limit(
        self,
        response: httpx.Response,
        attempt: int,
        query: str,
    ) -> float:
        self._consecutive_rate_limits += 1
        retry_after = response.headers.get("retry-after", "")
        try:
            wait = float(retry_after)
        except ValueError:
            wait = _RATE_LIMIT_COOLDOWN * attempt + random.uniform(0, 10)
        now = time.monotonic()
        self._cooldown_until = max(self._cooldown_until, now + wait)
        if self._consecutive_rate_limits >= _MAX_CONSECUTIVE_RATE_LIMITS:
            self._circuit_open_until = max(
                self._circuit_open_until,
                now + _CIRCUIT_BREAKER_SECONDS,
            )
            logger.warning(
                "arXiv circuit opened for %.0fs after %d consecutive 429s; latest query='%s'",
                _CIRCUIT_BREAKER_SECONDS,
                self._consecutive_rate_limits,
                query[:80],
            )
        return wait

    def _register_timeout(self, attempt: int, query: str) -> float:
        self._consecutive_timeouts += 1
        wait = _TIMEOUT_COOLDOWN * attempt + random.uniform(0, 10)
        now = time.monotonic()
        self._cooldown_until = max(self._cooldown_until, now + wait)
        if self._consecutive_timeouts >= _MAX_CONSECUTIVE_TIMEOUTS:
            self._circuit_open_until = max(
                self._circuit_open_until,
                now + _CIRCUIT_BREAKER_SECONDS,
            )
            logger.warning(
                "arXiv circuit opened for %.0fs after %d consecutive timeouts; latest query='%s'",
                _CIRCUIT_BREAKER_SECONDS,
                self._consecutive_timeouts,
                query[:80],
            )
        return wait

    def _register_success(self) -> None:
        self._consecutive_rate_limits = 0
        self._consecutive_timeouts = 0

    def _raise_if_circuit_open(self, query: str) -> None:
        wait = self._circuit_open_until - time.monotonic()
        if wait > 0:
            raise RuntimeError(
                f"arXiv API temporarily disabled for this run after repeated rate limits/timeouts; "
                f"retry after about {wait:.0f}s. Query skipped: {query}"
            )

    async def download_pdf(
        self,
        arxiv_id_or_url: str,
        output_dir: str | Path,
        filename: str | None = None,
    ) -> Path:
        """Download an arXiv PDF to ``output_dir`` and return the saved path."""
        arxiv_id = self._extract_arxiv_id(arxiv_id_or_url)
        pdf_url = self._to_pdf_url(arxiv_id_or_url)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        target = output_path / (filename or f"{arxiv_id.replace('/', '_')}.pdf")

        logger.info("arXiv PDF download: %s", pdf_url)
        client = await self._get_client()
        response = await client.get(pdf_url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
            raise RuntimeError(f"arXiv PDF download did not return a PDF: {pdf_url}")

        target.write_bytes(response.content)
        return target

    @staticmethod
    def _extract_arxiv_id(value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if "/pdf/" in cleaned:
            return cleaned.rsplit("/pdf/", 1)[-1].removesuffix(".pdf")
        if "/abs/" in cleaned:
            return cleaned.rsplit("/abs/", 1)[-1]
        return cleaned.removesuffix(".pdf")

    @classmethod
    def _to_pdf_url(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned.startswith("http://"):
            cleaned = "https://" + cleaned.removeprefix("http://")
        if cleaned.startswith("https://") and "/pdf/" in cleaned:
            return cleaned
        arxiv_id = cls._extract_arxiv_id(cleaned)
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    def _parse_response(self, xml_text: str) -> list[dict]:
        """Parse arXiv Atom XML response into list of dicts."""
        root = ET.fromstring(xml_text)
        papers = []

        for entry in root.findall("atom:entry", ARXIV_NS):
            paper = {
                "arxiv_id": self._text(entry, "atom:id", "").split("/abs/")[-1],
                "title": self._text(entry, "atom:title", "").replace("\n", " ").strip(),
                "summary": self._text(entry, "atom:summary", "").strip(),
                "published": self._text(entry, "atom:published", ""),
                "updated": self._text(entry, "atom:updated", ""),
                "authors": [
                    a.find("atom:name", ARXIV_NS).text
                    for a in entry.findall("atom:author", ARXIV_NS)
                    if a.find("atom:name", ARXIV_NS) is not None
                ],
                "categories": [
                    c.get("term", "")
                    for c in entry.findall("atom:category", ARXIV_NS)
                ],
                "pdf_url": "",
            }
            # Find PDF link
            for link in entry.findall("atom:link", ARXIV_NS):
                if link.get("title") == "pdf":
                    paper["pdf_url"] = self._to_pdf_url(link.get("href", ""))
                    break
            papers.append(paper)

        return papers

    @staticmethod
    def _text(element: ET.Element, tag: str, default: str = "") -> str:
        child = element.find(tag, ARXIV_NS)
        return child.text if child is not None and child.text else default

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> ArxivClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
