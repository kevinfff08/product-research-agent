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

# arXiv rate limit: no more than 1 request per 3 seconds for sustained use.
# We add jitter to avoid synchronisation with server-side rate windows.
_MIN_INTERVAL = 5.0
_MAX_JITTER = 2.0


class ArxivClient:
    """Client for arXiv API - preprint paper search.

    Uses an asyncio.Lock + mandatory delay to serialize all requests.
    This is essential because multiple research paths launch concurrent
    queries that would otherwise hit arXiv simultaneously and cause 429s.
    """

    BASE_URL = "https://export.arxiv.org/api/query"
    USER_AGENT = "ProductResearch/0.2 (research agent; contact: local)"

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0
        self._rate_lock = asyncio.Lock()
        # Pre-mark as if a request just happened, so the very first call
        # also waits.  This prevents the initial burst problem.
        self._last_request_time = time.monotonic()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
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

        for attempt in range(3):
            async with self._rate_lock:
                gap = time.monotonic() - self._last_request_time
                needed = _MIN_INTERVAL + random.uniform(0, _MAX_JITTER)
                if gap < needed:
                    await asyncio.sleep(needed - gap)

                logger.info("arXiv search: query=%s, max=%d", query, max_results)
                client = await self._get_client()
                response = await client.get(self.BASE_URL, params=params)
                self._last_request_time = time.monotonic()

            if response.status_code == 429:
                wait = 10.0 * (attempt + 1) + random.uniform(0, 5)
                logger.warning(
                    "arXiv 429 on attempt %d/3 for '%s', waiting %.0fs...",
                    attempt + 1, query[:60], wait,
                )
                await asyncio.sleep(wait)
                continue
            if response.status_code >= 400:
                response.raise_for_status()
            break

        papers = self._parse_response(response.text)
        logger.info("arXiv returned %d papers for: %s", len(papers), query)
        return papers

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
