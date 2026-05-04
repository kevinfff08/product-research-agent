"""arXiv API client for preprint paper search."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from src.logging_config import get_logger

logger = get_logger("apis.arxiv")

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivClient:
    """Client for arXiv API - preprint paper search.

    Not extending BaseAPIClient because arXiv uses XML, not JSON.
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 20,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[dict]:
        """Search arXiv papers.

        Args:
            query: Search query (supports arXiv query syntax).
            max_results: Max results to return.
            sort_by: "relevance", "lastUpdatedDate", or "submittedDate".
            sort_order: "ascending" or "descending".
        """
        params = {
            "search_query": f"all:{query}",
            "max_results": min(max_results, 50),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        logger.info("arXiv search: query=%s, max=%d", query, max_results)
        client = await self._get_client()
        response = await client.get(self.BASE_URL, params=params)
        response.raise_for_status()

        papers = self._parse_response(response.text)
        logger.info("arXiv returned %d papers for: %s", len(papers), query)
        return papers

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
                    paper["pdf_url"] = link.get("href", "")
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
