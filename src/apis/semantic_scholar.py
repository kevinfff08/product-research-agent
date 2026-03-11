"""Semantic Scholar Academic Graph API client."""

from __future__ import annotations

from pathlib import Path

from src.apis.base import BaseAPIClient
from src.logging_config import get_logger

logger = get_logger("apis.semantic_scholar")


class SemanticScholarClient(BaseAPIClient):
    """Client for Semantic Scholar Academic Graph API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    PAPER_FIELDS = (
        "title,authors,year,venue,citationCount,externalIds,"
        "abstract,tldr,fieldsOfStudy,publicationTypes,openAccessPdf"
    )

    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None):
        super().__init__(
            api_key=api_key,
            cache_dir=cache_dir,
            requests_per_second=10.0 if api_key else 1.0,
        )

    @property
    def headers(self) -> dict[str, str]:
        headers = {"User-Agent": "ProductResearch/0.1 (research agent)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def search_papers(
        self,
        query: str,
        limit: int = 20,
        year: str | None = None,
        fields_of_study: list[str] | None = None,
    ) -> list[dict]:
        """Search for papers by keyword query."""
        params: dict = {
            "query": query,
            "limit": min(limit, 100),
            "fields": self.PAPER_FIELDS,
        }
        if year:
            params["year"] = year
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        logger.info("S2 search: query=%s, limit=%d", query, limit)
        data = await self.get("/paper/search", params=params)
        papers = data.get("data", [])
        logger.info("S2 returned %d papers for: %s", len(papers), query)
        return papers

    async def get_paper(self, paper_id: str) -> dict:
        """Get paper details by S2 ID, arXiv ID ('ARXIV:xxx'), or DOI ('DOI:xxx')."""
        params = {"fields": self.PAPER_FIELDS}
        return await self.get(f"/paper/{paper_id}", params=params)

    async def get_paper_citations(self, paper_id: str, limit: int = 50) -> list[dict]:
        """Get papers that cite this paper."""
        params = {"fields": self.PAPER_FIELDS, "limit": min(limit, 100)}
        data = await self.get(f"/paper/{paper_id}/citations", params=params)
        return [item.get("citingPaper", {}) for item in data.get("data", [])]

    async def get_paper_references(self, paper_id: str, limit: int = 50) -> list[dict]:
        """Get papers referenced by this paper."""
        params = {"fields": self.PAPER_FIELDS, "limit": min(limit, 100)}
        data = await self.get(f"/paper/{paper_id}/references", params=params)
        return [item.get("citedPaper", {}) for item in data.get("data", [])]
