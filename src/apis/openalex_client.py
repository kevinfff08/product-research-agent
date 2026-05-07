"""OpenAlex API client for high-quality academic paper search."""

from __future__ import annotations

from pathlib import Path

from src.apis.base import BaseAPIClient
from src.logging_config import get_logger

logger = get_logger("apis.openalex")

# Polite pool email for higher rate limits
_OPENALEX_EMAIL = "product-research@example.com"


class OpenAlexClient(BaseAPIClient):
    """Client for OpenAlex API - 250M+ papers with citation and venue data.

    OpenAlex is free, open-source (CC0), and requires no API key.
    It returns citation counts, venue metadata, and publication types
    that make it possible to filter for top-conference, high-impact papers.
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(self, cache_dir: Path | None = None):
        super().__init__(
            api_key=None,
            cache_dir=cache_dir,
            requests_per_second=8.0,
        )

    @property
    def headers(self) -> dict[str, str]:
        return {
            "User-Agent": f"ProductResearch/0.2 (mailto:{_OPENALEX_EMAIL})",
            "Accept": "application/json",
        }

    async def search_papers(
        self,
        query: str,
        limit: int = 20,
        sort: str = "cited_by_count:desc",
        year_from: int | None = None,
        year_to: int | None = None,
        peer_reviewed: bool = True,
    ) -> list[dict]:
        """Search OpenAlex for papers with rich metadata.

        Args:
            query: Search query string.
            limit: Max results (1-200).
            sort: Sort order. Options: cited_by_count:desc, relevance_score:desc,
                  publication_date:desc, cited_by_count:asc.
            year_from: Filter papers published from this year.
            year_to: Filter papers published up to this year.
            peer_reviewed: If True, prefer peer-reviewed papers (journals, conferences).

        Returns:
            List of paper dicts with: id, title, doi, publication_date,
            cited_by_count, primary_location (venue name, type),
            authorships, abstract_inverted_index, topics, type.
        """
        params: dict = {
            "search": query,
            "per_page": min(limit, 200),
            "sort": sort,
        }

        # Build filter string
        filters: list[str] = []
        current_year = 2026
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        elif peer_reviewed:
            filters.append(f"from_publication_date:{max(2018, current_year - 8)}-01-01")
        if year_to:
            filters.append(f"to_publication_date:{year_to}-12-31")
        if peer_reviewed:
            filters.append("type:article|conference-paper")

        if filters:
            params["filter"] = ",".join(filters)

        logger.info(
            "OpenAlex search: query=%s, limit=%d, sort=%s, filters=%s",
            query, limit, sort, filters,
        )

        try:
            data = await self.get("/works", params=params)
        except Exception as exc:
            logger.warning("OpenAlex search failed for '%s': %s", query, exc)
            return []

        results = data.get("results", [])
        papers = [self._normalize_paper(paper) for paper in results]
        logger.info("OpenAlex returned %d papers for: %s", len(papers), query)
        return papers

    async def get_paper_by_doi(self, doi: str) -> dict | None:
        """Look up a paper by DOI."""
        try:
            return await self.get(f"/works/doi:{doi}")
        except Exception:
            return None

    async def search_high_impact(
        self,
        query: str,
        limit: int = 15,
        min_citations: int = 10,
        year_from: int | None = None,
    ) -> list[dict]:
        """Search for high-impact papers (cited, peer-reviewed, top venues).

        This is the recommended method for finding 'mature technology'
        papers with meaningful academic validation.
        """
        papers = await self.search_papers(
            query=query,
            limit=limit * 2,  # Fetch more, then filter
            sort="cited_by_count:desc",
            year_from=year_from,
            peer_reviewed=True,
        )

        filtered = [
            p for p in papers
            if p.get("cited_by_count", 0) >= min_citations
        ]
        return filtered[:limit]

    def _normalize_paper(self, raw: dict) -> dict:
        """Normalize OpenAlex paper into our standard paper dict."""
        # Extract venue info
        primary_location = raw.get("primary_location") or {}
        source = primary_location.get("source") or {}
        venue_name = source.get("display_name", "")
        venue_type = raw.get("type", "")  # "article", "conference-paper", etc.

        # Extract abstract
        abstract = ""
        abstract_index = raw.get("abstract_inverted_index")
        if abstract_index and isinstance(abstract_index, dict):
            abstract = self._decode_inverted_index(abstract_index)

        # Extract authors
        authors = []
        for authorship in raw.get("authorships", [])[:10]:
            author_data = authorship.get("author", {})
            name = author_data.get("display_name", "")
            if name:
                authors.append(name)

        # Extract topics
        topics = []
        for topic in raw.get("topics", [])[:5]:
            topic_name = topic.get("display_name", "")
            if topic_name:
                topics.append(topic_name)

        # Extract DOI
        doi = raw.get("doi", "") or ""
        doi = doi.removeprefix("https://doi.org/")

        # Build normalized dict
        paper_id = raw.get("id", "").split("/")[-1] if raw.get("id") else ""

        return {
            "paper_id": paper_id,
            "title": raw.get("title", ""),
            "authors": authors,
            "year": raw.get("publication_year", 0) or 0,
            "venue": venue_name,
            "venue_type": venue_type,
            "doi": doi,
            "url": f"https://doi.org/{doi}" if doi else raw.get("id", ""),
            "abstract": abstract,
            "cited_by_count": raw.get("cited_by_count", 0) or 0,
            "citation_count": raw.get("cited_by_count", 0) or 0,
            "topics": topics,
            "is_oa": raw.get("open_access", {}).get("is_oa", False),
            "publication_date": raw.get("publication_date", ""),
            "_source": "openalex",
        }

    @staticmethod
    def _decode_inverted_index(inverted: dict) -> str:
        """Decode OpenAlex abstract inverted index to plain text."""
        if not inverted:
            return ""
        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)


__all__ = ["OpenAlexClient"]
