"""Tavily AI Search API client."""

from __future__ import annotations

from pathlib import Path

from src.apis.base import BaseAPIClient
from src.logging_config import get_logger

logger = get_logger("apis.tavily")


class TavilyClient(BaseAPIClient):
    """Client for Tavily AI-native search API.

    Tavily returns pre-processed, LLM-optimized search results
    with content snippets alongside URLs.
    """

    BASE_URL = "https://api.tavily.com"

    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None):
        super().__init__(
            api_key=api_key,
            cache_dir=cache_dir,
            requests_per_second=5.0,
        )
        self._tavily_key = api_key or ""

    async def search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int = 10,
        include_answer: bool = True,
        include_raw_content: bool = False,
        topic: str = "general",
    ) -> dict:
        """Search the web using Tavily API.

        Args:
            query: Search query string.
            search_depth: "basic" or "advanced".
            max_results: Maximum results to return (1-20).
            include_answer: Whether to include an AI-generated answer.
            include_raw_content: Whether to include raw HTML content.
            topic: "general" or "news".

        Returns:
            Dict with keys: query, answer, results, response_time.
            Each result has: title, url, content, score.
        """
        payload = {
            "api_key": self._tavily_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": min(max_results, 20),
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "topic": topic,
        }

        logger.info("Tavily search: query=%s, depth=%s, max=%d", query, search_depth, max_results)
        result = await self.post("/search", json_data=payload)
        result_count = len(result.get("results", []))
        logger.info("Tavily returned %d results for: %s", result_count, query)
        return result
