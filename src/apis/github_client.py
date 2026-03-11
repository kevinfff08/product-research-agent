"""GitHub API client for repository analysis."""

from __future__ import annotations

from pathlib import Path

from src.apis.base import BaseAPIClient
from src.logging_config import get_logger

logger = get_logger("apis.github")


class GitHubClient(BaseAPIClient):
    """Client for GitHub API - repository search and analysis."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None, cache_dir: Path | None = None):
        super().__init__(
            api_key=token,
            cache_dir=cache_dir,
            requests_per_second=5.0 if token else 0.5,
        )

    @property
    def headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "ProductResearch/0.1",
            "Accept": "application/vnd.github.v3+json",
        }
        if self.api_key:
            headers["Authorization"] = f"token {self.api_key}"
        return headers

    async def get_repo(self, owner: str, repo: str) -> dict:
        """Get repository information."""
        return await self.get(f"/repos/{owner}/{repo}")

    async def get_repo_details(self, owner: str, repo: str) -> dict:
        """Get detailed repository metrics for research analysis."""
        repo_data = await self.get_repo(owner, repo)
        return {
            "full_name": repo_data.get("full_name", ""),
            "html_url": repo_data.get("html_url", ""),
            "stars": repo_data.get("stargazers_count", 0),
            "forks": repo_data.get("forks_count", 0),
            "open_issues": repo_data.get("open_issues_count", 0),
            "language": repo_data.get("language", ""),
            "description": repo_data.get("description", ""),
            "updated_at": repo_data.get("updated_at", ""),
            "created_at": repo_data.get("created_at", ""),
            "license": (repo_data.get("license") or {}).get("spdx_id", ""),
            "archived": repo_data.get("archived", False),
            "topics": repo_data.get("topics", []),
            "size": repo_data.get("size", 0),
            "default_branch": repo_data.get("default_branch", "main"),
            "watchers": repo_data.get("subscribers_count", 0),
        }

    async def search_repos(
        self,
        query: str,
        language: str | None = None,
        sort: str = "stars",
        limit: int = 10,
    ) -> list[dict]:
        """Search for repositories."""
        search_query = query
        if language:
            search_query = f"{query} language:{language}"
        params = {
            "q": search_query,
            "sort": sort,
            "order": "desc",
            "per_page": min(limit, 30),
        }
        logger.info("GitHub search: query=%s, sort=%s, limit=%d", search_query, sort, limit)
        data = await self.get("/search/repositories", params=params)
        items = data.get("items", [])
        logger.info("GitHub returned %d repos for: %s", len(items), query)
        return items

    async def get_readme(self, owner: str, repo: str) -> str:
        """Get repository README content."""
        try:
            client = await self._get_client()
            await self.rate_limiter.acquire()
            response = await client.get(
                f"/repos/{owner}/{repo}/readme",
                headers={**self.headers, "Accept": "application/vnd.github.raw+json"},
            )
            response.raise_for_status()
            return response.text
        except Exception:
            logger.debug("Could not fetch README for %s/%s", owner, repo)
            return ""

    @staticmethod
    def parse_repo_url(url: str) -> tuple[str, str]:
        """Parse owner and repo name from a GitHub URL."""
        url = url.rstrip("/")
        if "github.com" in url:
            parts = url.split("github.com/")[-1].split("/")
            if len(parts) >= 2:
                return parts[0], parts[1].split("?")[0].split("#")[0]
        return "", ""
