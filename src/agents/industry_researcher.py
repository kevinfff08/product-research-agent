"""Agent that conducts industry research using web search and GitHub."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from src.agents.base import BaseAgent
from src.apis.tavily_client import TavilyClient
from src.apis.github_client import GitHubClient
from src.apis.web_scraper import WebScraper
from src.models.plan import ResearchPath, SearchQuery
from src.models.industry import (
    ProductInfo, CompanyInfo, RepoAnalysis, BlogSummary, IndustryResearchResult,
)
from src.models.common import SourceReference, SourceType


class IndustryResearcher(BaseAgent):
    """Conducts industry research: products, companies, repos, blogs."""

    agent_name = "industry_researcher"

    def __init__(self, llm, store, tavily: TavilyClient, github: GitHubClient, scraper: WebScraper):
        super().__init__(llm, store)
        self.tavily = tavily
        self.github = github
        self.scraper = scraper

    async def run(
        self,
        *,
        path: ResearchPath,
        web_queries: list[SearchQuery] | None = None,
        code_queries: list[SearchQuery] | None = None,
    ) -> IndustryResearchResult:
        """Run industry research for a single path."""
        self.logger.info("Industry research for path: %s", path.title)

        # Collect web and code queries
        w_queries = web_queries or [
            SearchQuery(query=q, source="tavily", path_id=path.path_id)
            for q in path.search_queries.get("web", [])
        ]
        c_queries = code_queries or [
            SearchQuery(query=q, source="github", path_id=path.path_id)
            for q in path.search_queries.get("code", [])
        ]

        # Run searches in parallel
        web_results, code_results = await asyncio.gather(
            self._search_web(w_queries),
            self._search_github(c_queries),
            return_exceptions=True,
        )

        if isinstance(web_results, Exception):
            self.logger.error("Web search failed: %s", web_results)
            web_results = []
        if isinstance(code_results, Exception):
            self.logger.error("GitHub search failed: %s", code_results)
            code_results = []

        # Analyze web results with LLM
        products, companies, blogs, sources = await self._analyze_web_results(
            web_results, path,
        )

        # Analyze GitHub repos
        repos, repo_sources = await self._analyze_repos(code_results, path)
        sources.extend(repo_sources)

        result = IndustryResearchResult(
            path_id=path.path_id,
            products=products,
            companies=companies,
            repos=repos,
            blog_summaries=blogs,
            sources=sources,
        )
        self.logger.info(
            "Industry research complete: %d products, %d repos, %d blogs",
            len(products), len(repos), len(blogs),
        )
        return result

    async def _search_web(self, queries: list[SearchQuery]) -> list[dict]:
        """Execute web searches via Tavily."""
        all_results = []
        for sq in queries[:10]:  # Limit total queries
            try:
                data = await self.tavily.search(sq.query, max_results=5)
                for r in data.get("results", []):
                    r["_query"] = sq.query
                    r["_path_id"] = sq.path_id
                all_results.extend(data.get("results", []))
            except Exception as exc:
                self.logger.warning("Tavily search failed for '%s': %s", sq.query, exc)
        return all_results

    async def _search_github(self, queries: list[SearchQuery]) -> list[dict]:
        """Search GitHub repos."""
        all_repos = []
        seen_urls = set()
        for sq in queries[:5]:
            try:
                repos = await self.github.search_repos(sq.query, language=None, limit=5)
                for r in repos:
                    url = r.get("html_url", "")
                    if url not in seen_urls:
                        seen_urls.add(url)
                        all_repos.append(r)
            except Exception as exc:
                self.logger.warning("GitHub search failed for '%s': %s", sq.query, exc)
        return all_repos

    async def _analyze_web_results(
        self, results: list[dict], path: ResearchPath,
    ) -> tuple[list[ProductInfo], list[CompanyInfo], list[BlogSummary], list[SourceReference]]:
        """Use LLM to analyze web search results."""
        products, companies, blogs, sources = [], [], [], []

        for r in results[:15]:  # Limit to avoid excessive LLM calls
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")

            if not content:
                continue

            sources.append(SourceReference(
                url=url,
                title=title,
                source_type=SourceType.WEB,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                relevance_score=float(r.get("score", 0.5)),
            ))

            # LLM analysis
            analysis = self._call_llm_json(
                prompt=self._render_template(
                    "analyze_industry_source",
                    {
                        "path_title": path.title,
                        "key_questions": "; ".join(path.key_questions[:3]),
                        "source_title": title,
                        "source_url": url,
                        "source_content": content[:5000],
                    },
                ),
                temperature=0.2,
            )

            if not analysis or not isinstance(analysis, dict):
                continue

            for p in analysis.get("products", []):
                products.append(ProductInfo(**{k: v for k, v in p.items() if k in ProductInfo.model_fields}))
            for c in analysis.get("companies", []):
                companies.append(CompanyInfo(**{k: v for k, v in c.items() if k in CompanyInfo.model_fields}))

            if analysis.get("key_insights"):
                blogs.append(BlogSummary(
                    title=title, url=url,
                    key_points=analysis.get("key_insights", []),
                ))

        return products, companies, blogs, sources

    async def _analyze_repos(
        self, repos: list[dict], path: ResearchPath,
    ) -> tuple[list[RepoAnalysis], list[SourceReference]]:
        """Analyze GitHub repositories."""
        analyzed_repos = []
        sources = []

        for repo_data in repos[:10]:
            url = repo_data.get("html_url", "")
            owner, name = GitHubClient.parse_repo_url(url)
            if not owner:
                continue

            # Get README for deeper analysis
            readme = ""
            try:
                readme = await self.github.get_readme(owner, name)
            except Exception:
                pass

            # LLM analysis of repo
            analysis = self._call_llm_json(
                prompt=self._render_template(
                    "analyze_repo",
                    {
                        "path_title": path.title,
                        "repo_name": repo_data.get("full_name", f"{owner}/{name}"),
                        "repo_url": url,
                        "stars": repo_data.get("stargazers_count", 0),
                        "forks": repo_data.get("forks_count", 0),
                        "language": repo_data.get("language", ""),
                        "license": (repo_data.get("license") or {}).get("spdx_id", ""),
                        "description": repo_data.get("description", ""),
                        "topics": ", ".join(repo_data.get("topics", [])),
                        "updated_at": repo_data.get("updated_at", ""),
                        "readme_excerpt": readme[:3000],
                    },
                ),
                temperature=0.2,
            )

            repo_analysis = RepoAnalysis(
                repo_url=url,
                name=repo_data.get("full_name", ""),
                stars=repo_data.get("stargazers_count", 0),
                forks=repo_data.get("forks_count", 0),
                language=repo_data.get("language", ""),
                last_updated=repo_data.get("updated_at", ""),
                license=(repo_data.get("license") or {}).get("spdx_id", ""),
                description=repo_data.get("description", ""),
                topics=repo_data.get("topics", []),
            )

            if analysis and isinstance(analysis, dict):
                repo_analysis.code_quality_notes = analysis.get("code_quality_notes", "")
                repo_analysis.deployment_readiness = analysis.get("deployment_readiness", "")
                repo_analysis.community_health = analysis.get("community_health", "")
                repo_analysis.key_dependencies = analysis.get("key_dependencies", [])

            analyzed_repos.append(repo_analysis)
            sources.append(SourceReference(
                url=url, title=repo_analysis.name,
                source_type=SourceType.REPO,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
            ))

        return analyzed_repos, sources
