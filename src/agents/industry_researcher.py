"""Agent that conducts industry research using web search."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.storage.local_store import LocalStore
from src.apis.tavily_client import TavilyClient
from src.apis.web_scraper import WebScraper
from src.models.plan import ResearchPath, SearchQuery
from src.models.industry import (
    ProductInfo, CompanyInfo, BlogSummary, IndustryResearchResult,
)
from src.models.common import MaturityStage
from src.models.common import SourceReference, SourceType


class IndustryResearcher(BaseAgent):
    """Conducts industry research: products, companies, market signals, blogs."""

    agent_name = "industry_researcher"

    def __init__(
        self,
        llm: LLMClient,
        store: LocalStore,
        tavily: TavilyClient,
        scraper: WebScraper,
        *,
        max_web_queries: int = 10,
        web_results_per_query: int = 5,
        max_web_analyses: int = 15,
        api_concurrency: int = 5,
        llm_concurrency: int = 3,
    ):
        super().__init__(llm, store)
        self.tavily = tavily
        self.scraper = scraper
        self.max_web_queries = max_web_queries
        self.web_results_per_query = web_results_per_query
        self.max_web_analyses = max_web_analyses
        self.api_concurrency = max(1, api_concurrency)
        self.llm_concurrency = max(1, llm_concurrency)

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
        if code_queries:
            self.logger.debug(
                "Ignoring %d code queries in industry research; engineering handles code search",
                len(code_queries),
            )

        web_results = await self._search_web(w_queries)

        # Analyze web results in one bounded LLM call instead of one call per source.
        products, companies, blogs, sources, market_trends = await self._analyze_web_results(
            web_results, path,
        )

        result = IndustryResearchResult(
            path_id=path.path_id,
            products=products,
            companies=companies,
            repos=[],
            blog_summaries=blogs,
            market_trends=market_trends,
            sources=sources,
        )
        self.logger.info(
            "Industry research complete: %d products, %d blogs",
            len(products), len(blogs),
        )
        return result

    async def _search_web(self, queries: list[SearchQuery]) -> list[dict]:
        """Execute web searches via Tavily."""
        semaphore = asyncio.Semaphore(self.api_concurrency)

        async def search_one(sq: SearchQuery) -> list[dict]:
            async with semaphore:
                try:
                    data = await self.tavily.search(
                        sq.query,
                        search_depth="basic",
                        max_results=self.web_results_per_query,
                        include_answer=False,
                    )
                except Exception as exc:
                    self.logger.warning("Tavily search failed for '%s': %s", sq.query, exc)
                    return []
            results = []
            for r in data.get("results", []):
                r["_query"] = sq.query
                r["_path_id"] = sq.path_id
                r["_intent"] = sq.intent
                results.append(r)
            return results

        groups = await asyncio.gather(
            *(search_one(sq) for sq in queries[: self.max_web_queries]),
        )
        all_results = []
        seen_urls: set[str] = set()
        for group in groups:
            for result in group:
                url = result.get("url", "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                all_results.append(result)
        return all_results

    async def _analyze_web_results(
        self, results: list[dict], path: ResearchPath,
    ) -> tuple[
        list[ProductInfo],
        list[CompanyInfo],
        list[BlogSummary],
        list[SourceReference],
        str,
    ]:
        """Use one LLM call to analyze the top web search results."""
        selected = [
            r for r in results[: self.max_web_analyses]
            if r.get("url") and (r.get("content") or r.get("raw_content"))
        ]
        sources = self._build_sources(selected)
        if not selected:
            return [], [], [], sources, ""

        source_payload = [
            {
                "index": i + 1,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "query": r.get("_query", ""),
                "content": (r.get("content", "") or r.get("raw_content", ""))[:1200],
            }
            for i, r in enumerate(selected)
        ]
        analysis = await self._call_llm_json_async(
            prompt=self._render_template(
                "analyze_industry_batch",
                {
                    "path_title": path.title,
                    "key_questions": "; ".join(path.key_questions[:3]),
                    "sources_json": json.dumps(source_payload, ensure_ascii=False, indent=2),
                },
            ),
            temperature=0.2,
        )

        if not analysis or not isinstance(analysis, dict):
            return [], [], [], sources, ""

        products: list[ProductInfo] = []
        companies: list[CompanyInfo] = []
        for product_data in analysis.get("products", []):
            if isinstance(product_data, dict):
                product = self._coerce_product(product_data)
                if product is not None:
                    products.append(product)
        for company_data in analysis.get("companies", []):
            if isinstance(company_data, dict):
                company = self._coerce_company(company_data)
                if company is not None:
                    companies.append(company)

        key_insights = self._as_str_list(analysis.get("key_insights", []))
        blogs = [
            BlogSummary(
                title=source.title,
                url=source.url,
                key_points=key_insights[:5],
            )
            for source in sources[: min(len(sources), 3)]
            if key_insights
        ]
        market_trends = str(analysis.get("market_trends", ""))
        return products, companies, blogs, sources, market_trends

    def _build_sources(self, results: list[dict]) -> list[SourceReference]:
        """Build source references from raw web results."""
        sources: list[SourceReference] = []
        for result in results:
            sources.append(SourceReference(
                url=result.get("url", ""),
                title=result.get("title", ""),
                source_type=SourceType.WEB,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                relevance_score=float(result.get("score", 0.5) or 0.5),
                snippet=(result.get("content", "") or result.get("raw_content", ""))[:500],
            ))
        return sources

    def _coerce_product(self, data: dict[str, Any]) -> ProductInfo | None:
        """Coerce tolerant LLM product JSON into the strict ProductInfo model."""
        payload = {k: v for k, v in data.items() if k in ProductInfo.model_fields}
        payload["capabilities"] = self._as_str_list(payload.get("capabilities", []))
        payload["limitations"] = self._as_str_list(payload.get("limitations", []))
        payload["is_open_source"] = self._as_bool(payload.get("is_open_source", False))
        maturity = str(payload.get("maturity", MaturityStage.DEVELOPMENT.value))
        if maturity not in {item.value for item in MaturityStage}:
            maturity = MaturityStage.DEVELOPMENT.value
        payload["maturity"] = maturity
        try:
            return ProductInfo(**payload)
        except ValidationError as exc:
            self.logger.debug("Dropping invalid product entry: %s", exc)
            return None

    def _coerce_company(self, data: dict[str, Any]) -> CompanyInfo | None:
        """Coerce tolerant LLM company JSON into the strict CompanyInfo model."""
        payload = {k: v for k, v in data.items() if k in CompanyInfo.model_fields}
        try:
            return CompanyInfo(**payload)
        except ValidationError as exc:
            self.logger.debug("Dropping invalid company entry: %s", exc)
            return None

    @staticmethod
    def _as_bool(value: Any) -> bool:
        """Parse common LLM boolean strings without raising model validation errors."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y", "1", "open source", "open-source"}:
                return True
            if normalized in {"false", "no", "n", "0", "closed source", "proprietary"}:
                return False
            return False
        return bool(value)

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        """Normalize scalar/list values into a list of strings."""
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []
