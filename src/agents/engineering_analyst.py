"""Agent that analyzes engineering feasibility of research paths."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.storage.local_store import LocalStore
from src.apis.tavily_client import TavilyClient
from src.models.plan import ResearchPath, SearchQuery
from src.models.engineering import CodeAnalysis, DeploymentAssessment, EngineeringAnalysis
from src.models.common import SourceReference, SourceType
from src.utils.text_utils import clean_search_query


class EngineeringAnalyst(BaseAgent):
    """Analyzes engineering feasibility using web-discovered code sources."""

    agent_name = "engineering_analyst"

    def __init__(
        self,
        llm: LLMClient,
        store: LocalStore,
        code_search: TavilyClient,
        *,
        max_code_queries: int = 4,
        code_results_per_query: int = 6,
        max_repo_analyses: int = 6,
        api_concurrency: int = 3,
        llm_concurrency: int = 1,
    ):
        super().__init__(llm, store)
        self.code_search = code_search
        self.max_code_queries = max_code_queries
        self.code_results_per_query = code_results_per_query
        self.max_repo_analyses = max_repo_analyses
        self.api_concurrency = max(1, api_concurrency)
        self.llm_concurrency = max(1, llm_concurrency)

    async def run(
        self,
        *,
        path: ResearchPath,
        code_queries: list[SearchQuery] | None = None,
    ) -> EngineeringAnalysis:
        """Run engineering analysis for a single path."""
        self.logger.info("Engineering analysis for path: %s", path.title)

        c_queries = code_queries or [
            SearchQuery(query=q, source="code_web", path_id=path.path_id)
            for q in path.search_queries.get("code", [])
        ]

        repos = await self._search_code_sources(c_queries)
        code_analyses, sources = self._build_code_analyses(repos)
        deployment, recommendations, tech_stack = await self._assess_deployment(
            code_analyses, path,
        )

        result = EngineeringAnalysis(
            path_id=path.path_id,
            code_analyses=code_analyses,
            deployment_assessment=deployment,
            implementation_recommendations=recommendations,
            technology_stack_recommendation=tech_stack,
            sources=sources,
        )
        self.logger.info(
            "Engineering analysis complete: %d code sources analyzed",
            len(code_analyses),
        )
        return result

    async def _search_code_sources(self, queries: list[SearchQuery]) -> list[dict]:
        """Search code platforms through the general web search provider."""
        semaphore = asyncio.Semaphore(self.api_concurrency)

        async def search_one(sq: SearchQuery) -> list[dict]:
            clean_q = clean_search_query(sq.query)
            query = (
                f"{clean_q} open source implementation repository "
                f"site:github.com OR site:gitlab.com OR site:huggingface.co"
            )
            async with semaphore:
                try:
                    data = await self.code_search.search(
                        query,
                        search_depth="basic",
                        max_results=self.code_results_per_query,
                        include_answer=False,
                    )
                except Exception as exc:
                    self.logger.warning("Code web search failed for '%s': %s", sq.query, exc)
                    return []
            results = []
            for item in data.get("results", []):
                item["_query"] = sq.query
                item["_intent"] = sq.intent
                item["_source"] = "code_web"
                results.append(item)
            return results

        groups = await asyncio.gather(
            *(search_one(sq) for sq in queries[: self.max_code_queries]),
        )
        all_results = []
        seen_urls: set[str] = set()
        for group in groups:
            for item in group:
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(item)
        return all_results[: self.max_repo_analyses]

    def _build_code_analyses(
        self,
        results: list[dict],
    ) -> tuple[list[CodeAnalysis], list[SourceReference]]:
        """Convert search result snippets into lightweight code analyses."""
        analyses: list[CodeAnalysis] = []
        sources: list[SourceReference] = []
        for item in results[: self.max_repo_analyses]:
            url = item.get("url", "")
            title = item.get("title", "")
            content = item.get("content", "") or item.get("raw_content", "")
            if not url:
                continue
            analyses.append(CodeAnalysis(
                repo_url=url,
                architecture_summary=content[:700],
                tech_stack=self._guess_stack(" ".join([title, content, url])),
                code_patterns=self._guess_code_host(url),
                documentation_quality=content[:500],
                api_design_notes=title,
                scalability_notes=f"Discovered via query: {item.get('_query', '')}",
            ))
            sources.append(SourceReference(
                url=url,
                title=title,
                source_type=SourceType.REPO,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                relevance_score=float(item.get("score", 0.5) or 0.5),
            ))
        return analyses, sources

    async def _assess_deployment(
        self,
        code_analyses: list[CodeAnalysis],
        path: ResearchPath,
    ) -> tuple[DeploymentAssessment, str, list[str]]:
        """Use one LLM call to assess deployment readiness across all code sources."""
        if not code_analyses:
            return DeploymentAssessment(approach=path.title), "", []

        repos_summary = "\n".join(
            f"- {ca.repo_url}: stack={ca.tech_stack}, notes={ca.architecture_summary[:350]}"
            for ca in code_analyses[: self.max_repo_analyses]
        )

        try:
            result = await self._call_llm_json_async(
                prompt=(
                    f"Based on these code/source analyses for '{path.title}':\n\n"
                    f"{repos_summary}\n\n"
                    f"Assess deployment feasibility. Return JSON:\n"
                    f'{{"deployment_complexity": "trivial|moderate|complex|very_complex", '
                    f'"infrastructure_requirements": ["req1"], '
                    f'"estimated_setup_effort": "description", '
                    f'"prerequisites": ["preq1"], '
                    f'"risks": ["risk1"], '
                    f'"implementation_recommendations": "recommendations text", '
                    f'"technology_stack_recommendation": ["tech1"]}}'
                ),
                temperature=0.2,
            )
        except Exception as exc:
            self.logger.warning("Deployment assessment failed for '%s': %s", path.title, exc)
            return DeploymentAssessment(approach=path.title), "", []

        deployment = DeploymentAssessment(approach=path.title)
        recommendations = ""
        tech_stack: list[str] = []

        if result and isinstance(result, dict):
            deployment = DeploymentAssessment(
                approach=path.title,
                deployment_complexity=result.get("deployment_complexity", "moderate"),
                infrastructure_requirements=result.get("infrastructure_requirements", []),
                estimated_setup_effort=result.get("estimated_setup_effort", ""),
                prerequisites=result.get("prerequisites", []),
                risks=result.get("risks", []),
            )
            recommendations = result.get("implementation_recommendations", "")
            tech_stack = result.get("technology_stack_recommendation", [])

        return deployment, recommendations, tech_stack

    @staticmethod
    def _guess_code_host(url: str) -> str:
        lowered = (url or "").lower()
        if "github.com" in lowered:
            return "GitHub"
        if "gitlab.com" in lowered:
            return "GitLab"
        if "huggingface.co" in lowered:
            return "Hugging Face"
        return "Code or documentation source"

    @staticmethod
    def _guess_stack(text: str) -> list[str]:
        candidates = [
            "Python", "PyTorch", "TensorFlow", "ONNX", "CUDA", "TensorRT",
            "FastAPI", "Docker", "WebSocket", "gRPC", "TypeScript", "Rust",
            "C++", "Transformers", "Hugging Face",
        ]
        lowered = text.lower()
        found = [name for name in candidates if re.search(rf"\b{re.escape(name.lower())}\b", lowered)]
        return found[:8]
