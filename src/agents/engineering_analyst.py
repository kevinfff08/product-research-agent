"""Agent that analyzes engineering feasibility of research paths."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.storage.local_store import LocalStore
from src.apis.github_client import GitHubClient
from src.models.plan import ResearchPath, SearchQuery
from src.models.engineering import CodeAnalysis, DeploymentAssessment, EngineeringAnalysis
from src.models.common import SourceReference, SourceType


class EngineeringAnalyst(BaseAgent):
    """Analyzes engineering feasibility: code quality, deployment readiness."""

    agent_name = "engineering_analyst"

    def __init__(
        self,
        llm: LLMClient,
        store: LocalStore,
        github: GitHubClient,
        *,
        max_code_queries: int = 8,
        github_results_per_query: int = 10,
        max_repo_analyses: int = 10,
        api_concurrency: int = 5,
        llm_concurrency: int = 3,
    ):
        super().__init__(llm, store)
        self.github = github
        self.max_code_queries = max_code_queries
        self.github_results_per_query = github_results_per_query
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
            SearchQuery(query=q, source="github", path_id=path.path_id)
            for q in path.search_queries.get("code", [])
        ]

        # Search for repos
        repos = await self._search_repos(c_queries)

        # Analyze each repo deeply
        code_analyses, sources = await self._analyze_repos(repos, path)

        # Generate deployment assessment and recommendations via LLM
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
            "Engineering analysis complete: %d repos analyzed",
            len(code_analyses),
        )
        return result

    async def _search_repos(self, queries: list[SearchQuery]) -> list[dict]:
        """Search GitHub for relevant repos."""
        semaphore = asyncio.Semaphore(self.api_concurrency)

        async def search_one(sq: SearchQuery) -> list[dict]:
            async with semaphore:
                try:
                    return await self.github.search_repos(
                        sq.query,
                        limit=self.github_results_per_query,
                    )
                except Exception as exc:
                    self.logger.warning("GitHub search failed for '%s': %s", sq.query, exc)
                    return []

        groups = await asyncio.gather(
            *(search_one(sq) for sq in queries[: self.max_code_queries]),
        )
        all_repos = []
        seen_urls = set()
        for group in groups:
            for r in group:
                url = r.get("html_url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_repos.append(r)
        return all_repos

    async def _analyze_repos(
        self, repos: list[dict], path: ResearchPath,
    ) -> tuple[list[CodeAnalysis], list[SourceReference]]:
        """Deeply analyze repos for engineering quality."""
        semaphore = asyncio.Semaphore(self.llm_concurrency)

        async def analyze_one(repo_data: dict) -> tuple[CodeAnalysis | None, SourceReference | None]:
            url = repo_data.get("html_url", "")
            owner, name = GitHubClient.parse_repo_url(url)
            if not owner:
                return None, None

            # Get README
            try:
                readme = await self.github.get_readme(owner, name)
            except Exception:
                readme = ""

            # LLM analysis
            async with semaphore:
                llm_result = await self._call_llm_json_async(
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

            code_analysis = CodeAnalysis(repo_url=url)
            if llm_result and isinstance(llm_result, dict):
                code_analysis.architecture_summary = llm_result.get("architecture_summary", "")
                code_analysis.tech_stack = llm_result.get("tech_stack", [])
                code_analysis.code_patterns = llm_result.get("code_quality_notes", "")
                code_analysis.documentation_quality = llm_result.get("deployment_readiness", "")
                code_analysis.api_design_notes = llm_result.get("community_health", "")
                code_analysis.scalability_notes = "; ".join(llm_result.get("strengths", []))
                code_analysis.test_coverage_notes = "; ".join(llm_result.get("weaknesses", []))

            source = SourceReference(
                url=url,
                title=repo_data.get("full_name", ""),
                source_type=SourceType.REPO,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
            )
            return code_analysis, source

        analyzed = await asyncio.gather(
            *(analyze_one(repo) for repo in repos[: self.max_repo_analyses]),
        )
        analyses = [analysis for analysis, _ in analyzed if analysis is not None]
        sources = [source for _, source in analyzed if source is not None]
        return analyses, sources

    async def _assess_deployment(
        self, code_analyses: list[CodeAnalysis], path: ResearchPath,
    ) -> tuple[DeploymentAssessment, str, list[str]]:
        """Use LLM to assess deployment readiness across all analyzed repos."""
        if not code_analyses:
            return DeploymentAssessment(), "", []

        repos_summary = "\n".join(
            f"- {ca.repo_url}: stack={ca.tech_stack}, arch={ca.architecture_summary[:200]}"
            for ca in code_analyses
        )

        result = await self._call_llm_json_async(
            prompt=(
                f"Based on these repository analyses for '{path.title}':\n\n"
                f"{repos_summary}\n\n"
                f"Assess deployment feasibility. Return JSON:\n"
                f'{{"deployment_complexity": "trivial|moderate|complex|very_complex", '
                f'"infrastructure_requirements": ["req1"], '
                f'"estimated_setup_effort": "description", '
                f'"prerequisites": ["prereq1"], '
                f'"risks": ["risk1"], '
                f'"implementation_recommendations": "recommendations text", '
                f'"technology_stack_recommendation": ["tech1"]}}'
            ),
            temperature=0.2,
        )

        deployment = DeploymentAssessment()
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
