"""Central pipeline coordinator that wires all agents together."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.logging_config import get_logger
from src.llm.client import LLMClient
from src.storage.local_store import LocalStore
from src.models.input import ResearchRequest
from src.models.common import ResearchWeight
from src.models.plan import ResearchPlan, SearchQuery
from src.models.report import ResearchReport

# API clients
from src.apis.tavily_client import TavilyClient
from src.apis.arxiv_client import ArxivClient
from src.apis.openalex_client import OpenAlexClient
from src.apis.web_scraper import WebScraper

# Agents
from src.agents.idea_decomposer import IdeaDecomposer
from src.agents.research_planner import ResearchPlanner
from src.agents.industry_researcher import IndustryResearcher
from src.agents.academic_researcher import AcademicResearcher
from src.agents.engineering_analyst import EngineeringAnalyst
from src.agents.reputation_scorer import ReputationScorer
from src.agents.maturity_mapper import MaturityMapper
from src.agents.report_generator import ReportGenerator

# Reporter
from src.reporters.docx_reporter import DocxReporter
from src.reporters.markdown_reporter import MarkdownReporter
from src.utils.naming import build_run_name

load_dotenv()
logger = get_logger("orchestrator")


_DEPTH_PROFILES: dict[str, dict[str, int]] = {
    "quick": {
        "max_paths": 1,
        "max_parallel_paths": 1,
        "api_concurrency": 2,
        "llm_analysis_concurrency": 1,
        "max_search_results_per_query": 5,
        "max_web_queries_per_path": 3,
        "max_code_queries_per_path": 2,
        "max_academic_queries_per_path": 2,
        "max_arxiv_queries_per_path": 2,
        "max_papers_per_query": 5,
        "max_papers_per_path": 4,
        "max_repos_per_path": 4,
        "max_blog_sources": 4,
    },
    "comprehensive": {
        "max_paths": 3,
        "max_parallel_paths": 2,
        "api_concurrency": 3,
        "llm_analysis_concurrency": 1,
        "max_search_results_per_query": 8,
        "max_web_queries_per_path": 5,
        "max_code_queries_per_path": 4,
        "max_academic_queries_per_path": 3,
        "max_arxiv_queries_per_path": 3,
        "max_papers_per_query": 8,
        "max_papers_per_path": 8,
        "max_repos_per_path": 6,
        "max_blog_sources": 6,
    },
    "deep": {
        "max_paths": 5,
        "max_parallel_paths": 3,
        "api_concurrency": 5,
        "llm_analysis_concurrency": 2,
        "max_search_results_per_query": 10,
        "max_web_queries_per_path": 8,
        "max_code_queries_per_path": 5,
        "max_academic_queries_per_path": 4,
        "max_arxiv_queries_per_path": 4,
        "max_papers_per_query": 10,
        "max_papers_per_path": 12,
        "max_repos_per_path": 8,
        "max_blog_sources": 8,
    },
}


class Orchestrator:
    """Coordinates the full research pipeline."""

    def __init__(
        self,
        config_path: str | Path = "config/default.yaml",
        data_dir: str | Path = "data",
        output_dir: str | Path = "output",
    ):
        self.config = self._load_config(config_path)
        self.store = LocalStore(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Build LLM client
        self.llm = LLMClient(
            max_tokens=self.config.get("llm", {}).get("default_max_tokens", 8192),
        )

        # Build API clients
        self.tavily = TavilyClient(
            api_key=os.environ.get("TAVILY_API_KEY", ""),
            cache_dir=self.store.cache_dir,
        )
        self.arxiv = ArxivClient()
        self.openalex = OpenAlexClient(cache_dir=self.store.cache_dir)
        self.scraper = WebScraper()

        # Build agents
        default_limits = self._depth_settings("comprehensive")
        self.decomposer = IdeaDecomposer(self.llm, self.store)
        self.planner = ResearchPlanner(self.llm, self.store)
        self.industry_researcher = IndustryResearcher(
            self.llm,
            self.store,
            self.tavily,
            self.scraper,
            max_web_queries=default_limits["max_web_queries_per_path"],
            web_results_per_query=default_limits["max_search_results_per_query"],
            max_web_analyses=default_limits["max_blog_sources"],
            api_concurrency=default_limits["api_concurrency"],
            llm_concurrency=default_limits["llm_analysis_concurrency"],
        )
        self.academic_researcher = AcademicResearcher(
            self.llm,
            self.store,
            self.tavily,
            self.arxiv,
            self.openalex,
            max_academic_queries=default_limits["max_academic_queries_per_path"],
            max_arxiv_queries=default_limits["max_arxiv_queries_per_path"],
            papers_per_query=default_limits["max_papers_per_query"],
            max_paper_analyses=default_limits["max_papers_per_path"],
            api_concurrency=default_limits["api_concurrency"],
            llm_concurrency=default_limits["llm_analysis_concurrency"],
        )
        self.engineering_analyst = EngineeringAnalyst(
            self.llm,
            self.store,
            self.tavily,
            max_code_queries=default_limits["max_code_queries_per_path"],
            code_results_per_query=default_limits["max_search_results_per_query"],
            max_repo_analyses=default_limits["max_repos_per_path"],
            api_concurrency=default_limits["api_concurrency"],
            llm_concurrency=default_limits["llm_analysis_concurrency"],
        )
        self.reputation_scorer = ReputationScorer(self.llm, self.store)
        self.maturity_mapper = MaturityMapper(self.llm, self.store)
        self.report_generator = ReportGenerator(self.llm, self.store)
        self.markdown_reporter = MarkdownReporter()
        self.docx_reporter = DocxReporter()
        self.output_paths: list[Path] = []

    def _load_config(self, path: str | Path) -> dict:
        """Load YAML configuration."""
        path = Path(path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        logger.warning("Config not found at %s, using defaults", path)
        return {}

    async def run(self, request: ResearchRequest) -> ResearchReport:
        """Execute the full research pipeline.

        Pipeline:
        1. Decompose idea into research paths
        2. Plan search queries per source
        3. Run industry/academic/engineering research in parallel
        4. Score reputation and map maturity
        5. Generate final report
        6. Write Markdown output
        """
        run_name = request.run_name or build_run_name(request.title)
        session_id = self.store.create_session(
            description=request.raw_input[:200],
            title=request.title,
            detailed_description=request.detailed_description,
            session_id=run_name,
        )
        logger.info(
            "Starting research pipeline: session=%s, title='%s', input='%s'",
            session_id,
            request.title,
            request.raw_input[:80],
        )
        depth_settings = self._depth_settings(request.depth)
        self._apply_depth_settings(depth_settings)
        effective_max_paths = min(
            request.max_paths or depth_settings["max_paths"],
            depth_settings["max_paths"],
        )
        logger.info(
            "Depth profile '%s': max_paths=%d, max_parallel_paths=%d",
            request.depth,
            effective_max_paths,
            depth_settings["max_parallel_paths"],
        )

        try:
            # Step 1: Decompose
            self.store.update_session_status(session_id, "decomposing")
            decomposition = self.decomposer.run(
                raw_input=request.raw_input,
                max_paths=effective_max_paths,
            )
            self.store.save_model(
                f"research/{session_id}/decomposition.json", decomposition,
            )
            logger.info("Decomposition: %d paths", len(decomposition.paths))

            if not decomposition.paths:
                logger.error("No paths generated from decomposition")
                self.store.update_session_status(session_id, "failed")
                return ResearchReport(
                    title="Failed Research",
                    session_id=session_id,
                    original_input=request.raw_input,
                    executive_summary="Failed to decompose the research idea into viable paths.",
                )

            # Step 2: Plan
            self.store.update_session_status(session_id, "planning")
            weights = ResearchWeight(
                industry=request.weights.industry if request.weights else 0.60,
                academic=request.weights.academic if request.weights else 0.25,
                community=request.weights.community if request.weights else 0.15,
            )
            plan = self.planner.run(decomposition=decomposition, weights=weights)
            self.store.save_model(f"research/{session_id}/plan.json", plan)

            # Group queries by path and source
            path_queries = self._group_queries(plan)

            # Step 3: Parallel research
            self.store.update_session_status(session_id, "researching")
            max_parallel_paths = depth_settings["max_parallel_paths"]
            path_semaphore = asyncio.Semaphore(max(1, int(max_parallel_paths)))
            research_tasks = [
                self._run_path_research(
                    path,
                    path_queries.get(path.path_id, {}),
                    path_semaphore,
                )
                for path in decomposition.paths
            ]
            outcomes = await asyncio.gather(*research_tasks, return_exceptions=True)

            industry_by_path: dict[str, Any] = {}
            academic_by_path: dict[str, Any] = {}
            engineering_by_path: dict[str, Any] = {}

            for path, outcome in zip(decomposition.paths, outcomes, strict=False):
                if isinstance(outcome, Exception):
                    logger.error("Research subagents failed for %s: %s", path.path_id, outcome)
                    continue
                path_id, industry_result, academic_result, engineering_result = outcome
                if industry_result is not None:
                    industry_by_path[path_id] = industry_result
                if academic_result is not None:
                    academic_by_path[path_id] = academic_result
                if engineering_result is not None:
                    engineering_by_path[path_id] = engineering_result

            industry_results = [
                industry_by_path[path.path_id]
                for path in decomposition.paths
                if path.path_id in industry_by_path
            ]
            academic_results = [
                academic_by_path[path.path_id]
                for path in decomposition.paths
                if path.path_id in academic_by_path
            ]
            engineering_results = [
                engineering_by_path[path.path_id]
                for path in decomposition.paths
                if path.path_id in engineering_by_path
            ]

            # Save intermediate results
            for i, ir in enumerate(industry_results):
                self.store.save_model(f"research/{session_id}/industry_{i}.json", ir)
            for i, ar in enumerate(academic_results):
                self.store.save_model(f"research/{session_id}/academic_{i}.json", ar)
            for i, er in enumerate(engineering_results):
                self.store.save_model(f"research/{session_id}/engineering_{i}.json", er)

            # Step 4: Reputation + Maturity
            self.store.update_session_status(session_id, "scoring")
            reputation_report = self.reputation_scorer.run(
                industry_results=industry_results,
                academic_results=academic_results,
            )
            self.store.save_model(f"research/{session_id}/reputation.json", reputation_report)

            maturity_assessments = []
            for path in decomposition.paths:
                ind = industry_by_path.get(path.path_id)
                acad = academic_by_path.get(path.path_id)
                eng = engineering_by_path.get(path.path_id)
                assessment = self.maturity_mapper.run(
                    path=path,
                    industry_result=ind,
                    academic_result=acad,
                    engineering_result=eng,
                )
                maturity_assessments.append(assessment)
            for i, ma in enumerate(maturity_assessments):
                self.store.save_model(f"research/{session_id}/maturity_{i}.json", ma)

            # Step 5: Generate report
            self.store.update_session_status(session_id, "generating_report")
            report = self.report_generator.run(
                decomposition=decomposition,
                industry_results=industry_results,
                academic_results=academic_results,
                engineering_results=engineering_results,
                maturity_assessments=maturity_assessments,
                reputation_report=reputation_report,
                session_id=session_id,
            )
            if request.title:
                report.title = f"Technology Landscape: {request.title}"
            self.store.save_model(f"research/{session_id}/report.json", report)

            # Step 6: Write requested outputs
            output_paths = self._write_outputs(report, request.output_format)

            self.store.update_session_status(session_id, "completed")
            logger.info(
                "Research pipeline complete: %s",
                ", ".join(str(p) for p in output_paths),
            )

            return report

        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            self.store.update_session_status(session_id, "failed")
            raise

        finally:
            # Clean up API clients
            await self._cleanup()

    def _group_queries(self, plan: ResearchPlan) -> dict[str, dict[str, list[SearchQuery]]]:
        """Group search queries by path_id and source."""
        grouped: dict = {}
        for q in plan.search_queries:
            pid = q.path_id or "default"
            source = self._normalize_query_source(q.source)
            normalized = q.model_copy(update={"source": source})
            grouped.setdefault(pid, {})
            grouped[pid].setdefault(source, [])
            grouped[pid][source].append(normalized)
        return grouped

    async def _run_path_research(
        self,
        path: Any,
        queries_by_source: dict[str, list[SearchQuery]],
        semaphore: asyncio.Semaphore,
    ) -> tuple[str, Any | None, Any | None, Any | None]:
        """Run industry, academic, and engineering subagents for one path."""
        async with semaphore:
            web_queries = queries_by_source.get("tavily", [])
            code_queries = queries_by_source.get("code_web", [])
            academic_queries = [
                *queries_by_source.get("academic_web", []),
                *queries_by_source.get("arxiv", []),
            ]
            logger.info(
                "Launching research subagents for %s: web=%d, code=%d, academic=%d",
                path.path_id,
                len(web_queries),
                len(code_queries),
                len(academic_queries),
            )
            results = await asyncio.gather(
                self.industry_researcher.run(
                    path=path,
                    web_queries=web_queries,
                ),
                self.academic_researcher.run(path=path, queries=academic_queries),
                self.engineering_analyst.run(path=path, code_queries=code_queries),
                return_exceptions=True,
            )

        labels = ["industry", "academic", "engineering"]
        normalized: list[Any | None] = []
        for label, result in zip(labels, results, strict=False):
            if isinstance(result, Exception):
                logger.error("%s subagent failed for %s: %s", label, path.path_id, result)
                normalized.append(None)
            else:
                normalized.append(result)
        return path.path_id, normalized[0], normalized[1], normalized[2]

    def _write_outputs(self, report: ResearchReport, output_format: str) -> list[Path]:
        """Write report outputs in per-run subdirectories matching the log naming convention."""
        normalized = output_format.strip().lower()
        if normalized not in {"markdown", "docx", "both"}:
            raise ValueError(
                "Unsupported output format. Expected one of: markdown, docx, both"
            )

        run_dir = self.output_dir / report.session_id
        run_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        if normalized in {"markdown", "both"}:
            markdown_path = run_dir / f"{report.session_id}.md"
            paths.append(self.markdown_reporter.generate(report, markdown_path))

        if normalized in {"docx", "both"}:
            docx_path = run_dir / f"{report.session_id}.docx"
            paths.append(self.docx_reporter.generate(report, docx_path))

        self.output_paths = paths
        return paths

    async def _cleanup(self) -> None:
        """Close API clients."""
        for client in [self.tavily, self.arxiv, self.openalex, self.scraper]:
            try:
                await client.close()
            except Exception:
                pass
        try:
            self.llm.close()
        except Exception:
            pass

    def _depth_settings(self, depth: str) -> dict[str, int]:
        """Return effective workload limits for a depth profile."""
        normalized = (depth or "comprehensive").strip().lower()
        settings = dict(_DEPTH_PROFILES.get(normalized, _DEPTH_PROFILES["comprehensive"]))
        research_config = self.config.get("research", {})

        # Explicit per-depth overrides keep quick/comprehensive/deep meaning stable.
        profile_overrides = research_config.get("depth_profiles", {}).get(normalized, {})
        for key, value in profile_overrides.items():
            if key in settings:
                settings[key] = int(value)
        return settings

    def _apply_depth_settings(self, settings: dict[str, int]) -> None:
        """Apply workload limits to the reusable subagent instances."""
        self.industry_researcher.max_web_queries = settings["max_web_queries_per_path"]
        self.industry_researcher.web_results_per_query = settings["max_search_results_per_query"]
        self.industry_researcher.max_web_analyses = settings["max_blog_sources"]
        self.industry_researcher.api_concurrency = max(1, settings["api_concurrency"])
        self.industry_researcher.llm_concurrency = max(1, settings["llm_analysis_concurrency"])

        self.academic_researcher.max_academic_queries = settings["max_academic_queries_per_path"]
        self.academic_researcher.max_arxiv_queries = settings["max_arxiv_queries_per_path"]
        self.academic_researcher.max_openalex_queries = settings.get(
            "max_openalex_queries_per_path", settings.get("max_arxiv_queries_per_path", 3),
        )
        self.academic_researcher.papers_per_query = settings["max_papers_per_query"]
        self.academic_researcher.max_paper_analyses = settings["max_papers_per_path"]
        self.academic_researcher.api_concurrency = max(1, settings["api_concurrency"])
        self.academic_researcher.llm_concurrency = max(1, settings["llm_analysis_concurrency"])

        self.engineering_analyst.max_code_queries = settings["max_code_queries_per_path"]
        self.engineering_analyst.code_results_per_query = settings["max_search_results_per_query"]
        self.engineering_analyst.max_repo_analyses = settings["max_repos_per_path"]
        self.engineering_analyst.api_concurrency = max(1, settings["api_concurrency"])
        self.engineering_analyst.llm_concurrency = max(1, settings["llm_analysis_concurrency"])

    @staticmethod
    def _normalize_query_source(source: str) -> str:
        """Normalize legacy planner source names to current search channels."""
        source_map = {
            "web": "tavily",
            "github": "code_web",
            "code": "code_web",
            "semantic_scholar": "academic_web",
            "academic": "academic_web",
        }
        normalized = (source or "tavily").strip().lower()
        return source_map.get(normalized, normalized or "tavily")
