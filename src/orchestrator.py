"""Central pipeline coordinator that wires all agents together."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

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
from src.apis.semantic_scholar import SemanticScholarClient
from src.apis.github_client import GitHubClient
from src.apis.arxiv_client import ArxivClient
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

load_dotenv()
logger = get_logger("orchestrator")


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
        self.s2 = SemanticScholarClient(
            api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
            cache_dir=self.store.cache_dir,
        )
        self.github = GitHubClient(
            token=os.environ.get("GITHUB_TOKEN"),
            cache_dir=self.store.cache_dir,
        )
        self.arxiv = ArxivClient()
        self.scraper = WebScraper()

        # Build agents
        self.decomposer = IdeaDecomposer(self.llm, self.store)
        self.planner = ResearchPlanner(self.llm, self.store)
        self.industry_researcher = IndustryResearcher(
            self.llm, self.store, self.tavily, self.github, self.scraper,
        )
        self.academic_researcher = AcademicResearcher(
            self.llm, self.store, self.s2, self.arxiv,
        )
        self.engineering_analyst = EngineeringAnalyst(
            self.llm, self.store, self.github,
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
        session_id = self.store.create_session(description=request.raw_input[:200])
        logger.info("Starting research pipeline: session=%s, input='%s'",
                     session_id, request.raw_input[:80])

        try:
            # Step 1: Decompose
            self.store.update_session_status(session_id, "decomposing")
            decomposition = self.decomposer.run(
                raw_input=request.raw_input,
                max_paths=request.max_paths or self.config.get("research", {}).get("max_paths", 5),
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
            industry_results = []
            academic_results = []
            engineering_results = []

            for path in decomposition.paths:
                pq = path_queries.get(path.path_id, {})
                web_queries = pq.get("tavily", [])
                code_queries = pq.get("github", [])
                academic_queries = pq.get("semantic_scholar", [])

                # Run all three researchers in parallel for this path
                ind_task = self.industry_researcher.run(
                    path=path, web_queries=web_queries, code_queries=code_queries,
                )
                acad_task = self.academic_researcher.run(
                    path=path, queries=academic_queries,
                )
                eng_task = self.engineering_analyst.run(
                    path=path, code_queries=code_queries,
                )

                results = await asyncio.gather(
                    ind_task, acad_task, eng_task,
                    return_exceptions=True,
                )

                if isinstance(results[0], Exception):
                    logger.error("Industry research failed for %s: %s", path.path_id, results[0])
                else:
                    industry_results.append(results[0])

                if isinstance(results[1], Exception):
                    logger.error("Academic research failed for %s: %s", path.path_id, results[1])
                else:
                    academic_results.append(results[1])

                if isinstance(results[2], Exception):
                    logger.error("Engineering analysis failed for %s: %s", path.path_id, results[2])
                else:
                    engineering_results.append(results[2])

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
            for i, path in enumerate(decomposition.paths):
                ind = industry_results[i] if i < len(industry_results) else None
                acad = academic_results[i] if i < len(academic_results) else None
                eng = engineering_results[i] if i < len(engineering_results) else None
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
            grouped.setdefault(pid, {})
            grouped[pid].setdefault(q.source, [])
            grouped[pid][q.source].append(q)
        return grouped

    def _write_outputs(self, report: ResearchReport, output_format: str) -> list[Path]:
        """Write report outputs in the requested format."""
        normalized = output_format.strip().lower()
        if normalized not in {"markdown", "docx", "both"}:
            raise ValueError(
                "Unsupported output format. Expected one of: markdown, docx, both"
            )

        paths: list[Path] = []
        if normalized in {"markdown", "both"}:
            markdown_path = self.output_dir / f"{report.session_id}.md"
            paths.append(self.markdown_reporter.generate(report, markdown_path))

        if normalized in {"docx", "both"}:
            docx_path = self.output_dir / f"{report.session_id}.docx"
            paths.append(self.docx_reporter.generate(report, docx_path))

        self.output_paths = paths
        return paths

    async def _cleanup(self) -> None:
        """Close API clients."""
        for client in [self.tavily, self.s2, self.github, self.arxiv, self.scraper]:
            try:
                await client.close()
            except Exception:
                pass
        try:
            self.llm.close()
        except Exception:
            pass
