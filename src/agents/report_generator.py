"""Agent that synthesizes all research findings into a final report."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.agents.base import BaseAgent
from src.models.plan import DecompositionResult
from src.models.industry import IndustryResearchResult
from src.models.academic import AcademicResearchResult
from src.models.engineering import EngineeringAnalysis
from src.models.maturity import MaturityAssessment
from src.models.reputation import ReputationReport
from src.models.report import (
    ResearchReport, TechnologyEntry, ImplementationWorkflow,
    WorkflowStep, FeasibilityAssessment,
)
from src.models.common import MaturityStage, SourceReference


class ReportGenerator(BaseAgent):
    """Synthesizes all research findings into a comprehensive report."""

    agent_name = "report_generator"

    MATURITY_MAP = {
        "early_prototype": MaturityStage.EARLY_PROTOTYPE,
        "development": MaturityStage.DEVELOPMENT,
        "mature": MaturityStage.MATURE,
        "cutting_edge": MaturityStage.CUTTING_EDGE,
        "academic_frontier": MaturityStage.ACADEMIC_FRONTIER,
    }

    def run(
        self,
        *,
        decomposition: DecompositionResult,
        industry_results: list[IndustryResearchResult],
        academic_results: list[AcademicResearchResult],
        engineering_results: list[EngineeringAnalysis],
        maturity_assessments: list[MaturityAssessment],
        reputation_report: ReputationReport,
        session_id: str = "",
    ) -> ResearchReport:
        """Generate final comprehensive report from all findings."""
        self.logger.info("Generating final report for: %s", decomposition.original_input[:80])

        # Build summaries for the LLM
        paths_summary = self._summarize_paths(decomposition)
        industry_summary = self._summarize_industry(industry_results)
        academic_summary = self._summarize_academic(academic_results)
        engineering_summary = self._summarize_engineering(engineering_results)
        reputation_summary = self._summarize_reputation(reputation_report)
        maturity_summary = self._summarize_maturity(maturity_assessments)

        # LLM synthesis
        result = self._call_llm_json(
            prompt=self._render_template(
                "synthesize_report",
                {
                    "original_input": decomposition.original_input,
                    "paths_summary": paths_summary,
                    "industry_summary": industry_summary,
                    "academic_summary": academic_summary,
                    "engineering_summary": engineering_summary,
                    "reputation_summary": reputation_summary,
                    "maturity_summary": maturity_summary,
                },
            ),
            temperature=0.3,
            max_tokens=16384,
        )

        # Build report
        report = ResearchReport(
            title=f"Technology Landscape: {decomposition.original_input[:100]}",
            session_id=session_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            original_input=decomposition.original_input,
            industry_findings=industry_results,
            academic_findings=academic_results,
            engineering_analyses=engineering_results,
            maturity_assessments=maturity_assessments,
            reputation_report=reputation_report,
            all_sources=self._collect_all_sources(
                industry_results, academic_results, engineering_results,
            ),
        )

        if result and isinstance(result, dict):
            report.executive_summary = result.get("executive_summary", "")
            report.recommendations = result.get("recommendations", "")

            for tech in result.get("technology_landscape", []):
                maturity_str = tech.get("maturity", "development")
                report.technology_landscape.append(TechnologyEntry(
                    name=tech.get("name", ""),
                    category=tech.get("category", ""),
                    maturity=self.MATURITY_MAP.get(maturity_str, MaturityStage.DEVELOPMENT),
                    description=tech.get("description", ""),
                    industry_score=float(tech.get("industry_score", 0)),
                    academic_score=float(tech.get("academic_score", 0)),
                    engineering_score=float(tech.get("engineering_score", 0)),
                    reputation_score=float(tech.get("reputation_score", 0)),
                ))

            for wf in result.get("workflows", []):
                steps = [
                    WorkflowStep(
                        step=s.get("step", 0),
                        description=s.get("description", ""),
                        technologies=s.get("technologies", []),
                        considerations=s.get("considerations", ""),
                    )
                    for s in wf.get("steps", [])
                ]
                report.workflows.append(ImplementationWorkflow(
                    name=wf.get("name", ""),
                    description=wf.get("description", ""),
                    steps=steps,
                    pros=wf.get("pros", []),
                    cons=wf.get("cons", []),
                ))

            for fa in result.get("feasibility_assessments", []):
                report.feasibility_assessments.append(FeasibilityAssessment(
                    path_id=fa.get("path_id", ""),
                    path_title=fa.get("path_title", ""),
                    overall_feasibility=float(fa.get("overall_feasibility", 0.5)),
                    technical_risk=fa.get("technical_risk", "medium"),
                    resource_requirements=fa.get("resource_requirements", ""),
                    time_estimate=fa.get("time_estimate", ""),
                    recommended=fa.get("recommended", False),
                    rationale=fa.get("rationale", ""),
                    key_challenges=fa.get("key_challenges", []),
                ))

        self.logger.info(
            "Report generated: %d techs, %d workflows, %d sources",
            len(report.technology_landscape),
            len(report.workflows),
            len(report.all_sources),
        )
        return report

    def _summarize_paths(self, decomp: DecompositionResult) -> str:
        parts = []
        for p in decomp.paths:
            parts.append(f"- {p.title} (priority={p.priority}): {p.description[:200]}")
        return "\n".join(parts)

    def _summarize_industry(self, results: list[IndustryResearchResult]) -> str:
        parts = []
        for r in results:
            parts.append(f"Path {r.path_id}: {len(r.products)} products, "
                         f"{len(r.repos)} repos, {len(r.blog_summaries)} blogs")
            for p in r.products[:3]:
                parts.append(f"  Product: {p.name} ({p.company}) - {p.description[:150]}")
            for repo in r.repos[:3]:
                parts.append(f"  Repo: {repo.name} ({repo.stars} stars)")
        return "\n".join(parts) if parts else "No industry data."

    def _summarize_academic(self, results: list[AcademicResearchResult]) -> str:
        parts = []
        for r in results:
            parts.append(f"Path {r.path_id}: {len(r.papers)} papers")
            for p in r.papers[:3]:
                parts.append(f"  Paper: \"{p.title}\" ({p.year}, {p.citation_count} cites)")
                if p.conclusions:
                    parts.append(f"    Conclusions: {p.conclusions[:200]}")
        return "\n".join(parts) if parts else "No academic data."

    def _summarize_engineering(self, results: list[EngineeringAnalysis]) -> str:
        parts = []
        for r in results:
            parts.append(f"Path {r.path_id}: {len(r.code_analyses)} repos analyzed")
            if r.deployment_assessment.deployment_complexity:
                parts.append(f"  Deployment: {r.deployment_assessment.deployment_complexity}")
            if r.implementation_recommendations:
                parts.append(f"  Recommendations: {r.implementation_recommendations[:200]}")
        return "\n".join(parts) if parts else "No engineering data."

    def _summarize_reputation(self, report: ReputationReport) -> str:
        parts = []
        for c in report.companies[:5]:
            parts.append(f"Company: {c.company_name} ({c.category}, score={c.reputation_score})")
        for lab in report.labs[:5]:
            parts.append(f"Lab: {lab.lab_name} @ {lab.institution} (score={lab.prestige_score})")
        if report.summary:
            parts.append(f"Summary: {report.summary[:300]}")
        return "\n".join(parts) if parts else "No reputation data."

    def _summarize_maturity(self, assessments: list[MaturityAssessment]) -> str:
        parts = []
        for a in assessments:
            parts.append(f"Path {a.path_id}: {len(a.technologies)} technologies")
            if a.mature_solutions:
                parts.append(f"  Mature: {', '.join(a.mature_solutions[:5])}")
            if a.cutting_edge:
                parts.append(f"  Cutting-edge: {', '.join(a.cutting_edge[:5])}")
            if a.academic_frontier:
                parts.append(f"  Academic: {', '.join(a.academic_frontier[:5])}")
        return "\n".join(parts) if parts else "No maturity data."

    def _collect_all_sources(
        self,
        industry: list[IndustryResearchResult],
        academic: list[AcademicResearchResult],
        engineering: list[EngineeringAnalysis],
    ) -> list[SourceReference]:
        """Collect all source references, deduplicated by URL."""
        seen_urls: set[str] = set()
        all_sources: list[SourceReference] = []
        for results_list in [industry, academic, engineering]:
            for r in results_list:
                for s in r.sources:
                    if s.url and s.url not in seen_urls:
                        seen_urls.add(s.url)
                        all_sources.append(s)
        return all_sources
