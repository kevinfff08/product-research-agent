"""Agent that synthesizes all research findings into a final report."""

from __future__ import annotations

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
    WorkflowStep, FeasibilityAssessment, DecisionMatrixRow, EvidenceClaim,
    PathDeepAnalysis, PathTechDetail, ProsConsSummary,
    CrossAnalysis, TechnologyRelationships,
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
        """Generate final report via three lightweight LLM calls.

        Instead of one mega-call that times out, we split into:
        1. Core – executive summary, methodology, decision matrix, key claims
        2. Landscape – tech landscape, relationships, workflows, feasibility
        3. Synthesis – cross-analysis, final strategy, recommendations

        Each call is small enough to complete reliably within 120s.
        """
        self.logger.info("Generating final report for: %s", decomposition.original_input[:80])

        # --- Map phase: LLM digests each path independently ---
        digests = self._build_path_digests(
            decomposition, industry_results, academic_results,
            engineering_results, maturity_assessments,
        )

        paths_summary = self._summarize_paths(decomposition)
        industry_summary = self._summarize_industry(industry_results)
        academic_summary = self._summarize_academic(academic_results)
        engineering_summary = self._summarize_engineering(engineering_results)
        maturity_summary = self._summarize_maturity(maturity_assessments)
        reputation_summary = self._summarize_reputation(reputation_report)

        shared_vars = {
            "original_input": decomposition.original_input,
            "paths_summary": paths_summary,
            "path_digests": digests,
            "industry_summary": industry_summary,
            "academic_summary": academic_summary,
            "engineering_summary": engineering_summary,
            "reputation_summary": reputation_summary,
            "maturity_summary": maturity_summary,
        }

        # --- Reduce phase: three parallel lightweight calls ---
        core = self._call_llm_json(
            prompt=self._render_template("synthesize_core", shared_vars),
            temperature=0.3, max_tokens=8192,
        )
        landscape = self._call_llm_json(
            prompt=self._render_template("synthesize_landscape", shared_vars),
            temperature=0.3, max_tokens=8192,
        )
        synthesis = self._call_llm_json(
            prompt=self._render_template("synthesize_synthesis", shared_vars),
            temperature=0.3, max_tokens=8192,
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

        # Populate from core call
        if core and isinstance(core, dict):
            report.executive_summary = core.get("executive_summary", "")
            report.research_questions = self._string_list(core.get("research_questions", []))
            report.methodology_summary = core.get("methodology_summary", "")
            for row in core.get("decision_matrix", []):
                report.decision_matrix.append(DecisionMatrixRow(
                    option=row.get("option", "") or row.get("path_title", ""),
                    path_id=row.get("path_id", ""),
                    user_value=row.get("user_value", ""),
                    technical_feasibility=row.get("technical_feasibility", ""),
                    maturity=row.get("maturity", ""),
                    ecosystem_strength=row.get("ecosystem_strength", ""),
                    cost_risk=row.get("cost_risk", ""),
                    evidence_strength=row.get("evidence_strength", ""),
                    verdict=row.get("verdict", ""),
                ))
            for claim in core.get("key_claims", []):
                report.key_claims.append(EvidenceClaim(
                    claim=claim.get("claim", ""),
                    supporting_evidence=self._string_list(claim.get("supporting_evidence", [])),
                    contradicting_evidence=self._string_list(claim.get("contradicting_evidence", [])),
                    confidence=claim.get("confidence", ""),
                    implication=claim.get("implication", ""),
                    source_urls=self._string_list(claim.get("source_urls", [])),
                ))
            report.evidence_gaps = self._string_list(core.get("evidence_gaps", []))
            report.assumptions = self._string_list(core.get("assumptions", []))
            report.confidence_assessment = core.get("confidence_assessment", "")

        # Populate from landscape call
        if landscape and isinstance(landscape, dict):
            for tech in landscape.get("technology_landscape", []):
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
            tr = landscape.get("technology_relationships", {}) or {}
            report.technology_relationships = TechnologyRelationships(
                complementary_pairs=tr.get("complementary_pairs", []),
                alternatives=tr.get("alternatives", []),
                dependency_chains=tr.get("dependency_chains", []),
            )
            for wf in landscape.get("workflows", []):
                steps = [
                    WorkflowStep(
                        step=s.get("step", 0), description=s.get("description", ""),
                        technologies=s.get("technologies", []),
                        considerations=s.get("considerations", ""),
                    ) for s in wf.get("steps", [])
                ]
                report.workflows.append(ImplementationWorkflow(
                    name=wf.get("name", ""), description=wf.get("description", ""),
                    steps=steps, pros=wf.get("pros", []), cons=wf.get("cons", []),
                ))
            for fa in landscape.get("feasibility_assessments", []):
                report.feasibility_assessments.append(FeasibilityAssessment(
                    path_id=fa.get("path_id", ""), path_title=fa.get("path_title", ""),
                    overall_feasibility=float(fa.get("overall_feasibility", 0.5)),
                    technical_risk=fa.get("technical_risk", "medium"),
                    resource_requirements=fa.get("resource_requirements", ""),
                    time_estimate=fa.get("time_estimate", ""),
                    recommended=fa.get("recommended", False),
                    rationale=fa.get("rationale", ""),
                    key_challenges=fa.get("key_challenges", []),
                ))

        # Populate from synthesis call
        if synthesis and isinstance(synthesis, dict):
            report.recommended_strategy = synthesis.get("recommended_strategy", "")
            report.recommendations = synthesis.get("recommendations", "")

        # Build path deep analysis from digests (already LLM-quality)
        for path in decomposition.paths:
            pid = path.path_id
            digest_block = digests.split(f"### {path.title} (path_id={pid})")[-1] if pid in digests else ""
            digest_block = digest_block.split("### ")[0].strip() if digest_block else ""
            report.path_deep_analysis.append(PathDeepAnalysis(
                path_id=pid, title=path.title, technical_overview=digest_block[:2000],
            ))

        self.logger.info(
            "Report generated: %d techs, %d workflows, %d sources",
            len(report.technology_landscape), len(report.workflows), len(report.all_sources),
        )
        return report

    @staticmethod
    def _string_list(value: object) -> list[str]:
        """Coerce an LLM value into a compact list of strings."""
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value]
        return []

    _DIGEST_SKIP_THRESHOLD = 800   # chars: raw data shorter than this skips LLM digestion
    _DIGEST_BASE_WORDS = 200       # minimum word count for a short path
    _DIGEST_RICH_WORDS = 700       # max word count for a data-rich path

    def _build_path_digests(
        self,
        decomposition: DecompositionResult,
        industry_results: list[IndustryResearchResult],
        academic_results: list[AcademicResearchResult],
        engineering_results: list[EngineeringAnalysis],
        maturity_assessments: list[MaturityAssessment],
    ) -> str:
        """Map phase: digest each path's raw data into a concise analysis.

        - Paths with little data (< 800 chars) skip the LLM call entirely
          and pass through the raw text directly.
        - Richer paths get proportionally longer digests (200-700 words),
          guided by the amount of raw data and path priority.
        - The LLM is free to vary digest length within that range based on
          how much substantive content the path actually has.
        """
        digests_parts: list[str] = []
        for path in decomposition.paths:
            pid = path.path_id

            # Gather raw data for this specific path
            ind = next((r for r in industry_results if r.path_id == pid), None)
            acad = next((r for r in academic_results if r.path_id == pid), None)
            eng = next((r for r in engineering_results if r.path_id == pid), None)
            mat = next((r for r in maturity_assessments if r.path_id == pid), None)

            raw_text = (
                f"路线: {path.title}\n"
                f"描述: {path.description}\n"
                f"关键技术: {', '.join(path.technologies_needed[:8])}\n"
                f"关键问题: {'; '.join(path.key_questions[:4])}\n\n"
            )
            if ind:
                raw_text += f"产业发现: {len(ind.products)}产品, {len(ind.repos)}仓库, "
                raw_text += f"市场趋势: {ind.market_trends[:300]}\n"
                for p in ind.products[:5]:
                    raw_text += (
                        f"  - {p.name} ({p.company}): {p.description[:200]}; "
                        f"能力: {', '.join(p.capabilities[:5])}\n"
                    )
            if acad:
                raw_text += f"\n学术发现: {len(acad.papers)}篇论文\n"
                for p in acad.papers[:6]:
                    raw_text += (
                        f"  - \"{p.title}\" ({p.year}, {p.venue}, "
                        f"{p.citation_count}引用): {p.conclusions[:300]}\n"
                    )
            if eng:
                raw_text += f"\n工程发现: {len(eng.code_analyses)}个仓库; "
                raw_text += f"部署复杂度: {eng.deployment_assessment.deployment_complexity}; "
                raw_text += f"推荐: {eng.implementation_recommendations[:300]}\n"
            if mat:
                stages = []
                if mat.mature_solutions:
                    stages.append(f"成熟: {', '.join(mat.mature_solutions[:5])}")
                if mat.cutting_edge:
                    stages.append(f"前沿: {', '.join(mat.cutting_edge[:5])}")
                if mat.academic_frontier:
                    stages.append(f"学术: {', '.join(mat.academic_frontier[:5])}")
                if mat.early_prototypes:
                    stages.append(f"早期: {', '.join(mat.early_prototypes[:5])}")
                raw_text += f"\n成熟度: {' | '.join(stages)}\n"

            # Decide whether to digest or pass through
            if len(raw_text) <= self._DIGEST_SKIP_THRESHOLD:
                # Data is sparse — skip LLM, pass through raw
                self.logger.debug(
                    "Path %s: %d chars ≤ threshold, skipping digest LLM call",
                    pid, len(raw_text),
                )
                digests_parts.append(
                    f"### {path.title} (path_id={pid})\n{raw_text}\n"
                )
                continue

            # Determine digest word budget: scale with raw data size and
            # path priority so richer / higher-priority paths get more space.
            data_ratio = min(len(raw_text) / 4000.0, 1.0)  # 0..1
            priority = float(path.priority)
            budget = int(
                self._DIGEST_BASE_WORDS
                + (self._DIGEST_RICH_WORDS - self._DIGEST_BASE_WORDS)
                * data_ratio * (0.5 + 0.5 * priority)
            )
            digest_result = self._call_llm_json(
                prompt=(
                    f"你是一位技术分析师。请将以下调研数据整理为一段精炼的中文分析摘要。"
                    f"保留所有关键信息（产品名、技术名、论文标题、数据、矛盾），"
                    f"去掉冗余描述。大约{budget}字，根据内容丰瘠可上下浮动20%。\n\n"
                    f"{raw_text}"
                ),
                system="只返回JSON：{\"digest\": \"摘要内容\"}",
                temperature=0.2,
                max_tokens=max(800, int(budget * 2.5)),
            )

            digest_text = ""
            if digest_result and isinstance(digest_result, dict):
                digest_text = str(digest_result.get("digest", ""))

            digests_parts.append(
                f"### {path.title} (path_id={pid})\n{digest_text}\n"
            )

        return "\n".join(digests_parts)

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
