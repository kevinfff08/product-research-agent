"""Generates Markdown report from ResearchReport model."""

from __future__ import annotations

from pathlib import Path

from src.models.report import ResearchReport
from src.models.common import MaturityStage
from src.logging_config import get_logger

logger = get_logger("reporters.markdown")


class MarkdownReporter:
    """Renders a ResearchReport into a Markdown file."""

    def generate(self, report: ResearchReport, output_path: Path | str) -> Path:
        """Generate Markdown report and write to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        md = self._render(report)
        output_path.write_text(md, encoding="utf-8")

        logger.info("Markdown report written to %s (%d chars)", output_path, len(md))
        return output_path

    def _render(self, report: ResearchReport) -> str:
        """Render the full report as Markdown."""
        sections = [
            self._header(report),
            self._executive_summary(report),
            self._research_method(report),
            self._decision_matrix(report),
            self._key_claims(report),
            self._technology_landscape(report),
            self._maturity_map(report),
            self._implementation_workflows(report),
            self._industry_findings(report),
            self._academic_findings(report),
            self._engineering_analysis(report),
            self._reputation_analysis(report),
            self._feasibility(report),
            self._confidence_and_gaps(report),
            self._recommendations(report),
            self._sources(report),
        ]
        return "\n\n".join(s for s in sections if s)

    def _header(self, report: ResearchReport) -> str:
        lines = [
            f"# {report.title}",
            "",
            f"**Generated:** {report.generated_at}  ",
            f"**Session:** {report.session_id}  ",
            f"**Input:** {report.original_input}",
        ]
        return "\n".join(lines)

    def _executive_summary(self, report: ResearchReport) -> str:
        if not report.executive_summary:
            return ""
        return f"## Executive Summary\n\n{report.executive_summary}"

    def _research_method(self, report: ResearchReport) -> str:
        if not report.research_questions and not report.methodology_summary:
            return ""
        lines = ["## Research Question and Method", ""]
        if report.research_questions:
            lines.append("**Research questions:**")
            for question in report.research_questions:
                lines.append(f"- {question}")
            lines.append("")
        if report.methodology_summary:
            lines.append(report.methodology_summary)
        return "\n".join(lines)

    def _decision_matrix(self, report: ResearchReport) -> str:
        if not report.decision_matrix:
            return ""
        lines = ["## Decision Matrix", ""]
        lines.append(
            "| Option | User Value | Feasibility | Maturity | Ecosystem | Cost/Risk | Evidence | Verdict |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in report.decision_matrix:
            option = row.option or row.path_id
            lines.append(
                f"| {option} | {row.user_value} | {row.technical_feasibility} | "
                f"{row.maturity} | {row.ecosystem_strength} | {row.cost_risk} | "
                f"{row.evidence_strength} | {row.verdict} |"
            )
        return "\n".join(lines)

    def _key_claims(self, report: ResearchReport) -> str:
        if not report.key_claims:
            return ""
        lines = ["## Key Claims and Evidence", ""]
        for claim in report.key_claims:
            lines.append(f"### {claim.claim}")
            if claim.confidence:
                lines.append(f"**Confidence:** {claim.confidence}")
            if claim.supporting_evidence:
                lines.append("\n**Supporting evidence:**")
                for evidence in claim.supporting_evidence:
                    lines.append(f"- {evidence}")
            if claim.contradicting_evidence:
                lines.append("\n**Contradicting evidence or caveats:**")
                for evidence in claim.contradicting_evidence:
                    lines.append(f"- {evidence}")
            if claim.implication:
                lines.append(f"\n**Implication:** {claim.implication}")
            if claim.source_urls:
                links = ", ".join(f"[source]({url})" for url in claim.source_urls)
                lines.append(f"\n**Sources:** {links}")
            lines.append("")
        return "\n".join(lines)

    def _technology_landscape(self, report: ResearchReport) -> str:
        if not report.technology_landscape:
            return ""
        lines = ["## Technology Landscape", ""]
        lines.append("| Technology | Category | Maturity | Industry | Academic | Engineering | Reputation |")
        lines.append("|---|---|---|---|---|---|---|")
        for t in report.technology_landscape:
            lines.append(
                f"| {t.name} | {t.category} | {t.maturity.value} | "
                f"{t.industry_score:.1f} | {t.academic_score:.1f} | "
                f"{t.engineering_score:.1f} | {t.reputation_score:.1f} |"
            )
        # Details
        lines.append("")
        for t in report.technology_landscape:
            if t.description:
                lines.append(f"**{t.name}**: {t.description}")
                lines.append("")
        return "\n".join(lines)

    def _maturity_map(self, report: ResearchReport) -> str:
        if not report.maturity_assessments:
            return ""
        lines = ["## Technology Maturity Map", ""]
        stage_labels = {
            MaturityStage.EARLY_PROTOTYPE: "Early Prototype",
            MaturityStage.DEVELOPMENT: "In Development",
            MaturityStage.MATURE: "Mature",
            MaturityStage.CUTTING_EDGE: "Cutting Edge",
            MaturityStage.ACADEMIC_FRONTIER: "Academic Frontier",
        }
        for assessment in report.maturity_assessments:
            lines.append(f"### Path: {assessment.path_id}")
            if assessment.overall_maturity_summary:
                lines.append(f"\n{assessment.overall_maturity_summary}\n")
            for tech in assessment.technologies:
                trend_badge = f" ({tech.trend})" if tech.trend else ""
                lines.append(
                    f"- **{tech.technology}** [{stage_labels.get(tech.stage, tech.stage.value)}]"
                    f"{trend_badge}: {tech.evidence[:200]}"
                )
            lines.append("")
        return "\n".join(lines)

    def _implementation_workflows(self, report: ResearchReport) -> str:
        if not report.workflows:
            return ""
        lines = ["## Implementation Workflows", ""]
        for i, wf in enumerate(report.workflows, 1):
            lines.append(f"### Workflow {i}: {wf.name}")
            if wf.description:
                lines.append(f"\n{wf.description}\n")
            lines.append("**Steps:**\n")
            for step in wf.steps:
                techs = ", ".join(step.technologies) if step.technologies else ""
                lines.append(f"{step.step}. {step.description}")
                if techs:
                    lines.append(f"   - Technologies: {techs}")
                if step.considerations:
                    lines.append(f"   - Considerations: {step.considerations}")
            lines.append("")
            if wf.pros:
                lines.append("**Pros:** " + "; ".join(wf.pros))
            if wf.cons:
                lines.append("**Cons:** " + "; ".join(wf.cons))
            lines.append("")
        return "\n".join(lines)

    def _industry_findings(self, report: ResearchReport) -> str:
        if not report.industry_findings:
            return ""
        lines = ["## Industry Findings", ""]
        for finding in report.industry_findings:
            lines.append(f"### Research Path: {finding.path_id}")
            if finding.products:
                lines.append("\n**Products:**\n")
                for p in finding.products:
                    os_badge = " (Open Source)" if p.is_open_source else ""
                    lines.append(f"- **{p.name}**{os_badge} by {p.company}: {p.description[:200]}")
                    if p.capabilities:
                        lines.append(f"  - Capabilities: {', '.join(p.capabilities[:5])}")
            if finding.repos:
                lines.append("\n**Notable Repositories:**\n")
                for r in finding.repos:
                    lines.append(
                        f"- [{r.name}]({r.repo_url}) - {r.stars} stars, {r.language}"
                    )
                    if r.description:
                        lines.append(f"  {r.description[:200]}")
            if finding.blog_summaries:
                lines.append("\n**Key Blog Insights:**\n")
                for b in finding.blog_summaries:
                    lines.append(f"- [{b.title}]({b.url})")
                    for point in b.key_points[:3]:
                        lines.append(f"  - {point}")
            lines.append("")
        return "\n".join(lines)

    def _academic_findings(self, report: ResearchReport) -> str:
        if not report.academic_findings:
            return ""
        lines = ["## Academic Research", ""]
        for finding in report.academic_findings:
            lines.append(f"### Research Path: {finding.path_id}")
            for paper in finding.papers:
                venue = f" @ {paper.venue}" if paper.venue else ""
                lines.append(f"\n**{paper.title}** ({paper.year}{venue}, {paper.citation_count} citations)")
                lines.append(f"- Authors: {', '.join(paper.authors[:5])}")
                if paper.principles:
                    lines.append(f"- **Principles:** {paper.principles[:300]}")
                if paper.methods:
                    lines.append(f"- **Methods:** {paper.methods[:300]}")
                if paper.conclusions:
                    lines.append(f"- **Conclusions:** {paper.conclusions[:300]}")
                if paper.deficiencies:
                    lines.append(f"- **Limitations:** {paper.deficiencies[:300]}")
            lines.append("")
        return "\n".join(lines)

    def _engineering_analysis(self, report: ResearchReport) -> str:
        if not report.engineering_analyses:
            return ""
        lines = ["## Engineering Analysis", ""]
        for eng in report.engineering_analyses:
            lines.append(f"### Path: {eng.path_id}")
            if eng.deployment_assessment.deployment_complexity:
                da = eng.deployment_assessment
                lines.append(f"\n**Deployment Complexity:** {da.deployment_complexity}")
                if da.infrastructure_requirements:
                    lines.append(f"**Infrastructure:** {', '.join(da.infrastructure_requirements)}")
                if da.risks:
                    lines.append(f"**Risks:** {', '.join(da.risks)}")
            if eng.implementation_recommendations:
                lines.append(f"\n**Recommendations:** {eng.implementation_recommendations[:500]}")
            if eng.technology_stack_recommendation:
                lines.append(f"\n**Recommended Stack:** {', '.join(eng.technology_stack_recommendation)}")
            lines.append("")
        return "\n".join(lines)

    def _reputation_analysis(self, report: ResearchReport) -> str:
        rep = report.reputation_report
        if not rep.companies and not rep.labs and not rep.technology_scores:
            return ""
        lines = ["## Reputation & Credibility", ""]
        if rep.summary:
            lines.append(rep.summary)
            lines.append("")
        if rep.companies:
            lines.append("**Companies:**\n")
            for c in rep.companies:
                lines.append(
                    f"- {c.company_name} ({c.category}): "
                    f"score={c.reputation_score:.2f}, sentiment={c.user_sentiment}"
                )
        if rep.labs:
            lines.append("\n**Research Labs:**\n")
            for lab in rep.labs:
                lines.append(
                    f"- {lab.lab_name} @ {lab.institution}: score={lab.prestige_score:.2f}"
                )
        if rep.technology_scores:
            lines.append("\n**Technology Credibility Scores:**\n")
            lines.append("| Technology | Source | Recency | Author | Cross-val | Community | Overall |")
            lines.append("|---|---|---|---|---|---|---|")
            for ts in rep.technology_scores:
                lines.append(
                    f"| {ts.name} | {ts.source_type_score:.1f} | {ts.recency_score:.1f} | "
                    f"{ts.author_reputation_score:.1f} | {ts.cross_validation_score:.1f} | "
                    f"{ts.community_reception_score:.1f} | {ts.overall_score:.2f} |"
                )
        return "\n".join(lines)

    def _feasibility(self, report: ResearchReport) -> str:
        if not report.feasibility_assessments:
            return ""
        lines = ["## Feasibility Assessment", ""]
        for fa in report.feasibility_assessments:
            rec = "Recommended" if fa.recommended else "Not Recommended"
            lines.append(f"### {fa.path_title} ({rec})")
            lines.append(f"- **Feasibility Score:** {fa.overall_feasibility:.0%}")
            lines.append(f"- **Technical Risk:** {fa.technical_risk}")
            if fa.resource_requirements:
                lines.append(f"- **Resources:** {fa.resource_requirements}")
            if fa.time_estimate:
                lines.append(f"- **Time Estimate:** {fa.time_estimate}")
            if fa.rationale:
                lines.append(f"- **Rationale:** {fa.rationale}")
            if fa.key_challenges:
                lines.append("- **Key Challenges:** " + "; ".join(fa.key_challenges))
            lines.append("")
        return "\n".join(lines)

    def _recommendations(self, report: ResearchReport) -> str:
        if not report.recommendations and not report.recommended_strategy:
            return ""
        body = report.recommended_strategy or report.recommendations
        if report.recommended_strategy and report.recommendations:
            body = f"{report.recommended_strategy}\n\n{report.recommendations}"
        return f"## Recommendations\n\n{body}"

    def _confidence_and_gaps(self, report: ResearchReport) -> str:
        if not report.confidence_assessment and not report.evidence_gaps and not report.assumptions:
            return ""
        lines = ["## Confidence, Gaps, and Assumptions", ""]
        if report.confidence_assessment:
            lines.append(report.confidence_assessment)
            lines.append("")
        if report.evidence_gaps:
            lines.append("**Evidence gaps:**")
            for gap in report.evidence_gaps:
                lines.append(f"- {gap}")
            lines.append("")
        if report.assumptions:
            lines.append("**Assumptions:**")
            for assumption in report.assumptions:
                lines.append(f"- {assumption}")
        return "\n".join(lines)

    def _sources(self, report: ResearchReport) -> str:
        if not report.all_sources:
            return ""
        lines = ["## Sources", ""]
        for s in report.all_sources:
            source_type = s.source_type.value if s.source_type else "web"
            lines.append(f"- [{s.title}]({s.url}) [{source_type}]")
        return "\n".join(lines)
