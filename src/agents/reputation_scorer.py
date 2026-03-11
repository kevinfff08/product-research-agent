"""Agent that scores reputation and credibility of research sources."""

from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.models.industry import IndustryResearchResult
from src.models.academic import AcademicResearchResult
from src.models.reputation import (
    CompanyReputation, LabPrestige, VenuePrestige,
    ReputationScore, ReputationReport,
)


class ReputationScorer(BaseAgent):
    """Scores reputation and credibility of entities found during research."""

    agent_name = "reputation_scorer"

    def run(
        self,
        *,
        industry_results: list[IndustryResearchResult],
        academic_results: list[AcademicResearchResult],
    ) -> ReputationReport:
        """Score reputation of all discovered entities."""
        self.logger.info("Scoring reputation across %d industry + %d academic results",
                         len(industry_results), len(academic_results))

        industry_summary = self._build_industry_summary(industry_results)
        academic_summary = self._build_academic_summary(academic_results)

        result = self._call_llm_json(
            prompt=self._render_template(
                "score_reputation",
                {
                    "industry_summary": industry_summary,
                    "academic_summary": academic_summary,
                },
            ),
            temperature=0.2,
        )

        if not result or not isinstance(result, dict):
            self.logger.warning("Reputation scoring failed, returning empty report")
            return ReputationReport()

        report = ReputationReport(
            summary=result.get("summary", ""),
        )

        for c in result.get("companies", []):
            report.companies.append(CompanyReputation(
                **{k: v for k, v in c.items() if k in CompanyReputation.model_fields}
            ))

        for lab in result.get("labs", []):
            report.labs.append(LabPrestige(
                **{k: v for k, v in lab.items() if k in LabPrestige.model_fields}
            ))

        for v in result.get("venues", []):
            report.venues.append(VenuePrestige(
                **{k: v_ for k, v_ in v.items() if k in VenuePrestige.model_fields}
            ))

        for ts in result.get("technology_scores", []):
            report.technology_scores.append(ReputationScore(
                **{k: v for k, v in ts.items() if k in ReputationScore.model_fields}
            ))

        self.logger.info(
            "Reputation scored: %d companies, %d labs, %d venues, %d tech scores",
            len(report.companies), len(report.labs),
            len(report.venues), len(report.technology_scores),
        )
        return report

    def _build_industry_summary(self, results: list[IndustryResearchResult]) -> str:
        """Build a text summary of industry findings for the LLM."""
        parts = []
        for r in results:
            for p in r.products[:5]:
                parts.append(f"Product: {p.name} by {p.company} - {p.description[:200]}")
            for c in r.companies[:5]:
                parts.append(f"Company: {c.name} ({c.size}) - {c.reputation_notes[:200]}")
            for repo in r.repos[:5]:
                parts.append(
                    f"Repo: {repo.name} ({repo.stars} stars, {repo.language}) - {repo.description[:200]}"
                )
        return "\n".join(parts) if parts else "No industry data collected."

    def _build_academic_summary(self, results: list[AcademicResearchResult]) -> str:
        """Build a text summary of academic findings for the LLM."""
        parts = []
        for r in results:
            for p in r.papers[:5]:
                venue_info = f" @ {p.venue}" if p.venue else ""
                lab_info = f" [{p.known_lab}]" if p.known_lab else ""
                parts.append(
                    f"Paper: \"{p.title}\" ({p.year}{venue_info}, {p.citation_count} cites{lab_info})"
                )
        return "\n".join(parts) if parts else "No academic data collected."
