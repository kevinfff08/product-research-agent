"""Agent that maps technologies to maturity stages."""

from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.models.plan import ResearchPath
from src.models.industry import IndustryResearchResult
from src.models.academic import AcademicResearchResult
from src.models.engineering import EngineeringAnalysis
from src.models.maturity import TechnologyMaturity, MaturityAssessment
from src.models.common import MaturityStage


class MaturityMapper(BaseAgent):
    """Maps discovered technologies to maturity stages."""

    agent_name = "maturity_mapper"

    STAGE_MAP = {
        "early_prototype": MaturityStage.EARLY_PROTOTYPE,
        "development": MaturityStage.DEVELOPMENT,
        "mature": MaturityStage.MATURE,
        "cutting_edge": MaturityStage.CUTTING_EDGE,
        "academic_frontier": MaturityStage.ACADEMIC_FRONTIER,
    }

    def run(
        self,
        *,
        path: ResearchPath,
        industry_result: IndustryResearchResult | None = None,
        academic_result: AcademicResearchResult | None = None,
        engineering_result: EngineeringAnalysis | None = None,
    ) -> MaturityAssessment:
        """Map technologies from research findings to maturity stages."""
        self.logger.info("Mapping maturity for path: %s", path.title)

        technologies = self._collect_technologies(
            industry_result, academic_result, engineering_result,
        )

        if not technologies:
            self.logger.warning("No technologies found for maturity mapping")
            return MaturityAssessment(path_id=path.path_id)

        technologies_json = json.dumps(technologies, indent=2)

        result = self._call_llm_json(
            prompt=self._render_template(
                "assess_maturity",
                {
                    "path_title": path.title,
                    "path_description": path.description,
                    "technologies_json": technologies_json,
                },
            ),
            temperature=0.2,
        )

        if not result or not isinstance(result, dict):
            self.logger.warning("Maturity mapping failed")
            return MaturityAssessment(path_id=path.path_id)

        assessment = MaturityAssessment(
            path_id=path.path_id,
            overall_maturity_summary=result.get("overall_maturity_summary", ""),
            early_prototypes=result.get("early_prototypes", []),
            in_development=result.get("in_development", []),
            mature_solutions=result.get("mature_solutions", []),
            cutting_edge=result.get("cutting_edge", []),
            academic_frontier=result.get("academic_frontier", []),
        )

        for t in result.get("technologies", []):
            stage_str = t.get("stage", "development")
            stage = self.STAGE_MAP.get(stage_str, MaturityStage.DEVELOPMENT)
            assessment.technologies.append(TechnologyMaturity(
                technology=t.get("technology", ""),
                stage=stage,
                evidence=t.get("evidence", ""),
                adoption_signals=t.get("adoption_signals", []),
                trend=t.get("trend", ""),
                years_in_production=t.get("years_in_production", 0),
            ))

        self.logger.info(
            "Maturity mapped: %d technologies across stages",
            len(assessment.technologies),
        )
        return assessment

    def _collect_technologies(
        self,
        industry: IndustryResearchResult | None,
        academic: AcademicResearchResult | None,
        engineering: EngineeringAnalysis | None,
    ) -> list[dict]:
        """Collect technology names and evidence from all sources."""
        techs: dict[str, dict] = {}

        if industry:
            for product in industry.products:
                for cap in product.capabilities:
                    name = cap.lower().strip()
                    if name and len(name) > 2:
                        techs.setdefault(name, {"name": name, "sources": []})
                        techs[name]["sources"].append(f"product:{product.name}")
            for repo in industry.repos:
                if repo.language:
                    name = repo.language.lower()
                    techs.setdefault(name, {"name": name, "sources": []})
                    techs[name]["sources"].append(f"repo:{repo.name}")
                for topic in repo.topics:
                    name = topic.lower().strip()
                    if name and len(name) > 2:
                        techs.setdefault(name, {"name": name, "sources": []})
                        techs[name]["sources"].append(f"repo:{repo.name}")

        if engineering:
            for ca in engineering.code_analyses:
                for tech in ca.tech_stack:
                    name = tech.lower().strip()
                    if name:
                        techs.setdefault(name, {"name": name, "sources": []})
                        techs[name]["sources"].append(f"code:{ca.repo_url}")

        if academic:
            for paper in academic.papers:
                if paper.methods:
                    name = f"method:{paper.title[:50]}"
                    techs.setdefault(name, {"name": name, "sources": []})
                    techs[name]["sources"].append(f"paper:{paper.title[:50]}")

        return list(techs.values())[:30]  # Limit
