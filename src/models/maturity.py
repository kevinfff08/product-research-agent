"""Technology maturity mapping models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.common import MaturityStage


class TechnologyMaturity(BaseModel):
    """Maturity assessment for a single technology."""

    technology: str = ""
    stage: MaturityStage = MaturityStage.DEVELOPMENT
    evidence: str = Field(
        default="",
        description="Evidence supporting this maturity classification",
    )
    adoption_signals: list[str] = Field(default_factory=list)
    trend: str = ""  # "emerging", "growing", "stable", "declining"
    first_appeared: str = ""
    years_in_production: int = 0


class MaturityAssessment(BaseModel):
    """Complete maturity map for a research path."""

    path_id: str = ""
    technologies: list[TechnologyMaturity] = Field(default_factory=list)
    overall_maturity_summary: str = ""

    # Technologies grouped by stage
    early_prototypes: list[str] = Field(default_factory=list)
    in_development: list[str] = Field(default_factory=list)
    mature_solutions: list[str] = Field(default_factory=list)
    cutting_edge: list[str] = Field(default_factory=list)
    academic_frontier: list[str] = Field(default_factory=list)
