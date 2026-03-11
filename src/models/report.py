"""Final research report model."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.common import MaturityStage, SourceReference
from src.models.industry import IndustryResearchResult
from src.models.academic import AcademicResearchResult
from src.models.engineering import EngineeringAnalysis
from src.models.maturity import MaturityAssessment
from src.models.reputation import ReputationReport


class TechnologyEntry(BaseModel):
    """A single technology in the landscape analysis."""

    name: str = ""
    category: str = ""  # "full_stack", "frontend", "backend", "ml", "infra", "data", "devops"
    maturity: MaturityStage = MaturityStage.DEVELOPMENT
    description: str = ""
    industry_score: float = Field(default=0.0, ge=0.0, le=1.0)
    academic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    engineering_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reputation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sources: list[SourceReference] = Field(default_factory=list)


class WorkflowStep(BaseModel):
    """A step in an implementation workflow."""

    step: int = 0
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    considerations: str = ""


class ImplementationWorkflow(BaseModel):
    """A possible implementation workflow."""

    name: str = ""
    description: str = ""
    steps: list[WorkflowStep] = Field(default_factory=list)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


class FeasibilityAssessment(BaseModel):
    """Feasibility assessment for one research path."""

    path_id: str = ""
    path_title: str = ""
    overall_feasibility: float = Field(default=0.5, ge=0.0, le=1.0)
    technical_risk: str = ""  # "low", "medium", "high", "very_high"
    resource_requirements: str = ""
    time_estimate: str = ""
    recommended: bool = False
    rationale: str = ""
    key_challenges: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Final comprehensive research report."""

    title: str = ""
    session_id: str = ""
    generated_at: str = ""
    original_input: str = ""

    # Executive summary
    executive_summary: str = ""

    # Technology landscape
    technology_landscape: list[TechnologyEntry] = Field(default_factory=list)

    # Implementation workflows
    workflows: list[ImplementationWorkflow] = Field(default_factory=list)

    # Maturity map
    maturity_assessments: list[MaturityAssessment] = Field(default_factory=list)

    # Detailed findings by type
    industry_findings: list[IndustryResearchResult] = Field(default_factory=list)
    academic_findings: list[AcademicResearchResult] = Field(default_factory=list)
    engineering_analyses: list[EngineeringAnalysis] = Field(default_factory=list)

    # Reputation
    reputation_report: ReputationReport = Field(default_factory=ReputationReport)

    # Feasibility
    feasibility_assessments: list[FeasibilityAssessment] = Field(default_factory=list)

    # All sources
    all_sources: list[SourceReference] = Field(default_factory=list)

    # Recommendations
    recommendations: str = ""
