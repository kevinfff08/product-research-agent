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


class DecisionMatrixRow(BaseModel):
    """Decision matrix row comparing one research path or technology option."""

    option: str = ""
    path_id: str = ""
    user_value: str = ""
    technical_feasibility: str = ""
    maturity: str = ""
    ecosystem_strength: str = ""
    cost_risk: str = ""
    evidence_strength: str = ""
    verdict: str = ""


class EvidenceClaim(BaseModel):
    """A key report claim tied to supporting and conflicting evidence."""

    claim: str = ""
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    confidence: str = ""
    implication: str = ""
    source_urls: list[str] = Field(default_factory=list)


class PathTechDetail(BaseModel):
    """Deep technical analysis of a key technology within a research path."""

    name: str = ""
    what_it_is: str = ""
    how_it_works: str = ""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    implementation_notes: str = ""
    industry_evidence: str = ""
    academic_evidence: str = ""
    engineering_evidence: str = ""


class ProsConsSummary(BaseModel):
    """Strengths and weaknesses summary for a path."""

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    best_for: str = ""
    not_suitable_for: str = ""


class PathDeepAnalysis(BaseModel):
    """Deep analytical breakdown of one research path."""

    path_id: str = ""
    title: str = ""
    technical_overview: str = ""
    key_technologies_detail: list[PathTechDetail] = Field(default_factory=list)
    cross_references: str = ""
    pros_cons_summary: ProsConsSummary = Field(default_factory=ProsConsSummary)


class CrossAnalysis(BaseModel):
    """Cross-referencing analysis connecting industry, academic, and engineering evidence."""

    industry_academic_alignment: str = ""
    academic_engineering_gap: str = ""
    evidence_quality_overview: str = ""
    key_contradictions: list[str] = Field(default_factory=list)


class TechnologyRelationships(BaseModel):
    """Relationship map between technologies."""

    complementary_pairs: list[list[str]] = Field(default_factory=list)
    alternatives: list[list[str]] = Field(default_factory=list)
    dependency_chains: list[list[str]] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Final comprehensive research report."""

    title: str = ""
    session_id: str = ""
    generated_at: str = ""
    original_input: str = ""

    # Executive summary
    executive_summary: str = ""
    research_questions: list[str] = Field(default_factory=list)
    methodology_summary: str = ""
    decision_matrix: list[DecisionMatrixRow] = Field(default_factory=list)
    key_claims: list[EvidenceClaim] = Field(default_factory=list)

    # Deep analysis
    path_deep_analysis: list[PathDeepAnalysis] = Field(default_factory=list)
    cross_analysis: CrossAnalysis = Field(default_factory=CrossAnalysis)
    technology_relationships: TechnologyRelationships = Field(
        default_factory=TechnologyRelationships,
    )
    evidence_gaps: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence_assessment: str = ""
    recommended_strategy: str = ""

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
