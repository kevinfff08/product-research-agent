"""Engineering analysis data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.common import SourceReference


class CodeAnalysis(BaseModel):
    """Detailed analysis of a code repository's engineering quality."""

    repo_url: str = ""
    architecture_summary: str = ""
    tech_stack: list[str] = Field(default_factory=list)
    code_patterns: str = ""
    test_coverage_notes: str = ""
    documentation_quality: str = ""
    api_design_notes: str = ""
    scalability_notes: str = ""


class DeploymentAssessment(BaseModel):
    """Assessment of how deployable a solution is."""

    approach: str = ""
    deployment_complexity: str = ""  # "trivial", "moderate", "complex", "very_complex"
    infrastructure_requirements: list[str] = Field(default_factory=list)
    estimated_setup_effort: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class EngineeringAnalysis(BaseModel):
    """Complete engineering analysis result for one research path."""

    path_id: str = ""
    code_analyses: list[CodeAnalysis] = Field(default_factory=list)
    deployment_assessment: DeploymentAssessment = Field(default_factory=DeploymentAssessment)
    implementation_recommendations: str = ""
    technology_stack_recommendation: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
