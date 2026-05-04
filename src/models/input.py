"""Input models for research requests."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.models.common import ResearchWeight


class ResearchRequest(BaseModel):
    """User's research request."""

    raw_input: str = Field(description="The product idea or concept to research")
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Optional focus areas to prioritize",
    )
    depth: str = Field(
        default="comprehensive",
        description="Research depth: 'quick', 'comprehensive', or 'deep'",
    )
    output_format: str = Field(
        default="markdown",
        description="Output format: 'markdown', 'docx', or 'both'",
    )
    max_paths: int = Field(
        default=5,
        description="Maximum number of research paths to explore",
        ge=1,
        le=10,
    )
    weights: ResearchWeight | None = Field(
        default=None,
        description="Research type weights (industry/academic/community)",
    )

    @field_validator("depth")
    @classmethod
    def validate_depth(cls, value: str) -> str:
        """Validate supported research depths."""
        normalized = value.strip().lower()
        if normalized not in {"quick", "comprehensive", "deep"}:
            raise ValueError("depth must be one of: quick, comprehensive, deep")
        return normalized

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, value: str) -> str:
        """Validate supported report output formats."""
        normalized = value.strip().lower()
        if normalized not in {"markdown", "docx", "both"}:
            raise ValueError("output_format must be one of: markdown, docx, both")
        return normalized
