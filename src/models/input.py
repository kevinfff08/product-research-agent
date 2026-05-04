"""Input models for research requests."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from src.models.common import ResearchWeight


class ResearchRequest(BaseModel):
    """User's research request."""

    title: str = Field(
        default="",
        description="Short title-style product idea or research topic",
    )
    detailed_description: str = Field(
        default="",
        description="Second-stage detailed description with context, goals, and constraints",
    )
    raw_input: str = Field(
        default="",
        description="Composed full input used by downstream research agents",
    )
    run_name: str = Field(
        default="",
        description="Date-plus-title run name used for session, logs, and report files",
    )
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

    @model_validator(mode="after")
    def compose_input(self) -> "ResearchRequest":
        """Keep old raw_input calls compatible while supporting title + details."""
        self.title = self.title.strip()
        self.detailed_description = self.detailed_description.strip()
        self.raw_input = self.raw_input.strip()

        if not self.title and self.raw_input:
            self.title = self.raw_input.splitlines()[0].strip()

        if not self.raw_input and self.title:
            parts = [f"Title: {self.title}"]
            if self.detailed_description:
                parts.append(f"Detailed description:\n{self.detailed_description}")
            self.raw_input = "\n\n".join(parts)

        if not self.title:
            raise ValueError("title or raw_input is required")
        return self

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
