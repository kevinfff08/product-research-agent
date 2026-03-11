"""Shared enums, base models, and common types."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MaturityStage(str, Enum):
    """Technology maturity stages from earliest to most advanced."""

    EARLY_PROTOTYPE = "early_prototype"
    DEVELOPMENT = "development"
    MATURE = "mature"
    CUTTING_EDGE = "cutting_edge"
    ACADEMIC_FRONTIER = "academic_frontier"


class SourceType(str, Enum):
    """Types of research sources."""

    WEB = "web"
    PAPER = "paper"
    REPO = "repo"
    BLOG = "blog"
    SOCIAL = "social"
    NEWS = "news"
    DOCUMENTATION = "documentation"


class SourceReference(BaseModel):
    """A reference to a source used in research."""

    url: str = ""
    title: str = ""
    source_type: SourceType = SourceType.WEB
    retrieved_at: str = ""
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    snippet: str = ""


class ResearchWeight(BaseModel):
    """Weight allocation for research types."""

    industry: float = 0.60
    academic: float = 0.25
    community: float = 0.15
