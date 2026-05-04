"""Research plan and idea decomposition models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchPath(BaseModel):
    """A concrete research path derived from idea decomposition."""

    path_id: str = Field(description="Unique identifier for this path")
    title: str = Field(description="Short title for the path")
    description: str = Field(description="What this path explores")
    technologies_needed: list[str] = Field(
        default_factory=list,
        description="Key technologies required for this approach",
    )
    key_questions: list[str] = Field(
        default_factory=list,
        description="Critical questions to answer for this path",
    )
    search_queries: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Search queries by source type: web, academic, code",
    )
    priority: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Priority weight for this path",
    )


class DecompositionResult(BaseModel):
    """Result of decomposing a vague idea into research paths."""

    original_input: str = Field(description="The user's original input")
    interpretation: str = Field(
        default="",
        description="How the system interpreted the input",
    )
    paths: list[ResearchPath] = Field(
        default_factory=list,
        description="Concrete research paths to explore",
    )
    shared_technologies: list[str] = Field(
        default_factory=list,
        description="Technologies common across all paths",
    )


class SearchQuery(BaseModel):
    """A search query with metadata for execution."""

    query: str
    source: str  # "tavily", "semantic_scholar", "github", "arxiv"
    path_id: str
    priority: float = 0.5
    intent: str = Field(
        default="general",
        description="Search intent such as product, repo, paper, benchmark, alternative",
    )


class ResearchPlan(BaseModel):
    """Complete research plan with all queries organized for execution."""

    original_input: str
    paths: list[ResearchPath] = Field(default_factory=list)
    search_queries: list[SearchQuery] = Field(default_factory=list)
    estimated_api_calls: int = 0
    estimated_llm_calls: int = 0
