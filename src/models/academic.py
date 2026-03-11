"""Academic research data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.common import SourceReference


class PaperAnalysis(BaseModel):
    """PhD-level analysis of an academic paper."""

    paper_id: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int = 0
    venue: str = ""
    citation_count: int = 0
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""

    # PhD-level analysis fields
    principles: str = Field(
        default="",
        description="Core theoretical principles and contributions",
    )
    methods: str = Field(
        default="",
        description="Methodology, approach, and implementation details",
    )
    experiments: str = Field(
        default="",
        description="Experimental setup, datasets, metrics, and results",
    )
    conclusions: str = Field(
        default="",
        description="Key conclusions and implications",
    )
    deficiencies: str = Field(
        default="",
        description="Limitations, gaps, and areas for improvement",
    )

    # Venue prestige
    venue_prestige: str = ""  # "top_conference", "top_journal", "workshop", "preprint"
    known_lab: str = ""  # Name of well-known lab if applicable


class AcademicResearchResult(BaseModel):
    """Complete academic research result for one research path."""

    path_id: str = ""
    papers: list[PaperAnalysis] = Field(default_factory=list)
    research_trends: str = ""
    frontier_directions: str = ""
    key_researchers: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
