"""Reputation and credibility scoring models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VenuePrestige(BaseModel):
    """Prestige classification for a publication venue."""

    venue_name: str = ""
    venue_type: str = ""  # "top_conference", "top_journal", "workshop", "preprint", "blog"
    prestige_score: float = Field(default=0.5, ge=0.0, le=1.0)
    notes: str = ""


class CompanyReputation(BaseModel):
    """Reputation assessment for a company."""

    company_name: str = ""
    category: str = ""  # "big_tech", "established", "startup", "open_source_community"
    market_presence: str = ""
    user_sentiment: str = ""  # "very_positive", "positive", "mixed", "negative"
    notable_users: list[str] = Field(default_factory=list)
    reputation_score: float = Field(default=0.5, ge=0.0, le=1.0)


class LabPrestige(BaseModel):
    """Prestige assessment for a research lab."""

    lab_name: str = ""
    institution: str = ""
    notable_researchers: list[str] = Field(default_factory=list)
    prestige_score: float = Field(default=0.5, ge=0.0, le=1.0)
    notes: str = ""


class ReputationScore(BaseModel):
    """Composite reputation score for a technology or approach."""

    name: str = ""
    source_type_score: float = Field(default=0.5, ge=0.0, le=1.0)
    recency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    author_reputation_score: float = Field(default=0.5, ge=0.0, le=1.0)
    cross_validation_score: float = Field(default=0.5, ge=0.0, le=1.0)
    community_reception_score: float = Field(default=0.5, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.5, ge=0.0, le=1.0)


class ReputationReport(BaseModel):
    """Complete reputation analysis."""

    companies: list[CompanyReputation] = Field(default_factory=list)
    labs: list[LabPrestige] = Field(default_factory=list)
    venues: list[VenuePrestige] = Field(default_factory=list)
    technology_scores: list[ReputationScore] = Field(default_factory=list)
    summary: str = ""
