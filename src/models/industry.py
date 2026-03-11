"""Industry research data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.common import MaturityStage, SourceReference


class ProductInfo(BaseModel):
    """Information about a product or tool found in industry research."""

    name: str = ""
    company: str = ""
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    pricing_model: str = ""
    url: str = ""
    maturity: MaturityStage = MaturityStage.DEVELOPMENT
    is_open_source: bool = False
    github_url: str = ""


class CompanyInfo(BaseModel):
    """Information about a company behind a product."""

    name: str = ""
    size: str = ""  # "startup", "mid", "large", "big_tech"
    funding: str = ""
    reputation_notes: str = ""


class RepoAnalysis(BaseModel):
    """Analysis of a GitHub repository."""

    repo_url: str = ""
    name: str = ""
    stars: int = 0
    forks: int = 0
    language: str = ""
    last_updated: str = ""
    license: str = ""
    description: str = ""
    topics: list[str] = Field(default_factory=list)
    code_quality_notes: str = ""
    deployment_readiness: str = ""
    community_health: str = ""
    key_dependencies: list[str] = Field(default_factory=list)


class BlogSummary(BaseModel):
    """Summary of a blog post or article."""

    title: str = ""
    url: str = ""
    author: str = ""
    published_date: str = ""
    key_points: list[str] = Field(default_factory=list)
    relevance_notes: str = ""


class IndustryResearchResult(BaseModel):
    """Complete industry research result for one research path."""

    path_id: str = ""
    products: list[ProductInfo] = Field(default_factory=list)
    companies: list[CompanyInfo] = Field(default_factory=list)
    repos: list[RepoAnalysis] = Field(default_factory=list)
    blog_summaries: list[BlogSummary] = Field(default_factory=list)
    market_trends: str = ""
    sources: list[SourceReference] = Field(default_factory=list)
