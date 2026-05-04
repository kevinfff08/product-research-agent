"""Tests for MarkdownReporter."""

from __future__ import annotations

import pytest
from pathlib import Path

from src.reporters.markdown_reporter import MarkdownReporter
from src.models.report import (
    ResearchReport, TechnologyEntry, ImplementationWorkflow,
    WorkflowStep, FeasibilityAssessment, DecisionMatrixRow, EvidenceClaim,
)
from src.models.industry import IndustryResearchResult, ProductInfo, RepoAnalysis
from src.models.academic import AcademicResearchResult, PaperAnalysis
from src.models.engineering import EngineeringAnalysis, DeploymentAssessment
from src.models.maturity import MaturityAssessment, TechnologyMaturity
from src.models.reputation import ReputationReport, CompanyReputation, ReputationScore
from src.models.common import MaturityStage, SourceReference, SourceType


@pytest.fixture
def reporter():
    return MarkdownReporter()


@pytest.fixture
def full_report():
    return ResearchReport(
        title="Tech Landscape: AI Code Review",
        session_id="test-123",
        generated_at="2025-01-01T00:00:00Z",
        original_input="AI code review tool",
        executive_summary="This report analyzes the AI code review landscape.",
        research_questions=["Should we build or reuse an AI code review stack?"],
        methodology_summary="Compared products, papers, repositories, and maturity signals.",
        decision_matrix=[
            DecisionMatrixRow(
                option="Static Analysis + LLM",
                path_id="p1",
                user_value="high",
                technical_feasibility="high",
                maturity="mature",
                ecosystem_strength="strong",
                cost_risk="medium",
                evidence_strength="moderate",
                verdict="recommended",
            ),
        ],
        key_claims=[
            EvidenceClaim(
                claim="Parser-grounded review is a practical first step.",
                supporting_evidence=["tree-sitter is mature."],
                contradicting_evidence=["Agentic approaches may cover broader context."],
                confidence="medium",
                implication="Start narrow, then expand.",
                source_urls=["https://example.com"],
            ),
        ],
        evidence_gaps=["Need latency benchmark."],
        assumptions=["Python-first MVP."],
        confidence_assessment="Medium confidence.",
        technology_landscape=[
            TechnologyEntry(
                name="tree-sitter",
                category="backend",
                maturity=MaturityStage.MATURE,
                description="Fast parser",
                industry_score=0.9,
                academic_score=0.5,
                engineering_score=0.95,
                reputation_score=0.85,
            ),
        ],
        workflows=[
            ImplementationWorkflow(
                name="Quick Start",
                description="Fast setup",
                steps=[WorkflowStep(step=1, description="Install deps", technologies=["Python"])],
                pros=["Simple"],
                cons=["Basic"],
            ),
        ],
        maturity_assessments=[
            MaturityAssessment(
                path_id="p1",
                technologies=[TechnologyMaturity(
                    technology="tree-sitter",
                    stage=MaturityStage.MATURE,
                    evidence="Widely used",
                    trend="stable",
                )],
                overall_maturity_summary="Mature landscape",
                mature_solutions=["tree-sitter"],
            ),
        ],
        industry_findings=[
            IndustryResearchResult(
                path_id="p1",
                products=[ProductInfo(name="CodeBot", company="Co", description="AI tool")],
                repos=[RepoAnalysis(name="org/repo", repo_url="https://github.com/org/repo",
                                    stars=1000, language="Python")],
            ),
        ],
        academic_findings=[
            AcademicResearchResult(
                path_id="p1",
                papers=[PaperAnalysis(
                    title="LLM Review", year=2024, venue="ICSE",
                    citation_count=50, authors=["Alice"],
                    conclusions="Works well",
                )],
            ),
        ],
        engineering_analyses=[
            EngineeringAnalysis(
                path_id="p1",
                deployment_assessment=DeploymentAssessment(deployment_complexity="moderate"),
                implementation_recommendations="Use Docker",
                technology_stack_recommendation=["Python", "FastAPI"],
            ),
        ],
        reputation_report=ReputationReport(
            summary="Good overall",
            companies=[CompanyReputation(
                company_name="TechCo", category="startup",
                reputation_score=0.75, user_sentiment="positive",
            )],
            technology_scores=[ReputationScore(
                name="LLM review", overall_score=0.8,
                source_type_score=0.8, recency_score=0.9,
                author_reputation_score=0.7, cross_validation_score=0.6,
                community_reception_score=0.8,
            )],
        ),
        feasibility_assessments=[
            FeasibilityAssessment(
                path_id="p1", path_title="Static Analysis",
                overall_feasibility=0.85, technical_risk="medium",
                recommended=True, rationale="Feasible",
            ),
        ],
        all_sources=[
            SourceReference(url="https://example.com", title="Source 1",
                            source_type=SourceType.WEB),
        ],
        recommendations="Start with the simple approach.",
    )


def test_generate_creates_file(reporter, full_report, tmp_path):
    output = tmp_path / "report.md"
    result = reporter.generate(full_report, output)

    assert result.exists()
    content = result.read_text(encoding="utf-8")
    assert "AI Code Review" in content
    assert "Executive Summary" in content
    assert "Technology Landscape" in content


def test_render_contains_all_sections(reporter, full_report):
    md = reporter._render(full_report)

    assert "# Tech Landscape:" in md
    assert "## Executive Summary" in md
    assert "## Research Question and Method" in md
    assert "## Decision Matrix" in md
    assert "## Key Claims and Evidence" in md
    assert "## Technology Landscape" in md
    assert "## Technology Maturity Map" in md
    assert "## Implementation Workflows" in md
    assert "## Industry Findings" in md
    assert "## Academic Research" in md
    assert "## Engineering Analysis" in md
    assert "## Reputation & Credibility" in md
    assert "## Feasibility Assessment" in md
    assert "## Confidence, Gaps, and Assumptions" in md
    assert "## Recommendations" in md
    assert "## Sources" in md


def test_empty_report(reporter, tmp_path):
    report = ResearchReport(title="Empty")
    output = tmp_path / "empty.md"
    result = reporter.generate(report, output)

    content = result.read_text(encoding="utf-8")
    assert "# Empty" in content


def test_output_dir_creation(reporter, full_report, tmp_path):
    output = tmp_path / "nested" / "dir" / "report.md"
    result = reporter.generate(full_report, output)
    assert result.exists()
