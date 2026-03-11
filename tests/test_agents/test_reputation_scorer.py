"""Tests for ReputationScorer agent."""

from __future__ import annotations

import json

import pytest

from src.agents.reputation_scorer import ReputationScorer
from src.models.industry import (
    IndustryResearchResult, ProductInfo, CompanyInfo, RepoAnalysis,
)
from src.models.academic import AcademicResearchResult, PaperAnalysis


@pytest.fixture
def scorer(mock_llm, temp_store):
    return ReputationScorer(mock_llm, temp_store)


@pytest.fixture
def industry_results():
    return [IndustryResearchResult(
        path_id="p1",
        products=[ProductInfo(name="ReviewBot", company="TechCo", description="AI reviewer")],
        companies=[CompanyInfo(name="TechCo", size="startup", reputation_notes="Good")],
        repos=[RepoAnalysis(name="org/tool", stars=1000, language="Python")],
    )]


@pytest.fixture
def academic_results():
    return [AcademicResearchResult(
        path_id="p1",
        papers=[PaperAnalysis(
            title="LLM Code Review", year=2024, venue="ICSE",
            citation_count=50, known_lab="Google Research",
        )],
    )]


def test_run_success(scorer, mock_llm, industry_results, academic_results):
    mock_llm.generate_json.return_value = json.dumps({
        "companies": [
            {"company_name": "TechCo", "category": "startup",
             "reputation_score": 0.7, "user_sentiment": "positive"},
        ],
        "labs": [
            {"lab_name": "Google Research", "institution": "Google",
             "prestige_score": 0.95},
        ],
        "venues": [
            {"venue_name": "ICSE", "venue_type": "top_conference",
             "prestige_score": 0.9},
        ],
        "technology_scores": [
            {"name": "LLM-based review", "overall_score": 0.8,
             "source_type_score": 0.8, "recency_score": 0.9,
             "author_reputation_score": 0.7, "cross_validation_score": 0.6,
             "community_reception_score": 0.8},
        ],
        "summary": "Generally positive reputation landscape.",
    })

    report = scorer.run(
        industry_results=industry_results,
        academic_results=academic_results,
    )

    assert len(report.companies) == 1
    assert report.companies[0].company_name == "TechCo"
    assert len(report.labs) == 1
    assert len(report.venues) == 1
    assert len(report.technology_scores) == 1
    assert report.summary


def test_run_empty_input(scorer, mock_llm):
    mock_llm.generate_json.return_value = json.dumps({
        "companies": [], "labs": [], "venues": [],
        "technology_scores": [], "summary": "No data.",
    })

    report = scorer.run(industry_results=[], academic_results=[])

    assert report.companies == []


def test_run_invalid_response(scorer, mock_llm):
    mock_llm.generate_json.return_value = "invalid"

    report = scorer.run(industry_results=[], academic_results=[])

    assert report.companies == []
    assert report.summary == ""


def test_build_industry_summary(scorer, industry_results):
    summary = scorer._build_industry_summary(industry_results)
    assert "ReviewBot" in summary
    assert "TechCo" in summary


def test_build_academic_summary(scorer, academic_results):
    summary = scorer._build_academic_summary(academic_results)
    assert "LLM Code Review" in summary
    assert "ICSE" in summary
