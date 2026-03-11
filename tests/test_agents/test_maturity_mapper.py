"""Tests for MaturityMapper agent."""

from __future__ import annotations

import json

import pytest

from src.agents.maturity_mapper import MaturityMapper
from src.models.industry import IndustryResearchResult, RepoAnalysis, ProductInfo
from src.models.academic import AcademicResearchResult, PaperAnalysis
from src.models.engineering import EngineeringAnalysis, CodeAnalysis
from src.models.common import MaturityStage


@pytest.fixture
def mapper(mock_llm, temp_store):
    return MaturityMapper(mock_llm, temp_store)


def test_run_success(mapper, mock_llm, sample_research_path):
    industry = IndustryResearchResult(
        path_id="p1",
        repos=[RepoAnalysis(name="org/tool", language="Python", topics=["ai", "code-review"])],
        products=[ProductInfo(name="ReviewBot", capabilities=["code analysis", "auto review"])],
    )

    mock_llm.generate_json.return_value = json.dumps({
        "technologies": [
            {
                "technology": "python",
                "stage": "mature",
                "evidence": "Widely used in production",
                "adoption_signals": ["PyPI downloads", "GitHub presence"],
                "trend": "stable",
                "years_in_production": 30,
            },
            {
                "technology": "code analysis",
                "stage": "cutting_edge",
                "evidence": "LLM-based analysis is new",
                "adoption_signals": ["Growing GitHub repos"],
                "trend": "growing",
                "years_in_production": 2,
            },
        ],
        "overall_maturity_summary": "Mixed maturity landscape",
        "early_prototypes": [],
        "in_development": [],
        "mature_solutions": ["python"],
        "cutting_edge": ["code analysis"],
        "academic_frontier": [],
    })

    assessment = mapper.run(
        path=sample_research_path,
        industry_result=industry,
    )

    assert assessment.path_id == "p1"
    assert len(assessment.technologies) == 2
    assert assessment.technologies[0].stage == MaturityStage.MATURE
    assert assessment.technologies[1].stage == MaturityStage.CUTTING_EDGE
    assert "python" in assessment.mature_solutions


def test_run_no_technologies(mapper, mock_llm, sample_research_path):
    result = mapper.run(path=sample_research_path)
    assert result.path_id == "p1"
    assert result.technologies == []


def test_collect_technologies(mapper):
    industry = IndustryResearchResult(
        path_id="p1",
        repos=[RepoAnalysis(name="test/repo", language="Python", topics=["machine-learning"])],
    )
    engineering = EngineeringAnalysis(
        path_id="p1",
        code_analyses=[CodeAnalysis(repo_url="https://example.com", tech_stack=["FastAPI", "Redis"])],
    )
    academic = AcademicResearchResult(
        path_id="p1",
        papers=[PaperAnalysis(title="Test Paper", methods="Transformer-based approach")],
    )

    techs = mapper._collect_technologies(industry, academic, engineering)

    tech_names = [t["name"] for t in techs]
    assert "python" in tech_names
    assert "machine-learning" in tech_names
    assert "fastapi" in tech_names
    assert "redis" in tech_names


def test_collect_technologies_empty(mapper):
    techs = mapper._collect_technologies(None, None, None)
    assert techs == []
