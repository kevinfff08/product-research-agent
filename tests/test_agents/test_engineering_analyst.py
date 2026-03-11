"""Tests for EngineeringAnalyst agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.engineering_analyst import EngineeringAnalyst
from src.apis.github_client import GitHubClient


@pytest.fixture
def mock_github():
    client = AsyncMock(spec=GitHubClient)
    client.search_repos.return_value = [
        {
            "full_name": "org/codereviewer",
            "html_url": "https://github.com/org/codereviewer",
            "stargazers_count": 2000,
            "forks_count": 300,
            "language": "Python",
            "description": "AI-powered code review tool",
            "license": {"spdx_id": "Apache-2.0"},
            "topics": ["code-review", "ai"],
            "updated_at": "2025-06-01",
        }
    ]
    client.get_readme.return_value = "# CodeReviewer\nAn AI tool."
    return client


@pytest.fixture
def analyst(mock_llm, temp_store, mock_github):
    return EngineeringAnalyst(mock_llm, temp_store, mock_github)


@pytest.mark.asyncio
async def test_run_success(analyst, mock_llm, sample_research_path):
    mock_llm.generate_json.side_effect = [
        # First call: analyze_repo template
        json.dumps({
            "architecture_summary": "Microservice architecture",
            "tech_stack": ["Python", "FastAPI", "Redis"],
            "code_quality_notes": "Well structured",
            "deployment_readiness": "Production ready",
            "community_health": "Active maintenance",
            "strengths": ["Good docs", "Fast"],
            "weaknesses": ["Limited language support"],
        }),
        # Second call: deployment assessment
        json.dumps({
            "deployment_complexity": "moderate",
            "infrastructure_requirements": ["Docker", "Redis"],
            "estimated_setup_effort": "2-3 days",
            "prerequisites": ["Python 3.10+"],
            "risks": ["API rate limits"],
            "implementation_recommendations": "Start with Docker setup",
            "technology_stack_recommendation": ["Python", "FastAPI"],
        }),
    ]

    result = await analyst.run(path=sample_research_path)

    assert result.path_id == "p1"
    assert len(result.code_analyses) == 1
    assert result.deployment_assessment.deployment_complexity == "moderate"
    assert len(result.sources) == 1


@pytest.mark.asyncio
async def test_run_no_repos(analyst, mock_llm, mock_github, sample_research_path):
    mock_github.search_repos.return_value = []

    result = await analyst.run(path=sample_research_path)

    assert result.path_id == "p1"
    assert result.code_analyses == []


@pytest.mark.asyncio
async def test_github_search_failure(analyst, mock_llm, mock_github, sample_research_path):
    mock_github.search_repos.side_effect = Exception("GitHub down")

    result = await analyst.run(path=sample_research_path)

    assert result.path_id == "p1"
    assert result.code_analyses == []
