"""Tests for EngineeringAnalyst agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.engineering_analyst import EngineeringAnalyst
from src.apis.tavily_client import TavilyClient


@pytest.fixture
def mock_tavily():
    client = AsyncMock(spec=TavilyClient)
    client.search.return_value = {
        "results": [
            {
                "title": "org/codereviewer GitHub repository",
                "url": "https://github.com/org/codereviewer",
                "content": "AI-powered code review tool in Python FastAPI Docker.",
                "score": 0.9,
            }
        ]
    }
    return client


@pytest.fixture
def analyst(mock_llm, temp_store, mock_tavily):
    return EngineeringAnalyst(mock_llm, temp_store, mock_tavily)


@pytest.mark.asyncio
async def test_run_success(analyst, mock_llm, sample_research_path):
    mock_llm.generate_json.side_effect = [
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
    assert "Python" in result.code_analyses[0].tech_stack


@pytest.mark.asyncio
async def test_run_no_repos(analyst, mock_llm, mock_tavily, sample_research_path):
    mock_tavily.search.return_value = {"results": []}

    result = await analyst.run(path=sample_research_path)

    assert result.path_id == "p1"
    assert result.code_analyses == []


@pytest.mark.asyncio
async def test_code_search_failure(analyst, mock_llm, mock_tavily, sample_research_path):
    mock_tavily.search.side_effect = Exception("Search down")

    result = await analyst.run(path=sample_research_path)

    assert result.path_id == "p1"
    assert result.code_analyses == []
