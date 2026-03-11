"""Tests for IndustryResearcher agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.industry_researcher import IndustryResearcher
from src.apis.tavily_client import TavilyClient
from src.apis.github_client import GitHubClient
from src.apis.web_scraper import WebScraper


@pytest.fixture
def mock_tavily():
    client = AsyncMock(spec=TavilyClient)
    client.search.return_value = {
        "results": [
            {
                "title": "AI Code Review Tool",
                "url": "https://example.com/tool",
                "content": "This tool reviews code using AI.",
                "score": 0.9,
            }
        ]
    }
    return client


@pytest.fixture
def mock_github():
    client = AsyncMock(spec=GitHubClient)
    client.search_repos.return_value = [
        {
            "full_name": "owner/repo",
            "html_url": "https://github.com/owner/repo",
            "stargazers_count": 500,
            "forks_count": 50,
            "language": "Python",
            "description": "A code review tool",
            "license": {"spdx_id": "MIT"},
            "topics": ["code-review"],
            "updated_at": "2025-01-01",
        }
    ]
    client.get_readme.return_value = "# README\nA great tool."
    return client


@pytest.fixture
def mock_scraper():
    return AsyncMock(spec=WebScraper)


@pytest.fixture
def researcher(mock_llm, temp_store, mock_tavily, mock_github, mock_scraper):
    return IndustryResearcher(mock_llm, temp_store, mock_tavily, mock_github, mock_scraper)


@pytest.mark.asyncio
async def test_run_success(researcher, mock_llm, sample_research_path):
    mock_llm.generate_json.return_value = json.dumps({
        "products": [
            {"name": "CodeReviewAI", "company": "TechCo", "description": "AI reviewer"}
        ],
        "companies": [
            {"name": "TechCo", "size": "startup", "reputation_notes": "Good"}
        ],
        "key_insights": ["AI improves code quality"],
    })

    result = await researcher.run(path=sample_research_path)

    assert result.path_id == "p1"
    assert len(result.products) >= 1
    assert len(result.sources) >= 1


@pytest.mark.asyncio
async def test_run_web_search_failure(researcher, mock_llm, mock_tavily, sample_research_path):
    mock_tavily.search.side_effect = Exception("Network error")
    mock_llm.generate_json.return_value = json.dumps({
        "products": [], "companies": [], "key_insights": [],
    })

    result = await researcher.run(path=sample_research_path)

    # Should not crash, just have fewer results
    assert result.path_id == "p1"


@pytest.mark.asyncio
async def test_render_template_not_generate(researcher, mock_llm, sample_research_path):
    """Verify render_template is used instead of generate_with_template."""
    mock_llm.generate_json.return_value = json.dumps({
        "products": [], "companies": [], "key_insights": ["insight"],
    })

    await researcher.run(path=sample_research_path)

    # render_template should be called (not generate_with_template)
    assert mock_llm.render_template.called
    mock_llm.generate_with_template.assert_not_called()
