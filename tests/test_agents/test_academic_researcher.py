"""Tests for AcademicResearcher agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.academic_researcher import AcademicResearcher
from src.apis.tavily_client import TavilyClient
from src.apis.arxiv_client import ArxivClient


@pytest.fixture
def mock_tavily():
    client = AsyncMock(spec=TavilyClient)
    client.search.return_value = {
        "results": [
            {
                "title": "LLM Code Review benchmark paper",
                "url": "https://arxiv.org/abs/2401.00001",
                "content": "A 2024 benchmark for LLM-based code review.",
                "score": 0.9,
            }
        ]
    }
    return client


@pytest.fixture
def mock_arxiv():
    client = AsyncMock(spec=ArxivClient)
    client.search.return_value = [
        {
            "arxiv_id": "2401.99999",
            "title": "Novel Code Analysis",
            "authors": ["Charlie", "Diana"],
            "summary": "A new approach to code analysis using transformers.",
            "published": "2024-01-20T00:00:00Z",
            "categories": ["cs.SE"],
            "pdf_url": "https://arxiv.org/pdf/2401.99999",
        }
    ]
    return client


@pytest.fixture
def researcher(mock_llm, temp_store, mock_tavily, mock_arxiv):
    return AcademicResearcher(mock_llm, temp_store, mock_tavily, mock_arxiv)


@pytest.mark.asyncio
async def test_run_success(researcher, mock_llm, sample_research_path):
    mock_llm.generate_json.return_value = json.dumps({
        "research_trends": "Transformer-based code review is moving toward benchmarks.",
        "frontier_directions": "Multilingual and repository-level evaluation.",
        "key_researchers": ["Alice", "Bob"],
    })

    result = await researcher.run(path=sample_research_path)

    assert result.path_id == "p1"
    assert len(result.papers) >= 1
    assert result.papers[0].title in ("LLM Code Review benchmark paper", "Novel Code Analysis")
    assert len(result.sources) >= 1
    assert result.research_trends


@pytest.mark.asyncio
async def test_merge_deduplication(researcher, mock_tavily, mock_arxiv, mock_llm, sample_research_path):
    # Both sources return same title
    mock_tavily.search.return_value = {
        "results": [
            {
                "title": "LLM Code Review",
                "url": "https://arxiv.org/abs/2401.00001",
                "content": "Same paper.",
                "score": 0.8,
            }
        ]
    }
    mock_arxiv.search.return_value = [
        {
            "arxiv_id": "2401.00001",
            "title": "LLM Code Review",  # Same as S2
            "authors": ["Alice"],
            "summary": "Same paper.",
            "published": "2024-01-15",
            "categories": ["cs.SE"],
            "pdf_url": "",
        }
    ]
    mock_llm.generate_json.return_value = json.dumps({
        "research_trends": "", "frontier_directions": "", "key_researchers": [],
    })

    result = await researcher.run(path=sample_research_path)

    # Should be deduplicated
    assert len(result.papers) == 1


@pytest.mark.asyncio
async def test_academic_web_failure_still_returns(
    researcher, mock_tavily, mock_llm, sample_research_path,
):
    mock_tavily.search.side_effect = Exception("API error")
    mock_llm.generate_json.return_value = json.dumps({
        "research_trends": "", "frontier_directions": "", "key_researchers": [],
    })

    result = await researcher.run(path=sample_research_path)

    # Should still have arXiv results
    assert result.path_id == "p1"
    assert len(result.papers) >= 1


@pytest.mark.asyncio
async def test_build_from_web_result(researcher):
    data = {
        "title": "Test 2023",
        "url": "https://openreview.net/forum?id=abc",
        "content": "A 2023 benchmark paper.",
    }
    paper = researcher._build_from_web_result(data)
    assert paper.title == "Test 2023"
    assert paper.year == 2023
    assert paper.venue == "OpenReview"


@pytest.mark.asyncio
async def test_build_from_arxiv(researcher):
    data = {
        "arxiv_id": "2301.12345",
        "title": "ArXiv Paper",
        "authors": ["A", "B"],
        "published": "2023-01-15T00:00:00Z",
        "pdf_url": "https://arxiv.org/pdf/2301.12345",
    }
    paper = researcher._build_from_arxiv(data)
    assert paper.title == "ArXiv Paper"
    assert paper.year == 2023
    assert paper.arxiv_id == "2301.12345"
