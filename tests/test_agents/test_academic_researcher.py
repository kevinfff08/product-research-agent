"""Tests for AcademicResearcher agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.academic_researcher import AcademicResearcher
from src.apis.semantic_scholar import SemanticScholarClient
from src.apis.arxiv_client import ArxivClient


@pytest.fixture
def mock_s2():
    client = AsyncMock(spec=SemanticScholarClient)
    client.search_papers.return_value = [
        {
            "paperId": "abc123",
            "title": "LLM Code Review",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "year": 2024,
            "venue": "ICSE",
            "citationCount": 50,
            "abstract": "We present an LLM-based code review approach.",
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2401.00001"},
        }
    ]
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
def researcher(mock_llm, temp_store, mock_s2, mock_arxiv):
    return AcademicResearcher(mock_llm, temp_store, mock_s2, mock_arxiv)


@pytest.mark.asyncio
async def test_run_success(researcher, mock_llm, sample_research_path):
    mock_llm.generate_json.return_value = json.dumps({
        "principles": "Transformer-based approach",
        "methods": "Fine-tuned LLM on code review data",
        "experiments": "Tested on 1000 PRs",
        "conclusions": "Improves review quality",
        "deficiencies": "Limited to Python only",
        "venue_prestige": "top_conference",
        "known_lab": "Google Research",
    })

    result = await researcher.run(path=sample_research_path)

    assert result.path_id == "p1"
    assert len(result.papers) >= 1
    assert result.papers[0].title in ("LLM Code Review", "Novel Code Analysis")
    assert len(result.sources) >= 1


@pytest.mark.asyncio
async def test_merge_deduplication(researcher, mock_s2, mock_arxiv, mock_llm, sample_research_path):
    # Both sources return same title
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
        "principles": "", "methods": "", "experiments": "",
        "conclusions": "", "deficiencies": "",
    })

    result = await researcher.run(path=sample_research_path)

    # Should be deduplicated
    assert len(result.papers) == 1


@pytest.mark.asyncio
async def test_s2_failure_still_returns(researcher, mock_s2, mock_llm, sample_research_path):
    mock_s2.search_papers.side_effect = Exception("API error")
    mock_llm.generate_json.return_value = json.dumps({
        "principles": "", "methods": "", "experiments": "",
        "conclusions": "", "deficiencies": "",
    })

    result = await researcher.run(path=sample_research_path)

    # Should still have arXiv results
    assert result.path_id == "p1"
    assert len(result.papers) >= 1


@pytest.mark.asyncio
async def test_build_from_s2(researcher):
    data = {
        "paperId": "xyz",
        "title": "Test",
        "authors": [{"name": "A"}],
        "year": 2023,
        "venue": "NeurIPS",
        "citationCount": 100,
        "externalIds": {"DOI": "10.1/test"},
    }
    paper = researcher._build_from_s2(data)
    assert paper.title == "Test"
    assert paper.year == 2023
    assert paper.citation_count == 100


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
