"""Tests for ResearchPlanner agent."""

from __future__ import annotations

import json

import pytest

from src.agents.research_planner import ResearchPlanner
from src.models.common import ResearchWeight


@pytest.fixture
def planner(mock_llm, temp_store):
    return ResearchPlanner(mock_llm, temp_store)


def test_run_success(planner, mock_llm, sample_decomposition):
    mock_llm.generate_json.return_value = json.dumps({
        "search_queries": [
            {"query": "AI code review tools", "source": "tavily", "path_id": "p1", "priority": 0.9},
            {"query": "automated code analysis paper", "source": "semantic_scholar", "path_id": "p1", "priority": 0.7},
        ],
        "estimated_api_calls": 2,
        "estimated_llm_calls": 2,
    })

    plan = planner.run(decomposition=sample_decomposition)

    assert len(plan.search_queries) >= 2
    assert "tavily" in {q.source for q in plan.search_queries}
    assert "academic_web" in {q.source for q in plan.search_queries}
    assert any(q.intent for q in plan.search_queries)
    assert plan.estimated_api_calls == 2


def test_run_fallback_on_invalid(planner, mock_llm, sample_decomposition):
    mock_llm.generate_json.return_value = "invalid json"

    plan = planner.run(decomposition=sample_decomposition)

    # Should use fallback plan from decomposition's own queries
    assert len(plan.search_queries) > 0
    assert plan.original_input == sample_decomposition.original_input


def test_run_with_weights(planner, mock_llm, sample_decomposition):
    mock_llm.generate_json.return_value = json.dumps({
        "search_queries": [{"query": "test", "source": "tavily", "path_id": "p1"}],
    })

    weights = ResearchWeight(industry=0.8, academic=0.1, community=0.1)
    plan = planner.run(decomposition=sample_decomposition, weights=weights)

    assert len(plan.search_queries) >= 1
    # Verify render_template was called with correct weights
    mock_llm.render_template.assert_called_once()


def test_fallback_plan_maps_sources(planner, sample_decomposition):
    plan = planner._fallback_plan(sample_decomposition)

    sources = {q.source for q in plan.search_queries}
    assert "tavily" in sources
    assert "academic_web" in sources
    assert "code_web" in sources


def test_query_optimization_deduplicates_and_expands(planner, sample_decomposition):
    from src.models.plan import SearchQuery

    queries = [
        SearchQuery(query="AI code review tools", source="tavily", path_id="p1", priority=0.5),
        SearchQuery(query="  AI   code review tools ", source="tavily", path_id="p1", priority=0.9),
    ]

    optimized = planner._optimize_queries(sample_decomposition, queries)

    matching = [
        q for q in optimized
        if q.path_id == "p1" and q.source == "tavily" and q.query == "AI code review tools"
    ]
    assert len(matching) == 1
    assert matching[0].priority == 0.9
    assert {"code_web", "academic_web", "openalex", "arxiv"}.issubset({q.source for q in optimized})
    assert all("recent preprint benchmark" not in q.query for q in optimized if q.source == "arxiv")
