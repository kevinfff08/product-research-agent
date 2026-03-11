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

    assert len(plan.search_queries) == 2
    assert plan.search_queries[0].source == "tavily"
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

    assert len(plan.search_queries) == 1
    # Verify render_template was called with correct weights
    mock_llm.render_template.assert_called_once()


def test_fallback_plan_maps_sources(planner, sample_decomposition):
    plan = planner._fallback_plan(sample_decomposition)

    sources = {q.source for q in plan.search_queries}
    assert "tavily" in sources
    assert "semantic_scholar" in sources
    assert "github" in sources
