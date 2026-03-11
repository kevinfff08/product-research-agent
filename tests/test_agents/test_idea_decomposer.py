"""Tests for IdeaDecomposer agent."""

from __future__ import annotations

import json

import pytest

from src.agents.idea_decomposer import IdeaDecomposer


@pytest.fixture
def decomposer(mock_llm, temp_store):
    return IdeaDecomposer(mock_llm, temp_store)


def test_run_success(decomposer, mock_llm):
    mock_llm.generate_json.return_value = json.dumps({
        "interpretation": "An AI-powered code review tool",
        "paths": [
            {
                "path_id": "p1",
                "title": "Static Analysis + LLM",
                "description": "Combine AST parsing with LLM",
                "technologies_needed": ["Python", "tree-sitter"],
                "key_questions": ["How to parse code?"],
                "search_queries": {"web": ["AI code review"]},
                "priority": 0.8,
            },
        ],
        "shared_technologies": ["Python"],
    })

    result = decomposer.run(raw_input="AI code review tool")

    assert result.original_input == "AI code review tool"
    assert len(result.paths) == 1
    assert result.paths[0].title == "Static Analysis + LLM"
    assert result.paths[0].priority == 0.8
    assert result.shared_technologies == ["Python"]


def test_run_empty_response(decomposer, mock_llm):
    mock_llm.generate_json.return_value = ""

    result = decomposer.run(raw_input="test")

    assert result.paths == []
    assert result.interpretation == "Failed to decompose"


def test_run_invalid_json(decomposer, mock_llm):
    mock_llm.generate_json.return_value = "not json at all"

    result = decomposer.run(raw_input="test")

    assert result.paths == []


def test_run_max_paths(decomposer, mock_llm):
    paths_data = [
        {"path_id": f"p{i}", "title": f"Path {i}", "description": f"Desc {i}"}
        for i in range(10)
    ]
    mock_llm.generate_json.return_value = json.dumps({
        "interpretation": "test",
        "paths": paths_data,
    })

    result = decomposer.run(raw_input="test", max_paths=3)

    assert len(result.paths) == 3


def test_render_template_called(decomposer, mock_llm):
    """Ensure _build_prompt uses render_template, not generate_with_template."""
    mock_llm.render_template.return_value = "rendered prompt"
    mock_llm.generate_json.return_value = json.dumps({
        "interpretation": "test",
        "paths": [],
    })

    decomposer.run(raw_input="test idea")

    mock_llm.render_template.assert_called_once()
    # generate_with_template should NOT be called (that was the bug)
    mock_llm.generate_with_template.assert_not_called()
