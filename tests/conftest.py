"""Shared test fixtures for ProductResearch tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.llm.client import LLMClient
from src.storage.local_store import LocalStore
from src.models.common import MaturityStage, SourceReference
from src.models.plan import ResearchPath, DecompositionResult, ResearchPlan


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_store(temp_dir):
    """Provide a LocalStore backed by a temporary directory."""
    return LocalStore(data_dir=temp_dir)


@pytest.fixture
def mock_llm():
    """LLMClient mock that returns configurable responses."""
    client = MagicMock(spec=LLMClient)
    client.generate.return_value = ""
    client.generate_json.return_value = "{}"
    client.generate_with_template.return_value = ""
    client.render_template.return_value = "rendered template prompt"
    client.model = "test-model"
    client.max_tokens = 4096
    return client


@pytest.fixture
def sample_research_path():
    """A sample ResearchPath for testing downstream agents."""
    return ResearchPath(
        path_id="p1",
        title="Static Analysis + LLM Code Review",
        description="Combine AST parsing with LLM-based code understanding",
        technologies_needed=["Python AST", "tree-sitter", "Codex API", "Git hooks"],
        key_questions=[
            "How to parse multiple programming languages?",
            "What is the latency for real-time code review?",
        ],
        search_queries={
            "web": ["AI code review tool", "LLM static analysis"],
            "academic": ["LLM code analysis paper", "automated code review"],
            "code": ["code review AI github", "static analysis LLM"],
        },
        priority=0.8,
    )


@pytest.fixture
def sample_decomposition(sample_research_path):
    """A sample DecompositionResult for testing."""
    return DecompositionResult(
        original_input="AI-powered code review tool",
        interpretation="A tool that uses AI to automatically review code changes",
        paths=[
            sample_research_path,
            ResearchPath(
                path_id="p2",
                title="Fine-tuned Code LLM",
                description="Fine-tune an LLM specifically for code review tasks",
                technologies_needed=["Code LLM", "PEFT/LoRA", "Training data"],
                key_questions=["What training data is needed?"],
                search_queries={
                    "web": ["fine-tuned code review model"],
                    "academic": ["code review language model fine-tuning"],
                    "code": ["code review model training"],
                },
                priority=0.6,
            ),
        ],
        shared_technologies=["Python", "Git integration", "Codex API"],
    )
