"""Tests for LLMClient."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from src.llm.client import LLMClient


def test_render_template():
    """Test that render_template reads file and formats without LLM call."""
    client = LLMClient.__new__(LLMClient)
    client.model = "test"
    client.max_tokens = 100
    client._client = None

    # Test with the decompose_idea template
    result = client.render_template("decompose_idea", {"raw_input": "test idea"})

    assert "test idea" in result
    assert isinstance(result, str)


def test_render_template_not_found():
    """Test FileNotFoundError for missing template."""
    client = LLMClient.__new__(LLMClient)

    with pytest.raises(FileNotFoundError):
        client.render_template("nonexistent_template", {})


def test_render_template_no_llm_call():
    """Ensure render_template does NOT make any API call."""
    with patch.dict("os.environ", {"LLM_MODE": "api-key", "ANTHROPIC_API_KEY": "sk-test"}):
        client = LLMClient()

    # If this tried to call the API, it would fail since "sk-test" is not valid
    result = client.render_template("decompose_idea", {"raw_input": "test"})
    assert isinstance(result, str)
    assert len(result) > 0
