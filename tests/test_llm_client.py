"""Tests for LLMClient."""

from __future__ import annotations

import json
import pytest
from unittest.mock import patch

import httpx
import respx

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
    with patch.dict("os.environ", {"LLM_MODE": "api-key", "OPENAI_API_KEY": "sk-test"}):
        client = LLMClient()

    # If this tried to call the API, it would fail since "sk-test" is not valid
    result = client.render_template("decompose_idea", {"raw_input": "test"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_setup_token_normalizes_proxy_url():
    with patch.dict(
        "os.environ",
        {
            "LLM_MODE": "setup-token",
            "LLM_PROXY_URL": "http://localhost:8317",
            "LLM_MODEL": "gpt-5.4",
        },
    ):
        client = LLMClient()

    assert client.base_url == "http://localhost:8317/v1"
    assert client.model == "gpt-5.4"


@respx.mock
def test_generate_posts_chat_completion():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )
    )

    with patch.dict(
        "os.environ",
        {"LLM_MODE": "api-key", "OPENAI_API_KEY": "sk-test", "LLM_MODEL": "gpt-5.4"},
    ):
        client = LLMClient()
        result = client.generate("Say OK", system="system prompt", max_tokens=8)
        client.close()

    assert result == "OK"
    assert route.called
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-test"
    payload = json.loads(request.content)
    assert payload["model"] == "gpt-5.4"
    assert payload["messages"][0]["role"] == "system"


@respx.mock
def test_generate_raises_model_routing_error():
    respx.post("http://localhost:8317/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"message": "unknown provider for model gpt-5.5"}},
        )
    )

    with patch.dict(
        "os.environ",
        {
            "LLM_MODE": "setup-token",
            "LLM_PROXY_URL": "http://localhost:8317",
            "LLM_MODEL": "gpt-5.5",
        },
    ):
        client = LLMClient()
        with pytest.raises(RuntimeError, match="Non-retryable LLM model routing error"):
            client.generate("test")
        client.close()
