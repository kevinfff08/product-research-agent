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
    with patch.dict(
        "os.environ",
        {"LLM_MODE": "api-key", "LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"},
    ):
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
            "LLM_PROVIDER": "openai",
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
        {
            "LLM_MODE": "api-key",
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "LLM_MODEL": "gpt-5.4",
        },
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
            "LLM_PROVIDER": "openai",
            "LLM_PROXY_URL": "http://localhost:8317",
            "LLM_MODEL": "gpt-5.5",
        },
    ):
        client = LLMClient()
        with pytest.raises(RuntimeError, match="Non-retryable LLM model routing error"):
            client.generate("test")
        client.close()


def test_api_key_mode_uses_deepseek_defaults():
    with patch.dict(
        "os.environ",
        {
            "LLM_MODE": "api-key",
            "LLM_PROVIDER": "deepseek",
            "LLM_MODEL": "",
            "DEEPSEEK_API_KEY": "ds-test",
        },
    ):
        client = LLMClient()

    assert client.base_url == "https://api.deepseek.com"
    assert client.model == "deepseek-v4-flash"
    assert client.api_key == "ds-test"


def test_api_key_mode_uses_google_openai_compatible_base_url():
    with patch.dict(
        "os.environ",
        {
            "LLM_MODE": "api-key",
            "LLM_PROVIDER": "google",
            "LLM_MODEL": "",
            "GOOGLE_API_KEY": "google-test",
        },
    ):
        client = LLMClient()

    assert client.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert client.model == "gemini-2.5-flash"
    assert client.api_key == "google-test"


def test_unsupported_provider_raises():
    with patch.dict("os.environ", {"LLM_PROVIDER": "unknown"}):
        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            LLMClient()


@respx.mock
def test_list_models_reads_openai_compatible_inventory():
    respx.get("http://localhost:8317/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "gpt-5.4"}, {"id": "gpt-5.3-codex"}]},
        )
    )

    with patch.dict(
        "os.environ",
        {
            "LLM_MODE": "setup-token",
            "LLM_PROVIDER": "openai",
            "LLM_PROXY_URL": "http://localhost:8317",
        },
    ):
        client = LLMClient()
        models = client.list_models()
        client.close()

    assert models == ["gpt-5.4", "gpt-5.3-codex"]
