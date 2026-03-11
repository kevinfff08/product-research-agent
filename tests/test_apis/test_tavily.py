"""Tests for Tavily API client."""

from __future__ import annotations

import pytest
import httpx
import respx

from src.apis.tavily_client import TavilyClient


@pytest.fixture
def tavily(tmp_path):
    return TavilyClient(api_key="test-key", cache_dir=tmp_path / "cache")


@pytest.mark.asyncio
async def test_search_success(tavily):
    mock_response = {
        "query": "AI code review",
        "results": [
            {
                "title": "Best AI Code Review Tools",
                "url": "https://example.com/review",
                "content": "Here are the top tools...",
                "score": 0.95,
            }
        ],
    }

    with respx.mock:
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        result = await tavily.search("AI code review")

    assert "results" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Best AI Code Review Tools"


@pytest.mark.asyncio
async def test_search_empty_results(tavily):
    with respx.mock:
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json={"query": "nothing", "results": []})
        )
        result = await tavily.search("nothing")

    assert result["results"] == []


@pytest.mark.asyncio
async def test_search_api_error(tavily):
    with respx.mock:
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(500, json={"error": "Internal Server Error"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            await tavily.search("test query")


@pytest.mark.asyncio
async def test_close(tavily):
    await tavily.close()
    # Should not error on double close
    await tavily.close()
