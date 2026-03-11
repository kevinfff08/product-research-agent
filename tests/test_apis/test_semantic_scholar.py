"""Tests for Semantic Scholar API client."""

from __future__ import annotations

import pytest
import httpx
import respx

from src.apis.semantic_scholar import SemanticScholarClient


@pytest.fixture
def s2_client(tmp_path):
    return SemanticScholarClient(api_key="test-key", cache_dir=tmp_path / "cache")


@pytest.fixture
def mock_paper():
    return {
        "paperId": "abc123",
        "title": "Attention Is All You Need",
        "authors": [{"name": "Vaswani"}, {"name": "Shazeer"}],
        "year": 2017,
        "venue": "NeurIPS",
        "citationCount": 100000,
        "abstract": "We propose the Transformer architecture...",
        "externalIds": {"DOI": "10.1234/test", "ArXiv": "1706.03762"},
    }


@pytest.mark.asyncio
async def test_search_papers(s2_client, mock_paper):
    with respx.mock:
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=httpx.Response(200, json={"data": [mock_paper]})
        )
        papers = await s2_client.search_papers("transformer attention")

    assert len(papers) == 1
    assert papers[0]["title"] == "Attention Is All You Need"


@pytest.mark.asyncio
async def test_get_paper(s2_client, mock_paper):
    with respx.mock:
        respx.get("https://api.semanticscholar.org/graph/v1/paper/abc123").mock(
            return_value=httpx.Response(200, json=mock_paper)
        )
        paper = await s2_client.get_paper("abc123")

    assert paper["paperId"] == "abc123"


@pytest.mark.asyncio
async def test_get_paper_citations(s2_client):
    with respx.mock:
        respx.get("https://api.semanticscholar.org/graph/v1/paper/abc123/citations").mock(
            return_value=httpx.Response(200, json={
                "data": [{"citingPaper": {"paperId": "cited1", "title": "Follow-up"}}]
            })
        )
        citations = await s2_client.get_paper_citations("abc123")

    assert len(citations) == 1
    assert citations[0]["paperId"] == "cited1"


@pytest.mark.asyncio
async def test_headers_with_key(s2_client):
    headers = s2_client.headers
    assert "x-api-key" in headers
    assert headers["x-api-key"] == "test-key"


@pytest.mark.asyncio
async def test_headers_without_key(tmp_path):
    client = SemanticScholarClient(api_key=None, cache_dir=tmp_path / "cache")
    headers = client.headers
    assert "x-api-key" not in headers


@pytest.mark.asyncio
async def test_close(s2_client):
    await s2_client.close()
