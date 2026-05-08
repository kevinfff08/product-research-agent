"""Tests for arXiv API client."""

from __future__ import annotations

import pytest
import httpx
import respx

from src.apis.arxiv_client import ArxivClient

MOCK_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v1</id>
    <title>Test Paper on LLMs</title>
    <summary>This paper presents a novel approach to LLM-based code review.</summary>
    <published>2023-01-15T00:00:00Z</published>
    <updated>2023-01-16T00:00:00Z</updated>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cs.CL"/>
    <category term="cs.SE"/>
    <link title="pdf" href="http://arxiv.org/pdf/2301.12345v1"/>
  </entry>
</feed>"""


@pytest.fixture
def arxiv():
    return ArxivClient()


@pytest.mark.asyncio
async def test_search(arxiv):
    with respx.mock:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=MOCK_ARXIV_XML)
        )
        papers = await arxiv.search("LLM code review")

    assert len(papers) == 1
    assert papers[0]["title"] == "Test Paper on LLMs"
    assert papers[0]["arxiv_id"] == "2301.12345v1"
    assert len(papers[0]["authors"]) == 2
    assert "cs.CL" in papers[0]["categories"]
    assert papers[0]["pdf_url"] == "https://arxiv.org/pdf/2301.12345v1"


@pytest.mark.asyncio
async def test_search_empty(arxiv):
    empty_xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    with respx.mock:
        respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=empty_xml)
        )
        papers = await arxiv.search("nonexistent topic xyz")

    assert papers == []


@pytest.mark.asyncio
async def test_search_rate_limit_opens_circuit(arxiv, monkeypatch):
    monkeypatch.setattr("src.apis.arxiv_client._MIN_INTERVAL", 0.0)
    monkeypatch.setattr("src.apis.arxiv_client._MAX_JITTER", 0.0)
    monkeypatch.setattr("src.apis.arxiv_client._RATE_LIMIT_COOLDOWN", 0.0)

    with respx.mock:
        route = respx.get("https://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(429, text="Rate exceeded.")
        )
        with pytest.raises(RuntimeError, match="temporarily disabled"):
            await arxiv.search("LLM code review")

    assert route.call_count == 2


@pytest.mark.asyncio
async def test_parse_response(arxiv):
    papers = arxiv._parse_response(MOCK_ARXIV_XML)
    assert len(papers) == 1
    paper = papers[0]
    assert paper["summary"].startswith("This paper presents")
    assert paper["published"] == "2023-01-15T00:00:00Z"


@pytest.mark.asyncio
async def test_download_pdf(arxiv, tmp_path):
    pdf_bytes = b"%PDF-1.4\nfake"
    with respx.mock:
        respx.get("https://arxiv.org/pdf/2301.12345.pdf").mock(
            return_value=httpx.Response(
                200,
                content=pdf_bytes,
                headers={"content-type": "application/pdf"},
            )
        )
        path = await arxiv.download_pdf("2301.12345", tmp_path)

    assert path.name == "2301.12345.pdf"
    assert path.read_bytes() == pdf_bytes


@pytest.mark.asyncio
async def test_close(arxiv):
    await arxiv.close()
    await arxiv.close()  # Double close should not error
