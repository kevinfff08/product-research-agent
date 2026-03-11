"""Tests for web scraper."""

from __future__ import annotations

import pytest
import httpx
import respx

from src.apis.web_scraper import WebScraper


@pytest.fixture
def scraper():
    return WebScraper()


@pytest.mark.asyncio
async def test_scrape_url_success(scraper):
    html = "<html><head><title>Test Page</title></head><body><p>Hello world</p></body></html>"
    with respx.mock:
        respx.get("https://example.com/test").mock(
            return_value=httpx.Response(200, text=html, headers={"content-type": "text/html"})
        )
        result = await scraper.scrape_url("https://example.com/test")

    assert result["success"] is True
    assert result["title"] == "Test Page"
    assert "Hello world" in result["text"]


@pytest.mark.asyncio
async def test_scrape_url_failure(scraper):
    with respx.mock:
        respx.get("https://example.com/404").mock(
            return_value=httpx.Response(404)
        )
        result = await scraper.scrape_url("https://example.com/404")

    assert result["success"] is False


@pytest.mark.asyncio
async def test_scrape_multiple(scraper):
    with respx.mock:
        respx.get("https://a.com").mock(
            return_value=httpx.Response(200, text="<html><body>A</body></html>",
                                        headers={"content-type": "text/html"})
        )
        respx.get("https://b.com").mock(
            return_value=httpx.Response(200, text="<html><body>B</body></html>",
                                        headers={"content-type": "text/html"})
        )
        results = await scraper.scrape_multiple(["https://a.com", "https://b.com"])

    assert len(results) == 2
    assert all(r["success"] for r in results)


@pytest.mark.asyncio
async def test_close(scraper):
    await scraper.close()
