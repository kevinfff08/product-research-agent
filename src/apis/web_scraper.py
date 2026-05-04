"""Simple web scraper for extracting content from URLs."""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

from src.logging_config import get_logger
from src.utils.text_utils import truncate

logger = get_logger("apis.web_scraper")


class WebScraper:
    """Lightweight web content scraper using httpx + BeautifulSoup.

    Extracts clean text content from web pages without requiring
    a browser. For JavaScript-heavy pages, rely on Tavily search
    which handles rendering.
    """

    def __init__(self, timeout: int = 30, max_content_length: int = 50000):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )
        return self._client

    async def scrape_url(self, url: str) -> dict:
        """Scrape a URL and return extracted content.

        Returns:
            Dict with keys: url, title, text, links, success, error.
        """
        result = {"url": url, "title": "", "text": "", "links": [], "success": False, "error": ""}

        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                result["error"] = f"Non-HTML content type: {content_type}"
                return result

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Extract title
            title_tag = soup.find("title")
            result["title"] = title_tag.get_text(strip=True) if title_tag else ""

            # Extract main content
            # Try article or main first, then fall back to body
            main = soup.find("article") or soup.find("main") or soup.find("body")
            if main:
                text = main.get_text(separator="\n", strip=True)
                result["text"] = truncate(text, self.max_content_length)
            else:
                result["text"] = truncate(soup.get_text(separator="\n", strip=True), self.max_content_length)

            # Extract links
            for a_tag in soup.find_all("a", href=True, limit=50):
                href = a_tag["href"]
                if href.startswith(("http://", "https://")):
                    link_text = a_tag.get_text(strip=True)
                    if link_text:
                        result["links"].append({"text": link_text, "url": href})

            result["success"] = True
            logger.info("Scraped %s: %d chars, %d links", url, len(result["text"]), len(result["links"]))

        except httpx.HTTPStatusError as exc:
            result["error"] = f"HTTP {exc.response.status_code}"
            logger.warning("Scrape failed for %s: HTTP %d", url, exc.response.status_code)
        except Exception as exc:
            result["error"] = str(exc)
            logger.warning("Scrape failed for %s: %s", url, exc)

        return result

    async def scrape_multiple(self, urls: list[str]) -> list[dict]:
        """Scrape multiple URLs concurrently."""
        import asyncio
        tasks = [self.scrape_url(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> WebScraper:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
