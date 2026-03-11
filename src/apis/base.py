"""Base HTTP client with rate limiting and caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx

from src.logging_config import get_logger

logger = get_logger("apis.base")


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, requests_per_second: float = 1.0):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class ResponseCache:
    """Simple file-based response cache."""

    def __init__(self, cache_dir: Path, ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, url: str, params: dict | None = None) -> str:
        raw = url + json.dumps(params or {}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, url: str, params: dict | None = None) -> dict | None:
        key = self._key(url, params)
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - data.get("_cached_at", 0) > self.ttl_seconds:
                path.unlink(missing_ok=True)
                return None
            return data.get("response")
        except (json.JSONDecodeError, KeyError):
            path.unlink(missing_ok=True)
            return None

    def set(self, url: str, params: dict | None, response: dict) -> None:
        key = self._key(url, params)
        path = self.cache_dir / f"{key}.json"
        data = {"_cached_at": time.time(), "response": response}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class BaseAPIClient:
    """Base class for API clients with rate limiting and caching."""

    BASE_URL: str = ""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        requests_per_second: float = 1.0,
    ):
        self.api_key = api_key
        self.rate_limiter = RateLimiter(requests_per_second)
        self.cache = ResponseCache(cache_dir) if cache_dir else None
        self._client: httpx.AsyncClient | None = None

    @property
    def headers(self) -> dict[str, str]:
        headers = {"User-Agent": "ProductResearch/0.1 (research agent)"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers=self.headers,
                timeout=30.0,
            )
        return self._client

    async def get(self, path: str, params: dict | None = None) -> dict:
        """Make a GET request with rate limiting and optional caching."""
        url = f"{self.BASE_URL}{path}"

        if self.cache:
            cached = self.cache.get(url, params)
            if cached is not None:
                logger.debug("Cache hit: %s", url)
                return cached

        await self.rate_limiter.acquire()
        client = await self._get_client()
        logger.debug("HTTP GET %s params=%s", url, params)
        try:
            response = await client.get(path, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("HTTP %d from %s: %s", exc.response.status_code, url, exc.response.text[:200])
            raise
        except httpx.RequestError as exc:
            logger.error("HTTP request error for %s: %s", url, exc)
            raise
        data = response.json()

        if self.cache:
            self.cache.set(url, params, data)

        return data

    async def post(self, path: str, json_data: dict | None = None) -> dict:
        """Make a POST request with rate limiting."""
        url = f"{self.BASE_URL}{path}"
        await self.rate_limiter.acquire()
        client = await self._get_client()
        logger.debug("HTTP POST %s", url)
        try:
            response = await client.post(path, json=json_data)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("HTTP %d from %s: %s", exc.response.status_code, url, exc.response.text[:200])
            raise
        except httpx.RequestError as exc:
            logger.error("HTTP request error for %s: %s", url, exc)
            raise
        return response.json()

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
