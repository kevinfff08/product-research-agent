"""Codex LLM client for OpenAI-compatible APIs and CLIProxyAPI."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from src.logging_config import get_logger

load_dotenv()

logger = get_logger("llm.client")

_DEFAULT_MODEL = "gpt-5.4"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_PROXY_URL = "http://localhost:8317"
_ROUTING_ERROR_MARKERS = (
    "unknown provider for model",
    "model_not_found",
    "invalid model",
    "model does not exist",
)


class LLMClient:
    """OpenAI-compatible client constrained to Codex-capable models.

    Supports two modes:
    - ``setup-token``: route through CLIProxyAPI, defaulting to localhost:8317.
    - ``api-key``: call an OpenAI-compatible endpoint with ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 8192,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or os.environ.get("LLM_MODEL", _DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: httpx.Client | None = None

        self.mode = os.environ.get("LLM_MODE", "api-key").strip().lower()
        if self.mode == "setup-token":
            proxy_url = base_url or os.environ.get("LLM_PROXY_URL", _DEFAULT_PROXY_URL)
            self.base_url = self._normalize_base_url(proxy_url)
            self.api_key = api_key or os.environ.get("LLM_API_KEY", "setup-token")
            logger.info(
                "LLMClient init: mode=setup-token, proxy=%s, model=%s",
                self.base_url,
                self.model,
            )
        else:
            openai_url = (
                base_url
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("LLM_BASE_URL")
                or _DEFAULT_OPENAI_BASE_URL
            )
            self.base_url = self._normalize_base_url(openai_url)
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            logger.info(
                "LLMClient init: mode=api-key, has_key=%s, base_url=%s, model=%s",
                bool(self.api_key),
                self.base_url,
                self.model,
            )

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Return a base URL ending in /v1 for OpenAI-compatible requests."""
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is required when LLM_MODE=api-key. "
                    "Use LLM_MODE=setup-token for CLIProxyAPI."
                )
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            logger.debug("OpenAI-compatible client created (base_url=%s)", self.base_url)
        return self._client

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            self._client.close()

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text from a prompt via chat completions."""
        tokens = max_tokens or self.max_tokens
        prompt_preview = prompt[:120].replace("\n", " ")
        logger.info(
            "LLM request: model=%s, max_tokens=%d, temp=%.1f, prompt=%s...",
            self.model,
            tokens,
            temperature,
            prompt_preview,
        )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": tokens,
        }

        t0 = time.perf_counter()
        try:
            response = self.client.post("/chat/completions", json=payload)
            elapsed = time.perf_counter() - t0
            if response.status_code >= 400:
                self._raise_api_error(response, elapsed)
            data = response.json()
            text = self._extract_message_text(data)
            usage = data.get("usage") or {}
            logger.info(
                "LLM response: %.1fs, input_tokens=%s, output_tokens=%s",
                elapsed,
                usage.get("prompt_tokens", "?"),
                usage.get("completion_tokens", "?"),
            )
            return text
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("LLM request failed after %.1fs: %s", elapsed, exc)
            raise

    def _raise_api_error(self, response: httpx.Response, elapsed: float) -> None:
        """Raise a clear error for API failures, especially model routing issues."""
        try:
            payload = response.json()
        except ValueError:
            payload = {}

        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            message = str(error.get("message") or error)
            code = str(error.get("code") or "")
        else:
            message = response.text
            code = ""

        marker_text = f"{code} {message}".lower()
        if any(marker in marker_text for marker in _ROUTING_ERROR_MARKERS):
            raise RuntimeError(
                f"Non-retryable LLM model routing error for model '{self.model}': {message}"
            )

        raise RuntimeError(
            f"LLM request failed with HTTP {response.status_code} after {elapsed:.1f}s: {message}"
        )

    @staticmethod
    def _extract_message_text(data: dict[str, Any]) -> str:
        """Extract text from an OpenAI-compatible chat completion response."""
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response did not contain any choices")

        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def generate_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Generate JSON output from a prompt."""
        json_system = (system + "\n\n" if system else "") + (
            "You must respond with valid JSON only. No markdown fences, no explanations, "
            "just the JSON object/array."
        )
        return self.generate(
            prompt,
            system=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def render_template(self, template_name: str, variables: dict[str, Any]) -> str:
        """Read a prompt template and format it with variables. No LLM call."""
        template_path = Path(__file__).parent / "prompts" / "v1" / f"{template_name}.txt"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        template = template_path.read_text(encoding="utf-8")
        return template.format(**variables)

    def generate_with_template(
        self,
        template_name: str,
        variables: dict[str, Any],
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Render a prompt template and generate a response via LLM."""
        prompt = self.render_template(template_name, variables)
        return self.generate(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
