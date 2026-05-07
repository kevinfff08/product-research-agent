"""LLM client for OpenAI-compatible providers and CLIProxyAPI."""

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

_DEFAULT_PROXY_URL = "http://localhost:8317"
_PROVIDER_DEFAULTS: dict[str, dict[str, str | bool]] = {
    "openai": {
        "model": "gpt-5.4",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "ensure_v1": True,
    },
    "deepseek": {
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "ensure_v1": False,
    },
    "google": {
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GOOGLE_API_KEY",
        "base_url_env": "GOOGLE_BASE_URL",
        "ensure_v1": False,
    },
}
_ROUTING_ERROR_MARKERS = (
    "unknown provider for model",
    "model_not_found",
    "invalid model",
    "model does not exist",
)
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY = 2.0
_RETRY_MAX_DELAY = 30.0


class LLMClient:
    """OpenAI-compatible client for direct providers or CLIProxyAPI.

    Supports two modes:
    - ``setup-token``: route through CLIProxyAPI, defaulting to localhost:8317.
    - ``api-key``: call provider APIs selected by ``LLM_PROVIDER``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 8192,
        base_url: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
        if self.provider not in _PROVIDER_DEFAULTS:
            supported = ", ".join(sorted(_PROVIDER_DEFAULTS))
            raise ValueError(f"Unsupported LLM_PROVIDER '{self.provider}'. Use one of: {supported}")

        self.mode = os.environ.get("LLM_MODE", "api-key").strip().lower()
        if self.mode not in {"setup-token", "api-key"}:
            raise ValueError("LLM_MODE must be one of: setup-token, api-key")

        provider_config = _PROVIDER_DEFAULTS[self.provider]
        self.model = model or self._default_model_for_mode()
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: httpx.Client | None = None

        if self.mode == "setup-token":
            proxy_url = base_url or os.environ.get("LLM_PROXY_URL", _DEFAULT_PROXY_URL)
            self.base_url = self._normalize_base_url(proxy_url, ensure_v1=True)
            self.api_key = api_key or os.environ.get("LLM_API_KEY", "setup-token")
            logger.info(
                "LLMClient init: mode=setup-token, provider=%s, proxy=%s, model=%s",
                self.provider,
                self.base_url,
                self.model,
            )
        else:
            base_url_env = str(provider_config["base_url_env"])
            openai_url = (
                base_url
                or os.environ.get(base_url_env)
                or os.environ.get("LLM_BASE_URL")
                or str(provider_config["base_url"])
            )
            self.base_url = self._normalize_base_url(
                openai_url,
                ensure_v1=bool(provider_config["ensure_v1"]),
            )
            api_key_env = str(provider_config["api_key_env"])
            self.api_key = api_key or os.environ.get(api_key_env) or os.environ.get("LLM_API_KEY", "")
            logger.info(
                "LLMClient init: mode=api-key, provider=%s, has_key=%s, base_url=%s, model=%s",
                self.provider,
                bool(self.api_key),
                self.base_url,
                self.model,
            )

    @staticmethod
    def _normalize_base_url(base_url: str, *, ensure_v1: bool = True) -> str:
        """Return a normalized OpenAI-compatible base URL."""
        normalized = base_url.rstrip("/")
        if ensure_v1 and not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized

    def _default_model_for_mode(self) -> str:
        """Pick a default model. LLM_MODEL overrides the provider default."""
        provider_config = _PROVIDER_DEFAULTS[self.provider]
        default_model = str(provider_config["model"])
        return os.environ.get("LLM_MODEL") or default_model

    @property
    def client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            if not self.api_key:
                raise RuntimeError(
                    f"{_PROVIDER_DEFAULTS[self.provider]['api_key_env']} is required "
                    "when LLM_MODE=api-key. Use LLM_MODE=setup-token for CLIProxyAPI."
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

    def list_models(self) -> list[str]:
        """Return model IDs exposed by the configured OpenAI-compatible endpoint."""
        response = self.client.get("/models")
        if response.status_code >= 400:
            raise RuntimeError(f"Failed to list models: HTTP {response.status_code}")
        payload = response.json()
        data = payload.get("data", []) if isinstance(payload, dict) else []
        models: list[str] = []
        for item in data:
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
        return models

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text from a prompt via chat completions.

        Retries on timeouts, connection errors, and transient server errors
        (HTTP 429/5xx) with exponential backoff.
        """
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

        last_exc: Exception | None = None
        t0 = time.perf_counter()
        for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
            try:
                response = self.client.post("/chat/completions", json=payload)
                elapsed = time.perf_counter() - t0
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _RETRY_MAX_ATTEMPTS:
                    delay = min(_RETRY_BASE_DELAY * (2 ** (attempt - 1)), _RETRY_MAX_DELAY)
                    logger.warning(
                        "LLM HTTP %d on attempt %d/%d, retrying in %.1fs...",
                        response.status_code, attempt, _RETRY_MAX_ATTEMPTS, delay,
                    )
                    time.sleep(delay)
                    t0 = time.perf_counter()
                    continue
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
            except (httpx.ReadTimeout, httpx.ConnectError,
                    httpx.RemoteProtocolError, httpx.ReadError) as exc:
                last_exc = exc
                elapsed = time.perf_counter() - t0
                if attempt < _RETRY_MAX_ATTEMPTS:
                    delay = min(_RETRY_BASE_DELAY * (2 ** (attempt - 1)), _RETRY_MAX_DELAY)
                    logger.warning(
                        "LLM %s on attempt %d/%d after %.1fs, retrying in %.1fs...",
                        type(exc).__name__, attempt, _RETRY_MAX_ATTEMPTS, elapsed, delay,
                    )
                    time.sleep(delay)
                    t0 = time.perf_counter()
                    continue
                logger.error("LLM request failed after %d attempts (%.1fs): %s",
                             _RETRY_MAX_ATTEMPTS, elapsed, exc)
                raise
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.error("LLM request failed after %.1fs: %s", elapsed, exc)
                raise

        assert last_exc is not None
        raise last_exc

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
            available = self._safe_list_models()
            suffix = ""
            if available:
                suffix = f" Available models from {self.base_url}: {', '.join(available[:20])}"
            raise RuntimeError(
                f"Non-retryable LLM model routing error for model '{self.model}': {message}"
                f"{suffix}"
            )

        raise RuntimeError(
            f"LLM request failed with HTTP {response.status_code} after {elapsed:.1f}s: {message}"
        )

    def _safe_list_models(self) -> list[str]:
        """Best-effort model inventory for clearer setup-token routing errors."""
        try:
            return self.list_models()
        except Exception as exc:
            logger.debug("Could not list models from %s: %s", self.base_url, exc)
            return []

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
            "just the JSON object/array. "
            "请只返回有效的JSON对象或数组，不要使用markdown代码块，不要添加任何解释。"
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
