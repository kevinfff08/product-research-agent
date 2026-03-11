"""LLM client wrapper for Anthropic API via CLIProxyAPI."""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

from src.logging_config import get_logger

load_dotenv()

logger = get_logger("llm.client")


class LLMClient:
    """Wrapper around Anthropic API for structured LLM interactions.

    Supports two modes:
    - ``api-key``: Direct API call with ANTHROPIC_API_KEY
    - ``setup-token``: Route through CLIProxyAPI at localhost:8317
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 8192,
        base_url: str | None = None,
    ):
        self.model = model or os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
        self.max_tokens = max_tokens
        self._client: Anthropic | None = None

        # Determine mode from env
        llm_mode = os.environ.get("LLM_MODE", "api-key").strip().lower()

        if llm_mode == "setup-token":
            # Route through CLIProxyAPI — api_key is ignored by proxy
            self.api_key = "setup-token-placeholder"
            self.base_url = base_url or os.environ.get(
                "LLM_PROXY_URL", "http://localhost:8317"
            )
            logger.info(
                "LLMClient init: mode=setup-token, proxy=%s, model=%s",
                self.base_url, self.model,
            )
        else:
            # Direct API key mode
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            self.base_url = base_url
            has_key = bool(self.api_key and self.api_key.startswith("sk-"))
            logger.info(
                "LLMClient init: mode=api-key, has_key=%s, model=%s",
                has_key, self.model,
            )

    @property
    def client(self) -> Anthropic:
        if self._client is None:
            kwargs: dict = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = Anthropic(**kwargs)
            logger.debug("Anthropic client created (base_url=%s)", self.base_url or "default")
        return self._client

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text from a prompt."""
        tokens = max_tokens or self.max_tokens
        prompt_preview = prompt[:120].replace("\n", " ")
        logger.info(
            "LLM request: model=%s, max_tokens=%d, temp=%.1f, prompt=%s...",
            self.model, tokens, temperature, prompt_preview,
        )

        kwargs: dict = {
            "model": self.model,
            "max_tokens": tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        try:
            response = self.client.messages.create(**kwargs)
            elapsed = time.perf_counter() - t0
            usage = response.usage
            logger.info(
                "LLM response: %.1fs, input_tokens=%d, output_tokens=%d, stop=%s",
                elapsed, usage.input_tokens, usage.output_tokens, response.stop_reason,
            )
            return response.content[0].text
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("LLM request failed after %.1fs: %s", elapsed, exc)
            raise

    def generate_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Generate JSON output from a prompt.

        Adds instruction to return valid JSON only.
        """
        json_system = (system + "\n\n" if system else "") + (
            "You must respond with valid JSON only. No markdown fences, no explanations, "
            "just the JSON object/array."
        )
        return self.generate(
            prompt, system=json_system, temperature=temperature,
            max_tokens=max_tokens,
        )

    def render_template(self, template_name: str, variables: dict) -> str:
        """Read a prompt template and format it with variables. No LLM call."""
        template_path = Path(__file__).parent / "prompts" / "v1" / f"{template_name}.txt"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        template = template_path.read_text(encoding="utf-8")
        return template.format(**variables)

    def generate_with_template(
        self,
        template_name: str,
        variables: dict,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Render a prompt template and generate a response via LLM."""
        prompt = self.render_template(template_name, variables)
        return self.generate(
            prompt, system=system, temperature=temperature,
            max_tokens=max_tokens,
        )
