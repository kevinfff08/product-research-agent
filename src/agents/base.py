"""Base class for all research agents."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from src.llm.client import LLMClient
from src.logging_config import get_logger
from src.storage.local_store import LocalStore
from src.utils.json_repair import repair_json


class BaseAgent(ABC):
    """Abstract base for all research agents.

    Every agent receives an LLMClient and a LocalStore. The agent's
    job is to take structured input, call the LLM with a carefully
    crafted prompt, parse the JSON response, and return a Pydantic model.
    """

    agent_name: str = "base"

    def __init__(self, llm: LLMClient, store: LocalStore):
        self.llm = llm
        self.store = store
        self.logger = get_logger(f"agents.{self.agent_name}")

    def _call_llm_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> dict | list | None:
        """Call LLM expecting JSON, with automatic repair."""
        raw = self.llm.generate_json(
            prompt, system=system, temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = repair_json(raw)
        if parsed is None:
            self.logger.warning(
                "Failed to parse LLM JSON response (first 200 chars): %s",
                raw[:200],
            )
        return parsed

    async def _call_llm_json_async(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> dict | list | None:
        """Call the sync LLM client from async agents without blocking the loop."""
        return await asyncio.to_thread(
            self._call_llm_json,
            prompt,
            system,
            temperature,
            max_tokens,
        )

    def _call_llm_text(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Call LLM expecting plain text."""
        return self.llm.generate(
            prompt, system=system, temperature=temperature,
            max_tokens=max_tokens,
        )

    def _render_template(self, template_name: str, variables: dict) -> str:
        """Render a prompt template without making an LLM call."""
        return self.llm.render_template(template_name, variables)

    def _call_llm_template(
        self,
        template_name: str,
        variables: dict,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Call LLM using a prompt template."""
        return self.llm.generate_with_template(
            template_name, variables, system=system,
            temperature=temperature, max_tokens=max_tokens,
        )

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute this agent's task. Subclasses must implement."""
        ...
