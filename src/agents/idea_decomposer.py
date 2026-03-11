"""Agent that decomposes vague product ideas into concrete research paths."""

from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.models.plan import DecompositionResult, ResearchPath


class IdeaDecomposer(BaseAgent):
    """Decomposes a vague product concept into multiple concrete research paths."""

    agent_name = "idea_decomposer"

    def run(self, *, raw_input: str, max_paths: int = 5) -> DecompositionResult:
        """Decompose a product idea into research paths.

        Args:
            raw_input: The user's vague product idea or concept.
            max_paths: Maximum number of paths to generate.

        Returns:
            DecompositionResult with interpreted paths.
        """
        self.logger.info("Decomposing idea: %s", raw_input[:100])

        result = self._call_llm_json(
            prompt=self._build_prompt(raw_input),
            system=(
                "You are an expert product strategist and technology analyst. "
                "Analyze product ideas and decompose them into concrete research paths."
            ),
            temperature=0.5,  # More creative for brainstorming
        )

        if not result or not isinstance(result, dict):
            self.logger.error("Failed to decompose idea - empty or invalid LLM response")
            return DecompositionResult(
                original_input=raw_input,
                interpretation="Failed to decompose",
                paths=[],
            )

        # Build paths from response
        paths = []
        for p in result.get("paths", [])[:max_paths]:
            paths.append(ResearchPath(
                path_id=p.get("path_id", f"p{len(paths)+1}"),
                title=p.get("title", ""),
                description=p.get("description", ""),
                technologies_needed=p.get("technologies_needed", []),
                key_questions=p.get("key_questions", []),
                search_queries=p.get("search_queries", {}),
                priority=float(p.get("priority", 0.5)),
            ))

        decomposition = DecompositionResult(
            original_input=raw_input,
            interpretation=result.get("interpretation", ""),
            paths=paths,
            shared_technologies=result.get("shared_technologies", []),
        )

        self.logger.info(
            "Decomposed into %d paths: %s",
            len(paths),
            [p.title for p in paths],
        )
        return decomposition

    def _build_prompt(self, raw_input: str) -> str:
        try:
            return self._render_template(
                "decompose_idea",
                {"raw_input": raw_input},
            )
        except FileNotFoundError:
            # Fallback if template not found
            return (
                f"Decompose this product idea into concrete research paths:\n\n"
                f"{raw_input}\n\n"
                f"Return JSON with: interpretation, paths (path_id, title, description, "
                f"technologies_needed, key_questions, search_queries, priority), shared_technologies"
            )
