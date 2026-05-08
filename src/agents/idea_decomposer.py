"""Agent that decomposes vague product ideas into concrete research paths."""

from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.models.plan import DecompositionResult, ResearchPath


class IdeaDecomposer(BaseAgent):
    """Decomposes a vague product concept into multiple concrete research paths."""

    agent_name = "idea_decomposer"

    def run(self, *, raw_input: str, max_paths: int = 1) -> DecompositionResult:
        """Decompose a product idea into research paths.

        Args:
            raw_input: The user's vague product idea or concept.
            max_paths: Maximum number of paths to generate.

        Returns:
            DecompositionResult with interpreted paths.
        """
        self.logger.info("Decomposing idea: %s", raw_input[:100])

        result = self._call_llm_json(
            prompt=self._build_prompt(raw_input, max_paths=max_paths),
            system=(
                "你是一位资深产品策略师和技术分析师。"
                "分析产品创意并将其拆解为具体的研究路线。所有文本内容必须使用中文输出。"
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

    def _build_prompt(self, raw_input: str, *, max_paths: int) -> str:
        try:
            prompt = self._render_template(
                "decompose_idea",
                {"raw_input": raw_input},
            )
            return (
                f"{prompt}\n\n"
                "## 路径数量约束\n"
                f"- 最多生成 {max_paths} 条路线。\n"
                "- 如果最多只能生成 1 条路线，请输出一条综合性的主路线，不要把实现细节拆成多条平行路线。\n"
                "- 只有当用户明确要求多路线对比，或上游确认需要分路径调研时，才输出多条互斥技术路线。\n"
                "- 不要为了覆盖组件而过度拆分；组件、模块、步骤应放在同一条路线的 technologies_needed/key_questions 中。\n"
            )
        except FileNotFoundError:
            # Fallback if template not found
            return (
                f"Decompose this product idea into concrete research paths:\n\n"
                f"{raw_input}\n\n"
                f"Return at most {max_paths} path(s). If max_paths is 1, return one integrated "
                f"primary path rather than multiple alternatives. Return JSON with: "
                f"interpretation, paths (path_id, title, description, "
                f"technologies_needed, key_questions, search_queries, priority), shared_technologies"
            )
