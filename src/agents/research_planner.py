"""Agent that generates a structured research plan from decomposed paths."""

from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.models.plan import DecompositionResult, ResearchPlan, SearchQuery
from src.models.common import ResearchWeight


class ResearchPlanner(BaseAgent):
    """Generates a structured research plan with search queries for each source."""

    agent_name = "research_planner"

    def run(
        self,
        *,
        decomposition: DecompositionResult,
        weights: ResearchWeight | None = None,
    ) -> ResearchPlan:
        """Generate a research plan from decomposition results.

        Args:
            decomposition: The decomposed research paths.
            weights: Research type weights (industry/academic/community).

        Returns:
            ResearchPlan with search queries organized by source.
        """
        weights = weights or ResearchWeight()
        self.logger.info("Generating research plan for %d paths", len(decomposition.paths))

        decomposition_json = json.dumps(
            [p.model_dump() for p in decomposition.paths],
            indent=2,
        )

        result = self._call_llm_json(
            prompt=self._build_prompt(decomposition, decomposition_json, weights),
            system=(
                "You are a research planning expert. Generate targeted search queries "
                "for multiple data sources to cover a product/technology landscape."
            ),
            temperature=0.3,
        )

        if not result or not isinstance(result, dict):
            self.logger.warning("LLM returned invalid plan, using fallback queries")
            return self._fallback_plan(decomposition)

        # Build search queries
        queries = []
        for q in result.get("search_queries", []):
            queries.append(SearchQuery(
                query=q.get("query", ""),
                source=q.get("source", "tavily"),
                path_id=q.get("path_id", ""),
                priority=float(q.get("priority", 0.5)),
            ))

        plan = ResearchPlan(
            original_input=decomposition.original_input,
            paths=decomposition.paths,
            search_queries=queries,
            estimated_api_calls=result.get("estimated_api_calls", len(queries)),
            estimated_llm_calls=result.get("estimated_llm_calls", len(queries)),
        )

        self.logger.info(
            "Research plan: %d queries (%d API, %d LLM calls)",
            len(queries), plan.estimated_api_calls, plan.estimated_llm_calls,
        )
        return plan

    def _build_prompt(
        self,
        decomposition: DecompositionResult,
        decomposition_json: str,
        weights: ResearchWeight,
    ) -> str:
        try:
            return self._render_template(
                "generate_research_plan",
                {
                    "original_input": decomposition.original_input,
                    "decomposition_json": decomposition_json,
                    "weight_industry": weights.industry,
                    "weight_academic": weights.academic,
                    "weight_community": weights.community,
                },
            )
        except FileNotFoundError:
            return (
                f"Generate search queries for:\n{decomposition_json}\n\n"
                f"Weights: industry={weights.industry}, academic={weights.academic}\n"
                f"Return JSON with search_queries list."
            )

    def _fallback_plan(self, decomposition: DecompositionResult) -> ResearchPlan:
        """Generate a basic plan from the decomposition's own search queries."""
        queries = []
        for path in decomposition.paths:
            for source, query_list in path.search_queries.items():
                source_map = {"web": "tavily", "academic": "semantic_scholar", "code": "github"}
                mapped_source = source_map.get(source, "tavily")
                for q in query_list:
                    queries.append(SearchQuery(
                        query=q,
                        source=mapped_source,
                        path_id=path.path_id,
                        priority=path.priority,
                    ))
        return ResearchPlan(
            original_input=decomposition.original_input,
            paths=decomposition.paths,
            search_queries=queries,
            estimated_api_calls=len(queries),
            estimated_llm_calls=len(queries),
        )
