"""Agent that generates a structured research plan from decomposed paths."""

from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.models.plan import DecompositionResult, ResearchPlan, SearchQuery
from src.models.common import ResearchWeight


_SOURCE_LIMITS = {
    "tavily": 8,
    "github": 8,
    "semantic_scholar": 5,
    "arxiv": 4,
}


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

        queries = []
        for q in result.get("search_queries", []):
            queries.append(SearchQuery(
                query=q.get("query", ""),
                source=q.get("source", "tavily"),
                path_id=q.get("path_id", ""),
                priority=float(q.get("priority", 0.5)),
                intent=q.get("intent", "general"),
            ))
        queries = self._optimize_queries(decomposition, queries)

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
        queries = self._optimize_queries(decomposition, queries)
        return ResearchPlan(
            original_input=decomposition.original_input,
            paths=decomposition.paths,
            search_queries=queries,
            estimated_api_calls=len(queries),
            estimated_llm_calls=len(queries),
        )

    def _optimize_queries(
        self,
        decomposition: DecompositionResult,
        queries: list[SearchQuery],
    ) -> list[SearchQuery]:
        """Expand, deduplicate, and cap queries by source/path for broader coverage."""
        expanded = list(queries)
        for path in decomposition.paths:
            base_terms = [path.title, *path.technologies_needed[:4]]
            base_terms.extend(path.search_queries.get("web", [])[:3])
            for term in self._unique_terms(base_terms):
                expanded.extend(
                    [
                        SearchQuery(
                            query=f"{term} product landscape competitors",
                            source="tavily",
                            path_id=path.path_id,
                            priority=path.priority,
                            intent="product_landscape",
                        ),
                        SearchQuery(
                            query=f"{term} alternatives comparison pricing",
                            source="tavily",
                            path_id=path.path_id,
                            priority=max(path.priority - 0.05, 0.1),
                            intent="alternatives",
                        ),
                        SearchQuery(
                            query=f"{term} open source github",
                            source="github",
                            path_id=path.path_id,
                            priority=path.priority,
                            intent="repo_discovery",
                        ),
                        SearchQuery(
                            query=f"{term} benchmark evaluation survey",
                            source="semantic_scholar",
                            path_id=path.path_id,
                            priority=max(path.priority - 0.1, 0.1),
                            intent="evidence",
                        ),
                        SearchQuery(
                            query=f"{term} survey benchmark",
                            source="arxiv",
                            path_id=path.path_id,
                            priority=max(path.priority - 0.15, 0.1),
                            intent="preprint",
                        ),
                    ]
                )

        deduped: dict[tuple[str, str, str], SearchQuery] = {}
        for query in expanded:
            if not query.query.strip():
                continue
            key = (query.path_id, query.source, self._normalize_query(query.query))
            existing = deduped.get(key)
            if existing is None or query.priority > existing.priority:
                deduped[key] = query

        grouped: dict[tuple[str, str], list[SearchQuery]] = {}
        for query in deduped.values():
            grouped.setdefault((query.path_id, query.source), []).append(query)

        optimized: list[SearchQuery] = []
        for (_, source), group in grouped.items():
            limit = _SOURCE_LIMITS.get(source, 5)
            optimized.extend(
                sorted(group, key=lambda q: q.priority, reverse=True)[:limit]
            )
        return sorted(optimized, key=lambda q: (q.path_id, q.source, -q.priority, q.query))

    @staticmethod
    def _unique_terms(terms: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for term in terms:
            cleaned = " ".join(term.split())
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                unique.append(cleaned)
        return unique[:8]

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.lower().split())
