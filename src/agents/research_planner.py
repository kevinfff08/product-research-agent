"""Agent that generates a structured research plan from decomposed paths."""

from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.models.plan import DecompositionResult, ResearchPath, ResearchPlan, SearchQuery
from src.models.common import ResearchWeight
from src.utils.text_utils import english_search_query, is_useful_english_query


_SOURCE_LIMITS = {
    "tavily": 8,
    "code_web": 5,
    "academic_web": 4,
    "openalex": 4,
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
                "你是一位研究规划专家。为多个数据源生成有针对性的搜索查询，"
                "以覆盖产品/技术全景。搜索查询使用英文以获得最佳结果。"
            ),
            temperature=0.3,
        )

        if not result or not isinstance(result, dict):
            self.logger.warning("LLM returned invalid plan, using fallback queries")
            return self._fallback_plan(decomposition)

        queries = []
        for q in result.get("search_queries", []):
            source = self._normalize_source(q.get("source", "tavily"))
            queries.append(SearchQuery(
                query=q.get("query", ""),
                source=source,
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
                source_map = {"web": "tavily", "academic": "academic_web", "code": "code_web"}
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
            base_terms = self._english_base_terms(path)
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
                            query=f"{term} open source implementation repository",
                            source="code_web",
                            path_id=path.path_id,
                            priority=path.priority,
                            intent="repo_discovery",
                        ),
                        SearchQuery(
                            query=f"{term} research paper",
                            source="academic_web",
                            path_id=path.path_id,
                            priority=max(path.priority - 0.1, 0.1),
                            intent="evidence",
                        ),
                        SearchQuery(
                            query=f"{term} peer reviewed research",
                            source="openalex",
                            path_id=path.path_id,
                            priority=max(path.priority - 0.12, 0.1),
                            intent="published_evidence",
                        ),
                        SearchQuery(
                            query=term,
                            source="arxiv",
                            path_id=path.path_id,
                            priority=max(path.priority - 0.15, 0.1),
                            intent="preprint",
                        ),
                    ]
                )

        deduped: dict[tuple[str, str, str], SearchQuery] = {}
        for query in expanded:
            normalized_query = english_search_query(query.query)
            if not is_useful_english_query(normalized_query):
                continue
            query.query = normalized_query
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
    def _english_base_terms(path: ResearchPath) -> list[str]:
        terms: list[str] = []
        for source in ("web", "academic", "code"):
            terms.extend(path.search_queries.get(source, [])[:4])
        terms.extend(path.technologies_needed[:4])
        terms.append(path.title)
        normalized = [english_search_query(term) for term in terms]
        return [term for term in normalized if is_useful_english_query(term)]

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

    @staticmethod
    def _normalize_source(source: str) -> str:
        source_map = {
            "web": "tavily",
            "semantic_scholar": "academic_web",
            "github": "code_web",
            "code": "code_web",
            "academic": "academic_web",
        }
        normalized = source.strip().lower()
        return source_map.get(normalized, normalized or "tavily")
