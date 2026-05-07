"""Agent that conducts academic research using arXiv, OpenAlex, and web search."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from datetime import datetime, timezone

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.storage.local_store import LocalStore
from src.apis.arxiv_client import ArxivClient
from src.apis.tavily_client import TavilyClient
from src.apis.openalex_client import OpenAlexClient
from src.models.plan import ResearchPath, SearchQuery
from src.models.academic import PaperAnalysis, AcademicResearchResult
from src.models.common import SourceReference, SourceType
from src.utils.text_utils import clean_search_query


_STOP_WORDS = {
    "this", "that", "with", "from", "they", "their", "have", "been",
    "were", "when", "where", "which", "what", "about", "into", "more",
    "some", "such", "than", "then", "also", "very", "just", "like",
    "over", "other", "these", "those", "each", "both", "after",
}

_VENUE_PRESTIGE_MAP: dict[str, str] = {
    "nature": "顶级期刊", "science": "顶级期刊", "cell": "顶级期刊",
    "neurips": "顶会", "icml": "顶会", "iclr": "顶会", "cvpr": "顶会",
    "iccv": "顶会", "eccv": "顶会", "acl": "顶会", "emnlp": "顶会",
    "naacl": "顶会", "aaai": "顶会", "ijcai": "顶会", "siggraph": "顶会",
    "kdd": "顶会", "www": "顶会", "sigir": "顶会", "sigmod": "顶会",
    "vldb": "顶会", "osdi": "顶会", "sosp": "顶会", "nsdi": "顶会",
    "mobicom": "顶会", "sigcomm": "顶会", "icse": "顶会", "fse": "顶会",
    "isca": "顶会", "micro": "顶会", "hpca": "顶会", "asplos": "顶会",
    "ieee": "期刊/会议", "acm": "期刊/会议", "springer": "期刊",
    "elsevier": "期刊", "arxiv": "预印本",
}


class AcademicResearcher(BaseAgent):
    """Conducts academic research across arXiv, OpenAlex, and web search.

    Three-tier search strategy:
    1. OpenAlex — high-impact published papers (cited, peer-reviewed, top venues)
    2. arXiv — latest preprints for cutting-edge work
    3. Tavily — academic web search for supplementary evidence
    """

    agent_name = "academic_researcher"

    def __init__(
        self,
        llm: LLMClient,
        store: LocalStore,
        academic_search: TavilyClient,
        arxiv: ArxivClient,
        openalex: OpenAlexClient | None = None,
        *,
        max_academic_queries: int = 4,
        max_arxiv_queries: int = 3,
        max_openalex_queries: int = 3,
        papers_per_query: int = 8,
        max_paper_analyses: int = 12,
        api_concurrency: int = 3,
        llm_concurrency: int = 1,
    ):
        super().__init__(llm, store)
        self.academic_search = academic_search
        self.arxiv = arxiv
        self.openalex = openalex
        self.max_academic_queries = max_academic_queries
        self.max_arxiv_queries = max_arxiv_queries
        self.max_openalex_queries = max_openalex_queries
        self.papers_per_query = papers_per_query
        self.max_paper_analyses = max_paper_analyses
        self.api_concurrency = max(1, api_concurrency)
        self.llm_concurrency = max(1, llm_concurrency)

    _current_path: ResearchPath | None = None

    async def run(
        self,
        *,
        path: ResearchPath,
        queries: list[SearchQuery] | None = None,
    ) -> AcademicResearchResult:
        """Run academic research for a single path."""
        self._current_path = path
        self.logger.info("Academic research for path: %s", path.title)

        a_queries = queries or [
            SearchQuery(query=q, source="academic_web", path_id=path.path_id)
            for q in path.search_queries.get("academic", [])
        ]

        web_queries = [
            q for q in a_queries
            if q.source in {"academic_web", "semantic_scholar", "tavily"}
        ] or a_queries
        arxiv_queries = [q for q in a_queries if q.source == "arxiv"]
        openalex_queries = [q for q in a_queries if q.source == "openalex"]
        if not openalex_queries and arxiv_queries:
            openalex_queries = arxiv_queries[: self.max_openalex_queries]
        if not arxiv_queries and not openalex_queries:
            arxiv_queries = a_queries[: self.max_arxiv_queries]

        tasks = [
            self._search_academic_web(web_queries),
            self._search_arxiv(arxiv_queries),
            self._search_openalex(openalex_queries),
        ]
        web_results, arxiv_results, openalex_results = await asyncio.gather(*tasks)

        all_papers = self._merge_papers(web_results, arxiv_results, openalex_results)
        analyzed_papers, sources = self._build_paper_records(all_papers)
        trends, frontiers, researchers = await self._synthesize_academic_context(
            analyzed_papers, path,
        )

        result = AcademicResearchResult(
            path_id=path.path_id,
            papers=analyzed_papers,
            research_trends=trends,
            frontier_directions=frontiers,
            key_researchers=researchers,
            sources=sources,
        )
        self.logger.info(
            "Academic research complete: %d papers analyzed",
            len(analyzed_papers),
        )
        return result

    async def _search_academic_web(self, queries: list[SearchQuery]) -> list[dict]:
        """Search academic evidence through the general web search provider."""
        semaphore = asyncio.Semaphore(self.api_concurrency)

        async def search_one(sq: SearchQuery) -> list[dict]:
            clean_q = clean_search_query(sq.query)
            query = (
                f'{clean_q} paper benchmark evaluation arxiv '
                f'OR "Papers with Code" OR proceedings'
            )
            async with semaphore:
                try:
                    data = await self.academic_search.search(
                        query,
                        search_depth="basic",
                        max_results=self.papers_per_query,
                        include_answer=False,
                    )
                except Exception as exc:
                    self.logger.warning("Academic web search failed for '%s': %s", sq.query, exc)
                    return []
            results = []
            for item in data.get("results", []):
                item["_query"] = sq.query
                item["_intent"] = sq.intent
                item["_source"] = "academic_web"
                results.append(item)
            return results

        groups = await asyncio.gather(
            *(search_one(sq) for sq in queries[: self.max_academic_queries]),
        )
        return self._dedupe_dicts((item for group in groups for item in group), "url")

    async def _search_openalex(self, queries: list[SearchQuery]) -> list[dict]:
        """Search OpenAlex for high-impact published papers."""
        if not self.openalex or not queries:
            return []

        semaphore = asyncio.Semaphore(self.api_concurrency)

        async def search_one(sq: SearchQuery) -> list[dict]:
            clean_q = clean_search_query(sq.query)
            async with semaphore:
                try:
                    papers = await self.openalex.search_high_impact(
                        clean_q,
                        limit=self.papers_per_query,
                        min_citations=0,
                    )
                except Exception as exc:
                    self.logger.warning(
                        "OpenAlex search failed for '%s': %s", sq.query, exc,
                    )
                    return []
            for paper in papers:
                paper["_query"] = sq.query
                paper["_intent"] = sq.intent
                paper["_source"] = "openalex"
            return papers

        groups = await asyncio.gather(
            *(search_one(sq) for sq in queries[: self.max_openalex_queries]),
        )
        return self._dedupe_dicts(
            (item for group in groups for item in group), "paper_id",
        )

    async def _search_arxiv(self, queries: list[SearchQuery]) -> list[dict]:
        """Search arXiv for papers."""
        semaphore = asyncio.Semaphore(self.api_concurrency)

        async def search_one(sq: SearchQuery) -> list[dict]:
            clean_q = clean_search_query(sq.query)
            # arXiv requires ASCII; skip queries that are primarily non-Latin
            ascii_only = clean_q.encode("ascii", errors="ignore").decode("ascii").strip()
            if len(ascii_only) < 5:
                self.logger.debug("Skipping arXiv query (non-Latin): %s", sq.query[:60])
                return []
            async with semaphore:
                try:
                    papers = await self.arxiv.search(
                        ascii_only,
                        max_results=self.papers_per_query,
                    )
                except Exception as exc:
                    self.logger.warning("arXiv search failed for '%s': %s", sq.query, exc)
                    return []
            for paper in papers:
                paper["_query"] = sq.query
                paper["_intent"] = sq.intent
                paper["_source"] = "arxiv"
            return papers

        groups = await asyncio.gather(
            *(search_one(sq) for sq in queries[: self.max_arxiv_queries]),
        )
        return self._dedupe_dicts((item for group in groups for item in group), "arxiv_id")

    def _merge_papers(
        self, web_papers: list[dict], arxiv_papers: list[dict],
        openalex_papers: list[dict] | None = None,
    ) -> list[dict]:
        """Merge, deduplicate, and sort papers. OpenAlex papers come first (highest quality)."""
        merged: list[dict] = []
        seen: set[str] = set()
        # OpenAlex first (highest quality), then arXiv, then web
        for item in [*(openalex_papers or []), *arxiv_papers, *web_papers]:
            key = (
                item.get("arxiv_id")
                or self._extract_arxiv_id(item.get("url", ""))
                or item.get("title", "").lower().strip()
            )
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
        return self._filter_relevant(merged)[: max(self.max_paper_analyses, 1)]

    def _filter_relevant(self, papers: list[dict]) -> list[dict]:
        """Remove papers whose titles show no topical overlap with the path.

        Extracts search terms from path technologies, key questions, AND the
        original search queries (which are in English).  This avoids the
        Chinese-vs-English mismatch that previously discarded all papers.
        """
        if not self._current_path:
            return papers

        path_terms: set[str] = set()
        # English tech terms (e.g. "transformer", "FastSpeech") — best match
        for tech in self._current_path.technologies_needed:
            for word in re.split(r"[\s/]+", tech.lower()):
                clean = word.strip(",.()[]{}:;")
                if len(clean) > 3 and clean not in _STOP_WORDS:
                    path_terms.add(clean)
        # Key questions may contain English keywords
        for q in self._current_path.key_questions:
            for word in re.findall(r"[a-zA-Z0-9_\-]{4,}", q):
                path_terms.add(word.lower())
        # Also break Chinese title into bigrams as fallback — this won't
        # match English papers, so we skip it.  Instead, look for any
        # English sub-strings in the path title/description.
        for text in [self._current_path.title, self._current_path.description]:
            for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-\.]{3,}", text):
                path_terms.add(word.lower())

        if not path_terms:
            return papers

        kept: list[dict] = []
        for item in papers:
            title = (item.get("title") or "").lower()
            summary = (item.get("summary") or item.get("content", "") or "").lower()
            combined = f"{title} {summary}"
            if any(term in combined for term in path_terms):
                kept.append(item)

        kept_count = len(kept)
        total = len(papers)
        if kept_count < total:
            self.logger.info(
                "Relevance filter: kept %d/%d papers (terms=%s)",
                kept_count, total,
                ", ".join(sorted(path_terms)[:10]),
            )
        return kept if kept else papers  # Fall back to all papers if nothing passes

    def _build_paper_records(
        self,
        papers: list[dict],
    ) -> tuple[list[PaperAnalysis], list[SourceReference]]:
        """Convert raw search results into paper records without per-paper LLM calls."""
        analyzed: list[PaperAnalysis] = []
        sources: list[SourceReference] = []
        for paper_data in papers[: self.max_paper_analyses]:
            source = paper_data.get("_source", "")
            if source == "openalex":
                paper = self._build_from_openalex(paper_data)
            elif source == "arxiv":
                paper = self._build_from_arxiv(paper_data)
            else:
                paper = self._build_from_web_result(paper_data)
            if not paper.title and not paper.url:
                continue
            analyzed.append(paper)
            sources.append(SourceReference(
                url=paper.url,
                title=paper.title,
                source_type=SourceType.PAPER,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                relevance_score=float(paper_data.get("score", 0.5) or 0.5),
            ))
        return analyzed, sources

    async def _synthesize_academic_context(
        self,
        papers: list[PaperAnalysis],
        path: ResearchPath,
    ) -> tuple[str, str, list[str]]:
        """Use one bounded LLM call to synthesize academic trends for a path."""
        if not papers:
            return "", "", []

        paper_summary = "\n".join(
            f"- {p.title} ({p.year or 'n.d.'}, {p.venue or 'web'}): "
            f"{(p.principles or p.methods or p.conclusions)[:500]}"
            for p in papers[: self.max_paper_analyses]
        )
        try:
            result = await self._call_llm_json_async(
                prompt=(
                    f"Summarize the academic evidence for research path '{path.title}'.\n\n"
                    f"Key questions: {'; '.join(path.key_questions[:3])}\n\n"
                    f"Papers and evidence:\n{paper_summary}\n\n"
                    "Return JSON with keys: research_trends, frontier_directions, "
                    "key_researchers. Keep it concise and evidence-grounded."
                ),
                temperature=0.2,
            )
        except Exception as exc:
            self.logger.warning("Academic synthesis failed for '%s': %s", path.title, exc)
            return "", "", []

        if not result or not isinstance(result, dict):
            return "", "", []
        researchers = result.get("key_researchers", [])
        if not isinstance(researchers, list):
            researchers = []
        return (
            str(result.get("research_trends", "")),
            str(result.get("frontier_directions", "")),
            [str(name) for name in researchers],
        )

    def _build_from_openalex(self, data: dict) -> PaperAnalysis:
        """Build PaperAnalysis from OpenAlex data (high-quality, rich metadata)."""
        venue = data.get("venue", "")
        venue_type = data.get("venue_type", "")
        venue_prestige = self._classify_venue(venue)

        return PaperAnalysis(
            paper_id=data.get("paper_id", ""),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            year=data.get("year", 0),
            venue=venue,
            citation_count=data.get("cited_by_count", 0),
            doi=data.get("doi", ""),
            url=data.get("url", ""),
            principles=data.get("abstract", "")[:800],
            conclusions=data.get("abstract", "")[:800],
            venue_prestige=venue_prestige,
        )

    def _build_from_arxiv(self, data: dict) -> PaperAnalysis:
        """Build PaperAnalysis from arXiv data."""
        year = self._extract_year(data.get("published", ""))
        summary = data.get("summary", "")
        arxiv_id = data.get("arxiv_id", "")
        abs_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else data.get("url", "")
        return PaperAnalysis(
            paper_id=arxiv_id,
            title=data.get("title", ""),
            authors=data.get("authors", []),
            year=year,
            venue="arXiv",
            arxiv_id=arxiv_id,
            url=abs_url,
            principles=summary[:800],
            conclusions=summary[:800],
            venue_prestige="preprint",
        )

    def _build_from_web_result(self, data: dict) -> PaperAnalysis:
        """Build PaperAnalysis from academic web search result metadata."""
        url = data.get("url", "")
        arxiv_id = self._extract_arxiv_id(url)
        content = data.get("content", "") or data.get("raw_content", "")
        return PaperAnalysis(
            paper_id=arxiv_id or url,
            title=data.get("title", ""),
            authors=[],
            year=self._extract_year(" ".join([data.get("title", ""), content])),
            venue=self._guess_venue(url),
            arxiv_id=arxiv_id,
            url=url,
            principles=content[:800],
            conclusions=content[:800],
            venue_prestige="preprint" if arxiv_id else "",
        )

    @classmethod
    def _classify_venue(cls, venue: str) -> str:
        """Classify venue prestige level from name."""
        if not venue:
            return ""
        lowered = venue.lower()
        for key, label in _VENUE_PRESTIGE_MAP.items():
            if key in lowered:
                return label
        return "学术来源"

    @staticmethod
    def _extract_year(text: str) -> int:
        match = re.search(r"\b(20[0-3][0-9]|19[8-9][0-9])\b", text or "")
        return int(match.group(1)) if match else 0

    @staticmethod
    def _extract_arxiv_id(url: str) -> str:
        match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#\s]+)", url or "")
        if not match:
            return ""
        return match.group(1).removesuffix(".pdf")

    @staticmethod
    def _guess_venue(url: str) -> str:
        lowered = (url or "").lower()
        if "arxiv.org" in lowered:
            return "arXiv"
        if "paperswithcode.com" in lowered:
            return "Papers with Code"
        if "aclanthology.org" in lowered:
            return "ACL Anthology"
        if "openreview.net" in lowered:
            return "OpenReview"
        return "Web"

    @staticmethod
    def _dedupe_dicts(items: Iterable[dict], key_name: str) -> list[dict]:
        deduped: list[dict] = []
        seen: set[str] = set()
        for item in items:
            key = item.get(key_name) or item.get("url") or item.get("title", "").lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped
