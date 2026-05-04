"""Agent that conducts academic research using Semantic Scholar and arXiv."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.storage.local_store import LocalStore
from src.apis.semantic_scholar import SemanticScholarClient
from src.apis.arxiv_client import ArxivClient
from src.models.plan import ResearchPath, SearchQuery
from src.models.academic import PaperAnalysis, AcademicResearchResult
from src.models.common import SourceReference, SourceType


class AcademicResearcher(BaseAgent):
    """Conducts academic research: papers, citations, research trends."""

    agent_name = "academic_researcher"

    def __init__(self, llm: LLMClient, store: LocalStore, semantic_scholar: SemanticScholarClient, arxiv: ArxivClient):
        super().__init__(llm, store)
        self.s2 = semantic_scholar
        self.arxiv = arxiv

    async def run(
        self,
        *,
        path: ResearchPath,
        queries: list[SearchQuery] | None = None,
    ) -> AcademicResearchResult:
        """Run academic research for a single path."""
        self.logger.info("Academic research for path: %s", path.title)

        a_queries = queries or [
            SearchQuery(query=q, source="semantic_scholar", path_id=path.path_id)
            for q in path.search_queries.get("academic", [])
        ]

        # Search both sources in parallel
        s2_results, arxiv_results = await asyncio.gather(
            self._search_semantic_scholar(a_queries),
            self._search_arxiv(a_queries),
            return_exceptions=True,
        )

        if isinstance(s2_results, Exception):
            self.logger.error("Semantic Scholar search failed: %s", s2_results)
            s2_results = []
        if isinstance(arxiv_results, Exception):
            self.logger.error("arXiv search failed: %s", arxiv_results)
            arxiv_results = []

        # Merge and deduplicate papers
        all_papers = self._merge_papers(s2_results, arxiv_results)

        # Analyze papers with LLM
        analyzed_papers, sources = await self._analyze_papers(all_papers, path)

        result = AcademicResearchResult(
            path_id=path.path_id,
            papers=analyzed_papers,
            sources=sources,
        )
        self.logger.info(
            "Academic research complete: %d papers analyzed",
            len(analyzed_papers),
        )
        return result

    async def _search_semantic_scholar(self, queries: list[SearchQuery]) -> list[dict]:
        """Search Semantic Scholar for papers."""
        all_papers = []
        seen_ids = set()
        for sq in queries[:5]:
            try:
                papers = await self.s2.search_papers(sq.query, limit=10)
                for p in papers:
                    pid = p.get("paperId", "")
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        p["_source"] = "semantic_scholar"
                        all_papers.append(p)
            except Exception as exc:
                self.logger.warning("S2 search failed for '%s': %s", sq.query, exc)
        return all_papers

    async def _search_arxiv(self, queries: list[SearchQuery]) -> list[dict]:
        """Search arXiv for papers."""
        all_papers = []
        seen_ids = set()
        for sq in queries[:3]:
            try:
                papers = await self.arxiv.search(sq.query, max_results=10)
                for p in papers:
                    aid = p.get("arxiv_id", "")
                    if aid and aid not in seen_ids:
                        seen_ids.add(aid)
                        p["_source"] = "arxiv"
                        all_papers.append(p)
            except Exception as exc:
                self.logger.warning("arXiv search failed for '%s': %s", sq.query, exc)
        return all_papers

    def _merge_papers(self, s2_papers: list[dict], arxiv_papers: list[dict]) -> list[dict]:
        """Merge and deduplicate papers from both sources."""
        merged = list(s2_papers)
        s2_titles = {p.get("title", "").lower().strip() for p in s2_papers}

        for arxiv_paper in arxiv_papers:
            title = arxiv_paper.get("title", "").lower().strip()
            if title not in s2_titles:
                merged.append(arxiv_paper)

        # Sort by citation count (S2 papers) then by recency
        merged.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
        return merged[:20]  # Limit total papers

    async def _analyze_papers(
        self, papers: list[dict], path: ResearchPath,
    ) -> tuple[list[PaperAnalysis], list[SourceReference]]:
        """Analyze papers with LLM for PhD-level insights."""
        analyzed = []
        sources = []

        for paper_data in papers[:10]:  # Limit LLM calls
            source = paper_data.get("_source", "unknown")

            if source == "semantic_scholar":
                paper_analysis = self._build_from_s2(paper_data)
            else:
                paper_analysis = self._build_from_arxiv(paper_data)

            # LLM analysis for deeper insights
            abstract = paper_data.get("abstract") or paper_data.get("summary", "")
            if abstract:
                llm_analysis = self._call_llm_json(
                    prompt=self._render_template(
                        "analyze_paper",
                        {
                            "path_title": path.title,
                            "key_questions": "; ".join(path.key_questions[:3]),
                            "paper_title": paper_analysis.title,
                            "paper_authors": ", ".join(paper_analysis.authors[:5]),
                            "paper_year": paper_analysis.year,
                            "paper_venue": paper_analysis.venue,
                            "citation_count": paper_analysis.citation_count,
                            "paper_abstract": abstract[:3000],
                        },
                    ),
                    temperature=0.2,
                )

                if llm_analysis and isinstance(llm_analysis, dict):
                    paper_analysis.principles = llm_analysis.get("principles", "")
                    paper_analysis.methods = llm_analysis.get("methods", "")
                    paper_analysis.experiments = llm_analysis.get("experiments", "")
                    paper_analysis.conclusions = llm_analysis.get("conclusions", "")
                    paper_analysis.deficiencies = llm_analysis.get("deficiencies", "")
                    paper_analysis.venue_prestige = llm_analysis.get("venue_prestige", "")
                    paper_analysis.known_lab = llm_analysis.get("known_lab", "")

            analyzed.append(paper_analysis)
            sources.append(SourceReference(
                url=paper_analysis.url,
                title=paper_analysis.title,
                source_type=SourceType.PAPER,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
            ))

        return analyzed, sources

    def _build_from_s2(self, data: dict) -> PaperAnalysis:
        """Build PaperAnalysis from Semantic Scholar data."""
        external_ids = data.get("externalIds") or {}
        authors = [a.get("name", "") for a in (data.get("authors") or [])]
        return PaperAnalysis(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            authors=authors,
            year=data.get("year") or 0,
            venue=data.get("venue", ""),
            citation_count=data.get("citationCount") or 0,
            doi=external_ids.get("DOI", ""),
            arxiv_id=external_ids.get("ArXiv", ""),
            url=f"https://www.semanticscholar.org/paper/{data.get('paperId', '')}",
        )

    def _build_from_arxiv(self, data: dict) -> PaperAnalysis:
        """Build PaperAnalysis from arXiv data."""
        year = 0
        published = data.get("published", "")
        if published and len(published) >= 4:
            try:
                year = int(published[:4])
            except ValueError:
                pass

        return PaperAnalysis(
            paper_id=data.get("arxiv_id", ""),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            year=year,
            arxiv_id=data.get("arxiv_id", ""),
            url=data.get("pdf_url", "") or f"https://arxiv.org/abs/{data.get('arxiv_id', '')}",
        )
