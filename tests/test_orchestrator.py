"""Tests for Orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.orchestrator import Orchestrator
from src.models.input import ResearchRequest


@pytest.fixture
def mock_config(tmp_path):
    """Create a minimal config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "research:\n  max_paths: 3\n"
        "llm:\n  default_max_tokens: 4096\n"
    )
    return config_path


@pytest.fixture
def orchestrator(tmp_path, mock_config):
    """Create an Orchestrator with mocked LLM and API clients."""
    with patch.dict("os.environ", {
        "LLM_MODE": "api-key",
        "OPENAI_API_KEY": "sk-test",
        "TAVILY_API_KEY": "test",
    }):
        orch = Orchestrator(
            config_path=mock_config,
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
        )

    # Mock the LLM client
    mock_llm = MagicMock()
    mock_llm.generate.return_value = ""
    mock_llm.generate_json.return_value = "{}"
    mock_llm.render_template.return_value = "rendered prompt"
    mock_llm.model = "test-model"
    orch.llm = mock_llm

    # Inject mocked LLM into all agents
    for agent in [
        orch.decomposer, orch.planner,
        orch.industry_researcher, orch.academic_researcher,
        orch.engineering_analyst, orch.reputation_scorer,
        orch.maturity_mapper, orch.report_generator,
    ]:
        agent.llm = mock_llm

    # Mock API clients
    orch.tavily = AsyncMock()
    orch.tavily.search.return_value = {"results": []}
    orch.tavily.close = AsyncMock()

    orch.s2 = AsyncMock()
    orch.s2.search_papers.return_value = []
    orch.s2.close = AsyncMock()

    orch.github = AsyncMock()
    orch.github.search_repos.return_value = []
    orch.github.get_readme.return_value = ""
    orch.github.close = AsyncMock()

    orch.arxiv = AsyncMock()
    orch.arxiv.search.return_value = []
    orch.arxiv.close = AsyncMock()

    orch.scraper = AsyncMock()
    orch.scraper.close = AsyncMock()

    # Inject mocked API clients into agents
    orch.industry_researcher.tavily = orch.tavily
    orch.industry_researcher.github = orch.github
    orch.industry_researcher.scraper = orch.scraper
    orch.academic_researcher.s2 = orch.s2
    orch.academic_researcher.arxiv = orch.arxiv
    orch.engineering_analyst.github = orch.github

    return orch


@pytest.mark.asyncio
async def test_run_full_pipeline(orchestrator):
    """Test the full pipeline with mocked components."""
    # Mock decomposer LLM response
    orchestrator.llm.generate_json.side_effect = [
        # 1. Decomposer
        json.dumps({
            "interpretation": "AI code review tool",
            "paths": [{
                "path_id": "p1",
                "title": "Static Analysis",
                "description": "Use AST + LLM",
                "search_queries": {"web": ["AI code review"], "code": ["code review"]},
                "priority": 0.8,
            }],
        }),
        # 2. Planner
        json.dumps({
            "search_queries": [
                {"query": "AI code review", "source": "tavily", "path_id": "p1"},
            ],
        }),
        # Industry/Academic/Engineering: no LLM calls (empty search results)
        # 3. Reputation
        json.dumps({
            "companies": [], "labs": [], "venues": [],
            "technology_scores": [], "summary": "No data.",
        }),
        # Maturity: skipped (no technologies found, returns early)
        # 4. Report synthesis
        json.dumps({
            "executive_summary": "A test report.",
            "technology_landscape": [],
            "workflows": [],
            "feasibility_assessments": [],
            "recommendations": "Start simple.",
        }),
    ]

    request = ResearchRequest(raw_input="AI code review tool", max_paths=1)
    report = await orchestrator.run(request)

    assert report.original_input == "AI code review tool"
    assert report.session_id  # Should have a session ID
    assert report.executive_summary == "A test report."

    # Check output file was created
    output_files = list(orchestrator.output_dir.glob("*.md"))
    assert len(output_files) == 1


@pytest.mark.asyncio
async def test_run_decompose_fails(orchestrator):
    """Test handling when decomposition produces no paths."""
    orchestrator.llm.generate_json.return_value = json.dumps({
        "interpretation": "Failed",
        "paths": [],
    })

    request = ResearchRequest(raw_input="impossible idea")
    report = await orchestrator.run(request)

    assert "Failed" in report.executive_summary


def test_load_config_missing(tmp_path):
    """Config loading should handle missing file."""
    orch = Orchestrator.__new__(Orchestrator)
    config = orch._load_config(tmp_path / "nonexistent.yaml")
    assert config == {}


def test_group_queries(orchestrator):
    """Test query grouping by path and source."""
    from src.models.plan import SearchQuery, ResearchPlan, ResearchPath

    plan = ResearchPlan(
        original_input="test",
        paths=[],
        search_queries=[
            SearchQuery(query="q1", source="tavily", path_id="p1"),
            SearchQuery(query="q2", source="github", path_id="p1"),
            SearchQuery(query="q3", source="tavily", path_id="p2"),
        ],
    )

    grouped = orchestrator._group_queries(plan)

    assert "p1" in grouped
    assert "p2" in grouped
    assert len(grouped["p1"]["tavily"]) == 1
    assert len(grouped["p1"]["github"]) == 1


def test_write_outputs_both(orchestrator):
    """Output writer should honor the requested combined format."""
    pytest.importorskip("docx")
    from src.models.report import ResearchReport

    report = ResearchReport(
        title="Combined Output",
        session_id="session-123",
        executive_summary="Summary",
    )

    paths = orchestrator._write_outputs(report, "both")

    assert {path.suffix for path in paths} == {".md", ".docx"}
    assert all(path.exists() for path in paths)
    assert orchestrator.output_paths == paths
