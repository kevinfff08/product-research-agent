"""Tests for Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.common import MaturityStage, SourceReference, SourceType, ResearchWeight
from src.models.input import ResearchRequest
from src.models.plan import ResearchPath, DecompositionResult, ResearchPlan, SearchQuery
from src.models.industry import ProductInfo, RepoAnalysis, IndustryResearchResult, CompanyInfo
from src.models.academic import PaperAnalysis, AcademicResearchResult
from src.models.engineering import CodeAnalysis, DeploymentAssessment, EngineeringAnalysis
from src.models.maturity import TechnologyMaturity, MaturityAssessment
from src.models.reputation import ReputationScore, VenuePrestige, ReputationReport
from src.models.report import TechnologyEntry, FeasibilityAssessment, ResearchReport


class TestCommonModels:
    def test_maturity_stage_values(self):
        assert MaturityStage.EARLY_PROTOTYPE == "early_prototype"
        assert MaturityStage.MATURE == "mature"
        assert MaturityStage.ACADEMIC_FRONTIER == "academic_frontier"

    def test_source_reference_defaults(self):
        ref = SourceReference()
        assert ref.url == ""
        assert ref.relevance_score == 0.5

    def test_source_reference_validation(self):
        ref = SourceReference(
            url="https://example.com",
            title="Test",
            source_type=SourceType.PAPER,
            relevance_score=0.9,
        )
        assert ref.source_type == SourceType.PAPER
        assert ref.relevance_score == 0.9

    def test_source_reference_score_bounds(self):
        with pytest.raises(ValidationError):
            SourceReference(relevance_score=1.5)
        with pytest.raises(ValidationError):
            SourceReference(relevance_score=-0.1)

    def test_research_weight_defaults(self):
        w = ResearchWeight()
        assert w.industry == 0.60
        assert w.academic == 0.25
        assert w.community == 0.15


class TestInputModels:
    def test_research_request_defaults(self):
        req = ResearchRequest(raw_input="AI code review tool")
        assert req.depth == "comprehensive"
        assert req.output_format == "markdown"
        assert req.max_paths == 5

    def test_research_request_custom(self):
        req = ResearchRequest(
            raw_input="real-time video translation",
            focus_areas=["latency", "accuracy"],
            depth="deep",
            output_format="both",
            max_paths=3,
        )
        assert len(req.focus_areas) == 2
        assert req.max_paths == 3

    def test_research_request_max_paths_validation(self):
        with pytest.raises(ValidationError):
            ResearchRequest(raw_input="test", max_paths=0)
        with pytest.raises(ValidationError):
            ResearchRequest(raw_input="test", max_paths=11)


class TestPlanModels:
    def test_research_path_creation(self):
        path = ResearchPath(
            path_id="p1",
            title="Test Path",
            description="A test research path",
            technologies_needed=["Python", "FastAPI"],
            priority=0.8,
        )
        assert path.path_id == "p1"
        assert path.priority == 0.8

    def test_decomposition_result(self, sample_decomposition):
        assert sample_decomposition.original_input == "AI-powered code review tool"
        assert len(sample_decomposition.paths) == 2
        assert len(sample_decomposition.shared_technologies) == 3

    def test_search_query(self):
        sq = SearchQuery(
            query="AI code review",
            source="tavily",
            path_id="p1",
            priority=0.9,
        )
        assert sq.source == "tavily"

    def test_research_plan(self):
        plan = ResearchPlan(
            original_input="test",
            estimated_api_calls=20,
            estimated_llm_calls=10,
        )
        assert plan.estimated_api_calls == 20


class TestIndustryModels:
    def test_product_info(self):
        prod = ProductInfo(
            name="GPT-Researcher",
            company="gptr.dev",
            description="Open source deep research agent",
            capabilities=["web search", "report generation"],
            is_open_source=True,
            maturity=MaturityStage.MATURE,
        )
        assert prod.is_open_source is True
        assert prod.maturity == MaturityStage.MATURE

    def test_repo_analysis(self):
        repo = RepoAnalysis(
            repo_url="https://github.com/test/repo",
            stars=5000,
            forks=800,
            language="Python",
        )
        assert repo.stars == 5000

    def test_industry_research_result(self):
        result = IndustryResearchResult(path_id="p1")
        assert result.path_id == "p1"
        assert len(result.products) == 0


class TestAcademicModels:
    def test_paper_analysis(self):
        paper = PaperAnalysis(
            paper_id="s2-12345",
            title="Attention Is All You Need",
            authors=["Vaswani et al."],
            year=2017,
            venue="NeurIPS",
            citation_count=100000,
            principles="Self-attention mechanism for sequence modeling",
            methods="Transformer architecture with multi-head attention",
            experiments="Machine translation benchmarks",
            conclusions="Transformers outperform RNNs on translation tasks",
            deficiencies="Quadratic complexity in sequence length",
            venue_prestige="top_conference",
        )
        assert paper.citation_count == 100000
        assert paper.venue_prestige == "top_conference"

    def test_academic_research_result(self):
        result = AcademicResearchResult(path_id="p1")
        assert len(result.papers) == 0


class TestEngineeringModels:
    def test_deployment_assessment(self):
        assess = DeploymentAssessment(
            approach="Docker + K8s",
            deployment_complexity="moderate",
            infrastructure_requirements=["Kubernetes cluster", "GPU nodes"],
        )
        assert assess.deployment_complexity == "moderate"

    def test_engineering_analysis(self):
        analysis = EngineeringAnalysis(path_id="p1")
        assert analysis.path_id == "p1"


class TestMaturityModels:
    def test_technology_maturity(self):
        tm = TechnologyMaturity(
            technology="Transformer",
            stage=MaturityStage.MATURE,
            trend="stable",
            years_in_production=7,
        )
        assert tm.stage == MaturityStage.MATURE

    def test_maturity_assessment(self):
        assess = MaturityAssessment(
            path_id="p1",
            mature_solutions=["Transformer", "BERT"],
            cutting_edge=["Mamba", "RWKV"],
        )
        assert len(assess.mature_solutions) == 2


class TestReputationModels:
    def test_reputation_score_bounds(self):
        score = ReputationScore(
            name="test",
            overall_score=0.85,
        )
        assert score.overall_score == 0.85

    def test_reputation_score_validation(self):
        with pytest.raises(ValidationError):
            ReputationScore(overall_score=1.5)


class TestReportModels:
    def test_technology_entry(self):
        entry = TechnologyEntry(
            name="FastAPI",
            category="backend",
            maturity=MaturityStage.MATURE,
            industry_score=0.9,
        )
        assert entry.industry_score == 0.9

    def test_feasibility_assessment(self):
        assess = FeasibilityAssessment(
            path_id="p1",
            path_title="Test Path",
            overall_feasibility=0.75,
            technical_risk="medium",
            recommended=True,
            rationale="Good balance of maturity and capability",
        )
        assert assess.recommended is True

    def test_research_report_defaults(self):
        report = ResearchReport()
        assert report.title == ""
        assert len(report.technology_landscape) == 0
        assert len(report.all_sources) == 0

    def test_research_report_serialization(self):
        report = ResearchReport(
            title="Test Report",
            original_input="AI tool",
            executive_summary="A test summary",
        )
        data = report.model_dump()
        assert data["title"] == "Test Report"

        # Roundtrip
        restored = ResearchReport.model_validate(data)
        assert restored.title == "Test Report"
