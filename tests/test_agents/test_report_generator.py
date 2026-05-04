"""Tests for ReportGenerator agent."""

from __future__ import annotations

import json

import pytest

from src.agents.report_generator import ReportGenerator
from src.models.plan import DecompositionResult, ResearchPath
from src.models.industry import IndustryResearchResult, ProductInfo
from src.models.academic import AcademicResearchResult, PaperAnalysis
from src.models.engineering import EngineeringAnalysis
from src.models.maturity import MaturityAssessment
from src.models.reputation import ReputationReport
from src.models.common import SourceReference, SourceType


@pytest.fixture
def generator(mock_llm, temp_store):
    return ReportGenerator(mock_llm, temp_store)


@pytest.fixture
def full_inputs(sample_decomposition):
    industry = [IndustryResearchResult(
        path_id="p1",
        products=[ProductInfo(name="Tool1", company="Co1", description="A tool")],
        sources=[SourceReference(url="https://example.com", title="Source 1",
                                 source_type=SourceType.WEB)],
    )]
    academic = [AcademicResearchResult(
        path_id="p1",
        papers=[PaperAnalysis(title="Paper 1", year=2024, citation_count=100)],
        sources=[SourceReference(url="https://s2.org/paper1", title="Paper 1",
                                 source_type=SourceType.PAPER)],
    )]
    engineering = [EngineeringAnalysis(
        path_id="p1",
        sources=[SourceReference(url="https://github.com/test", title="Repo",
                                 source_type=SourceType.REPO)],
    )]
    maturity = [MaturityAssessment(path_id="p1")]
    reputation = ReputationReport(summary="Good overall")

    return {
        "decomposition": sample_decomposition,
        "industry_results": industry,
        "academic_results": academic,
        "engineering_results": engineering,
        "maturity_assessments": maturity,
        "reputation_report": reputation,
    }


def test_run_success(generator, mock_llm, full_inputs):
    mock_llm.generate_json.return_value = json.dumps({
        "executive_summary": "A comprehensive analysis...",
        "research_questions": ["Should we build or reuse an AI code review stack?"],
        "methodology_summary": "Compared products, papers, repositories, and maturity signals.",
        "decision_matrix": [
            {
                "option": "Static Analysis + LLM",
                "path_id": "p1",
                "user_value": "high",
                "technical_feasibility": "high",
                "maturity": "mature",
                "ecosystem_strength": "strong",
                "cost_risk": "medium",
                "evidence_strength": "moderate",
                "verdict": "recommended",
            }
        ],
        "key_claims": [
            {
                "claim": "AST plus LLM is the lowest-risk starting point.",
                "supporting_evidence": ["Existing products and repos use this split."],
                "contradicting_evidence": ["End-to-end agents may catch broader issues."],
                "confidence": "medium",
                "implication": "Start with parser-grounded reviews.",
                "source_urls": ["https://example.com"],
            }
        ],
        "technology_landscape": [
            {
                "name": "LLM Code Review",
                "category": "ml",
                "maturity": "cutting_edge",
                "description": "AI-powered code review",
                "industry_score": 0.8,
                "academic_score": 0.7,
                "engineering_score": 0.9,
                "reputation_score": 0.85,
            }
        ],
        "workflows": [
            {
                "name": "Quick Start",
                "description": "Simple approach",
                "steps": [{"step": 1, "description": "Install", "technologies": ["Python"]}],
                "pros": ["Simple"],
                "cons": ["Limited"],
            }
        ],
        "feasibility_assessments": [
            {
                "path_id": "p1",
                "path_title": "Static Analysis + LLM",
                "overall_feasibility": 0.8,
                "technical_risk": "medium",
                "recommended": True,
                "rationale": "Feasible with current tech",
                "key_challenges": ["Integration complexity"],
            }
        ],
        "evidence_gaps": ["No latency benchmark was available."],
        "assumptions": ["The product targets Python first."],
        "confidence_assessment": "Medium confidence due to limited benchmark data.",
        "recommended_strategy": "Build a narrow parser-grounded MVP first.",
        "recommendations": "Start with the quick approach.",
    })

    report = generator.run(**full_inputs, session_id="test-session")

    assert report.session_id == "test-session"
    assert report.executive_summary
    assert len(report.technology_landscape) == 1
    assert report.technology_landscape[0].name == "LLM Code Review"
    assert len(report.workflows) == 1
    assert len(report.feasibility_assessments) == 1
    assert report.feasibility_assessments[0].recommended is True
    assert report.research_questions
    assert len(report.decision_matrix) == 1
    assert len(report.key_claims) == 1
    assert report.evidence_gaps == ["No latency benchmark was available."]
    assert len(report.all_sources) == 3  # Deduplicated


def test_run_invalid_llm(generator, mock_llm, full_inputs):
    mock_llm.generate_json.return_value = "invalid"

    report = generator.run(**full_inputs)

    # Should still have raw findings populated
    assert len(report.industry_findings) == 1
    assert len(report.academic_findings) == 1
    assert report.executive_summary == ""


def test_collect_sources_dedup(generator):
    from src.models.industry import IndustryResearchResult
    from src.models.academic import AcademicResearchResult
    from src.models.engineering import EngineeringAnalysis

    same_url = "https://example.com"
    ind = [IndustryResearchResult(
        path_id="p1",
        sources=[SourceReference(url=same_url, title="S1", source_type=SourceType.WEB)],
    )]
    acad = [AcademicResearchResult(
        path_id="p1",
        sources=[SourceReference(url=same_url, title="S1", source_type=SourceType.WEB)],
    )]
    eng = [EngineeringAnalysis(path_id="p1", sources=[])]

    sources = generator._collect_all_sources(ind, acad, eng)
    assert len(sources) == 1
