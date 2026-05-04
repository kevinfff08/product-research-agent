"""Tests for DocxReporter."""

from __future__ import annotations

import pytest

docx = pytest.importorskip("docx")

from docx import Document

from src.models.common import MaturityStage, SourceReference, SourceType
from src.models.report import ResearchReport, TechnologyEntry
from src.reporters.docx_reporter import DocxReporter


@pytest.fixture
def docx_report() -> ResearchReport:
    return ResearchReport(
        title="Tech Landscape: AI Code Review",
        session_id="test-123",
        generated_at="2026-05-04T00:00:00Z",
        original_input="AI code review tool",
        executive_summary="This report analyzes the AI code review landscape.",
        technology_landscape=[
            TechnologyEntry(
                name="tree-sitter",
                category="backend",
                maturity=MaturityStage.MATURE,
                description="Fast parser",
                industry_score=0.9,
                academic_score=0.5,
                engineering_score=0.95,
                reputation_score=0.85,
            ),
        ],
        recommendations="Start with the simple approach.",
        all_sources=[
            SourceReference(
                url="https://example.com",
                title="Source 1",
                source_type=SourceType.WEB,
            ),
        ],
    )


def test_generate_creates_docx(tmp_path, docx_report):
    output = tmp_path / "report.docx"
    result = DocxReporter().generate(docx_report, output)

    assert result.exists()
    document = Document(result)
    text = "\n".join(p.text for p in document.paragraphs)
    assert "Tech Landscape: AI Code Review" in text
    assert "Executive Summary" in text
    assert "Start with the simple approach." in text
    assert document.tables


def test_empty_report(tmp_path):
    output = tmp_path / "empty.docx"
    result = DocxReporter().generate(ResearchReport(title="Empty"), output)

    document = Document(result)
    text = "\n".join(p.text for p in document.paragraphs)
    assert "Empty" in text
