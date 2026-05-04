"""Tests for CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli import app
from src.models.report import ResearchReport


runner = CliRunner()


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # Typer with no_args_is_help exits with 0 or 2 depending on version
    assert result.exit_code in (0, 2)
    assert "Usage" in result.stdout or "product-research" in result.stdout


def test_list_sessions_empty(tmp_path):
    result = runner.invoke(app, ["list-sessions", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No research sessions" in result.stdout


def test_list_sessions_with_data(tmp_path):
    # Create a fake session
    session_dir = tmp_path / "research" / "20250101_120000_abc123"
    session_dir.mkdir(parents=True)
    meta = {
        "session_id": "20250101_120000_abc123",
        "status": "completed",
        "created_at": "2025-01-01T12:00:00Z",
        "description": "Test session",
    }
    (session_dir / "meta.json").write_text(json.dumps(meta))

    result = runner.invoke(app, ["list-sessions", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "20250101_120000_abc123" in result.stdout


def test_show_session_not_found(tmp_path):
    result = runner.invoke(app, ["show", "nonexistent", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1


def test_show_session_with_data(tmp_path):
    session_dir = tmp_path / "research" / "test-session"
    session_dir.mkdir(parents=True)
    meta = {"session_id": "test-session", "status": "completed"}
    (session_dir / "meta.json").write_text(json.dumps(meta))

    result = runner.invoke(app, ["show", "test-session", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "test-session" in result.stdout


def test_status_not_found(tmp_path):
    result = runner.invoke(app, ["status", "nonexistent", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1


def test_status_with_data(tmp_path):
    session_dir = tmp_path / "research" / "test-session"
    session_dir.mkdir(parents=True)
    meta = {"session_id": "test-session", "status": "completed"}
    (session_dir / "meta.json").write_text(json.dumps(meta))

    result = runner.invoke(app, ["status", "test-session", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "completed" in result.stdout


def test_start_guided_flow(monkeypatch, tmp_path):
    class FakeOrchestrator:
        def __init__(self, config_path, data_dir="data", output_dir="output"):
            self.output_dir = Path(output_dir)
            self.output_paths = []

        async def run(self, request):
            output = self.output_dir / f"{request.run_name}.md"
            self.output_paths = [output]
            return ResearchReport(
                title=f"Technology Landscape: {request.title}",
                session_id=request.run_name,
                original_input=request.raw_input,
                executive_summary="Done",
            )

    monkeypatch.setattr("src.cli.Orchestrator", FakeOrchestrator)
    monkeypatch.setattr("src.cli.build_run_name", lambda title: "20260504_120000_Test_Title")

    result = runner.invoke(
        app,
        [
            "start",
            "--data-dir", str(tmp_path / "data"),
            "--output-dir", str(tmp_path / "output"),
            "--log-dir", str(tmp_path / "logs"),
        ],
        input=(
            "Test Title\n"
            "Detailed paragraph one.\n"
            "Detailed paragraph two.\n"
            "END\n"
            "latency, open source\n"
            "quick\n"
            "2\n"
            "markdown\n"
        ),
    )

    assert result.exit_code == 0
    assert "标题式输入" in result.stdout
    assert "二阶段细致描述" in result.stdout
    assert "20260504_120000_Test_Title" in result.stdout
