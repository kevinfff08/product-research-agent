"""Tests for LocalStore."""

from __future__ import annotations

import json
import pytest
from pydantic import BaseModel

from src.storage.local_store import LocalStore


class SampleModel(BaseModel):
    name: str = ""
    value: int = 0


class TestLocalStore:
    def test_create_session(self, temp_store):
        session_id = temp_store.create_session("Test research")
        assert session_id is not None
        assert len(session_id) > 0

        # Meta file should exist
        meta = temp_store.load_json(f"research/{session_id}/meta.json")
        assert meta is not None
        assert meta["status"] == "started"
        assert meta["description"] == "Test research"

    def test_create_session_with_date_title_name(self, temp_store):
        session_id = temp_store.create_session(
            "Detailed context",
            title="AI Code Review",
            detailed_description="Long description",
            session_id="20260504_120000_AI_Code_Review",
        )

        assert session_id == "20260504_120000_AI_Code_Review"
        meta = temp_store.load_json(f"research/{session_id}/meta.json")
        assert meta["title"] == "AI Code Review"
        assert meta["detailed_description"] == "Long description"

    def test_save_and_load_json(self, temp_store):
        data = {"key": "value", "number": 42}
        temp_store.save_json("test/data.json", data)
        loaded = temp_store.load_json("test/data.json")
        assert loaded == data

    def test_load_nonexistent_json(self, temp_store):
        assert temp_store.load_json("nonexistent.json") is None

    def test_save_model(self, temp_store):
        model = SampleModel(name="test", value=42)
        temp_store.save_model("test/model.json", model)
        loaded = temp_store.load_json("test/model.json")
        assert loaded["name"] == "test"
        assert loaded["value"] == 42

    def test_save_text(self, temp_store):
        temp_store.save_text("test/report.md", "# Test Report\n\nContent here.")
        path = temp_store.data_dir / "test" / "report.md"
        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("# Test Report")

    def test_list_sessions(self, temp_store):
        s1 = temp_store.create_session("First")
        s2 = temp_store.create_session("Second")
        sessions = temp_store.list_sessions()
        assert len(sessions) >= 2
        session_ids = [s["session_id"] for s in sessions]
        assert s1 in session_ids
        assert s2 in session_ids

    def test_list_sessions_empty(self, temp_store):
        sessions = temp_store.list_sessions()
        assert sessions == []

    def test_update_session_status(self, temp_store):
        session_id = temp_store.create_session("Test")
        temp_store.update_session_status(session_id, "completed")
        meta = temp_store.load_json(f"research/{session_id}/meta.json")
        assert meta["status"] == "completed"
        assert "updated_at" in meta

    def test_cache_dir(self, temp_store):
        assert temp_store.cache_dir.exists()
        assert temp_store.cache_dir.name == "cache"

    def test_ensure_dirs(self, temp_dir):
        store = LocalStore(data_dir=temp_dir / "subdir")
        assert (temp_dir / "subdir" / "research").exists()
        assert (temp_dir / "subdir" / "cache").exists()
