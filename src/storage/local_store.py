"""JSON file-based local storage for research sessions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from src.logging_config import get_logger

logger = get_logger("storage")


class LocalStore:
    """Simple file-based storage for research data.

    Organizes data by research sessions under ``data/research/<session_id>/``.
    Also manages a cache directory at ``data/cache/``.
    """

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for subdir in ("research", "cache"):
            (self.data_dir / subdir).mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def create_session(self, description: str = "") -> str:
        """Create a new research session and return its ID."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        session_dir = self.data_dir / "research" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "status": "started",
        }
        self.save_json(f"research/{session_id}/meta.json", meta)
        logger.info("Created research session: %s", session_id)
        return session_id

    def save_json(self, relative_path: str, data: dict | list) -> Path:
        """Save data as JSON to a path relative to data_dir."""
        path = self.data_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug("Saved JSON: %s (%d bytes)", relative_path, path.stat().st_size)
        return path

    def load_json(self, relative_path: str) -> dict | list | None:
        """Load JSON from a path relative to data_dir."""
        path = self.data_dir / relative_path
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load JSON from %s: %s", relative_path, exc)
            return None

    def save_model(self, relative_path: str, model: BaseModel) -> Path:
        """Save a Pydantic model as JSON."""
        return self.save_json(relative_path, model.model_dump(mode="json"))

    def save_text(self, relative_path: str, text: str) -> Path:
        """Save plain text to a path relative to data_dir."""
        path = self.data_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logger.debug("Saved text: %s (%d bytes)", relative_path, path.stat().st_size)
        return path

    def list_sessions(self) -> list[dict]:
        """List all research sessions with metadata."""
        research_dir = self.data_dir / "research"
        sessions = []
        if not research_dir.exists():
            return sessions

        for session_dir in sorted(research_dir.iterdir(), reverse=True):
            if session_dir.is_dir():
                meta = self.load_json(f"research/{session_dir.name}/meta.json")
                if meta:
                    sessions.append(meta)
                else:
                    sessions.append({
                        "session_id": session_dir.name,
                        "status": "unknown",
                    })
        return sessions

    def update_session_status(self, session_id: str, status: str) -> None:
        """Update the status field in a session's metadata."""
        meta_path = f"research/{session_id}/meta.json"
        meta = self.load_json(meta_path)
        if meta and isinstance(meta, dict):
            meta["status"] = status
            meta["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.save_json(meta_path, meta)
