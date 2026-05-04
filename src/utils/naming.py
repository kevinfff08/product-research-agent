"""Helpers for stable run, log, and output names."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime

_INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_SEPARATORS = re.compile(r"[\s._-]+")


def safe_title_for_path(title: str, *, max_length: int = 80) -> str:
    """Return a filesystem-safe title while preserving readable Unicode."""
    normalized = unicodedata.normalize("NFKC", title).strip()
    cleaned = _INVALID_PATH_CHARS.sub(" ", normalized)
    cleaned = _SEPARATORS.sub("_", cleaned).strip("_")
    if not cleaned:
        return "research"
    return cleaned[:max_length].strip("_") or "research"


def build_run_name(title: str, *, now: datetime | None = None) -> str:
    """Build the canonical date-plus-title run name used by logs and reports."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{safe_title_for_path(title)}"
