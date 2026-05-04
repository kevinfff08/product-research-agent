"""Centralized logging for ProductResearch.

Each research run writes to one UTF-8 log file named with date + title.
Logs do not rotate by file size.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils.naming import safe_title_for_path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_ROOT_LOGGER_NAME = "productresearch"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-40s | %(message)s"

_INITIALIZED = False
_CURRENT_LOG_PATH: Path | None = None


def setup_logging(
    level: int = logging.INFO,
    *,
    run_name: str = "product_research",
    log_dir: str | Path = "logs",
    force: bool = False,
) -> Path:
    """Configure project-wide logging and return the active log path.

    The handler is a plain ``FileHandler``. It never rotates by size; each
    research run should pass a date-plus-title ``run_name``.
    """
    global _INITIALIZED, _CURRENT_LOG_PATH

    resolved_log_dir = Path(log_dir)
    if not resolved_log_dir.is_absolute():
        resolved_log_dir = _PROJECT_ROOT / resolved_log_dir
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    log_path = resolved_log_dir / f"{safe_title_for_path(run_name, max_length=120)}.log"
    if _INITIALIZED and not force and _CURRENT_LOG_PATH == log_path:
        return log_path

    formatter = logging.Formatter(fmt=_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)
    root.propagate = False

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _INITIALIZED = True
    _CURRENT_LOG_PATH = log_path

    root.info("=" * 72)
    root.info("ProductResearch logging started - log file: %s", log_path)
    root.info("=" * 72)
    return log_path


def current_log_path() -> Path | None:
    """Return the active log file path, if logging has been configured."""
    return _CURRENT_LOG_PATH


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the ``productresearch`` namespace."""
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
