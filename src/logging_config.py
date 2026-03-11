"""Centralized logging for ProductResearch with RotatingFileHandler.

Logs rotate at 10 MB per file. Old logs are NEVER deleted (backupCount=999).
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_ROOT_LOGGER_NAME = "productresearch"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-40s | %(message)s"

_INITIALIZED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure project-wide logging with rotating file handler.

    Call once at application entry point (cli.py).
    Subsequent calls are no-ops.

    Log files rotate at 10 MB. Up to 999 backup files are kept
    (effectively unlimited -- old logs are NEVER destroyed).
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # RotatingFileHandler: 10 MB per file, keep 999 backups (never destroy)
    file_handler = RotatingFileHandler(
        _LOG_DIR / "product_research.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=999,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler - only WARNING+ to keep CLI output clean
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    root.info("=" * 72)
    root.info("ProductResearch logging started - log dir: %s", _LOG_DIR)
    root.info("=" * 72)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the ``productresearch`` namespace.

    Usage::

        from src.logging_config import get_logger
        logger = get_logger("agents.decomposer")
        logger.info("decomposing idea ...")
    """
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
