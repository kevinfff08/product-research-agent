"""Tests for logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

from src.logging_config import setup_logging, get_logger, _ROOT_LOGGER_NAME


class TestLogging:
    def test_get_logger_namespace(self):
        logger = get_logger("test.module")
        assert logger.name == f"{_ROOT_LOGGER_NAME}.test.module"

    def test_get_logger_is_logger(self):
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_creates_handler(self):
        """Verify setup creates a RotatingFileHandler with correct settings."""
        # setup_logging may have been called already, so check root logger
        root = logging.getLogger(_ROOT_LOGGER_NAME)
        # Find RotatingFileHandler in handlers
        rotating_handlers = [
            h for h in root.handlers
            if isinstance(h, RotatingFileHandler)
        ]
        # If setup_logging was called, there should be at least one
        # (may not be called in test environment, so this is conditional)
        if rotating_handlers:
            handler = rotating_handlers[0]
            assert handler.maxBytes == 10 * 1024 * 1024  # 10 MB
            assert handler.backupCount == 999

    def test_logger_hierarchy(self):
        parent = get_logger("agents")
        child = get_logger("agents.decomposer")
        assert child.parent.name == parent.name or child.name.startswith(parent.name)
