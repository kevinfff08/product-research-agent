"""Tests for logging configuration."""

from __future__ import annotations

import logging
from logging import FileHandler

from src.logging_config import current_log_path, setup_logging, get_logger, _ROOT_LOGGER_NAME


class TestLogging:
    def test_get_logger_namespace(self):
        logger = get_logger("test.module")
        assert logger.name == f"{_ROOT_LOGGER_NAME}.test.module"

    def test_get_logger_is_logger(self):
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_creates_file_handler(self, tmp_path):
        """Verify setup creates a non-rotating FileHandler named by run."""
        log_path = setup_logging(
            run_name="20260504_Test_Title",
            log_dir=tmp_path,
            force=True,
        )
        root = logging.getLogger(_ROOT_LOGGER_NAME)
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, FileHandler)
        ]
        assert file_handlers
        assert log_path.name == "20260504_Test_Title.log"
        assert current_log_path() == log_path
        assert type(file_handlers[0]).__name__ == "FileHandler"

    def test_logger_hierarchy(self):
        parent = get_logger("agents")
        child = get_logger("agents.decomposer")
        assert child.parent.name == parent.name or child.name.startswith(parent.name)
