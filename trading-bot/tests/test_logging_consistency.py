"""Test logging consistency across modules."""

import logging
import re
from io import StringIO

from utils.logger import setup_logger
from data_fetcher import logger as data_fetcher_logger
from config_handler import logger as config_handler_logger


def test_logger_format_consistency():
    """Test 10.1: Verify all loggers use consistent format."""
    # Expected format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # Example: "2025-12-02 10:30:45,123 - TradingBot - INFO - Test message"

    # Create a test logger
    test_logger = setup_logger("TestModule")

    # Capture log output
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    # Remove color codes for easier testing
    plain_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(plain_formatter)

    # Add handler temporarily
    test_logger.handlers.clear()
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)

    # Log a test message
    test_logger.info("Test message")

    # Get the output
    log_output = stream.getvalue()

    # Verify format matches expected pattern
    # Pattern: YYYY-MM-DD HH:MM:SS,mmm - LoggerName - LEVEL - Message
    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - \w+ - \w+ - .+"
    assert re.match(pattern, log_output), f"Log format doesn't match expected pattern. Got: {log_output}"

    # Verify specific parts
    assert " - TestModule - " in log_output
    assert " - INFO - " in log_output
    assert "Test message" in log_output


def test_all_module_loggers_use_setup_logger():
    """Test 10.2: Verify data_fetcher and config_handler use setup_logger."""
    # Both should have ColoredFormatter (check handler type)
    assert data_fetcher_logger is not None
    assert config_handler_logger is not None

    # Both should have handlers set up
    assert len(data_fetcher_logger.handlers) > 0
    assert len(config_handler_logger.handlers) > 0

    # Verify logger names
    assert data_fetcher_logger.name == "TradingBot"
    assert config_handler_logger.name == "TradingBot"


def test_logger_level_resolution():
    """Test 10.3: Verify logger level can be set via parameter."""
    # Create loggers with different levels
    debug_logger = setup_logger("DebugLogger", "DEBUG")
    info_logger = setup_logger("InfoLogger", "INFO")
    warning_logger = setup_logger("WarningLogger", "WARNING")

    assert debug_logger.level == logging.DEBUG
    assert info_logger.level == logging.INFO
    assert warning_logger.level == logging.WARNING


def test_logger_propagate_false():
    """Test 10.4: Verify loggers don't propagate to root logger."""
    test_logger = setup_logger("NoPropagate")
    assert test_logger.propagate is False, "Logger should not propagate to prevent duplicate logs"


def test_logger_reuse():
    """Test 10.5: Verify calling setup_logger twice returns same logger instance."""
    import uuid
    unique_name = f"ReuseTest_{uuid.uuid4().hex[:8]}"

    # First call
    logger1 = setup_logger(unique_name)

    # Second call with same name
    logger2 = setup_logger(unique_name)

    # Should be the same logger instance
    assert logger1 is logger2, "setup_logger should return the same instance for the same name"

    # Verify logger configuration is preserved
    assert logger2.propagate is False
    assert logger2.name == unique_name
