"""Test logging consistency across modules."""

import logging
import re
from io import StringIO

from trading_bot.config_handler import logger as config_handler_logger
from trading_bot.data_fetcher import logger as data_fetcher_logger
from trading_bot.utils.logger import ColoredFormatter, _color_enabled, get_logger


def test_logger_format_consistency():
    """Test 10.1: Verify all loggers use consistent format."""
    test_logger = get_logger("tests.logging_consistency")

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    plain_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(plain_formatter)

    test_logger.addHandler(handler)
    original_propagate = test_logger.propagate
    test_logger.propagate = False

    try:
        test_logger.info("Test message")
    finally:
        test_logger.removeHandler(handler)
        test_logger.propagate = original_propagate

    log_output = stream.getvalue()

    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - [\w\.]+ - \w+ - .+"
    assert re.match(pattern, log_output), f"Log format doesn't match expected pattern. Got: {log_output}"
    assert " - trading_bot.tests.logging_consistency - " in log_output
    assert " - INFO - " in log_output
    assert "Test message" in log_output


def test_all_module_loggers_use_shared_root():
    """Test 10.2: Verify module loggers share the root trading_bot logger."""
    root_logger = get_logger()

    assert data_fetcher_logger.propagate is True
    assert config_handler_logger.propagate is True
    assert data_fetcher_logger.name.startswith("trading_bot.")
    assert config_handler_logger.name.startswith("trading_bot.")
    assert root_logger.handlers
    assert all(h.formatter is not None for h in root_logger.handlers)
    for handler in root_logger.handlers:
        if _color_enabled(handler.stream):
            assert isinstance(handler.formatter, ColoredFormatter)


def test_logger_level_resolution():
    """Test 10.3: Verify logger level can be set via parameter."""
    debug_logger = get_logger(None, "DEBUG")
    assert debug_logger.getEffectiveLevel() == logging.DEBUG
    info_logger = get_logger(None, "INFO")
    assert info_logger.getEffectiveLevel() == logging.INFO


def test_logger_propagation_and_handlers():
    """Test 10.4: Children should propagate and keep handlers on the root logger."""
    child_logger = get_logger("propagation.check")
    assert child_logger.propagate is True
    assert child_logger.handlers == []
    root_logger = get_logger()
    assert root_logger.propagate is False


def test_logger_reuse():
    """Test 10.5: Verify calling get_logger twice returns same logger instance."""
    import uuid

    unique_name = f"ReuseTest_{uuid.uuid4().hex[:8]}"

    logger1 = get_logger(unique_name)
    logger2 = get_logger(unique_name)

    assert logger1 is logger2, "get_logger should return the same instance for the same name"
    assert logger2.propagate is True
    assert logger2.name.startswith("trading_bot.")
