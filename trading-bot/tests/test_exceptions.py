"""Tests for custom exceptions."""

import pytest

from trading_bot.core.exceptions import (
    ConfigurationError,
    DataFetchError,
    DataValidationError,
    InsufficientDataError,
    ModelPersistenceError,
    RiskViolationError,
    TradingBotError,
)


def test_base_exception():
    """Test that TradingBotError can be raised and caught."""
    with pytest.raises(TradingBotError):
        raise TradingBotError("Test error")


def test_insufficient_data_error():
    """Test InsufficientDataError inheritance."""
    with pytest.raises(InsufficientDataError):
        raise InsufficientDataError("Not enough rows")

    # Should also be catchable as TradingBotError
    with pytest.raises(TradingBotError):
        raise InsufficientDataError("Not enough rows")


def test_model_persistence_error():
    """Test ModelPersistenceError inheritance."""
    with pytest.raises(ModelPersistenceError):
        raise ModelPersistenceError("Failed to save model")

    with pytest.raises(TradingBotError):
        raise ModelPersistenceError("Failed to save model")


def test_risk_violation_error():
    """Test RiskViolationError inheritance."""
    with pytest.raises(RiskViolationError):
        raise RiskViolationError("Position size too large")

    with pytest.raises(TradingBotError):
        raise RiskViolationError("Position size too large")


def test_data_validation_error():
    """Test DataValidationError inheritance."""
    with pytest.raises(DataValidationError):
        raise DataValidationError("Missing required columns")

    with pytest.raises(TradingBotError):
        raise DataValidationError("Missing required columns")


def test_configuration_error():
    """Test ConfigurationError inheritance."""
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("Invalid strategy")

    with pytest.raises(TradingBotError):
        raise ConfigurationError("Invalid strategy")


def test_data_fetch_error():
    """Test DataFetchError inheritance."""
    with pytest.raises(DataFetchError):
        raise DataFetchError("API rate limit exceeded")

    with pytest.raises(TradingBotError):
        raise DataFetchError("API rate limit exceeded")


def test_exception_messages():
    """Test that custom messages are preserved."""
    error_msg = "Custom error message"

    try:
        raise InsufficientDataError(error_msg)
    except InsufficientDataError as exc:
        assert str(exc) == error_msg

    try:
        raise ModelPersistenceError(error_msg)
    except ModelPersistenceError as exc:
        assert str(exc) == error_msg


def test_exception_chaining():
    """Test exception chaining with from clause."""
    original_error = ValueError("Original error")

    try:
        try:
            raise original_error
        except ValueError as exc:
            raise ModelPersistenceError("Failed to load model") from exc
    except ModelPersistenceError as exc:
        assert exc.__cause__ is original_error
        assert isinstance(exc.__cause__, ValueError)
