"""Custom exceptions for trading bot.

This module defines a hierarchy of custom exceptions for the trading bot,
making error handling more specific and maintainable.
"""


class TradingBotError(Exception):
    """Base exception for all trading bot errors.

    All custom exceptions in the trading bot should inherit from this class.
    """
    pass


class InsufficientDataError(TradingBotError):
    """Raised when insufficient data for strategy execution.

    Examples:
        - Not enough rows for training
        - Missing required historical data for indicators
        - Dataset too small for time series split
    """
    pass


class ModelPersistenceError(TradingBotError):
    """Raised when model save/load operations fail.

    Examples:
        - Failed to save model to disk
        - Failed to load model from disk
        - Corrupted model file
        - Version mismatch
    """
    pass


class RiskViolationError(TradingBotError):
    """Raised when trade violates risk management rules.

    Examples:
        - Position size exceeds max_position_size
        - Insufficient balance for trade
        - Risk limits breached
    """
    pass


class DataValidationError(TradingBotError):
    """Raised when data validation fails.

    Examples:
        - Missing required columns
        - Invalid data types
        - Non-positive prices
        - Empty dataset
    """
    pass


class ConfigurationError(TradingBotError):
    """Raised when configuration is invalid.

    Examples:
        - Invalid strategy name
        - Unsupported data source
        - Missing required config fields
        - Invalid parameter values
    """
    pass


class DataFetchError(TradingBotError):
    """Raised when data fetching fails.

    Examples:
        - API rate limit exceeded
        - Network timeout
        - Invalid ticker symbol
        - API authentication failed
    """
    pass
