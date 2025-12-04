"""Feature engineering utilities for trading strategies.

This module provides reusable functions for creating common technical features
used across multiple trading strategies.
"""

import pandas as pd


def add_lag_features(data: pd.DataFrame, lags: list[int], price_col: str = "Close") -> pd.DataFrame:
    """Add lag-based features to the dataset.

    Creates four types of lag features for each lag period:
    - Price lag: Historical price at lag periods ago
    - Return lag: Percentage return over lag periods
    - Volatility: Rolling standard deviation of returns
    - Drawdown: Percentage decline from rolling maximum

    Args:
        data: DataFrame with OHLCV data
        lags: List of lag periods to compute (e.g., [5, 10, 20])
        price_col: Column name to use for price (default: "Close")

    Returns:
        DataFrame with added lag features (modifies in-place and returns)

    Example:
        >>> data = pd.DataFrame({'Close': [100, 101, 102, 103, 104]})
        >>> data = add_lag_features(data, lags=[2, 3])
        >>> 'Close_lag_2' in data.columns
        True
    """
    for lag in lags:
        data[f"{price_col}_lag_{lag}"] = data[price_col].shift(lag)
        data[f"Return_lag_{lag}"] = data[price_col].pct_change(lag)
        data[f"Volatility_{lag}"] = data[price_col].pct_change().rolling(lag).std()
        rolling_max = data[price_col].rolling(lag).max()
        data[f"Drawdown_{lag}"] = (data[price_col] / rolling_max) - 1

    return data


def create_forward_return_target(
    data: pd.DataFrame, horizon: int, price_col: str = "Close", target_col: str = "target"
) -> pd.DataFrame:
    """Create forward-looking return target variable.

    Computes future return over specified horizon using percentage change.
    Uses negative shift to look forward in time.

    Args:
        data: DataFrame with price data
        horizon: Number of periods to look ahead (e.g., 5 for 5-day return)
        price_col: Column name to use for price (default: "Close")
        target_col: Name for the target column (default: "target")

    Returns:
        DataFrame with added target column (modifies in-place and returns)

    Example:
        >>> data = pd.DataFrame({'Close': [100, 105, 110, 115, 120]})
        >>> data = create_forward_return_target(data, horizon=2)
        >>> 'target' in data.columns
        True
    """
    data[target_col] = data[price_col].pct_change(horizon).shift(-horizon)
    return data


def build_lag_feature_columns(lags: list[int], price_col: str = "Close") -> list[str]:
    """Build list of lag-based feature column names.

    Generates column names for all four lag feature types (price, return,
    volatility, drawdown) for each lag period.

    Args:
        lags: List of lag periods (e.g., [5, 10, 20])
        price_col: Price column name used in features (default: "Close")

    Returns:
        List of feature column names in consistent order

    Example:
        >>> cols = build_lag_feature_columns([5, 10])
        >>> cols
        ['Close_lag_5', 'Return_lag_5', 'Volatility_5', 'Drawdown_5',
         'Close_lag_10', 'Return_lag_10', 'Volatility_10', 'Drawdown_10']
    """
    feature_columns = []
    for lag in lags:
        feature_columns.append(f"{price_col}_lag_{lag}")
        feature_columns.append(f"Return_lag_{lag}")
        feature_columns.append(f"Volatility_{lag}")
        feature_columns.append(f"Drawdown_{lag}")
    return feature_columns


def add_indicator_columns(
    feature_columns: list[str], data: pd.DataFrame, indicator_names: list[str]
) -> list[str]:
    """Add indicator columns to feature list if they exist in data.

    Filters indicator names to only include those present in the DataFrame.

    Args:
        feature_columns: Existing feature column list
        data: DataFrame to check for indicator presence
        indicator_names: List of indicator column names to potentially add

    Returns:
        Updated feature column list with available indicators appended

    Example:
        >>> data = pd.DataFrame({'Close': [100], 'MACD': [0.5]})
        >>> cols = add_indicator_columns([], data, ['MACD', 'RSI'])
        >>> cols
        ['MACD']
    """
    result = feature_columns.copy()
    for col in indicator_names:
        if col in data.columns:
            result.append(col)
    return result
