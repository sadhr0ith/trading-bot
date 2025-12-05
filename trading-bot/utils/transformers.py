"""
Feature Engineering Transformers for Trading Bot.
All transformers follow scikit-learn API (BaseEstimator + TransformerMixin).
"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Creates lagged features from specified columns."""

    def __init__(self, columns: list[str] | None = None, lags: list[int] | None = None):
        """
        Args:
            columns: List of column names to create lags for. If None, defaults to ['Close'].
            lags: List of lag periods. If None, defaults to [1, 3, 5, 10].
        """
        self.columns = columns or ["Close"]
        self.lags = lags or [1, 3, 5, 10]

    def fit(self, X, y=None):
        """Fit does nothing, returns self for pipeline compatibility."""
        return self

    def transform(self, X):
        """Create lagged features."""
        X_copy = X.copy()

        for col in self.columns:
            if col not in X_copy.columns:
                continue
            for lag in self.lags:
                X_copy[f"{col}_lag_{lag}"] = X_copy[col].shift(lag)

        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output."""
        feature_names = list(input_features) if input_features is not None else []
        for col in self.columns:
            for lag in self.lags:
                feature_names.append(f"{col}_lag_{lag}")
        return np.array(feature_names)


class RollingStatsTransformer(BaseEstimator, TransformerMixin):
    """Creates rolling window statistics (mean, std, etc.)."""

    def __init__(
        self,
        price_col: str = "Close",
        volume_col: str | None = "Volume",
        windows: list[int] | None = None,
    ):
        """
        Args:
            price_col: Column name for price data.
            volume_col: Column name for volume data. Set to None to skip volume features.
            windows: List of rolling window sizes. If None, defaults to [5, 10].
        """
        self.price_col = price_col
        self.volume_col = volume_col
        self.windows = windows or [5, 10]

    def fit(self, X, y=None):
        """Fit does nothing, returns self for pipeline compatibility."""
        return self

    def transform(self, X):
        """Create rolling statistics features."""
        X_copy = X.copy()

        for window in self.windows:
            # SMA
            if self.price_col in X_copy.columns:
                X_copy[f"SMA_{window}"] = X_copy[self.price_col].rolling(window).mean()

            # Volatility (rolling std of returns)
            if self.price_col in X_copy.columns:
                returns = X_copy[self.price_col].pct_change()
                X_copy[f"Volatility_{window}"] = returns.rolling(window).std()

            # Volume MA
            if self.volume_col and self.volume_col in X_copy.columns:
                X_copy[f"Volume_MA_{window}"] = X_copy[self.volume_col].rolling(window).mean()

        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output."""
        feature_names = list(input_features) if input_features is not None else []
        for window in self.windows:
            feature_names.append(f"SMA_{window}")
            feature_names.append(f"Volatility_{window}")
            if self.volume_col:
                feature_names.append(f"Volume_MA_{window}")
        return np.array(feature_names)


class ReturnFeatureTransformer(BaseEstimator, TransformerMixin):
    """Creates return features (pct_change over different periods)."""

    def __init__(self, price_col: str = "Close", periods: list[int] | None = None):
        """
        Args:
            price_col: Column name for price data.
            periods: List of periods for calculating returns. If None, defaults to [1, 3, 5, 10].
        """
        self.price_col = price_col
        self.periods = periods or [1, 3, 5, 10]

    def fit(self, X, y=None):
        """Fit does nothing, returns self for pipeline compatibility."""
        return self

    def transform(self, X):
        """Create return features."""
        X_copy = X.copy()

        if self.price_col in X_copy.columns:
            for period in self.periods:
                X_copy[f"Return_lag_{period}"] = X_copy[self.price_col].pct_change(period)

        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output."""
        feature_names = list(input_features) if input_features is not None else []
        for period in self.periods:
            feature_names.append(f"Return_lag_{period}")
        return np.array(feature_names)


class CalendarFeatureTransformer(BaseEstimator, TransformerMixin):
    """Adds simple calendar-based features from the DatetimeIndex."""

    def __init__(self):
        self.feature_names_ = [
            "day_of_week",
            "month",
            "is_month_start",
            "is_month_end",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        idx = X_copy.index
        if not isinstance(idx, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(idx)
            except Exception:  # noqa: BLE001
                return X_copy

        X_copy["day_of_week"] = idx.dayofweek
        X_copy["month"] = idx.month
        X_copy["is_month_start"] = idx.is_month_start.astype(int)
        X_copy["is_month_end"] = idx.is_month_end.astype(int)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        feature_names = list(input_features) if input_features is not None else []
        feature_names.extend(self.feature_names_)
        return np.array(feature_names)


class IndicatorLagTransformer(BaseEstimator, TransformerMixin):
    """Creates lagged versions of technical indicators (MACD, RSI, etc.)."""

    def __init__(
        self,
        indicator_columns: list[str] | None = None,
        lags: list[int] | None = None,
        column_mapping: dict[str, str] | None = None,
    ):
        """
        Args:
            indicator_columns: List of indicator column names.
                If None, defaults to ['MACD', 'Signal', 'RSI'].
            lags: List of lag periods. If None, defaults to [1].
            column_mapping: Optional mapping to normalize input column names
                to canonical names before lagging.
        """
        self.indicator_columns = indicator_columns or ["MACD", "Signal", "RSI"]
        self.lags = lags or [1]
        self.column_mapping = column_mapping or {}

    def fit(self, X, y=None):
        """Fit does nothing, returns self for pipeline compatibility."""
        return self

    def transform(self, X):
        """Create lagged indicator features."""
        X_copy = X.copy()
        # Rename columns to canonical names if provided
        if self.column_mapping:
            X_copy = X_copy.rename(columns=self.column_mapping)

        for col in self.indicator_columns:
            if col not in X_copy.columns:
                continue
            for lag in self.lags:
                X_copy[f"{col}_lag_{lag}"] = X_copy[col].shift(lag)

        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output."""
        feature_names = list(input_features) if input_features is not None else []
        for col in self.indicator_columns:
            for lag in self.lags:
                feature_names.append(f"{col}_lag_{lag}")
        return np.array(feature_names)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects specific columns and handles missing values."""

    def __init__(self, feature_columns: list[str], handle_missing: str = "drop"):
        """
        Args:
            feature_columns: List of feature column names to select.
            handle_missing: How to handle missing values: 'drop', 'ffill', or 'median'.
        """
        self.feature_columns = feature_columns
        self.handle_missing = handle_missing
        self.medians_ = {}

    def fit(self, X, y=None):
        """Fit: compute medians for 'median' strategy."""
        if self.handle_missing == "median":
            available = [col for col in self.feature_columns if col in X.columns]
            self.medians_ = X[available].median().to_dict()
        return self

    def transform(self, X):
        """Select features and handle missing values."""
        available_features = [col for col in self.feature_columns if col in X.columns]
        X_selected = X[available_features].copy()

        if self.handle_missing == "ffill":
            X_selected = X_selected.ffill().bfill()
        elif self.handle_missing == "median":
            X_selected = X_selected.fillna(self.medians_)
        # 'drop' is handled later in the pipeline or by the caller

        return X_selected

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output."""
        return np.array(self.feature_columns)
