"""
Feature Engineering Transformers for Trading Bot.
All transformers follow scikit-learn API (BaseEstimator + TransformerMixin).
"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from trading_bot.utils.feature_engineering import apply_configured_indicators

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


class LagReturnDrawdownTransformer(BaseEstimator, TransformerMixin):
    """Creates lag, return, volatility, and drawdown features for specified lags."""

    def __init__(self, price_col: str = "Close", lags: list[int] | None = None):
        self.price_col = price_col
        self.lags = lags or [5, 10, 20]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.price_col not in X_copy.columns:
            return X_copy

        for lag in self.lags:
            X_copy[f"{self.price_col}_lag_{lag}"] = X_copy[self.price_col].shift(lag)
            X_copy[f"Return_lag_{lag}"] = X_copy[self.price_col].pct_change(lag)
            X_copy[f"Volatility_{lag}"] = X_copy[self.price_col].pct_change().rolling(lag).std()
            rolling_max = X_copy[self.price_col].rolling(lag).max()
            X_copy[f"Drawdown_{lag}"] = (X_copy[self.price_col] / rolling_max) - 1

        return X_copy

    def get_feature_names_out(self, input_features=None):
        feature_names = list(input_features) if input_features is not None else []
        for lag in self.lags:
            feature_names.append(f"{self.price_col}_lag_{lag}")
            feature_names.append(f"Return_lag_{lag}")
            feature_names.append(f"Volatility_{lag}")
            feature_names.append(f"Drawdown_{lag}")
        return np.array(feature_names)


class IndicatorTransformer(BaseEstimator, TransformerMixin):
    """Adds configured technical indicators to the DataFrame."""

    def __init__(
        self,
        indicators: list[str] | set[str] | None = None,
        use_indicators: bool = True,
        rsi_period: int = 14,
        sma_window: int = 200,
        ema_span: int = 50,
    ):
        self.indicators = indicators
        self.use_indicators = use_indicators
        self.rsi_period = rsi_period
        self.sma_window = sma_window
        self.ema_span = ema_span

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X
        if not self.use_indicators or not self.indicators:
            return X.copy()
        out, _ = apply_configured_indicators(
            X,
            indicators=self.indicators,
            use_indicators=self.use_indicators,
            logger=None,
            rsi_period=self.rsi_period,
            sma_window=self.sma_window,
            ema_span=self.ema_span,
        )
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else None


class ReturnOutlierClipper(BaseEstimator, TransformerMixin):
    """Clips extreme returns using past-only quantiles (causal)."""

    def __init__(
        self,
        price_columns: list[str] | None = None,
        lower_pct: float = 0.1,
        upper_pct: float = 99.9,
        window: int | None = None,
        min_periods: int = 30,
        min_return: float = -0.99,
    ):
        self.price_columns = price_columns or ["Close"]
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.window = window
        self.min_periods = min_periods
        self.min_return = min_return

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X

        X_copy = X.copy()
        for col in self.price_columns:
            if col not in X_copy.columns:
                continue
            series = pd.to_numeric(X_copy[col], errors="coerce")
            if series.isna().all():
                continue

            returns = series.pct_change()
            if self.window:
                lower = (
                    returns.rolling(self.window, min_periods=self.min_periods)
                    .quantile(self.lower_pct / 100)
                    .shift(1)
                )
                upper = (
                    returns.rolling(self.window, min_periods=self.min_periods)
                    .quantile(self.upper_pct / 100)
                    .shift(1)
                )
            else:
                lower = (
                    returns.expanding(min_periods=self.min_periods)
                    .quantile(self.lower_pct / 100)
                    .shift(1)
                )
                upper = (
                    returns.expanding(min_periods=self.min_periods)
                    .quantile(self.upper_pct / 100)
                    .shift(1)
                )

            lower = lower.clip(lower=self.min_return)
            clipped = returns.clip(lower, upper)
            mask = lower.isna() | upper.isna()
            if mask.any():
                clipped = clipped.where(~mask, returns)

            adjusted = (1.0 + clipped.fillna(0.0)).cumprod()
            if not adjusted.empty:
                adjusted.iloc[0] = 1.0
            X_copy[col] = series.iloc[0] * adjusted

        return X_copy

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else None


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
            except (ValueError, TypeError):
                # Index cannot be converted to datetime - skip calendar features
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

    def __init__(
        self,
        feature_columns: list[str],
        handle_missing: str = "drop",
        add_missing_flags: bool = False,
    ):
        """
        Args:
            feature_columns: List of feature column names to select.
            handle_missing: How to handle missing values: 'drop', 'ffill', or 'median'.
            add_missing_flags: Whether to add *_missing indicator columns.
        """
        self.feature_columns = feature_columns
        self.handle_missing = handle_missing
        self.add_missing_flags = add_missing_flags
        self.medians_ = {}
        self._missing_suffix = "_missing"

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
        missing_mask = X_selected.isna() if self.add_missing_flags else None

        if self.handle_missing == "ffill":
            # Forward-fill only; remaining NaNs are filled with a constant to avoid future leakage.
            X_selected = X_selected.ffill()
            X_selected = X_selected.apply(pd.to_numeric, errors="coerce")
            if X_selected.isna().any().any():
                X_selected = X_selected.fillna(0.0)
        elif self.handle_missing == "median":
            X_selected = X_selected.fillna(self.medians_)
        # 'drop' is handled later in the pipeline or by the caller

        if self.add_missing_flags and missing_mask is not None:
            for col in available_features:
                X_selected[f"{col}{self._missing_suffix}"] = missing_mask[col].astype(int)

        return X_selected

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output."""
        feature_names = list(input_features) if input_features is not None else list(self.feature_columns)
        if self.add_missing_flags:
            base_features = [col for col in feature_names if not col.endswith(self._missing_suffix)]
            for col in base_features:
                flag = f"{col}{self._missing_suffix}"
                if flag not in feature_names:
                    feature_names.append(flag)
        return np.array(feature_names)
