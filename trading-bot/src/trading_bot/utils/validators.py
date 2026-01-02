from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from trading_bot.models.config import parse_strategy_config
from trading_bot.utils.logger import get_logger
from trading_bot.utils.time_utils import interval_to_timedelta


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: pd.DataFrame | None = None


class ConfigValidator:
    REQUIRED_FIELDS = {"strategy", "data_source", "ticker", "period", "interval", "indicators"}
    ALLOWED_DATA_SOURCES = {"yahoo", "binance"}
    ALLOWED_STRATEGIES = {"day_trading", "short_term", "mid_term", "long_term"}

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def validate(self, config: dict) -> ValidationResult:
        if config is None:
            return ValidationResult(False, ["Config is None."])

        try:
            parse_strategy_config(config)
            return ValidationResult(is_valid=True)
        except PydanticValidationError as exc:
            errors = []
            for err in exc.errors():
                loc = " -> ".join(str(piece) for piece in err.get("loc", ()))
                errors.append(f"{loc}: {err.get('msg')}")
                self.logger.error(errors[-1])
            return ValidationResult(is_valid=False, errors=errors)


class DataValidator:
    def __init__(
        self,
        min_rows: int = 50,
        require_ohlcv: bool = True,
        fill_method: str | None = "ffill",
        outlier_clip_pct: tuple[float, float] | None = None,
        expected_interval: str | None = None,
        assume_normalized: bool = False,
    ):
        self.min_rows = min_rows or 50
        self.require_ohlcv = require_ohlcv
        self.fill_method = fill_method
        self.outlier_clip_pct = outlier_clip_pct
        self.expected_interval = expected_interval
        self.logger = get_logger(self.__class__.__name__)
        self.assume_normalized = assume_normalized

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if data is None or data.empty:
            errors.append("DataFrame is empty.")
            return ValidationResult(False, errors)

        df = data.copy()

        if self.require_ohlcv:
            required_columns = {"Open", "High", "Low", "Close", "Volume"}
        else:
            required_columns = {"Close"}
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {sorted(missing_cols)}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, data=None)

        if not self.assume_normalized:
            # Normalize timezone if DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
                self.logger.debug("Localized index to UTC in validator")

            # Deduplicate and sort index
            if df.index.duplicated().any():
                warnings.append("Duplicate index entries detected; keeping first occurrence.")
                df = df[~df.index.duplicated(keep="first")]

            if not df.index.is_monotonic_increasing:
                warnings.append("Index not sorted; sorting now.")
                df = df.sort_index()

        if len(df) < self.min_rows:
            warnings.append(f"DataFrame has only {len(df)} rows; minimum recommended is {self.min_rows}.")

        # Interval sanity check (warn-only; fetcher may already resample)
        if self.expected_interval and isinstance(df.index, pd.DatetimeIndex):
            expected_delta = interval_to_timedelta(self.expected_interval)
            if expected_delta:
                diffs = df.index.to_series().diff().dropna()
                if not diffs.empty:
                    median_diff = diffs.median()
                    gap_count = int((diffs > expected_delta * 1.5).sum())
                    if abs(median_diff - expected_delta) > expected_delta * 0.25 or gap_count > 0:
                        warnings.append(
                            f"Interval mismatch: expected {self.expected_interval} "
                            f"({expected_delta}), median diff {median_diff}, gaps>{expected_delta*1.5}: {gap_count}."
                        )

        # Clean all OHLCV columns, not just Close
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        available_ohlcv = [c for c in ohlcv_cols if c in df.columns]

        # Drop rows with NaN in any OHLCV column
        nan_before = len(df)
        df = df.dropna(subset=available_ohlcv)
        nan_dropped = nan_before - len(df)
        if nan_dropped > 0:
            warnings.append(f"Dropped {nan_dropped} rows with NaN in OHLCV columns.")

        # Filter non-positive prices and volume
        if "Close" in df.columns and (df["Close"] <= 0).any():
            errors.append("Non-positive Close prices detected; aborting.")
        if "Volume" in df.columns and (df["Volume"] <= 0).any():
            before = len(df)
            df = df[df["Volume"] > 0]
            warnings.append(f"Filtered {before - len(df)} rows with Volume<=0.")

        # Outlier detection and clipping
        if self.outlier_clip_pct and "Close" in df.columns:
            returns = df["Close"].pct_change()
            lower_pct, upper_pct = self.outlier_clip_pct
            lower_bound = returns.expanding(min_periods=30).quantile(lower_pct / 100).shift(1)
            upper_bound = returns.expanding(min_periods=30).quantile(upper_pct / 100).shift(1)
            bounds_ready = lower_bound.notna() & upper_bound.notna()
            outlier_mask = bounds_ready & (returns.lt(lower_bound) | returns.gt(upper_bound))
            outlier_indices = returns.index[outlier_mask]
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                df = df.drop(index=outlier_indices)
                last_lower = float(lower_bound[outlier_mask].iloc[-1])
                last_upper = float(upper_bound[outlier_mask].iloc[-1])
                warnings.append(
                    f"Dropped {outlier_count} rows with outlier returns outside "
                    f"[{last_lower:.4f}, {last_upper:.4f}]."
                )

        # Optional gap filling
        if self.fill_method:
            filled_cols = df[available_ohlcv].isna().sum()
            if filled_cols.sum() > 0:
                if self.fill_method == "ffill":
                    df[available_ohlcv] = df[available_ohlcv].ffill()
                warnings.append(f"Forward-filled missing values in OHLCV columns using {self.fill_method}.")

        is_valid = len(errors) == 0
        for warn in warnings:
            self.logger.warning(warn)
        if not is_valid:
            for err in errors:
                self.logger.error(err)
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, data=df)
