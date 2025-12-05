from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from models.config import parse_strategy_config
from utils.logger import setup_logger


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
        self.logger = setup_logger(self.__class__.__name__)

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
        outlier_clip_pct: tuple[float, float] = (0.1, 99.9),
    ):
        self.min_rows = min_rows or 50
        self.require_ohlcv = require_ohlcv
        self.fill_method = fill_method
        self.outlier_clip_pct = outlier_clip_pct
        self.logger = setup_logger(self.__class__.__name__)

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
            lower_bound = returns.quantile(lower_pct / 100)
            upper_bound = returns.quantile(upper_pct / 100)
            clipped_returns = returns.clip(lower=lower_bound, upper=upper_bound)
            outlier_count = (returns != clipped_returns).sum()
            if outlier_count > 0:
                warnings.append(f"Clipped {outlier_count} outlier returns to [{lower_bound:.4f}, {upper_bound:.4f}].")

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
