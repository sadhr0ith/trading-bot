from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from models.config import ALLOWED_INDICATORS, StrategyConfig, parse_strategy_config
from utils.logger import setup_logger

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data: pd.DataFrame | None = None


class ConfigValidator:
    REQUIRED_FIELDS = {"strategy", "data_source", "ticker", "period", "interval", "indicators"}
    ALLOWED_DATA_SOURCES = {"yahoo", "binance"}
    ALLOWED_STRATEGIES = {"day_trading", "short_term", "mid_term", "long_term"}

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    def validate(self, config: Dict) -> ValidationResult:
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
    def __init__(self, min_rows: int = 50, require_ohlcv: bool = True):
        self.min_rows = min_rows
        self.require_ohlcv = require_ohlcv
        self.logger = setup_logger(self.__class__.__name__)

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

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
            # Return immediately to prevent KeyError when accessing missing columns
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                data=None
            )

        if len(df) < self.min_rows:
            warnings.append(f"DataFrame has only {len(df)} rows; minimum recommended is {self.min_rows}.")

        if df.index.duplicated().any():
            warnings.append("Duplicate index entries detected; keeping first occurrence.")
            df = df[~df.index.duplicated(keep="first")]

        if df[['Close']].isna().any().any():
            warnings.append("NaN values detected in Close; dropping those rows.")
            df = df.dropna(subset=['Close'])

        if 'Volume' in df.columns and (df['Volume'] <= 0).any():
            warnings.append("Non-positive volume rows detected; filtering them out.")
            df = df[df['Volume'] > 0]

        if (df['Close'] <= 0).any():
            errors.append("Non-positive Close prices detected; aborting.")

        is_valid = len(errors) == 0
        for warn in warnings:
            self.logger.warning(warn)
        if not is_valid:
            for err in errors:
                self.logger.error(err)
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, data=df)
