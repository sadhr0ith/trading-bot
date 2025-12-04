from __future__ import annotations

from pydantic import BaseModel, EmailStr, root_validator, validator

from utils.time_utils import parse_period_to_timedelta

ALLOWED_STRATEGIES = {"day_trading", "short_term", "mid_term", "long_term"}
ALLOWED_DATA_SOURCES = {"yahoo", "binance"}
ALLOWED_INDICATORS = {"rsi", "macd", "sma", "ema", "stochastic", "bollinger_bands", "adx"}
BINANCE_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}
YAHOO_INTERVALS = {"1d", "1wk", "1mo", "1h", "90m"}


class _DictLikeModel(BaseModel):
    class Config:
        extra = "forbid"

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, key, default=None):
        return getattr(self, key, default)


class RiskConfig(_DictLikeModel):
    stop_loss: float | None = None
    take_profit: float | None = None
    max_position_size: float = 0.1
    trading_fee: float = 0.001

    @validator("stop_loss", "take_profit", "max_position_size", "trading_fee", pre=True, always=True)
    def _ensure_range(cls, value):
        if value is None:
            return value
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError("must be between 0 and 1")
        return value


class StrategyConfig(_DictLikeModel):
    strategy: str
    data_source: str
    ticker: str
    period: str
    interval: str
    indicators: set[str] = set()
    use_indicators: bool = True
    log_level: str | int | None = None
    notification_email: EmailStr | list[EmailStr] | None = None
    seed: int | None = None
    risk_management: RiskConfig = RiskConfig()
    min_rows: int | None = None

    @validator("strategy")
    def _validate_strategy(cls, value):
        if value not in ALLOWED_STRATEGIES:
            raise ValueError(f"Unsupported strategy '{value}'. Allowed: {sorted(ALLOWED_STRATEGIES)}")
        return value

    @validator("data_source")
    def _validate_data_source(cls, value):
        if value not in ALLOWED_DATA_SOURCES:
            raise ValueError(f"Unsupported data_source '{value}'. Allowed: {sorted(ALLOWED_DATA_SOURCES)}")
        return value

    @validator("indicators", pre=True)
    def _normalize_indicators(cls, value):
        if value is None:
            return set()
        if isinstance(value, str):
            value = [value]
        return {str(v).lower() for v in value}

    @validator("indicators")
    def _validate_indicators(cls, value, values):
        invalid = value - ALLOWED_INDICATORS
        if invalid:
            raise ValueError(f"Indicators not implemented: {sorted(invalid)}")
        return value

    @validator("period")
    def _validate_period(cls, value):
        try:
            parse_period_to_timedelta(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(str(exc))
        return value

    @validator("log_level")
    def _normalize_log_level(cls, value):
        if value is None:
            return value
        if isinstance(value, str):
            return value.upper()
        if isinstance(value, int):
            return value
        raise ValueError("log_level should be a string or int")

    @validator("notification_email", pre=True)
    def _coerce_email_list(cls, value):
        if value is None:
            return value
        if isinstance(value, list):
            return value
        return [value]

    @root_validator(skip_on_failure=True)
    def _validate_interval_and_indicators(cls, values):
        data_source = values.get("data_source")
        interval = values.get("interval")
        use_indicators = values.get("use_indicators", True)
        indicators = values.get("indicators") or set()

        if not use_indicators:
            values["indicators"] = set()

        if data_source == "binance":
            allowed = BINANCE_INTERVALS
        else:
            allowed = YAHOO_INTERVALS

        if interval not in allowed:
            raise ValueError(
                f"Unsupported interval '{interval}' for source '{data_source}'. Allowed: {sorted(allowed)}"
            )

        return values


def parse_strategy_config(raw_config) -> StrategyConfig:
    """
    Parse a raw config (dict or StrategyConfig) into a validated StrategyConfig.
    Raises ValidationError on failure.
    """
    if isinstance(raw_config, StrategyConfig):
        return raw_config
    return StrategyConfig(**raw_config)
