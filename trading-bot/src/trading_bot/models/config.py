from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from trading_bot.utils.time_utils import interval_to_seconds, parse_period_to_timedelta

ALLOWED_STRATEGIES = {
    "day_trading",
    "short_term",
    "mid_term",
    "long_term",
    "atr_breakout",
    "mean_reversion",
    "regime_switch",
    "day_trading_ml",
}
ALLOWED_DATA_SOURCES = {"yahoo", "binance"}
ALLOWED_INDICATORS = {"rsi", "macd", "sma", "ema", "stochastic", "bollinger_bands", "adx"}
BINANCE_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}
YAHOO_INTERVALS = {"1d", "1wk", "1mo", "1h", "90m"}


class _DictLikeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, key, default=None):
        return getattr(self, key, default)


class RiskConfig(_DictLikeModel):
    stop_loss: float | None = None
    take_profit: float | None = None
    max_position_size: float = 0.1
    trading_fee: float = 0.001
    trailing_stop: float | None = None

    @field_validator("stop_loss", "take_profit", "max_position_size", "trading_fee", "trailing_stop", mode="before")
    @classmethod
    def _ensure_range(cls, value):
        if value is None:
            return value
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError("must be between 0 and 1")
        return value


class AdaptiveThresholdConfig(_DictLikeModel):
    volatility_window: int = Field(20, ge=2)
    reference_volatility: float | None = Field(0.02, ge=0)
    min_multiplier: float = Field(0.5, gt=0)
    max_multiplier: float = Field(3.0, gt=0)

    @model_validator(mode="after")
    def _validate_multipliers(self):
        if self.min_multiplier > self.max_multiplier:
            raise ValueError("min_multiplier must be <= max_multiplier")
        return self


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
    risk_management: RiskConfig = Field(default_factory=RiskConfig)
    min_rows: int = Field(50, ge=1)
    train_window_rows: int | None = Field(None, ge=1)
    drop_nonpositive_volume: bool = True
    cache_ttl_seconds: int | None = Field(None, ge=0)
    sleep_seconds: int | None = Field(None, ge=0)
    inference_only: bool = False
    lstm_quality_gate_enabled: bool = True
    lstm_quality_gate_ratio: float = Field(1.0, gt=0)
    force_retrain_on_drift: bool = False

    # Decision thresholds (strategy-specific, can be overridden)
    buy_threshold: float = Field(0.005, ge=0)
    sell_threshold: float = Field(-0.005, le=0)
    min_inference_rows: int | None = None
    use_adaptive_thresholds: bool = False
    adaptive_threshold_config: AdaptiveThresholdConfig = Field(default_factory=AdaptiveThresholdConfig)

    # ATR breakout / trend parameters
    donchian_window: int | None = Field(None, ge=1)
    atr_window: int | None = Field(None, ge=1)
    atr_stop_mult: float | None = Field(None, ge=0)
    atr_trail_mult: float | None = Field(None, ge=0)
    time_stop_bars: int | None = Field(None, ge=1)

    # Mean reversion parameters
    bb_window: int | None = Field(None, ge=1)
    bb_num_std: float | None = Field(None, gt=0)
    rsi_period: int | None = Field(None, ge=1)
    rsi_oversold: float | None = Field(None, ge=0, le=100)
    rsi_overbought: float | None = Field(None, ge=0, le=100)
    max_loss_pct: float | None = Field(None, ge=0, le=1)
    partial_take_profit_pct: float | None = Field(None, ge=0, le=1)

    # Regime switching parameters
    adx_window: int | None = Field(None, ge=1)
    adx_trend_threshold: float | None = Field(None, ge=0)
    volatility_window: int | None = Field(None, ge=1)
    volatility_high_threshold: float | None = Field(None, ge=0)
    cooldown_bars: int | None = Field(None, ge=0)

    # ML strategy parameters
    return_horizon: int | None = Field(None, ge=1)
    prediction_threshold: float | None = Field(None, ge=0)
    slippage_rate: float | None = Field(None, ge=0)
    ml_min_improvement: float | None = Field(None, ge=0)
    ml_max_drawdown: float | None = Field(None, ge=0, le=1)
    ml_min_hit_rate: float | None = Field(None, ge=0, le=1)
    training_tickers: list[str] | None = None


    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, value):
        if value not in ALLOWED_STRATEGIES:
            raise ValueError(f"Unsupported strategy '{value}'. Allowed: {sorted(ALLOWED_STRATEGIES)}")
        return value

    @field_validator("data_source")
    @classmethod
    def _validate_data_source(cls, value):
        if value not in ALLOWED_DATA_SOURCES:
            raise ValueError(f"Unsupported data_source '{value}'. Allowed: {sorted(ALLOWED_DATA_SOURCES)}")
        return value

    @field_validator("indicators", mode="before")
    @classmethod
    def _normalize_indicators(cls, value):
        if value is None:
            return set()
        if isinstance(value, str):
            value = [value]
        return {str(v).lower() for v in value}

    @field_validator("indicators")
    @classmethod
    def _validate_indicators(cls, value):
        invalid = value - ALLOWED_INDICATORS
        if invalid:
            raise ValueError(f"Indicators not implemented: {sorted(invalid)}")
        return value

    @field_validator("period")
    @classmethod
    def _validate_period(cls, value):
        try:
            parse_period_to_timedelta(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value):
        if value is None:
            return value
        if isinstance(value, str):
            return value.upper()
        if isinstance(value, int):
            return value
        raise ValueError("log_level should be a string or int")

    @field_validator("notification_email", mode="before")
    @classmethod
    def _coerce_email_list(cls, value):
        if value is None:
            return value
        if isinstance(value, list):
            return value
        return [value]

    @model_validator(mode="after")
    def _validate_interval_and_indicators(self):
        data_source = self.data_source
        interval = self.interval
        if not self.use_indicators:
            self.indicators = set()

        allowed = BINANCE_INTERVALS if data_source == "binance" else YAHOO_INTERVALS
        if interval not in allowed:
            raise ValueError(
                f"Unsupported interval '{interval}' for source '{data_source}'. Allowed: {sorted(allowed)}"
            )
        interval_seconds = interval_to_seconds(interval)
        if (
            self.sleep_seconds is not None
            and interval_seconds is not None
            and self.sleep_seconds < interval_seconds
        ):
            raise ValueError(
                f"sleep_seconds ({self.sleep_seconds}) must be greater than or equal to interval length "
                f"in seconds ({interval_seconds})"
            )
        return self


def parse_strategy_config(raw_config) -> StrategyConfig:
    """
    Parse a raw config (dict or StrategyConfig) into a validated StrategyConfig.
    Raises ValidationError on failure.
    """
    if isinstance(raw_config, StrategyConfig):
        return raw_config
    return StrategyConfig(**raw_config)
