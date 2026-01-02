import pytest
from pydantic import ValidationError

from trading_bot.models.config import StrategyConfig, parse_strategy_config


def _base_config(**overrides):
    base = {
        "strategy": "short_term",
        "data_source": "binance",
        "ticker": "BTCUSDT",
        "period": "1d",
        "interval": "1h",
        "indicators": ["macd", "rsi"],
        "use_indicators": True,
        "notification_email": "user@example.com",
        "risk_management": {"stop_loss": 0.03, "take_profit": 0.05, "max_position_size": 0.1, "trading_fee": 0.001},
    }
    base.update(overrides)
    return base


def test_invalid_indicator_raises():
    with pytest.raises(ValidationError):
        StrategyConfig(**_base_config(indicators=["unknown"]))


def test_invalid_data_source():
    with pytest.raises(ValidationError):
        StrategyConfig(**_base_config(data_source="invalid"))


def test_invalid_email():
    with pytest.raises(ValidationError):
        StrategyConfig(**_base_config(notification_email="not-an-email"))


def test_risk_range_validation():
    with pytest.raises(ValidationError):
        StrategyConfig(**_base_config(risk_management={"stop_loss": 2.0}))


def test_interval_validation_per_source():
    with pytest.raises(ValidationError):
        StrategyConfig(**_base_config(interval="2h"))  # not in allowed set for binance list here

    cfg = StrategyConfig(**_base_config(data_source="yahoo", interval="1wk"))
    assert cfg.interval == "1wk"


def test_use_indicators_false_clears_list():
    cfg = StrategyConfig(**_base_config(use_indicators=False, indicators=["macd"]))
    assert cfg.use_indicators is False
    assert cfg.indicators == set()


def test_parse_strategy_config_accepts_valid_dict():
    cfg = parse_strategy_config(_base_config())
    assert isinstance(cfg, StrategyConfig)
    assert cfg.strategy == "short_term"


def test_adaptive_threshold_config_parses():
    cfg = StrategyConfig(
        **_base_config(
            use_adaptive_thresholds=True,
            adaptive_threshold_config={"volatility_window": 10, "reference_volatility": None},
        )
    )
    assert cfg.use_adaptive_thresholds is True
    assert cfg.adaptive_threshold_config.volatility_window == 10
    assert cfg.adaptive_threshold_config.reference_volatility is None


def test_sleep_seconds_respects_interval_length():
    cfg = _base_config(interval="1h")
    with pytest.raises(ValidationError):
        StrategyConfig(**cfg, sleep_seconds=300)
