import pandas as pd

from strategies.long_term_strategy import LongTermStrategy
from strategies.mid_term_strategy import MidTermStrategy
from strategies.short_term_strategy import ShortTermStrategy


def _make_dummy_data(rows=120):
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = pd.DataFrame(
        {
            "Open": pd.Series(range(rows), index=idx) + 100,
            "High": pd.Series(range(rows), index=idx) + 101,
            "Low": pd.Series(range(rows), index=idx) + 99,
            "Close": pd.Series(range(rows), index=idx) + 100,
            "Volume": pd.Series(range(rows), index=idx) * 10 + 1000,
        },
        index=idx,
    )
    return data


def test_short_term_pipeline_runs(monkeypatch, tmp_path):
    data = _make_dummy_data()
    config = {
        "ticker": "TEST",
        "strategy": "short_term",
        "indicators": ["macd", "rsi", "adx"],
        "use_indicators": True,
        "notification_email": "none@example.com",
        "seed": 42,
    }
    monkeypatch.setenv("TRADING_BOT_SEED", "42")
    strategy = ShortTermStrategy(config, data)
    strategy.execute()


def test_mid_term_pipeline_runs(monkeypatch):
    data = _make_dummy_data(rows=160)
    config = {
        "ticker": "TEST",
        "strategy": "mid_term",
        "indicators": ["macd", "bollinger_bands"],
        "use_indicators": True,
        "notification_email": "none@example.com",
        "seed": 42,
    }
    monkeypatch.setenv("TRADING_BOT_SEED", "42")
    strategy = MidTermStrategy(config, data)
    strategy.execute()


def test_long_term_pipeline_runs(monkeypatch):
    data = _make_dummy_data(rows=300)
    config = {
        "ticker": "TEST",
        "strategy": "long_term",
        "indicators": ["sma", "ema", "macd"],
        "use_indicators": True,
        "notification_email": "none@example.com",
        "seed": 42,
    }
    monkeypatch.setenv("TRADING_BOT_SEED", "42")
    strategy = LongTermStrategy(config, data)
    strategy.execute()
