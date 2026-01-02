from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.strategies.day_trading_strategy import DayTradingStrategy
from trading_bot.utils.model_persistence import ModelPersistence


class _DummyPaperExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def process_signal(self, *args, **kwargs):
        return {"status": "noop"}


@pytest.fixture(autouse=True)
def _patch_paper_executor(monkeypatch):
    from trading_bot.strategies import strategy_base
    from trading_bot.utils import paper_trading

    monkeypatch.setattr(paper_trading, "PaperTradingExecutor", _DummyPaperExecutor, raising=True)
    monkeypatch.setattr(strategy_base, "PaperTradingExecutor", _DummyPaperExecutor, raising=True)


def _dummy_ohlcv(rows: int = 2) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="1H", tz="UTC")
    close = [100.0 + i for i in range(rows)]
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": [1.0] * rows,
        },
        index=index,
    )


def test_lstm_benchmark_required_blocks_when_underperforming(monkeypatch):
    def fake_load_metadata(self, strategy: str, version: str | None = None):
        if strategy.startswith("day_trading_ml__"):
            return {"pnl_proxy_net": 0.10}
        return None

    monkeypatch.setattr(ModelPersistence, "load_metadata", fake_load_metadata, raising=True)

    config = {
        "strategy": "day_trading",
        "data_source": "binance",
        "ticker": "BTCUSDT",
        "period": "1y",
        "interval": "1h",
        "indicators": [],
        "use_indicators": False,
        "notification_email": None,
        "lstm_quality_gate_enabled": False,
        "lstm_benchmark_enabled": True,
        "lstm_benchmark_strategy": "day_trading_ml",
        "lstm_benchmark_required": True,
        "lstm_benchmark_min_delta": 0.0,
        "risk_management": {"trading_fee": 0.0},
    }
    strategy = DayTradingStrategy(config, _dummy_ohlcv())

    lstm_meta = {"pnl_proxy_net": 0.05}
    assert strategy._gate_from_metadata(lstm_meta) is False


def test_lstm_benchmark_not_required_allows_missing_benchmark(monkeypatch):
    def fake_load_metadata(self, strategy: str, version: str | None = None):
        return None

    monkeypatch.setattr(ModelPersistence, "load_metadata", fake_load_metadata, raising=True)

    config = {
        "strategy": "day_trading",
        "data_source": "binance",
        "ticker": "BTCUSDT",
        "period": "1y",
        "interval": "1h",
        "indicators": [],
        "use_indicators": False,
        "notification_email": None,
        "lstm_quality_gate_enabled": False,
        "lstm_benchmark_enabled": True,
        "lstm_benchmark_strategy": "day_trading_ml",
        "lstm_benchmark_required": False,
        "risk_management": {"trading_fee": 0.0},
    }
    strategy = DayTradingStrategy(config, _dummy_ohlcv())

    lstm_meta = {"pnl_proxy_net": 0.05}
    assert strategy._gate_from_metadata(lstm_meta) is True

