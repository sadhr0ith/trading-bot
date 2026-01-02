from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtest import portfolio_cli


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


def _frame(rows: int = 30) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="1H", tz="UTC")
    close = [100.0 + i for i in range(rows)]
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": [1.0] * rows},
        index=index,
    )


def test_portfolio_cli_runs(monkeypatch, tmp_path):
    def fake_fetch_data_online(*args, **kwargs):
        return _frame()

    monkeypatch.setattr(portfolio_cli, "fetch_data_online", fake_fetch_data_online, raising=True)

    code = portfolio_cli.main(
        [
            "--strategy",
            "atr_breakout",
            "--tickers",
            "AAA,BBB",
            "--interval",
            "1h",
            "--period",
            "1d",
            "--report-dir",
            str(tmp_path),
            "--no-vol-targeting",
        ]
    )

    assert code == 0
    assert list(tmp_path.glob("backtest_*.json"))

