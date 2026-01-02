import pandas as pd
import pytest

from trading_bot.backtest.engine import run_backtest
from trading_bot.backtest.interfaces import Signal


class BuyThenSellStrategy:
    def __init__(self, last_index: pd.Timestamp):
        self.last_index = last_index

    def on_bar(self, state, window_df):
        if state.position == 0:
            return Signal.BUY
        if window_df.index[-1] == self.last_index:
            return Signal.SELL
        return Signal.HOLD


def _sample_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    close = [100.0, 110.0, 120.0, 130.0]
    data = {
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
        "Volume": [1.0] * 4,
    }
    return pd.DataFrame(data, index=index)


def test_backtest_no_costs_pnl():
    df = _sample_frame()
    strategy = BuyThenSellStrategy(df.index[-1])
    result = run_backtest(df, strategy, initial_cash=1000.0, fee_rate=0.0, slippage_rate=0.0, close_out=False)

    assert len(result.trades) == 1
    assert len(result.equity_curve) == len(df)

    expected_final = 1000.0 * (130.0 / 100.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(expected_final, rel=1e-8)


def test_backtest_costs_reduce_equity():
    df = _sample_frame()
    strategy = BuyThenSellStrategy(df.index[-1])
    result = run_backtest(df, strategy, initial_cash=1000.0, fee_rate=0.001, slippage_rate=0.0002, close_out=False)

    assert len(result.trades) == 1
    trade = result.trades[0]

    assert trade.entry_fee > 0
    assert trade.exit_fee > 0
    assert result.equity_curve.iloc[-1] < 1000.0 * (130.0 / 100.0)
    assert result.equity_curve.iloc[-1] > 0
