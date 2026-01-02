import pandas as pd

from trading_bot.backtest.engine import run_backtest
from trading_bot.backtest.interfaces import Signal


class NoLookaheadStrategy:
    def on_bar(self, state, window_df):
        assert window_df.index[-1] == state.current_time
        assert (window_df.index <= state.current_time).all()
        return Signal.HOLD


def test_no_lookahead_window():
    index = pd.date_range("2024-01-01", periods=10, freq="1H", tz="UTC")
    data = {
        "Open": range(10),
        "High": range(10),
        "Low": range(10),
        "Close": range(10),
        "Volume": [1.0] * 10,
    }
    df = pd.DataFrame(data, index=index)

    strategy = NoLookaheadStrategy()
    result = run_backtest(df, strategy, close_out=False)

    assert result.equity_curve.shape[0] == len(df)
