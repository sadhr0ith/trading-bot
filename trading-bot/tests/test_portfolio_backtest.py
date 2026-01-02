import pandas as pd

from trading_bot.backtest.interfaces import Signal
from trading_bot.backtest.portfolio import PortfolioConstraints, run_portfolio_backtest


class BuySellLastStrategy:
    def __init__(self, last_index: pd.Timestamp):
        self.last_index = last_index

    def on_bar(self, state, window_df):
        if state.position == 0:
            return Signal.BUY
        if window_df.index[-1] == self.last_index:
            return Signal.SELL
        return Signal.HOLD


class BuySellOnIndexStrategy:
    def __init__(self, buy_index: pd.Timestamp, sell_index: pd.Timestamp):
        self.buy_index = buy_index
        self.sell_index = sell_index

    def on_bar(self, state, window_df):
        ts = window_df.index[-1]
        if state.position == 0 and ts == self.buy_index:
            return Signal.BUY
        if state.position > 0 and ts == self.sell_index:
            return Signal.SELL
        return Signal.HOLD


def _make_frame(start: str, periods: int, freq: str) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = list(range(100, 100 + periods))
    data = {
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
        "Volume": [1.0] * periods,
    }
    return pd.DataFrame(data, index=index)


def test_portfolio_backtest_runs():
    asset_a = _make_frame("2024-01-01", periods=4, freq="1D")
    asset_b = _make_frame("2024-01-01", periods=4, freq="1D")
    data_by_asset = {"AAA": asset_a, "BBB": asset_b}

    def factory(_ticker: str, df: pd.DataFrame):
        return BuySellLastStrategy(df.index[-1])

    result = run_portfolio_backtest(
        data_by_asset,
        factory,
        initial_cash=10_000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        constraints=PortfolioConstraints(max_positions=2, max_exposure_per_asset=0.6, vol_targeting=False),
    )

    assert len(result.equity_curve) == 4
    assert result.trades
    assert result.metrics.get("total_return") is not None


def test_portfolio_correlation_filter_limits_positions():
    asset_a = _make_frame("2024-01-01", periods=6, freq="1D")
    asset_b = asset_a.copy()
    data_by_asset = {"AAA": asset_a, "BBB": asset_b}

    def factory(_ticker: str, df: pd.DataFrame):
        return BuySellOnIndexStrategy(df.index[-2], df.index[-1])

    constraints = PortfolioConstraints(
        max_positions=2,
        max_exposure_per_asset=1.0,
        vol_targeting=False,
        correlation_filter=True,
        correlation_window=3,
        max_pairwise_correlation=0.5,
    )
    result = run_portfolio_backtest(
        data_by_asset,
        factory,
        initial_cash=10_000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        constraints=constraints,
    )

    assert len(result.trades) == 1
