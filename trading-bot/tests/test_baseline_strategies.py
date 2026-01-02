import pandas as pd

from trading_bot.backtest.interfaces import Signal, StrategyState
from trading_bot.strategies.atr_breakout_strategy import ATRBreakoutStrategy
from trading_bot.strategies.mean_reversion_strategy import MeanReversionStrategy
from trading_bot.strategies.regime_switch_strategy import RegimeSwitchStrategy


def test_atr_breakout_buy_signal():
    index = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    data = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 110],
            "High": [101, 102, 103, 104, 110],
            "Low": [99, 100, 101, 102, 109],
            "Close": [100, 101, 102, 103, 110],
            "Volume": [1.0] * 5,
        },
        index=index,
    )
    config = {
        "strategy": "atr_breakout",
        "ticker": "TEST",
        "donchian_window": 3,
        "atr_window": 2,
        "atr_stop_mult": 2.0,
        "atr_trail_mult": 2.0,
    }
    strategy = ATRBreakoutStrategy(config, data)
    prepared = strategy.prepare_data(data)
    state = StrategyState(cash=1000.0, position=0.0, equity=1000.0)

    signal = strategy.on_bar(state, prepared)
    assert signal == Signal.BUY


def test_mean_reversion_buy_signal():
    index = pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC")
    data = pd.DataFrame(
        {
            "Close": [90.0],
            "BB_Lower": [95.0],
            "BB_Middle": [100.0],
            "RSI": [25.0],
        },
        index=index,
    )
    config = {
        "strategy": "mean_reversion",
        "ticker": "TEST",
        "rsi_oversold": 30.0,
    }
    strategy = MeanReversionStrategy(config, data)
    state = StrategyState(cash=1000.0, position=0.0, equity=1000.0)

    signal = strategy.on_bar(state, data)
    assert signal == Signal.BUY


def test_regime_switch_trend_buy_signal():
    index = pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC")
    data = pd.DataFrame(
        {
            "Close": [105.0],
            "ADX": [30.0],
            "Volatility": [0.01],
            "ATR": [1.0],
            "DonchianHigh": [100.0],
        },
        index=index,
    )
    config = {
        "strategy": "regime_switch",
        "ticker": "TEST",
        "adx_trend_threshold": 25.0,
        "volatility_high_threshold": 0.05,
        "atr_stop_mult": 2.0,
        "atr_trail_mult": 2.0,
    }
    strategy = RegimeSwitchStrategy(config, data)
    state = StrategyState(cash=1000.0, position=0.0, equity=1000.0)

    signal = strategy.on_bar(state, data)
    assert signal == Signal.BUY
