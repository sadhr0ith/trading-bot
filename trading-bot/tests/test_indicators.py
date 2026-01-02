import pandas as pd

from trading_bot.indicators.bollinger_bands import BollingerBands
from trading_bot.indicators.ema import EMA
from trading_bot.indicators.macd import MACD
from trading_bot.indicators.rsi import RSI
from trading_bot.indicators.sma import SMA
from trading_bot.indicators.stochastic import StochasticOscillator


def sample_data(n: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": range(1, n + 1),
            "High": [x + 1 for x in range(1, n + 1)],
            "Low": [x - 1 for x in range(1, n + 1)],
            "Close": range(1, n + 1),
            "Volume": [1000] * n,
        },
        index=idx,
    )


def test_macd_columns_present():
    df = sample_data()
    macd = MACD(df)
    result = macd.calculate()
    assert {"MACD", "Signal", "MACD_Histogram"}.issubset(result.columns)
    assert not result.tail(1).isna().any().any()


def test_rsi_range():
    df = sample_data()
    rsi = RSI(df)
    result = rsi.calculate()
    assert (result.dropna() <= 100).all()
    assert (result.dropna() >= 0).all()


def test_bollinger_bands_columns():
    df = sample_data()
    bb = BollingerBands(df, window=20, num_std=2)
    result = bb.calculate()
    expected = {"BB_Middle", "BB_Upper", "BB_Lower", "BB_Width"}
    assert expected.issubset(result.columns)


def test_sma_ema_columns():
    df = sample_data()
    sma = SMA(df, window=5)
    ema = EMA(df, span=5)
    sma_result = sma.calculate()
    ema_result = ema.calculate()
    assert sma_result.columns[0].startswith("SMA")
    assert ema_result.columns[0].startswith("EMA")


def test_stochastic_columns():
    df = sample_data()
    stoch = StochasticOscillator(df, k_period=14, d_period=3)
    result = stoch.calculate()
    assert {"Stochastic_K", "Stochastic_D"}.issubset(result.columns)
