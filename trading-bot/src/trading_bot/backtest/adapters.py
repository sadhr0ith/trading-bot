"""Backtest adapters for indicators and helpers."""

from __future__ import annotations

import pandas as pd

from trading_bot.indicators.adx import ADX
from trading_bot.indicators.bollinger_bands import BollingerBands
from trading_bot.indicators.ema import EMA
from trading_bot.indicators.macd import MACD
from trading_bot.indicators.rsi import RSI
from trading_bot.indicators.sma import SMA
from trading_bot.indicators.stochastic import StochasticOscillator
from trading_bot.utils.feature_engineering import apply_configured_indicators


def add_indicators(data: pd.DataFrame, indicators: list[str] | set[str]) -> pd.DataFrame:
    """Add configured indicators to the data frame."""
    out, _added = apply_configured_indicators(data, indicators, use_indicators=True)
    return out


def compute_atr(data: pd.DataFrame, window: int = 14, column: str = "ATR") -> pd.Series:
    """Compute Average True Range (ATR)."""
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr.rename(column)


def compute_donchian_channels(data: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    """Compute Donchian channel upper/lower bands."""
    upper = data["High"].rolling(window=window, min_periods=window).max()
    lower = data["Low"].rolling(window=window, min_periods=window).min()
    return upper, lower


def add_indicator_set(data: pd.DataFrame, indicator_set: set[str]) -> pd.DataFrame:
    """Compatibility helper for backtest strategies using indicator classes directly."""
    out = data.copy()

    if "rsi" in indicator_set:
        out["RSI"] = RSI(out).calculate()
    if "macd" in indicator_set:
        macd = MACD(out).calculate()
        out = out.join(macd[["MACD", "Signal", "MACD_Histogram"]])
    if "adx" in indicator_set:
        adx = ADX(out).calculate()
        out = out.join(adx[["ADX", "Plus_DI", "Minus_DI"]])
    if "bollinger_bands" in indicator_set:
        bb = BollingerBands(out).calculate()
        out = out.join(bb[["BB_Middle", "BB_Upper", "BB_Lower", "BB_Width"]])
    if "sma" in indicator_set:
        out = out.join(SMA(out, window=20, alias="SMA").calculate())
    if "ema" in indicator_set:
        out = out.join(EMA(out, span=20, alias="EMA").calculate())
    if "stochastic" in indicator_set:
        stoch = StochasticOscillator(out).calculate()
        out = out.join(stoch[["Stochastic_K", "Stochastic_D"]])

    return out
