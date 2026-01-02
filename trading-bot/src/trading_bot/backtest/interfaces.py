"""Interfaces and data structures for backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import pandas as pd


class Signal(str, Enum):
    """Trading signal for backtest strategies."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Trade:
    """Executed trade record."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    entry_fee: float
    exit_fee: float
    entry_value: float
    exit_value: float
    pnl: float
    return_pct: float


@dataclass
class StrategyState:
    """Mutable backtest state passed to strategy on each bar."""

    cash: float
    position: float
    equity: float
    current_time: pd.Timestamp | None = None
    entry_price: float | None = None
    entry_time: pd.Timestamp | None = None
    entry_fee: float = 0.0
    last_signal: Signal | None = None
    trades: list[Trade] = field(default_factory=list)


class BacktestStrategy(Protocol):
    """Protocol for backtest strategies."""

    def on_bar(self, state: StrategyState, window_df: pd.DataFrame) -> Signal:  # pragma: no cover - interface
        """Return BUY/SELL/HOLD for the current bar."""
        ...

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - optional
        """Optional hook to add indicators or features ahead of backtest loop."""
        return data
