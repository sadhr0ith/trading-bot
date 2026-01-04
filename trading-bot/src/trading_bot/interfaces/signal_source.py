"""Signal source abstraction for multi-strategy coordination.

This module provides abstractions for collecting signals from strategies,
designed to support future multiprocess/distributed signal generation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd

from trading_bot.models.signal import SignalDecision

if TYPE_CHECKING:
    from trading_bot.strategies.strategy_base import StrategyBase


class MarketContext:
    """
    Container for market data required by strategies to generate signals.

    Designed to be passed to signal sources, encapsulating all data needed
    for signal generation without coupling to specific data fetching logic.
    """

    def __init__(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_price: float | None = None,
        timeframe: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.symbol = symbol
        self.data = data
        self.current_price = current_price if current_price is not None else float(data["Close"].iloc[-1])
        self.timeframe = timeframe
        self.metadata = metadata or {}


@runtime_checkable
class SignalSource(Protocol):
    """
    Protocol for signal sources that collect trading signals.

    This abstraction enables future multiprocessing support:
    - InProcessSignalSource: strategies run in-process (current implementation)
    - Future: RemoteSignalSource for multiprocess/distributed signal generation

    The key invariant is that signal sources produce SignalDecision objects
    without mutating any portfolio state.
    """

    def collect(self, market_context: MarketContext) -> list[SignalDecision]:
        """
        Collect signals from all underlying strategies for the given market context.

        Args:
            market_context: Market data and metadata for signal generation

        Returns:
            List of SignalDecision objects, one per strategy that generated a signal.
            Strategies that cannot generate a signal (e.g., insufficient data) may
            return HOLD or be omitted.
        """
        ...


class InProcessSignalSource:
    """
    Signal source that collects signals from strategies running in the same process.

    This is the default implementation for multi-strategy mode, where all strategies
    share the same process but generate signals independently before aggregation.
    """

    def __init__(self, strategies: list["StrategyBase"]) -> None:
        """
        Initialize with a list of strategy instances.

        Args:
            strategies: List of StrategyBase subclass instances, each configured
                       with its own config and ready to generate signals.
        """
        self._strategies = strategies

    @property
    def strategies(self) -> list["StrategyBase"]:
        """Access to underlying strategies for inspection."""
        return self._strategies

    def collect(self, market_context: MarketContext) -> list[SignalDecision]:
        """
        Collect signals from all strategies for the given market context.

        Each strategy's generate_signal method is called, and valid SignalDecision
        objects are collected. Strategies that raise exceptions or return None
        are logged but do not prevent other strategies from contributing signals.

        Args:
            market_context: Market data and metadata for signal generation

        Returns:
            List of SignalDecision objects from all strategies that successfully
            generated a signal.
        """
        signals: list[SignalDecision] = []

        for strategy in self._strategies:
            try:
                signal = strategy.generate_signal(market_context)
                if signal is not None:
                    signals.append(signal)
            except Exception as exc:
                # Log but don't fail - other strategies can still contribute
                strategy.log_action(
                    f"Signal generation failed for {strategy.__class__.__name__}: {exc}",
                    "warning",
                )

        return signals
