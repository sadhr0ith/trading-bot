from __future__ import annotations

from typing import Any

import pandas as pd

from trading_bot.backtest.adapters import compute_atr, compute_donchian_channels
from trading_bot.backtest.interfaces import Signal, StrategyState
from trading_bot.models.signal import SignalAction
from trading_bot.strategies.strategy_base import StrategyBase


class ATRBreakoutStrategy(StrategyBase):
    """ATR breakout strategy with trailing and time-based exits."""

    MIN_ROWS = 50

    def __init__(self, config, data: pd.DataFrame) -> None:
        super().__init__(config, data)
        self.donchian_window = int(config.get("donchian_window", 20))
        self.atr_window = int(config.get("atr_window", 14))
        self.atr_stop_mult = float(config.get("atr_stop_mult", 2.0))
        self.atr_trail_mult = float(config.get("atr_trail_mult", 3.0))
        self.time_stop_bars = config.get("time_stop_bars")
        if self.time_stop_bars is not None:
            self.time_stop_bars = int(self.time_stop_bars)

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out["ATR"] = compute_atr(out, window=self.atr_window)
        upper, lower = compute_donchian_channels(out, window=self.donchian_window)
        out["DonchianHigh"] = upper.shift(1)
        out["DonchianLow"] = lower.shift(1)
        return out

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.prepare_data(data)

    def _bars_held(self, window_df: pd.DataFrame, entry_time: pd.Timestamp | None) -> int:
        if entry_time is None:
            return 0
        if entry_time in window_df.index:
            return len(window_df.loc[entry_time:]) - 1
        return int((window_df.index >= entry_time).sum())

    def _max_close_since_entry(self, window_df: pd.DataFrame, entry_time: pd.Timestamp | None) -> float:
        if entry_time is None:
            return float(window_df["Close"].iloc[-1])
        if entry_time in window_df.index:
            return float(window_df.loc[entry_time:, "Close"].max())
        return float(window_df.loc[window_df.index >= entry_time, "Close"].max())

    def _signal(
        self,
        window_df: pd.DataFrame,
        in_position: bool,
        entry_price: float | None,
        entry_time: pd.Timestamp | None,
    ) -> Signal:
        row = window_df.iloc[-1]
        close = float(row["Close"])
        atr = float(row.get("ATR", 0.0))
        donchian_high = row.get("DonchianHigh")

        if not in_position:
            if pd.notna(donchian_high) and close > float(donchian_high):
                return Signal.BUY
            return Signal.HOLD

        if entry_price is None or atr <= 0:
            return Signal.HOLD

        max_close = self._max_close_since_entry(window_df, entry_time)
        initial_stop = entry_price - self.atr_stop_mult * atr
        trailing_stop = max_close - self.atr_trail_mult * atr
        stop_level = max(initial_stop, trailing_stop)

        if close < stop_level:
            return Signal.SELL
        if self.time_stop_bars is not None:
            bars_held = self._bars_held(window_df, entry_time)
            if bars_held >= self.time_stop_bars:
                return Signal.SELL
        return Signal.HOLD

    def on_bar(self, state: StrategyState, window_df: pd.DataFrame) -> Signal:
        return self._signal(window_df, state.position > 0, state.entry_price, state.entry_time)

    def _run_strategy(self, data: pd.DataFrame) -> None:
        asset = self.config["ticker"]
        position = self.order_executor.state.get("positions", {}).get(asset)
        in_position = position is not None
        entry_price = float(position["entry_price"]) if position else None

        entry_time = None
        if position and position.get("opened_at"):
            try:
                entry_time = pd.to_datetime(position["opened_at"], utc=True)
            except (ValueError, TypeError):
                entry_time = None

        signal = self._signal(data, in_position, entry_price, entry_time)
        price = float(data["Close"].iloc[-1])
        self.log_action(f"ATR breakout signal: {signal}", "info" if signal != Signal.HOLD else "warning")
        self.order_executor.process_signal(asset, signal.value, price, self.risk_manager)

    def _compute_signal_action(
        self,
        data: pd.DataFrame,
    ) -> tuple[SignalAction, float | None, float | None, dict[str, Any]] | None:
        """Compute signal action for multi-strategy mode."""
        asset = self.config["ticker"]
        position = self.order_executor.state.get("positions", {}).get(asset)
        in_position = position is not None
        entry_price = float(position["entry_price"]) if position else None

        entry_time = None
        if position and position.get("opened_at"):
            try:
                entry_time = pd.to_datetime(position["opened_at"], utc=True)
            except (ValueError, TypeError):
                entry_time = None

        signal = self._signal(data, in_position, entry_price, entry_time)

        # Convert Signal enum to SignalAction
        action_map = {
            Signal.BUY: SignalAction.BUY,
            Signal.SELL: SignalAction.SELL,
            Signal.HOLD: SignalAction.HOLD,
        }
        action = action_map.get(signal, SignalAction.HOLD)

        metadata = {
            "strategy_type": "atr_breakout",
            "in_position": in_position,
            "signal": signal.value,
        }

        return (action, None, None, metadata)
