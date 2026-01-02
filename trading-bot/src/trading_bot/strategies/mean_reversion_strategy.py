from __future__ import annotations

import pandas as pd

from trading_bot.backtest.interfaces import Signal, StrategyState
from trading_bot.indicators.bollinger_bands import BollingerBands
from trading_bot.indicators.rsi import RSI
from trading_bot.strategies.strategy_base import StrategyBase


class MeanReversionStrategy(StrategyBase):
    """Mean reversion strategy using Bollinger Bands and RSI."""

    MIN_ROWS = 50

    def __init__(self, config, data: pd.DataFrame) -> None:
        super().__init__(config, data)
        self.bb_window = int(config.get("bb_window", 20))
        self.bb_num_std = float(config.get("bb_num_std", 2.0))
        self.rsi_period = int(config.get("rsi_period", 14))
        self.rsi_oversold = float(config.get("rsi_oversold", 30.0))
        self.time_stop_bars = config.get("time_stop_bars")
        if self.time_stop_bars is not None:
            self.time_stop_bars = int(self.time_stop_bars)
        self.max_loss_pct = config.get("max_loss_pct")
        if self.max_loss_pct is not None:
            self.max_loss_pct = float(self.max_loss_pct)
        self.partial_take_profit_pct = config.get("partial_take_profit_pct")
        if self.partial_take_profit_pct is not None:
            self.partial_take_profit_pct = float(self.partial_take_profit_pct)

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        bb = BollingerBands(out, window=self.bb_window, num_std=self.bb_num_std).calculate()
        out = out.join(bb[["BB_Middle", "BB_Upper", "BB_Lower", "BB_Width"]])
        out["RSI"] = RSI(out, period=self.rsi_period).calculate()
        return out

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.prepare_data(data)

    def _bars_held(self, window_df: pd.DataFrame, entry_time: pd.Timestamp | None) -> int:
        if entry_time is None:
            return 0
        if entry_time in window_df.index:
            return len(window_df.loc[entry_time:]) - 1
        return int((window_df.index >= entry_time).sum())

    def _signal(
        self,
        window_df: pd.DataFrame,
        in_position: bool,
        entry_price: float | None,
        entry_time: pd.Timestamp | None,
    ) -> Signal:
        row = window_df.iloc[-1]
        close = float(row["Close"])
        bb_lower = row.get("BB_Lower")
        bb_middle = row.get("BB_Middle")
        rsi = row.get("RSI")

        if not in_position:
            if pd.notna(bb_lower) and pd.notna(rsi):
                if close < float(bb_lower) and float(rsi) <= self.rsi_oversold:
                    return Signal.BUY
            return Signal.HOLD

        if entry_price is None:
            return Signal.HOLD

        if self.max_loss_pct is not None and close <= entry_price * (1.0 - self.max_loss_pct):
            return Signal.SELL

        if self.partial_take_profit_pct is not None and close >= entry_price * (1.0 + self.partial_take_profit_pct):
            return Signal.SELL

        if pd.notna(bb_middle) and close >= float(bb_middle):
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
        self.log_action(f"Mean reversion signal: {signal}", "info" if signal != Signal.HOLD else "warning")
        self.order_executor.process_signal(asset, signal.value, price, self.risk_manager)
