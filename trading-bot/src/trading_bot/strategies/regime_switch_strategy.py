from __future__ import annotations

import pandas as pd

from trading_bot.backtest.adapters import compute_atr, compute_donchian_channels
from trading_bot.backtest.interfaces import Signal, StrategyState
from trading_bot.indicators.adx import ADX
from trading_bot.indicators.bollinger_bands import BollingerBands
from trading_bot.indicators.rsi import RSI
from trading_bot.strategies.strategy_base import StrategyBase


class RegimeSwitchStrategy(StrategyBase):
    """Regime switching strategy: trend vs range vs high-vol no-trade."""

    MIN_ROWS = 50

    def __init__(self, config, data: pd.DataFrame) -> None:
        super().__init__(config, data)
        self.adx_window = int(config.get("adx_window", 14))
        self.adx_trend_threshold = float(config.get("adx_trend_threshold", 25.0))
        self.volatility_window = int(config.get("volatility_window", 20))
        self.volatility_high_threshold = float(config.get("volatility_high_threshold", 0.05))
        self.cooldown_bars = int(config.get("cooldown_bars", 0))

        self.donchian_window = int(config.get("donchian_window", 20))
        self.atr_window = int(config.get("atr_window", 14))
        self.atr_stop_mult = float(config.get("atr_stop_mult", 2.0))
        self.atr_trail_mult = float(config.get("atr_trail_mult", 3.0))
        self.time_stop_bars = config.get("time_stop_bars")
        if self.time_stop_bars is not None:
            self.time_stop_bars = int(self.time_stop_bars)

        self.bb_window = int(config.get("bb_window", 20))
        self.bb_num_std = float(config.get("bb_num_std", 2.0))
        self.rsi_period = int(config.get("rsi_period", 14))
        self.rsi_oversold = float(config.get("rsi_oversold", 30.0))
        self.max_loss_pct = config.get("max_loss_pct")
        if self.max_loss_pct is not None:
            self.max_loss_pct = float(self.max_loss_pct)
        self.partial_take_profit_pct = config.get("partial_take_profit_pct")
        if self.partial_take_profit_pct is not None:
            self.partial_take_profit_pct = float(self.partial_take_profit_pct)

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        adx = ADX(out, period=self.adx_window).calculate()
        out = out.join(adx[["ADX", "Plus_DI", "Minus_DI"]])
        returns = out["Close"].pct_change()
        out["Volatility"] = returns.rolling(self.volatility_window).std()

        out["ATR"] = compute_atr(out, window=self.atr_window)
        upper, lower = compute_donchian_channels(out, window=self.donchian_window)
        out["DonchianHigh"] = upper.shift(1)
        out["DonchianLow"] = lower.shift(1)

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

    def _max_close_since_entry(self, window_df: pd.DataFrame, entry_time: pd.Timestamp | None) -> float:
        if entry_time is None:
            return float(window_df["Close"].iloc[-1])
        if entry_time in window_df.index:
            return float(window_df.loc[entry_time:, "Close"].max())
        return float(window_df.loc[window_df.index >= entry_time, "Close"].max())

    def _atr_signal(
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

    def _mean_reversion_signal(
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

    def _cooldown_active(self, window_df: pd.DataFrame, last_exit: pd.Timestamp | None) -> bool:
        if self.cooldown_bars <= 0 or last_exit is None:
            return False
        if last_exit in window_df.index:
            bars_since = len(window_df.loc[last_exit:]) - 1
            return bars_since < self.cooldown_bars
        return int((window_df.index >= last_exit).sum()) < self.cooldown_bars

    def _signal(
        self,
        window_df: pd.DataFrame,
        in_position: bool,
        entry_price: float | None,
        entry_time: pd.Timestamp | None,
        last_exit: pd.Timestamp | None,
    ) -> Signal:
        row = window_df.iloc[-1]
        adx = float(row.get("ADX", 0.0))
        volatility = float(row.get("Volatility", 0.0))

        if not in_position and self._cooldown_active(window_df, last_exit):
            return Signal.HOLD

        if volatility >= self.volatility_high_threshold and not in_position:
            return Signal.HOLD

        if adx >= self.adx_trend_threshold:
            return self._atr_signal(window_df, in_position, entry_price, entry_time)
        return self._mean_reversion_signal(window_df, in_position, entry_price, entry_time)

    def on_bar(self, state: StrategyState, window_df: pd.DataFrame) -> Signal:
        last_exit = state.trades[-1].exit_time if state.trades else None
        return self._signal(window_df, state.position > 0, state.entry_price, state.entry_time, last_exit)

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

        last_exit = None
        history = self.order_executor.state.get("history", [])
        for record in reversed(history):
            if record.get("symbol") == asset and record.get("action") == "CLOSE":
                try:
                    last_exit = pd.to_datetime(record.get("timestamp"), utc=True)
                except (ValueError, TypeError):
                    last_exit = None
                break

        signal = self._signal(data, in_position, entry_price, entry_time, last_exit)
        price = float(data["Close"].iloc[-1])
        self.log_action(f"Regime switch signal: {signal}", "info" if signal != Signal.HOLD else "warning")
        self.order_executor.process_signal(asset, signal.value, price, self.risk_manager)
