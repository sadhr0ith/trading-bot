"""Core backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.backtest.costs import fee_from_notional, fill_price, split_notional_for_fee
from trading_bot.backtest.interfaces import BacktestStrategy, Signal, StrategyState, Trade


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    trades: list[Trade]
    equity_curve: pd.Series
    positions: pd.Series
    signals: pd.Series


def _ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.copy()
        data.index = pd.to_datetime(data.index)
    if data.index.tz is None:
        data = data.copy()
        data.index = data.index.tz_localize("UTC")
    return data


def run_backtest(
    data: pd.DataFrame,
    strategy: BacktestStrategy,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0002,
    window_size: int | None = None,
    close_out: bool = True,
) -> BacktestResult:
    """Run a long-only backtest for a given strategy.

    Args:
        data: OHLCV DataFrame with DatetimeIndex in UTC.
        strategy: Strategy implementing on_bar(state, window_df).
        initial_cash: Starting cash balance.
        fee_rate: Per-side fee rate (e.g., 0.001 = 0.10%).
        slippage_rate: Per-side slippage rate (e.g., 0.0002 = 0.02%).
        window_size: Optional rolling window size for strategy inputs.
        close_out: If True, force close any open position on the last bar.

    Returns:
        BacktestResult with trades, equity curve, and positions.
    """
    if data is None or data.empty:
        raise ValueError("Backtest data is empty")
    if "Close" not in data.columns:
        raise ValueError("Backtest data must include a Close column")

    df = data.copy().sort_index()
    df = _ensure_datetime_index(df)

    prepare = getattr(strategy, "prepare_data", None)
    if callable(prepare):
        df = prepare(df)

    state = StrategyState(cash=float(initial_cash), position=0.0, equity=float(initial_cash))

    positions: list[float] = []
    equity_values: list[float] = []
    signals: list[str] = []

    total_bars = len(df)
    for idx, (ts, row) in enumerate(df.iterrows()):
        price = float(row["Close"])
        state.current_time = ts
        state.equity = state.cash + state.position * price

        if window_size is None:
            window_df = df.iloc[: idx + 1]
        else:
            start = max(0, idx + 1 - window_size)
            window_df = df.iloc[start : idx + 1]

        signal = strategy.on_bar(state, window_df)
        state.last_signal = signal

        if signal == Signal.BUY and state.position <= 0:
            notional = split_notional_for_fee(state.cash, fee_rate)
            fill = fill_price(price, "BUY", slippage_rate)
            if fill > 0 and notional > 0:
                size = notional / fill
                fee = fee_from_notional(notional, fee_rate)
                state.cash -= notional + fee
                state.position = size
                state.entry_price = fill
                state.entry_time = ts
                state.entry_fee = fee
        elif signal == Signal.SELL and state.position > 0:
            fill = fill_price(price, "SELL", slippage_rate)
            notional = state.position * fill
            fee = fee_from_notional(notional, fee_rate)
            state.cash += notional - fee
            entry_value = (state.entry_price or 0.0) * state.position
            exit_value = notional
            pnl = exit_value - entry_value - state.entry_fee - fee
            return_pct = pnl / entry_value if entry_value else 0.0
            state.trades.append(
                Trade(
                    entry_time=state.entry_time or ts,
                    exit_time=ts,
                    entry_price=state.entry_price or 0.0,
                    exit_price=fill,
                    size=state.position,
                    entry_fee=state.entry_fee,
                    exit_fee=fee,
                    entry_value=entry_value,
                    exit_value=exit_value,
                    pnl=pnl,
                    return_pct=return_pct,
                )
            )
            state.position = 0.0
            state.entry_price = None
            state.entry_time = None
            state.entry_fee = 0.0

        if close_out and idx == total_bars - 1 and state.position > 0:
            fill = fill_price(price, "SELL", slippage_rate)
            notional = state.position * fill
            fee = fee_from_notional(notional, fee_rate)
            state.cash += notional - fee
            entry_value = (state.entry_price or 0.0) * state.position
            exit_value = notional
            pnl = exit_value - entry_value - state.entry_fee - fee
            return_pct = pnl / entry_value if entry_value else 0.0
            state.trades.append(
                Trade(
                    entry_time=state.entry_time or ts,
                    exit_time=ts,
                    entry_price=state.entry_price or 0.0,
                    exit_price=fill,
                    size=state.position,
                    entry_fee=state.entry_fee,
                    exit_fee=fee,
                    entry_value=entry_value,
                    exit_value=exit_value,
                    pnl=pnl,
                    return_pct=return_pct,
                )
            )
            state.position = 0.0
            state.entry_price = None
            state.entry_time = None
            state.entry_fee = 0.0

        state.equity = state.cash + state.position * price
        positions.append(state.position)
        equity_values.append(state.equity)
        signals.append(signal.value)

    equity_curve = pd.Series(equity_values, index=df.index, name="equity")
    positions_series = pd.Series(positions, index=df.index, name="position")
    signals_series = pd.Series(signals, index=df.index, name="signal")

    return BacktestResult(
        trades=state.trades,
        equity_curve=equity_curve,
        positions=positions_series,
        signals=signals_series,
    )
