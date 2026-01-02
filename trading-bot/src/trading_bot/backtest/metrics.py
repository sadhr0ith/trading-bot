"""Backtest performance metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from trading_bot.backtest.interfaces import Trade

_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float | None:
    if index is None or len(index) < 2:
        return None
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return None
    median = float(deltas.median())
    if median <= 0:
        return None
    return _SECONDS_PER_YEAR / median


def _safe_std(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    std = float(np.std(values, ddof=1))
    if std == 0:
        return None
    return std


def total_return(equity_curve: pd.Series) -> float | None:
    if equity_curve.empty:
        return None
    start = float(equity_curve.iloc[0])
    end = float(equity_curve.iloc[-1])
    if start == 0:
        return None
    return end / start - 1.0


def cagr(equity_curve: pd.Series) -> float | None:
    if equity_curve.empty:
        return None
    start = equity_curve.index.min()
    end = equity_curve.index.max()
    if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
        return None
    seconds = (end - start).total_seconds()
    if seconds <= 0:
        return None
    total_ret = total_return(equity_curve)
    if total_ret is None:
        return None
    years = seconds / _SECONDS_PER_YEAR
    if years <= 0:
        return None
    return (1.0 + total_ret) ** (1.0 / years) - 1.0


def max_drawdown(equity_curve: pd.Series) -> float | None:
    if equity_curve.empty:
        return None
    cumulative_max = equity_curve.cummax()
    drawdowns = equity_curve / cumulative_max - 1.0
    return float(drawdowns.min())


def sharpe_ratio(equity_curve: pd.Series) -> float | None:
    if equity_curve.empty:
        return None
    returns = equity_curve.pct_change().dropna().values
    if returns.size < 2:
        return None
    periods_per_year = _infer_periods_per_year(equity_curve.index) or 0.0
    std = _safe_std(returns)
    if std is None or periods_per_year <= 0:
        return None
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def sortino_ratio(equity_curve: pd.Series) -> float | None:
    if equity_curve.empty:
        return None
    returns = equity_curve.pct_change().dropna().values
    if returns.size < 2:
        return None
    negative = returns[returns < 0]
    if negative.size == 0:
        return None
    periods_per_year = _infer_periods_per_year(equity_curve.index) or 0.0
    std = _safe_std(negative)
    if std is None or periods_per_year <= 0:
        return None
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def hit_rate(trades: Iterable[Trade]) -> float | None:
    trades_list = list(trades)
    if not trades_list:
        return None
    wins = sum(1 for trade in trades_list if trade.pnl > 0)
    return wins / len(trades_list)


def avg_trade_return(trades: Iterable[Trade]) -> float | None:
    trades_list = list(trades)
    if not trades_list:
        return None
    return float(np.mean([trade.return_pct for trade in trades_list]))


def avg_trade_pnl(trades: Iterable[Trade]) -> float | None:
    trades_list = list(trades)
    if not trades_list:
        return None
    return float(np.mean([trade.pnl for trade in trades_list]))


def turnover(trades: Iterable[Trade], equity_curve: pd.Series) -> float | None:
    trades_list = list(trades)
    if not trades_list or equity_curve.empty:
        return None
    total_notional = sum(abs(t.entry_value) + abs(t.exit_value) for t in trades_list)
    avg_equity = float(equity_curve.mean()) if not equity_curve.empty else 0.0
    if avg_equity <= 0:
        return None
    return total_notional / avg_equity


def compute_metrics(equity_curve: pd.Series, trades: Iterable[Trade]) -> dict[str, float | None]:
    return {
        "total_return": total_return(equity_curve),
        "cagr": cagr(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "sharpe": sharpe_ratio(equity_curve),
        "sortino": sortino_ratio(equity_curve),
        "hit_rate": hit_rate(trades),
        "avg_trade_return": avg_trade_return(trades),
        "avg_trade_pnl": avg_trade_pnl(trades),
        "turnover": turnover(trades, equity_curve),
    }
