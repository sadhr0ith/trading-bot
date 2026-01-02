"""Lightweight proxy metrics for ML trading signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def long_only_proxy(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    *,
    threshold: float,
    cost_per_side: float = 0.0,
) -> dict[str, float | int | None]:
    """Compute a long-only proxy equity curve from 1-step forward returns.

    The proxy assumes:
    - Position is 1 when y_pred > threshold, otherwise 0 (flat).
    - Per-side costs are paid on position changes (entry/exit).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = min(len(y_true_arr), len(y_pred_arr))
    if n == 0:
        return {"hit_rate": None, "pnl_proxy": None, "max_drawdown": None, "entries": 0, "exits": 0}

    y_true_arr = y_true_arr[:n]
    y_pred_arr = y_pred_arr[:n]
    y_true_arr = np.nan_to_num(y_true_arr, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred_arr = np.nan_to_num(y_pred_arr, nan=0.0, posinf=0.0, neginf=0.0)

    positions = (y_pred_arr > threshold).astype(int)
    prev_positions = np.concatenate(([0], positions[:-1]))
    changes = np.abs(positions - prev_positions)
    entries = int(((positions == 1) & (prev_positions == 0)).sum())
    exits = int(((positions == 0) & (prev_positions == 1)).sum())

    net_returns = positions * y_true_arr - changes * float(cost_per_side)
    equity = pd.Series((1.0 + net_returns).cumprod())
    if equity.empty:
        return {"hit_rate": None, "pnl_proxy": None, "max_drawdown": None, "entries": entries, "exits": exits}

    pnl_proxy = float(equity.iloc[-1] - 1.0)
    max_drawdown = float((equity / equity.cummax() - 1.0).min())

    traded = positions == 1
    if not np.any(traded):
        hit_rate = None
    else:
        hit_rate = float(np.mean(y_true_arr[traded] > 0))

    return {
        "hit_rate": hit_rate,
        "pnl_proxy": pnl_proxy,
        "max_drawdown": max_drawdown,
        "entries": entries,
        "exits": exits,
    }

