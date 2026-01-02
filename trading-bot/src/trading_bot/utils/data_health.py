"""Data health and drift reporting utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_bot.utils.time_utils import interval_to_timedelta


def data_health_report(data: pd.DataFrame, interval: str | None) -> dict[str, Any]:
    """Return basic data health metrics for an OHLCV frame."""
    if data is None or data.empty:
        return {"status": "empty"}
    report: dict[str, Any] = {"status": "ok", "rows": int(len(data))}
    index = data.index

    expected_delta = interval_to_timedelta(interval)
    if isinstance(index, pd.DatetimeIndex) and expected_delta is not None:
        gaps = index.to_series().diff().dropna()
        expected_seconds = expected_delta.total_seconds()
        if expected_seconds > 0:
            missing = (gaps / expected_delta - 1).clip(lower=0)
            report["missing_bars_estimate"] = int(missing.sum())
            report["gap_count"] = int((gaps > expected_delta * 1.5).sum())
            report["max_gap_seconds"] = float(gaps.max().total_seconds()) if not gaps.empty else 0.0

    returns = data["Close"].pct_change().dropna()
    if not returns.empty:
        vol = returns.rolling(20).std().dropna()
        if not vol.empty:
            current_vol = float(vol.iloc[-1])
            percentile = float((vol < current_vol).mean())
            if percentile < 0.33:
                regime = "low"
            elif percentile > 0.67:
                regime = "high"
            else:
                regime = "mid"
            report.update({"volatility": current_vol, "volatility_regime": regime, "volatility_percentile": percentile})

    return report


def feature_shift_report(data: pd.DataFrame, window: int = 200) -> dict[str, float] | None:
    """Compare recent vs baseline return distributions to flag drift."""
    if data is None or data.empty or "Close" not in data.columns:
        return None
    returns = data["Close"].pct_change().dropna()
    if len(returns) < window * 2:
        return None
    baseline = returns.iloc[:window]
    recent = returns.iloc[-window:]
    return {
        "return_mean_shift": float(recent.mean() - baseline.mean()),
        "return_vol_shift": float(recent.std() - baseline.std()),
    }
