"""Reporting utilities for backtest runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.backtest.metrics import compute_metrics


def build_report(
    *,
    strategy: str,
    ticker: str,
    interval: str,
    period: str,
    data: pd.DataFrame,
    initial_cash: float,
    fee_rate: float,
    slippage_rate: float,
    trades: list,
    equity_curve: pd.Series,
) -> dict[str, Any]:
    metrics = compute_metrics(equity_curve, trades)
    start = data.index.min() if not data.empty else None
    end = data.index.max() if not data.empty else None

    report = {
        "strategy": strategy,
        "ticker": ticker,
        "interval": interval,
        "period": period,
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
        "bars": int(len(data)),
        "initial_cash": float(initial_cash),
        "final_equity": float(equity_curve.iloc[-1]) if not equity_curve.empty else None,
        "trades": len(trades),
        "fee_rate": float(fee_rate),
        "slippage_rate": float(slippage_rate),
        "metrics": metrics,
    }

    return report


def write_report(report: dict[str, Any], output_dir: str | Path, filename: str | None = None) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        strategy = report.get("strategy", "strategy")
        ticker = report.get("ticker", "asset")
        interval = report.get("interval", "interval")
        filename = f"backtest_{strategy}_{ticker}_{interval}_{timestamp}.json"

    path = output_path / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    return path


def print_summary(report: dict[str, Any]) -> None:
    metrics = report.get("metrics", {})
    summary = (
        f"Backtest {report.get('strategy')} {report.get('ticker')} {report.get('interval')}\n"
        f"Bars: {report.get('bars')} | Trades: {report.get('trades')}\n"
        f"Total return: {metrics.get('total_return')} | Max DD: {metrics.get('max_drawdown')}\n"
        f"Sharpe: {metrics.get('sharpe')} | Sortino: {metrics.get('sortino')}"
    )
    print(summary)
