"""Backtest package for strategy evaluation on historical OHLCV data."""

from trading_bot.backtest.engine import BacktestResult, run_backtest
from trading_bot.backtest.interfaces import BacktestStrategy, Signal, StrategyState, Trade
from trading_bot.backtest.metrics import compute_metrics
from trading_bot.backtest.portfolio import (
    PortfolioConstraints,
    PortfolioResult,
    load_top_volume_universe,
    run_portfolio_backtest,
)
from trading_bot.backtest.report import build_report, print_summary, write_report

__all__ = [
    "BacktestResult",
    "run_backtest",
    "BacktestStrategy",
    "Signal",
    "StrategyState",
    "Trade",
    "compute_metrics",
    "build_report",
    "write_report",
    "print_summary",
    "PortfolioConstraints",
    "PortfolioResult",
    "load_top_volume_universe",
    "run_portfolio_backtest",
]
