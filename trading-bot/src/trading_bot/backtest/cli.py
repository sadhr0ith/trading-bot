"""Command line interface for backtesting."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from trading_bot.backtest.engine import run_backtest
from trading_bot.backtest.report import build_report, print_summary, write_report
from trading_bot.data_fetcher import fetch_data_online
from trading_bot.models.config import parse_strategy_config
from trading_bot.utils.logger import setup_logger

logger = setup_logger("BacktestCLI")

_STRATEGY_REGISTRY: dict[str, str] = {
    "atr_breakout": "trading_bot.strategies.atr_breakout_strategy.ATRBreakoutStrategy",
    "mean_reversion": "trading_bot.strategies.mean_reversion_strategy.MeanReversionStrategy",
    "regime_switch": "trading_bot.strategies.regime_switch_strategy.RegimeSwitchStrategy",
}


def _import_strategy(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _load_config(strategy: str) -> dict:
    try:
        module = importlib.import_module(f"trading_bot.configs.config_{strategy}")
    except ModuleNotFoundError:
        return {}
    return getattr(module, "CONFIG", {}) or {}


def _load_backtest_defaults() -> dict:
    try:
        module = importlib.import_module("trading_bot.configs.config_backtest_defaults")
    except ModuleNotFoundError:
        return {}
    return getattr(module, "CONFIG", {}) or {}


def _load_cached_frame(source: str, ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        from trading_bot import data_fetcher

        cache_path = data_fetcher._cache_paths(source, ticker, period, interval)
        if cache_path.exists():
            return pd.read_pickle(cache_path, compression="gzip")
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def parse_args() -> argparse.Namespace:
    defaults = _load_backtest_defaults()
    parser = argparse.ArgumentParser(description="Run a backtest on cached OHLCV data.")
    parser.add_argument("--strategy", required=True, choices=sorted(_STRATEGY_REGISTRY))
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--interval", default=None)
    parser.add_argument("--period", default=None)
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--initial-cash", type=float, default=defaults.get("initial_cash", 10_000.0))
    parser.add_argument("--fee", type=float, default=defaults.get("fee", 0.001))
    parser.add_argument("--slippage", type=float, default=defaults.get("slippage", 0.0002))
    parser.add_argument("--report-dir", default=defaults.get("report_dir", "reports"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_config = _load_config(args.strategy)

    ticker = args.ticker or raw_config.get("ticker", "BTCUSDT")
    interval = args.interval or raw_config.get("interval", "1h")
    period = args.period or raw_config.get("period", "1y")
    data_source = args.data_source or raw_config.get("data_source", "binance")

    if not raw_config:
        raw_config = {
            "strategy": args.strategy,
            "data_source": data_source,
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "indicators": [],
            "use_indicators": False,
        }

    raw_config.update({"ticker": ticker, "interval": interval, "period": period, "data_source": data_source})

    try:
        config = parse_strategy_config(raw_config)
    except ValidationError as exc:
        logger.error("Invalid backtest config: %s", exc)
        return 1

    logger.info(
        "Starting backtest %s %s %s %s",
        args.strategy,
        ticker,
        interval,
        period,
    )

    data = fetch_data_online(
        source=data_source,
        ticker=ticker,
        period=period,
        interval=interval,
        cache_ttl_seconds=0,
    )

    if data.empty:
        cached = _load_cached_frame(data_source, ticker, period, interval)
        if cached.empty:
            logger.error("No cached data available. Run once online to cache data.")
            return 1
        logger.warning("Fetch failed or returned empty data; using cached frame.")
        data = cached

    strategy_cls = _import_strategy(_STRATEGY_REGISTRY[args.strategy])
    strategy = strategy_cls(config, data)

    result = run_backtest(
        data,
        strategy,
        initial_cash=args.initial_cash,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
    )

    report = build_report(
        strategy=args.strategy,
        ticker=ticker,
        interval=interval,
        period=period,
        data=data,
        initial_cash=args.initial_cash,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        trades=result.trades,
        equity_curve=result.equity_curve,
    )
    report_path = write_report(report, Path(args.report_dir))
    print_summary(report)
    logger.info("Backtest report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
