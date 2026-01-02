"""Command line interface for portfolio backtesting."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Iterable

import pandas as pd
from pydantic import ValidationError

from trading_bot.backtest.portfolio import PortfolioConstraints, load_top_volume_universe, run_portfolio_backtest
from trading_bot.backtest.report import write_report
from trading_bot.data_fetcher import fetch_data_online
from trading_bot.models.config import parse_strategy_config
from trading_bot.utils.logger import setup_logger

logger = setup_logger("PortfolioBacktestCLI")

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


def _load_cached_frame(source: str, ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        from trading_bot import data_fetcher

        cache_path = data_fetcher._cache_paths(source, ticker, period, interval)
        if cache_path.exists():
            return pd.read_pickle(cache_path, compression="gzip")
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _parse_tickers(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-asset portfolio backtest.")
    parser.add_argument("--strategy", required=True, choices=sorted(_STRATEGY_REGISTRY))
    parser.add_argument("--tickers", default=None, help="Comma-separated list of tickers (overrides universe).")
    parser.add_argument("--universe-limit", type=int, default=10)
    parser.add_argument("--universe-quote", default="USDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--data-source", default="binance")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-exposure-per-asset", type=float, default=0.2)
    parser.add_argument("--vol-targeting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--correlation-filter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--correlation-window", type=int, default=48)
    parser.add_argument("--max-pairwise-correlation", type=float, default=0.9)
    parser.add_argument("--report-dir", default="reports")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    tickers = _parse_tickers(args.tickers)
    if not tickers:
        tickers = load_top_volume_universe(
            limit=args.universe_limit,
            quote_asset=args.universe_quote,
        )
        if not tickers:
            logger.error("Universe is empty; provide --tickers or ensure universe cache is available.")
            return 1

    raw_config = _load_config(args.strategy)
    if not raw_config:
        raw_config = {
            "strategy": args.strategy,
            "data_source": args.data_source,
            "ticker": tickers[0],
            "period": args.period,
            "interval": args.interval,
            "indicators": [],
            "use_indicators": False,
            "notification_email": None,
        }
    raw_config.update(
        {
            "strategy": args.strategy,
            "data_source": args.data_source,
            "ticker": tickers[0],
            "period": args.period,
            "interval": args.interval,
        }
    )
    try:
        base_config = parse_strategy_config(raw_config)
    except ValidationError as exc:
        logger.error("Invalid portfolio backtest config: %s", exc)
        return 1

    data_by_asset: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = fetch_data_online(
            source=args.data_source,
            ticker=ticker,
            period=args.period,
            interval=args.interval,
            cache_ttl_seconds=0,
        )
        if df.empty:
            cached = _load_cached_frame(args.data_source, ticker, args.period, args.interval)
            if cached.empty:
                logger.warning("No data for %s; skipping.", ticker)
                continue
            logger.warning("Fetch failed or returned empty data for %s; using cached frame.", ticker)
            df = cached
        data_by_asset[ticker] = df

    if not data_by_asset:
        logger.error("No data available for any ticker; aborting.")
        return 1

    strategy_cls = _import_strategy(_STRATEGY_REGISTRY[args.strategy])

    def factory(ticker: str, df: pd.DataFrame):
        cfg = base_config.model_copy(update={"ticker": ticker}) if hasattr(base_config, "model_copy") else dict(base_config)
        if isinstance(cfg, dict):
            cfg["ticker"] = ticker
        return strategy_cls(cfg, df)

    constraints = PortfolioConstraints(
        max_positions=args.max_positions,
        max_exposure_per_asset=args.max_exposure_per_asset,
        vol_targeting=args.vol_targeting,
        correlation_filter=args.correlation_filter,
        correlation_window=args.correlation_window,
        max_pairwise_correlation=args.max_pairwise_correlation,
    )

    result = run_portfolio_backtest(
        data_by_asset,
        factory,
        initial_cash=args.initial_cash,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        constraints=constraints,
    )

    report = {
        "strategy": args.strategy,
        "ticker": "portfolio",
        "tickers": sorted(data_by_asset),
        "interval": args.interval,
        "period": args.period,
        "data_source": args.data_source,
        "initial_cash": float(args.initial_cash),
        "fee_rate": float(args.fee),
        "slippage_rate": float(args.slippage),
        "constraints": {
            "max_positions": args.max_positions,
            "max_exposure_per_asset": args.max_exposure_per_asset,
            "vol_targeting": args.vol_targeting,
            "correlation_filter": args.correlation_filter,
            "correlation_window": args.correlation_window,
            "max_pairwise_correlation": args.max_pairwise_correlation,
        },
        "bars_by_asset": {ticker: int(len(df)) for ticker, df in data_by_asset.items()},
        "trades": len(result.trades),
        "metrics": result.metrics,
    }

    output_path = write_report(report, Path(args.report_dir), filename=None)
    logger.info("Portfolio backtest report written to %s", output_path)
    print(
        f"Portfolio backtest {args.strategy} ({len(data_by_asset)} assets)\n"
        f"Trades: {len(result.trades)} | Total return: {result.metrics.get('total_return')} | "
        f"Max DD: {result.metrics.get('max_drawdown')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
