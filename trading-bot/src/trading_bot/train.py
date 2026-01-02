"""Training entry point for ML strategies."""

from __future__ import annotations

import argparse

from pydantic import ValidationError

from trading_bot.config_handler import load_config
from trading_bot.data_fetcher import fetch_data_online
from trading_bot.models.config import parse_strategy_config
from trading_bot.strategy_manager import select_strategy
from trading_bot.utils.data_health import data_health_report, feature_shift_report
from trading_bot.utils.experiment_tracking import dataset_hash, write_experiment_report
from trading_bot.utils.logger import get_logger
from trading_bot.utils.model_persistence import ModelPersistence
from trading_bot.utils.strategy_helpers import _build_config_signature, build_persistence_key

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a strategy and persist artifacts.")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--interval", default=None)
    parser.add_argument("--period", default=None)
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--report-dir", default="reports")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_config = load_config(args.strategy)
    if base_config is None:
        logger.error("Strategy config not found: %s", args.strategy)
        return 1

    config_dict = base_config.model_dump() if hasattr(base_config, "model_dump") else dict(base_config)

    if args.ticker:
        config_dict["ticker"] = args.ticker
    if args.interval:
        config_dict["interval"] = args.interval
    if args.period:
        config_dict["period"] = args.period
    if args.data_source:
        config_dict["data_source"] = args.data_source

    config_dict["inference_only"] = False

    try:
        config = parse_strategy_config(config_dict)
    except ValidationError as exc:
        logger.error("Invalid training config: %s", exc)
        return 1

    data = fetch_data_online(
        source=config.get("data_source"),
        ticker=config.get("ticker"),
        period=config.get("period"),
        interval=config.get("interval"),
        cache_ttl_seconds=0,
    )
    if data is None or data.empty:
        logger.error("No data available for training.")
        return 1

    strategy = select_strategy(config, data)
    strategy.execute()

    persistence_key = build_persistence_key(
        strategy=config.get("strategy"),
        data_source=config.get("data_source"),
        ticker=config.get("ticker"),
        interval=config.get("interval"),
    )
    metadata = ModelPersistence().load_metadata(persistence_key) or {}

    config_signature = _build_config_signature(config_dict)
    report = {
        "strategy": config.get("strategy"),
        "ticker": config.get("ticker"),
        "interval": config.get("interval"),
        "period": config.get("period"),
        "data_source": config.get("data_source"),
        "config_signature": config_signature,
        "dataset_hash": dataset_hash(data),
        "horizon": config.get("return_horizon"),
        "costs": {
            "fee": config.get("risk_management").trading_fee if hasattr(config, "risk_management") else None,
            "slippage": config.get("slippage_rate"),
        },
        "metrics": {
            "cv_mae": metadata.get("cv_mae"),
            "baseline_mae": metadata.get("baseline_mae"),
            "hit_rate": metadata.get("hit_rate"),
            "pnl_proxy": metadata.get("pnl_proxy"),
            "max_drawdown_proxy": metadata.get("max_drawdown_proxy"),
        },
        "feature_version": metadata.get("fe_version"),
        "feature_columns": metadata.get("feature_columns"),
        "data_health": data_health_report(data, config.get("interval")),
        "feature_shift": feature_shift_report(data),
    }

    report_path = write_experiment_report(report, args.report_dir, prefix="train")
    logger.info("Training report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
