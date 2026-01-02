import argparse
import signal
import sys
import time
from datetime import datetime, timedelta

from trading_bot.config_handler import get_sleep_duration, load_config
from trading_bot.core.exceptions import InsufficientDataError, ModelPersistenceError, TradingBotError
from trading_bot.data_fetcher import fetch_data_online
from trading_bot.strategy_manager import select_strategy
from trading_bot.utils.data_health import data_health_report, feature_shift_report
from trading_bot.utils.logger import get_logger
from trading_bot.utils.model_persistence import ModelPersistence
from trading_bot.utils.strategy_helpers import _build_config_signature, build_persistence_key
from trading_bot.utils.validators import ConfigValidator, DataValidator

logger = get_logger(__name__)

# Global shutdown flag
shutdown_requested = False

_ML_STRATEGIES = {"day_trading", "short_term", "mid_term", "long_term", "day_trading_ml"}


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global shutdown_requested
    signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    logger.info(f"Received {signal_name}, initiating graceful shutdown...")
    shutdown_requested = True


def _warn_on_config_drift(strategy_name, config) -> bool:
    """Compare runtime config hash with persisted metadata if available."""
    try:
        config_blob = config.model_dump() if hasattr(config, "model_dump") else config
        current_signature = _build_config_signature(config_blob)
    except (ValueError, TypeError, AttributeError):
        return False

    persistence_key = build_persistence_key(
        strategy=strategy_name,
        data_source=config.get("data_source") if isinstance(config, dict) else getattr(config, "data_source", None),
        ticker=config.get("ticker") if isinstance(config, dict) else getattr(config, "ticker", None),
        interval=config.get("interval") if isinstance(config, dict) else getattr(config, "interval", None),
    )
    metadata = ModelPersistence().load_metadata(persistence_key)
    if not metadata:
        return False

    persisted_signature = metadata.get("config_signature")
    if persisted_signature and persisted_signature != current_signature:
        logger.warning("Config drift detected: runtime config differs from persisted model metadata.")
        return True
    return False


def _ensure_inference_only(config, strategy: str):
    if strategy not in _ML_STRATEGIES:
        return config
    if config.get("inference_only"):
        return config
    logger.info("Enabling inference_only for ML strategy; use `python -m trading_bot.train` to retrain.")
    if hasattr(config, "model_copy"):
        return config.model_copy(update={"inference_only": True})
    if isinstance(config, dict):
        config["inference_only"] = True
        return config
    try:
        setattr(config, "inference_only", True)
    except AttributeError:
        pass
    return config


def run_trading_bot(strategy: str):
    """
    Run the trading bot loop for the given strategy: load config, validate, fetch data, execute strategy with backoff.

    Supports graceful shutdown via SIGTERM/SIGINT.
    """
    global shutdown_requested
    global logger

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    config = load_config(strategy)
    if config is None:
        logger.error("Invalid strategy or configuration. Exiting.")
        return

    config_result = ConfigValidator().validate(config)
    if not config_result.is_valid:
        logger.error("Configuration validation failed; aborting.")
        return

    config = _ensure_inference_only(config, strategy)

    drift_detected = _warn_on_config_drift(strategy, config)
    if drift_detected and config.get("force_retrain_on_drift"):
        persistence_key = build_persistence_key(
            strategy=strategy,
            data_source=config.get("data_source"),
            ticker=config.get("ticker"),
            interval=config.get("interval"),
        )
        ModelPersistence().purge(persistence_key)
        logger.warning("Purged persisted model due to config drift.")

    # Respect log level configured for the strategy
    logger = get_logger(__name__, config.get("log_level"))

    asset = config["ticker"]
    logger.info(f"Running {strategy} strategy for asset {asset} at {datetime.now()}")

    base_backoff_seconds = 60
    max_backoff_seconds = 1800
    current_backoff_seconds = base_backoff_seconds

    while not shutdown_requested:
        try:
            logger.info(f"Fetching data for asset {asset} with strategy {strategy}")
            data = fetch_data_online(
                source=config["data_source"],
                ticker=config["ticker"],
                period=config.get("period", "1y"),
                interval=config.get("interval", "1d"),
                cache_ttl_seconds=config.get("cache_ttl_seconds"),
            )

            min_rows = config.get("min_rows", 50) or 50
            validation = DataValidator(
                min_rows=min_rows,
                expected_interval=config.get("interval"),
                assume_normalized=True,
                drop_nonpositive_volume=config.get("drop_nonpositive_volume", True),
            ).validate(data)
            if not validation.is_valid or validation.data is None:
                logger.warning("Data validation failed or returned empty dataset; skipping strategy execution.")
                sleep_duration = current_backoff_seconds
                logger.info(f"Sleeping for {timedelta(seconds=sleep_duration)} before retrying.")
                time.sleep(sleep_duration)
                current_backoff_seconds = min(current_backoff_seconds * 2, max_backoff_seconds)
                continue
            data = validation.data

            health = data_health_report(data, config.get("interval"))
            if health.get("status") == "ok":
                logger.info(f"Data health: {health}")
            else:
                logger.warning(f"Data health: {health}")

            drift = feature_shift_report(data)
            if drift:
                logger.info(f"Feature shift: {drift}")

            persistence_key = build_persistence_key(
                strategy=strategy,
                data_source=config.get("data_source"),
                ticker=config.get("ticker"),
                interval=config.get("interval"),
            )
            metadata = ModelPersistence().load_metadata(persistence_key) or {}
            baseline_mae = metadata.get("baseline_mae")
            cv_mae = metadata.get("cv_mae")
            if baseline_mae is not None and cv_mae is not None and cv_mae >= baseline_mae:
                logger.warning(
                    "Performance decay detected: cv_mae %.6f >= baseline_mae %.6f", cv_mae, baseline_mae
                )

            current_backoff_seconds = base_backoff_seconds
            strategy_instance = select_strategy(config, data)
            strategy_instance.execute()

        except InsufficientDataError as exc:
            logger.warning(f"Insufficient data for strategy execution: {exc}")
            logger.info(f"Sleeping for {timedelta(seconds=current_backoff_seconds)} before retrying with more data.")
            time.sleep(current_backoff_seconds)
            current_backoff_seconds = min(current_backoff_seconds * 2, max_backoff_seconds)
            continue

        except ModelPersistenceError as exc:
            logger.error(f"Model persistence error: {exc}")
            logger.info(f"Sleeping for {timedelta(seconds=current_backoff_seconds)} before retrying.")
            time.sleep(current_backoff_seconds)
            current_backoff_seconds = min(current_backoff_seconds * 2, max_backoff_seconds)
            continue

        except TradingBotError as exc:
            logger.error(f"Trading bot error: {exc}")
            logger.info(f"Sleeping for {timedelta(seconds=current_backoff_seconds)} before retrying.")
            time.sleep(current_backoff_seconds)
            current_backoff_seconds = min(current_backoff_seconds * 2, max_backoff_seconds)
            continue

        except Exception as exc:  # Intentional catch-all for main loop resilience
            logger.error(f"Unexpected error in main loop: {exc}", exc_info=True)
            logger.info(f"Sleeping for {timedelta(seconds=current_backoff_seconds)} before retrying.")
            time.sleep(current_backoff_seconds)
            current_backoff_seconds = min(current_backoff_seconds * 2, max_backoff_seconds)
            continue

        sleep_duration = get_sleep_duration(config)
        logger.info(
            f"Sleeping for {timedelta(seconds=sleep_duration)} before fetching data again for {asset} using {strategy}."
        )

        # Sleep in chunks to allow faster response to shutdown signal
        sleep_chunk = 60  # Wake up every 60 seconds to check shutdown flag
        elapsed = 0
        while elapsed < sleep_duration and not shutdown_requested:
            chunk_duration = min(sleep_chunk, sleep_duration - elapsed)
            time.sleep(chunk_duration)
            elapsed += chunk_duration

    logger.info("Graceful shutdown completed. Exiting trading bot.")
    sys.exit(0)


if __name__ == "__main__":
    from trading_bot.models.config import ALLOWED_STRATEGIES

    parser = argparse.ArgumentParser(description="Run the trading bot with the specified strategy.")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=sorted(ALLOWED_STRATEGIES),
        help="Strategy to run (see configs/ for options).",
    )
    args = parser.parse_args()

    run_trading_bot(args.strategy)
