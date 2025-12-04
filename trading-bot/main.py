import argparse
import time
from datetime import datetime, timedelta

from config_handler import get_sleep_duration, load_config
from core.exceptions import InsufficientDataError, ModelPersistenceError, TradingBotError
from data_fetcher import fetch_data_online
from strategy_manager import select_strategy
from utils.logger import setup_logger
from utils.model_persistence import ModelPersistence
from utils.strategy_helpers import _build_config_signature
from utils.validators import ConfigValidator, DataValidator

logger = setup_logger("TradingBot")


def _warn_on_config_drift(strategy_name, config):
    """Compare runtime config hash with persisted metadata if available."""
    try:
        config_blob = config.model_dump() if hasattr(config, "model_dump") else config
        current_signature = _build_config_signature(config_blob)
    except Exception:  # noqa: BLE001
        return

    try:
        artifact = ModelPersistence().load(strategy_name)
    except Exception:  # noqa: BLE001
        return

    if not artifact:
        return

    persisted_signature = (artifact.get("metadata") or {}).get("config_signature")
    if persisted_signature and persisted_signature != current_signature:
        logger.warning("Config drift detected: runtime config differs from persisted model metadata.")


def run_trading_bot(strategy: str):
    """
    Run the trading bot loop for the given strategy: load config, validate, fetch data, execute strategy with backoff.
    """
    config = load_config(strategy)
    if config is None:
        logger.error("Invalid strategy or configuration. Exiting.")
        return

    config_result = ConfigValidator().validate(config)
    if not config_result.is_valid:
        logger.error("Configuration validation failed; aborting.")
        return

    _warn_on_config_drift(strategy, config)

    # Respect log level configured for the strategy
    global logger
    logger = setup_logger("TradingBot", config.get("log_level"))

    asset = config["ticker"]
    logger.info(f"Running {strategy} strategy for asset {asset} at {datetime.now()}")

    base_backoff_seconds = 60
    max_backoff_seconds = 1800
    current_backoff_seconds = base_backoff_seconds

    while True:
        try:
            logger.info(f"Fetching data for asset {asset} with strategy {strategy}")
            data = fetch_data_online(
                source=config["data_source"],
                ticker=config["ticker"],
                period=config.get("period", "1y"),
                interval=config.get("interval", "1d"),
            )

            validation = DataValidator(min_rows=config.get("min_rows", 50)).validate(data)
            if not validation.is_valid or validation.data is None:
                logger.warning("Data validation failed or returned empty dataset; skipping strategy execution.")
                sleep_duration = current_backoff_seconds
                logger.info(f"Sleeping for {timedelta(seconds=sleep_duration)} before retrying.")
                time.sleep(sleep_duration)
                current_backoff_seconds = min(current_backoff_seconds * 2, max_backoff_seconds)
                continue
            data = validation.data

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

        except Exception as exc:  # noqa: BLE001
            logger.error(f"Unexpected error in main loop: {exc}", exc_info=True)
            logger.info(f"Sleeping for {timedelta(seconds=current_backoff_seconds)} before retrying.")
            time.sleep(current_backoff_seconds)
            current_backoff_seconds = min(current_backoff_seconds * 2, max_backoff_seconds)
            continue

        sleep_duration = get_sleep_duration(strategy)
        logger.info(
            f"Sleeping for {timedelta(seconds=sleep_duration)} before fetching data again for {asset} using {strategy}."
        )
        time.sleep(sleep_duration)


if __name__ == "__main__":
    from models.config import ALLOWED_STRATEGIES

    parser = argparse.ArgumentParser(description="Run the trading bot with the specified strategy.")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=sorted(ALLOWED_STRATEGIES),
        help="Strategy to run: 'day_trading', 'short_term', 'mid_term', 'long_term'",
    )
    args = parser.parse_args()

    run_trading_bot(args.strategy)
