import argparse
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from trading_bot.config_handler import get_sleep_duration, load_config
from trading_bot.core.exceptions import InsufficientDataError, ModelPersistenceError, TradingBotError
from trading_bot.data_fetcher import fetch_data_online
from trading_bot.interfaces.signal_source import InProcessSignalSource, MarketContext
from trading_bot.models.signal import SignalAction
from trading_bot.strategy_manager import select_strategy
from trading_bot.utils.data_health import data_health_report, feature_shift_report
from trading_bot.utils.logger import get_logger
from trading_bot.utils.model_persistence import ModelPersistence
from trading_bot.utils.paper_trading import PaperTradingExecutor
from trading_bot.utils.risk_management import RiskManager
from trading_bot.utils.signal_aggregator import SignalAggregator
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


def run_multi_strategy_bot(strategies: list[str]):
    """
    Run multiple strategies in a single process with a shared portfolio.

    Each strategy generates signals independently, then a SignalAggregator
    combines them into one final decision per symbol. A single PaperTradingExecutor
    executes trades and maintains a shared state file.
    """
    global shutdown_requested
    global logger

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    if not strategies:
        logger.error("No strategies provided for multi-strategy mode.")
        return

    logger.info(f"Starting multi-strategy bot with strategies: {strategies}")

    # Load and validate configs for all strategies
    configs = {}
    for strategy in strategies:
        config = load_config(strategy)
        if config is None:
            logger.error(f"Failed to load config for strategy '{strategy}'. Aborting.")
            return

        config_result = ConfigValidator().validate(config)
        if not config_result.is_valid:
            logger.error(f"Configuration validation failed for '{strategy}'. Aborting.")
            return

        config = _ensure_inference_only(config, strategy)
        configs[strategy] = config

    # Use the first strategy's config as the "primary" for shared settings
    primary_config = configs[strategies[0]]
    primary_asset = primary_config["ticker"]
    logger = get_logger(__name__, primary_config.get("log_level"))

    # Create a shared PaperTradingExecutor with a unified state file
    shared_state_path = Path("paper_trading_state.json")
    shared_executor = PaperTradingExecutor(
        initial_balance=float(primary_config.get("initial_balance", 10000.0)),
        state_path=str(shared_state_path),
    )
    logger.info(f"Using shared paper trading state file: {shared_state_path}")

    # Create a shared RiskManager from primary config's risk_management section
    risk_config = primary_config.get("risk_management", {})
    if isinstance(risk_config, dict):
        risk_dict = {
            "stop_loss": risk_config.get("stop_loss", 0.03),
            "take_profit": risk_config.get("take_profit", 0.05),
            "max_position_size": risk_config.get("max_position_size", 0.1),
            "trading_fee": risk_config.get("trading_fee", 0.001),
            "trailing_stop": risk_config.get("trailing_stop"),
        }
    else:
        risk_dict = {
            "stop_loss": 0.03,
            "take_profit": 0.05,
            "max_position_size": 0.1,
            "trading_fee": 0.001,
        }
    shared_risk_manager = RiskManager(risk_config=risk_dict)

    # Create SignalAggregator
    aggregator = SignalAggregator(
        enable_long_term_filter="long_term" in strategies,
        enable_mid_term_bias="mid_term" in strategies,
    )

    logger.info(f"Running multi-strategy mode for asset {primary_asset} at {datetime.now()}")

    base_backoff_seconds = 60
    max_backoff_seconds = 1800
    current_backoff_seconds = base_backoff_seconds

    while not shutdown_requested:
        try:
            # Fetch data (use primary config for data fetching parameters)
            logger.info(f"Fetching data for asset {primary_asset} (multi-strategy mode)")
            data = fetch_data_online(
                source=primary_config["data_source"],
                ticker=primary_config["ticker"],
                period=primary_config.get("period", "1y"),
                interval=primary_config.get("interval", "1d"),
                cache_ttl_seconds=primary_config.get("cache_ttl_seconds"),
            )

            min_rows = primary_config.get("min_rows", 50) or 50
            validation = DataValidator(
                min_rows=min_rows,
                expected_interval=primary_config.get("interval"),
                assume_normalized=True,
                drop_nonpositive_volume=primary_config.get("drop_nonpositive_volume", True),
            ).validate(data)

            if not validation.is_valid or validation.data is None:
                logger.warning("Data validation failed; skipping strategy execution.")
                sleep_duration = current_backoff_seconds
                logger.info(f"Sleeping for {timedelta(seconds=sleep_duration)} before retrying.")
                time.sleep(sleep_duration)
                current_backoff_seconds = min(current_backoff_seconds * 2, max_backoff_seconds)
                continue
            data = validation.data

            health = data_health_report(data, primary_config.get("interval"))
            if health.get("status") == "ok":
                logger.info(f"Data health: {health}")
            else:
                logger.warning(f"Data health: {health}")

            current_backoff_seconds = base_backoff_seconds

            # Initialize strategy instances with shared executor
            strategy_instances = []
            for strategy_name in strategies:
                config = configs[strategy_name]
                strategy_instance = select_strategy(config, data)
                # Point the strategy's executor to the shared one
                strategy_instance.order_executor = shared_executor
                strategy_instance.risk_manager = shared_risk_manager
                strategy_instances.append(strategy_instance)

            # Create signal source
            signal_source = InProcessSignalSource(strategy_instances)

            # Create market context
            market_context = MarketContext(
                symbol=primary_asset,
                data=data,
                current_price=float(data["Close"].iloc[-1]),
                timeframe=primary_config.get("interval"),
            )

            # Collect signals from all strategies
            signals = signal_source.collect(market_context)

            # Log individual signals
            for sig in signals:
                logger.info(
                    f"[{sig.symbol}] {sig.strategy_name}: {sig.action.value} "
                    f"(confidence={sig.confidence}, edge={sig.edge})"
                )

            # Position check callback for position-aware aggregation
            def has_position_for_symbol(symbol: str) -> bool:
                return shared_executor._current_position(symbol) is not None

            # Aggregate signals (position-aware: long_term SELL -> EXIT if position exists)
            aggregated_decisions = aggregator.aggregate(signals, has_position=has_position_for_symbol)
            aggregator.log_aggregation(aggregated_decisions)

            # Execute aggregated decisions
            for decision in aggregated_decisions:
                if decision.final_action in (SignalAction.HOLD,):
                    logger.info(f"[{decision.symbol}] Final action: HOLD - no trade executed")
                    continue

                # Map SignalAction to string for executor
                action_str = decision.final_action.value
                if decision.final_action in (SignalAction.EXIT, SignalAction.RISK_EXIT):
                    action_str = "SELL"  # Exit signals become sells

                price = float(data["Close"].iloc[-1])
                trade_summary = shared_executor.process_signal(
                    decision.symbol,
                    action_str,
                    price,
                    shared_risk_manager,
                    votes=[v.to_dict() for v in decision.votes],
                    aggregation_reason=decision.reason,
                )

                if trade_summary.get("status") not in {"noop", "already_long"}:
                    logger.info(f"[{decision.symbol}] Trade executed: {trade_summary}")

        except InsufficientDataError as exc:
            logger.warning(f"Insufficient data: {exc}")
            time.sleep(current_backoff_seconds)
            current_backoff_seconds = min(current_backoff_seconds * 2, max_backoff_seconds)
            continue

        except ModelPersistenceError as exc:
            logger.error(f"Model persistence error: {exc}")
            time.sleep(current_backoff_seconds)
            current_backoff_seconds = min(current_backoff_seconds * 2, max_backoff_seconds)
            continue

        except TradingBotError as exc:
            logger.error(f"Trading bot error: {exc}")
            time.sleep(current_backoff_seconds)
            current_backoff_seconds = min(current_backoff_seconds * 2, max_backoff_seconds)
            continue

        except Exception as exc:
            logger.error(f"Unexpected error in multi-strategy loop: {exc}", exc_info=True)
            time.sleep(current_backoff_seconds)
            current_backoff_seconds = min(current_backoff_seconds * 2, max_backoff_seconds)
            continue

        # Sleep between cycles
        sleep_duration = get_sleep_duration(primary_config)
        logger.info(f"Sleeping for {timedelta(seconds=sleep_duration)} before next cycle.")

        elapsed = 0
        sleep_chunk = 60
        while elapsed < sleep_duration and not shutdown_requested:
            chunk_duration = min(sleep_chunk, sleep_duration - elapsed)
            time.sleep(chunk_duration)
            elapsed += chunk_duration

    logger.info("Graceful shutdown completed. Exiting multi-strategy bot.")
    sys.exit(0)


if __name__ == "__main__":
    from trading_bot.models.config import ALLOWED_STRATEGIES

    parser = argparse.ArgumentParser(description="Run the trading bot with the specified strategy.")

    # Mutually exclusive: either --strategy (single) or --strategies (multi)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--strategy",
        type=str,
        choices=sorted(ALLOWED_STRATEGIES),
        help="Single strategy to run (see configs/ for options).",
    )
    group.add_argument(
        "--strategies",
        type=str,
        help="Comma-separated list of strategies for multi-strategy mode (e.g., 'long_term,mid_term,atr_breakout').",
    )

    args = parser.parse_args()

    if args.strategy:
        # Single-strategy mode (backward compatible)
        run_trading_bot(args.strategy)
    else:
        # Multi-strategy mode
        strategy_list = [s.strip() for s in args.strategies.split(",") if s.strip()]

        # Validate all strategies
        invalid = [s for s in strategy_list if s not in ALLOWED_STRATEGIES]
        if invalid:
            parser.error(f"Invalid strategies: {invalid}. Allowed: {sorted(ALLOWED_STRATEGIES)}")

        if len(strategy_list) < 2:
            parser.error("Multi-strategy mode requires at least 2 strategies. Use --strategy for single strategy.")

        run_multi_strategy_bot(strategy_list)
