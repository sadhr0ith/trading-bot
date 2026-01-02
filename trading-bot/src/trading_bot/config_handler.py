import importlib
from typing import Any

from pydantic import ValidationError

from trading_bot.models.config import StrategyConfig, parse_strategy_config
from trading_bot.models.env_settings import RuntimeSettings, load_runtime_settings
from trading_bot.utils.logger import get_logger
from trading_bot.utils.time_utils import interval_to_seconds

logger = get_logger(__name__)

_STRATEGY_SLEEP_DEFAULTS: dict[str, int] = {
    "day_trading": 3600,
    "short_term": 86_400,
    "mid_term": 604_800,
    "long_term": 2_592_000,
}


def _resolve_sleep_seconds(
    strategy: str, interval: str | None, configured_sleep: int | None, settings: RuntimeSettings | None
) -> int:
    interval_seconds = interval_to_seconds(interval) or 0
    min_sleep = settings.min_sleep_seconds if settings else 60

    if configured_sleep is not None:
        sleep_seconds = configured_sleep
    elif settings and settings.sleep_seconds is not None:
        sleep_seconds = settings.sleep_seconds
    elif interval_seconds:
        sleep_seconds = max(interval_seconds, min_sleep)
    else:
        sleep_seconds = max(_STRATEGY_SLEEP_DEFAULTS.get(strategy, 3600), min_sleep)

    if interval_seconds:
        sleep_seconds = max(sleep_seconds, interval_seconds, min_sleep)
    return sleep_seconds


def _apply_runtime_defaults(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Enrich raw config with settings-driven defaults."""
    settings = load_runtime_settings(logger)
    cfg = dict(raw_config)

    if settings:
        cfg.setdefault("min_rows", settings.min_rows)
        cfg.setdefault("cache_ttl_seconds", settings.cache_ttl_seconds)
        if settings.sleep_seconds is not None:
            cfg.setdefault("sleep_seconds", settings.sleep_seconds)

    cfg["sleep_seconds"] = _resolve_sleep_seconds(
        cfg.get("strategy", ""),
        cfg.get("interval"),
        cfg.get("sleep_seconds"),
        settings,
    )
    return cfg


def load_config(strategy) -> StrategyConfig | None:
    """Load the appropriate config file based on the strategy."""
    try:
        config_module = importlib.import_module(f"trading_bot.configs.config_{strategy}")
    except ModuleNotFoundError:
        logger.error(f"Configuration for strategy '{strategy}' not found.")
        return None

    raw_config = getattr(config_module, "CONFIG", None)
    if raw_config is None:
        logger.error(f"CONFIG not defined for strategy '{strategy}'.")
        return None

    try:
        config_model = parse_strategy_config(_apply_runtime_defaults(raw_config))
        logger.info(f"Loaded configuration for {config_model.strategy} strategy")
        return config_model
    except ValidationError as exc:
        for err in exc.errors():
            loc = " -> ".join(str(piece) for piece in err.get("loc", ()))
            logger.error(f"Config validation error at {loc}: {err.get('msg')}")
        logger.error("Invalid strategy configuration. Aborting load.")
        return None


def get_sleep_duration(config: StrategyConfig | dict) -> int:
    """Return the validated sleep duration for a strategy config."""
    if config is None:
        return _STRATEGY_SLEEP_DEFAULTS["day_trading"]
    sleep_seconds = config.get("sleep_seconds") if hasattr(config, "get") else getattr(config, "sleep_seconds", None)
    if sleep_seconds is not None:
        return sleep_seconds
    return _resolve_sleep_seconds(
        config.get("strategy") if hasattr(config, "get") else getattr(config, "strategy", ""),
        config.get("interval") if hasattr(config, "get") else getattr(config, "interval", None),
        None,
        load_runtime_settings(logger),
    )
