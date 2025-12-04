import importlib

from pydantic import ValidationError

from models.config import StrategyConfig, parse_strategy_config
from utils.logger import setup_logger

logger = setup_logger("TradingBot")

def load_config(strategy):
    """
    Load the appropriate config file based on the strategy.
    """
    try:
        config_module = importlib.import_module(f"configs.config_{strategy}")
    except ModuleNotFoundError:
        logger.error(f"Configuration for strategy '{strategy}' not found.")
        return None
    raw_config = getattr(config_module, "CONFIG", None)
    if raw_config is None:
        logger.error(f"CONFIG not defined for strategy '{strategy}'.")
        return None
    try:
        config_model = parse_strategy_config(raw_config)
        logger.info(f"Loaded configuration for {config_model.strategy} strategy")
        return config_model
    except ValidationError as exc:
        for err in exc.errors():
            loc = " -> ".join(str(piece) for piece in err.get("loc", ()))
            logger.error(f"Config validation error at {loc}: {err.get('msg')}")
        logger.error("Invalid strategy configuration. Aborting load.")
        return None

def get_sleep_duration(strategy):
    """
    Determines the sleep duration based on the strategy.
    """
    if strategy == "day_trading":
        return 3600
    elif strategy == "short_term":
        return 86400
    elif strategy == "mid_term":
        return 86400 * 7
    elif strategy == "long_term":
        return 86400 * 30
    else:
        return 3600  # Default to 1 hour
