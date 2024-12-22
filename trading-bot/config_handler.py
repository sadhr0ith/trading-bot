import importlib
import logging

logger = logging.getLogger("TradingBot")

def load_config(strategy):
    """
    Load the appropriate config file based on the strategy.
    """
    try:
        config_module = importlib.import_module(f"configs.config_{strategy}")
        logger.info(f"Loaded configuration for {strategy} strategy")
        return config_module.CONFIG
    except ModuleNotFoundError:
        logger.error(f"Configuration for strategy '{strategy}' not found.")
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
