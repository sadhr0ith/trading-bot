import importlib
import pandas as pd

from trading_bot.strategies.strategy_base import StrategyBase


# Registry of strategy name -> class (lazy imports to avoid heavy deps at module import)
_STRATEGY_REGISTRY: dict[str, str] = {
    "day_trading": "trading_bot.strategies.day_trading_strategy.DayTradingStrategy",
    "short_term": "trading_bot.strategies.short_term_strategy.ShortTermStrategy",
    "mid_term": "trading_bot.strategies.mid_term_strategy.MidTermStrategy",
    "long_term": "trading_bot.strategies.long_term_strategy.LongTermStrategy",
    "atr_breakout": "trading_bot.strategies.atr_breakout_strategy.ATRBreakoutStrategy",
    "mean_reversion": "trading_bot.strategies.mean_reversion_strategy.MeanReversionStrategy",
    "regime_switch": "trading_bot.strategies.regime_switch_strategy.RegimeSwitchStrategy",
    "day_trading_ml": "trading_bot.strategies.day_trading_ml_strategy.DayTradingMLStrategy",
}


def _import_strategy(path: str) -> type[StrategyBase]:
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def select_strategy(config: dict, data: pd.DataFrame) -> StrategyBase:
    """
    Select and instantiate the appropriate strategy class based on the config.

    Args:
        config: Strategy configuration (dict or Pydantic model with 'strategy' key).
        data: Market data DataFrame (expects OHLCV columns for most strategies).

    Returns:
        Concrete StrategyBase subclass instance.

    Raises:
        ValueError: If strategy type is unsupported.
    """
    name = config.get("strategy")
    path = _STRATEGY_REGISTRY.get(name)
    if not path:
        raise ValueError(f"Unsupported strategy type: {name}")
    strategy_cls = _import_strategy(path)
    return strategy_cls(config, data)
