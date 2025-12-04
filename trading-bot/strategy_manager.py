import pandas as pd

from strategies.strategy_base import StrategyBase


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
    if config["strategy"] == "day_trading":
        from strategies.day_trading_strategy import DayTradingStrategy

        return DayTradingStrategy(config, data)
    if config["strategy"] == "short_term":
        from strategies.short_term_strategy import ShortTermStrategy

        return ShortTermStrategy(config, data)
    if config["strategy"] == "mid_term":
        from strategies.mid_term_strategy import MidTermStrategy

        return MidTermStrategy(config, data)
    if config["strategy"] == "long_term":
        from strategies.long_term_strategy import LongTermStrategy

        return LongTermStrategy(config, data)
    raise ValueError(f"Unsupported strategy type: {config.get('strategy')}")
