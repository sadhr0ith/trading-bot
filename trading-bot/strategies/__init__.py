"""Trading strategies module.

This module contains implementations of different trading strategies:
- DayTradingStrategy: LSTM-based intraday trading (1h candles)
- ShortTermStrategy: XGBoost for 5-day swings (1d candles)
- MidTermStrategy: RandomForest for 20-day trends (1d candles)
- LongTermStrategy: RandomForest for 50-day positions (1d candles)

All strategies inherit from StrategyBase and implement the execute() method.
"""

from strategies.strategy_base import StrategyBase
from strategies.day_trading_strategy import DayTradingStrategy
from strategies.short_term_strategy import ShortTermStrategy
from strategies.mid_term_strategy import MidTermStrategy
from strategies.long_term_strategy import LongTermStrategy

__all__ = [
    "StrategyBase",
    "DayTradingStrategy",
    "ShortTermStrategy",
    "MidTermStrategy",
    "LongTermStrategy",
]
