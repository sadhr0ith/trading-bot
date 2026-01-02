"""Trading strategies module.

This module contains implementations of different trading strategies:
- DayTradingStrategy: LSTM-based intraday trading (1h candles)
- ShortTermStrategy: XGBoost for 5-day swings (1d candles)
- MidTermStrategy: RandomForest for 20-day trends (1d candles)
- LongTermStrategy: RandomForest for 50-day positions (1d candles)
- ATRBreakoutStrategy: trend breakout with ATR stops
- MeanReversionStrategy: Bollinger + RSI mean reversion
- RegimeSwitchStrategy: trend/range routing with vol filter
- DayTradingMLStrategy: tabular ML intraday returns

All strategies inherit from StrategyBase and implement the execute() method.
"""

from trading_bot.strategies.strategy_base import StrategyBase
from trading_bot.strategies.day_trading_strategy import DayTradingStrategy
from trading_bot.strategies.short_term_strategy import ShortTermStrategy
from trading_bot.strategies.mid_term_strategy import MidTermStrategy
from trading_bot.strategies.long_term_strategy import LongTermStrategy
from trading_bot.strategies.atr_breakout_strategy import ATRBreakoutStrategy
from trading_bot.strategies.mean_reversion_strategy import MeanReversionStrategy
from trading_bot.strategies.regime_switch_strategy import RegimeSwitchStrategy
from trading_bot.strategies.day_trading_ml_strategy import DayTradingMLStrategy

__all__ = [
    "StrategyBase",
    "DayTradingStrategy",
    "ShortTermStrategy",
    "MidTermStrategy",
    "LongTermStrategy",
    "ATRBreakoutStrategy",
    "MeanReversionStrategy",
    "RegimeSwitchStrategy",
    "DayTradingMLStrategy",
]
