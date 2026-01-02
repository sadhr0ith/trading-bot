"""Configuration files module.

This module contains strategy-specific configuration files:
- config_day_trading.py: LSTM intraday trading (1h interval)
- config_short_term.py: XGBoost 5-day horizon (1d interval)
- config_mid_term.py: RandomForest 20-day trends (1d interval)
- config_long_term.py: RandomForest 50-day positions (1d interval)

Each config exports a 'config' dictionary with strategy parameters.
"""

# Note: Configs are loaded dynamically by config_handler.py
# No imports here to avoid circular dependencies
