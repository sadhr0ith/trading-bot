# config.py
CONFIG = {
    "strategy": "day_trading",  # Change to short_term, mid_term, or long_term
    "data_source": "binance",  # 'yahoo' for stocks, 'binance' for crypto
    "ticker": "BTCUSDT",  # Stock ticker symbol for 'yahoo', or trading pair for 'binance' (e.g., 'BTCUSDT')
    "period": "1y",  # For yahoo: '1d', '5d', '1mo', '1y' etc.
    "interval": "1d",  # For binance: '1m', '1h', '1d', etc.; For yahoo: '1d', '1h', etc.
    "indicators": ["rsi", "stochastic"],  # Specify which indicators to calculate
    "log_level": "DEBUG",
    "models": {
        "day_trading": "LSTM",           # Use LSTM for day trading
        "short_term": "XGBoost",         # Use XGBoost for short-term
        "mid_term": "RandomForest"       # Use RandomForest for mid-term
    },
    "risk_management": {
        "stop_loss": 0.02,   # 2% stop loss
        "take_profit": 0.04  # 4% take profit
    }
}
