# configs/config_long_term.py
CONFIG = {
    "strategy": "long_term",
    "data_source": "yahoo",  # or 'binance'
    "ticker": "AAPL",
    "period": "5y",
    "interval": "1wk",
    "indicators": ["sma_200", "fundamental_analysis"],
    "log_level": "DEBUG",
    "models": {
        "long_term": "RandomForest"
    },
    "risk_management": {
        "stop_loss": 0.10,   # 10% stop loss
        "take_profit": 0.30  # 30% take profit
    }
}
