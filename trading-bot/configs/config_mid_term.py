# configs/config_mid_term.py
CONFIG = {
    "strategy": "mid_term",
    "data_source": "yahoo",  # or 'binance'
    "ticker": "AAPL",
    "period": "1y",
    "interval": "1d",
    "indicators": ["macd", "bollinger_bands"],
    "log_level": "DEBUG",
    "models": {
        "mid_term": "RandomForest"
    },
    "risk_management": {
        "stop_loss": 0.05,   # 5% stop loss
        "take_profit": 0.10  # 10% take profit
    }
}
