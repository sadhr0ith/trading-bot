# configs/config_day_trading.py
CONFIG = {
    "strategy": "day_trading",
    "data_source": "binance",  # or 'binance'
    "ticker": "BTCUSDT",  # Example for stock (use "BTCUSDT" for crypto)
    "period": "6mo",
    "interval": "1h",  # Intraday data for day trading
    "indicators": ["rsi", "stochastic"],
    "log_level": "DEBUG",
    "notification_email": ['mateusz.dziuk@gmail.com', 'biuro@aszenbrener.pl'],
    "models": {
        "day_trading": "LSTM"
    },
    "risk_management": {
        "stop_loss": 0.01,   # 1% stop loss
        "take_profit": 0.02  # 2% take profit
    }
}
