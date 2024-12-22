# configs/config_short_term.py
CONFIG = {
    "strategy": "short_term",
    "data_source": "binance",  # or 'binance'
    "ticker": "BTCUSDT",  # Stock example
    "period": "6mo",
    "interval": "1d",
    "indicators": ["macd", "rsi"],
    "log_level": "DEBUG",
    "notification_email": ['mateusz.dziuk@gmail.com', 'biuro@aszenbrener.pl'],
    "models": {
        "short_term": "LSTM"
        #"short_term": "XGBoost"
    },
    "risk_management": {
        "stop_loss": 0.03,   # 3% stop loss
        "take_profit": 0.05  # 5% take profit
    }
}
