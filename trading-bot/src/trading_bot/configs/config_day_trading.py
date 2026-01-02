# configs/config_day_trading.py
import os

# Load notification emails from environment (comma-separated)
_notification_emails = os.getenv("NOTIFICATION_EMAILS", "").strip()
NOTIFICATION_EMAILS = [email.strip() for email in _notification_emails.split(",") if email.strip()]

CONFIG = {
    "strategy": "day_trading",
    "data_source": "binance",  # or 'binance'
    "ticker": "BTCUSDT",  # Example for stock (use "BTCUSDT" for crypto)
    "period": "6M",
    "interval": "1h",  # Intraday data for day trading
    "indicators": ["rsi", "stochastic"],
    "use_indicators": True,
    "log_level": "DEBUG",
    "notification_email": NOTIFICATION_EMAILS,
    "seed": 42,
    "risk_management": {
        "stop_loss": 0.01,  # 1% stop loss
        "take_profit": 0.02,  # 2% take profit
        "max_position_size": 0.1,
        "trading_fee": 0.001,
    },
}
