# configs/config_long_term.py
import os

# Load notification emails from environment (comma-separated)
_notification_emails = os.getenv("NOTIFICATION_EMAILS", "").strip()
NOTIFICATION_EMAILS = [email.strip() for email in _notification_emails.split(",") if email.strip()]

CONFIG = {
    "strategy": "long_term",
    "data_source": "yahoo",  # or 'binance'
    "ticker": "AAPL",
    "period": "5y",
    "interval": "1wk",
    "indicators": ["sma", "ema", "macd"],
    "use_indicators": True,
    "log_level": "DEBUG",
    "notification_email": NOTIFICATION_EMAILS,
    "seed": 42,
    "risk_management": {
        "stop_loss": 0.10,   # 10% stop loss
        "take_profit": 0.30,  # 30% take profit
        "max_position_size": 0.1,
        "trading_fee": 0.001
    }
}
