# configs/config_mid_term.py
import os

# Load notification emails from environment (comma-separated)
_notification_emails = os.getenv("NOTIFICATION_EMAILS", "").strip()
NOTIFICATION_EMAILS = [email.strip() for email in _notification_emails.split(",") if email.strip()]

CONFIG = {
    "strategy": "mid_term",
    "data_source": "yahoo",  # or 'binance'
    "ticker": "AAPL",
    "period": "1y",
    "interval": "1d",
    "indicators": ["macd", "bollinger_bands"],
    "use_indicators": True,
    "log_level": "DEBUG",
    "notification_email": NOTIFICATION_EMAILS,
    "seed": 42,
    "risk_management": {
        "stop_loss": 0.05,   # 5% stop loss
        "take_profit": 0.10,  # 10% take profit
        "max_position_size": 0.1,
        "trading_fee": 0.001
    }
}
