# configs/config_short_term.py
import os

# Load notification emails from environment (comma-separated)
_notification_emails = os.getenv("NOTIFICATION_EMAILS", "").strip()
NOTIFICATION_EMAILS = [email.strip() for email in _notification_emails.split(",") if email.strip()]

CONFIG = {
    "strategy": "short_term",
    "data_source": "binance",  # or 'binance'
    "ticker": "BTCUSDT",  # Stock example
    "period": "6M",
    "interval": "1d",
    "indicators": ["macd", "rsi"],
    "use_indicators": True,
    "log_level": "DEBUG",
    "notification_email": NOTIFICATION_EMAILS,
    "seed": 42,
    "risk_management": {
        "stop_loss": 0.03,   # 3% stop loss
        "take_profit": 0.05,  # 5% take profit
        "max_position_size": 0.1,
        "trading_fee": 0.001
    }
}
