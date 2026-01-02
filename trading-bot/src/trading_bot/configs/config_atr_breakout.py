# configs/config_atr_breakout.py
import os

_notification_emails = os.getenv("NOTIFICATION_EMAILS", "").strip()
NOTIFICATION_EMAILS = [email.strip() for email in _notification_emails.split(",") if email.strip()]

CONFIG = {
    "strategy": "atr_breakout",
    "data_source": "binance",
    "ticker": "BTCUSDT",
    "period": "1y",
    "interval": "1h",
    "indicators": [],
    "use_indicators": False,
    "log_level": "INFO",
    "notification_email": NOTIFICATION_EMAILS,
    "seed": 42,
    "donchian_window": 20,
    "atr_window": 14,
    "atr_stop_mult": 2.0,
    "atr_trail_mult": 3.0,
    "time_stop_bars": 48,
    "risk_management": {
        "stop_loss": 0.02,
        "take_profit": 0.05,
        "max_position_size": 0.1,
        "trading_fee": 0.001,
    },
}
