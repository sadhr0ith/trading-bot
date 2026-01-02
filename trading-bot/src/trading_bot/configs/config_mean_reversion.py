# configs/config_mean_reversion.py
import os

_notification_emails = os.getenv("NOTIFICATION_EMAILS", "").strip()
NOTIFICATION_EMAILS = [email.strip() for email in _notification_emails.split(",") if email.strip()]

CONFIG = {
    "strategy": "mean_reversion",
    "data_source": "binance",
    "ticker": "BTCUSDT",
    "period": "6mo",
    "interval": "15m",
    "indicators": ["rsi", "bollinger_bands"],
    "use_indicators": True,
    "log_level": "INFO",
    "notification_email": NOTIFICATION_EMAILS,
    "seed": 42,
    "bb_window": 20,
    "bb_num_std": 2.0,
    "rsi_period": 14,
    "rsi_oversold": 30.0,
    "time_stop_bars": 32,
    "max_loss_pct": 0.01,
    "partial_take_profit_pct": 0.015,
    "risk_management": {
        "stop_loss": 0.01,
        "take_profit": 0.02,
        "max_position_size": 0.1,
        "trading_fee": 0.001,
    },
}
