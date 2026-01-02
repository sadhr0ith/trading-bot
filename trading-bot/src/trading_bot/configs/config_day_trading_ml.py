# configs/config_day_trading_ml.py
import os

_notification_emails = os.getenv("NOTIFICATION_EMAILS", "").strip()
NOTIFICATION_EMAILS = [email.strip() for email in _notification_emails.split(",") if email.strip()]

CONFIG = {
    "strategy": "day_trading_ml",
    "data_source": "binance",
    "ticker": "BTCUSDT",
    "period": "1y",
    "interval": "1h",
    "indicators": ["rsi", "bollinger_bands", "adx"],
    "use_indicators": True,
    "log_level": "INFO",
    "notification_email": NOTIFICATION_EMAILS,
    "seed": 42,
    "return_horizon": 1,
    "prediction_threshold": 0.0005,
    "slippage_rate": 0.0002,
    "ml_min_improvement": 0.05,
    "ml_max_drawdown": 0.2,
    "min_inference_rows": 120,
    "risk_management": {
        "stop_loss": 0.01,
        "take_profit": 0.02,
        "max_position_size": 0.1,
        "trading_fee": 0.001,
    },
}
