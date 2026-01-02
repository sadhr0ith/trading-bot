# configs/config_regime_switch.py
import os

_notification_emails = os.getenv("NOTIFICATION_EMAILS", "").strip()
NOTIFICATION_EMAILS = [email.strip() for email in _notification_emails.split(",") if email.strip()]

CONFIG = {
    "strategy": "regime_switch",
    "data_source": "binance",
    "ticker": "BTCUSDT",
    "period": "1y",
    "interval": "1h",
    "indicators": ["adx", "rsi", "bollinger_bands"],
    "use_indicators": True,
    "log_level": "INFO",
    "notification_email": NOTIFICATION_EMAILS,
    "seed": 42,
    "adx_window": 14,
    "adx_trend_threshold": 25.0,
    "volatility_window": 20,
    "volatility_high_threshold": 0.05,
    "cooldown_bars": 4,
    "donchian_window": 20,
    "atr_window": 14,
    "atr_stop_mult": 2.0,
    "atr_trail_mult": 3.0,
    "time_stop_bars": 48,
    "bb_window": 20,
    "bb_num_std": 2.0,
    "rsi_period": 14,
    "rsi_oversold": 30.0,
    "max_loss_pct": 0.01,
    "partial_take_profit_pct": 0.015,
    "risk_management": {
        "stop_loss": 0.02,
        "take_profit": 0.05,
        "max_position_size": 0.1,
        "trading_fee": 0.001,
    },
}
