from __future__ import annotations

from typing import TypeVar

from pydantic import EmailStr, Field, ValidationError
from pydantic_settings import BaseSettings

T = TypeVar("T", bound=BaseSettings)


class BinanceSettings(BaseSettings):
    api_key: str = Field(..., env="BINANCE_API_KEY")
    api_secret: str = Field(..., env="BINANCE_API_SECRET")

    class Config:
        extra = "ignore"


class EmailSettings(BaseSettings):
    sender_email: EmailStr = Field(..., env="GMAIL_SENDER_EMAIL")
    app_password: str = Field(..., env="GMAIL_APP_PASSWORD")
    smtp_server: str = Field("smtp.gmail.com", env="SMTP_SERVER")
    smtp_port: int = Field(587, env="SMTP_PORT")

    class Config:
        extra = "ignore"


class CacheSettings(BaseSettings):
    ttl_seconds: int = Field(0, ge=0, env="TRADING_BOT_CACHE_TTL_SECONDS")

    class Config:
        extra = "ignore"


class LoggingSettings(BaseSettings):
    level: str | None = Field(None, env="TRADING_BOT_LOG_LEVEL")

    class Config:
        extra = "ignore"


def _load_settings(settings_cls: type[T], logger=None) -> T | None:
    try:
        return settings_cls()
    except ValidationError as exc:
        if logger:
            logger.error(f"{settings_cls.__name__} validation failed: {exc}")
        return None


def load_binance_settings(logger=None) -> BinanceSettings | None:
    return _load_settings(BinanceSettings, logger)


def load_email_settings(logger=None) -> EmailSettings | None:
    return _load_settings(EmailSettings, logger)


def load_cache_settings(logger=None) -> CacheSettings | None:
    return _load_settings(CacheSettings, logger)


def load_logging_settings(logger=None) -> LoggingSettings | None:
    return _load_settings(LoggingSettings, logger)
