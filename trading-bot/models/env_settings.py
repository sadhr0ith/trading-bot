from __future__ import annotations

from typing import Optional, Type, TypeVar

from pydantic import BaseSettings, EmailStr, Field, ValidationError

T = TypeVar("T", bound=BaseSettings)


class BinanceSettings(BaseSettings):
    api_key: str = Field(..., env="BINANCE_API_KEY")
    api_secret: str = Field(..., env="BINANCE_API_SECRET")

    class Config:
        extra = "ignore"


class EmailSettings(BaseSettings):
    sender_email: EmailStr = Field(..., env="GMAIL_SENDER_EMAIL")
    app_password: str = Field(..., env="GMAIL_APP_PASSWORD")

    class Config:
        extra = "ignore"


class CacheSettings(BaseSettings):
    ttl_seconds: int = Field(0, ge=0, env="TRADING_BOT_CACHE_TTL_SECONDS")

    class Config:
        extra = "ignore"


class LoggingSettings(BaseSettings):
    level: Optional[str] = Field(None, env="TRADING_BOT_LOG_LEVEL")

    class Config:
        extra = "ignore"


def _load_settings(settings_cls: Type[T], logger=None) -> Optional[T]:
    try:
        return settings_cls()
    except ValidationError as exc:
        if logger:
            logger.error(f"{settings_cls.__name__} validation failed: {exc}")
        return None


def load_binance_settings(logger=None) -> Optional[BinanceSettings]:
    return _load_settings(BinanceSettings, logger)


def load_email_settings(logger=None) -> Optional[EmailSettings]:
    return _load_settings(EmailSettings, logger)


def load_cache_settings(logger=None) -> Optional[CacheSettings]:
    return _load_settings(CacheSettings, logger)


def load_logging_settings(logger=None) -> Optional[LoggingSettings]:
    return _load_settings(LoggingSettings, logger)
