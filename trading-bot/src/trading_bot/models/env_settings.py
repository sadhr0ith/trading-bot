from __future__ import annotations

from typing import TypeVar

from pydantic import AliasChoices, EmailStr, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

T = TypeVar("T", bound=BaseSettings)


class BinanceSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_prefix="BINANCE_", populate_by_name=True)

    api_key: str
    api_secret: str


class EmailSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    sender_email: EmailStr = Field(
        ...,
        validation_alias=AliasChoices("GMAIL_SENDER_EMAIL", "sender_email"),
    )
    app_password: str = Field(
        ...,
        validation_alias=AliasChoices("GMAIL_APP_PASSWORD", "app_password"),
    )
    smtp_server: str = Field(
        "smtp.gmail.com",
        validation_alias=AliasChoices("SMTP_SERVER", "smtp_server"),
    )
    smtp_port: int = Field(
        587,
        validation_alias=AliasChoices("SMTP_PORT", "smtp_port"),
    )


class CacheSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_prefix="TRADING_BOT_", populate_by_name=True)

    cache_ttl_seconds: int = Field(3600, ge=0)
    min_rows: int = Field(50, ge=1)
    sleep_seconds: int | None = Field(None, ge=0)
    min_sleep_seconds: int = Field(60, ge=1)


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_prefix="TRADING_BOT_", populate_by_name=True)

    level: str | None = Field(None)


# Backward compatible alias for runtime settings that include cache TTL, sleep defaults and min_rows.
class RuntimeSettings(CacheSettings):
    @property
    def ttl_seconds(self) -> int:
        """Compatibility property for older callers expecting ttl_seconds."""
        return self.cache_ttl_seconds


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


def load_cache_settings(logger=None) -> RuntimeSettings | None:
    return _load_settings(RuntimeSettings, logger)


def load_logging_settings(logger=None) -> LoggingSettings | None:
    return _load_settings(LoggingSettings, logger)


def load_runtime_settings(logger=None) -> RuntimeSettings | None:
    return _load_settings(RuntimeSettings, logger)
