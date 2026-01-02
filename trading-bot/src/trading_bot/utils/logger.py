import logging
import os

from trading_bot.models.env_settings import load_logging_settings

_ROOT_LOGGER_NAME = "trading_bot"
_BASE_CONFIGURED = False


class ColoredFormatter(logging.Formatter):
    COLOR_CODES = {
        "BUY": "\033[92m",  # Green for BUY
        "SELL": "\033[91m",  # Red for SELL
        "HOLD": "\033[93m",  # Yellow for HOLD
        "DEBUG": "\033[94m",  # Blue for debugging
        "INFO": "\033[92m",  # Green for general info
        "WARNING": "\033[93m",  # Yellow for warnings
        "ERROR": "\033[91m",  # Red for errors
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if "BUY signal" in message:
            color = self.COLOR_CODES["BUY"]
        elif "SELL signal" in message:
            color = self.COLOR_CODES["SELL"]
        elif "HOLD signal" in message:
            color = self.COLOR_CODES["HOLD"]
        else:
            color = self.COLOR_CODES.get(record.levelname, self.COLOR_CODES["RESET"])

        reset = self.COLOR_CODES["RESET"]
        return f"{color}{message}{reset}"


def _color_enabled(stream) -> bool:
    env_flag = os.getenv("TRADING_BOT_COLOR_LOGS")
    if env_flag is not None and env_flag.strip().lower() in {"0", "false", "no"}:
        return False
    try:
        return bool(getattr(stream, "isatty", lambda: False)())
    except (OSError, ValueError):
        return False


def _resolve_level(level: str | int | None) -> int:
    env_settings = load_logging_settings()
    env_level = env_settings.level if env_settings else os.getenv("TRADING_BOT_LOG_LEVEL")
    if level is None:
        level = env_level

    if isinstance(level, str):
        level = level.upper()
        return getattr(logging, level, logging.INFO)
    if isinstance(level, int):
        return level
    return logging.INFO


def _configure_root_logger(level: str | int | None = None) -> logging.Logger:
    global _BASE_CONFIGURED
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    resolved_level = _resolve_level(level)

    if not _BASE_CONFIGURED:
        handler = logging.StreamHandler()
        if _color_enabled(handler.stream):
            formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        else:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        _BASE_CONFIGURED = True

    logger.setLevel(resolved_level)
    # Root logger should not propagate further.
    logger.propagate = False
    return logger


def get_logger(name: str | None = None, level: str | int | None = None) -> logging.Logger:
    """Return a child of the shared trading_bot logger."""
    root_logger = _configure_root_logger(level)
    if name in (None, _ROOT_LOGGER_NAME, "TradingBot"):
        return root_logger

    target_name = name if str(name).startswith(f"{_ROOT_LOGGER_NAME}.") else f"{_ROOT_LOGGER_NAME}.{name}"
    logger = logging.getLogger(target_name)
    # Keep handlers on the root logger only; children propagate up.
    logger.setLevel(logging.NOTSET)
    return logger


def setup_logger(name: str, level: str | int | None = None) -> logging.Logger:
    """Backward-compatible alias to get_logger."""
    return get_logger(name, level)
