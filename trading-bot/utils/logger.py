# utils/logger.py
import logging
import os


class ColoredFormatter(logging.Formatter):
    COLOR_CODES = {
        "BUY": "\033[92m",      # Green for BUY
        "SELL": "\033[91m",     # Red for SELL
        "HOLD": "\033[93m",     # Yellow for HOLD
        "DEBUG": "\033[94m",    # Blue for debugging
        "INFO": "\033[92m",     # Green for general info
        "WARNING": "\033[93m",  # Yellow for warnings
        "ERROR": "\033[91m",    # Red for errors
        "RESET": "\033[0m"
    }

    def format(self, record):
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


def _resolve_level(level):
    env_level = os.getenv("TRADING_BOT_LOG_LEVEL")
    if level is None:
        level = env_level

    if isinstance(level, str):
        level = level.upper()
        return getattr(logging, level, logging.INFO)
    if isinstance(level, int):
        return level
    return logging.INFO


def setup_logger(name, level=None):
    logger = logging.getLogger(name)
    resolved_level = _resolve_level(level)

    if not logger.hasHandlers():
        formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(resolved_level)
    logger.propagate = False
    return logger
