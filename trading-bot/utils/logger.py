# utils/logger.py
import logging

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
        if "BUY signal" in record.msg:
            color = self.COLOR_CODES["BUY"]
        elif "SELL signal" in record.msg:
            color = self.COLOR_CODES["SELL"]
        elif "HOLD signal" in record.msg:
            color = self.COLOR_CODES["HOLD"]
        else:
            color = self.COLOR_CODES.get(record.levelname, self.COLOR_CODES["RESET"])
        
        reset = self.COLOR_CODES["RESET"]
        record.msg = f"{color}{record.msg}{reset}"
        return super().format(record)

def setup_logger(name):
    logger = logging.getLogger(name)

    # Check if the logger already has handlers to prevent duplicate logs
    if not logger.hasHandlers():
        formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Set the logging level (you can adjust this based on the required level)
    logger.setLevel(logging.DEBUG)
    return logger
