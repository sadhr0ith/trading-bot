# strategies/strategy_base.py
from abc import ABC, abstractmethod
from utils.logger import setup_logger

class StrategyBase(ABC):
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.logger = setup_logger(self.__class__.__name__)

    @abstractmethod
    def execute(self):
        pass

    def log_action(self, message, level="info"):
        getattr(self.logger, level.lower())(message)
