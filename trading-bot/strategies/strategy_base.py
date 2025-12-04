# strategies/strategy_base.py
from abc import ABC, abstractmethod

from utils.logger import setup_logger
from utils.paper_trading import PaperTradingExecutor
from utils.risk_management import RiskManager
from utils.seeding import set_global_seeds


class StrategyBase(ABC):
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.logger = setup_logger(self.__class__.__name__, config.get("log_level") if config else None)
        risk_cfg = None
        if config:
            risk_cfg = getattr(config, "risk_management", None) if hasattr(config, "risk_management") else config.get("risk_management")
        self.risk_manager = RiskManager(risk_cfg)

        # Use strategy-specific state file to prevent position leakage between strategies
        strategy_name = config.get("strategy") if config else None
        self.order_executor = PaperTradingExecutor(strategy_name=strategy_name)

        self.seed = set_global_seeds(config.get("seed") if config else None)

    @abstractmethod
    def execute(self):
        pass

    def log_action(self, message, level="info"):
        getattr(self.logger, level.lower())(message)
