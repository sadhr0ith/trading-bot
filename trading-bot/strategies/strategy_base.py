# strategies/strategy_base.py
from abc import ABC, abstractmethod

from utils.logger import setup_logger
from utils.paper_trading import PaperTradingExecutor
from utils.risk_management import RiskManager
from utils.seeding import set_global_seeds


class StrategyBase(ABC):
    """Common template for all strategies (data prep → validate → indicators → strategy run)."""

    MIN_ROWS = 50

    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.logger = setup_logger(self.__class__.__name__, config.get("log_level") if config else None)
        risk_cfg = None
        if config:
            risk_cfg = (
                getattr(config, "risk_management", None)
                if hasattr(config, "risk_management")
                else config.get("risk_management")
            )
        self.risk_manager = RiskManager(risk_cfg)

        # Use strategy-specific state file to prevent position leakage between strategies
        strategy_name = config.get("strategy") if config else None
        self.order_executor = PaperTradingExecutor(strategy_name=strategy_name)

        self.seed = set_global_seeds(config.get("seed") if config else None)

    def execute(self):
        """Template method orchestrating the strategy flow."""
        prepared = self._prepare_data(self.data)
        if not self._validate_data(prepared):
            return
        enriched = self._add_indicators(prepared)
        return self._run_strategy(enriched)

    def _prepare_data(self, data):
        """Make a sorted copy of input data."""
        if data is None:
            return None
        return data.copy().sort_index()

    def _validate_data(self, data) -> bool:
        if data is None or data.empty:
            self.log_action("Input data frame is empty; skipping execution.", "warning")
            return False
        if len(data) < self.MIN_ROWS:
            self.log_action(f"Not enough data rows; need at least {self.MIN_ROWS}, got {len(data)}.", "warning")
            return False
        return True

    def _add_indicators(self, data):
        """Hook for subclasses to add indicators. Default: no-op."""
        return data

    @abstractmethod
    def _run_strategy(self, data):
        """Core strategy implementation (training/inference/decisions)."""
        raise NotImplementedError

    def log_action(self, message, level="info"):
        getattr(self.logger, level.lower())(message)
