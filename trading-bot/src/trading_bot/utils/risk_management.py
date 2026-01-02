from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError as PydanticValidationError

from trading_bot.models.config import RiskConfig
from trading_bot.utils.logger import setup_logger


@dataclass
class Position:
    symbol: str
    entry_price: float
    size: float
    stop_loss: float | None
    take_profit: float | None
    peak_price: float | None = None


class RiskManager:
    """
    Risk management for position sizing and exit conditions.

    Config schema:
        - stop_loss: float (e.g., 0.03 = 3% stop loss below entry)
        - take_profit: float (e.g., 0.05 = 5% take profit above entry)
        - max_position_size: float (e.g., 0.1 = 10% of balance max per position)
        - trading_fee: float (e.g., 0.001 = 0.1% trading fee per transaction)

    Example config:
        {
            "stop_loss": 0.03,
            "take_profit": 0.05,
            "max_position_size": 0.1,
            "trading_fee": 0.001
        }
    """

    def __init__(
        self,
        risk_config: RiskConfig | dict | None = None,
        default_max_position: float = 0.1,
        trading_fee: float = 0.001,
    ):
        self.logger = setup_logger(self.__class__.__name__)

        if risk_config is None:
            self.config = RiskConfig(max_position_size=default_max_position, trading_fee=trading_fee)
        elif isinstance(risk_config, RiskConfig):
            self.config = risk_config
        else:
            try:
                allowed_keys = set(RiskConfig.model_fields.keys())
                filtered = {k: v for k, v in risk_config.items() if k in allowed_keys}
                self.config = RiskConfig(**filtered)
            except PydanticValidationError as exc:
                self.logger.error(f"Invalid risk configuration; using defaults. Details: {exc}")
                self.config = RiskConfig(max_position_size=default_max_position, trading_fee=trading_fee)

        self.stop_loss = self.config.stop_loss
        self.take_profit = self.config.take_profit
        self.max_position = self.config.max_position_size
        self.trading_fee = self.config.trading_fee
        self.trailing_stop = getattr(self.config, "trailing_stop", None)

    def calculate_position_size(self, balance: float, price: float) -> float:
        if price <= 0 or balance <= 0:
            return 0.0
        raw_size = (balance * self.max_position) / price
        size = max(raw_size, 0.0)
        return size

    def evaluate_exit(self, position: Position, current_price: float) -> bool:
        if position is None:
            return False
        peak_price = position.peak_price or position.entry_price
        if self.trailing_stop is not None:
            peak_price = max(peak_price, current_price)
            trail_threshold = peak_price * (1 - self.trailing_stop)
            if current_price <= trail_threshold:
                self.logger.info(
                    f"Trailing stop triggered at {current_price:.4f} "
                    f"(peak {peak_price:.4f}, threshold {trail_threshold:.4f})."
                )
                return True
        if position.stop_loss is not None:
            stop_price = position.entry_price * (1 - position.stop_loss)
            if current_price <= stop_price:
                self.logger.info(f"Stop-loss triggered at {current_price:.4f} (threshold {stop_price:.4f}).")
                return True
        if position.take_profit is not None:
            tp_price = position.entry_price * (1 + position.take_profit)
            if current_price >= tp_price:
                self.logger.info(f"Take-profit triggered at {current_price:.4f} (threshold {tp_price:.4f}).")
                return True
        return False
