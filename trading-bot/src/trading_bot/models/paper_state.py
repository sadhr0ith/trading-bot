from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PositionState(BaseModel):
    symbol: str
    entry_price: float
    size: float
    stop_loss: float | None = None
    take_profit: float | None = None
    entry_notional: float | None = None
    opened_at: datetime


class TradeRecord(BaseModel):
    symbol: str
    action: str
    price: float
    size: float
    timestamp: datetime
    pnl: float | None = None
    reason: str | None = None

    @field_validator("action")
    @classmethod
    def _validate_action(cls, value):
        if value not in {"BUY", "SELL", "CLOSE"}:
            raise ValueError("action must be BUY, SELL, or CLOSE")
        return value


class PaperState(BaseModel):
    """Paper trading state with drawdown tracking.

    Tracks balance, positions, history, and critical risk metrics like max drawdown.
    Max drawdown is automatically updated on every state validation.
    """

    balance: float
    positions: dict[str, PositionState] = Field(default_factory=dict)
    history: list[TradeRecord] = Field(default_factory=list)
    strategy_name: str | None = None

    # Drawdown tracking fields
    peak_balance: float | None = Field(default=None, description="Highest balance reached")
    max_drawdown: float = Field(default=0.0, ge=0.0, le=1.0, description="Maximum drawdown as fraction (0.0-1.0)")

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def _update_drawdown_metrics(self) -> "PaperState":
        """Update peak balance and max drawdown after every state change.

        Max drawdown = (peak_balance - current_balance) / peak_balance
        """
        # Initialize peak_balance if not set
        if self.peak_balance is None:
            self.peak_balance = self.balance

        # Update peak if current balance is higher
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # Calculate current drawdown
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
            # Track maximum drawdown seen
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

        return self
