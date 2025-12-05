from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    balance: float
    positions: dict[str, PositionState] = Field(default_factory=dict)
    history: list[TradeRecord] = Field(default_factory=list)
    strategy_name: str | None = None

    model_config = ConfigDict(extra="ignore")
