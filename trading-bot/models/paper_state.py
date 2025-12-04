from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class PositionState(BaseModel):
    symbol: str
    entry_price: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_notional: Optional[float] = None
    opened_at: datetime


class TradeRecord(BaseModel):
    symbol: str
    action: str
    price: float
    size: float
    timestamp: datetime
    pnl: Optional[float] = None
    reason: Optional[str] = None

    @validator("action")
    def _validate_action(cls, value):
        if value not in {"BUY", "SELL", "CLOSE"}:
            raise ValueError("action must be BUY, SELL, or CLOSE")
        return value


class PaperState(BaseModel):
    balance: float
    positions: Dict[str, PositionState] = Field(default_factory=dict)
    history: List[TradeRecord] = Field(default_factory=list)
    strategy_name: Optional[str] = None

    class Config:
        extra = "ignore"
