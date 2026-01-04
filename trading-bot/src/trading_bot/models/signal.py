"""Signal decision models for multi-strategy trading."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SignalAction(str, Enum):
    """Trading signal action types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"  # Explicit exit signal (close position)
    RISK_EXIT = "RISK_EXIT"  # Exit due to risk management (highest priority)


class SignalDecision(BaseModel):
    """
    Immutable signal decision from a strategy.

    This model is designed to be JSON-serializable for future IPC/queue use.
    It represents a single strategy's recommendation for a specific symbol.
    """

    strategy_name: str = Field(..., description="Name of the strategy that generated this signal")
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT, AAPL)")
    action: SignalAction = Field(..., description="Recommended action")
    timeframe: str | None = Field(None, description="Timeframe of the strategy (e.g., 1h, 1d)")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence level (0-1)")
    edge: float | None = Field(None, description="Expected edge/return (e.g., predicted return)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when signal was generated",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional strategy-specific data")

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_utc(cls, v: datetime | str) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-safe dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "action": self.action.value,
            "timeframe": self.timeframe,
            "confidence": self.confidence,
            "edge": self.edge,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignalDecision":
        """Create from JSON dictionary."""
        if "action" in data and isinstance(data["action"], str):
            data = dict(data)
            data["action"] = SignalAction(data["action"])
        return cls(**data)


class AggregatedDecision(BaseModel):
    """
    Final aggregated decision for a symbol after combining multiple strategy signals.

    Contains the final action along with the votes that led to it for audit/reproducibility.
    """

    symbol: str = Field(..., description="Trading symbol")
    final_action: SignalAction = Field(..., description="Final aggregated action")
    votes: list[SignalDecision] = Field(..., description="All strategy votes that contributed to this decision")
    reason: str = Field(..., description="Human-readable explanation of aggregation logic")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of aggregation",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_utc(cls, v: datetime | str) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-safe dictionary for history storage."""
        return {
            "symbol": self.symbol,
            "final_action": self.final_action.value,
            "votes": [v.to_dict() for v in self.votes],
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }
