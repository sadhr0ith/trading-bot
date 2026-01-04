"""Tests for SignalDecision JSON serialization and deserialization."""
from __future__ import annotations

import json
from datetime import timezone

import pytest

from trading_bot.models.signal import AggregatedDecision, SignalAction, SignalDecision


class TestSignalDecisionSerialization:
    """Test SignalDecision pydantic model serialization."""

    def test_signal_decision_to_dict(self):
        """SignalDecision.to_dict() should produce JSON-safe dictionary."""
        signal = SignalDecision(
            strategy_name="test_strategy",
            symbol="BTCUSDT",
            action=SignalAction.BUY,
            timeframe="1h",
            confidence=0.85,
            edge=0.02,
            metadata={"reason": "bullish"},
        )

        result = signal.to_dict()

        assert result["strategy_name"] == "test_strategy"
        assert result["symbol"] == "BTCUSDT"
        assert result["action"] == "BUY"
        assert result["timeframe"] == "1h"
        assert result["confidence"] == 0.85
        assert result["edge"] == 0.02
        assert result["metadata"] == {"reason": "bullish"}
        assert "timestamp" in result

    def test_signal_decision_json_serializable(self):
        """SignalDecision should be JSON serializable via to_dict()."""
        signal = SignalDecision(
            strategy_name="test_strategy",
            symbol="BTCUSDT",
            action=SignalAction.SELL,
        )

        # Should not raise
        json_str = json.dumps(signal.to_dict())
        assert json_str is not None

        # Should roundtrip
        parsed = json.loads(json_str)
        assert parsed["strategy_name"] == "test_strategy"
        assert parsed["action"] == "SELL"

    def test_signal_decision_from_dict(self):
        """SignalDecision.from_dict() should reconstruct from dictionary."""
        data = {
            "strategy_name": "long_term",
            "symbol": "ETHUSDT",
            "action": "HOLD",
            "timeframe": "1d",
            "confidence": 0.5,
            "edge": None,
            "timestamp": "2025-01-01T12:00:00+00:00",
            "metadata": {"test": True},
        }

        signal = SignalDecision.from_dict(data)

        assert signal.strategy_name == "long_term"
        assert signal.symbol == "ETHUSDT"
        assert signal.action == SignalAction.HOLD
        assert signal.timeframe == "1d"
        assert signal.confidence == 0.5
        assert signal.metadata == {"test": True}

    def test_signal_decision_roundtrip(self):
        """SignalDecision should survive full JSON roundtrip."""
        original = SignalDecision(
            strategy_name="mid_term",
            symbol="AAPL",
            action=SignalAction.EXIT,
            timeframe="4h",
            confidence=0.9,
            edge=-0.01,
            metadata={"indicators": ["RSI", "MACD"]},
        )

        # Serialize to JSON string
        json_str = json.dumps(original.to_dict())

        # Deserialize
        parsed = json.loads(json_str)
        reconstructed = SignalDecision.from_dict(parsed)

        assert reconstructed.strategy_name == original.strategy_name
        assert reconstructed.symbol == original.symbol
        assert reconstructed.action == original.action
        assert reconstructed.timeframe == original.timeframe
        assert reconstructed.confidence == original.confidence
        assert reconstructed.edge == original.edge
        assert reconstructed.metadata == original.metadata

    def test_signal_action_enum_values(self):
        """SignalAction enum should have expected values."""
        assert SignalAction.BUY.value == "BUY"
        assert SignalAction.SELL.value == "SELL"
        assert SignalAction.HOLD.value == "HOLD"
        assert SignalAction.EXIT.value == "EXIT"
        assert SignalAction.RISK_EXIT.value == "RISK_EXIT"

    def test_signal_decision_timestamp_utc(self):
        """SignalDecision timestamp should be UTC."""
        signal = SignalDecision(
            strategy_name="test",
            symbol="TEST",
            action=SignalAction.HOLD,
        )

        assert signal.timestamp.tzinfo == timezone.utc

    def test_signal_decision_confidence_validation(self):
        """SignalDecision confidence should be validated (0-1)."""
        # Valid confidence
        signal = SignalDecision(
            strategy_name="test",
            symbol="TEST",
            action=SignalAction.BUY,
            confidence=0.5,
        )
        assert signal.confidence == 0.5

        # Invalid confidence should raise
        with pytest.raises(ValueError):
            SignalDecision(
                strategy_name="test",
                symbol="TEST",
                action=SignalAction.BUY,
                confidence=1.5,  # > 1.0
            )

        with pytest.raises(ValueError):
            SignalDecision(
                strategy_name="test",
                symbol="TEST",
                action=SignalAction.BUY,
                confidence=-0.1,  # < 0.0
            )


class TestAggregatedDecisionSerialization:
    """Test AggregatedDecision serialization."""

    def test_aggregated_decision_to_dict(self):
        """AggregatedDecision should serialize to JSON-safe dict."""
        votes = [
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s2", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s3", symbol="BTC", action=SignalAction.HOLD),
        ]

        decision = AggregatedDecision(
            symbol="BTC",
            final_action=SignalAction.BUY,
            votes=votes,
            reason="Majority vote: BUY (2 votes)",
        )

        result = decision.to_dict()

        assert result["symbol"] == "BTC"
        assert result["final_action"] == "BUY"
        assert len(result["votes"]) == 3
        assert result["reason"] == "Majority vote: BUY (2 votes)"

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_aggregated_decision_votes_serialized(self):
        """AggregatedDecision.to_dict() should serialize nested votes."""
        votes = [
            SignalDecision(
                strategy_name="test",
                symbol="SYM",
                action=SignalAction.SELL,
                metadata={"test": True},
            ),
        ]

        decision = AggregatedDecision(
            symbol="SYM",
            final_action=SignalAction.SELL,
            votes=votes,
            reason="Single vote",
        )

        result = decision.to_dict()

        assert result["votes"][0]["strategy_name"] == "test"
        assert result["votes"][0]["action"] == "SELL"
        assert result["votes"][0]["metadata"] == {"test": True}
