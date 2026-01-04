"""Tests for multi-strategy shared state functionality."""
from __future__ import annotations

import json
from pathlib import Path

from trading_bot.utils.paper_trading import PaperTradingExecutor
from trading_bot.utils.risk_management import RiskManager


class TestSharedState:
    """Test that multi-strategy mode uses a single shared state file."""

    def test_single_shared_state_file(self, tmp_path: Path):
        """Multiple executors pointing to same file should share state."""
        state_file = tmp_path / "shared_state.json"

        # Create two executors pointing to the same state file
        executor1 = PaperTradingExecutor(
            state_path=str(state_file),
            initial_balance=10000.0,
        )
        executor2 = PaperTradingExecutor(
            state_path=str(state_file),
            initial_balance=10000.0,
        )

        risk_manager = RiskManager(risk_config={
            "stop_loss": 0.03,
            "take_profit": 0.05,
            "max_position_size": 0.1,
            "trading_fee": 0.001,
        })

        # Executor 1 opens a position
        result = executor1.process_signal("BTCUSDT", "BUY", 50000.0, risk_manager)
        assert result["status"] == "opened"

        # Reload executor 2's state
        executor2.state = executor2._load_state()

        # Executor 2 should see the position opened by executor 1
        position = executor2._current_position("BTCUSDT")
        assert position is not None
        assert position["entry_price"] == 50000.0

    def test_shared_state_netted_positions(self, tmp_path: Path):
        """Shared state should net positions per symbol."""
        state_file = tmp_path / "shared_state.json"

        executor = PaperTradingExecutor(
            state_path=str(state_file),
            initial_balance=100000.0,
        )

        risk_manager = RiskManager(risk_config={
            "stop_loss": 0.03,
            "take_profit": 0.05,
            "max_position_size": 0.1,
            "trading_fee": 0.001,
        })

        # Open position
        executor.process_signal("BTCUSDT", "BUY", 50000.0, risk_manager)

        # Another BUY should not open a second position (already_long)
        result = executor.process_signal("BTCUSDT", "BUY", 51000.0, risk_manager)
        assert result["status"] == "already_long"

        # Only one position should exist
        assert len(executor.state.get("positions", {})) == 1

    def test_vote_logging_in_history(self, tmp_path: Path):
        """History entries should include vote information when provided."""
        state_file = tmp_path / "shared_state.json"

        executor = PaperTradingExecutor(
            state_path=str(state_file),
            initial_balance=100000.0,
        )

        risk_manager = RiskManager(risk_config={
            "stop_loss": 0.03,
            "take_profit": 0.05,
            "max_position_size": 0.1,
            "trading_fee": 0.001,
        })

        votes = [
            {
                "strategy_name": "long_term",
                "symbol": "BTCUSDT",
                "action": "BUY",
                "confidence": 0.8,
            },
            {
                "strategy_name": "mid_term",
                "symbol": "BTCUSDT",
                "action": "BUY",
                "confidence": 0.7,
            },
        ]

        executor.process_signal(
            "BTCUSDT",
            "BUY",
            50000.0,
            risk_manager,
            votes=votes,
            aggregation_reason="Majority vote: BUY (2 votes)",
        )

        # Check history has vote information
        history = executor.state.get("history", [])
        assert len(history) == 1

        entry = history[0]
        assert entry["action"] == "BUY"
        assert "votes" in entry
        assert len(entry["votes"]) == 2
        assert entry["aggregation_reason"] == "Majority vote: BUY (2 votes)"

    def test_state_file_persistence(self, tmp_path: Path):
        """State changes should persist to disk."""
        state_file = tmp_path / "shared_state.json"

        executor = PaperTradingExecutor(
            state_path=str(state_file),
            initial_balance=100000.0,
        )

        risk_manager = RiskManager(risk_config={
            "max_position_size": 0.1,
            "trading_fee": 0.001,
        })

        # Execute a trade
        executor.process_signal("BTCUSDT", "BUY", 50000.0, risk_manager)

        # Read the file directly
        with open(state_file, "r", encoding="utf-8") as f:
            saved_state = json.load(f)

        assert "positions" in saved_state
        assert "BTCUSDT" in saved_state["positions"]
        assert saved_state["positions"]["BTCUSDT"]["entry_price"] == 50000.0

    def test_multiple_symbols_shared_state(self, tmp_path: Path):
        """Shared state should handle multiple symbols correctly."""
        state_file = tmp_path / "shared_state.json"

        executor = PaperTradingExecutor(
            state_path=str(state_file),
            initial_balance=100000.0,
        )

        risk_manager = RiskManager(risk_config={
            "max_position_size": 0.05,  # Smaller position to fit multiple trades
            "trading_fee": 0.001,
        })

        # Open positions in multiple symbols
        executor.process_signal("BTCUSDT", "BUY", 50000.0, risk_manager)
        executor.process_signal("ETHUSDT", "BUY", 3000.0, risk_manager)

        positions = executor.state.get("positions", {})
        assert len(positions) == 2
        assert "BTCUSDT" in positions
        assert "ETHUSDT" in positions

        # Close one position
        executor.process_signal("BTCUSDT", "SELL", 51000.0, risk_manager)

        positions = executor.state.get("positions", {})
        assert len(positions) == 1
        assert "BTCUSDT" not in positions
        assert "ETHUSDT" in positions

    def test_shared_executor_no_strategy_name_mismatch_warning(self, tmp_path: Path):
        """Shared executor should not have strategy name conflicts."""
        state_file = tmp_path / "shared_state.json"

        # Create executor without strategy_name (multi-strategy mode)
        executor = PaperTradingExecutor(
            state_path=str(state_file),
            initial_balance=100000.0,
            strategy_name=None,  # Multi-strategy uses shared state without specific name
        )

        # State should not have strategy_name
        assert executor.strategy_name is None


class TestSharedStateIsolation:
    """Test that different state files remain isolated."""

    def test_different_state_files_isolated(self, tmp_path: Path):
        """Different state files should not affect each other."""
        state_file1 = tmp_path / "state1.json"
        state_file2 = tmp_path / "state2.json"

        executor1 = PaperTradingExecutor(
            state_path=str(state_file1),
            initial_balance=100000.0,
        )
        executor2 = PaperTradingExecutor(
            state_path=str(state_file2),
            initial_balance=100000.0,
        )

        risk_manager = RiskManager(risk_config={
            "max_position_size": 0.1,
            "trading_fee": 0.001,
        })

        # Execute trade on executor1
        executor1.process_signal("BTCUSDT", "BUY", 50000.0, risk_manager)

        # Executor2 should not have this position
        assert executor2._current_position("BTCUSDT") is None

        # Executor1 should have it
        assert executor1._current_position("BTCUSDT") is not None
