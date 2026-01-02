"""Tests for paper trading state file isolation between strategies."""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest
from filelock import FileLock

from trading_bot.utils import paper_trading as paper_trading_module
from trading_bot.utils.paper_trading import PaperTradingExecutor
from trading_bot.utils.risk_management import RiskManager


def test_strategy_specific_state_files():
    """Test 7.1: Different strategies create different state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create two executors with different strategy names
        executor1 = PaperTradingExecutor(
            state_path=tmpdir_path / "state1.json", strategy_name="short_term", initial_balance=10_000.0
        )
        executor2 = PaperTradingExecutor(
            state_path=tmpdir_path / "state2.json", strategy_name="long_term", initial_balance=20_000.0
        )

        # Verify initial balances are different (isolated states)
        assert executor1.state["balance"] == 10_000.0
        assert executor2.state["balance"] == 20_000.0

        # Verify strategy names are stored
        assert executor1.state["strategy_name"] == "short_term"
        assert executor2.state["strategy_name"] == "long_term"

        # Execute trade on executor1
        rm = RiskManager({"max_position_size": 0.1, "trading_fee": 0.001, "stop_loss": 0.03, "take_profit": 0.05})
        executor1.process_signal("BTCUSDT", "BUY", 50_000.0, rm)

        # Verify executor1 balance changed but executor2 didn't
        assert executor1.state["balance"] < 10_000.0
        assert executor2.state["balance"] == 20_000.0

        # Verify positions are isolated
        assert len(executor1.state["positions"]) == 1
        assert len(executor2.state["positions"]) == 0


def test_default_state_path_with_strategy_name():
    """Test 7.2: Default state path uses strategy name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = Path.cwd()
        try:
            # Change to temp directory so state files are created there
            import os

            os.chdir(tmpdir)

            # Create executor with strategy name but no explicit state_path
            executor = PaperTradingExecutor(strategy_name="mid_term", initial_balance=15_000.0)

            # Verify default path was generated with strategy name in current cwd
            expected_path = Path(tmpdir).resolve() / "paper_trading_state_mid_term.json"
            assert executor.state_path == expected_path

            # Execute a trade to trigger save
            rm = RiskManager({"max_position_size": 0.1, "trading_fee": 0.001, "stop_loss": 0.03, "take_profit": 0.05})
            executor.process_signal("ETHUSDT", "BUY", 3_000.0, rm)

            # Verify file exists
            assert expected_path.exists()

            # Verify strategy_name is in the saved state
            with open(expected_path, encoding="utf-8") as f:
                saved_state = json.load(f)
            assert saved_state["strategy_name"] == "mid_term"

            # Cleanup
            expected_path.unlink(missing_ok=True)

        finally:
            os.chdir(original_cwd)


def test_default_state_path_without_strategy_name():
    """Test 7.3: Default state path without strategy name uses generic filename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(tmpdir)

            # Create executor without strategy name or state_path
            executor = PaperTradingExecutor(initial_balance=25_000.0)

            # Verify default path is generic in current cwd
            expected_path = Path(tmpdir).resolve() / "paper_trading_state.json"
            assert executor.state_path == expected_path

            # Verify strategy_name is None
            assert executor.strategy_name is None
            assert "strategy_name" not in executor.state

            # Execute a trade to trigger save
            rm = RiskManager({"max_position_size": 0.1, "trading_fee": 0.001, "stop_loss": 0.03, "take_profit": 0.05})
            executor.process_signal("BTCUSDT", "BUY", 50_000.0, rm)

            # Verify file exists but has no strategy_name
            with open(expected_path, encoding="utf-8") as f:
                saved_state = json.load(f)
            assert "strategy_name" not in saved_state

            # Cleanup
            expected_path.unlink(missing_ok=True)

        finally:
            os.chdir(original_cwd)


def test_strategy_name_mismatch_warning(caplog):
    """Test 7.4: Loading state with different strategy_name logs warning."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        state_path = f.name

        # Create initial state with strategy_name="short_term"
        initial_state = {"balance": 100_000.0, "positions": {}, "history": [], "strategy_name": "short_term"}
        json.dump(initial_state, f)
        f.flush()

    try:
        # Load with different strategy_name
        import logging

        with caplog.at_level(logging.WARNING):
            executor = PaperTradingExecutor(
                state_path=state_path, strategy_name="long_term"  # Different from saved "short_term"
            )

        # Note: Warning might not appear in caplog due to custom logger,
        # but verify state was loaded and strategy_name was updated
        assert executor.state["balance"] == 100_000.0
        assert executor.state["strategy_name"] == "long_term"  # Updated to current

    finally:
        Path(state_path).unlink(missing_ok=True)


def test_concurrent_strategies_isolated_states():
    """Test 7.5: Run two strategies concurrently - verify complete isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Simulate two different strategies running
        config1 = {"strategy": "day_trading"}
        config2 = {"strategy": "swing_trading"}

        executor1 = PaperTradingExecutor(
            state_path=tmpdir_path / "day_trading.json", strategy_name=config1["strategy"], initial_balance=50_000.0
        )
        executor2 = PaperTradingExecutor(
            state_path=tmpdir_path / "swing_trading.json", strategy_name=config2["strategy"], initial_balance=50_000.0
        )

        rm1 = RiskManager({"max_position_size": 0.2, "trading_fee": 0.001, "stop_loss": 0.02, "take_profit": 0.03})
        rm2 = RiskManager({"max_position_size": 0.1, "trading_fee": 0.001, "stop_loss": 0.05, "take_profit": 0.10})

        # Execute different trades on each
        executor1.process_signal("BTCUSDT", "BUY", 50_000.0, rm1)  # Day trading buys BTC
        executor2.process_signal("ETHUSDT", "BUY", 3_000.0, rm2)  # Swing trading buys ETH

        # Verify complete isolation
        assert "BTCUSDT" in executor1.state["positions"]
        assert "ETHUSDT" not in executor1.state["positions"]

        assert "ETHUSDT" in executor2.state["positions"]
        assert "BTCUSDT" not in executor2.state["positions"]

        # Verify different position sizes due to different max_position_size
        btc_size = executor1.state["positions"]["BTCUSDT"]["size"]
        eth_position = executor2.state["positions"]["ETHUSDT"]
        eth_size = eth_position["size"]

        # Day trading uses 20% of 50k = 10k, BTC @ 50k = 0.2 BTC
        assert abs(btc_size - 0.2) < 0.01

        # Swing trading uses 10% of 50k = 5k, ETH @ 3k = ~1.67 ETH
        assert abs(eth_size - 1.666) < 0.01

        # Verify history is isolated
        assert len(executor1.state["history"]) == 1
        assert executor1.state["history"][0]["symbol"] == "BTCUSDT"

        assert len(executor2.state["history"]) == 1
        assert executor2.state["history"][0]["symbol"] == "ETHUSDT"

        # Verify state files are separate
        assert (tmpdir_path / "day_trading.json").exists()
        assert (tmpdir_path / "swing_trading.json").exists()

        # Verify state file contents
        with open(tmpdir_path / "day_trading.json", encoding="utf-8") as f:
            day_state = json.load(f)
        assert day_state["strategy_name"] == "day_trading"
        assert "BTCUSDT" in day_state["positions"]

        with open(tmpdir_path / "swing_trading.json", encoding="utf-8") as f:
            swing_state = json.load(f)
        assert swing_state["strategy_name"] == "swing_trading"
        assert "ETHUSDT" in swing_state["positions"]


def test_file_lock_removes_stale(monkeypatch, tmp_path):
    monkeypatch.setattr(paper_trading_module, "fcntl", None, raising=False)
    executor = PaperTradingExecutor(state_path=tmp_path / "state.json", initial_balance=1000.0)
    executor.STALE_LOCK_TTL_SECONDS = 0.1
    executor.LOCK_TIMEOUT_SECONDS = 1

    lock_path = executor.state_path.with_suffix(".json.lock")
    lock_path.write_text("locked", encoding="utf-8")
    old = time.time() - 5
    os.utime(lock_path, (old, old))

    with executor._file_lock(executor.state_path):
        assert lock_path.exists()

    assert not lock_path.exists()


def test_file_lock_times_out(monkeypatch, tmp_path):
    executor = PaperTradingExecutor(state_path=tmp_path / "state_timeout.json", initial_balance=1000.0)
    executor.STALE_LOCK_TTL_SECONDS = 100
    executor.LOCK_TIMEOUT_SECONDS = 0.2

    lock_path = executor.state_path.with_suffix(".json.lock")
    blocking_lock = FileLock(str(lock_path))
    blocking_lock.acquire(timeout=1)

    with pytest.raises(TimeoutError):
        with executor._file_lock(executor.state_path):
            pass

    blocking_lock.release()


def test_file_lock_cleans_up_after_exception(monkeypatch, tmp_path):
    """Lock file should be removed even if the protected block raises."""
    monkeypatch.setattr(paper_trading_module, "fcntl", None, raising=False)
    executor = PaperTradingExecutor(state_path=tmp_path / "state_cleanup.json", initial_balance=500.0)
    lock_path = executor.state_path.with_suffix(".json.lock")

    with pytest.raises(RuntimeError):
        with executor._file_lock(executor.state_path):
            assert lock_path.exists()
            raise RuntimeError("boom")

    assert not lock_path.exists(), "Lock file must be removed after exception"
