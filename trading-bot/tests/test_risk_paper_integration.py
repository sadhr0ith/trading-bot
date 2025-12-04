import pandas as pd
import tempfile
from pathlib import Path

from utils.paper_trading import PaperTradingExecutor
from utils.risk_management import RiskManager


def test_risk_and_executor_integration_buy_sell_hold():
    # Use unique state file for isolation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_path = f.name

    # Fix: use correct config keys (stop_loss, not stop_loss_pct) and reasonable max_position_size
    rm = RiskManager({"max_position_size": 0.1, "stop_loss": 0.05, "take_profit": 0.1, "trading_fee": 0.001})
    executor = PaperTradingExecutor(state_path=state_path, initial_balance=100_000.0)
    price = 100.0

    buy_summary = executor.process_signal("TEST", "BUY", price, rm)
    assert buy_summary["status"] in {"opened", "already_long", "noop"}

    hold_summary = executor.process_signal("TEST", "HOLD", price, rm)
    assert hold_summary["status"] in {"noop", "already_long"}

    sell_summary = executor.process_signal("TEST", "SELL", price, rm)
    assert sell_summary["status"] in {"closed", "noop"}

    # Cleanup
    Path(state_path).unlink(missing_ok=True)


def test_buy_cash_accounting():
    """Test 1.4: BUY 1 BTC @ 50k with fee 0.001 - verify balance accounting"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_path = f.name

    # To buy exactly 1 BTC @ 50k, we need max_position_size to calculate to 1.0
    # size = (balance * max_position) / price
    # 1.0 = (100_000 * max_position) / 50_000
    # max_position = 1.0 * 50_000 / 100_000 = 0.5 (50% of balance)
    rm = RiskManager({"max_position_size": 0.5, "trading_fee": 0.001, "stop_loss": 0.03, "take_profit": 0.05})
    executor = PaperTradingExecutor(state_path=state_path, initial_balance=100_000.0)

    # BUY 1 BTC @ 50,000
    result = executor.process_signal("BTCUSDT", "BUY", 50_000.0, rm)

    assert result["status"] == "opened"
    assert result["size"] == 1.0

    # Expected balance: 100,000 - notional - fee
    # notional = 50,000 * 1.0 = 50,000
    # fee = 50,000 * 0.001 = 50
    # balance = 100,000 - 50,000 - 50 = 49,950
    assert executor.state["balance"] == 49_950.0

    Path(state_path).unlink(missing_ok=True)


def test_sell_cash_accounting():
    """Test 1.5: SELL 1 BTC @ 55k with fee 0.001 - verify balance accounting"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_path = f.name

    rm = RiskManager({"max_position_size": 0.5, "trading_fee": 0.001, "stop_loss": 0.03, "take_profit": 0.05})
    executor = PaperTradingExecutor(state_path=state_path, initial_balance=100_000.0)

    # First BUY 1 BTC @ 50,000
    executor.process_signal("BTCUSDT", "BUY", 50_000.0, rm)
    assert executor.state["balance"] == 49_950.0

    # Now SELL 1 BTC @ 55,000
    result = executor.process_signal("BTCUSDT", "SELL", 55_000.0, rm)

    assert result["status"] == "closed"

    # Expected balance: 49,950 + notional - fee
    # notional = 55,000 * 1.0 = 55,000
    # fee = 55,000 * 0.001 = 55
    # balance = 49,950 + 55,000 - 55 = 104,895
    assert executor.state["balance"] == 104_895.0

    # Expected PnL in history: (55k - 50k) * 1.0 - entry_fee - exit_fee
    # = 5,000 - 50 - 55 = 4,895
    assert result["pnl"] == 4_895.0

    Path(state_path).unlink(missing_ok=True)


def test_full_cycle_buy_sell_integration():
    """Test 1.6: Full cycle BUY@50k→SELL@55k - integration test"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_path = f.name

    rm = RiskManager({"max_position_size": 0.5, "trading_fee": 0.001, "stop_loss": 0.03, "take_profit": 0.05})
    executor = PaperTradingExecutor(state_path=state_path, initial_balance=100_000.0)

    # Initial state
    assert executor.state["balance"] == 100_000.0
    assert len(executor.state["positions"]) == 0
    assert len(executor.state["history"]) == 0

    # BUY 1 BTC @ 50,000
    buy_result = executor.process_signal("BTCUSDT", "BUY", 50_000.0, rm)
    assert buy_result["status"] == "opened"
    assert executor.state["balance"] == 49_950.0
    assert len(executor.state["positions"]) == 1
    assert len(executor.state["history"]) == 1

    # SELL 1 BTC @ 55,000
    sell_result = executor.process_signal("BTCUSDT", "SELL", 55_000.0, rm)
    assert sell_result["status"] == "closed"
    assert executor.state["balance"] == 104_895.0
    assert len(executor.state["positions"]) == 0  # Position closed
    assert len(executor.state["history"]) == 2  # BUY + CLOSE entries

    # Net profit: 104,895 - 100,000 = 4,895
    net_profit = executor.state["balance"] - 100_000.0
    assert net_profit == 4_895.0

    # Verify history entries
    assert executor.state["history"][0]["action"] == "BUY"
    assert executor.state["history"][1]["action"] == "CLOSE"
    assert executor.state["history"][1]["pnl"] == 4_895.0

    Path(state_path).unlink(missing_ok=True)


def test_insufficient_balance():
    """Test 1.7: BUY with insufficient balance - should fail gracefully"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_path = f.name

    rm = RiskManager({"max_position_size": 1.0, "trading_fee": 0.001, "stop_loss": 0.03, "take_profit": 0.05})
    executor = PaperTradingExecutor(state_path=state_path, initial_balance=1_000.0)  # Only $1,000

    # Try to BUY BTC @ 50,000 - requires 50,000 + fee but only have 1,000
    result = executor.process_signal("BTCUSDT", "BUY", 50_000.0, rm)

    assert result["status"] == "insufficient_balance"
    assert executor.state["balance"] == 1_000.0  # Balance unchanged
    assert len(executor.state["positions"]) == 0  # No position opened
    assert len(executor.state["history"]) == 0  # No history entry

    Path(state_path).unlink(missing_ok=True)


def test_risk_manager_rejects_unknown_keys_gracefully():
    """RiskManager should ignore/replace invalid keys via Pydantic validation."""
    rm = RiskManager({
        "stop_loss": 0.03,
        "take_profit": 0.05,
        "max_position_size": 0.1,
        "trading_fee": 0.001,
        "unknown_key": 123,
    })

    assert rm.stop_loss == 0.03
    assert rm.take_profit == 0.05
    assert rm.max_position == 0.1
    assert rm.trading_fee == 0.001
