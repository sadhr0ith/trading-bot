import pytest

from trading_bot.utils.risk_management import Position, RiskManager


def test_trailing_stop_triggers_below_peak():
    rm = RiskManager({"max_position_size": 0.1, "trading_fee": 0.0, "trailing_stop": 0.05})
    pos = Position(symbol="T", entry_price=100.0, size=1.0, stop_loss=None, take_profit=None, peak_price=110.0)
    assert rm.evaluate_exit(pos, 104.0) is True  # 5% trail below peak


def test_trailing_stop_does_not_trigger_above_threshold():
    rm = RiskManager({"max_position_size": 0.1, "trading_fee": 0.0, "trailing_stop": 0.05})
    pos = Position(symbol="T", entry_price=100.0, size=1.0, stop_loss=None, take_profit=None, peak_price=110.0)
    assert rm.evaluate_exit(pos, 107.0) is False
