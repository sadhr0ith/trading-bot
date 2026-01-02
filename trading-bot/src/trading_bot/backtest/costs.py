"""Cost model helpers for backtests."""

from __future__ import annotations


def fill_price(price: float, side: str, slippage_rate: float) -> float:
    """Return fill price after applying slippage for a given side."""
    side_upper = side.upper()
    if side_upper == "BUY":
        return price * (1.0 + slippage_rate)
    if side_upper == "SELL":
        return price * (1.0 - slippage_rate)
    return price


def fee_from_notional(notional: float, fee_rate: float) -> float:
    """Return fee amount for a notional value."""
    return notional * fee_rate


def split_notional_for_fee(cash: float, fee_rate: float) -> float:
    """Return trade notional that fully uses cash after fees.

    For buys we want: notional + fee = cash. So notional = cash / (1 + fee_rate).
    """
    if fee_rate <= 0:
        return cash
    return cash / (1.0 + fee_rate)
