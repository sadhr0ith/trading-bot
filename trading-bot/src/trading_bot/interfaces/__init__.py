"""Interfaces for extensibility and future multiprocessing support."""
from trading_bot.interfaces.signal_source import InProcessSignalSource, SignalSource

__all__ = ["SignalSource", "InProcessSignalSource"]
