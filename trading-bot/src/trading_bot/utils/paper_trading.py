from __future__ import annotations

import contextlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock, Timeout
from pydantic import ValidationError as PydanticValidationError

from trading_bot.models.paper_state import PaperState
from trading_bot.utils.logger import setup_logger
from trading_bot.utils.metrics import TradingMetrics
from trading_bot.utils.risk_management import Position, RiskManager


class PaperTradingExecutor:
    LOCK_TIMEOUT_SECONDS = 10
    STALE_LOCK_TTL_SECONDS = 300

    def __init__(
        self,
        state_path: Path | str | None = None,
        initial_balance: float = 100_000.0,
        strategy_name: str | None = None,
    ):
        base_dir = Path(os.getenv("TRADING_BOT_STATE_DIR", ".")).resolve()
        # Generate default path based on strategy name if not provided
        if state_path is None:
            if strategy_name:
                state_path = base_dir / f"paper_trading_state_{strategy_name}.json"
            else:
                state_path = base_dir / "paper_trading_state.json"
        else:
            state_path = Path(state_path)
            if not state_path.is_absolute():
                state_path = (base_dir / state_path).resolve()

        self.state_path = state_path
        self.initial_balance = initial_balance
        self.strategy_name = strategy_name
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(f"Paper trading state path: {self.state_path}")
        self.state = self._load_state()

    @contextlib.contextmanager
    def _file_lock(self, path: Path):
        """Context manager for file locking to prevent concurrent access.

        Args:
            path: Path to the file being locked

        Yields:
            None (lock is held during context)

        Example:
            >>> with self._file_lock(self.state_path):
            ...     # Critical section - file operations
            ...     pass
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        start = time.monotonic()
        if lock_path.exists():
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > self.STALE_LOCK_TTL_SECONDS:
                    self.logger.warning(
                        f"Stale paper-trading lock detected for {lock_path}; removing after {age:.1f}s."
                    )
                    lock_path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass

        file_lock = FileLock(str(lock_path))
        try:
            file_lock.acquire(timeout=self.LOCK_TIMEOUT_SECONDS)
            self.logger.debug(f"Acquired lock on {lock_path}")
        except Timeout as exc:
            raise TimeoutError(f"Timed out acquiring lock at {lock_path}") from exc

        try:
            yield
        finally:
            # Cleanup: silently ignore errors releasing lock (may already be released)
            try:
                if file_lock.is_locked:
                    file_lock.release()
                self.logger.debug(f"Released lock on {lock_path}")
            except (OSError, RuntimeError):
                pass
            # Remove lock file (may already be deleted by another process)
            try:
                lock_path.unlink()
            except OSError:
                pass

    def _load_state(self) -> dict:
        """Load paper trading state from file with file locking."""
        with self._file_lock(self.state_path):
            if self.state_path.exists():
                try:
                    with open(self.state_path, encoding="utf-8") as f:
                        state = json.load(f)

                    # Validate strategy_name if both are present
                    loaded_strategy = state.get("strategy_name")
                    if self.strategy_name and loaded_strategy and loaded_strategy != self.strategy_name:
                        logger = setup_logger(self.__class__.__name__)
                        logger.warning(
                            f"State file strategy mismatch: file has '{loaded_strategy}', "
                            f"current is '{self.strategy_name}'. This may indicate state file reuse across strategies."
                        )

                    # Ensure strategy_name is set in state
                    if self.strategy_name:
                        state["strategy_name"] = self.strategy_name

                    validated = PaperState.model_validate(state)
                    return validated.model_dump(exclude_none=True)
                except (json.JSONDecodeError, PydanticValidationError, OSError) as exc:
                    self.logger.warning(f"Stored paper trading state invalid ({exc}); resetting to initial state.")
                    return self._create_initial_state()
            return self._create_initial_state()

    def _create_initial_state(self) -> dict:
        """Create initial state with strategy_name if provided."""
        state = {"balance": self.initial_balance, "positions": {}, "history": []}
        if self.strategy_name:
            state["strategy_name"] = self.strategy_name
        return state

    # Drawdown warning threshold (20%)
    DRAWDOWN_WARNING_THRESHOLD = 0.20

    def _save_state(self) -> None:
        """Save paper trading state to file with file locking.

        Also updates drawdown metrics and logs warnings for high drawdown.
        """
        try:
            state_copy = dict(self.state)
            if state_copy.get("strategy_name") is None:
                state_copy.pop("strategy_name", None)
            validated = PaperState.model_validate(state_copy)

            # Sync drawdown metrics back to internal state
            self.state["peak_balance"] = validated.peak_balance
            self.state["max_drawdown"] = validated.max_drawdown

            # Log warning if drawdown exceeds threshold
            if validated.max_drawdown >= self.DRAWDOWN_WARNING_THRESHOLD:
                self.logger.warning(
                    f"High drawdown alert: {validated.max_drawdown:.1%} "
                    f"(peak: {validated.peak_balance:.2f}, current: {validated.balance:.2f})"
                )

            # Record metrics
            if self.strategy_name:
                TradingMetrics.update_portfolio_value(self.strategy_name, validated.balance)
                TradingMetrics.update_max_drawdown(self.strategy_name, validated.max_drawdown)

        except PydanticValidationError as exc:
            self.logger.error(f"Paper trading state failed validation before save: {exc}")
            validated = PaperState(
                balance=self.state.get("balance", self.initial_balance),
                positions={},
                history=[],
                strategy_name=self.strategy_name,
            )

        with self._file_lock(self.state_path):
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(validated.model_dump(exclude_none=True), f, indent=2, default=str)

    def _current_position(self, symbol: str) -> dict | None:
        return self.state.get("positions", {}).get(symbol)

    def get_drawdown_metrics(self) -> dict:
        """Get current drawdown metrics.

        Returns:
            dict with keys:
                - balance: Current balance
                - peak_balance: Highest balance reached
                - max_drawdown: Maximum drawdown as fraction (0.0-1.0)
                - current_drawdown: Current drawdown from peak
        """
        balance = self.state.get("balance", self.initial_balance)
        peak = self.state.get("peak_balance", balance)
        max_dd = self.state.get("max_drawdown", 0.0)

        current_dd = (peak - balance) / peak if peak > 0 else 0.0

        return {
            "balance": balance,
            "peak_balance": peak,
            "max_drawdown": max_dd,
            "current_drawdown": current_dd,
        }

    def _close_position(self, symbol: str, price: float, reason: str, risk_manager: RiskManager) -> dict:
        position = self._current_position(symbol)
        if not position:
            return {"status": "no_position"}

        entry_price = position["entry_price"]
        size = position["size"]

        # Calculate exit notional and fee
        notional = price * size
        exit_fee = notional * risk_manager.trading_fee

        # Return notional minus exit fee
        self.state["balance"] += notional - exit_fee

        # Calculate net PnL after both entry and exit fees
        entry_notional = position.get("entry_notional", entry_price * size)
        entry_fee = entry_notional * risk_manager.trading_fee
        pnl = (price - entry_price) * size - entry_fee - exit_fee
        self.state["history"].append(
            {
                "symbol": symbol,
                "action": "CLOSE",
                "price": price,
                "size": size,
                "pnl": pnl,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.state["positions"].pop(symbol, None)
        self._save_state()
        # Record trade metric
        if self.strategy_name:
            TradingMetrics.record_trade(self.strategy_name, "SELL")
        return {"status": "closed", "pnl": pnl, "reason": reason}

    def process_signal(self, symbol: str, signal: str, price: float, risk_manager: RiskManager) -> dict:
        """
        Execute BUY/SELL/HOLD in paper-trading mode with simple risk controls.
        """
        position = self._current_position(symbol)
        summary: dict = {"status": "noop", "balance": self.state["balance"]}
        state_dirty = False

        if signal == "BUY":
            if position:
                summary["status"] = "already_long"
                return summary

            size = risk_manager.calculate_position_size(self.state["balance"], price)
            if size <= 0:
                summary["status"] = "insufficient_balance"
                return summary

            notional = price * size
            fee = notional * risk_manager.trading_fee

            # Validate sufficient balance before executing
            if self.state["balance"] < notional + fee:
                self.logger.error(f"Insufficient balance: need {notional + fee:.2f}, have {self.state['balance']:.2f}")
                summary["status"] = "insufficient_balance"
                return summary

            self.state["positions"][symbol] = {
                "symbol": symbol,
                "entry_price": price,
                "size": size,
                "entry_notional": notional,
                "stop_loss": risk_manager.stop_loss,
                "take_profit": risk_manager.take_profit,
                "peak_price": price,
                "opened_at": datetime.now(timezone.utc).isoformat(),
            }
            self.state["balance"] -= notional + fee
            self.state["history"].append(
                {
                    "symbol": symbol,
                    "action": "BUY",
                    "price": price,
                    "size": size,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            self._save_state()
            # Record trade metric
            if self.strategy_name:
                TradingMetrics.record_trade(self.strategy_name, "BUY")
            summary.update({"status": "opened", "size": size})
            return summary

        if signal == "SELL":
            if not position:
                summary["status"] = "no_position_to_sell"
                return summary
            return self._close_position(symbol, price, reason="manual_sell", risk_manager=risk_manager)

        if signal == "HOLD":
            if position:
                # update trailing peak
                peak = position.get("peak_price", position["entry_price"])
                peak = max(peak, price)
                previous_peak = position.get("peak_price", position["entry_price"])
                if peak != previous_peak:
                    position["peak_price"] = peak
                    state_dirty = True
                if risk_manager.evaluate_exit(
                    Position(
                        symbol=symbol,
                        entry_price=position["entry_price"],
                        size=position["size"],
                        stop_loss=position.get("stop_loss"),
                        take_profit=position.get("take_profit"),
                        peak_price=peak,
                    ),
                    price,
                ):
                    return self._close_position(symbol, price, reason="risk_exit", risk_manager=risk_manager)

        if state_dirty:
            self._save_state()

        return summary
