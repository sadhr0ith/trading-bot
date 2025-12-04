from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from models.paper_state import PaperState
from utils.logger import setup_logger
from utils.risk_management import Position, RiskManager


class PaperTradingExecutor:
    def __init__(
        self,
        state_path: Optional[Path | str] = None,
        initial_balance: float = 100_000.0,
        strategy_name: Optional[str] = None,
    ):
        # Generate default path based on strategy name if not provided
        if state_path is None:
            if strategy_name:
                state_path = Path(f"paper_trading_state_{strategy_name}.json")
            else:
                state_path = Path("paper_trading_state.json")

        self.state_path = Path(state_path)
        self.initial_balance = initial_balance
        self.strategy_name = strategy_name
        self.logger = setup_logger(self.__class__.__name__)
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
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

                validated = PaperState.parse_obj(state)
                return validated.dict()
            except Exception:  # noqa: BLE001
                self.logger.warning("Stored paper trading state invalid; resetting to initial state.")
                return self._create_initial_state()
        return self._create_initial_state()

    def _create_initial_state(self) -> Dict:
        """Create initial state with strategy_name if provided."""
        state = {"balance": self.initial_balance, "positions": {}, "history": []}
        if self.strategy_name:
            state["strategy_name"] = self.strategy_name
        return state

    def _save_state(self) -> None:
        try:
            validated = PaperState.parse_obj(self.state)
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"Paper trading state failed validation before save: {exc}")
            validated = PaperState(balance=self.state.get("balance", self.initial_balance), positions={}, history=[], strategy_name=self.strategy_name)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(validated.dict(), f, indent=2, default=str)

    def _current_position(self, symbol: str) -> Optional[Dict]:
        return self.state.get("positions", {}).get(symbol)

    def _close_position(self, symbol: str, price: float, reason: str, risk_manager: RiskManager) -> Dict:
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
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
        self.state["positions"].pop(symbol, None)
        self._save_state()
        return {"status": "closed", "pnl": pnl, "reason": reason}

    def process_signal(self, symbol: str, signal: str, price: float, risk_manager: RiskManager) -> Dict:
        """
        Execute BUY/SELL/HOLD in paper-trading mode with simple risk controls.
        """
        position = self._current_position(symbol)
        summary: Dict = {"status": "noop", "balance": self.state["balance"]}

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
                self.logger.error(
                    f"Insufficient balance: need {notional + fee:.2f}, have {self.state['balance']:.2f}"
                )
                summary["status"] = "insufficient_balance"
                return summary

            self.state["positions"][symbol] = {
                "entry_price": price,
                "size": size,
                "entry_notional": notional,
                "stop_loss": risk_manager.stop_loss,
                "take_profit": risk_manager.take_profit,
                "opened_at": datetime.utcnow().isoformat(),
            }
            self.state["balance"] -= notional + fee
            self.state["history"].append(
                {
                    "symbol": symbol,
                    "action": "BUY",
                    "price": price,
                    "size": size,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            self._save_state()
            summary.update({"status": "opened", "size": size})
            return summary

        if signal == "SELL":
            if not position:
                summary["status"] = "no_position_to_sell"
                return summary
            return self._close_position(symbol, price, reason="manual_sell", risk_manager=risk_manager)

        if signal == "HOLD":
            if position and risk_manager.evaluate_exit(
                Position(
                    symbol=symbol,
                    entry_price=position["entry_price"],
                    size=position["size"],
                    stop_loss=position.get("stop_loss"),
                    take_profit=position.get("take_profit"),
                ),
                price,
            ):
                return self._close_position(symbol, price, reason="risk_exit", risk_manager=risk_manager)

        return summary
