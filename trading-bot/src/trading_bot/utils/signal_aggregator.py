"""Signal aggregation for multi-strategy trading.

This module provides logic to combine signals from multiple strategies into
a single actionable decision per symbol. The aggregation is deterministic
and purely data-driven (no dependency on strategy classes).
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from trading_bot.models.signal import AggregatedDecision, SignalAction, SignalDecision
from trading_bot.utils.logger import setup_logger


class SignalAggregator:
    """
    Aggregates signals from multiple strategies into final trading decisions.

    Aggregation Policy (deterministic, priority-based):
    1. RISK_EXIT signals have highest priority - always exit
    2. EXIT signals have second priority - close position
    3. Long-term filter: if long_term strategy signals NO_TRADE -> HOLD
    4. Mid-term directional bias: if mid_term is BUY, block SELL (and vice versa)
    5. Majority vote among remaining non-filtered signals
    6. Tie -> HOLD

    The aggregator operates purely on SignalDecision data, with no direct
    dependency on strategy implementations. This enables future multiprocess
    signal collection where signals arrive as serialized data.
    """

    # Strategy names that act as filters (case-insensitive matching)
    LONG_TERM_STRATEGIES = {"long_term"}
    MID_TERM_STRATEGIES = {"mid_term"}

    def __init__(
        self,
        enable_long_term_filter: bool = True,
        enable_mid_term_bias: bool = True,
    ) -> None:
        """
        Initialize the aggregator with optional filter toggles.

        Args:
            enable_long_term_filter: If True, long_term HOLD/SELL blocks all trades
            enable_mid_term_bias: If True, mid_term direction filters opposing signals
        """
        self.enable_long_term_filter = enable_long_term_filter
        self.enable_mid_term_bias = enable_mid_term_bias
        self.logger = setup_logger(self.__class__.__name__)

    def aggregate(
        self,
        signals: list[SignalDecision],
        has_position: Callable[[str], bool] | None = None,
    ) -> list[AggregatedDecision]:
        """
        Aggregate signals by symbol and return final decisions.

        Args:
            signals: List of SignalDecision from various strategies
            has_position: Optional callback that returns True if a position exists for the
                given symbol. Used to make long_term SELL position-aware: if a position
                exists, SELL becomes EXIT; otherwise it acts as a filter (HOLD).

        Returns:
            List of AggregatedDecision, one per unique symbol
        """
        if not signals:
            return []

        # Group signals by symbol
        by_symbol: dict[str, list[SignalDecision]] = {}
        for sig in signals:
            by_symbol.setdefault(sig.symbol, []).append(sig)

        results: list[AggregatedDecision] = []
        for symbol, symbol_signals in by_symbol.items():
            decision = self._aggregate_symbol(symbol, symbol_signals, has_position)
            results.append(decision)

        return results

    def _aggregate_symbol(
        self,
        symbol: str,
        signals: list[SignalDecision],
        has_position: Callable[[str], bool] | None = None,
    ) -> AggregatedDecision:
        """
        Aggregate signals for a single symbol.

        Implements the priority-based aggregation policy.
        """
        # Priority 1: Check for RISK_EXIT (highest priority)
        risk_exits = [s for s in signals if s.action == SignalAction.RISK_EXIT]
        if risk_exits:
            return AggregatedDecision(
                symbol=symbol,
                final_action=SignalAction.RISK_EXIT,
                votes=signals,
                reason=f"RISK_EXIT from {[s.strategy_name for s in risk_exits]}",
            )

        # Priority 2: Check for EXIT signals
        exits = [s for s in signals if s.action == SignalAction.EXIT]
        if exits:
            return AggregatedDecision(
                symbol=symbol,
                final_action=SignalAction.EXIT,
                votes=signals,
                reason=f"EXIT from {[s.strategy_name for s in exits]}",
            )

        # Priority 3: Long-term filter
        if self.enable_long_term_filter:
            long_term_signals = [
                s for s in signals if s.strategy_name.lower() in self.LONG_TERM_STRATEGIES
            ]
            for lt_sig in long_term_signals:
                if lt_sig.action == SignalAction.SELL:
                    # Position-aware: if position exists, long_term SELL triggers EXIT
                    if has_position is not None and has_position(symbol):
                        return AggregatedDecision(
                            symbol=symbol,
                            final_action=SignalAction.EXIT,
                            votes=signals,
                            reason=f"Long-term SELL -> EXIT (position exists for {symbol})",
                        )
                    # No position or no callback: filter only (block new trades)
                    return AggregatedDecision(
                        symbol=symbol,
                        final_action=SignalAction.HOLD,
                        votes=signals,
                        reason=f"Long-term filter: {lt_sig.strategy_name} signaled SELL (no position)",
                    )
                if lt_sig.action == SignalAction.HOLD:
                    return AggregatedDecision(
                        symbol=symbol,
                        final_action=SignalAction.HOLD,
                        votes=signals,
                        reason=f"Long-term filter: {lt_sig.strategy_name} signaled HOLD",
                    )

        # Priority 4: Mid-term directional bias
        mid_term_bias: SignalAction | None = None
        if self.enable_mid_term_bias:
            mid_term_signals = [
                s for s in signals if s.strategy_name.lower() in self.MID_TERM_STRATEGIES
            ]
            for mt_sig in mid_term_signals:
                if mt_sig.action == SignalAction.BUY:
                    mid_term_bias = SignalAction.BUY
                elif mt_sig.action == SignalAction.SELL:
                    mid_term_bias = SignalAction.SELL
                # HOLD doesn't set a bias

        # Collect voting signals (exclude long_term and mid_term from vote, they are filters)
        filter_strategies = self.LONG_TERM_STRATEGIES | self.MID_TERM_STRATEGIES
        voting_signals = [
            s for s in signals
            if s.strategy_name.lower() not in filter_strategies
            and s.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD)
        ]

        # Apply mid-term bias filter
        if mid_term_bias is not None:
            filtered_voting = []
            for vs in voting_signals:
                # Block opposing directions
                if mid_term_bias == SignalAction.BUY and vs.action == SignalAction.SELL:
                    continue  # Blocked by bullish bias
                if mid_term_bias == SignalAction.SELL and vs.action == SignalAction.BUY:
                    continue  # Blocked by bearish bias
                filtered_voting.append(vs)
            voting_signals = filtered_voting

        # Priority 5: Majority vote
        if not voting_signals:
            # No voting signals left after filtering
            reason = "No actionable signals after filtering"
            if mid_term_bias:
                reason += f" (mid-term bias: {mid_term_bias.value})"
            return AggregatedDecision(
                symbol=symbol,
                final_action=SignalAction.HOLD,
                votes=signals,
                reason=reason,
            )

        # Count votes (only BUY and SELL participate in voting; HOLD is neutral)
        action_counts: Counter[SignalAction] = Counter()
        for vs in voting_signals:
            if vs.action in (SignalAction.BUY, SignalAction.SELL):
                action_counts[vs.action] += 1

        if not action_counts:
            # All voting signals were HOLD
            return AggregatedDecision(
                symbol=symbol,
                final_action=SignalAction.HOLD,
                votes=signals,
                reason="All voting strategies signaled HOLD",
            )

        # Find the winner
        most_common = action_counts.most_common()
        top_action, top_count = most_common[0]

        # Check for tie
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            return AggregatedDecision(
                symbol=symbol,
                final_action=SignalAction.HOLD,
                votes=signals,
                reason=f"Tie between {[a.value for a, _ in most_common[:2]]} ({top_count} votes each)",
            )

        # Clear winner
        vote_breakdown = ", ".join(f"{a.value}={c}" for a, c in most_common)
        reason = f"Majority vote: {top_action.value} ({vote_breakdown})"
        if mid_term_bias:
            reason += f" [mid-term bias: {mid_term_bias.value}]"

        return AggregatedDecision(
            symbol=symbol,
            final_action=top_action,
            votes=signals,
            reason=reason,
        )

    def log_aggregation(self, decisions: list[AggregatedDecision]) -> None:
        """Log aggregation results for debugging/auditing."""
        for dec in decisions:
            vote_summary = ", ".join(
                f"{v.strategy_name}={v.action.value}" for v in dec.votes
            )
            self.logger.info(
                f"[{dec.symbol}] Final: {dec.final_action.value} | "
                f"Votes: [{vote_summary}] | Reason: {dec.reason}"
            )
