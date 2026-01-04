"""Tests for multi-strategy signal aggregation."""
from __future__ import annotations

from trading_bot.models.signal import SignalAction, SignalDecision
from trading_bot.utils.signal_aggregator import SignalAggregator


class TestSignalAggregation:
    """Test SignalAggregator aggregation logic."""

    def test_empty_signals_returns_empty(self):
        """Empty signal list should return empty aggregation list."""
        aggregator = SignalAggregator()
        result = aggregator.aggregate([])
        assert result == []

    def test_single_buy_signal(self):
        """Single BUY signal should result in BUY decision."""
        aggregator = SignalAggregator(
            enable_long_term_filter=False,
            enable_mid_term_bias=False,
        )
        signals = [
            SignalDecision(strategy_name="test", symbol="BTC", action=SignalAction.BUY),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        assert result[0].symbol == "BTC"
        assert result[0].final_action == SignalAction.BUY

    def test_majority_vote_buy(self):
        """Majority BUY signals should result in BUY decision."""
        aggregator = SignalAggregator(
            enable_long_term_filter=False,
            enable_mid_term_bias=False,
        )
        signals = [
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s2", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s3", symbol="BTC", action=SignalAction.SELL),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        assert result[0].final_action == SignalAction.BUY
        assert "Majority" in result[0].reason

    def test_tie_results_in_hold(self):
        """Tied votes should result in HOLD."""
        aggregator = SignalAggregator(
            enable_long_term_filter=False,
            enable_mid_term_bias=False,
        )
        signals = [
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s2", symbol="BTC", action=SignalAction.SELL),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        assert result[0].final_action == SignalAction.HOLD
        assert "Tie" in result[0].reason

    def test_risk_exit_highest_priority(self):
        """RISK_EXIT should override all other signals."""
        aggregator = SignalAggregator()
        signals = [
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s2", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="risk", symbol="BTC", action=SignalAction.RISK_EXIT),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        assert result[0].final_action == SignalAction.RISK_EXIT
        assert "RISK_EXIT" in result[0].reason

    def test_exit_second_priority(self):
        """EXIT should override BUY/SELL but not RISK_EXIT."""
        aggregator = SignalAggregator()
        signals = [
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s2", symbol="BTC", action=SignalAction.EXIT),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        assert result[0].final_action == SignalAction.EXIT

    def test_long_term_filter_hold_blocks_all(self):
        """Long-term HOLD signal should block all trades."""
        aggregator = SignalAggregator(enable_long_term_filter=True)
        signals = [
            SignalDecision(strategy_name="long_term", symbol="BTC", action=SignalAction.HOLD),
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s2", symbol="BTC", action=SignalAction.BUY),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        assert result[0].final_action == SignalAction.HOLD
        assert "Long-term filter" in result[0].reason

    def test_long_term_filter_sell_blocks_all(self):
        """Long-term SELL signal should block all trades (acts as no-trade filter)."""
        aggregator = SignalAggregator(enable_long_term_filter=True)
        signals = [
            SignalDecision(strategy_name="long_term", symbol="BTC", action=SignalAction.SELL),
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        assert result[0].final_action == SignalAction.HOLD
        assert "Long-term filter" in result[0].reason

    def test_long_term_buy_allows_trading(self):
        """Long-term BUY should allow other strategies to trade."""
        aggregator = SignalAggregator(enable_long_term_filter=True)
        signals = [
            SignalDecision(strategy_name="long_term", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        assert result[0].final_action == SignalAction.BUY

    def test_mid_term_buy_bias_blocks_sells(self):
        """Mid-term BUY bias should block SELL signals."""
        aggregator = SignalAggregator(
            enable_long_term_filter=False,
            enable_mid_term_bias=True,
        )
        signals = [
            SignalDecision(strategy_name="mid_term", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.SELL),
            SignalDecision(strategy_name="s2", symbol="BTC", action=SignalAction.SELL),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        # With mid-term BUY bias, SELLs are blocked, so no votes remain
        assert result[0].final_action == SignalAction.HOLD
        assert "mid-term bias" in result[0].reason.lower() or "no actionable" in result[0].reason.lower()

    def test_mid_term_sell_bias_blocks_buys(self):
        """Mid-term SELL bias should block BUY signals."""
        aggregator = SignalAggregator(
            enable_long_term_filter=False,
            enable_mid_term_bias=True,
        )
        signals = [
            SignalDecision(strategy_name="mid_term", symbol="BTC", action=SignalAction.SELL),
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s2", symbol="BTC", action=SignalAction.HOLD),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        # BUY is blocked by mid-term SELL bias, only HOLD remains
        assert result[0].final_action == SignalAction.HOLD

    def test_multiple_symbols(self):
        """Aggregator should handle multiple symbols independently."""
        aggregator = SignalAggregator(
            enable_long_term_filter=False,
            enable_mid_term_bias=False,
        )
        signals = [
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s1", symbol="ETH", action=SignalAction.SELL),
            SignalDecision(strategy_name="s2", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s2", symbol="ETH", action=SignalAction.SELL),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 2

        btc_result = next(r for r in result if r.symbol == "BTC")
        eth_result = next(r for r in result if r.symbol == "ETH")

        assert btc_result.final_action == SignalAction.BUY
        assert eth_result.final_action == SignalAction.SELL

    def test_votes_preserved_in_result(self):
        """All original votes should be preserved in the aggregated decision."""
        aggregator = SignalAggregator()
        signals = [
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY, confidence=0.8),
            SignalDecision(strategy_name="s2", symbol="BTC", action=SignalAction.HOLD, confidence=0.5),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        assert len(result[0].votes) == 2
        assert any(v.strategy_name == "s1" for v in result[0].votes)
        assert any(v.strategy_name == "s2" for v in result[0].votes)

    def test_all_hold_results_in_hold(self):
        """All HOLD signals should result in HOLD."""
        aggregator = SignalAggregator(
            enable_long_term_filter=False,
            enable_mid_term_bias=False,
        )
        signals = [
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.HOLD),
            SignalDecision(strategy_name="s2", symbol="BTC", action=SignalAction.HOLD),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        assert result[0].final_action == SignalAction.HOLD
        assert "HOLD" in result[0].reason

    def test_disabled_filters(self):
        """With filters disabled, long_term/mid_term should vote normally."""
        aggregator = SignalAggregator(
            enable_long_term_filter=False,
            enable_mid_term_bias=False,
        )
        signals = [
            SignalDecision(strategy_name="long_term", symbol="BTC", action=SignalAction.HOLD),
            SignalDecision(strategy_name="mid_term", symbol="BTC", action=SignalAction.BUY),
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
        ]

        result = aggregator.aggregate(signals)

        assert len(result) == 1
        # With filters disabled, all strategies vote
        # long_term: HOLD (neutral), mid_term: BUY, s1: BUY -> BUY wins
        assert result[0].final_action == SignalAction.BUY

    def test_long_term_sell_no_position_returns_hold(self):
        """Long-term SELL with no position should return HOLD (filter only)."""
        aggregator = SignalAggregator(enable_long_term_filter=True)
        signals = [
            SignalDecision(strategy_name="long_term", symbol="BTC", action=SignalAction.SELL),
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
        ]

        # No has_position callback: backward-compatible filter behavior
        result = aggregator.aggregate(signals)
        assert len(result) == 1
        assert result[0].final_action == SignalAction.HOLD

        # With explicit "no position" callback
        result = aggregator.aggregate(signals, has_position=lambda s: False)
        assert len(result) == 1
        assert result[0].final_action == SignalAction.HOLD
        assert "no position" in result[0].reason.lower()

    def test_long_term_sell_with_position_returns_exit(self):
        """Long-term SELL with existing position should return EXIT."""
        aggregator = SignalAggregator(enable_long_term_filter=True)
        signals = [
            SignalDecision(strategy_name="long_term", symbol="BTC", action=SignalAction.SELL),
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
        ]

        # Provide position context: BTC has a position
        positions = {"BTC": True}
        result = aggregator.aggregate(signals, has_position=lambda s: positions.get(s, False))

        assert len(result) == 1
        assert result[0].final_action == SignalAction.EXIT
        assert "EXIT" in result[0].reason
        assert "position" in result[0].reason.lower()

    def test_long_term_sell_position_aware_multiple_symbols(self):
        """Position-aware long_term SELL handles multiple symbols independently."""
        aggregator = SignalAggregator(enable_long_term_filter=True)
        signals = [
            # BTC: has position -> should EXIT
            SignalDecision(strategy_name="long_term", symbol="BTC", action=SignalAction.SELL),
            SignalDecision(strategy_name="s1", symbol="BTC", action=SignalAction.BUY),
            # ETH: no position -> should HOLD (filter)
            SignalDecision(strategy_name="long_term", symbol="ETH", action=SignalAction.SELL),
            SignalDecision(strategy_name="s1", symbol="ETH", action=SignalAction.BUY),
        ]

        # Only BTC has a position
        positions = {"BTC": True, "ETH": False}
        result = aggregator.aggregate(signals, has_position=lambda s: positions.get(s, False))

        assert len(result) == 2

        btc_result = next(r for r in result if r.symbol == "BTC")
        eth_result = next(r for r in result if r.symbol == "ETH")

        assert btc_result.final_action == SignalAction.EXIT
        assert "EXIT" in btc_result.reason

        assert eth_result.final_action == SignalAction.HOLD
        assert "no position" in eth_result.reason.lower()
