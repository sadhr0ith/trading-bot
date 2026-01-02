"""Tests for Prometheus metrics module."""

import pytest

from trading_bot.utils.metrics import TradingMetrics, PROMETHEUS_AVAILABLE


@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus-client not installed")
class TestTradingMetrics:
    """Tests for TradingMetrics class."""

    def test_is_available(self):
        """Check prometheus availability detection."""
        assert TradingMetrics.is_available() is True

    def test_record_trade_no_error(self):
        """Recording a trade should not raise errors."""
        TradingMetrics.record_trade("test_strategy", "BUY")
        TradingMetrics.record_trade("test_strategy", "SELL")
        TradingMetrics.record_trade("test_strategy", "HOLD")

    def test_record_prediction_error_no_error(self):
        """Recording prediction error should not raise errors."""
        TradingMetrics.record_prediction_error("test_strategy", 0.005)
        TradingMetrics.record_prediction_error("test_strategy", 0.1)

    def test_update_portfolio_value_no_error(self):
        """Updating portfolio value should not raise errors."""
        TradingMetrics.update_portfolio_value("test_strategy", 100000.0)
        TradingMetrics.update_portfolio_value("test_strategy", 105000.0)

    def test_update_max_drawdown_no_error(self):
        """Updating max drawdown should not raise errors."""
        TradingMetrics.update_max_drawdown("test_strategy", 0.05)
        TradingMetrics.update_max_drawdown("test_strategy", 0.15)

    def test_record_training_duration_no_error(self):
        """Recording training duration should not raise errors."""
        TradingMetrics.record_training_duration("test_strategy", 30.5)

    def test_record_inference_duration_no_error(self):
        """Recording inference duration should not raise errors."""
        TradingMetrics.record_inference_duration("test_strategy", 0.5)

    def test_record_data_fetch_duration_no_error(self):
        """Recording data fetch duration should not raise errors."""
        TradingMetrics.record_data_fetch_duration("yahoo", "AAPL", 2.5)
        TradingMetrics.record_data_fetch_duration("binance", "BTCUSDT", 1.2)

    def test_update_circuit_breaker_state_no_error(self):
        """Updating circuit breaker state should not raise errors."""
        TradingMetrics.update_circuit_breaker_state("yahoo", "closed")
        TradingMetrics.update_circuit_breaker_state("binance", "open")
        TradingMetrics.update_circuit_breaker_state("yahoo", "half-open")

    def test_training_timer_context_manager(self):
        """Training timer context manager should work."""
        with TradingMetrics.training_timer("test_strategy"):
            pass  # Simulate training

    def test_inference_timer_context_manager(self):
        """Inference timer context manager should work."""
        with TradingMetrics.inference_timer("test_strategy"):
            pass  # Simulate inference

    def test_data_fetch_timer_context_manager(self):
        """Data fetch timer context manager should work."""
        with TradingMetrics.data_fetch_timer("yahoo", "AAPL"):
            pass  # Simulate data fetch

    def test_get_registry_returns_registry(self):
        """Should return the Prometheus registry."""
        registry = TradingMetrics.get_registry()
        assert registry is not None


class TestMetricsGracefulDegradation:
    """Tests for graceful degradation when prometheus is unavailable."""

    def test_methods_do_not_raise_when_called(self):
        """All metric methods should be callable without errors."""
        # These should all work regardless of prometheus availability
        TradingMetrics.record_trade("test", "BUY")
        TradingMetrics.record_prediction_error("test", 0.01)
        TradingMetrics.update_portfolio_value("test", 1000.0)
        TradingMetrics.update_max_drawdown("test", 0.1)
        TradingMetrics.record_training_duration("test", 10.0)
        TradingMetrics.record_inference_duration("test", 1.0)
        TradingMetrics.record_data_fetch_duration("yahoo", "TEST", 0.5)
        TradingMetrics.update_circuit_breaker_state("yahoo", "closed")

    def test_context_managers_work(self):
        """Context managers should work without raising."""
        with TradingMetrics.training_timer("test"):
            pass
        with TradingMetrics.inference_timer("test"):
            pass
        with TradingMetrics.data_fetch_timer("yahoo", "TEST"):
            pass
