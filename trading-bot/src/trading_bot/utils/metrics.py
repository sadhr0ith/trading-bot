"""Prometheus metrics for trading bot observability.

This module provides Prometheus metrics for monitoring the trading bot's performance,
including trade counts, prediction errors, portfolio value, and model training times.

Usage:
    from trading_bot.utils.metrics import TradingMetrics

    # Record a trade
    TradingMetrics.record_trade("short_term", "BUY")

    # Update portfolio value
    TradingMetrics.update_portfolio_value("short_term", 10500.0)

    # Record prediction error
    TradingMetrics.record_prediction_error("short_term", 0.0025)

    # Record training duration
    with TradingMetrics.training_timer("short_term"):
        # ... training code ...

    # Start HTTP server for /metrics endpoint
    TradingMetrics.start_http_server(port=8000)
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Generator

try:
    from prometheus_client import (
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

logger = logging.getLogger(__name__)

# Metrics definitions (lazy-initialized to avoid import-time side effects)
_metrics_initialized = False
_trades_total: Counter | None = None
_prediction_error: Histogram | None = None
_portfolio_value: Gauge | None = None
_training_duration: Histogram | None = None
_inference_duration: Histogram | None = None
_data_fetch_duration: Histogram | None = None
_circuit_breaker_state: Gauge | None = None
_max_drawdown: Gauge | None = None


def _initialize_metrics() -> None:
    """Initialize Prometheus metrics (lazy initialization)."""
    global _metrics_initialized
    global _trades_total, _prediction_error, _portfolio_value
    global _training_duration, _inference_duration, _data_fetch_duration
    global _circuit_breaker_state, _max_drawdown

    if _metrics_initialized or not PROMETHEUS_AVAILABLE:
        return

    _trades_total = Counter(
        "trading_bot_trades_total",
        "Total number of trades executed",
        ["strategy", "signal"],
    )

    _prediction_error = Histogram(
        "trading_bot_prediction_error",
        "Prediction error (MAE) for strategy predictions",
        ["strategy"],
        buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
    )

    _portfolio_value = Gauge(
        "trading_bot_portfolio_value",
        "Current portfolio value in base currency",
        ["strategy"],
    )

    _training_duration = Histogram(
        "trading_bot_training_duration_seconds",
        "Time spent training models",
        ["strategy"],
        buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800),
    )

    _inference_duration = Histogram(
        "trading_bot_inference_duration_seconds",
        "Time spent on model inference",
        ["strategy"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10),
    )

    _data_fetch_duration = Histogram(
        "trading_bot_data_fetch_duration_seconds",
        "Time spent fetching market data",
        ["source", "ticker"],
        buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60),
    )

    _circuit_breaker_state = Gauge(
        "trading_bot_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open, 2=half-open)",
        ["source"],
    )

    _max_drawdown = Gauge(
        "trading_bot_max_drawdown",
        "Maximum drawdown observed for paper trading",
        ["strategy"],
    )

    _metrics_initialized = True
    logger.info("Prometheus metrics initialized")


class TradingMetrics:
    """Static class for recording trading bot metrics.

    All methods are no-ops if prometheus-client is not installed,
    ensuring graceful degradation without breaking the bot.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if Prometheus metrics are available."""
        return PROMETHEUS_AVAILABLE

    @staticmethod
    def record_trade(strategy: str, signal: str) -> None:
        """Record a trade execution.

        Args:
            strategy: Strategy name (e.g., "short_term", "day_trading")
            signal: Trade signal ("BUY", "SELL", "HOLD")
        """
        _initialize_metrics()
        if _trades_total is not None:
            _trades_total.labels(strategy=strategy, signal=signal).inc()

    @staticmethod
    def record_prediction_error(strategy: str, error: float) -> None:
        """Record prediction error (MAE).

        Args:
            strategy: Strategy name
            error: Mean absolute error value
        """
        _initialize_metrics()
        if _prediction_error is not None:
            _prediction_error.labels(strategy=strategy).observe(error)

    @staticmethod
    def update_portfolio_value(strategy: str, value: float) -> None:
        """Update current portfolio value.

        Args:
            strategy: Strategy name
            value: Current portfolio value
        """
        _initialize_metrics()
        if _portfolio_value is not None:
            _portfolio_value.labels(strategy=strategy).set(value)

    @staticmethod
    def update_max_drawdown(strategy: str, drawdown: float) -> None:
        """Update maximum drawdown.

        Args:
            strategy: Strategy name
            drawdown: Maximum drawdown ratio (0.0 to 1.0)
        """
        _initialize_metrics()
        if _max_drawdown is not None:
            _max_drawdown.labels(strategy=strategy).set(drawdown)

    @staticmethod
    def record_training_duration(strategy: str, duration: float) -> None:
        """Record model training duration.

        Args:
            strategy: Strategy name
            duration: Training time in seconds
        """
        _initialize_metrics()
        if _training_duration is not None:
            _training_duration.labels(strategy=strategy).observe(duration)

    @staticmethod
    def record_inference_duration(strategy: str, duration: float) -> None:
        """Record model inference duration.

        Args:
            strategy: Strategy name
            duration: Inference time in seconds
        """
        _initialize_metrics()
        if _inference_duration is not None:
            _inference_duration.labels(strategy=strategy).observe(duration)

    @staticmethod
    def record_data_fetch_duration(source: str, ticker: str, duration: float) -> None:
        """Record data fetch duration.

        Args:
            source: Data source ("yahoo", "binance")
            ticker: Ticker symbol
            duration: Fetch time in seconds
        """
        _initialize_metrics()
        if _data_fetch_duration is not None:
            _data_fetch_duration.labels(source=source, ticker=ticker).observe(duration)

    @staticmethod
    def update_circuit_breaker_state(source: str, state: str) -> None:
        """Update circuit breaker state.

        Args:
            source: Data source ("yahoo", "binance")
            state: State string ("closed", "open", "half-open")
        """
        _initialize_metrics()
        if _circuit_breaker_state is not None:
            state_map = {"closed": 0, "open": 1, "half-open": 2}
            _circuit_breaker_state.labels(source=source).set(state_map.get(state.lower(), -1))

    @staticmethod
    @contextlib.contextmanager
    def training_timer(strategy: str) -> Generator[None, None, None]:
        """Context manager for timing model training.

        Args:
            strategy: Strategy name

        Usage:
            with TradingMetrics.training_timer("short_term"):
                model.fit(X, y)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            TradingMetrics.record_training_duration(strategy, duration)

    @staticmethod
    @contextlib.contextmanager
    def inference_timer(strategy: str) -> Generator[None, None, None]:
        """Context manager for timing model inference.

        Args:
            strategy: Strategy name

        Usage:
            with TradingMetrics.inference_timer("short_term"):
                predictions = model.predict(X)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            TradingMetrics.record_inference_duration(strategy, duration)

    @staticmethod
    @contextlib.contextmanager
    def data_fetch_timer(source: str, ticker: str) -> Generator[None, None, None]:
        """Context manager for timing data fetches.

        Args:
            source: Data source
            ticker: Ticker symbol

        Usage:
            with TradingMetrics.data_fetch_timer("yahoo", "AAPL"):
                data = yf.download(...)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            TradingMetrics.record_data_fetch_duration(source, ticker, duration)

    @staticmethod
    def start_http_server(port: int = 8000, addr: str = "") -> None:
        """Start HTTP server for Prometheus scraping.

        Args:
            port: Port number (default: 8000)
            addr: Address to bind (default: all interfaces)
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus-client not installed; cannot start metrics server")
            return

        _initialize_metrics()
        try:
            start_http_server(port, addr)
            logger.info(f"Prometheus metrics server started on port {port}")
        except OSError as e:
            logger.warning(f"Failed to start metrics server on port {port}: {e}")

    @staticmethod
    def get_registry() -> CollectorRegistry | None:
        """Get the Prometheus registry for testing or custom export.

        Returns:
            The default Prometheus registry, or None if not available
        """
        if not PROMETHEUS_AVAILABLE:
            return None
        return REGISTRY
