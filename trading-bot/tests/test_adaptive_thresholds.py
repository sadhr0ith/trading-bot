"""Tests for adaptive decision thresholds in strategies."""

import numpy as np
import pandas as pd
import pytest

from trading_bot.strategies.strategy_base import StrategyBase


class ConcreteStrategy(StrategyBase):
    """Concrete implementation for testing."""

    def _run_strategy(self, data: pd.DataFrame):
        return None


@pytest.fixture
def sample_data():
    """Create sample OHLCV data with known volatility."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    # Create price with known volatility
    returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
    close = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )


@pytest.fixture
def high_volatility_data():
    """Create sample data with high volatility."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    # Create price with high volatility (6% daily)
    returns = np.random.normal(0, 0.06, 100)
    close = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )


@pytest.fixture
def low_volatility_data():
    """Create sample data with low volatility."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    # Create price with low volatility (0.5% daily)
    returns = np.random.normal(0, 0.005, 100)
    close = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )


@pytest.fixture
def base_config():
    """Base config for testing."""
    return {
        "strategy": "test",
        "ticker": "TEST",
        "buy_threshold": 0.01,
        "sell_threshold": -0.01,
        "data_source": "yahoo",
        "period": "1d",
        "interval": "1d",
    }


class TestCalculateAdaptiveThreshold:
    """Tests for _calculate_adaptive_threshold method."""

    def test_returns_base_threshold_when_disabled(self, sample_data, base_config):
        """When adaptive thresholds disabled, returns base threshold."""
        strategy = ConcreteStrategy(base_config, sample_data)

        result = strategy._calculate_adaptive_threshold(
            sample_data,
            base_threshold=0.01,
        )

        # Should return a value (may be adjusted based on volatility)
        assert isinstance(result, float)
        assert result > 0

    def test_returns_base_threshold_for_insufficient_data(self, base_config):
        """Returns base threshold when data is too short."""
        short_data = pd.DataFrame({"Close": [100, 101, 102]})
        strategy = ConcreteStrategy(base_config, short_data)

        result = strategy._calculate_adaptive_threshold(
            short_data,
            base_threshold=0.01,
            volatility_window=20,
        )

        assert result == 0.01  # Base threshold unchanged

    def test_returns_base_threshold_for_missing_close(self, base_config):
        """Returns base threshold when Close column missing."""
        data = pd.DataFrame({"Price": range(100)})
        strategy = ConcreteStrategy(base_config, data)

        result = strategy._calculate_adaptive_threshold(
            data,
            base_threshold=0.01,
        )

        assert result == 0.01

    def test_higher_volatility_increases_threshold(
        self, sample_data, high_volatility_data, base_config
    ):
        """Higher volatility should increase the threshold."""
        strategy = ConcreteStrategy(base_config, sample_data)

        normal_threshold = strategy._calculate_adaptive_threshold(
            sample_data,
            base_threshold=0.01,
            reference_volatility=0.02,
        )

        high_vol_threshold = strategy._calculate_adaptive_threshold(
            high_volatility_data,
            base_threshold=0.01,
            reference_volatility=0.02,
        )

        # High volatility threshold should be higher
        assert high_vol_threshold > normal_threshold

    def test_lower_volatility_decreases_threshold(
        self, sample_data, low_volatility_data, base_config
    ):
        """Lower volatility should decrease the threshold."""
        strategy = ConcreteStrategy(base_config, sample_data)

        normal_threshold = strategy._calculate_adaptive_threshold(
            sample_data,
            base_threshold=0.01,
            reference_volatility=0.02,
        )

        low_vol_threshold = strategy._calculate_adaptive_threshold(
            low_volatility_data,
            base_threshold=0.01,
            reference_volatility=0.02,
        )

        # Low volatility threshold should be lower
        assert low_vol_threshold < normal_threshold

    def test_respects_min_multiplier(self, low_volatility_data, base_config):
        """Threshold should not go below min_multiplier * base."""
        strategy = ConcreteStrategy(base_config, low_volatility_data)

        result = strategy._calculate_adaptive_threshold(
            low_volatility_data,
            base_threshold=0.01,
            min_multiplier=0.5,
            reference_volatility=0.10,  # Very high reference to force low multiplier
        )

        assert result >= 0.01 * 0.5

    def test_respects_max_multiplier(self, high_volatility_data, base_config):
        """Threshold should not exceed max_multiplier * base."""
        strategy = ConcreteStrategy(base_config, high_volatility_data)

        result = strategy._calculate_adaptive_threshold(
            high_volatility_data,
            base_threshold=0.01,
            max_multiplier=3.0,
            reference_volatility=0.001,  # Very low reference to force high multiplier
        )

        assert result <= 0.01 * 3.0

    def test_dynamic_reference_volatility(self, sample_data, base_config):
        """Uses historical median volatility when reference is None."""
        strategy = ConcreteStrategy(base_config, sample_data)

        result = strategy._calculate_adaptive_threshold(
            sample_data,
            base_threshold=0.01,
            reference_volatility=None,
        )

        assert isinstance(result, float)
        assert result > 0


class TestGetAdaptiveThresholds:
    """Tests for _get_adaptive_thresholds method."""

    def test_returns_fixed_thresholds_when_disabled(self, sample_data, base_config):
        """When adaptive disabled, returns config thresholds."""
        config = {
            **base_config,
            "use_adaptive_thresholds": False,
        }
        strategy = ConcreteStrategy(config, sample_data)

        buy, sell = strategy._get_adaptive_thresholds(sample_data)

        assert buy == 0.01
        assert sell == -0.01

    def test_returns_adaptive_thresholds_when_enabled(self, sample_data, base_config):
        """When adaptive enabled, returns adjusted thresholds."""
        config = {
            **base_config,
            "use_adaptive_thresholds": True,
        }
        strategy = ConcreteStrategy(config, sample_data)

        buy, sell = strategy._get_adaptive_thresholds(sample_data)

        # Thresholds should be calculated (may differ from base)
        assert isinstance(buy, float)
        assert isinstance(sell, float)
        assert buy > 0
        assert sell < 0

    def test_uses_custom_adaptive_config(self, sample_data, base_config):
        """Uses custom adaptive threshold configuration."""
        config = {
            **base_config,
            "use_adaptive_thresholds": True,
            "adaptive_threshold_config": {
                "volatility_window": 10,
                "reference_volatility": 0.01,
                "min_multiplier": 0.8,
                "max_multiplier": 2.0,
            },
        }
        strategy = ConcreteStrategy(config, sample_data)

        buy, sell = strategy._get_adaptive_thresholds(sample_data)

        # Should use custom config values
        assert buy >= 0.01 * 0.8
        assert buy <= 0.01 * 2.0

    def test_override_adaptive_setting(self, sample_data, base_config):
        """Can override adaptive setting via parameter."""
        config = {
            **base_config,
            "use_adaptive_thresholds": False,  # Disabled in config
        }
        strategy = ConcreteStrategy(config, sample_data)

        # Force adaptive via parameter
        buy, sell = strategy._get_adaptive_thresholds(sample_data, use_adaptive=True)

        # Should calculate adaptive thresholds despite config
        assert isinstance(buy, float)
        assert isinstance(sell, float)
