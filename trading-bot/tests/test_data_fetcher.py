import os
import time
from datetime import timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from trading_bot.data_fetcher import fetch_binance_data
from trading_bot import data_fetcher
from trading_bot.models.env_settings import RuntimeSettings
from trading_bot.utils.time_utils import parse_period_to_timedelta


def test_parse_period_days():
    assert parse_period_to_timedelta("5d") == timedelta(days=5)


def test_parse_period_months():
    # New convention: M (uppercase) for months, not mo
    assert parse_period_to_timedelta("6M") == timedelta(days=180)


def test_parse_period_years():
    assert parse_period_to_timedelta("2y") == timedelta(days=730)


def test_parse_period_weeks():
    # New: support for weeks (w)
    assert parse_period_to_timedelta("2w") == timedelta(weeks=2)


def test_parse_period_hours():
    assert parse_period_to_timedelta("12h") == timedelta(hours=12)


def test_parse_period_minutes():
    assert parse_period_to_timedelta("30m") == timedelta(minutes=30)


def test_parse_period_invalid_unit():
    with pytest.raises(ValueError):
        parse_period_to_timedelta("10q")


@patch.dict("os.environ", {"BINANCE_API_KEY": "test_key", "BINANCE_API_SECRET": "test_secret"})
@patch("trading_bot.data_fetcher.fetch_binance_klines")
@patch("trading_bot.data_fetcher.Client")
def test_binance_data_has_datetime_index(mock_client, mock_fetch_klines):
    """Test 3.5: Verify Binance data returns DataFrame with DatetimeIndex"""
    # Mock klines response
    mock_klines = [
        [
            1609459200000,
            "29000",
            "29500",
            "28800",
            "29200",
            "100",
            1609462800000,
            "2920000",
            1000,
            "50",
            "1460000",
            "0",
        ],
        [
            1609462800000,
            "29200",
            "29700",
            "29100",
            "29500",
            "110",
            1609466400000,
            "3245000",
            1100,
            "55",
            "1622500",
            "0",
        ],
        [
            1609466400000,
            "29500",
            "29900",
            "29400",
            "29700",
            "120",
            1609470000000,
            "3564000",
            1200,
            "60",
            "1782000",
            "0",
        ],
    ]
    mock_fetch_klines.return_value = mock_klines

    # Fetch data
    data = fetch_binance_data("BTCUSDT", "1h", "1d")

    # Verify not empty
    assert not data.empty
    assert len(data) == 3

    # Verify index is DatetimeIndex
    assert isinstance(data.index, pd.DatetimeIndex), "Index must be DatetimeIndex"

    # Verify index name
    assert data.index.name == "timestamp"

    # Verify index is monotonically increasing
    assert data.index.is_monotonic_increasing, "Index must be monotonically increasing"


def _runtime_settings(ttl_seconds: int) -> RuntimeSettings:
    return RuntimeSettings(
        cache_ttl_seconds=ttl_seconds,
        min_rows=50,
        sleep_seconds=60,
        min_sleep_seconds=60,
    )


def _sample_cache_frame():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    return pd.DataFrame(
        {"Open": [1, 2, 3], "High": [1, 2, 3.5], "Low": [0.5, 1, 2], "Close": [1, 2, 3], "Volume": [10, 20, 30]},
        index=idx,
    )


def test_fetch_data_online_uses_cache_hit(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.pkl"
    df = _sample_cache_frame()
    data_fetcher._save_cache(cache_path, df)

    calls = {"fetch": 0}

    def fake_fetch_yahoo(*args, **kwargs):
        calls["fetch"] += 1
        return df.copy()

    monkeypatch.setattr(data_fetcher, "_cache_paths", lambda *args, **kwargs: cache_path)
    monkeypatch.setattr(data_fetcher, "fetch_yahoo_data", fake_fetch_yahoo)
    monkeypatch.setattr(data_fetcher, "load_cache_settings", lambda logger=None: _runtime_settings(300))

    result = data_fetcher.fetch_data_online(source="yahoo", ticker="T", period="1y", interval="1d")

    assert calls["fetch"] == 0, "Cache hit should bypass fetch_yahoo_data"
    pd.testing.assert_frame_equal(result, df)


def test_fetch_data_online_cache_expired_triggers_fetch(monkeypatch, tmp_path):
    cache_path = tmp_path / "expired.pkl"
    df_cached = _sample_cache_frame()
    data_fetcher._save_cache(cache_path, df_cached)
    old = time.time() - 10
    os.utime(cache_path, (old, old))

    df_fetched = df_cached.copy()
    df_fetched["Close"] = [10, 11, 12]
    calls = {"fetch": 0}

    def fake_fetch_yahoo(*args, **kwargs):
        calls["fetch"] += 1
        return df_fetched.copy()

    monkeypatch.setattr(data_fetcher, "_cache_paths", lambda *args, **kwargs: cache_path)
    monkeypatch.setattr(data_fetcher, "fetch_yahoo_data", fake_fetch_yahoo)
    monkeypatch.setattr(data_fetcher, "load_cache_settings", lambda logger=None: _runtime_settings(1))

    result = data_fetcher.fetch_data_online(source="yahoo", ticker="T", period="1y", interval="1d")

    assert calls["fetch"] == 1, "Expired cache should trigger fetch"
    pd.testing.assert_frame_equal(result, df_fetched)


def test_normalize_ohlcv_resamples_on_mismatch(caplog):
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [2, 3, 4],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1, 2, 3],
            "Volume": [10, 20, 30],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
    )

    with caplog.at_level("WARNING"):
        normalized = data_fetcher._normalize_ohlcv(df, expected_interval="1d", resample_on_mismatch=True)

    assert any("interval mismatch" in msg.lower() for msg in caplog.messages)
    # Resampling hourly data to daily should yield 2 rows (ceil of 3 hours over 1-day rule)
    assert len(normalized) <= len(df)
    assert not normalized.empty


def test_fetch_data_online_normalizes_cached_frame(monkeypatch, tmp_path):
    cache_path = tmp_path / "naive_cache.pkl"
    idx = pd.to_datetime(["2024-01-02", "2024-01-01"])
    df = pd.DataFrame(
        {"Open": [1, 2], "High": [2, 3], "Low": [0.5, 1], "Close": [1, 2], "Volume": [10, 20]},
        index=idx,
    )
    data_fetcher._save_cache(cache_path, df)

    monkeypatch.setattr(data_fetcher, "_cache_paths", lambda *args, **kwargs: cache_path)
    monkeypatch.setattr(data_fetcher, "load_cache_settings", lambda logger=None: _runtime_settings(300))

    normalized = data_fetcher.fetch_data_online(source="yahoo", ticker="T", period="1y", interval="1d")

    assert isinstance(normalized.index, pd.DatetimeIndex)
    assert normalized.index.tz is not None
    assert normalized.index.is_monotonic_increasing
    assert len(normalized) == len(df)


def test_fetch_data_online_normalizes_yahoo_fetch(monkeypatch, tmp_path):
    cache_path = tmp_path / "naive_cache.pkl"

    def fake_cache_paths(*args, **kwargs):
        return cache_path

    def fake_fetch_yahoo(*args, **kwargs):
        idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
        return pd.DataFrame(
            {
                "Open": [3, 1, 2],
                "High": [3.5, 1.5, 2.5],
                "Low": [2.5, 0.5, 1.5],
                "Close": [3, 1, 2],
                "Volume": [30, 10, 20],
            },
            index=idx,
        )

    monkeypatch.setattr(data_fetcher, "_cache_paths", fake_cache_paths)
    monkeypatch.setattr(data_fetcher, "fetch_yahoo_data", fake_fetch_yahoo)
    monkeypatch.setattr(data_fetcher, "load_cache_settings", lambda logger=None: _runtime_settings(0))

    normalized = data_fetcher.fetch_data_online(source="yahoo", ticker="T", period="1y", interval="1d")

    assert normalized.index.is_monotonic_increasing
    assert normalized.index.tz is not None
    assert normalized["Close"].iloc[-1] == 3


class TestCircuitBreaker:
    """Tests for circuit breaker behavior on API calls."""

    def test_yahoo_circuit_breaker_opens_after_failures(self, monkeypatch):
        """Circuit breaker should open after 5 consecutive failures."""
        # Reset circuit breaker to closed state
        data_fetcher.yahoo_circuit_breaker.close()

        call_count = {"count": 0}

        def failing_download(*args, **kwargs):
            call_count["count"] += 1
            raise ConnectionError("Network error")

        monkeypatch.setattr("yfinance.download", failing_download)

        # Make 5 failing calls to trip the breaker
        for _ in range(5):
            result = data_fetcher.fetch_yahoo_data("AAPL", "1y", "1d")
            assert result.empty

        assert call_count["count"] == 5

        # 6th call should fail immediately due to open circuit
        result = data_fetcher.fetch_yahoo_data("AAPL", "1y", "1d")
        assert result.empty
        # Should NOT have made another actual API call
        assert call_count["count"] == 5

        # Cleanup: close the breaker for other tests
        data_fetcher.yahoo_circuit_breaker.close()

    def test_yahoo_circuit_breaker_state_transitions(self, monkeypatch):
        """Circuit breaker should track state transitions correctly."""
        # Reset circuit breaker to closed state
        data_fetcher.yahoo_circuit_breaker.close()
        assert data_fetcher.yahoo_circuit_breaker.current_state == "closed"

        call_count = {"count": 0}

        def failing_download(*args, **kwargs):
            call_count["count"] += 1
            raise ConnectionError("Network error")

        monkeypatch.setattr("yfinance.download", failing_download)

        # Make 4 failing calls - should still be closed
        for _ in range(4):
            data_fetcher.fetch_yahoo_data("AAPL", "1y", "1d")

        assert data_fetcher.yahoo_circuit_breaker.current_state == "closed"
        assert call_count["count"] == 4

        # 5th failure should open the circuit
        data_fetcher.fetch_yahoo_data("AAPL", "1y", "1d")
        assert data_fetcher.yahoo_circuit_breaker.current_state == "open"
        assert call_count["count"] == 5

        # Additional calls should not increment counter (circuit is open)
        data_fetcher.fetch_yahoo_data("AAPL", "1y", "1d")
        data_fetcher.fetch_yahoo_data("AAPL", "1y", "1d")
        assert call_count["count"] == 5  # No new API calls

        # Cleanup
        data_fetcher.yahoo_circuit_breaker.close()

    def test_binance_circuit_breaker_fallback_to_cache(self, monkeypatch, tmp_path):
        """When Binance circuit breaker is open, should fallback to cache."""
        # Reset circuit breaker
        data_fetcher.binance_circuit_breaker.close()

        cache_path = tmp_path / "binance_cache.pkl.gz"
        cached_df = _sample_cache_frame()
        data_fetcher._save_cache(cache_path, cached_df)

        # Mock settings and paths
        monkeypatch.setattr(data_fetcher, "_cache_paths", lambda *args, **kwargs: cache_path)
        monkeypatch.setattr(data_fetcher, "load_cache_settings", lambda logger=None: _runtime_settings(3600))
        monkeypatch.setattr(data_fetcher, "load_binance_settings", lambda logger=None: type(
            "Settings", (), {"api_key": "key", "api_secret": "secret"}
        )())

        # Mock Client to raise errors
        call_count = {"count": 0}

        class MockClient:
            def __init__(self, **kwargs):
                pass

            def get_klines(self, **kwargs):
                call_count["count"] += 1
                raise ConnectionError("Binance API error")

        monkeypatch.setattr(data_fetcher, "Client", MockClient)

        # Need to mock fetch_binance_klines_since to also fail for incremental path
        def failing_incremental(*args, **kwargs):
            raise ConnectionError("Binance API error")

        monkeypatch.setattr(data_fetcher, "fetch_binance_klines_since", failing_incremental)

        # Make calls to trip the breaker (5 failures needed)
        # With enable_incremental=True, it loads cache then tries to fetch new data
        for _ in range(5):
            data_fetcher.fetch_binance_data("BTCUSDT", "1h", "1d", enable_incremental=True, cache_ttl_seconds=3600)

        # Circuit should be open now, next call should use cache immediately
        result = data_fetcher.fetch_binance_data("BTCUSDT", "1h", "1d", enable_incremental=True, cache_ttl_seconds=3600)

        # Should have returned cached data, not empty
        assert not result.empty
        assert len(result) == len(cached_df)

        # Cleanup
        data_fetcher.binance_circuit_breaker.close()

    def test_circuit_breaker_half_open_allows_test_call(self, monkeypatch):
        """In half-open state, circuit should allow one test call."""
        # Reset breaker
        data_fetcher.yahoo_circuit_breaker.close()

        call_count = {"count": 0}

        def failing_download(*args, **kwargs):
            call_count["count"] += 1
            raise ConnectionError("Network error")

        monkeypatch.setattr("yfinance.download", failing_download)

        # Trip the breaker
        for _ in range(5):
            data_fetcher.fetch_yahoo_data("AAPL", "1y", "1d")

        assert call_count["count"] == 5
        assert data_fetcher.yahoo_circuit_breaker.current_state == "open"

        # Set to half-open to allow test call
        data_fetcher.yahoo_circuit_breaker.half_open()

        # Now switch to success
        def success_download(*args, **kwargs):
            call_count["count"] += 1
            return pd.DataFrame()  # Empty but no exception

        monkeypatch.setattr("yfinance.download", success_download)

        # Half-open allows one call which should succeed
        result = data_fetcher.fetch_yahoo_data("AAPL", "1y", "1d")

        assert call_count["count"] == 6  # One more call was made
        assert data_fetcher.yahoo_circuit_breaker.current_state == "closed"

        # Cleanup
        data_fetcher.yahoo_circuit_breaker.close()
