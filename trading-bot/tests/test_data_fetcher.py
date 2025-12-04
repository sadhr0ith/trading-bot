import pathlib
import sys
from datetime import timedelta
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from data_fetcher import fetch_binance_data
from utils.time_utils import parse_period_to_timedelta


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
@patch("data_fetcher.fetch_binance_klines")
@patch("data_fetcher.Client")
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
