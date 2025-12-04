import pytest

from utils.time_utils import parse_period_to_timedelta


@pytest.mark.parametrize("period,expected_days", [
    ("1d", 1),
    ("1w", 7),
    ("1M", 30),  # Fixed: M (uppercase) for months, not m
    ("1y", 365),
])
def test_parse_period_extended(period, expected_days):
    """Test period parsing with new convention: m=minutes, M=months, w=weeks"""
    delta = parse_period_to_timedelta(period)
    assert delta.days == expected_days
