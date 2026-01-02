"""Time utilities for period parsing and time-based operations."""

import re
from datetime import timedelta
from typing import Mapping

_INTERVAL_TO_TIMDELTA: Mapping[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "90m": timedelta(minutes=90),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
    "1wk": timedelta(weeks=1),
    "1mo": timedelta(days=30),  # approximate
}


def parse_period_to_timedelta(period: str) -> timedelta:
    """
    Parse a period string into a timedelta object.

    Period format convention (Option A - standard in finance/pandas):
    - 'm' = minutes
    - 'M' = months (approximated as 30 days)
    - 'w' = weeks
    - 'd' = days
    - 'h' = hours
    - 'y' = years (approximated as 365 days)

    Examples:
        - "1m" = 1 minute
        - "5m" = 5 minutes
        - "1h" = 1 hour
        - "1d" = 1 day
        - "1w" = 1 week
        - "1M" = 1 month (30 days)
        - "1mo" = 1 month (30 days, alias)
        - "6M" = 6 months (180 days)
        - "6mo" = 6 months (180 days, alias)
        - "1y" = 1 year (365 days)

    Args:
        period: Period string (e.g., "1d", "6M", "1y")

    Returns:
        timedelta object

    Raises:
        ValueError: If period format is invalid or unit is unknown
    """
    match = re.match(r"(\d+)([a-zA-Z]+)", period)
    if not match:
        raise ValueError(f"Invalid period format: {period}")

    value, unit = match.groups()
    value = int(value)

    # Map canonical units to timedelta
    unit_map: dict[str, timedelta] = {
        "m": timedelta(minutes=value),
        "M": timedelta(days=value * 30),  # Approximate month as 30 days
        "w": timedelta(weeks=value),
        "d": timedelta(days=value),
        "h": timedelta(hours=value),
        "y": timedelta(days=value * 365),  # Approximate year as 365 days
    }

    # Friendly aliases (common in broker/finance APIs)
    aliases: dict[str, str] = {
        "min": "m",
        "mins": "m",
        "minute": "m",
        "minutes": "m",
        "hr": "h",
        "hrs": "h",
        "hour": "h",
        "hours": "h",
        "day": "d",
        "days": "d",
        "wk": "w",
        "wks": "w",
        "week": "w",
        "weeks": "w",
        "mo": "M",
        "mon": "M",
        "month": "M",
        "months": "M",
        "yr": "y",
        "yrs": "y",
        "year": "y",
        "years": "y",
    }

    if unit not in unit_map:
        normalized = unit.lower()
        canonical = aliases.get(normalized)
        if canonical is not None:
            unit = canonical

    if unit not in unit_map:
        raise ValueError(
            f"Unknown period unit: {unit}. "
            "Valid units: m (minutes), h (hours), d (days), w (weeks), M (months), y (years) "
            "plus aliases like mo, wk, min, hr."
        )

    return unit_map[unit]


def interval_to_timedelta(interval: str | None) -> timedelta | None:
    """Map interval strings (e.g., '1h') to timedeltas. Returns None for unknown intervals."""
    if interval is None:
        return None
    return _INTERVAL_TO_TIMDELTA.get(interval)


def interval_to_seconds(interval: str | None) -> int | None:
    """Return interval length in seconds if known."""
    delta = interval_to_timedelta(interval)
    return int(delta.total_seconds()) if delta else None
