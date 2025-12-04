"""Time utilities for period parsing and time-based operations."""

import re
from datetime import timedelta


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
        - "6M" = 6 months (180 days)
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

    # Map units to timedelta
    unit_map = {
        'm': timedelta(minutes=value),
        'M': timedelta(days=value * 30),  # Approximate month as 30 days
        'w': timedelta(weeks=value),
        'd': timedelta(days=value),
        'h': timedelta(hours=value),
        'y': timedelta(days=value * 365),  # Approximate year as 365 days
    }

    if unit not in unit_map:
        raise ValueError(
            f"Unknown period unit: {unit}. "
            f"Valid units: m (minutes), M (months), w (weeks), d (days), h (hours), y (years)"
        )

    return unit_map[unit]
