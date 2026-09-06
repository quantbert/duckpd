"""Shared validation for fixed-duration temporal operations."""

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


def fixed_duration_ns(value: object, *, parameter: str) -> int:
    """Parse a positive, fixed duration at nanosecond precision."""
    if not isinstance(value, (str, timedelta)):
        raise TypeError(f"{parameter} must be a fixed-duration string or datetime.timedelta")
    try:
        duration = pd.Timedelta(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{parameter} must be a positive fixed duration") from error
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError(f"{parameter} must be a positive fixed duration")
    return int(duration.value)


def timezone_name(value: object) -> str:
    """Validate and canonicalize an IANA timezone name without row execution."""
    if not isinstance(value, str) or not value:
        raise TypeError("timezone must be a non-empty string")
    try:
        zone = ZoneInfo(value)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown timezone {value!r}") from error
    return zone.key
