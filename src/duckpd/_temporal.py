"""Shared fixed-duration and series-temporal operations."""

from __future__ import annotations

import calendar
import math
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

if TYPE_CHECKING:
    from duckpd.embeddings import (
        SeriesCadenceSpec,
        SeriesFrequencyInputSpec,
        SeriesTemporalInputSpec,
    )

_ELAPSED_SECONDS = {
    "second": 1,
    "minute": 60,
    "hour": 3_600,
    "day": 86_400,
}
_PERIODS = {
    "second_of_minute": 60,
    "minute_of_hour": 60,
    "hour_of_day": 24,
    "day_of_week": 7,
    "day_of_month": 31,
    "day_of_year": 366,
    "week_of_year": 53,
    "month_of_year": 12,
}
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


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


def cadence_nanoseconds(cadence: SeriesCadenceSpec) -> int:
    """Return one elapsed cadence step in nanoseconds."""
    if cadence.mode != "elapsed" or cadence.unit not in _ELAPSED_SECONDS:
        raise ValueError("fixed-grid representations require an elapsed cadence")
    return cadence.multiple * _ELAPSED_SECONDS[cadence.unit] * 1_000_000_000


def resolve_timesfm_frequency(specification: SeriesFrequencyInputSpec) -> int:
    """Resolve the immutable TimesFM categorical frequency index."""
    value = specification.timesfm_frequency
    if value != "auto":
        return value
    if specification.cadence.unit in {"second", "minute", "hour", "day"}:
        return 0
    if specification.cadence.unit in {"week", "month"}:
        return 1
    return 2


def _localize_naive(value: datetime, zone: ZoneInfo, *, field: str) -> datetime:
    candidates: list[datetime] = []
    for fold in (0, 1):
        candidate = value.replace(tzinfo=zone, fold=fold)
        round_trip = candidate.astimezone(UTC).astimezone(zone)
        if round_trip.replace(tzinfo=None) == value and all(
            existing.utcoffset() != candidate.utcoffset() for existing in candidates
        ):
            candidates.append(candidate)
    if not candidates:
        raise ValueError(f"{field} is a nonexistent local time in {zone.key!r}")
    if len(candidates) > 1:
        raise ValueError(f"{field} is an ambiguous local time in {zone.key!r}")
    return candidates[0]


def resolve_series_timestamp(value: object, timezone: str, *, field: str) -> datetime:
    """Resolve a datetime to one unambiguous instant expressed in the declared zone."""
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime")
    zone = ZoneInfo(timezone)
    if value.tzinfo is None or value.utcoffset() is None:
        return _localize_naive(value.replace(tzinfo=None), zone, field=field)
    try:
        return value.astimezone(zone)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field} is outside the supported timestamp range") from error


def utc_nanoseconds(value: datetime) -> int:
    """Convert an aware datetime to an exact UTC nanosecond cache coordinate."""
    delta = value.astimezone(UTC) - _EPOCH
    return (
        delta.days * 86_400_000_000_000 + delta.seconds * 1_000_000_000 + delta.microseconds * 1_000
    )


def _shift_months(value: datetime, months: int, zone: ZoneInfo) -> datetime:
    local = value.astimezone(zone).replace(tzinfo=None)
    ordinal = local.year * 12 + local.month - 1 + months
    year, zero_month = divmod(ordinal, 12)
    if year < 1 or year > 9999:
        raise OverflowError
    month = zero_month + 1
    day = min(local.day, calendar.monthrange(year, month)[1])
    shifted = local.replace(year=year, month=month, day=day)
    return _localize_naive(shifted, zone, field="generated timestamp")


def shift_series_timestamp(value: datetime, cadence: SeriesCadenceSpec, count: int) -> datetime:
    """Shift directly from one anchor without iterative clipping drift."""
    if type(count) is not int:
        raise TypeError("cadence shift count must be an integer")
    zone = value.tzinfo
    if not isinstance(zone, ZoneInfo):
        raise TypeError("series timestamps must be resolved to an IANA timezone")
    amount = count * cadence.multiple
    try:
        if cadence.mode == "elapsed":
            seconds = _ELAPSED_SECONDS[cadence.unit] * amount
            return (value.astimezone(UTC) + timedelta(seconds=seconds)).astimezone(zone)
        if cadence.unit in {"day", "week"}:
            local = value.astimezone(zone).replace(tzinfo=None)
            days = amount * (7 if cadence.unit == "week" else 1)
            return _localize_naive(local + timedelta(days=days), zone, field="generated timestamp")
        months = amount * {"month": 1, "quarter": 3, "year": 12}[cadence.unit]
        return _shift_months(value, months, zone)
    except (OverflowError, ValueError) as error:
        if isinstance(error, ValueError) and "local time" in str(error):
            raise
        raise ValueError("series cadence shift overflowed the supported timestamp range") from error


def reconstruct_series_timestamps(
    temporal: SeriesTemporalInputSpec,
    anchor: object,
    length: int,
) -> tuple[datetime, ...]:
    """Reconstruct all observation instants from one semantic anchor."""
    resolved = resolve_series_timestamp(anchor, temporal.timezone, field="time")
    end_offset = length - 1 if temporal.anchor == "last" else length
    timestamps = tuple(
        shift_series_timestamp(resolved, temporal.cadence, index - end_offset)
        for index in range(length)
    )
    instants = tuple(value.astimezone(UTC) for value in timestamps)
    if any(right <= left for left, right in pairwise(instants)):
        raise ValueError("generated series timestamps must be strictly increasing")
    return timestamps


def _cadence_index(
    start: datetime,
    value: datetime,
    cadence: SeriesCadenceSpec,
) -> int:
    start_utc = start.astimezone(UTC)
    value_utc = value.astimezone(UTC)
    if value_utc < start_utc:
        raise ValueError("series_start must be no later than the first generated timestamp")
    if cadence.mode == "elapsed":
        step_us = cadence.multiple * _ELAPSED_SECONDS[cadence.unit] * 1_000_000
        delta = value_utc - start_utc
        delta_us = delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds
        index, remainder = divmod(delta_us, step_us)
        if remainder:
            raise ValueError("series_start and time are not aligned to the declared cadence")
    elif cadence.unit in {"day", "week"}:
        start_local = start.replace(tzinfo=None)
        value_local = value.replace(tzinfo=None)
        if start_local.timetz() != value_local.timetz():
            raise ValueError("series_start and time are not aligned to the declared cadence")
        days = (value_local.date() - start_local.date()).days
        step_days = cadence.multiple * (7 if cadence.unit == "week" else 1)
        index, remainder = divmod(days, step_days)
        if remainder:
            raise ValueError("series_start and time are not aligned to the declared cadence")
    else:
        months = (value.year - start.year) * 12 + value.month - start.month
        unit_months = cadence.multiple * {"month": 1, "quarter": 3, "year": 12}[cadence.unit]
        index, remainder = divmod(months, unit_months)
        if remainder:
            raise ValueError("series_start and time are not aligned to the declared cadence")
    if shift_series_timestamp(start, cadence, index).astimezone(UTC) != value_utc:
        raise ValueError("series_start and time are not aligned to the declared cadence")
    return index


def _calendar_component(feature: str, value: datetime) -> int:
    if feature == "second_of_minute":
        return value.second
    if feature == "minute_of_hour":
        return value.minute
    if feature == "hour_of_day":
        return value.hour
    if feature == "day_of_week":
        return value.weekday()
    if feature == "day_of_month":
        return value.day - 1
    if feature == "day_of_year":
        return value.timetuple().tm_yday - 1
    if feature == "week_of_year":
        return value.isocalendar().week - 1
    if feature == "month_of_year":
        return value.month - 1
    raise ValueError(f"Unsupported periodic feature {feature!r}")


def generate_series_time_features(
    temporal: SeriesTemporalInputSpec,
    *,
    anchor: object,
    series_start: object | None,
    length: int,
) -> tuple[tuple[float, ...], ...]:
    """Generate one ordered feature row per reconstructed timestamp."""
    timestamps = reconstruct_series_timestamps(temporal, anchor, length)
    resolved_start: datetime | None = None
    if "age_log10" in temporal.features:
        if series_start is None:
            raise ValueError("series_start is required when age_log10 is declared")
        resolved_start = resolve_series_timestamp(
            series_start,
            temporal.timezone,
            field="series_start",
        )
    elif series_start is not None:
        raise ValueError("series_start is valid only when age_log10 is declared")

    rows: list[tuple[float, ...]] = []
    for timestamp in timestamps:
        output: list[float] = []
        for feature in temporal.features:
            if feature == "age_log10":
                assert resolved_start is not None
                cadence_index = _cadence_index(
                    resolved_start,
                    timestamp,
                    temporal.cadence,
                )
                output.append(math.log10(2 + cadence_index))
                continue
            component = _calendar_component(feature, timestamp)
            period = _PERIODS[feature]
            if temporal.recipe == "gluonts-calendar-v1":
                output.append(component / (period - 1) - 0.5)
            else:
                angle = 2 * math.pi * component / period
                output.extend((math.sin(angle), math.cos(angle)))
        rows.append(tuple(output))
    return tuple(rows)
