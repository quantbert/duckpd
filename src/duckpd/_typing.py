"""Shared public and internal type definitions."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Literal, TypeAlias, TypeGuard
from uuid import UUID

ScalarValue: TypeAlias = (
    int
    | float
    | bytes
    | bytearray
    | str
    | bool
    | date
    | datetime
    | time
    | timedelta
    | UUID
    | Decimal
    | memoryview
    | None
)

ParquetCompression: TypeAlias = Literal[
    "uncompressed",
    "brotli",
    "snappy",
    "lz4",
    "lz4_raw",
    "gzip",
    "zstd",
]

_SCALAR_TYPES = (
    int,
    float,
    bytes,
    bytearray,
    str,
    bool,
    date,
    datetime,
    time,
    timedelta,
    UUID,
    Decimal,
    memoryview,
)


def is_scalar_value(value: object) -> TypeGuard[ScalarValue]:
    """Return whether DuckPD accepts a value as a scalar literal."""
    return value is None or isinstance(value, _SCALAR_TYPES)
