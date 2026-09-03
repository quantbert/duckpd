"""Shared public and internal type definitions."""

from __future__ import annotations

import re
from collections.abc import Iterable
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

_INTEGER_BOUNDS: dict[str, tuple[int, int]] = {
    "TINYINT": (-(2**7), 2**7 - 1),
    "SMALLINT": (-(2**15), 2**15 - 1),
    "INTEGER": (-(2**31), 2**31 - 1),
    "BIGINT": (-(2**63), 2**63 - 1),
    "HUGEINT": (-(2**127), 2**127 - 1),
    "UTINYINT": (0, 2**8 - 1),
    "USMALLINT": (0, 2**16 - 1),
    "UINTEGER": (0, 2**32 - 1),
    "UBIGINT": (0, 2**64 - 1),
    "UHUGEINT": (0, 2**128 - 1),
}

_SIGNED_INTEGER_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT")
_UNSIGNED_INTEGER_TYPES = (
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
)


def common_union_type(types: Iterable[str]) -> str:
    """Return a lossless DuckDB type for a row-wise union.

    DuckPD rejects heterogeneous non-numeric columns rather than coercing
    values to strings and pretending to match pandas object semantics.
    """
    unique = tuple(dict.fromkeys(types))
    if not unique:
        raise ValueError("At least one dtype is required")
    if len(unique) == 1:
        return "VARCHAR" if unique[0] == "UNKNOWN" else unique[0]

    concrete = tuple(dtype for dtype in unique if dtype != "UNKNOWN")
    if not concrete:
        return "VARCHAR"
    if len(concrete) == 1:
        return concrete[0]

    if all(dtype in _INTEGER_BOUNDS for dtype in concrete):
        return _common_integer_type(concrete)

    decimal_parts = tuple(_decimal_parts(dtype) for dtype in concrete)
    if all(
        dtype in _INTEGER_BOUNDS or parts is not None
        for dtype, parts in zip(concrete, decimal_parts, strict=True)
    ):
        return _common_decimal_type(concrete, decimal_parts)

    if all(
        dtype in _INTEGER_BOUNDS or dtype in {"FLOAT", "DOUBLE"} or parts is not None
        for dtype, parts in zip(concrete, decimal_parts, strict=True)
    ):
        return "DOUBLE"

    joined = ", ".join(concrete)
    raise TypeError(f"concat cannot losslessly reconcile DuckDB types: {joined}")


def _common_integer_type(types: tuple[str, ...]) -> str:
    minimum = min(_INTEGER_BOUNDS[dtype][0] for dtype in types)
    maximum = max(_INTEGER_BOUNDS[dtype][1] for dtype in types)
    candidates = (
        _SIGNED_INTEGER_TYPES
        if minimum < 0
        else (*_UNSIGNED_INTEGER_TYPES, *_SIGNED_INTEGER_TYPES)
    )
    for candidate in candidates:
        lower, upper = _INTEGER_BOUNDS[candidate]
        if lower <= minimum and maximum <= upper:
            if candidate not in {"HUGEINT", "UHUGEINT"}:
                return candidate
            break
    joined = ", ".join(types)
    raise TypeError(f"concat cannot losslessly reconcile integer types: {joined}")


def _decimal_parts(dtype: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"DECIMAL\((\d+),(\d+)\)", dtype)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _common_decimal_type(
    types: tuple[str, ...],
    decimal_parts: tuple[tuple[int, int] | None, ...],
) -> str:
    scale = max((parts[1] for parts in decimal_parts if parts is not None), default=0)
    integer_digits = 0
    for dtype, parts in zip(types, decimal_parts, strict=True):
        if parts is not None:
            integer_digits = max(integer_digits, parts[0] - parts[1])
            continue
        lower, upper = _INTEGER_BOUNDS[dtype]
        integer_digits = max(integer_digits, len(str(max(abs(lower), abs(upper)))))
    precision = integer_digits + scale
    if precision > 38:
        joined = ", ".join(types)
        raise TypeError(f"concat decimal precision exceeds 38 digits: {joined}")
    return f"DECIMAL({precision},{scale})"


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

_PANDAS_TO_DUCKDB_DTYPES: dict[str, str] = {
    "int8": "TINYINT",
    "int16": "SMALLINT",
    "int32": "INTEGER",
    "int64": "BIGINT",
    "int": "BIGINT",
    "uint8": "UTINYINT",
    "uint16": "USMALLINT",
    "uint32": "UINTEGER",
    "uint64": "UBIGINT",
    "float32": "FLOAT",
    "float64": "DOUBLE",
    "float": "DOUBLE",
    "bool": "BOOLEAN",
    "boolean": "BOOLEAN",
    "str": "VARCHAR",
    "string": "VARCHAR",
    "object": "VARCHAR",
    "date": "DATE",
    "timestamp": "TIMESTAMP",
    "datetime64[ns]": "TIMESTAMP",
    "datetime64[us]": "TIMESTAMP",
    "datetime64[ms]": "TIMESTAMP",
    "datetime64[s]": "TIMESTAMP",
}


def normalize_dtype(dtype: object) -> str:
    """Normalize a pandas / numpy / string dtype to DuckDB type string."""
    dtype_str = str(dtype).lower()
    if dtype_str in _PANDAS_TO_DUCKDB_DTYPES:
        return _PANDAS_TO_DUCKDB_DTYPES[dtype_str]
    # Check upper case DuckDB types directly
    upper = str(dtype).upper()
    known_duckdb = {
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
        "FLOAT",
        "DOUBLE",
        "BOOLEAN",
        "VARCHAR",
        "DATE",
        "TIMESTAMP",
        "TIME",
        "INTERVAL",
    }
    if upper in known_duckdb or upper.startswith("DECIMAL("):
        return upper
    raise TypeError(f"Unsupported dtype for conversion: {dtype!r}")


def is_scalar_value(value: object) -> TypeGuard[ScalarValue]:
    """Return whether DuckPD accepts a value as a scalar literal."""
    return value is None or isinstance(value, _SCALAR_TYPES)
