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
