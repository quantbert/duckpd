"""Synthetic market data generation for DuckPD benchmarks."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb

PRESET_SIZES: Final[dict[str, int]] = {
    "5mb": 5_000_000,
    "50mb": 50_000_000,
    "500m": 500_000_000,
    "500mb": 500_000_000,
    "5g": 5_000_000_000,
    "5gb": 5_000_000_000,
    "50g": 50_000_000_000,
    "50gb": 50_000_000_000,
}

CANONICAL_PRESETS: Final[tuple[str, ...]] = ("5mb", "50mb", "500m", "5g", "50g")

CALIBRATION_ROWS: Final[int] = 100_000


def canonicalize_preset(name: str) -> str:
    """Map aliases to canonical preset names (e.g. 500mb -> 500m, 5gb -> 5g)."""
    clean = name.strip().lower()
    if clean in ("500mb", "500m"):
        return "500m"
    if clean in ("5gb", "5g"):
        return "5g"
    if clean in ("50gb", "50g"):
        return "50g"
    if clean == "5mb":
        return "5mb"
    if clean == "50mb":
        return "50mb"
    if clean in PRESET_SIZES:
        return clean
    raise ValueError(f"Unknown preset size {name!r}. Available: {', '.join(CANONICAL_PRESETS)}")


def market_query(rows: int) -> str:
    """Return deterministic DuckDB SQL for synthetic ticker bars."""
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    return f"""
        WITH base AS (
            SELECT
                row_id,
                TIMESTAMP '2020-01-01'
                    + floor(row_id / 8)::BIGINT * INTERVAL '1 second'
                    AS datetime,
                CASE row_id % 8
                    WHEN 0 THEN 'AAPL'
                    WHEN 1 THEN 'MSFT'
                    WHEN 2 THEN 'NVDA'
                    WHEN 3 THEN 'AMZN'
                    WHEN 4 THEN 'GOOGL'
                    WHEN 5 THEN 'META'
                    WHEN 6 THEN 'TSLA'
                    ELSE 'JPM'
                END AS ticker,
                50.0
                    + (row_id % 8) * 25.0
                    + sin(row_id / 1800.0) * 2.0
                    + ((hash(row_id) % 1000)::DOUBLE - 500.0) / 10000.0
                    AS open,
                ((hash(row_id + 17) % 2000)::DOUBLE - 1000.0) / 10000.0
                    AS move,
                0.01 + (hash(row_id + 29) % 100)::DOUBLE / 1000.0
                    AS spread
            FROM range({rows}) AS generated(row_id)
        )
        SELECT
            datetime,
            ticker,
            round(open, 4) AS open,
            round(greatest(open, open + move) + spread, 4) AS high,
            round(least(open, open + move) - spread, 4) AS low,
            round(open + move, 4) AS close
        FROM base
    """


def estimate_row_count(calibration_rows: int, calibration_bytes: int, target_bytes: int) -> int:
    """Estimate rows required for a compressed target size."""
    if calibration_rows <= 0:
        raise ValueError("calibration_rows must be positive")
    if calibration_bytes <= 0:
        raise ValueError("calibration_bytes must be positive")
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    return max(1, round(calibration_rows * target_bytes / calibration_bytes))


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata for a generated benchmark dataset."""

    preset: str
    path: Path
    file_size_bytes: int
    row_count: int


def calibrate(data_dir: Path) -> tuple[int, int]:
    """Perform a small calibration write to measure compressed bytes/row."""
    data_dir.mkdir(parents=True, exist_ok=True)
    calib_path = data_dir / ".calibration.parquet"
    con = duckdb.connect()
    try:
        sql = (
            f"COPY ({market_query(CALIBRATION_ROWS)}) TO '{calib_path}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.execute(sql)
        size_bytes = calib_path.stat().st_size
        return CALIBRATION_ROWS, size_bytes
    finally:
        con.close()
        if calib_path.exists():
            with contextlib.suppress(OSError):
                calib_path.unlink()


def generate_dataset(
    preset_name: str,
    output_dir: Path,
    *,
    force: bool = False,
    calibration: tuple[int, int] | None = None,
) -> DatasetMetadata:
    """Generate a single dataset preset file if it does not already exist."""
    canonical = canonicalize_preset(preset_name)
    target_bytes = PRESET_SIZES[canonical]
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"market-data-{canonical}.parquet"

    if target_path.exists() and not force:
        actual_size = target_path.stat().st_size
        if actual_size > 0:
            # Query actual row count
            con = duckdb.connect()
            try:
                row_count = int(
                    con.execute(f"SELECT count(*) FROM '{target_path}'").fetchone()[0]  # type: ignore[index]
                )
            finally:
                con.close()
            return DatasetMetadata(canonical, target_path, actual_size, row_count)

    if calibration is None:
        calibration = calibrate(output_dir)

    calib_rows, calib_bytes = calibration
    rows = estimate_row_count(calib_rows, calib_bytes, target_bytes)

    temp_path = output_dir / f".market-data-{canonical}.tmp.parquet"
    con = duckdb.connect()
    try:
        sql = f"COPY ({market_query(rows)}) TO '{temp_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        con.execute(sql)
        if target_path.exists():
            target_path.unlink()
        temp_path.rename(target_path)
    finally:
        con.close()
        if temp_path.exists():
            with contextlib.suppress(OSError):
                temp_path.unlink()

    actual_size = target_path.stat().st_size
    return DatasetMetadata(canonical, target_path, actual_size, rows)


def ensure_dataset(
    preset_name: str,
    output_dir: Path,
    *,
    force: bool = False,
) -> DatasetMetadata:
    """Ensure a dataset exists for the given preset, generating if needed."""
    return generate_dataset(preset_name, output_dir, force=force)
