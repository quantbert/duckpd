"""Generate calibrated synthetic OHLC market-data Parquet files."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import duckpd as pd

TARGET_SIZES = {
    "smoke": 5_000_000,
    "100mb": 100_000_000,
    "1gb": 1_000_000_000,
    "5gb": 5_000_000_000,
}
CALIBRATION_ROWS = 250_000


@dataclass(frozen=True)
class Arguments:
    """Validated command-line arguments."""

    sizes: tuple[str, ...]
    output_dir: Path
    force: bool
    memory_limit: str
    calibration_rows: int


def market_query(rows: int) -> str:
    """Return deterministic DuckDB SQL for synthetic ticker bars."""
    if rows <= 0:
        raise ValueError("rows must be positive")
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
    if min(calibration_rows, calibration_bytes, target_bytes) <= 0:
        raise ValueError("calibration and target values must be positive")
    return max(1, round(calibration_rows * target_bytes / calibration_bytes))


def write_market_data(session: pd.Session, rows: int, path: Path) -> None:
    """Stream generated rows through DuckPD directly into Parquet."""
    session.sql(market_query(rows)).write_parquet(
        path,
        compression="zstd",
        overwrite=True,
    )


def human_size(size: int) -> str:
    """Format a byte count for console output."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1000.0
    raise AssertionError("unreachable")


def generate(arguments: Arguments) -> None:
    """Generate all requested presets after one calibration write."""
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[str, int, Path]] = []
    for name in arguments.sizes:
        target_bytes = TARGET_SIZES[name]
        path = arguments.output_dir / f"market-data-{name}.parquet"
        if path.exists() and not arguments.force:
            print(f"Skipping existing {path}; pass --force to replace it")
            continue
        pending.append((name, target_bytes, path))

    if not pending:
        return

    required_bytes = sum(target_bytes for _, target_bytes, _ in pending)
    free_bytes = shutil.disk_usage(arguments.output_dir).free
    if required_bytes * 1.05 > free_bytes:
        raise RuntimeError(
            f"Requested about {human_size(required_bytes)}, but only "
            f"{human_size(free_bytes)} is free"
        )

    with TemporaryDirectory(
        prefix="duckpd-market-data-", dir=arguments.output_dir
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        calibration_path = temporary / "calibration.parquet"
        spill_directory = temporary / "spill"
        with pd.connect(
            memory_limit=arguments.memory_limit,
            temp_directory=spill_directory,
        ) as session:
            print(f"Calibrating with {arguments.calibration_rows:,} rows...")
            write_market_data(
                session,
                arguments.calibration_rows,
                calibration_path,
            )
            calibration_bytes = calibration_path.stat().st_size
            bytes_per_row = calibration_bytes / arguments.calibration_rows
            print(
                f"Calibration: {human_size(calibration_bytes)}, "
                f"{bytes_per_row:.2f} compressed bytes/row"
            )

            for name, target_bytes, path in pending:
                rows = estimate_row_count(
                    arguments.calibration_rows,
                    calibration_bytes,
                    target_bytes,
                )
                print(
                    f"Writing {name} preset: {rows:,} rows to {path} "
                    f"(target {human_size(target_bytes)})..."
                )
                write_market_data(session, rows, path)
                actual_bytes = path.stat().st_size
                error = (actual_bytes - target_bytes) / target_bytes * 100.0
                print(f"Created {path}: {human_size(actual_bytes)} ({error:+.1f}% from target)")


def parse_args(argv: Sequence[str] | None = None) -> Arguments:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sizes",
        nargs="*",
        default=None,
        metavar="SIZE",
        help=(
            f"Size presets to generate ({', '.join(TARGET_SIZES)}); defaults to a 5 MB smoke file"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo/data"),
        help="Directory for generated Parquet files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace files that already exist",
    )
    parser.add_argument(
        "--memory-limit",
        default="1GB",
        help="DuckDB working-memory limit",
    )
    parser.add_argument(
        "--calibration-rows",
        type=int,
        default=CALIBRATION_ROWS,
        help="Rows used to estimate compressed bytes per row",
    )
    namespace = parser.parse_args(argv)
    parsed_sizes = cast("list[str] | None", namespace.sizes)
    invalid = [size for size in parsed_sizes or () if size not in TARGET_SIZES]
    if invalid:
        valid = ", ".join(TARGET_SIZES)
        parser.error(f"argument sizes: invalid choice: {invalid[0]!r} (choose from {valid})")
    sizes = tuple(parsed_sizes) if parsed_sizes else ("smoke",)
    calibration_rows = cast("int", namespace.calibration_rows)
    if calibration_rows <= 0:
        parser.error("--calibration-rows must be positive")
    return Arguments(
        sizes=sizes,
        output_dir=cast("Path", namespace.output_dir),
        force=cast("bool", namespace.force),
        memory_limit=cast("str", namespace.memory_limit),
        calibration_rows=calibration_rows,
    )


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
