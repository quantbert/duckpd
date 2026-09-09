"""Atomically migrate legacy yearly/monthly feature datasets to UTC-day partitions."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from build_catalog import build_catalog
from gendata import DAILY_PARTITION_LAYOUT, NEWS_SCHEMA, daily_partition_path

LEGACY_PATTERNS = {
    "ohlcv": "year=*/data.parquet",
    "sma": "year=*/data.parquet",
    "news": "year=*/month=*/data.parquet",
}


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _metadata_bounds(paths: list[Path], time_column: str) -> tuple[date, date]:
    minimum: datetime | None = None
    maximum: datetime | None = None
    for path in paths:
        parquet = pq.ParquetFile(path)
        column_index = parquet.schema_arrow.get_field_index(time_column)
        if column_index < 0:
            raise ValueError(f"{path} does not contain {time_column!r}")
        for row_group_index in range(parquet.metadata.num_row_groups):
            statistics = parquet.metadata.row_group(row_group_index).column(column_index).statistics
            if statistics is None or not statistics.has_min_max:
                raise ValueError(f"{path} lacks {time_column!r} min/max statistics")
            lower = statistics.min
            upper = statistics.max
            if not isinstance(lower, datetime) or not isinstance(upper, datetime):
                raise TypeError(f"{path} has non-timestamp {time_column!r} statistics")
            minimum = lower if minimum is None else min(minimum, lower)
            maximum = upper if maximum is None else max(maximum, upper)
    if minimum is None or maximum is None:
        raise ValueError("Legacy dataset has no timestamp bounds")
    return minimum.date(), maximum.date()


def _row_count(paths: list[Path]) -> int:
    return sum(int(pq.ParquetFile(path).metadata.num_rows) for path in paths)


def _write_with_schema(table: pa.Table, destination: Path, schema: pa.Schema) -> None:
    """Atomically write a table using the dataset's original Arrow schema."""
    temporary = destination.with_name(f".{destination.name}.tmp")
    cast_table = table if table.schema.equals(schema) else table.cast(schema)
    pq.write_table(  # pyright: ignore[reportUnknownMemberType]
        cast_table,
        temporary,
        compression="zstd",
    )
    temporary.replace(destination)


def _compact_partition(partition_dir: Path, schema: pa.Schema) -> None:
    """Compact DuckDB's partition fragments into the single-file store contract."""
    fragments = sorted(partition_dir.glob("part*.parquet"))
    if not fragments:
        raise ValueError(f"No Parquet files found in {partition_dir}")
    destination = partition_dir / "part.parquet"
    if len(fragments) == 1 and pq.ParquetFile(fragments[0]).schema_arrow.equals(schema):
        if fragments[0] != destination:
            fragments[0].replace(destination)
        return

    tables = [pq.ParquetFile(fragment).read() for fragment in fragments]  # pyright: ignore[reportUnknownMemberType]
    table = pa.concat_tables(tables)
    _write_with_schema(table, destination, schema)
    for fragment in fragments:
        if fragment != destination:
            fragment.unlink()


def _repair_daily_schemas(paths: list[Path], schema: pa.Schema) -> None:
    """Repair daily partitions produced by an interrupted older migration."""
    for path in paths:
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow.equals(schema):
            continue
        table = parquet.read()  # pyright: ignore[reportUnknownMemberType]
        _write_with_schema(table, path, schema)


def migrate_dataset(
    dataset_root: Path,
    legacy_pattern: str,
    target_schema: pa.Schema | None = None,
) -> int:
    """Replace one legacy dataset with verified daily partitions and return its row count."""
    legacy_paths = sorted(dataset_root.glob(legacy_pattern))
    daily_paths = sorted(dataset_root.glob("year=*/month=*/day=*/part.parquet"))
    if not legacy_paths:
        if daily_paths:
            if target_schema is not None:
                _repair_daily_schemas(daily_paths, target_schema)
            return _row_count(daily_paths)
        raise FileNotFoundError(f"No legacy or daily Parquet partitions found under {dataset_root}")
    if daily_paths:
        raise ValueError(f"Mixed legacy and daily partitions under {dataset_root}")

    staging = dataset_root.with_name(f".{dataset_root.name}-daily-migration")
    backup = dataset_root.with_name(f".{dataset_root.name}-legacy-backup")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    staging.mkdir(parents=True)

    source_rows = _row_count(legacy_paths)
    schema = pq.ParquetFile(legacy_paths[0]).schema_arrow
    output_schema = target_schema or schema
    first_day, last_day = _metadata_bounds(legacy_paths, "datetime")
    paths_sql = ", ".join(_quoted(str(path.resolve())) for path in legacy_paths)
    staging_sql = _quoted(str(staging.resolve()))

    connection = duckdb.connect()
    connection.execute("SET threads = 1")
    try:
        connection.execute(
            f"""
            COPY (
                SELECT *,
                       strftime(datetime, '%Y') AS year,
                       strftime(datetime, '%m') AS month,
                       strftime(datetime, '%d') AS day
                FROM read_parquet([{paths_sql}], hive_partitioning=false)
            ) TO {staging_sql} (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                PARTITION_BY (year, month, day),
                FILENAME_PATTERN 'part'
            )
            """
        )
    finally:
        connection.close()
    for partition_dir in sorted(staging.glob("year=*/month=*/day=*")):
        _compact_partition(partition_dir, output_schema)

    empty = pa.Table.from_batches([], schema=output_schema)
    current = first_day
    while current <= last_day:
        destination = daily_partition_path(staging, current)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(empty, destination, compression="zstd")  # pyright: ignore[reportUnknownMemberType]
        current += timedelta(days=1)

    staged_paths = sorted(staging.glob("year=*/month=*/day=*/part.parquet"))
    staged_rows = _row_count(staged_paths)
    if staged_rows != source_rows:
        raise ValueError(
            f"Daily migration row count mismatch: source={source_rows}, staged={staged_rows}"
        )

    for auxiliary in dataset_root.iterdir():
        if auxiliary.name.startswith("year="):
            continue
        target = staging / auxiliary.name
        if auxiliary.is_file():
            shutil.copy2(auxiliary, target)
        elif auxiliary.is_dir():
            shutil.copytree(auxiliary, target)
    completion = staging / "_SUCCESS.json"
    if completion.is_file():
        identity = json.loads(completion.read_text(encoding="utf-8"))
        identity["partition_layout"] = DAILY_PARTITION_LAYOUT
        completion.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")

    dataset_root.replace(backup)
    try:
        staging.replace(dataset_root)
    except BaseException:
        backup.replace(dataset_root)
        raise
    shutil.rmtree(backup)
    return source_rows


def migrate_store(data_root: Path) -> dict[str, int]:
    """Migrate every generated timeseries dataset and rebuild its catalog."""
    catalog_path = data_root / "catalog.json"
    store_name = data_root.name
    if catalog_path.is_file():
        existing_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog_name = existing_catalog.get("name")
        if isinstance(catalog_name, str):
            store_name = catalog_name

    results: dict[str, int] = {}
    for dataset, pattern in LEGACY_PATTERNS.items():
        dataset_root = data_root / dataset
        if dataset_root.exists():
            target_schema = NEWS_SCHEMA if dataset == "news" else None
            results[dataset] = migrate_dataset(dataset_root, pattern, target_schema)
    if results:
        build_catalog(data_root, store_name=store_name, source=str(data_root))
    return results


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    return parser.parse_args()


if __name__ == "__main__":
    settings = arguments()
    migrated = migrate_store(settings.data.resolve())
    for name, rows in migrated.items():
        print(f"Migrated {name}: {rows:,} rows")
