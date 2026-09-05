"""Local and remote partition resolver and JIT cache manager."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from duckpd._feature_catalog import parse_timestamp
from duckpd._quoting import quote_identifier, quote_literal


def load_dataset_metadata(
    source_root: Path | str,
    entry: dict[str, Any],
    fs: Any = None,
) -> dict[str, Any]:
    """Load detailed metadata document for a dataset if available."""
    metadata_rel = entry.get("metadata")
    if not metadata_rel or not isinstance(metadata_rel, str):
        return {}
    if isinstance(source_root, Path):
        meta_path = source_root / metadata_rel
        if not meta_path.is_file():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    elif fs is not None:
        remote_path = f"{str(source_root).rstrip('/')}/{metadata_rel}"
        try:
            if fs.exists(remote_path):
                with fs.open(remote_path, "r") as f:
                    return json.load(f)
        except Exception:
            return {}
    return {}


def get_dataset_path_template(
    source_root: Path | str,
    entry: dict[str, Any],
    fs: Any = None,
) -> str:
    """Determine the path template relative to the store root."""
    path_template = entry.get("path_template")
    if path_template is None:
        meta = load_dataset_metadata(source_root, entry, fs=fs)
        path_template = meta.get("storage", {}).get("path_template")
    if not isinstance(path_template, str) or not path_template:
        dataset_name = entry["name"]
        if entry["kind"] == "timeseries":
            path_template = f"{dataset_name}/year={{year}}/data.parquet"
        else:
            path_template = f"{dataset_name}/data.parquet"
    return path_template


def available_interval(
    entry: dict[str, Any],
    start: datetime,
    end: datetime,
) -> tuple[datetime, datetime] | None:
    """Clip a half-open interval to dataset's min_time and max_time bounds."""
    min_time = entry.get("min_time")
    max_time = entry.get("max_time")
    if min_time is not None:
        start = max(start, parse_timestamp(min_time))
    if max_time is not None:
        end = min(end, parse_timestamp(max_time) + timedelta(microseconds=1))
    if end <= start:
        return None
    return start, end


def partition_years_for_interval(
    entry: dict[str, Any],
    start: datetime,
    end: datetime,
) -> list[int]:
    """Return all partition calendar years intersecting [start, end)."""
    interval = available_interval(entry, start, end)
    if interval is None:
        return []
    avail_start, avail_end = interval
    final_time = avail_end - timedelta(microseconds=1)
    return list(range(avail_start.year, final_time.year + 1))


def resolve_local_partition_paths(
    source_root: Path,
    entry: dict[str, Any],
    start: datetime,
    end: datetime,
) -> list[str]:
    """Resolve existing partition parquet paths for the requested interval in a local directory."""
    path_template = get_dataset_path_template(source_root, entry)
    paths: list[str] = []
    if "{year}" in path_template:
        years = partition_years_for_interval(entry, start, end)
        for year in years:
            pattern = path_template.format(year=year)
            target = source_root / pattern
            if "*" in pattern or "?" in pattern:
                matched = sorted(source_root.glob(pattern))
                paths.extend(str(p.resolve()) for p in matched if p.is_file())
            elif target.is_file():
                paths.append(str(target.resolve()))
    else:
        target = source_root / path_template
        if target.is_file():
            paths.append(str(target.resolve()))
    return paths


def file_contains_columns(file_path: Path, needed_columns: list[str]) -> bool:
    """Check if a local parquet file contains all required columns without reading row data."""
    if not file_path.is_file():
        return False
    try:
        parquet_file = pq.ParquetFile(file_path)
        existing_cols = set(parquet_file.schema_arrow.names)
        return set(needed_columns).issubset(existing_cols)
    except Exception:
        return False


def ensure_cached_partition(
    source_uri: str,
    cache_root: Path,
    relative_path: str,
    needed_columns: list[str],
    con: duckdb.DuckDBPyConnection,
    filters_sql: str | None = None,
) -> Path:
    """Ensure a partition is mirrored locally with the needed columns projected."""
    local_target = cache_root / relative_path
    if local_target.is_file() and file_contains_columns(local_target, needed_columns):
        return local_target

    local_target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = local_target.with_suffix(".parquet.tmp")
    remote_file_url = f"{source_uri.rstrip('/')}/{relative_path}"

    cols_sql = ", ".join(quote_identifier(c) for c in needed_columns)
    where_clause = f" WHERE {filters_sql}" if filters_sql else ""
    copy_query = (
        f"COPY (SELECT {cols_sql} FROM read_parquet({quote_literal(remote_file_url)})"
        f"{where_clause}) TO {quote_literal(str(temp_target))} "
        f"(FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    temp_target.unlink(missing_ok=True)
    try:
        con.execute(copy_query)
        temp_target.replace(local_target)
    except BaseException:
        temp_target.unlink(missing_ok=True)
        raise

    return local_target


def ensure_cached_table(
    source_uri: str,
    cache_root: Path,
    relative_path: str,
    fs: Any,
) -> Path:
    """Ensure a static reference table is downloaded in full to the local cache."""
    local_target = cache_root / relative_path
    if local_target.is_file():
        return local_target

    local_target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = local_target.with_suffix(".parquet.tmp")
    # For HfFileSystem, path doesn't include 'hf://' prefix
    remote_file_path = source_uri.replace("hf://", "").rstrip("/") + f"/{relative_path}"

    temp_target.unlink(missing_ok=True)
    try:
        with fs.open(remote_file_path, "rb") as src, temp_target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        temp_target.replace(local_target)
    except BaseException:
        temp_target.unlink(missing_ok=True)
        raise

    return local_target


def resolve_partition_paths(
    source_root: Path,
    entry: dict[str, Any],
    start: datetime,
    end: datetime,
) -> list[str]:
    """Resolve partition parquet paths for the requested interval."""
    return resolve_local_partition_paths(source_root, entry, start, end)
