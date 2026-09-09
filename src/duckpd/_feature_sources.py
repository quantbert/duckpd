"""Local and remote partition resolver and JIT cache manager."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from io import BytesIO, TextIOWrapper
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO, TextIO, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from duckpd._feature_catalog import parse_timestamp
from duckpd._locking import exclusive_file_lock
from duckpd._quoting import quote_identifier, quote_literal

if TYPE_CHECKING:
    from duckpd._logical import FeatureParquetSource

_TRANSFER_METRICS: ContextVar[dict[str, int | float] | None] = ContextVar(
    "_TRANSFER_METRICS",
    default=None,
)


@contextmanager
def capture_transfer_metrics(
    metrics: dict[str, int | float],
) -> Generator[None, None, None]:
    token = _TRANSFER_METRICS.set(metrics)
    try:
        yield
    finally:
        _TRANSFER_METRICS.reset(token)


def _validated_relative_path(relative_path: str) -> PurePosixPath:
    """Return a normalized catalog path that cannot escape its configured root."""
    path = PurePosixPath(relative_path)
    if not relative_path or path.is_absolute() or ".." in path.parts or "\\" in relative_path:
        raise ValueError(f"Catalog path must remain relative to its root: {relative_path!r}")
    return path


def _rooted_path(root: Path, relative_path: str) -> Path:
    relative = _validated_relative_path(relative_path)
    target = (root / Path(*relative.parts)).resolve()
    resolved_root = root.resolve()
    if target == resolved_root or resolved_root not in target.parents:
        raise ValueError(f"Catalog path escapes its configured root: {relative_path!r}")
    return target


class HttpFileSystem:
    """Minimal read-only filesystem adapter for HTTP feature stores."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token

    def _request(self, path: str, *, method: str = "GET") -> Request:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        return Request(path, headers=headers, method=method)

    def open(self, path: str, mode: str = "rb") -> BinaryIO | TextIO:
        if mode not in {"r", "rb"}:
            raise ValueError("HTTP feature stores are read-only")
        with urlopen(self._request(path), timeout=30) as response:
            payload = response.read()
        raw = BytesIO(payload)
        if mode == "rb":
            return raw
        return TextIOWrapper(raw, encoding="utf-8")

    def exists(self, path: str) -> bool:
        try:
            with urlopen(self._request(path, method="HEAD"), timeout=30):
                return True
        except HTTPError as error:
            if error.code not in {405, 501}:
                return False
        try:
            with urlopen(self._request(path), timeout=30):
                return True
        except HTTPError:
            return False


def remote_file_path(source_root: str, relative_path: str) -> str:
    """Join a provider root and validated relative path."""
    remote_root = source_root.removeprefix("hf://").rstrip("/")
    remote_path = _validated_relative_path(relative_path).as_posix()
    return f"{remote_root}/{remote_path}"


def load_dataset_metadata(
    source_root: Path | str,
    entry: dict[str, Any],
    fs: Any = None,
) -> dict[str, Any]:
    """Load detailed metadata document for a dataset if available."""
    metadata_rel = entry.get("metadata")
    if not metadata_rel or not isinstance(metadata_rel, str):
        return {}
    metadata_path = _validated_relative_path(metadata_rel).as_posix()
    if isinstance(source_root, Path):
        meta_path = _rooted_path(source_root, metadata_path)
        if not meta_path.is_file():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    if fs is not None:
        remote_path = remote_file_path(str(source_root), metadata_path)
        try:
            if fs.exists(remote_path):
                with fs.open(remote_path, "r") as file:
                    return json.load(file)
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
            partitioning = entry.get("partitioning")
            unit = (
                cast("dict[str, Any]", partitioning).get("unit")
                if isinstance(partitioning, dict)
                else None
            )
            if unit == "day":
                path_template = (
                    f"{dataset_name}/year={{year}}/month={{month:02d}}/day={{day:02d}}/part.parquet"
                )
            elif unit == "month":
                path_template = f"{dataset_name}/year={{year}}/month={{month:02d}}/data.parquet"
            else:
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


def partition_paths_for_interval(
    entry: dict[str, Any],
    path_template: str,
    start: datetime,
    end: datetime,
) -> list[str]:
    """Format all partition paths intersecting the half-open interval."""
    interval = available_interval(entry, start, end)
    if interval is None:
        return []
    avail_start, avail_end = interval
    final_time = avail_end - timedelta(microseconds=1)

    if "{day" in path_template:
        cursor = avail_start.replace(hour=0, minute=0, second=0, microsecond=0)
        paths: list[str] = []
        while cursor <= final_time:
            paths.append(path_template.format(year=cursor.year, month=cursor.month, day=cursor.day))
            cursor += timedelta(days=1)
        return paths

    if "{month" in path_template:
        cursor = avail_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        paths: list[str] = []
        while cursor <= final_time:
            paths.append(path_template.format(year=cursor.year, month=cursor.month))
            if cursor.month == 12:
                cursor = cursor.replace(year=cursor.year + 1, month=1)
            else:
                cursor = cursor.replace(month=cursor.month + 1)
        return paths

    if "{year" in path_template:
        return [
            path_template.format(year=year) for year in range(avail_start.year, final_time.year + 1)
        ]
    return [path_template]


def resolve_local_partition_paths(
    source_root: Path,
    entry: dict[str, Any],
    start: datetime,
    end: datetime,
) -> list[str]:
    """Resolve existing partition parquet paths for the requested interval in a local directory."""
    path_template = get_dataset_path_template(source_root, entry)
    _validated_relative_path(path_template)
    paths: list[str] = []
    for pattern in partition_paths_for_interval(entry, path_template, start, end):
        if "*" in pattern or "?" in pattern:
            matched = sorted(source_root.glob(pattern))
            root = source_root.resolve()
            paths.extend(
                str(path.resolve())
                for path in matched
                if path.is_file() and root in path.resolve().parents
            )
        else:
            target = _rooted_path(source_root, pattern)
            if target.is_file():
                paths.append(str(target))
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


def _restore_fixed_size_columns(
    path: Path,
    embedding_columns: dict[str, int],
) -> None:
    """Restore Arrow fixed-size-list metadata stripped by DuckDB's Parquet writer."""
    table = pq.read_table(path)  # pyright: ignore[reportUnknownMemberType]
    changed = False
    for column, dimension in embedding_columns.items():
        index = table.schema.get_field_index(column)
        if index < 0:
            continue
        field = cast("Any", table.schema.field(index))
        expected_type = pa.list_(pa.float32(), dimension)
        if field.type.equals(expected_type):
            continue
        cast_column = table.column(index).cast(expected_type)
        expected_field = pa.field(
            field.name,
            expected_type,
            nullable=field.nullable,
            metadata=field.metadata,
        )
        table = table.set_column(index, expected_field, cast_column)
        changed = True
    if changed:
        pq.write_table(  # pyright: ignore[reportUnknownMemberType]
            table,
            path,
            compression="zstd",
        )


def ensure_cached_partition(
    source_uri: str,
    cache_root: Path,
    relative_path: str,
    needed_columns: list[str],
    con: duckdb.DuckDBPyConnection,
    filters_sql: str | None = None,
    embedding_columns: dict[str, int] | None = None,
) -> Path:
    """Ensure a partition contains the cumulative requested projection."""
    local_target = _rooted_path(cache_root, relative_path)
    metrics = _TRANSFER_METRICS.get()
    lock_path = local_target.with_name(f".{local_target.name}.lock")
    remote_path = _validated_relative_path(relative_path).as_posix()
    remote_file_url = f"{source_uri.rstrip('/')}/{remote_path}"

    with exclusive_file_lock(lock_path):
        existing_columns: list[str] = []
        if local_target.is_file():
            try:
                existing_columns = list(pq.ParquetFile(local_target).schema_arrow.names)
            except Exception:
                existing_columns = []
        if set(needed_columns).issubset(existing_columns):
            if metrics is not None:
                metrics["partition_cache_reuses"] = (
                    int(metrics.get("partition_cache_reuses", 0)) + 1
                )
            return local_target

        projected_columns = list(dict.fromkeys([*existing_columns, *needed_columns]))
        local_target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = local_target.with_name(
            f".{local_target.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        cols_sql = ", ".join(quote_identifier(column) for column in projected_columns)
        where_clause = f" WHERE {filters_sql}" if filters_sql else ""
        copy_query = (
            f"COPY (SELECT {cols_sql} FROM read_parquet({quote_literal(remote_file_url)})"
            f"{where_clause}) TO {quote_literal(str(temp_target))} "
            f"(FORMAT PARQUET, COMPRESSION ZSTD)"
        )

        try:
            con.execute(copy_query)
            if embedding_columns:
                _restore_fixed_size_columns(temp_target, embedding_columns)
            os.replace(temp_target, local_target)
            if metrics is not None:
                metrics["partition_transfer_bytes"] = (
                    int(metrics.get("partition_transfer_bytes", 0)) + local_target.stat().st_size
                )
                metrics["partition_transfers"] = int(metrics.get("partition_transfers", 0)) + 1
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
    metrics = _TRANSFER_METRICS.get()
    local_target = _rooted_path(cache_root, relative_path)
    lock_path = local_target.with_name(f".{local_target.name}.lock")
    remote_path = _validated_relative_path(relative_path).as_posix()
    remote_file_pathname = remote_file_path(source_uri, remote_path)

    with exclusive_file_lock(lock_path):
        if local_target.is_file():
            if metrics is not None:
                metrics["partition_cache_reuses"] = (
                    int(metrics.get("partition_cache_reuses", 0)) + 1
                )
            return local_target

        local_target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = local_target.with_name(
            f".{local_target.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        try:
            with fs.open(remote_file_pathname, "rb") as src, temp_target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(temp_target, local_target)
            if metrics is not None:
                metrics["partition_transfer_bytes"] = (
                    int(metrics.get("partition_transfer_bytes", 0)) + local_target.stat().st_size
                )
                metrics["partition_transfers"] = int(metrics.get("partition_transfers", 0)) + 1
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


def materialize_feature_source(
    source: FeatureParquetSource,
    con: duckdb.DuckDBPyConnection,
    filesystem: Any = None,
) -> list[str]:
    """Resolve a feature scan and populate remote cache files at compilation."""
    entry: dict[str, Any] = {
        "name": "feature_source",
        "kind": "table" if source.table else "timeseries",
        "path_template": source.path_template,
    }
    if source.min_time is not None:
        entry["min_time"] = source.min_time
    if source.max_time is not None:
        entry["max_time"] = source.max_time

    if source.table:
        if source.cache_root is None:
            target = _rooted_path(Path(source.source_root), source.path_template)
            return [str(target)] if target.is_file() else []
        if filesystem is None:
            raise RuntimeError("Feature table filesystem is unavailable")
        return [
            str(
                ensure_cached_table(
                    source.source_root,
                    Path(source.cache_root),
                    source.path_template,
                    filesystem,
                ).resolve()
            )
        ]

    if source.start is None or source.end is None:
        raise AssertionError("Timeseries feature scans require start and end")
    start = parse_timestamp(source.start)
    end = parse_timestamp(source.end)
    if source.cache_root is None:
        return resolve_local_partition_paths(Path(source.source_root), entry, start, end)

    path_template = source.path_template
    _validated_relative_path(path_template)
    relative_paths = partition_paths_for_interval(entry, path_template, start, end)
    return [
        str(
            ensure_cached_partition(
                source.source_root,
                Path(source.cache_root),
                relative_path,
                list(source.needed_columns),
                con,
                embedding_columns=dict(source.embedding_columns),
            ).resolve()
        )
        for relative_path in relative_paths
    ]
