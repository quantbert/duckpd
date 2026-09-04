"""The only layer that triggers DuckDB result production."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from duckpd._logical import (
    AggregatePlan,
    ColumnId,
    ColumnRef,
    FilterPlan,
    JoinPlan,
    JoinType,
    LimitPlan,
    LocIndexPlan,
    PandasSource,
    ParquetSource,
    ProjectPlan,
    SamplePlan,
    ScanPlan,
    SortKey,
    SortPlan,
    SourceKind,
    TopKPlan,
    UnionPlan,
)
from duckpd._optimizer import plan_to_dict
from duckpd._quoting import quote_identifier
from duckpd._typing import ParquetCompression
from duckpd.errors import (
    ConcurrentModificationError,
    MaterializationError,
    MergeError,
    UnsupportedOperationError,
)

if TYPE_CHECKING:
    from duckpd._compiler import DuckDBCompiler
    from duckpd._logical import LogicalPlan
    from duckpd.session import Session


def _replace_file_preserving_metadata(source: Path, staging: Path) -> None:
    """Atomically replace source while retaining available file metadata."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        replace_file = ctypes.WinDLL("kernel32", use_last_error=True).ReplaceFileW
        replace_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPVOID,
        ]
        replace_file.restype = wintypes.BOOL
        if not replace_file(str(source), str(staging), None, 0, None, None):
            raise ctypes.WinError(ctypes.get_last_error())
        return

    shutil.copystat(source, staging)
    os.utime(staging, None)
    os.replace(staging, source)


CommitFailurePoint = Literal[
    "before_staging",
    "during_write",
    "after_staging_write",
    "during_validation",
    "before_backup",
    "after_backup",
    "before_replace",
]


@dataclass(frozen=True)
class CommitReport:
    """Structured report returned by DataFrame.commit()."""

    source_path: str
    staging_path: str
    backup_path: str | None
    rows_written: int
    bytes_written: int
    duration_seconds: float


@dataclass(frozen=True)
class ProfileResult:
    """Structured execution profiling metrics for a DuckPD plan."""

    latency: float
    cpu_time: float
    rows_scanned: int
    rows_returned: int
    bytes_read: int
    bytes_written: int
    peak_buffer_memory: int
    peak_temp_dir_size: int
    raw: dict[str, Any]
    planning_seconds: float = 0.0
    execution_seconds: float = 0.0
    optimization: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return DuckDB metrics plus DuckPD planning and optimizer details."""
        return {
            **self.raw,
            "duckpd": {
                "planning_seconds": self.planning_seconds,
                "execution_seconds": self.execution_seconds,
                "optimization": self.optimization,
            },
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return all profile metrics as formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Format a concise human-readable profiling summary."""
        lines = [
            "DuckPD Query Profile Summary",
            f"  Execution Latency:      {self.latency * 1000:.3f} ms",
            f"  Planning Time:          {self.planning_seconds * 1000:.3f} ms",
            f"  Execution Time:         {self.execution_seconds * 1000:.3f} ms",
            f"  CPU Time:               {self.cpu_time * 1000:.3f} ms",
            f"  Rows Scanned:           {self.rows_scanned:,}",
            f"  Rows Returned:          {self.rows_returned:,}",
            f"  Bytes Read:             {self.bytes_read:,} bytes",
            f"  Bytes Written:          {self.bytes_written:,} bytes",
            f"  Peak Buffer Memory:     {self.peak_buffer_memory:,} bytes",
            f"  Peak Temp Spill Size:   {self.peak_temp_dir_size:,} bytes",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return Decimal(str(value))


def _bytes_or_none(value: object) -> bytes | None:
    if value is None or value is pd.NA:
        return None
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"Expected a binary value, got {type(value).__name__}")
    return bytes(value)


_NULLABLE_INTEGER_DTYPE_BY_DUCKDB = {
    "TINYINT": "Int8",
    "SMALLINT": "Int16",
    "INTEGER": "Int32",
    "BIGINT": "Int64",
    "UTINYINT": "UInt8",
    "USMALLINT": "UInt16",
    "UINTEGER": "UInt32",
    "UBIGINT": "UInt64",
}


def _is_null_safe_pandas_dtype(dtype: str) -> bool:
    """Return whether a source dtype can represent join-introduced missing values."""
    return dtype in {"boolean", "string"} or dtype.startswith(
        ("Int", "UInt", "datetime64[", "timedelta64[")
    )


class Executor:
    """Execute compiled plans and track observable execution boundaries."""

    def __init__(self, session: Session, compiler: DuckDBCompiler) -> None:
        self._session = session
        self._compiler = compiler

    def collect(self, plan: LogicalPlan) -> pd.DataFrame:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        rel = compiled.relation
        if plan.metadata.ordering.keys:
            order_exprs = [
                self._compiler._compile_sort_key(
                    SortKey(
                        ColumnRef(k.column_id),
                        k.direction,
                        k.null_placement,
                    ),
                    compiled.bindings,
                )
                for k in plan.metadata.ordering.keys
                if k.column_id in compiled.bindings
            ]
            if order_exprs:
                rel = rel.sort(*order_exprs)
        decimal_labels = {
            compiled.bindings[column.id]
            for column in plan.metadata.columns
            if column.duckdb_type.startswith("DECIMAL(")
            and column.id in compiled.bindings
        }
        if decimal_labels:
            rel = rel.project(
                *(
                    duckdb.SQLExpression(quote_identifier(label))
                    .cast("VARCHAR")
                    .alias(label)
                    if label in decimal_labels
                    else duckdb.SQLExpression(quote_identifier(label))
                    for label in compiled.bindings.values()
                )
            )
        result = rel.to_df()
        type_by_label = {
            compiled.bindings[column.id]: column.duckdb_type
            for column in plan.metadata.columns
            if column.id in compiled.bindings
        }
        for label, duckdb_type in type_by_label.items():
            if duckdb_type.startswith("DECIMAL("):
                result[label] = result[label].map(_decimal_or_none)
            elif duckdb_type == "BLOB":
                result[label] = result[label].map(_bytes_or_none)
            elif duckdb_type == "DATE":
                result[label] = result[label].map(
                    lambda value: (
                        None
                        if value is None or value is pd.NaT
                        else cast("pd.Timestamp", value).date()
                    )
                )
        source_dtypes = self._pandas_dtypes(plan)
        source_labels: dict[str, str] = {
            compiled.bindings[column_id]: dtype
            for column_id, dtype in source_dtypes.items()
            if column_id in compiled.bindings
        }
        preserved_labels = {
            label: dtype
            for label, dtype in source_labels.items()
            if _is_null_safe_pandas_dtype(dtype)
        }
        for label, orig_dtype in preserved_labels.items():
            if str(result[label].dtype) != orig_dtype:
                result[label] = result[label].astype(orig_dtype)  # type: ignore[arg-type]
        for label in result.columns:
            dtype_name = str(result[label].dtype)
            if (
                label not in preserved_labels
                and dtype_name.startswith(("Int", "UInt"))
                and result[label].isna().any()
            ):
                result[label] = result[label].astype("float64")
            elif (
                source_labels.get(label) == "bool"
                and dtype_name == "boolean"
                and result[label].isna().any()
            ):
                result[label] = (
                    result[label].astype(object).where(result[label].notna(), np.nan)
                )

        index_ids = plan.metadata.index.columns
        if index_ids:
            index_labels = [compiled.bindings[column_id] for column_id in index_ids]
            result = result.set_index(index_labels, drop=plan.metadata.index.drop)
        hidden_labels = [
            compiled.bindings[column.id]
            for column in plan.metadata.columns
            if column.hidden and column.id not in index_ids
        ]
        if hidden_labels:
            result = result.drop(columns=hidden_labels)
        return result

    def _pandas_dtypes(self, plan: LogicalPlan) -> dict[ColumnId, str]:
        """Track pandas source dtypes through identity-preserving plans."""
        if isinstance(plan, ScanPlan):
            if not isinstance(plan.source, PandasSource):
                return {}
            source = self._session._get_registered_source(plan.source.key)
            if not isinstance(source, pd.DataFrame):
                raise TypeError("Registered pandas source must be a DataFrame")
            return {
                column.id: str(source[column.label].dtype)
                for column in plan.metadata.columns
                if column.label in source.columns
            }
        if isinstance(plan, JoinPlan):
            left_dtypes = self._pandas_dtypes(plan.left)
            right_dtypes = self._pandas_dtypes(plan.right)
            return {**left_dtypes, **right_dtypes}
        if isinstance(plan, UnionPlan):
            nullable_labels: set[str] = set()
            for input_plan in plan.inputs:
                input_dtypes = self._pandas_dtypes(input_plan)
                nullable_labels.update(
                    column.label
                    for column in input_plan.metadata.columns
                    if column.id in input_dtypes
                    and input_dtypes[column.id].startswith(("Int", "UInt"))
                )
            return {
                column.id: _NULLABLE_INTEGER_DTYPE_BY_DUCKDB[column.duckdb_type]
                for column in plan.metadata.columns
                if column.label in nullable_labels
                and column.duckdb_type in _NULLABLE_INTEGER_DTYPE_BY_DUCKDB
            }
        if isinstance(plan, LocIndexPlan):
            return self._pandas_dtypes(plan.input)
        return self._pandas_dtypes(plan.input)

    def to_arrow(self, plan: LogicalPlan) -> pa.Table:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        return self._compiler.project_visible(compiled, plan).relation.to_arrow_table()

    def to_arrow_batches(
        self, plan: LogicalPlan, *, batch_size: int
    ) -> pa.RecordBatchReader:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        return self._compiler.project_visible(compiled, plan).relation.to_arrow_reader(
            batch_size
        )

    def write_parquet(
        self,
        plan: LogicalPlan,
        path: str,
        *,
        compression: ParquetCompression,
        overwrite: bool,
    ) -> None:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        self._compiler.project_visible(compiled, plan).relation.write_parquet(
            path,
            compression=compression,
            overwrite=overwrite,
        )

    def write_csv(
        self,
        plan: LogicalPlan,
        path: str,
        *,
        sep: str = ",",
        header: bool = True,
    ) -> None:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        self._compiler.project_visible(compiled, plan).relation.write_csv(
            path,
            sep=sep,
            header=header,
        )

    def persist(
        self,
        plan: LogicalPlan,
        name: str,
    ) -> None:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        compiled.relation.create(name)

    def save_as_table(
        self,
        plan: LogicalPlan,
        name: str,
        *,
        mode: Literal["error", "overwrite", "append"] = "error",
    ) -> None:
        """Save the plan to a DuckDB table with mode and schema validation."""
        valid_modes = {"error", "overwrite", "append"}
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown mode: {mode!r}; expected one of {sorted(valid_modes)}"
            )

        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        visible = self._compiler.project_visible(compiled, plan)
        visible_rel = visible.relation

        con = self._session._connection
        self._session._begin_execution()

        escaped_table = quote_identifier(name)
        tables_query = (
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        )
        existing_tables = {row[0] for row in con.sql(tables_query).fetchall()}
        table_exists = name in existing_tables

        con.execute("BEGIN TRANSACTION")
        try:
            if mode == "error":
                if table_exists:
                    raise ValueError(f"Table '{name}' already exists in DuckDB session")
                visible_rel.create(name)
            elif mode == "overwrite":
                if table_exists:
                    con.execute(f"DROP TABLE {escaped_table}")
                visible_rel.create(name)
            else:  # mode == "append"
                if not table_exists:
                    visible_rel.create(name)
                else:
                    info_rows = con.sql(
                        f"PRAGMA table_info({escaped_table})"
                    ).fetchall()
                    existing_columns = [str(row[1]) for row in info_rows]
                    existing_types = {
                        str(row[1]): str(row[2]).upper() for row in info_rows
                    }

                    incoming_columns = list(visible_rel.columns)
                    incoming_types = {
                        str(col): str(dtype).upper()
                        for col, dtype in zip(
                            visible_rel.columns, visible_rel.dtypes, strict=True
                        )
                    }

                    if set(existing_columns) != set(incoming_columns):
                        missing = set(existing_columns) - set(incoming_columns)
                        extra = set(incoming_columns) - set(existing_columns)
                        details: list[str] = []
                        if missing:
                            details.append(f"missing columns {sorted(missing)}")
                        if extra:
                            details.append(f"extra columns {sorted(extra)}")
                        msg = (
                            f"Schema mismatch when appending to table '{name}': "
                            f"{', '.join(details)}"
                        )
                        raise ValueError(msg)

                    type_mismatches = [
                        f"column '{col}': expected {existing_types[col]}, "
                        f"got {incoming_types[col]}"
                        for col in existing_columns
                        if existing_types[col] != incoming_types[col]
                    ]
                    if type_mismatches:
                        msg = (
                            f"Schema mismatch when appending to table '{name}': "
                            f"{', '.join(type_mismatches)}"
                        )
                        raise ValueError(msg)

                    if existing_columns != incoming_columns:
                        project_exprs = [
                            duckdb.SQLExpression(quote_identifier(col))
                            for col in existing_columns
                        ]
                        insert_rel = visible_rel.project(*project_exprs)
                    else:
                        insert_rel = visible_rel

                    insert_rel.insert_into(name)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

    def commit(
        self,
        plan: LogicalPlan,
        *,
        compression: ParquetCompression = "snappy",
        retain_previous: bool = False,
        _failure_injector: Callable[[CommitFailurePoint], None] | None = None,
        _replace_file: Callable[[Path, Path], None] = (
            _replace_file_preserving_metadata
        ),
    ) -> CommitReport:
        import time

        t0 = time.perf_counter()

        # 1. Find the single Parquet source scan
        scans: list[ScanPlan] = []

        def _walk(p: LogicalPlan) -> None:
            if isinstance(p, ScanPlan):
                scans.append(p)
                return
            if isinstance(
                p,
                (
                    FilterPlan,
                    ProjectPlan,
                    SortPlan,
                    LimitPlan,
                    TopKPlan,
                    AggregatePlan,
                    SamplePlan,
                    LocIndexPlan,
                ),
            ):
                _walk(p.input)
                return
            if isinstance(p, JoinPlan):
                _walk(p.left)
                _walk(p.right)
                return
            for inp in p.inputs:
                _walk(inp)
            return

        _walk(plan)
        if len(scans) != 1:
            raise UnsupportedOperationError(
                f"commit() requires exactly one source scan, found {len(scans)}"
            )
        scan = scans[0]
        if not isinstance(scan.source, ParquetSource):
            msg = (
                f"commit() only supports ParquetSource, "
                f"got {type(scan.source).__name__}"
            )
            raise UnsupportedOperationError(msg)
        if len(scan.source.paths) != 1:
            msg = (
                "commit() currently requires a single Parquet file, "
                "not multi-file paths"
            )
            raise UnsupportedOperationError(msg)
        provenance = plan.metadata.provenance
        if provenance.kind is not SourceKind.PARQUET or not provenance.writable:
            raise UnsupportedOperationError(
                "commit() requires writable local Parquet source provenance"
            )

        source_path_str = scan.source.paths[0]
        if any(c in source_path_str for c in "*?[]"):
            raise UnsupportedOperationError(
                "commit() does not support wildcard or glob paths"
            )

        source_path = Path(source_path_str).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Source Parquet file does not exist: {source_path}"
            )
        # 2. Capture initial source fingerprint before reading and compilation
        initial_stat = source_path.stat()
        initial_mtime_ns = initial_stat.st_mtime_ns
        initial_size = initial_stat.st_size
        source_arrow_schema = pq.ParquetFile(source_path).schema_arrow

        # 3. Inspect original source schema, columns, and row count
        con = self._session._connection
        orig_rel = con.read_parquet(str(source_path))
        orig_columns = list(orig_rel.columns)
        orig_types = [str(t).upper() for t in orig_rel.dtypes]
        orig_count_row = orig_rel.count("*").fetchone()
        if orig_count_row is None:
            raise MaterializationError("Failed to count rows in source Parquet file")
        orig_count = int(cast("int", orig_count_row[0]))

        # 3. Validate plan columns preserve schema (including hidden index columns)
        col_by_label = {col.label: col for col in plan.metadata.columns}
        missing_cols = set(orig_columns) - set(col_by_label.keys())
        extra_cols = set(col_by_label.keys()) - set(orig_columns)
        if missing_cols or extra_cols:
            details: list[str] = []
            if missing_cols:
                details.append(f"missing {sorted(missing_cols)}")
            if extra_cols:
                details.append(f"extra {sorted(extra_cols)}")
            raise UnsupportedOperationError(
                f"commit() requires schema preservation; {', '.join(details)}"
            )

        # 4. Compile plan and project in the original source-column order
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()

        project_exprs = [
            duckdb.SQLExpression(
                quote_identifier(compiled.bindings[col_by_label[col].id])
            ).alias(col)
            for col in orig_columns
        ]
        commit_rel = compiled.relation.project(*project_exprs)

        # Check types match original source types
        commit_types = [str(t).upper() for t in commit_rel.dtypes]
        type_mismatches = [
            f"column '{col}': expected {orig_t}, got {cur_t}"
            for col, orig_t, cur_t in zip(
                orig_columns, orig_types, commit_types, strict=True
            )
            if orig_t != cur_t
        ]
        if type_mismatches:
            msg = (
                f"commit() cannot alter source column types: "
                f"{', '.join(type_mismatches)}"
            )
            raise UnsupportedOperationError(msg)

        # 6. Create a unique staging file in the same directory.
        if _failure_injector is not None:
            _failure_injector("before_staging")
        dest_dir = source_path.parent
        staging_name = f".duckpd_staging_{source_path.stem}_{uuid4().hex}.parquet"
        staging_path = dest_dir / staging_name
        backup_file: Path | None = None
        replaced = False

        try:
            # 7. Stream Arrow batches while retaining Parquet/Arrow metadata.
            arrow_compression = None if compression == "uncompressed" else compression
            reader = commit_rel.to_arrow_reader()
            with pq.ParquetWriter(
                staging_path,
                source_arrow_schema,
                compression=cast("Any", arrow_compression),
            ) as writer:
                injected_during_write = False
                for batch in reader:
                    writer.write_batch(batch.cast(source_arrow_schema))
                    if _failure_injector is not None and not injected_during_write:
                        injected_during_write = True
                        _failure_injector("during_write")
            if _failure_injector is not None:
                _failure_injector("after_staging_write")

            if _failure_injector is not None:
                _failure_injector("during_validation")

            # 8. Validate output readability, row-preservation, and schema
            staging_rel = con.read_parquet(str(staging_path))
            staging_count_row = staging_rel.count("*").fetchone()
            if staging_count_row is None:
                raise MaterializationError(
                    "Failed to count rows in staging Parquet file"
                )
            staging_count = int(cast("int", staging_count_row[0]))
            if staging_count != orig_count:
                msg = (
                    f"commit() requires a row-preserving plan; "
                    f"row count changed from {orig_count} to {staging_count}"
                )
                raise UnsupportedOperationError(msg)

            staging_columns = list(staging_rel.columns)
            if staging_columns != orig_columns:
                msg = (
                    f"Committed schema {staging_columns} "
                    f"does not match original {orig_columns}"
                )
                raise ValueError(msg)
            staging_arrow_schema = pq.ParquetFile(staging_path).schema_arrow
            if not staging_arrow_schema.equals(
                source_arrow_schema, check_metadata=True
            ):
                raise ValueError(
                    "Committed Parquet schema metadata does not match source"
                )

            # 9. Concurrency guard: verify source has not been modified
            current_stat = source_path.stat()
            if (
                current_stat.st_mtime_ns != initial_mtime_ns
                or current_stat.st_size != initial_size
            ):
                msg = (
                    f"Source file '{source_path}' was modified concurrently "
                    "during commit"
                )
                raise ConcurrentModificationError(msg)

            # 10. Optional retention of previous version
            backup_path: str | None = None
            if retain_previous:
                if _failure_injector is not None:
                    _failure_injector("before_backup")
                backup_file = (
                    dest_dir / f"{source_path.stem}_backup_{uuid4().hex[:8]}.parquet"
                )
                shutil.copy2(source_path, backup_file)
                backup_path = str(backup_file)
                if _failure_injector is not None:
                    _failure_injector("after_backup")

            # Recheck after backup creation, then atomically preserve metadata.
            post_backup_stat = source_path.stat()
            if (
                post_backup_stat.st_mtime_ns != initial_mtime_ns
                or post_backup_stat.st_size != initial_size
            ):
                raise ConcurrentModificationError(
                    f"Source file '{source_path}' changed while retaining backup"
                )
            if _failure_injector is not None:
                _failure_injector("before_replace")
            _replace_file(source_path, staging_path)
            replaced = True
            bytes_written = source_path.stat().st_size
            t1 = time.perf_counter()

            return CommitReport(
                source_path=str(source_path),
                staging_path=str(staging_path),
                backup_path=backup_path,
                rows_written=staging_count,
                bytes_written=bytes_written,
                duration_seconds=t1 - t0,
            )
        finally:
            if staging_path.exists():
                staging_path.unlink(missing_ok=True)
            if not replaced and backup_file is not None and backup_file.exists():
                backup_file.unlink(missing_ok=True)

    def explain(
        self,
        plan: LogicalPlan,
        *,
        mode: Literal["all", "logical", "optimized", "json", "sql", "physical"] = "all",
    ) -> str:
        optimization = self._compiler.optimize(plan)
        logical = json.dumps(plan_to_dict(plan), indent=2)
        optimized = json.dumps(plan_to_dict(optimization.plan), indent=2)
        if mode == "logical":
            return f"DuckPD logical plan:\n{logical}"
        if mode == "optimized":
            return f"DuckPD optimized logical plan:\n{optimized}"
        if mode == "json":
            return json.dumps(optimization.to_dict(), indent=2)

        compiled = self._compiler.compile(optimization.plan, optimize=False)
        relation = compiled.relation
        self._session._begin_execution()
        if mode == "sql":
            return f"DuckDB SQL:\n{relation.sql_query()}"
        if mode == "physical":
            return f"DuckDB physical plan:\n{relation.explain()}"
        if mode == "all":
            changed = [
                snapshot.name for snapshot in optimization.snapshots if snapshot.changed
            ]
            return (
                f"DuckPD logical plan:\n{logical}\n\n"
                f"DuckPD optimized logical plan:\n{optimized}\n"
                f"Applied rewrites: {changed}\n\n"
                f"DuckDB SQL:\n{relation.sql_query()}\n\n"
                f"DuckDB physical plan:\n{relation.explain()}"
            )
        msg = (
            f"Unknown explain mode: {mode!r}; expected 'all', 'logical', "
            "'optimized', 'json', 'sql', or 'physical'"
        )
        raise ValueError(msg)

    def explain_write(
        self,
        plan: LogicalPlan,
        path: str,
        *,
        compression: ParquetCompression = "snappy",
    ) -> str:
        """Inspect write strategy and execution plan without writing rows."""
        compiled = self._compiler.compile(plan)
        visible_rel = self._compiler.project_visible(compiled, plan).relation
        self._session._begin_execution()
        return (
            f"Write target: {path}\n"
            f"Compression: {compression}\n"
            f"Output columns: {list(plan.metadata.visible_columns)}\n"
            f"DuckDB physical plan:\n{visible_rel.explain()}"
        )

    def profile(self, plan: LogicalPlan) -> ProfileResult:
        """Execute plan and separate planning from engine execution time."""
        self._validate_execution(plan)
        con = self._session._connection
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            temp_path = tmp.name

        try:
            con.execute("PRAGMA enable_profiling = 'json'")
            con.execute(f"PRAGMA profiling_output = '{temp_path}'")
            self._session._begin_execution()
            planning_started = perf_counter()
            optimization = self._compiler.optimize(plan)
            compiled = self._compiler.compile(optimization.plan, optimize=False)
            visible_rel = self._compiler.project_visible(
                compiled, optimization.plan
            ).relation
            planning_seconds = perf_counter() - planning_started
            execution_started = perf_counter()
            reader = visible_rel.to_arrow_reader()
            for _ in reader:
                pass
            execution_seconds = perf_counter() - execution_started
        finally:
            con.execute("PRAGMA disable_profiling")

        try:
            with open(temp_path, encoding="utf-8") as f:
                raw_data = cast("dict[str, Any]", json.loads(f.read()))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return ProfileResult(
            latency=float(raw_data.get("latency") or 0.0),
            cpu_time=float(raw_data.get("cpu_time") or 0.0),
            rows_scanned=int(raw_data.get("cumulative_rows_scanned") or 0),
            rows_returned=int(raw_data.get("rows_returned") or 0),
            bytes_read=int(raw_data.get("total_bytes_read") or 0),
            bytes_written=int(raw_data.get("total_bytes_written") or 0),
            peak_buffer_memory=int(raw_data.get("system_peak_buffer_memory") or 0),
            peak_temp_dir_size=int(raw_data.get("system_peak_temp_dir_size") or 0),
            raw=raw_data,
            planning_seconds=planning_seconds,
            execution_seconds=execution_seconds,
            optimization=optimization.to_dict(),
        )

    def reduce_scalar(self, plan: LogicalPlan) -> object:
        """Execute a one-column, one-row aggregate plan."""
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        if len(plan.metadata.visible_columns) != 1:
            raise MaterializationError("Scalar reduction requires one output column")
        self._session._begin_execution()
        result = compiled.relation.to_df()
        if result.shape != (1, 1):
            raise MaterializationError("Scalar reduction did not produce one value")
        value = cast("object", result.iloc[0, 0])
        return np.nan if value is None else value

    def reduce_columns(self, plan: LogicalPlan) -> pd.Series:
        """Execute a one-row aggregate plan as a label-indexed pandas Series."""
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        result = compiled.relation.to_df()
        if result.shape != (1, len(plan.metadata.visible_columns)):
            raise MaterializationError("Column reduction did not produce one row")
        reduced = result.iloc[0]
        reduced.index = [column.label for column in plan.metadata.visible_columns]
        reduced.name = None
        if reduced.isna().all():
            return pd.Series(np.nan, index=reduced.index, dtype="float64")
        if reduced.isna().any():
            reduced = reduced.map(lambda value: np.nan if value is None else value)
        return reduced.infer_objects()

    def _validate_execution(self, plan: LogicalPlan) -> None:
        if isinstance(plan, JoinPlan):
            self._validate_execution(plan.left)
            self._validate_execution(plan.right)
            if plan.validate and plan.validate not in {"m:m", "many_to_many"}:
                self._validate_join(plan)
            return
        if isinstance(plan, UnionPlan):
            for input_plan in plan.inputs:
                self._validate_execution(input_plan)
            return
        if isinstance(plan, LocIndexPlan):
            self._validate_execution(plan.input)
            self._validate_loc_plan(plan)
            return
        if isinstance(plan, SamplePlan):
            self._validate_execution(plan.input)
            if plan.n:
                compiled_input = self._compiler.compile(plan.input)
                count_row = compiled_input.relation.count("*").fetchone()
                if count_row is None:
                    raise MaterializationError("Failed to count sample population")
                population = int(cast("int", count_row[0]))
                if plan.n > population:
                    raise ValueError(
                        "Cannot take a larger sample than population "
                        "when 'replace=False'"
                    )
            return
        if isinstance(
            plan,
            (FilterPlan, ProjectPlan, SortPlan, LimitPlan, AggregatePlan),
        ):
            self._validate_execution(plan.input)

    def _validate_loc_plan(self, plan: LocIndexPlan) -> None:
        compiled_input = self._compiler.compile(plan.input)
        index_ids = plan.input.metadata.index.columns
        index_labels = [compiled_input.bindings[column_id] for column_id in index_ids]

        keys_df = cast(
            "pd.DataFrame", self._session._get_registered_source(plan.source_key)
        )
        keys_relation = self._session._connection.from_df(keys_df).set_alias(
            "__duckpd_loc_keys__"
        )

        input_alias = "__duckpd_loc_input__"
        matched_label = f"__duckpd_loc_matched_{plan.source_key}__"
        flagged_input = compiled_input.relation.project(
            f"*, 1 AS {quote_identifier(matched_label)}"
        ).set_alias(input_alias)

        conditions = [
            f"__duckpd_loc_keys__.{quote_identifier(key_label)} "
            f"IS NOT DISTINCT FROM {input_alias}.{quote_identifier(index_label)}"
            for key_label, index_label in zip(
                plan.key_labels, index_labels, strict=True
            )
        ]
        joined = keys_relation.join(flagged_input, " AND ".join(conditions), how="left")
        self._session._begin_execution()

        key_projection = ", ".join(quote_identifier(label) for label in plan.key_labels)
        missing_rows = (
            joined.filter(f"{quote_identifier(matched_label)} IS NULL")
            .project(key_projection)
            .limit(1)
            .fetchall()
        )
        if missing_rows:
            row = missing_rows[0]
            missing_value = row[0] if len(row) == 1 else row
            raise KeyError(f"[{missing_value!r}] not in index")

    def _validate_join(self, join: JoinPlan) -> None:
        check_left = join.validate in {"1:1", "1:m", "one_to_one", "one_to_many"}
        check_right = join.validate in {"1:1", "m:1", "one_to_one", "many_to_one"}
        relationship = (
            "one-to-one"
            if join.validate in {"1:1", "one_to_one"}
            else "one-to-many"
            if join.validate in {"1:m", "one_to_many"}
            else "many-to-one"
        )

        if check_left:
            self._check_uniqueness(
                join.left,
                join.left_keys,
                side="left",
                relationship=relationship,
                is_cross=join.how is JoinType.CROSS,
            )
        if check_right:
            self._check_uniqueness(
                join.right,
                join.right_keys,
                side="right",
                relationship=relationship,
                is_cross=join.how is JoinType.CROSS,
            )

    def _check_uniqueness(
        self,
        input_plan: LogicalPlan,
        key_ids: tuple[ColumnId, ...],
        *,
        side: str,
        relationship: str,
        is_cross: bool,
    ) -> None:
        compiled = self._compiler.compile(input_plan)
        self._session._begin_execution()
        if is_cross:
            limit_rel = compiled.relation.limit(2)
            if len(limit_rel.fetchall()) > 1:
                raise MergeError(
                    f"Merge keys are not unique in {side} dataset; "
                    f"not a {relationship} merge"
                )
            return

        key_cols = [quote_identifier(compiled.bindings[k_id]) for k_id in key_ids]
        group_keys = ", ".join(key_cols)
        dup_check = (
            compiled.relation.aggregate("COUNT(*) AS __duckpd_dup_cnt__", group_keys)
            .filter("__duckpd_dup_cnt__ > 1")
            .limit(1)
        )
        if len(dup_check.fetchall()) > 0:
            raise MergeError(
                f"Merge keys are not unique in {side} dataset; "
                f"not a {relationship} merge"
            )
