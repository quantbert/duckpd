"""The only layer that triggers DuckDB result production."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Concatenate, Literal, ParamSpec, TypeVar, cast
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from duckpd._logical import (
    AggregatePlan,
    AsOfJoinPlan,
    BinaryExpression,
    ColumnId,
    ColumnRef,
    FeatureParquetSource,
    FilterPlan,
    JoinPlan,
    JoinType,
    LimitPlan,
    LiteralValue,
    LocIndexPlan,
    PandasSource,
    ParquetSource,
    ProjectPlan,
    RemoteTableSource,
    SamplePlan,
    ScanPlan,
    SortKey,
    SortPlan,
    SourceCapabilities,
    SourceFragment,
    SourceKind,
    SourceOperation,
    TopKPlan,
    UnaryExpression,
    UnionPlan,
    expression_metadata,
    sanitize_source_location,
)
from duckpd._optimizer import plan_to_dict
from duckpd._quoting import quote_identifier
from duckpd._typing import ParquetCompression
from duckpd.errors import (
    ConcurrentModificationError,
    MaterializationError,
    MergeError,
    RemoteScanWarning,
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
    files_written: int
    columns_written: int
    row_groups_written: int
    bytes_written: int
    duration_seconds: float


@dataclass(frozen=True)
class MaterializationReport:
    """Measured contract for one explicit Python materialization."""

    reason: str
    estimated_bytes: int | None
    actual_bytes: int
    limit_bytes: int | None


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
    remote_boundaries: tuple[dict[str, object], ...] = ()
    source_fragments: tuple[dict[str, object], ...] = ()
    movement_plans: tuple[dict[str, object], ...] = ()
    measured_transfer_bytes: int | None = None
    optimization: dict[str, Any] | None = None
    fallback_boundaries: tuple[dict[str, object], ...] = ()
    materialization_boundaries: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return DuckDB metrics plus DuckPD planning and optimizer details."""
        return {
            **self.raw,
            "duckpd": {
                "planning_seconds": self.planning_seconds,
                "execution_seconds": self.execution_seconds,
                "remote_boundaries": list(self.remote_boundaries),
                "source_fragments": list(self.source_fragments),
                "movement": list(self.movement_plans),
                "measured_transfer_bytes": self.measured_transfer_bytes,
                "measured_source_bytes_read": self.bytes_read,
                "optimization": self.optimization,
                "fallback_policy": "error",
                "fallback_boundaries": list(self.fallback_boundaries),
                "materialization_boundaries": list(self.materialization_boundaries),
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


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _execution_context(
    operation: str,
) -> Callable[
    [Callable[Concatenate[Executor, LogicalPlan, _P], _R]],
    Callable[Concatenate[Executor, LogicalPlan, _P], _R],
]:
    """Attach credential-safe DuckPD context to DuckDB execution failures."""

    def decorate(
        function: Callable[Concatenate[Executor, LogicalPlan, _P], _R],
    ) -> Callable[Concatenate[Executor, LogicalPlan, _P], _R]:
        @wraps(function)
        def wrapped(
            self: Executor,
            plan: LogicalPlan,
            *args: _P.args,
            **kwargs: _P.kwargs,
        ) -> _R:
            try:
                return function(self, plan, *args, **kwargs)
            except duckdb.Error:
                provenance = plan.metadata.provenance
                locations = tuple(
                    sanitize_source_location(location) for location in provenance.locations
                )
                context = (
                    f"plan={type(plan).__name__}, "
                    f"source_kind={provenance.kind.value}, "
                    f"locations={locations}"
                )
                raise MaterializationError(
                    f"DuckPD operation {operation!r} failed ({context})"
                ) from None

        return wrapped

    return decorate


def _plan_nodes(plan: LogicalPlan) -> Iterator[LogicalPlan]:
    """Yield a plan tree without compiling or executing it."""
    yield plan
    if isinstance(plan, (JoinPlan, AsOfJoinPlan)):
        yield from _plan_nodes(plan.left)
        yield from _plan_nodes(plan.right)
    elif isinstance(plan, UnionPlan):
        for input_plan in plan.inputs:
            yield from _plan_nodes(input_plan)
    elif isinstance(
        plan,
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
        yield from _plan_nodes(plan.input)


def _redact_plan_text(text: str, plan: LogicalPlan) -> str:
    """Remove credentials and query parameters from engine plan text."""
    for location in plan.metadata.provenance.locations:
        text = text.replace(location, sanitize_source_location(location))
    return text


def _materialization_upper_bound(plan: LogicalPlan) -> int | None:
    """Prove a conservative pandas-memory bound for a narrow plan subset."""
    if _fallback_boundaries(plan):
        return None

    fixed_width_types = {
        "BOOLEAN",
        "TINYINT",
        "UTINYINT",
        "SMALLINT",
        "USMALLINT",
        "INTEGER",
        "UINTEGER",
        "BIGINT",
        "UBIGINT",
        "HUGEINT",
        "UHUGEINT",
        "FLOAT",
        "REAL",
        "DOUBLE",
        "TIMESTAMP",
        "TIMESTAMP_S",
        "TIMESTAMP_MS",
        "TIMESTAMP_NS",
        "TIMESTAMP WITH TIME ZONE",
        "INTERVAL",
    }
    index_ids = set(plan.metadata.index.columns)
    materialized_columns = tuple(
        column for column in plan.metadata.columns if not column.hidden or column.id in index_ids
    )
    if any(column.duckdb_type.upper() not in fixed_width_types for column in materialized_columns):
        return None

    def row_upper_bound(node: LogicalPlan) -> int | None:
        if isinstance(node, ScanPlan):
            if not isinstance(node.source, ParquetSource):
                return None
            total_rows = 0
            for location in node.source.paths:
                if "://" in location or any(char in location for char in "*?[]"):
                    return None
                path = Path(location)
                if not path.is_file():
                    return None
                total_rows += pq.ParquetFile(path).metadata.num_rows
            return total_rows
        if isinstance(node, (FilterPlan, ProjectPlan, SortPlan)):
            return row_upper_bound(node.input)
        if isinstance(node, (LimitPlan, TopKPlan)):
            input_rows = row_upper_bound(node.input)
            return None if input_rows is None else min(input_rows, node.count)
        if isinstance(node, AggregatePlan):
            input_rows = row_upper_bound(node.input)
            if input_rows is None:
                return None
            return input_rows if node.keys else 1
        if isinstance(node, SamplePlan):
            input_rows = row_upper_bound(node.input)
            if input_rows is None:
                return None
            if node.n is not None:
                return min(input_rows, node.n)
            if node.frac is not None:
                return min(input_rows, math.ceil(input_rows * node.frac))
        return None

    rows = row_upper_bound(plan)
    if rows is None:
        return None
    label_bytes = sum(len(str(column.label).encode()) for column in materialized_columns)
    per_row = 64 * (len(materialized_columns) + max(len(index_ids), 1))
    return 65_536 + label_bytes * 4 + rows * per_row


def _fallback_boundaries(plan: LogicalPlan) -> tuple[dict[str, object], ...]:
    """Describe explicit Python execution encoded in a logical plan."""
    found: dict[str, dict[str, object]] = {}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            mapping = cast("dict[str, object]", value)
            if mapping.get("is_arrow_udf") is True:
                name = str(mapping["name"])
                found[name] = {
                    "kind": "arrow_udf",
                    "name": name,
                    "batch_independent": True,
                    "estimated_transfer_bytes": None,
                }
            for child in mapping.values():
                visit(child)
        elif isinstance(value, list):
            for child in cast("list[object]", value):
                visit(child)

    visit(plan_to_dict(plan))
    return tuple(found.values())


_SOURCE_OPERATION_ORDER = tuple(SourceOperation)
_REMOTE_PARQUET_CAPABILITIES = SourceCapabilities(projection=True, filter=True)


def _source_filter_expression(expression: object) -> bool:
    """Accept only scalar operators known safe for source predicate pushdown."""
    if isinstance(expression, (ColumnRef, LiteralValue)):
        return True
    if isinstance(expression, BinaryExpression):
        return _source_filter_expression(expression.left) and _source_filter_expression(
            expression.right
        )
    if isinstance(expression, UnaryExpression):
        return _source_filter_expression(expression.operand)
    return False


def _source_fragments(plan: LogicalPlan) -> tuple[SourceFragment, ...]:
    """Split remote branches into pushdown candidates and required local work."""
    fragments: list[SourceFragment] = []

    def visit(
        node: LogicalPlan,
        requested: frozenset[SourceOperation],
        blocked: frozenset[SourceOperation],
    ) -> None:
        if isinstance(node, ScanPlan):
            source = node.source
            if isinstance(source, RemoteTableSource):
                kind = node.metadata.provenance.kind
                source_name = source.qualified_name
                capabilities = source.capabilities
            elif isinstance(source, FeatureParquetSource) and source.cache_root is not None:
                kind = SourceKind.FEATURE_STORE
                source_name = sanitize_source_location(source.source_root)
                capabilities = _REMOTE_PARQUET_CAPABILITIES
            elif isinstance(source, ParquetSource) and any("://" in path for path in source.paths):
                kind = SourceKind.PARQUET
                source_name = ",".join(sanitize_source_location(path) for path in source.paths)
                capabilities = _REMOTE_PARQUET_CAPABILITIES
            else:
                return
            ordered = tuple(
                operation for operation in _SOURCE_OPERATION_ORDER if operation in requested
            )
            pushdown_candidates = tuple(
                operation
                for operation in ordered
                if operation not in blocked and getattr(capabilities, operation.value)
            )
            local_required = tuple(
                operation
                for operation in ordered
                if operation in blocked or not getattr(capabilities, operation.value)
            )
            fragments.append(
                SourceFragment(
                    kind=kind,
                    source=source_name,
                    capabilities=capabilities,
                    requested=ordered,
                    pushdown_candidates=pushdown_candidates,
                    local_required=local_required,
                )
            )
            return
        if isinstance(node, (JoinPlan, AsOfJoinPlan)):
            branch_operations = requested | {SourceOperation.JOIN}
            visit(node.left, branch_operations, blocked)
            visit(node.right, branch_operations, blocked)
            return
        if isinstance(node, UnionPlan):
            for item in node.inputs:
                visit(item, requested, blocked)
            return

        operations: set[SourceOperation] = set()
        blocked_operations: set[SourceOperation] = set()
        if isinstance(node, ProjectPlan):
            operations.add(SourceOperation.PROJECTION)
            if not all(isinstance(item.expression, ColumnRef) for item in node.projections):
                blocked_operations.add(SourceOperation.PROJECTION)
            if any(expression_metadata(item.expression).has_window for item in node.projections):
                operations.add(SourceOperation.WINDOW)
        elif isinstance(node, FilterPlan):
            operations.add(SourceOperation.FILTER)
            if not _source_filter_expression(node.predicate):
                blocked_operations.add(SourceOperation.FILTER)
        elif isinstance(node, AggregatePlan):
            operations.add(SourceOperation.AGGREGATION)
        elif isinstance(node, LimitPlan):
            operations.add(SourceOperation.LIMIT)
        elif isinstance(node, TopKPlan):
            operations.update((SourceOperation.LIMIT, SourceOperation.SORT))
        elif isinstance(node, SortPlan):
            operations.add(SourceOperation.SORT)
        visit(
            node.input,
            requested | operations,
            blocked | blocked_operations,
        )

    visit(plan, frozenset(), frozenset())
    return tuple(fragments)


def _fragment_to_dict(fragment: SourceFragment) -> dict[str, object]:
    return {
        "kind": fragment.kind.value,
        "source": fragment.source,
        "requested": [operation.value for operation in fragment.requested],
        "pushdown_candidates": [operation.value for operation in fragment.pushdown_candidates],
        "local_required": [operation.value for operation in fragment.local_required],
        "estimated_transfer_bytes": fragment.estimated_transfer_bytes,
    }


def _movement_plans(plan: LogicalPlan) -> tuple[dict[str, object], ...]:
    """Describe joins whose inputs must meet inside the owning DuckDB session."""
    movements: list[dict[str, object]] = []

    def sources(node: LogicalPlan) -> tuple[tuple[str, tuple[str, ...]], ...]:
        if isinstance(node, ScanPlan):
            provenance = node.metadata.provenance
            return (
                (
                    provenance.kind.value,
                    tuple(sanitize_source_location(location) for location in provenance.locations),
                ),
            )
        if isinstance(node, (JoinPlan, AsOfJoinPlan)):
            return (*sources(node.left), *sources(node.right))
        if isinstance(node, UnionPlan):
            return tuple(item for child in node.inputs for item in sources(child))
        return sources(node.input)

    def visit(node: LogicalPlan) -> None:
        if isinstance(node, (JoinPlan, AsOfJoinPlan)):
            left = sources(node.left)
            right = sources(node.right)
            if set(left) != set(right):
                movements.append(
                    {
                        "kind": "cross_source_join",
                        "strategy": "stream_inputs_to_duckdb",
                        "left": [
                            {"kind": kind, "locations": list(locations)} for kind, locations in left
                        ],
                        "right": [
                            {"kind": kind, "locations": list(locations)}
                            for kind, locations in right
                        ],
                        "estimated_transfer_bytes": None,
                        "materializes_in_python": False,
                    }
                )
            visit(node.left)
            visit(node.right)
        elif isinstance(node, UnionPlan):
            for item in node.inputs:
                visit(item)
        elif not isinstance(node, ScanPlan):
            visit(node.input)

    visit(plan)
    return tuple(movements)


def _remote_boundaries(plan: LogicalPlan) -> tuple[dict[str, object], ...]:
    """Describe remote source movement and known native pushdown."""
    boundaries: list[dict[str, object]] = []
    for node in _plan_nodes(plan):
        if not isinstance(node, ScanPlan):
            continue
        if isinstance(node.source, FeatureParquetSource) and node.source.cache_root is not None:
            boundaries.append(
                {
                    "kind": "feature_store_cache",
                    "source": sanitize_source_location(node.source.source_root),
                    "location": str(node.source.cache_root),
                    "estimated_transfer_bytes": None,
                    "unbounded_scan": "allow",
                    "pushdown": {
                        "projection": True,
                        "filter": False,
                        "aggregation": False,
                        "join": False,
                        "window": False,
                        "limit": False,
                        "sort": False,
                    },
                }
            )
            continue
        if not isinstance(node.source, RemoteTableSource):
            continue
        source = node.source
        capabilities = source.capabilities
        boundaries.append(
            {
                "kind": "remote_table",
                "engine": source.engine,
                "source": source.qualified_name,
                "location": sanitize_source_location(source.location),
                "estimated_transfer_bytes": None,
                "unbounded_scan": source.unbounded_scan,
                "pushdown": {
                    "projection": capabilities.projection,
                    "filter": capabilities.filter,
                    "aggregation": capabilities.aggregation,
                    "join": capabilities.join,
                    "window": capabilities.window,
                    "limit": capabilities.limit,
                    "sort": capabilities.sort,
                },
            }
        )
    return tuple(boundaries)


class Executor:
    """Execute compiled plans and track observable execution boundaries."""

    def __init__(self, session: Session, compiler: DuckDBCompiler) -> None:
        self._session = session
        self._compiler = compiler

    @_execution_context("collect")
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
            if column.duckdb_type.startswith("DECIMAL(") and column.id in compiled.bindings
        }
        if decimal_labels:
            rel = rel.project(
                *(
                    duckdb.SQLExpression(quote_identifier(label)).cast("VARCHAR").alias(label)
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
                result[label] = result[label].astype(object).where(result[label].notna(), np.nan)

        index_ids = plan.metadata.index.columns
        if index_ids:
            index_labels = [compiled.bindings[column_id] for column_id in index_ids]
            result = result.set_index(index_labels, drop=plan.metadata.index.drop)
            if plan.metadata.index.names:
                result.index.names = list(plan.metadata.index.names)
        hidden_labels = [
            compiled.bindings[column.id]
            for column in plan.metadata.columns
            if column.hidden and column.id not in index_ids
        ]
        if hidden_labels:
            result = result.drop(columns=hidden_labels)
        self._session._last_materialization_report = MaterializationReport(
            reason="explicit collect",
            estimated_bytes=_materialization_upper_bound(plan),
            actual_bytes=int(result.memory_usage(index=True, deep=True).sum()),
            limit_bytes=None,
        )
        return result

    def collect_small(self, plan: LogicalPlan, *, max_bytes: int) -> pd.DataFrame:
        """Collect only when plan shape and fixed-width types prove a byte bound."""
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        estimate = _materialization_upper_bound(plan)
        if estimate is None:
            raise UnsupportedOperationError(
                "collect_small() requires a non-expanding local Parquet plan "
                "with fixed-width output types and no Python UDF"
            )
        if estimate > max_bytes:
            raise MaterializationError(
                f"collect_small() estimated {estimate:,} bytes, exceeding "
                f"the {max_bytes:,}-byte limit"
            )
        result = self.collect(plan)
        actual = int(result.memory_usage(index=True, deep=True).sum())
        report = MaterializationReport(
            reason="explicit collect_small",
            estimated_bytes=estimate,
            actual_bytes=actual,
            limit_bytes=max_bytes,
        )
        self._session._last_materialization_report = report
        if actual > max_bytes:
            raise MaterializationError(
                f"collect_small() materialized {actual:,} bytes, exceeding "
                f"the {max_bytes:,}-byte limit"
            )
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
        if isinstance(plan, (JoinPlan, AsOfJoinPlan)):
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

    @_execution_context("to_arrow")
    def to_arrow(self, plan: LogicalPlan) -> pa.Table:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        result = self._compiler.project_visible(compiled, plan).relation.to_arrow_table()
        self._session._last_materialization_report = MaterializationReport(
            reason="explicit to_arrow",
            estimated_bytes=_materialization_upper_bound(plan),
            actual_bytes=result.nbytes,
            limit_bytes=None,
        )
        return result

    @_execution_context("to_arrow_batches")
    def to_arrow_batches(self, plan: LogicalPlan, *, batch_size: int) -> pa.RecordBatchReader:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        return self._compiler.project_visible(compiled, plan).relation.to_arrow_reader(batch_size)

    @_execution_context("write_parquet")
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

    @_execution_context("write_csv")
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

    @_execution_context("persist")
    def persist(
        self,
        plan: LogicalPlan,
        name: str,
    ) -> None:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        compiled.relation.create(name)

    @_execution_context("save_as_table")
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
            raise ValueError(f"Unknown mode: {mode!r}; expected one of {sorted(valid_modes)}")

        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        visible = self._compiler.project_visible(compiled, plan)
        visible_rel = visible.relation

        con = self._session._connection
        self._session._begin_execution()

        escaped_table = quote_identifier(name)
        tables_query = (
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
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
                    info_rows = con.sql(f"PRAGMA table_info({escaped_table})").fetchall()
                    existing_columns = [str(row[1]) for row in info_rows]
                    existing_types = {str(row[1]): str(row[2]).upper() for row in info_rows}

                    incoming_columns = list(visible_rel.columns)
                    incoming_types = {
                        str(col): str(dtype).upper()
                        for col, dtype in zip(visible_rel.columns, visible_rel.dtypes, strict=True)
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
                        f"column '{col}': expected {existing_types[col]}, got {incoming_types[col]}"
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
                            duckdb.SQLExpression(quote_identifier(col)) for col in existing_columns
                        ]
                        insert_rel = visible_rel.project(*project_exprs)
                    else:
                        insert_rel = visible_rel

                    insert_rel.insert_into(name)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

    @_execution_context("commit")
    def commit(
        self,
        plan: LogicalPlan,
        *,
        compression: ParquetCompression = "snappy",
        retain_previous: bool = False,
        _failure_injector: Callable[[CommitFailurePoint], None] | None = None,
        _replace_file: Callable[[Path, Path], None] = (_replace_file_preserving_metadata),
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
            if isinstance(p, (JoinPlan, AsOfJoinPlan)):
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
            msg = f"commit() only supports ParquetSource, got {type(scan.source).__name__}"
            raise UnsupportedOperationError(msg)
        if len(scan.source.paths) != 1:
            msg = "commit() currently requires a single Parquet file, not multi-file paths"
            raise UnsupportedOperationError(msg)
        provenance = plan.metadata.provenance
        if provenance.kind is not SourceKind.PARQUET or not provenance.writable:
            raise UnsupportedOperationError(
                "commit() requires writable local Parquet source provenance"
            )

        source_path_str = scan.source.paths[0]
        if any(c in source_path_str for c in "*?[]"):
            raise UnsupportedOperationError("commit() does not support wildcard or glob paths")

        source_path = Path(source_path_str).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Source Parquet file does not exist: {source_path}")
        # 2. Capture initial source fingerprint before reading and compilation
        initial_stat = source_path.stat()
        initial_mtime_ns = initial_stat.st_mtime_ns
        initial_size = initial_stat.st_size
        source_parquet = pq.ParquetFile(source_path)
        source_arrow_schema = source_parquet.schema_arrow

        # 3. Inspect schema and row count from Parquet metadata without scanning rows.
        con = self._session._connection
        orig_rel = con.read_parquet(str(source_path))
        orig_columns = list(orig_rel.columns)
        orig_types = [str(t).upper() for t in orig_rel.dtypes]
        orig_count = source_parquet.metadata.num_rows

        # 3. Validate plan columns preserve schema (including hidden index columns)
        col_by_label = {
            col.label: col
            for col in plan.metadata.columns
            if col.label != scan.source.stable_order_label
        }
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
            duckdb.SQLExpression(quote_identifier(compiled.bindings[col_by_label[col].id])).alias(
                col
            )
            for col in orig_columns
        ]
        commit_rel = compiled.relation.project(*project_exprs)

        # Check types match original source types
        commit_types = [str(t).upper() for t in commit_rel.dtypes]
        type_mismatches = [
            f"column '{col}': expected {orig_t}, got {cur_t}"
            for col, orig_t, cur_t in zip(orig_columns, orig_types, commit_types, strict=True)
            if orig_t != cur_t
        ]
        if type_mismatches:
            msg = f"commit() cannot alter source column types: {', '.join(type_mismatches)}"
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
            # 7. Stream through DuckDB COPY and retain Arrow key/value metadata.
            metadata = source_arrow_schema.metadata or {}
            metadata_fields = ", ".join(
                f"{quote_identifier(key.decode('utf-8'))}: ?" for key in metadata
            )
            metadata_option = f", KV_METADATA {{{metadata_fields}}}" if metadata_fields else ""
            copy_sql = (
                f"COPY ({commit_rel.sql_query()}) TO ? "
                f"(FORMAT PARQUET, COMPRESSION ?, RETURN_STATS{metadata_option})"
            )
            copy_parameters: list[object] = [
                str(staging_path),
                compression,
                *metadata.values(),
            ]
            if _failure_injector is not None:
                _failure_injector("during_write")
            copy_row = con.execute(copy_sql, copy_parameters).fetchone()
            if copy_row is None:
                raise MaterializationError("DuckDB COPY returned no write statistics")
            copy_rows_written = int(cast("int", copy_row[1]))
            copy_bytes_written = int(cast("int", copy_row[2]))
            copy_column_stats = cast("dict[str, object]", copy_row[4])
            if _failure_injector is not None:
                _failure_injector("after_staging_write")

            if _failure_injector is not None:
                _failure_injector("during_validation")

            # 8. Validate output readability, row-preservation, and schema
            staging_parquet = pq.ParquetFile(staging_path)
            staging_rel = con.read_parquet(str(staging_path))
            staging_count = copy_rows_written
            if staging_count != orig_count:
                msg = (
                    f"commit() requires a row-preserving plan; "
                    f"row count changed from {orig_count} to {staging_count}"
                )
                raise UnsupportedOperationError(msg)

            staging_columns = list(staging_rel.columns)
            if staging_columns != orig_columns:
                msg = f"Committed schema {staging_columns} does not match original {orig_columns}"
                raise ValueError(msg)
            staging_arrow_schema = staging_parquet.schema_arrow
            if not staging_arrow_schema.equals(source_arrow_schema, check_metadata=True):
                raise ValueError("Committed Parquet schema metadata does not match source")

            # 9. Concurrency guard: verify source has not been modified
            current_stat = source_path.stat()
            if current_stat.st_mtime_ns != initial_mtime_ns or current_stat.st_size != initial_size:
                msg = f"Source file '{source_path}' was modified concurrently during commit"
                raise ConcurrentModificationError(msg)

            # 10. Optional retention of previous version
            backup_path: str | None = None
            if retain_previous:
                if _failure_injector is not None:
                    _failure_injector("before_backup")
                backup_file = dest_dir / f"{source_path.stem}_backup_{uuid4().hex[:8]}.parquet"
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
            bytes_written = copy_bytes_written
            t1 = time.perf_counter()

            return CommitReport(
                source_path=str(source_path),
                staging_path=str(staging_path),
                backup_path=backup_path,
                rows_written=staging_count,
                files_written=1,
                columns_written=len(copy_column_stats),
                row_groups_written=staging_parquet.metadata.num_row_groups,
                bytes_written=bytes_written,
                duration_seconds=t1 - t0,
            )
        finally:
            if staging_path.exists():
                staging_path.unlink(missing_ok=True)
            if not replaced and backup_file is not None and backup_file.exists():
                backup_file.unlink(missing_ok=True)

    @_execution_context("explain")
    def explain(
        self,
        plan: LogicalPlan,
        *,
        mode: Literal["all", "logical", "optimized", "json", "sql", "physical", "analyze"] = "all",
    ) -> str:
        optimization = self._compiler.optimize(plan)
        logical = json.dumps(plan_to_dict(plan), indent=2)
        optimized = json.dumps(plan_to_dict(optimization.plan), indent=2)
        fallback_boundaries = _fallback_boundaries(plan)
        fallback_text = (
            f"explicit typed Arrow UDFs {fallback_boundaries}" if fallback_boundaries else "none"
        )
        remote_boundaries = _remote_boundaries(plan)
        remote_text = json.dumps(remote_boundaries, sort_keys=True) if remote_boundaries else "none"
        source_fragments = tuple(
            _fragment_to_dict(fragment) for fragment in _source_fragments(optimization.plan)
        )
        movement_plans = _movement_plans(optimization.plan)
        boundaries = (
            f"Fallback boundaries: {fallback_text} (policy=error)\n"
            "Materialization boundaries: none in the logical plan\n"
            f"Remote source boundaries: {remote_text}\n"
            f"Source fragments: {json.dumps(source_fragments, sort_keys=True)}\n"
            f"Cross-source movement: {json.dumps(movement_plans, sort_keys=True)}"
        )
        if mode == "logical":
            return f"{boundaries}\nDuckPD logical plan:\n{logical}"
        if mode == "optimized":
            return f"{boundaries}\nDuckPD optimized logical plan:\n{optimized}"
        if mode == "json":
            payload = optimization.to_dict()
            payload["execution_boundaries"] = {
                "fallback_policy": "error",
                "fallback": list(fallback_boundaries),
                "materialization": [],
                "remote": list(remote_boundaries),
                "source_fragments": list(source_fragments),
                "movement": list(movement_plans),
            }
            return json.dumps(payload, indent=2)
        if mode == "analyze":
            self._validate_execution(optimization.plan)

        compiled = self._compiler.compile(optimization.plan, optimize=False)
        relation = compiled.relation
        self._session._begin_execution()
        if mode == "sql":
            sql = _redact_plan_text(relation.sql_query(), plan)
            return f"{boundaries}\nDuckDB SQL:\n{sql}"
        if mode == "physical":
            physical = _redact_plan_text(relation.explain(), plan)
            return f"{boundaries}\nDuckDB physical plan:\n{physical}"
        if mode == "analyze":
            row = self._session._connection.execute(
                f"EXPLAIN ANALYZE {relation.sql_query()}"
            ).fetchone()
            if row is None:
                raise MaterializationError("EXPLAIN ANALYZE returned no plan")
            analyzed = _redact_plan_text(str(row[1]), plan)
            return f"{boundaries}\nDuckDB analyzed physical plan:\n{analyzed}"
        if mode == "all":
            changed = [snapshot.name for snapshot in optimization.snapshots if snapshot.changed]
            sql = _redact_plan_text(relation.sql_query(), plan)
            physical = _redact_plan_text(relation.explain(), plan)
            return (
                f"{boundaries}\n"
                f"DuckPD logical plan:\n{logical}\n\n"
                f"DuckPD optimized logical plan:\n{optimized}\n"
                f"Applied rewrites: {changed}\n\n"
                f"DuckDB SQL:\n{sql}\n\n"
                f"DuckDB physical plan:\n{physical}"
            )
        msg = (
            f"Unknown explain mode: {mode!r}; expected 'all', 'logical', "
            "'optimized', 'json', 'sql', 'physical', or 'analyze'"
        )
        raise ValueError(msg)

    @_execution_context("explain_write")
    def explain_write(
        self,
        plan: LogicalPlan,
        path: str,
        *,
        compression: ParquetCompression = "snappy",
    ) -> str:
        """Inspect write strategy and estimates without counting or writing rows."""
        compiled = self._compiler.compile(plan)
        visible_rel = self._compiler.project_visible(compiled, plan).relation
        self._session._begin_execution()
        nodes = tuple(_plan_nodes(plan))
        blocking_types = (
            SortPlan,
            TopKPlan,
            AggregatePlan,
            JoinPlan,
            AsOfJoinPlan,
            LocIndexPlan,
        )
        blocking = tuple(
            dict.fromkeys(type(node).__name__ for node in nodes if isinstance(node, blocking_types))
        )
        locations = plan.metadata.provenance.locations
        local_paths = tuple(
            Path(location)
            for location in locations
            if "://" not in location and not any(char in location for char in "*?[]")
        )
        estimate = (
            sum(local_path.stat().st_size for local_path in local_paths)
            if locations
            and len(local_paths) == len(locations)
            and all(local_path.is_file() for local_path in local_paths)
            else None
        )
        estimate_text = f"{estimate:,} bytes" if estimate is not None else "unknown"
        extra_disk = estimate_text if blocking and estimate is not None else "engine-dependent"
        physical = _redact_plan_text(visible_rel.explain(), plan)
        target = sanitize_source_location(path)
        return (
            "Fallback boundaries: none (policy=error)\n"
            "Materialization boundary: explicit Parquet write sink\n"
            f"Write target: {target}\n"
            f"Compression: {compression}\n"
            f"Output columns: {list(plan.metadata.visible_columns)}\n"
            f"Estimated input bytes: {estimate_text} "
            "(estimate from local file metadata; no row count executed)\n"
            f"Blocking operators: {blocking or ('none',)}\n"
            "Known non-spillable aggregate states: none; "
            "list/string_agg are rejected before execution\n"
            f"Expected extra disk use: {extra_disk} (estimate)\n"
            f"DuckDB physical plan:\n{physical}"
        )

    @_execution_context("profile")
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
            visible_rel = self._compiler.project_visible(compiled, optimization.plan).relation
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
            fallback_boundaries=_fallback_boundaries(plan),
            remote_boundaries=_remote_boundaries(plan),
            source_fragments=tuple(
                _fragment_to_dict(fragment) for fragment in _source_fragments(optimization.plan)
            ),
            movement_plans=_movement_plans(optimization.plan),
            measured_transfer_bytes=None,
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
        if isinstance(plan, ScanPlan) and isinstance(plan.source, RemoteTableSource):
            source = plan.source
            message = (
                f"Remote {source.engine} scan {source.qualified_name!r} has no "
                "proven transfer bound; projection/filter pushdown is "
                f"{'available' if source.capabilities.filter else 'limited'}"
            )
            if source.unbounded_scan == "error":
                raise MaterializationError(
                    f"{message}; set unbounded_scan='warn' or 'allow' explicitly"
                )
            if source.unbounded_scan == "warn":
                warnings.warn(message, RemoteScanWarning, stacklevel=3)
            return
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
                        "Cannot take a larger sample than population when 'replace=False'"
                    )
            return
        if isinstance(
            plan,
            (FilterPlan, ProjectPlan, SortPlan, TopKPlan, LimitPlan, AggregatePlan),
        ):
            self._validate_execution(plan.input)

    def _validate_loc_plan(self, plan: LocIndexPlan) -> None:
        compiled_input = self._compiler.compile(plan.input)
        index_ids = plan.input.metadata.index.columns
        index_labels = [compiled_input.bindings[column_id] for column_id in index_ids]

        keys_df = cast("pd.DataFrame", self._session._get_registered_source(plan.source_key))
        keys_relation = self._session._connection.from_df(keys_df).set_alias("__duckpd_loc_keys__")

        input_alias = "__duckpd_loc_input__"
        matched_label = f"__duckpd_loc_matched_{plan.source_key}__"
        flagged_input = compiled_input.relation.project(
            f"*, 1 AS {quote_identifier(matched_label)}"
        ).set_alias(input_alias)

        conditions = [
            f"__duckpd_loc_keys__.{quote_identifier(key_label)} "
            f"IS NOT DISTINCT FROM {input_alias}.{quote_identifier(index_label)}"
            for key_label, index_label in zip(plan.key_labels, index_labels, strict=True)
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
                    f"Merge keys are not unique in {side} dataset; not a {relationship} merge"
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
                f"Merge keys are not unique in {side} dataset; not a {relationship} merge"
            )
