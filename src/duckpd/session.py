"""DuckDB connection and resource ownership."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import duckdb
import pandas as pd
import pyarrow as pa

from duckpd._compiler import DuckDBCompiler
from duckpd._executor import Executor
from duckpd._logical import (
    ArrowSource,
    ColumnRef,
    CsvSource,
    FrameMetadata,
    NullPlacement,
    OrderColumn,
    OrderSpec,
    PandasSource,
    ParquetSource,
    ScanPlan,
    SortDirection,
    SortKey,
    SortPlan,
    SqlSource,
    TableSource,
)
from duckpd._metadata import after_sort, sort_keys_for_labels, source_metadata
from duckpd.errors import SessionClosedError, UnsupportedOperationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckpd.frame import DataFrame


class Session:
    """Own a DuckDB connection and every source referenced by its plans."""

    def __init__(
        self,
        database: str | Path = ":memory:",
        *,
        read_only: bool = False,
        memory_limit: str | None = None,
        temp_directory: str | Path | None = None,
        max_temp_directory_size: str | None = None,
        threads: int | None = None,
    ) -> None:
        config: dict[str, str | bool | int | float | list[str]] = {}
        if memory_limit is not None:
            config["memory_limit"] = memory_limit
        if temp_directory is not None:
            config["temp_directory"] = str(temp_directory)
        if max_temp_directory_size is not None:
            config["max_temp_directory_size"] = max_temp_directory_size
        if threads is not None:
            config["threads"] = threads

        self._connection = duckdb.connect(
            database=str(database),
            read_only=read_only,
            config=config,
        )
        self._registered_sources: dict[str, object] = {}
        self._closed = False
        self._execution_count = 0
        self._compiler = DuckDBCompiler(self)
        self._executor = Executor(self, self._compiler)

    @property
    def execution_count(self) -> int:
        """Number of explicit execution boundaries entered by this session."""
        return self._execution_count

    @property
    def closed(self) -> bool:
        """Whether this session has released its connection."""
        return self._closed

    def from_pandas(
        self,
        value: pd.DataFrame,
        *,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Create a lazy frame while retaining the pandas source."""
        from duckpd.frame import DataFrame

        self._ensure_open()
        if not value.columns.is_unique:
            msg = "DuckPD does not yet support duplicate column labels"
            raise ValueError(msg)
        labels = cast("list[object]", value.columns.to_list())
        if not all(isinstance(label, str) for label in labels):
            msg = "DuckPD currently requires string column labels"
            raise TypeError(msg)

        key = uuid4().hex
        ordinal_label = f"__duckpd_row_ordinal_{key}__"
        snapshot = value.copy()
        snapshot[ordinal_label] = range(len(snapshot))
        self._registered_sources[key] = snapshot
        source = PandasSource(key)
        plan = self._source_plan(
            source,
            index=index,
            order_by=order_by,
            stable_order_label=ordinal_label,
        )
        return DataFrame(self, plan)

    def from_arrow(
        self,
        value: pa.Table | pa.RecordBatch,
        *,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Create a lazy frame while retaining the Arrow source."""
        from duckpd.frame import DataFrame

        self._ensure_open()
        key = uuid4().hex
        ordinal_label = f"__duckpd_row_ordinal_{key}__"
        ordered_value = value.append_column(
            ordinal_label, pa.array(range(value.num_rows), type=pa.int64())
        )
        self._registered_sources[key] = ordered_value
        source = ArrowSource(key)
        plan = self._source_plan(
            source,
            index=index,
            order_by=order_by,
            stable_order_label=ordinal_label,
        )
        return DataFrame(self, plan)

    def read_parquet(
        self,
        path: str | Path | Sequence[str | Path],
        *,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Create a lazy scan over one or more Parquet files."""
        from duckpd.frame import DataFrame

        self._ensure_open()
        if isinstance(path, (str, Path)):
            raw_paths = (str(path),)
        else:
            raw_paths = tuple(str(item) for item in path)
        paths = tuple(
            item if "://" in item else str(Path(item).expanduser().resolve())
            for item in raw_paths
        )
        if not paths:
            msg = "At least one Parquet path is required"
            raise ValueError(msg)

        source = ParquetSource(paths, hive_partitioning, union_by_name)
        plan = self._source_plan(source, index=index, order_by=order_by)
        return DataFrame(self, plan)

    def read_csv(
        self,
        path: str | Path | Sequence[str | Path],
        *,
        header: bool = True,
        delimiter: str = ",",
        auto_detect: bool = True,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Create a lazy scan over one or more CSV files."""
        from duckpd.frame import DataFrame

        self._ensure_open()
        if isinstance(path, (str, Path)):
            paths = (str(path),)
        else:
            paths = tuple(str(item) for item in path)
        if not paths:
            msg = "At least one CSV path is required"
            raise ValueError(msg)

        source = CsvSource(
            paths,
            header=header,
            delimiter=delimiter,
            auto_detect=auto_detect,
        )
        plan = self._source_plan(source, index=index, order_by=order_by)
        return DataFrame(self, plan)

    def table(
        self,
        name: str,
        *,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Create a lazy frame for a table in this session."""
        from duckpd.frame import DataFrame

        self._ensure_open()
        source = TableSource(name)
        plan = self._source_plan(source, index=index, order_by=order_by)
        return DataFrame(self, plan)

    def sql(
        self,
        query: str,
        *,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Create a lazy frame from exactly one read-only SQL query."""
        from duckpd.frame import DataFrame

        self._ensure_open()
        statements = self._connection.extract_statements(query)
        if len(statements) != 1:
            msg = "Session.sql() requires exactly one SELECT statement"
            raise UnsupportedOperationError(msg)
        if statements[0].type != duckdb.StatementType.SELECT:
            msg = "Session.sql() only accepts read-only SELECT statements"
            raise UnsupportedOperationError(msg)

        source = SqlSource(query)
        plan = self._source_plan(source, index=index, order_by=order_by)
        return DataFrame(self, plan)

    def close(self) -> None:
        """Release the connection and retained Python sources."""
        if self._closed:
            return
        self._registered_sources.clear()
        self._connection.close()
        self._closed = True

    def __enter__(self) -> Session:
        self._ensure_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            msg = "DuckPD session is closed"
            raise SessionClosedError(msg)

    def _get_registered_source(self, key: str) -> object:
        self._ensure_open()
        return self._registered_sources[key]

    def _begin_execution(self) -> None:
        self._ensure_open()
        self._execution_count += 1

    def _source_plan(
        self,
        source: (
            ArrowSource
            | CsvSource
            | PandasSource
            | ParquetSource
            | SqlSource
            | TableSource
        ),
        *,
        index: str | Sequence[str] | None,
        order_by: str | Sequence[str] | None,
        stable_order_label: str | None = None,
    ) -> ScanPlan | SortPlan:
        columns = self._compiler.inspect_source(source)
        index_labels = self._normalize_labels(index)
        metadata = source_metadata(columns, index_labels=index_labels)
        stable_order_key: OrderColumn | None = None
        if stable_order_label is not None:
            stable_column = next(
                column
                for column in metadata.columns
                if column.label == stable_order_label
            )
            stable_order_key = OrderColumn(
                stable_column.id,
                SortDirection.ASCENDING,
                NullPlacement.LAST,
            )
            metadata = FrameMetadata(
                tuple(
                    replace(column, hidden=True, row_identity=True)
                    if column.id == stable_column.id
                    else column
                    for column in metadata.columns
                ),
                metadata.index,
                OrderSpec((stable_order_key,)),
            )
        scan = ScanPlan(source, metadata)
        order_labels = self._normalize_labels(order_by)
        if not order_labels:
            return scan
        keys = sort_keys_for_labels(metadata, order_labels)
        if stable_order_key is not None:
            keys = (
                *keys,
                SortKey(
                    ColumnRef(stable_order_key.column_id),
                    stable_order_key.direction,
                    stable_order_key.null_placement,
                ),
            )
        return SortPlan(scan, keys, after_sort(metadata, keys))

    @staticmethod
    def _normalize_labels(value: str | Sequence[str] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        labels = (value,) if isinstance(value, str) else tuple(value)
        if len(labels) != len(set(labels)):
            raise ValueError("Metadata column labels must be unique")
        return labels


def connect(
    database: str | Path = ":memory:",
    *,
    read_only: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
    max_temp_directory_size: str | None = None,
    threads: int | None = None,
) -> Session:
    """Create a DuckPD session."""
    return Session(
        database,
        read_only=read_only,
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        max_temp_directory_size=max_temp_directory_size,
        threads=threads,
    )
