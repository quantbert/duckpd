"""Lazy DataFrame public API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import pandas as pd
import pyarrow as pa

from duckpd._logical import (
    AggregateOperator,
    Column,
    ColumnId,
    ColumnRef,
    FilterPlan,
    LimitPlan,
    LiteralValue,
    NamedExpression,
    NullPlacement,
    ProjectPlan,
    SortDirection,
    SortKey,
    SortPlan,
)
from duckpd._metadata import (
    after_projection,
    after_sort,
    find_column,
    projection_columns,
    protected_column_ids,
)
from duckpd._metadata import reset_index as reset_index_metadata
from duckpd._metadata import set_index as set_index_metadata
from duckpd._reductions import (
    aggregate_plan,
    expression_type,
    is_numeric_type,
    materialized_int,
    validate_axis,
    validate_min_count,
)
from duckpd._typing import ParquetCompression, is_scalar_value
from duckpd.errors import AlignmentError, UnsupportedOperationError

if TYPE_CHECKING:
    from duckpd._logical import Expression, LogicalPlan
    from duckpd.groupby import DataFrameGroupBy
    from duckpd.series import Series
    from duckpd.session import Session


class DataFrame:
    """A lazy pandas-shaped DataFrame backed by a DuckDB query plan."""

    def __init__(self, session: Session, plan: LogicalPlan) -> None:
        self._session = session
        self._plan = plan

    @property
    def columns(self) -> tuple[str, ...]:
        """Displayed column labels without executing the plan."""
        return tuple(column.label for column in self._plan.metadata.visible_columns)

    @property
    def index_names(self) -> tuple[str, ...]:
        """Names of explicit index columns without executing the plan."""
        return tuple(
            self._column_by_id(column_id).label
            for column_id in self._plan.metadata.index.columns
        )

    @property
    def ordering(self) -> tuple[str, ...]:
        """Names of columns that guarantee the current row order."""
        return tuple(
            self._column_by_id(key.column_id).label
            for key in self._plan.metadata.ordering.keys
        )

    def collect(self) -> pd.DataFrame:
        """Execute the complete plan and return a pandas DataFrame."""
        return self._session._executor.collect(self._plan)

    def to_pandas(self) -> pd.DataFrame:
        """Alias for :meth:`collect`."""
        return self.collect()

    def to_arrow(self) -> pa.Table:
        """Execute the complete plan and return an Arrow table."""
        return self._session._executor.to_arrow(self._plan)

    def to_arrow_batches(self, batch_size: int = 1_000_000) -> pa.RecordBatchReader:
        """Execute the plan and stream its rows as Arrow record batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return self._session._executor.to_arrow_batches(
            self._plan, batch_size=batch_size
        )

    def write_parquet(
        self,
        path: str | Path,
        *,
        compression: ParquetCompression = "snappy",
        overwrite: bool = False,
    ) -> None:
        """Execute the plan directly into a Parquet file."""
        self._session._executor.write_parquet(
            self._plan,
            str(path),
            compression=compression,
            overwrite=overwrite,
        )

    def explain(
        self,
        mode: Literal["all", "logical", "sql", "physical"] = "all",
    ) -> str:
        """Return DuckDB's execution plan views without fetching result rows."""
        return self._session._executor.explain(self._plan, mode=mode)

    def explain_write(
        self,
        path: str | Path,
        *,
        compression: ParquetCompression = "snappy",
    ) -> str:
        """Inspect write strategy and execution plan without writing rows."""
        return self._session._executor.explain_write(
            self._plan, str(path), compression=compression
        )

    @property
    def size(self) -> int:
        """Return the number of visible elements in the frame."""
        plan = aggregate_plan(
            self._plan,
            (("__duckpd_size__", None, None),),
            AggregateOperator.SIZE,
        )
        rows = materialized_int(self._session._executor.reduce_scalar(plan))
        return rows * len(self._plan.metadata.visible_columns)

    def count(
        self,
        axis: int | str = 0,
        numeric_only: bool = False,
    ) -> pd.Series:
        """Return the number of non-null values in each supported column."""
        validate_axis(axis, series=False)
        columns = self._reduction_columns(numeric_only=numeric_only)
        return self._reduce_columns(AggregateOperator.COUNT, columns)

    def sum(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
    ) -> pd.Series:
        """Return column-wise sums for supported numeric and boolean columns."""
        validate_axis(axis, series=False)
        validate_min_count(min_count)
        columns = self._reduction_columns(
            numeric_only=numeric_only,
            require_numeric=True,
        )
        return self._reduce_columns(
            AggregateOperator.SUM,
            columns,
            skipna=skipna,
            min_count=min_count,
        )

    def mean(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> pd.Series:
        """Return column-wise means for supported numeric and boolean columns."""
        validate_axis(axis, series=False)
        columns = self._reduction_columns(
            numeric_only=numeric_only,
            require_numeric=True,
        )
        return self._reduce_columns(
            AggregateOperator.MEAN,
            columns,
            skipna=skipna,
        )

    def min(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> pd.Series:
        """Return column-wise minima for supported numeric and boolean columns."""
        validate_axis(axis, series=False)
        columns = self._reduction_columns(
            numeric_only=numeric_only,
            require_numeric=True,
        )
        return self._reduce_columns(
            AggregateOperator.MIN,
            columns,
            skipna=skipna,
        )

    def max(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> pd.Series:
        """Return column-wise maxima for supported numeric and boolean columns."""
        validate_axis(axis, series=False)
        columns = self._reduction_columns(
            numeric_only=numeric_only,
            require_numeric=True,
        )
        return self._reduce_columns(
            AggregateOperator.MAX,
            columns,
            skipna=skipna,
        )

    @overload
    def __getitem__(self, key: str) -> Series: ...

    @overload
    def __getitem__(self, key: list[str] | tuple[str, ...]) -> DataFrame: ...

    @overload
    def __getitem__(self, key: Series) -> DataFrame: ...

    def __getitem__(self, key: str | Sequence[str] | Series) -> Series | DataFrame:
        from duckpd.series import Series

        if isinstance(key, str):
            column = self._column(key)
            return Series(self._session, self._plan, ColumnRef(column.id), key)
        if isinstance(key, Series):
            self._require_same_plan(key)
            return DataFrame(
                self._session,
                FilterPlan(self._plan, key._expression, self._plan.metadata),
            )

        labels = tuple(key)
        if not labels:
            raise ValueError("DuckPD does not yet support empty projections")
        if len(labels) != len(set(labels)):
            raise ValueError("DuckPD does not yet support duplicate column labels")
        projections = tuple(
            NamedExpression(column, ColumnRef(column.id))
            for column in (self._column(label) for label in labels)
        )
        selected = tuple(projection.column for projection in projections)
        columns = projection_columns(self._plan.metadata, selected)
        projections = tuple(
            NamedExpression(column, ColumnRef(column.id)) for column in columns
        )
        return DataFrame(
            self._session,
            ProjectPlan(
                self._plan,
                projections,
                after_projection(self._plan.metadata, columns),
            ),
        )

    def assign(
        self,
        **columns: object | Callable[[DataFrame], object],
    ) -> DataFrame:
        """Assign columns sequentially while keeping the result lazy."""
        original_plan = self._plan
        frame = self
        for label, value in columns.items():
            resolved = value(frame) if callable(value) else value
            expression = frame._coerce_expression(
                resolved, alternate_plan=original_plan
            )
            try:
                existing = find_column(frame._plan.metadata, label, include_hidden=True)
            except KeyError:
                existing = None
            if existing is not None and existing.id in protected_column_ids(
                frame._plan.metadata
            ):
                raise ValueError(
                    "Cannot replace an index or ordering column; reset metadata first"
                )
            output = Column(
                ColumnId.create(),
                label,
                expression_type(frame._plan, expression),
            )
            projections = (
                *(
                    NamedExpression(column, ColumnRef(column.id))
                    for column in frame._plan.columns
                    if column.label != label
                ),
                NamedExpression(output, expression),
            )
            projected_columns = tuple(projection.column for projection in projections)
            plan = ProjectPlan(
                frame._plan,
                projections,
                after_projection(frame._plan.metadata, projected_columns),
            )
            frame = DataFrame(frame._session, plan)
        return frame

    def sort_values(
        self,
        by: str | Sequence[str],
        *,
        ascending: bool | Sequence[bool] = True,
        na_position: Literal["first", "last"] = "last",
    ) -> DataFrame:
        """Return a lazy frame ordered by one or more columns."""
        labels = (by,) if isinstance(by, str) else tuple(by)
        if not labels:
            raise ValueError("sort_values requires at least one column")
        if na_position not in {"first", "last"}:
            raise ValueError("na_position must be 'first' or 'last'")

        directions = (
            (ascending,) * len(labels)
            if isinstance(ascending, bool)
            else tuple(ascending)
        )
        if len(directions) != len(labels):
            raise ValueError("Length of ascending must match length of by")

        null_placement = (
            NullPlacement.FIRST if na_position == "first" else NullPlacement.LAST
        )
        keys = tuple(
            SortKey(
                ColumnRef(self._column(label).id),
                SortDirection.ASCENDING if direction else SortDirection.DESCENDING,
                null_placement,
            )
            for label, direction in zip(labels, directions, strict=True)
        )
        return DataFrame(
            self._session,
            SortPlan(self._plan, keys, after_sort(self._plan.metadata, keys)),
        )

    def groupby(
        self,
        by: str | Sequence[str],
        *,
        as_index: bool = True,
        sort: bool = True,
        dropna: bool = True,
        observed: bool = False,
    ) -> DataFrameGroupBy:
        """Group DataFrame using a mapper or by a Series of columns."""
        from duckpd.groupby import DataFrameGroupBy

        return DataFrameGroupBy(
            self,
            by=by,
            as_index=as_index,
            sort=sort,
            dropna=dropna,
            observed=observed,
        )

    def merge(
        self,
        right: DataFrame,
        how: Literal["left", "right", "outer", "inner", "cross"] = "inner",
        on: str | Sequence[str] | None = None,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = False,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> DataFrame:
        """Merge DataFrame or named Series objects with a database-style join."""
        from duckpd._merging import plan_merge

        if self._session is not right._session:
            raise AlignmentError("Cannot merge frames from different sessions")

        plan = plan_merge(
            self,
            right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
        )
        return DataFrame(self._session, plan)

    def join(
        self,
        other: DataFrame,
        on: str | Sequence[str] | None = None,
        how: Literal["left", "right", "outer", "inner"] = "left",
        lsuffix: str = "",
        rsuffix: str = "",
        sort: bool = False,
    ) -> DataFrame:
        """Join columns of another DataFrame using index or a key column."""
        return self.merge(
            other,
            how=how,
            left_on=on,
            right_index=True,
            left_index=(on is None),
            sort=sort,
            suffixes=(lsuffix, rsuffix),
        )

    def limit(self, count: int, *, offset: int = 0) -> DataFrame:
        """Return a lazy frame containing at most ``count`` rows."""
        if count < 0:
            raise ValueError("count must be non-negative")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        return DataFrame(
            self._session,
            LimitPlan(self._plan, count, offset, self._plan.metadata),
        )

    def set_index(self, keys: str | Sequence[str], *, drop: bool = True) -> DataFrame:
        """Set one or more existing columns as an explicit lazy index."""
        labels = (keys,) if isinstance(keys, str) else tuple(keys)
        if not labels:
            raise ValueError("set_index requires at least one column")
        if len(labels) != len(set(labels)):
            raise ValueError("set_index keys must be unique")
        columns = tuple(self._column(label) for label in labels)
        metadata = set_index_metadata(self._plan.metadata, columns, drop=drop)
        return self._identity_project(metadata)

    def reset_index(self, *, drop: bool = False) -> DataFrame:
        """Remove the explicit index without executing the plan."""
        metadata = reset_index_metadata(self._plan.metadata, drop=drop)
        return self._identity_project(metadata)

    def head(self, count: int = 5) -> pd.DataFrame:
        """Execute a bounded preview and return a pandas DataFrame."""
        if count < 0:
            raise ValueError("count must be non-negative")
        return self.limit(count).collect()

    def __repr__(self) -> str:
        labels = ", ".join(repr(label) for label in self.columns)
        plan_name = type(self._plan).__name__
        return f"DuckPD DataFrame\nColumns: [{labels}]\nPlan: {plan_name}"

    def _column(self, label: str) -> Column:
        return find_column(self._plan.metadata, label)

    def _column_by_id(self, column_id: ColumnId) -> Column:
        for column in self._plan.columns:
            if column.id == column_id:
                return column
        raise AssertionError(f"Column ID is missing from metadata: {column_id}")

    def _identity_project(self, metadata: object) -> DataFrame:
        from duckpd._logical import FrameMetadata

        if not isinstance(metadata, FrameMetadata):
            raise TypeError("Expected FrameMetadata")
        projections = tuple(
            NamedExpression(column, ColumnRef(column.id)) for column in metadata.columns
        )
        return DataFrame(
            self._session,
            ProjectPlan(self._plan, projections, metadata),
        )

    def _require_same_plan(self, series: Series) -> None:
        if series._session is not self._session or series._plan is not self._plan:
            raise AlignmentError(
                "Series from a different frame requires explicit index alignment"
            )

    def _coerce_expression(
        self, value: object, *, alternate_plan: LogicalPlan | None = None
    ) -> Expression:
        from duckpd.series import Series

        if isinstance(value, Series):
            same_session = value._session is self._session
            compatible_plan = value._plan is self._plan or value._plan is alternate_plan
            if not same_session or not compatible_plan:
                raise AlignmentError(
                    "Series from a different frame requires explicit index alignment"
                )
            return value._expression
        if isinstance(value, DataFrame):
            raise TypeError("assign values must be scalar or Series expressions")
        if not is_scalar_value(value):
            raise TypeError("DuckPD does not support this scalar literal type")
        return LiteralValue(value)

    def _reduction_columns(
        self,
        *,
        numeric_only: bool,
        require_numeric: bool = False,
    ) -> tuple[Column, ...]:
        visible = self._plan.metadata.visible_columns
        numeric = tuple(
            column for column in visible if is_numeric_type(column.duckdb_type)
        )
        if numeric_only:
            if not numeric:
                raise UnsupportedOperationError(
                    "No numeric columns are available for this reduction"
                )
            return numeric
        if require_numeric and len(numeric) != len(visible):
            unsupported = next(
                column for column in visible if not is_numeric_type(column.duckdb_type)
            )
            raise UnsupportedOperationError(
                "This reduction currently supports only numeric and boolean data; "
                f"column {unsupported.label!r} has DuckDB type "
                f"{unsupported.duckdb_type}. Pass numeric_only=True to exclude it."
            )
        return visible

    def _reduce_columns(
        self,
        operator: AggregateOperator,
        columns: tuple[Column, ...],
        *,
        skipna: bool = True,
        min_count: int = 0,
    ) -> pd.Series:
        requests = tuple(
            (column.label, ColumnRef(column.id), column.duckdb_type)
            for column in columns
        )
        plan = aggregate_plan(
            self._plan,
            requests,
            operator,
            skipna=skipna,
            min_count=min_count,
        )
        return self._session._executor.reduce_columns(plan)
