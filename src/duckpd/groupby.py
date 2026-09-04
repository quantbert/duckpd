"""Lazy GroupBy public API and aggregation execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast, overload

from duckpd._logical import (
    AggregateExpression,
    AggregateOperator,
    AggregatePlan,
    Column,
    ColumnId,
    ColumnRef,
    NullPlacement,
    OrderColumn,
    SortDirection,
)
from duckpd._metadata import after_aggregate, find_column
from duckpd._reductions import is_numeric_type
from duckpd.errors import AlignmentError, UnsupportedOperationError

if TYPE_CHECKING:
    from duckpd.frame import DataFrame
    from duckpd.series import Series
    from duckpd.session import Session


class DataFrameGroupBy:
    """A lazy GroupBy wrapper over a DataFrame."""

    def __init__(
        self,
        frame: DataFrame,
        by: str | Sequence[str],
        *,
        as_index: bool = True,
        sort: bool = True,
        dropna: bool = True,
        observed: bool = True,
    ) -> None:
        self._frame = frame
        self._session: Session = frame._session
        self._by = (by,) if isinstance(by, str) else tuple(by)
        if not self._by:
            raise ValueError("No group keys were provided")
        if len(self._by) != len(set(self._by)):
            raise ValueError("Duplicate group keys are not supported")

        # validate that all group keys exist in frame
        self._key_columns: list[Column] = []
        for key in self._by:
            self._key_columns.append(find_column(frame._plan.metadata, key))

        self._as_index = as_index
        self._sort = sort
        self._dropna = dropna
        if not observed:
            raise UnsupportedOperationError(
                "DuckPD does not support unobserved categorical groups"
            )

    @overload
    def __getitem__(self, key: str) -> SeriesGroupBy: ...

    @overload
    def __getitem__(self, key: list[str] | tuple[str, ...]) -> DataFrameGroupBy: ...

    def __getitem__(self, key: str | Sequence[str]) -> SeriesGroupBy | DataFrameGroupBy:
        if isinstance(key, str):
            series = self._frame[key]
            return SeriesGroupBy(
                series,
                by=self._by,
                as_index=self._as_index,
                sort=self._sort,
                dropna=self._dropna,
            )
        labels = tuple(key)
        # Project frame down to group keys + selected columns
        all_labels = list(self._by) + [
            label for label in labels if label not in self._by
        ]
        projected_frame = self._frame[all_labels]
        return DataFrameGroupBy(
            projected_frame,
            by=self._by,
            as_index=self._as_index,
            sort=self._sort,
            dropna=self._dropna,
        )

    def agg(
        self,
        func: object = None,
        *args: object,
        **kwargs: object,
    ) -> DataFrame:
        """Aggregate using one or more operations."""
        if func is not None and kwargs:
            raise ValueError("Cannot pass both func and named aggregation kwargs")

        if kwargs:
            return self._named_agg(**kwargs)

        if func is None:
            raise ValueError("Must provide at least one aggregation function")

        if isinstance(func, str):
            func_name = func.lower()
            visible = self._frame._plan.metadata.visible_columns
            key_labels = set(self._by)
            data_cols = [col for col in visible if col.label not in key_labels]
            named_kwargs: dict[str, tuple[str, str]] = {}
            for col in data_cols:
                if func_name in {
                    "sum",
                    "mean",
                    "min",
                    "max",
                    "std",
                    "var",
                    "median",
                } and not is_numeric_type(col.duckdb_type):
                    continue
                named_kwargs[col.label] = (col.label, func_name)
            if not named_kwargs and func_name != "size":
                raise UnsupportedOperationError(
                    f"No valid columns for aggregation {func!r}"
                )
            return self._named_agg(**named_kwargs)

        if isinstance(func, dict):
            dict_map = cast("dict[object, object]", func)
            named_kwargs = {}
            for col_name_obj, agg_spec in dict_map.items():
                if not isinstance(col_name_obj, str):
                    raise TypeError(
                        "Dictionary aggregation keys must be column name strings"
                    )
                if isinstance(agg_spec, str):
                    named_kwargs[col_name_obj] = (col_name_obj, agg_spec)
                else:
                    raise UnsupportedOperationError(
                        "Currently only single string function per column "
                        "is supported in dict agg"
                    )
            return self._named_agg(**named_kwargs)

        raise UnsupportedOperationError(
            f"Unsupported agg argument type: {type(func).__name__}"
        )

    def sum(self, numeric_only: bool = False) -> DataFrame:
        """Compute sum of column values for each group."""
        return self._convenience_agg("sum", numeric_only=numeric_only)

    def mean(self, numeric_only: bool = False) -> DataFrame:
        """Compute mean of column values for each group."""
        return self._convenience_agg("mean", numeric_only=numeric_only)

    def min(self, numeric_only: bool = False) -> DataFrame:
        """Compute minimum of column values for each group."""
        return self._convenience_agg("min", numeric_only=numeric_only)

    def max(self, numeric_only: bool = False) -> DataFrame:
        """Compute maximum of column values for each group."""
        return self._convenience_agg("max", numeric_only=numeric_only)

    def count(self) -> DataFrame:
        """Compute count of non-null values for each group."""
        return self._convenience_agg("count", numeric_only=False)

    def size(self) -> DataFrame:
        """Compute group sizes."""
        from duckpd.frame import DataFrame

        aggregates: list[AggregateExpression] = []
        output_columns: list[Column] = []

        key_ids = tuple(col.id for col in self._key_columns)
        for col in self._key_columns:
            out_col = Column(
                ColumnId.create(),
                col.label,
                col.duckdb_type,
                hidden=self._as_index,
            )
            output_columns.append(out_col)
            aggregates.append(
                AggregateExpression(
                    out_col,
                    operator=None,
                    expression=ColumnRef(col.id),
                    input_duckdb_type=col.duckdb_type,
                )
            )

        size_col = Column(ColumnId.create(), "size", "BIGINT", hidden=False)
        output_columns.append(size_col)
        aggregates.append(
            AggregateExpression(
                size_col,
                operator=AggregateOperator.SIZE,
                expression=None,
                input_duckdb_type=None,
            )
        )

        index_ids = tuple(
            col.id for col in output_columns[: len(self._key_columns)] if col.hidden
        )
        ordering_keys = (
            tuple(
                OrderColumn(col.id, SortDirection.ASCENDING, NullPlacement.LAST)
                for col in output_columns[: len(self._key_columns)]
            )
            if self._sort
            else ()
        )
        if not self._sort:
            identity_id = next(
                iter(self._frame._plan.metadata.row_identity.columns),
                None,
            )
            row_identity = next(
                (
                    column
                    for column in self._frame._plan.metadata.columns
                    if column.id == identity_id
                ),
                None,
            )
            if row_identity is not None:
                first_seen = Column(
                    ColumnId.create(),
                    "__duckpd_group_first_seen__",
                    row_identity.duckdb_type,
                    hidden=True,
                )
                output_columns.append(first_seen)
                aggregates.append(
                    AggregateExpression(
                        first_seen,
                        operator=AggregateOperator.MIN,
                        expression=ColumnRef(row_identity.id),
                        input_duckdb_type=row_identity.duckdb_type,
                    )
                )
                ordering_keys = (
                    OrderColumn(
                        first_seen.id, SortDirection.ASCENDING, NullPlacement.LAST
                    ),
                )

        metadata = after_aggregate(
            tuple(output_columns),
            index_ids=index_ids,
            ordering_keys=ordering_keys,
        )

        plan = AggregatePlan(
            self._frame._plan,
            tuple(aggregates),
            metadata,
            keys=key_ids,
            dropna=self._dropna,
            sort=self._sort,
        )
        return DataFrame(self._session, plan)

    def _convenience_agg(self, func_name: str, *, numeric_only: bool) -> DataFrame:
        visible = self._frame._plan.metadata.visible_columns
        key_labels = set(self._by)
        data_cols = [col for col in visible if col.label not in key_labels]
        named_kwargs: dict[str, tuple[str, str]] = {}
        for col in data_cols:
            if numeric_only and not is_numeric_type(col.duckdb_type):
                continue
            if func_name in {"sum", "mean", "min", "max"} and not is_numeric_type(
                col.duckdb_type
            ):
                if numeric_only:
                    continue
                if func_name in {"min", "max"}:
                    pass
                else:
                    continue
            named_kwargs[col.label] = (col.label, func_name)
        if not named_kwargs:
            raise UnsupportedOperationError(
                f"No numeric columns available for {func_name}"
            )
        return self._named_agg(**named_kwargs)

    def _named_agg(
        self,
        **kwargs: object,
    ) -> DataFrame:
        from duckpd.frame import DataFrame

        aggregates: list[AggregateExpression] = []
        output_columns: list[Column] = []

        # 1. Group keys as identity expressions
        key_ids = tuple(col.id for col in self._key_columns)
        for col in self._key_columns:
            out_col = Column(
                ColumnId.create(),
                col.label,
                col.duckdb_type,
                hidden=self._as_index,
            )
            output_columns.append(out_col)
            aggregates.append(
                AggregateExpression(
                    out_col,
                    operator=None,
                    expression=ColumnRef(col.id),
                    input_duckdb_type=col.duckdb_type,
                )
            )

        # 2. Aggregations from kwargs
        for output_name, raw_spec in kwargs.items():
            if not isinstance(raw_spec, tuple):
                msg = (
                    f"Named aggregation for {output_name!r} must be a "
                    "(column_name, agg_func) tuple"
                )
                raise TypeError(msg)
            tup = cast("tuple[object, ...]", raw_spec)
            if len(tup) != 2:
                msg = (
                    f"Named aggregation for {output_name!r} must be a "
                    "(column_name, agg_func) tuple"
                )
                raise TypeError(msg)
            target_col_name: object = tup[0]
            func_name: object = tup[1]
            if not isinstance(target_col_name, str):
                raise TypeError(
                    f"Target column name in {output_name!r} must be a string"
                )
            if not isinstance(func_name, str):
                raise UnsupportedOperationError(
                    f"Callable aggregators in {output_name!r} are not yet supported; "
                    "use a string like 'sum', 'mean', 'count', 'size'"
                )

            func_lower = func_name.lower()
            if func_lower in {"list", "string_agg"}:
                raise UnsupportedOperationError(
                    f"Aggregate {func_name!r} has an unbounded, non-spillable "
                    "state and is rejected by DuckPD's resource policy"
                )
            if func_lower == "size":
                op = AggregateOperator.SIZE
                target_expr = None
                input_type = None
                out_type = "BIGINT"
            else:
                target_col = find_column(
                    self._frame._plan.metadata, target_col_name, include_hidden=True
                )
                if func_lower == "count":
                    op = AggregateOperator.COUNT
                    target_expr = ColumnRef(target_col.id)
                    input_type = target_col.duckdb_type
                    out_type = "BIGINT"
                elif func_lower in {
                    "sum",
                    "mean",
                    "min",
                    "max",
                    "std",
                    "var",
                    "median",
                }:
                    op_map = {
                        "sum": AggregateOperator.SUM,
                        "mean": AggregateOperator.MEAN,
                        "min": AggregateOperator.MIN,
                        "max": AggregateOperator.MAX,
                        "std": AggregateOperator.STD,
                        "var": AggregateOperator.VAR,
                        "median": AggregateOperator.MEDIAN,
                    }
                    op = op_map[func_lower]
                    if not is_numeric_type(target_col.duckdb_type):
                        raise UnsupportedOperationError(
                            f"{func_lower} requires numeric or boolean column; "
                            f"{target_col_name!r} is {target_col.duckdb_type}"
                        )
                    target_expr = ColumnRef(target_col.id)
                    input_type = target_col.duckdb_type
                    if func_lower in {"mean", "std", "var", "median"}:
                        out_type = "DOUBLE"
                    elif func_lower == "sum" and target_col.duckdb_type == "BOOLEAN":
                        out_type = "BIGINT"
                    else:
                        out_type = target_col.duckdb_type
                else:
                    raise UnsupportedOperationError(
                        f"Unsupported aggregate function: {func_name!r}"
                    )

            out_col = Column(ColumnId.create(), output_name, out_type, hidden=False)
            output_columns.append(out_col)
            aggregates.append(
                AggregateExpression(
                    out_col,
                    operator=op,
                    expression=target_expr,
                    input_duckdb_type=input_type,
                )
            )

        # Build metadata
        index_ids = tuple(
            col.id for col in output_columns[: len(self._key_columns)] if col.hidden
        )
        ordering_keys = (
            tuple(
                OrderColumn(col.id, SortDirection.ASCENDING, NullPlacement.LAST)
                for col in output_columns[: len(self._key_columns)]
            )
            if self._sort
            else ()
        )
        if not self._sort:
            identity_id = next(
                iter(self._frame._plan.metadata.row_identity.columns),
                None,
            )
            row_identity = next(
                (
                    column
                    for column in self._frame._plan.metadata.columns
                    if column.id == identity_id
                ),
                None,
            )
            if row_identity is not None:
                first_seen = Column(
                    ColumnId.create(),
                    "__duckpd_group_first_seen__",
                    row_identity.duckdb_type,
                    hidden=True,
                )
                output_columns.append(first_seen)
                aggregates.append(
                    AggregateExpression(
                        first_seen,
                        operator=AggregateOperator.MIN,
                        expression=ColumnRef(row_identity.id),
                        input_duckdb_type=row_identity.duckdb_type,
                    )
                )
                ordering_keys = (
                    OrderColumn(
                        first_seen.id, SortDirection.ASCENDING, NullPlacement.LAST
                    ),
                )

        metadata = after_aggregate(
            tuple(output_columns),
            index_ids=index_ids,
            ordering_keys=ordering_keys,
        )

        plan = AggregatePlan(
            self._frame._plan,
            tuple(aggregates),
            metadata,
            keys=key_ids,
            dropna=self._dropna,
            sort=self._sort,
        )
        return DataFrame(self._session, plan)


class SeriesGroupBy:
    """A lazy GroupBy wrapper over a single Series."""

    def __init__(
        self,
        series: Series,
        by: object,
        *,
        as_index: bool = True,
        sort: bool = True,
        dropna: bool = True,
        observed: bool = True,
    ) -> None:
        from duckpd.frame import DataFrame
        from duckpd.series import Series

        self._series = series
        self._session: Session = series._session
        self._as_index = as_index
        self._sort = sort
        self._dropna = dropna
        if not observed:
            raise UnsupportedOperationError(
                "DuckPD does not support unobserved categorical groups"
            )

        # If by is string or sequence of strings, series must have that column
        # in its underlying plan
        if isinstance(by, str):
            by_labels = (by,)
            self._frame = DataFrame(series._session, series._plan)
            self._df_groupby = DataFrameGroupBy(
                self._frame,
                by=by_labels,
                as_index=as_index,
                sort=sort,
                dropna=dropna,
            )
        elif isinstance(by, Series):
            if by._session is not series._session or by._plan is not series._plan:
                raise AlignmentError(
                    "Grouping Series from a different frame "
                    "requires explicit index alignment"
                )
            by_name = by.name if by.name is not None else "__duckpd_grp__"
            self._frame = DataFrame(series._session, series._plan)
            if by.name is None:
                self._frame = self._frame.assign(__duckpd_grp__=by)
            self._df_groupby = DataFrameGroupBy(
                self._frame,
                by=(by_name,),
                as_index=as_index,
                sort=sort,
                dropna=dropna,
            )
        elif isinstance(by, Sequence):
            by_seq = tuple(cast("Sequence[object]", by))
            if not by_seq:
                raise ValueError("No group keys were provided")
            if all(isinstance(item, str) for item in by_seq):
                str_labels = cast("tuple[str, ...]", by_seq)
                self._frame = DataFrame(series._session, series._plan)
                self._df_groupby = DataFrameGroupBy(
                    self._frame,
                    by=str_labels,
                    as_index=as_index,
                    sort=sort,
                    dropna=dropna,
                )
            elif all(isinstance(item, Series) for item in by_seq):
                ser_seq = cast("tuple[Series, ...]", by_seq)
                for s in ser_seq:
                    if s._session is not series._session or s._plan is not series._plan:
                        raise AlignmentError(
                            "Grouping Series from a different frame "
                            "requires explicit index alignment"
                        )
                ser_names: list[str] = []
                self._frame = DataFrame(series._session, series._plan)
                for i, s in enumerate(ser_seq):
                    name = s.name if s.name is not None else f"__duckpd_grp_{i}__"
                    if s.name is None:
                        self._frame = self._frame.assign(**{name: s})
                    ser_names.append(name)
                self._df_groupby = DataFrameGroupBy(
                    self._frame,
                    by=ser_names,
                    as_index=as_index,
                    sort=sort,
                    dropna=dropna,
                )
            else:
                raise TypeError("Group keys must be all strings or all Series")
        else:
            raise TypeError(
                "by must be a string, Series, or sequence of strings/Series"
            )

    def agg(self, func: object) -> DataFrame:
        """Aggregate the grouped Series using the specified function."""
        if not isinstance(func, str):
            raise UnsupportedOperationError(
                "SeriesGroupBy.agg currently supports a function name string"
            )
        target_name = (
            self._series.name if self._series.name is not None else "__duckpd_val__"
        )
        if self._series.name is None:
            self._df_groupby._frame = self._df_groupby._frame.assign(
                __duckpd_val__=self._series
            )
        return self._df_groupby._named_agg(**{target_name: (target_name, func)})

    def sum(self) -> DataFrame:
        """Compute sum of values for each group."""
        return self.agg("sum")

    def mean(self) -> DataFrame:
        """Compute mean of values for each group."""
        return self.agg("mean")

    def min(self) -> DataFrame:
        """Compute minimum of values for each group."""
        return self.agg("min")

    def max(self) -> DataFrame:
        """Compute maximum of values for each group."""
        return self.agg("max")

    def count(self) -> DataFrame:
        """Compute count of non-null values for each group."""
        return self.agg("count")

    def size(self) -> DataFrame:
        """Compute group sizes."""
        return self._df_groupby.size()

    def std(self) -> DataFrame:
        """Compute sample standard deviation for each group."""
        return self.agg("std")

    def var(self) -> DataFrame:
        """Compute sample variance for each group."""
        return self.agg("var")

    def median(self) -> DataFrame:
        """Compute median for each group."""
        return self.agg("median")
