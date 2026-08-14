"""Lazy GroupBy public API and aggregation execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

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
from duckpd.errors import UnsupportedOperationError

if TYPE_CHECKING:
    from duckpd.frame import DataFrame
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
        observed: bool = False,
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
        if observed:
            pass

    def agg(
        self,
        func: object = None,
        *args: object,
        **kwargs: object,
    ) -> DataFrame:
        """Aggregate using one or more operations."""
        if func is not None:
            raise UnsupportedOperationError(
                "Positional aggregation func is not yet supported; "
                "use named aggregation kwargs"
            )

        if not kwargs:
            raise ValueError("Must provide at least one aggregation function")

        return self._named_agg(**kwargs)

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
            target_col = find_column(self._frame._plan.metadata, target_col_name)

            if func_lower == "size":
                op = AggregateOperator.SIZE
                target_expr = None
                input_type = None
                out_type = "BIGINT"
            elif func_lower == "count":
                op = AggregateOperator.COUNT
                target_expr = ColumnRef(target_col.id)
                input_type = target_col.duckdb_type
                out_type = "BIGINT"
            elif func_lower in {"sum", "mean", "min", "max"}:
                op = {
                    "sum": AggregateOperator.SUM,
                    "mean": AggregateOperator.MEAN,
                    "min": AggregateOperator.MIN,
                    "max": AggregateOperator.MAX,
                }[func_lower]
                if not is_numeric_type(target_col.duckdb_type):
                    raise UnsupportedOperationError(
                        f"{func_lower} requires numeric or boolean column; "
                        f"{target_col_name!r} is {target_col.duckdb_type}"
                    )
                target_expr = ColumnRef(target_col.id)
                input_type = target_col.duckdb_type
                out_type = "DOUBLE" if func_lower == "mean" else target_col.duckdb_type
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
        index_ids = tuple(col.id for col in output_columns if col.hidden)
        ordering_keys = (
            tuple(
                OrderColumn(col.id, SortDirection.ASCENDING, NullPlacement.LAST)
                for col in output_columns[: len(self._key_columns)]
            )
            if self._sort
            else ()
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
