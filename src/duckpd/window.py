"""Rolling and expanding window objects for lazy evaluation."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING, Literal, TypeAlias

import pandas as pd

from duckpd._logical import (
    BinaryExpression,
    BinaryOperator,
    CaseWhen,
    CastExpression,
    Column,
    ColumnId,
    ColumnRef,
    Expression,
    FilterPlan,
    FunctionCall,
    IndexSpec,
    IndexUniqueness,
    LiteralValue,
    NamedExpression,
    NullPlacement,
    OrderSpec,
    ProjectPlan,
    RowIdentity,
    SortDirection,
    SortKey,
    SortPlan,
    WindowClosed,
    WindowExpression,
    WindowFrame,
    WindowFrameKind,
)
from duckpd._metadata import (
    after_filter,
    after_projection,
    after_sort,
    validate_metadata,
)
from duckpd._reductions import expression_type, is_numeric_type
from duckpd.errors import UnorderedOperationError, UnsupportedOperationError

if TYPE_CHECKING:
    from duckpd._logical import LogicalPlan
    from duckpd.frame import DataFrame
    from duckpd.series import Series


Closed: TypeAlias = Literal["right", "left", "both", "neither"]
RollingWindow: TypeAlias = int | str | timedelta


def _is_timestamp_type(duckdb_type: str) -> bool:
    normalized = duckdb_type.upper()
    return normalized == "UNKNOWN" or normalized.startswith("TIMESTAMP")


class WindowBase:
    """Base class for rolling and expanding window calculations."""

    def __init__(
        self,
        parent: DataFrame | Series,
        min_periods: int,
        frame: WindowFrame,
        *,
        partition_by: tuple[Expression, ...] = (),
        order_by: tuple[SortKey, ...] | None = None,
        on_column: Column | None = None,
    ) -> None:
        self._parent = parent
        self._min_periods = min_periods
        self._frame = frame
        self._partition_by = partition_by
        self._range_order_by = order_by
        self._on_column = on_column

    def _require_order(self) -> tuple[SortKey, ...]:
        if self._range_order_by is not None:
            return self._range_order_by
        ordering = self._parent._plan.metadata.ordering
        if not ordering.keys:
            raise UnorderedOperationError(
                "Window operations require a guaranteed row ordering. Specify "
                "order_by when creating a SQL/table source or sort first using "
                ".sort_values(...)"
            )
        return tuple(
            SortKey(
                ColumnRef(key.column_id),
                key.direction,
                key.null_placement,
            )
            for key in ordering.keys
        )

    def _apply_series_agg(
        self,
        s: Series,
        func: Literal["count", "sum", "mean", "min", "max", "std", "var"],
        ddof: int = 1,
    ) -> Series:
        from duckpd.series import Series as SeriesClass

        order_keys = self._require_order()
        in_type = expression_type(s._plan, s._expression)

        if func == "count":
            window_cnt = WindowExpression(
                function="count",
                arguments=(s._expression,),
                partition_by=self._partition_by,
                order_by=order_keys,
                frame=self._frame,
            )
            observation_count = WindowExpression(
                function="count",
                arguments=(LiteralValue(1),),
                partition_by=self._partition_by,
                order_by=order_keys,
                frame=self._frame,
            )
            cnt_expr: Expression
            if self._min_periods > 1:
                cnt_expr = CaseWhen(
                    BinaryExpression(
                        observation_count,
                        BinaryOperator.LESS_THAN,
                        LiteralValue(self._min_periods),
                    ),
                    LiteralValue(None),
                    CastExpression(window_cnt, "DOUBLE"),
                )
            else:
                cnt_expr = CastExpression(window_cnt, "DOUBLE")
            return SeriesClass(s._session, s._plan, cnt_expr, s.name)

        duck_func: str
        if func == "mean":
            duck_func = "avg"
        elif func == "std":
            duck_func = "stddev_samp" if ddof == 1 else "stddev_pop"
        elif func == "var":
            duck_func = "var_samp" if ddof == 1 else "var_pop"
        else:
            duck_func = func

        op = (
            CastExpression(s._expression, "BIGINT")
            if in_type == "BOOLEAN" and func in {"sum", "mean"}
            else s._expression
        )
        if func in {"std", "var"} and in_type in {
            "BOOLEAN",
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
        }:
            op = CastExpression(s._expression, "DOUBLE")

        window_val = WindowExpression(
            function=duck_func,
            arguments=(op,),
            partition_by=self._partition_by,
            order_by=order_keys,
            frame=self._frame,
        )
        if func == "sum" and self._min_periods == 0:
            window_val = FunctionCall("coalesce", (window_val, LiteralValue(0)))
        window_cnt = WindowExpression(
            function="count",
            arguments=(s._expression,),
            partition_by=self._partition_by,
            order_by=order_keys,
            frame=self._frame,
        )

        min_req = self._min_periods
        if func in {"std", "var"}:
            min_req = max(min_req, 2 if ddof == 1 else 1)

        result_expr: Expression = CaseWhen(
            BinaryExpression(
                window_cnt,
                BinaryOperator.LESS_THAN,
                LiteralValue(min_req),
            ),
            LiteralValue(None),
            CastExpression(window_val, "DOUBLE"),
        )
        return SeriesClass(s._session, s._plan, result_expr, s.name)

    def _apply_frame_agg(
        self,
        df: DataFrame,
        func: Literal["count", "sum", "mean", "min", "max", "std", "var"],
        numeric_only: bool = False,
        ddof: int = 1,
    ) -> DataFrame:
        from duckpd.frame import DataFrame as DataFrameClass
        from duckpd.series import Series as SeriesClass

        visible = df._plan.metadata.visible_columns
        protected = [col for col in df._plan.metadata.columns if col.hidden]
        new_projections: list[NamedExpression] = []

        for col in visible:
            if self._on_column is not None and col.id == self._on_column.id:
                new_projections.append(NamedExpression(col, ColumnRef(col.id)))
                continue
            is_num = is_numeric_type(col.duckdb_type)
            if numeric_only and not is_num:
                continue
            s = SeriesClass(df._session, df._plan, ColumnRef(col.id), col.label)
            res_s = self._apply_series_agg(s, func, ddof=ddof)
            out_col = Column(
                ColumnId.create(),
                col.label,
                expression_type(df._plan, res_s._expression),
            )
            new_projections.append(NamedExpression(out_col, res_s._expression))

        for col in protected:
            new_projections.append(NamedExpression(col, ColumnRef(col.id)))

        output_columns = tuple(p.column for p in new_projections)
        metadata = after_projection(df._plan.metadata, output_columns)
        return DataFrameClass(
            df._session,
            ProjectPlan(df._plan, tuple(new_projections), metadata),
        )


class Rolling(WindowBase):
    """A rolling window calculation object."""

    def __init__(
        self,
        parent: DataFrame | Series,
        window: RollingWindow,
        min_periods: int | None = None,
        *,
        center: bool = False,
        on: str | None = None,
        closed: Closed | None = None,
        _partition_by: tuple[Expression, ...] = (),
        _key_columns: tuple[Column, ...] = (),
    ) -> None:
        if center:
            raise UnsupportedOperationError("DuckPD does not support center=True in rolling")
        if min_periods is not None and type(min_periods) is not int:
            raise ValueError("min_periods must be an integer")

        if type(window) is int:
            if on is not None:
                raise UnsupportedOperationError(
                    "DuckPD supports on= only for fixed-duration rolling windows"
                )
            if closed is not None:
                raise UnsupportedOperationError(
                    "DuckPD supports closed= only for fixed-duration rolling windows"
                )
            if window <= 0:
                raise ValueError("window must be a positive integer")
            min_p = window if min_periods is None else min_periods
            if min_p < 0:
                raise ValueError("min_periods must be non-negative")
            if min_p > window:
                raise ValueError("min_periods must not exceed window")
            frame = WindowFrame(WindowFrameKind.ROWS, window - 1)
            super().__init__(
                parent,
                min_p,
                frame,
                partition_by=_partition_by,
            )
            self._window = window
            return

        if not isinstance(window, (str, timedelta)):
            raise ValueError("window must be an integer or fixed duration")
        if closed not in {None, "right", "left", "both", "neither"}:
            raise ValueError("closed must be 'right', 'left', 'both' or 'neither'")
        try:
            duration = pd.Timedelta(window)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("window must be a positive fixed duration") from error
        if duration <= pd.Timedelta(0):
            raise ValueError("window must be a positive fixed duration")
        duration_ns = int(duration.value)
        min_p = 1 if min_periods is None else min_periods
        if min_p < 0:
            raise ValueError("min_periods must be non-negative")

        on_column = self._resolve_time_column(parent, on)
        range_order = self._range_order(parent, on_column, _key_columns)
        frame = WindowFrame(
            WindowFrameKind.RANGE,
            duration_ns,
            WindowClosed(closed or "right"),
        )
        super().__init__(
            parent,
            min_p,
            frame,
            partition_by=_partition_by,
            order_by=range_order,
            on_column=on_column,
        )
        self._window = window

    @staticmethod
    def _resolve_time_column(parent: DataFrame | Series, on: str | None) -> Column:
        from duckpd.frame import DataFrame

        metadata = parent._plan.metadata
        if isinstance(parent, DataFrame):
            if not isinstance(on, str) or not on:
                raise ValueError("DataFrame time-based rolling requires on= timestamp column")
            column = next(
                (candidate for candidate in metadata.visible_columns if candidate.label == on),
                None,
            )
            if column is None:
                raise KeyError(on)
        else:
            if on is not None:
                raise TypeError("Series.rolling() does not accept on=")
            if len(metadata.index.columns) != 1:
                raise ValueError("Series time-based rolling requires one explicit datetime index")
            index_id = metadata.index.columns[0]
            column = next(candidate for candidate in metadata.columns if candidate.id == index_id)
        if not _is_timestamp_type(column.duckdb_type):
            raise ValueError(
                f"Time-based rolling requires a timestamp column; "
                f"{column.label!r} is {column.duckdb_type}"
            )
        return column

    @staticmethod
    def _range_order(
        parent: DataFrame | Series,
        on_column: Column,
        key_columns: tuple[Column, ...],
    ) -> tuple[SortKey, ...]:
        key_ids = {column.id for column in key_columns}
        relevant = tuple(
            key for key in parent._plan.metadata.ordering.keys if key.column_id not in key_ids
        )
        if (
            not relevant
            or relevant[0].column_id != on_column.id
            or relevant[0].direction is not SortDirection.ASCENDING
        ):
            raise UnorderedOperationError(
                "Time-based rolling requires an ascending order_by beginning with "
                f"{on_column.label!r} after group keys"
            )
        return (
            SortKey(
                ColumnRef(on_column.id),
                SortDirection.ASCENDING,
                relevant[0].null_placement,
            ),
        )

    def count(self, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "count")
        return self._apply_frame_agg(self._parent, "count", numeric_only=numeric_only)

    def sum(self, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "sum")
        return self._apply_frame_agg(self._parent, "sum", numeric_only=numeric_only)

    def mean(self, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "mean")
        return self._apply_frame_agg(self._parent, "mean", numeric_only=numeric_only)

    def min(self, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "min")
        return self._apply_frame_agg(self._parent, "min", numeric_only=numeric_only)

    def max(self, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "max")
        return self._apply_frame_agg(self._parent, "max", numeric_only=numeric_only)

    def std(self, ddof: int = 1, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "std", ddof=ddof)
        return self._apply_frame_agg(self._parent, "std", numeric_only=numeric_only, ddof=ddof)

    def var(self, ddof: int = 1, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "var", ddof=ddof)
        return self._apply_frame_agg(self._parent, "var", numeric_only=numeric_only, ddof=ddof)


class GroupedRolling(Rolling):
    """Rolling windows partitioned by lazy GroupBy keys."""

    def __init__(
        self,
        parent: DataFrame | Series,
        key_columns: tuple[Column, ...],
        window: RollingWindow,
        min_periods: int | None = None,
        *,
        center: bool = False,
        on: str | None = None,
        closed: Closed | None = None,
        as_index: bool = True,
        sort: bool = True,
        dropna: bool = True,
        alignment_source: LogicalPlan | None = None,
    ) -> None:
        self._key_columns = key_columns
        self._as_index = as_index
        self._sort_groups = sort
        self._dropna = dropna
        self._alignment_source = alignment_source or parent._plan
        super().__init__(
            parent,
            window,
            min_periods,
            center=center,
            on=on,
            closed=closed,
            _partition_by=tuple(ColumnRef(column.id) for column in key_columns),
            _key_columns=key_columns,
        )

    def count(self, numeric_only: bool = False) -> DataFrame | Series:
        return self._calculate("count", numeric_only=numeric_only)

    def sum(self, numeric_only: bool = False) -> DataFrame | Series:
        return self._calculate("sum", numeric_only=numeric_only)

    def mean(self, numeric_only: bool = False) -> DataFrame | Series:
        return self._calculate("mean", numeric_only=numeric_only)

    def min(self, numeric_only: bool = False) -> DataFrame | Series:
        return self._calculate("min", numeric_only=numeric_only)

    def max(self, numeric_only: bool = False) -> DataFrame | Series:
        return self._calculate("max", numeric_only=numeric_only)

    def std(self, ddof: int = 1, numeric_only: bool = False) -> DataFrame | Series:
        return self._calculate("std", numeric_only=numeric_only, ddof=ddof)

    def var(self, ddof: int = 1, numeric_only: bool = False) -> DataFrame | Series:
        return self._calculate("var", numeric_only=numeric_only, ddof=ddof)

    def _calculate(
        self,
        func: Literal["count", "sum", "mean", "min", "max", "std", "var"],
        *,
        numeric_only: bool,
        ddof: int = 1,
    ) -> DataFrame | Series:
        from duckpd.frame import DataFrame
        from duckpd.series import Series

        source_plan = self._parent._plan
        order_keys = self._require_order()
        key_ids = {column.id for column in self._key_columns}
        calculations: list[tuple[str, Expression, str]] = []

        if isinstance(self._parent, Series):
            result = self._apply_series_agg(self._parent, func, ddof=ddof)
            calculations.append(
                (
                    self._parent.name or "0",
                    result._expression,
                    expression_type(source_plan, result._expression),
                )
            )
        else:
            for column in source_plan.metadata.visible_columns:
                if column.id in key_ids:
                    continue
                if self._on_column is not None and column.id == self._on_column.id:
                    calculations.append(
                        (
                            column.label,
                            ColumnRef(column.id),
                            column.duckdb_type,
                        )
                    )
                    continue
                if numeric_only and not is_numeric_type(column.duckdb_type):
                    continue
                series = Series(
                    self._parent._session,
                    source_plan,
                    ColumnRef(column.id),
                    column.label,
                )
                result = self._apply_series_agg(series, func, ddof=ddof)
                calculations.append(
                    (
                        column.label,
                        result._expression,
                        expression_type(source_plan, result._expression),
                    )
                )

        if not calculations:
            raise UnsupportedOperationError(f"No valid columns for grouped rolling {func}")

        result_source = source_plan
        source_index: list[tuple[Expression, str, str]] = []
        index_names: tuple[str | None, ...]
        source_metadata = source_plan.metadata
        if source_metadata.index.columns:
            index_names = source_metadata.index.names or tuple(
                next(column.label for column in source_metadata.columns if column.id == column_id)
                for column_id in source_metadata.index.columns
            )
            for column_id in source_metadata.index.columns:
                column = next(
                    column for column in source_metadata.columns if column.id == column_id
                )
                source_index.append((ColumnRef(column.id), column.label, column.duckdb_type))
        elif (
            source_metadata.row_identity.stable
            and source_metadata.row_identity.unique
            and len(source_metadata.row_identity.columns) == 1
        ):
            identity_id = source_metadata.row_identity.columns[0]
            identity_column = next(
                column for column in source_metadata.columns if column.id == identity_id
            )
            source_index.append(
                (
                    ColumnRef(identity_column.id),
                    "__duckpd_grouped_rolling_index__",
                    identity_column.duckdb_type,
                )
            )
            index_names = (None,)
        else:
            ordinal_column = Column(
                ColumnId.create(),
                "__duckpd_grouped_rolling_index__",
                "BIGINT",
                hidden=True,
            )
            ordinal_expression = BinaryExpression(
                WindowExpression(function="row_number", order_by=order_keys),
                BinaryOperator.SUBTRACT,
                LiteralValue(1),
            )
            base_projections = (
                *(
                    NamedExpression(column, ColumnRef(column.id))
                    for column in source_metadata.columns
                ),
                NamedExpression(ordinal_column, ordinal_expression),
            )
            base_columns = tuple(item.column for item in base_projections)
            base_metadata = after_projection(source_metadata, base_columns)
            result_source = ProjectPlan(
                source_plan,
                base_projections,
                base_metadata,
            )
            source_index.append(
                (
                    ColumnRef(ordinal_column.id),
                    ordinal_column.label,
                    ordinal_column.duckdb_type,
                )
            )
            index_names = (None,)

        valid_group: Expression | None = None
        for column in self._key_columns:
            present = FunctionCall("notnull", (ColumnRef(column.id),))
            valid_group = (
                present
                if valid_group is None
                else BinaryExpression(valid_group, BinaryOperator.AND, present)
            )
        if self._dropna:
            if valid_group is None:
                raise AssertionError("Grouped rolling requires at least one key")
            result_source = FilterPlan(
                result_source,
                valid_group,
                after_filter(result_source.metadata),
            )

        projections: list[NamedExpression] = []
        key_outputs: list[Column] = []
        for column in self._key_columns:
            output = replace(column, id=ColumnId.create(), hidden=self._as_index)
            key_outputs.append(output)
            projections.append(NamedExpression(output, ColumnRef(column.id)))

        index_outputs: list[Column] = []
        for expression, label, duckdb_type in source_index:
            output = Column(ColumnId.create(), label, duckdb_type, hidden=True)
            index_outputs.append(output)
            projections.append(NamedExpression(output, expression))

        data_outputs: list[Column] = []
        alignment_expressions: list[Expression] = []
        for label, expression, duckdb_type in calculations:
            output = Column(ColumnId.create(), label, duckdb_type)
            data_outputs.append(output)
            projections.append(NamedExpression(output, expression))
            alignment_expressions.append(
                CaseWhen(valid_group, expression, LiteralValue(None))
                if self._dropna and valid_group is not None
                else expression
            )

        group_order_outputs: list[Column] = []
        if not self._sort_groups:
            for position, key in enumerate(order_keys):
                expression = WindowExpression(
                    function="first_value",
                    arguments=(key.expression,),
                    partition_by=self._partition_by,
                    order_by=order_keys,
                )
                output = Column(
                    ColumnId.create(),
                    f"__duckpd_group_order_{position}__",
                    expression_type(source_plan, expression),
                    hidden=True,
                )
                group_order_outputs.append(output)
                projections.append(NamedExpression(output, expression))

        row_order_outputs: list[Column] = []
        for position, key in enumerate(order_keys):
            output = Column(
                ColumnId.create(),
                f"__duckpd_group_row_order_{position}__",
                expression_type(source_plan, key.expression),
                hidden=True,
            )
            row_order_outputs.append(output)
            projections.append(NamedExpression(output, key.expression))

        output_columns = tuple(item.column for item in projections)
        metadata = after_projection(result_source.metadata, output_columns)
        index_columns = (*key_outputs, *index_outputs) if self._as_index else tuple(index_outputs)
        grouped_index_names = (
            (*tuple(column.label for column in self._key_columns), *index_names)
            if self._as_index
            else index_names
        )
        metadata = replace(
            metadata,
            index=IndexSpec(
                tuple(column.id for column in index_columns),
                drop=True,
                uniqueness=IndexUniqueness.UNKNOWN,
                names=grouped_index_names,
            ),
            ordering=OrderSpec(),
            row_identity=RowIdentity(),
        )
        validate_metadata(metadata)
        projected = ProjectPlan(result_source, tuple(projections), metadata)

        sort_keys: list[SortKey] = []
        if self._sort_groups:
            sort_keys.extend(
                SortKey(
                    ColumnRef(column.id),
                    SortDirection.ASCENDING,
                    NullPlacement.LAST,
                )
                for column in key_outputs
            )
        else:
            sort_keys.extend(
                SortKey(
                    ColumnRef(output.id),
                    source_key.direction,
                    source_key.null_placement,
                )
                for output, source_key in zip(group_order_outputs, order_keys, strict=True)
            )
        sort_keys.extend(
            SortKey(
                ColumnRef(output.id),
                source_key.direction,
                source_key.null_placement,
            )
            for output, source_key in zip(row_order_outputs, order_keys, strict=True)
        )
        ordered_plan = SortPlan(
            projected,
            tuple(sort_keys),
            after_sort(metadata, tuple(sort_keys)),
        )

        if isinstance(self._parent, Series):
            return Series(
                self._parent._session,
                ordered_plan,
                ColumnRef(data_outputs[0].id),
                self._parent.name,
                alignment_source=self._alignment_source,
                alignment_expression=alignment_expressions[0],
            )
        return DataFrame(
            self._parent._session,
            ordered_plan,
            alignment_source=self._alignment_source,
            alignment_expressions=tuple(alignment_expressions),
        )


class Expanding(WindowBase):
    """An expanding window calculation object."""

    def __init__(
        self,
        parent: DataFrame | Series,
        min_periods: int = 1,
    ) -> None:
        if type(min_periods) is not int:
            raise ValueError("min_periods must be an integer")
        if min_periods < 0:
            raise ValueError("min_periods must be non-negative")
        min_p = min_periods
        frame = WindowFrame(WindowFrameKind.ROWS, None)
        super().__init__(parent, min_p, frame)

    def count(self, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "count")
        return self._apply_frame_agg(self._parent, "count", numeric_only=numeric_only)

    def sum(self, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "sum")
        return self._apply_frame_agg(self._parent, "sum", numeric_only=numeric_only)

    def mean(self, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "mean")
        return self._apply_frame_agg(self._parent, "mean", numeric_only=numeric_only)

    def min(self, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "min")
        return self._apply_frame_agg(self._parent, "min", numeric_only=numeric_only)

    def max(self, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "max")
        return self._apply_frame_agg(self._parent, "max", numeric_only=numeric_only)

    def std(self, ddof: int = 1, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "std", ddof=ddof)
        return self._apply_frame_agg(self._parent, "std", numeric_only=numeric_only, ddof=ddof)

    def var(self, ddof: int = 1, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "var", ddof=ddof)
        return self._apply_frame_agg(self._parent, "var", numeric_only=numeric_only, ddof=ddof)
