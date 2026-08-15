"""Rolling and expanding window objects for lazy evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from duckpd._logical import (
    BinaryExpression,
    BinaryOperator,
    CaseWhen,
    CastExpression,
    Column,
    ColumnId,
    ColumnRef,
    Expression,
    FunctionCall,
    LiteralValue,
    NamedExpression,
    ProjectPlan,
    SortKey,
    WindowExpression,
)
from duckpd._metadata import after_projection
from duckpd._reductions import expression_type, is_numeric_type
from duckpd.errors import UnorderedOperationError, UnsupportedOperationError

if TYPE_CHECKING:
    from duckpd.frame import DataFrame
    from duckpd.series import Series


class WindowBase:
    """Base class for rolling and expanding window calculations."""

    def __init__(
        self,
        parent: DataFrame | Series,
        min_periods: int,
        frame_spec: str,
    ) -> None:
        self._parent = parent
        self._min_periods = min_periods
        self._frame_spec = frame_spec

    def _require_order(self) -> tuple[SortKey, ...]:
        ordering = self._parent._plan.metadata.ordering
        if not ordering.keys:
            raise UnorderedOperationError(
                "Window operations require a guaranteed row order"
            )
        return tuple(
            SortKey(
                ColumnRef(k.column_id),
                k.direction,
                k.null_placement,
            )
            for k in ordering.keys
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
                order_by=order_keys,
                frame_spec=self._frame_spec,
            )
            row_num = WindowExpression(
                function="row_number",
                order_by=order_keys,
            )
            cnt_expr: Expression
            if self._min_periods > 1:
                cnt_expr = CaseWhen(
                    BinaryExpression(
                        row_num,
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
            order_by=order_keys,
            frame_spec=self._frame_spec,
        )
        if func == "sum" and self._min_periods == 0:
            window_val = FunctionCall("coalesce", (window_val, LiteralValue(0)))
        window_cnt = WindowExpression(
            function="count",
            arguments=(s._expression,),
            order_by=order_keys,
            frame_spec=self._frame_spec,
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
        window: int,
        min_periods: int | None = None,
        *,
        center: bool = False,
    ) -> None:
        if type(window) is not int:
            raise ValueError("window must be an integer")
        if window <= 0:
            raise ValueError("window must be a positive integer")
        if center:
            raise UnsupportedOperationError(
                "DuckPD does not support center=True in rolling"
            )
        if min_periods is not None and (type(min_periods) is not int):
            raise ValueError("min_periods must be an integer")
        min_p = window if min_periods is None else min_periods
        if min_p < 0:
            raise ValueError("min_periods must be non-negative")
        if min_p > window:
            raise ValueError("min_periods must not exceed window")
        frame_spec = f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        super().__init__(parent, min_p, frame_spec)
        self._window = window

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
        return self._apply_frame_agg(
            self._parent, "std", numeric_only=numeric_only, ddof=ddof
        )

    def var(self, ddof: int = 1, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "var", ddof=ddof)
        return self._apply_frame_agg(
            self._parent, "var", numeric_only=numeric_only, ddof=ddof
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
        frame_spec = "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        super().__init__(parent, min_p, frame_spec)

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
        return self._apply_frame_agg(
            self._parent, "std", numeric_only=numeric_only, ddof=ddof
        )

    def var(self, ddof: int = 1, numeric_only: bool = False) -> DataFrame | Series:
        from duckpd.series import Series

        if isinstance(self._parent, Series):
            return self._apply_series_agg(self._parent, "var", ddof=ddof)
        return self._apply_frame_agg(
            self._parent, "var", numeric_only=numeric_only, ddof=ddof
        )
