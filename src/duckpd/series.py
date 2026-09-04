"""Lazy Series expression API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace as dataclass_replace
from decimal import Decimal
from math import isfinite
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd

from duckpd._logical import (
    AggregateExpression,
    AggregateOperator,
    AggregatePlan,
    BinaryExpression,
    BinaryOperator,
    CaseWhen,
    CastExpression,
    ColumnId,
    ColumnRef,
    FilterPlan,
    FunctionCall,
    IndexSpec,
    LiteralValue,
    NullPlacement,
    OrderColumn,
    OrderSpec,
    SamplePlan,
    SortDirection,
    SortKey,
    SortPlan,
    UnaryExpression,
    UnaryOperator,
    WindowExpression,
    expression_nullability,
)
from duckpd._metadata import after_aggregate, after_filter, after_sort
from duckpd._metadata import reset_index as reset_index_metadata
from duckpd._reductions import (
    aggregate_plan,
    expression_type,
    is_numeric_type,
    materialized_int,
    validate_axis,
    validate_ddof,
    validate_min_count,
    validate_quantile,
)
from duckpd._typing import ScalarValue, is_scalar_value, normalize_dtype
from duckpd.errors import (
    AlignmentError,
    UnorderedOperationError,
    UnsupportedOperationError,
)

if TYPE_CHECKING:
    from duckpd._logical import Expression, LogicalPlan
    from duckpd.accessors import DatetimeProperties, StringMethods
    from duckpd.frame import DataFrame
    from duckpd.groupby import SeriesGroupBy
    from duckpd.session import Session
    from duckpd.window import Expanding, Rolling


class Series:
    """A typed expression bound to one immutable frame plan."""

    def __init__(
        self,
        session: Session,
        plan: LogicalPlan,
        expression: Expression,
        name: str | None = None,
    ) -> None:
        self._session = session
        self._plan = plan
        self._expression = expression
        self.name = name

    def rename(
        self,
        index: object | None = None,
        *,
        axis: int | str | None = None,
        copy: bool = True,
        inplace: bool = False,
        level: object | None = None,
        errors: str = "raise",
    ) -> Series:
        """Alter Series name."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if not copy:
            raise UnsupportedOperationError("DuckPD does not support copy=False")
        if level is not None:
            raise UnsupportedOperationError("DuckPD does not support MultiIndex levels")
        if isinstance(index, dict):
            raise UnsupportedOperationError(
                "DuckPD does not yet support renaming index labels"
            )
        new_name = str(index) if index is not None else None
        return Series(self._session, self._plan, self._expression, new_name)

    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights: object = None,
        random_state: int | None = None,
        axis: int | str | None = None,
        ignore_index: bool = False,
    ) -> Series:
        """Return a random sample of items from the Series."""
        if axis not in (0, "index", None):
            raise UnsupportedOperationError(
                "DuckPD sample supports only axis=0 or axis='index'"
            )
        if replace is not False:
            raise UnsupportedOperationError(
                "DuckPD sample does not currently support replace=True"
            )
        if weights is not None:
            raise UnsupportedOperationError(
                "DuckPD sample does not currently support weights"
            )
        if n is not None and frac is not None:
            raise ValueError("Only one of 'n' or 'frac' can be specified")
        if n is None and frac is None:
            n = 1
        if n is not None:
            if isinstance(n, bool) or not isinstance(cast("object", n), int):
                raise ValueError(f"'n' must be an integer, got {type(n).__name__}")
            if n < 0:
                raise ValueError("A negative number of rows was requested")
        if frac is not None:
            if isinstance(frac, bool) or not isinstance(
                cast("object", frac), (int, float)
            ):
                raise ValueError(f"'frac' must be a float, got {type(frac).__name__}")
            if not isfinite(float(frac)):
                raise ValueError("'frac' must be finite")
            if frac < 0.0:
                raise ValueError("A negative fraction of rows was requested")
            if frac > 1.0:
                raise ValueError("Replace has to be set to True when frac > 1")
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(cast("object", random_state), int)
        ):
            raise ValueError("random_state must be an integer seed or None")
        if random_state is not None and not 0 <= random_state <= 2_147_483_647:
            raise ValueError("random_state must be between 0 and 2**31 - 1")

        metadata = dataclass_replace(self._plan.metadata, ordering=OrderSpec())
        if ignore_index:
            if metadata.index.columns:
                metadata = reset_index_metadata(metadata, drop=True)
            else:
                metadata = dataclass_replace(
                    metadata,
                    index=IndexSpec(),
                    ordering=OrderSpec(),
                )

        plan = SamplePlan(
            input=self._plan,
            n=n,
            frac=float(frac) if frac is not None else None,
            seed=random_state,
            metadata=metadata,
        )
        return Series(self._session, plan, self._expression, self.name)

    def to_frame(self, name: str | None = None) -> DataFrame:
        """Convert Series to a DataFrame."""
        from duckpd._logical import Column, ColumnId, NamedExpression, ProjectPlan
        from duckpd._metadata import after_projection, projection_columns
        from duckpd.frame import DataFrame

        out_label = name if name is not None else (self.name or "0")
        out_col = Column(
            ColumnId.create(),
            out_label,
            expression_type(self._plan, self._expression),
            nullable=expression_nullability(self._expression, self._plan.metadata),
            alias_of=(
                self._expression.column_id
                if isinstance(self._expression, ColumnRef)
                else None
            ),
        )
        all_cols = projection_columns(self._plan.metadata, (out_col,))
        projections = [
            (
                NamedExpression(col, self._expression)
                if col.id == out_col.id
                else NamedExpression(col, ColumnRef(col.id))
            )
            for col in all_cols
        ]
        metadata = after_projection(self._plan.metadata, all_cols)
        return DataFrame(
            self._session, ProjectPlan(self._plan, tuple(projections), metadata)
        )

    def collect(self) -> pd.Series:
        """Execute the Series plan and return a pandas Series."""
        df = self.to_frame()
        pdf = df.collect()
        col_label = df.columns[0]
        s = pdf[col_label]
        s.name = self.name
        return s

    def __add__(self, other: object) -> Series:
        return self._binary(other, BinaryOperator.ADD)

    def __radd__(self, other: object) -> Series:
        return self._rbinary(other, BinaryOperator.ADD)

    def __sub__(self, other: object) -> Series:
        return self._binary(other, BinaryOperator.SUBTRACT)

    def __rsub__(self, other: object) -> Series:
        return self._rbinary(other, BinaryOperator.SUBTRACT)

    def __mul__(self, other: object) -> Series:
        return self._binary(other, BinaryOperator.MULTIPLY)

    def __rmul__(self, other: object) -> Series:
        return self._rbinary(other, BinaryOperator.MULTIPLY)

    def __truediv__(self, other: object) -> Series:
        return self._binary(other, BinaryOperator.TRUE_DIVIDE)

    def __rtruediv__(self, other: object) -> Series:
        return self._rbinary(other, BinaryOperator.TRUE_DIVIDE)

    def __mod__(self, other: object) -> Series:
        return self._binary(other, BinaryOperator.MODULO)

    def __rmod__(self, other: object) -> Series:
        return self._rbinary(other, BinaryOperator.MODULO)

    def __eq__(self, other: object) -> Series:  # type: ignore[override]
        return self._binary(other, BinaryOperator.EQUAL)

    def __ne__(self, other: object) -> Series:  # type: ignore[override]
        return self._binary(other, BinaryOperator.NOT_EQUAL)

    def __lt__(self, other: object) -> Series:
        return self._binary(other, BinaryOperator.LESS_THAN)

    def __le__(self, other: object) -> Series:
        return self._binary(other, BinaryOperator.LESS_EQUAL)

    def __gt__(self, other: object) -> Series:
        return self._binary(other, BinaryOperator.GREATER_THAN)

    def __ge__(self, other: object) -> Series:
        return self._binary(other, BinaryOperator.GREATER_EQUAL)

    def __and__(self, other: object) -> Series:
        return self._binary(other, BinaryOperator.AND)

    def __or__(self, other: object) -> Series:
        return self._binary(other, BinaryOperator.OR)

    def __invert__(self) -> Series:
        return self._unary(UnaryOperator.INVERT)

    def __neg__(self) -> Series:
        return self._unary(UnaryOperator.NEGATE)

    def __pos__(self) -> Series:
        return self._unary(UnaryOperator.POSITIVE)

    def __bool__(self) -> bool:
        raise ValueError("The truth value of a DuckPD Series is ambiguous")

    def isna(self) -> Series:
        """Detect missing values."""
        return self._call_function("isnull")

    def notna(self) -> Series:
        """Detect existing (non-missing) values."""
        return self._call_function("notnull")

    def isnull(self) -> Series:
        """Alias for :meth:`isna`."""
        return self.isna()

    def notnull(self) -> Series:
        """Alias for :meth:`notna`."""
        return self.notna()

    def astype(
        self,
        dtype: object,
        *,
        copy: bool = True,
        errors: Literal["raise", "ignore"] = "raise",
    ) -> Series:
        """Cast Series values to a specified dtype."""
        if not copy:
            raise UnsupportedOperationError("DuckPD does not support copy=False")
        if errors not in {"raise", "ignore"}:
            raise ValueError("errors must be 'raise' or 'ignore'")
        try:
            target_type = normalize_dtype(dtype)
        except (TypeError, ValueError):
            if errors == "ignore":
                return self
            raise
        return Series(
            self._session,
            self._plan,
            CastExpression(self._expression, target_type),
            self.name,
        )

    def fillna(
        self,
        value: object = None,
        *,
        inplace: bool = False,
        limit: int | None = None,
    ) -> Series:
        """Fill NA/NaN values using the specified value."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if limit is not None:
            raise UnsupportedOperationError(
                "DuckPD does not support limit in fillna without explicit windows"
            )
        if value is None:
            raise ValueError("Must specify a value to fill NA/NaN values with")
        fill_expr = self._coerce_other(value)
        return self._call_function("coalesce", fill_expr)

    def dropna(
        self,
        *,
        axis: int | str = 0,
        inplace: bool = False,
        how: str | None = None,
        ignore_index: bool = False,
    ) -> Series:
        """Return a Series with missing values removed."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if ignore_index:
            raise UnsupportedOperationError(
                "DuckPD does not support ignore_index=True in Series.dropna"
            )
        if axis not in {0, "index"}:
            raise ValueError("Series.dropna supports only axis=0 or axis='index'")

        not_null_pred = FunctionCall("notnull", (self._expression,))
        filtered_plan = FilterPlan(
            self._plan, not_null_pred, after_filter(self._plan.metadata)
        )
        return Series(self._session, filtered_plan, self._expression, self.name)

    def where(
        self,
        cond: object,
        other: object = None,
        *,
        inplace: bool = False,
    ) -> Series:
        """Replace values where the condition is False."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        cond_expr = self._coerce_cond(cond)
        other_expr = self._coerce_other(other)
        return Series(
            self._session,
            self._plan,
            CaseWhen(cond_expr, self._expression, other_expr),
            self.name,
        )

    def mask(
        self,
        cond: object,
        other: object = None,
        *,
        inplace: bool = False,
    ) -> Series:
        """Replace values where the condition is True."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        cond_expr = self._coerce_cond(cond)
        other_expr = self._coerce_other(other)
        return Series(
            self._session,
            self._plan,
            CaseWhen(cond_expr, other_expr, self._expression),
            self.name,
        )

    def clip(
        self,
        lower: object = None,
        upper: object = None,
        *,
        axis: int | str | None = None,
        inplace: bool = False,
    ) -> Series:
        """Trim values at input thresholds."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if axis not in {0, "index", None}:
            raise UnsupportedOperationError(
                "DuckPD clip currently supports only axis=0 or axis='index'"
            )
        if lower is None and upper is None:
            return self

        if (
            isinstance(lower, (int, float))
            and isinstance(upper, (int, float))
            and lower > upper
        ):
            raise ValueError("Cannot set lower > upper")

        lower_expr: Expression | None = None
        if isinstance(lower, Series):
            if lower._session is not self._session or lower._plan is not self._plan:
                raise AlignmentError("Cannot align Series from different frames")
            lower_expr = lower._expression
        elif lower is not None:
            lower_expr = LiteralValue(cast("ScalarValue", lower))

        upper_expr: Expression | None = None
        if isinstance(upper, Series):
            if upper._session is not self._session or upper._plan is not self._plan:
                raise AlignmentError("Cannot align Series from different frames")
            upper_expr = upper._expression
        elif upper is not None:
            upper_expr = LiteralValue(cast("ScalarValue", upper))
        expr = self._expression
        cond_lower = (
            BinaryExpression(expr, BinaryOperator.LESS_THAN, lower_expr)
            if lower_expr is not None
            else None
        )
        cond_upper = (
            BinaryExpression(expr, BinaryOperator.GREATER_THAN, upper_expr)
            if upper_expr is not None
            else None
        )

        if cond_lower is not None and cond_upper is not None:
            clipped: Expression = CaseWhen(
                cond_lower,
                lower_expr,  # type: ignore[arg-type]
                CaseWhen(cond_upper, upper_expr, expr),  # type: ignore[arg-type]
            )
        elif cond_lower is not None:
            clipped = CaseWhen(cond_lower, lower_expr, expr)  # type: ignore[arg-type]
        elif cond_upper is not None:
            clipped = CaseWhen(cond_upper, upper_expr, expr)  # type: ignore[arg-type]
        else:
            clipped = expr

        return Series(self._session, self._plan, clipped, self.name)

    def replace(
        self,
        to_replace: object = None,
        value: object = None,
        *,
        inplace: bool = False,
        limit: int | None = None,
        regex: bool = False,
        method: str | None = None,
    ) -> Series:
        """Replace values given in to_replace with value."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if regex:
            raise UnsupportedOperationError(
                "DuckPD replace does not yet support regex=True"
            )
        if limit is not None or method is not None:
            raise UnsupportedOperationError(
                "DuckPD replace does not support limit or method parameters"
            )
        if to_replace is None and value is None:
            return self

        pairs: list[tuple[object, object]] = []
        if isinstance(to_replace, dict):
            pairs = list(cast("dict[object, object]", to_replace).items())
        elif isinstance(to_replace, (list, tuple)):
            replace_list = cast("Sequence[object]", to_replace)
            if isinstance(value, (list, tuple)):
                value_list = cast("Sequence[object]", value)
                if len(replace_list) != len(value_list):
                    raise ValueError(
                        "Replacement list lengths must match: "
                        f"len(to_replace)={len(replace_list)} vs "
                        f"len(value)={len(value_list)}"
                    )
                pairs = list(zip(replace_list, value_list, strict=True))
            else:
                pairs = [(old_val, value) for old_val in replace_list]
        else:
            pairs = [(to_replace, value)]

        def _is_replace_compatible(val: object, duckdb_type: str) -> bool:
            if val is None or val is pd.NA:
                return True
            if is_numeric_type(duckdb_type):
                return isinstance(val, (int, float, Decimal)) and not isinstance(
                    val, bool
                )
            if duckdb_type in {"VARCHAR", "TEXT"}:
                return isinstance(val, str)
            if duckdb_type == "BOOLEAN":
                return isinstance(val, bool)
            return True

        target_type = expression_type(self._plan, self._expression)
        applicable_pairs = [
            p for p in pairs if _is_replace_compatible(p[0], target_type)
        ]

        expr = self._expression
        cur_expr: Expression = expr
        for old_v, new_v in reversed(applicable_pairs):
            is_null_val = (
                old_v is None
                or old_v is pd.NA
                or (isinstance(old_v, float) and pd.isna(old_v))
            )
            if is_null_val:
                cond: Expression = FunctionCall("isnull", (expr,))
            else:
                old_scalar = cast("ScalarValue", old_v)
                cond = BinaryExpression(
                    expr, BinaryOperator.EQUAL, LiteralValue(old_scalar)
                )
            new_scalar = cast("ScalarValue", new_v)
            cur_expr = CaseWhen(cond, LiteralValue(new_scalar), cur_expr)
        return Series(self._session, self._plan, cur_expr, self.name)

    def _coerce_cond(self, cond: object) -> Expression:
        if isinstance(cond, Series):
            if cond._session is not self._session or cond._plan is not self._plan:
                raise AlignmentError(
                    "Condition Series from a different frame "
                    "requires explicit index alignment"
                )
            return cond._expression
        if isinstance(cond, bool):
            return LiteralValue(cond)
        raise TypeError("cond must be a Series or boolean")

    @property
    def size(self) -> int:
        """Return the number of elements, including null values."""
        plan = aggregate_plan(
            self._plan,
            ((self.name or "__duckpd_size__", None, None),),
            AggregateOperator.SIZE,
        )
        return materialized_int(self._session._executor.reduce_scalar(plan))

    def count(self) -> int:
        """Return the number of non-null elements."""
        return materialized_int(self._reduce(AggregateOperator.COUNT))

    def sum(
        self,
        *,
        axis: int | str | None = None,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
    ) -> object:
        """Return the sum of supported numeric or boolean values."""
        validate_axis(axis, series=True)
        self._validate_numeric_only(numeric_only)
        validate_min_count(min_count)
        return self._reduce(
            AggregateOperator.SUM,
            skipna=skipna,
            min_count=min_count,
        )

    def mean(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> object:
        """Return the mean of supported numeric or boolean values."""
        validate_axis(axis, series=True)
        self._validate_numeric_only(numeric_only)
        return self._reduce(AggregateOperator.MEAN, skipna=skipna)

    def min(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> object:
        """Return the minimum of supported numeric or boolean values."""
        validate_axis(axis, series=True)
        self._validate_numeric_only(numeric_only)
        return self._reduce(AggregateOperator.MIN, skipna=skipna)

    def max(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> object:
        """Return the maximum of supported numeric or boolean values."""
        validate_axis(axis, series=True)
        self._validate_numeric_only(numeric_only)
        return self._reduce(AggregateOperator.MAX, skipna=skipna)

    def std(
        self,
        *,
        axis: int | str | None = None,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> object:
        """Return the sample standard deviation."""
        validate_axis(axis, series=True)
        self._validate_numeric_only(numeric_only)
        validate_ddof(ddof)
        return self._reduce(AggregateOperator.STD, skipna=skipna, ddof=ddof)

    def var(
        self,
        *,
        axis: int | str | None = None,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> object:
        """Return the sample variance."""
        validate_axis(axis, series=True)
        self._validate_numeric_only(numeric_only)
        validate_ddof(ddof)
        return self._reduce(AggregateOperator.VAR, skipna=skipna, ddof=ddof)

    def median(
        self,
        *,
        axis: int | str | None = None,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> object:
        """Return the median value."""
        validate_axis(axis, series=True)
        self._validate_numeric_only(numeric_only)
        return self._reduce(AggregateOperator.MEDIAN, skipna=skipna)

    def quantile(
        self,
        q: float = 0.5,
        *,
        interpolation: str = "linear",
    ) -> object:
        """Return the quantile value."""
        if interpolation != "linear":
            raise UnsupportedOperationError(
                "DuckPD quantile currently supports only interpolation='linear'"
            )
        q_val = validate_quantile(q)
        return self._reduce(AggregateOperator.QUANTILE, q=q_val)

    def any(
        self,
        *,
        axis: int | str | None = 0,
        bool_only: bool = False,
        skipna: bool = True,
    ) -> bool:
        """Return True if any element is True."""
        validate_axis(axis, series=True)
        if bool_only and not is_numeric_type(
            expression_type(self._plan, self._expression)
        ):
            return False
        res = self._reduce(AggregateOperator.ANY, skipna=skipna)
        return bool(res)

    def all(
        self,
        *,
        axis: int | str | None = 0,
        bool_only: bool = False,
        skipna: bool = True,
    ) -> bool:
        """Return True if all elements are True."""
        validate_axis(axis, series=True)
        if bool_only and not is_numeric_type(
            expression_type(self._plan, self._expression)
        ):
            return True
        res = self._reduce(AggregateOperator.ALL, skipna=skipna)
        return bool(res)

    def nunique(self) -> int:
        """Return the number of unique non-null values."""
        return materialized_int(self._reduce(AggregateOperator.NUNIQUE))

    def unique(self) -> pd.Series:
        """Return unique non-null values as a pandas Series."""
        from duckpd._logical import Column, ColumnId

        out_col = Column(
            ColumnId.create(),
            self.name or "__duckpd_unique__",
            expression_type(self._plan, self._expression),
        )
        agg = AggregateExpression(
            out_col,
            operator=None,
            expression=self._expression,
            input_duckdb_type=expression_type(self._plan, self._expression),
        )
        plan = AggregatePlan(
            self._plan,
            (agg,),
            after_aggregate((out_col,)),
        )
        result = self._session._executor.collect(plan)
        return result.iloc[:, 0]

    def value_counts(
        self,
        *,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> pd.Series:
        """Return a Series containing counts of unique values."""
        from duckpd._logical import Column, ColumnId

        if not isinstance(self._expression, ColumnRef):
            raise UnsupportedOperationError(
                "value_counts currently supports only direct column references"
            )
        input_type = expression_type(self._plan, self._expression)
        key_col = Column(
            ColumnId.create(),
            self.name or "__duckpd_value__",
            input_type,
        )
        count_col = Column(ColumnId.create(), "count", "BIGINT")
        aggregates = (
            AggregateExpression(
                key_col,
                operator=None,
                expression=self._expression,
                input_duckdb_type=input_type,
            ),
            AggregateExpression(
                count_col,
                operator=AggregateOperator.SIZE,
                expression=None,
                input_duckdb_type=None,
            ),
        )
        ordering_keys = (
            (
                OrderColumn(
                    count_col.id,
                    SortDirection.ASCENDING if ascending else SortDirection.DESCENDING,
                    NullPlacement.LAST,
                ),
            )
            if sort
            else ()
        )
        metadata = after_aggregate(
            (key_col, count_col),
            index_ids=(key_col.id,),
            ordering_keys=ordering_keys,
        )
        plan = AggregatePlan(
            self._plan,
            aggregates,
            metadata,
            keys=(self._expression.column_id,),
            dropna=dropna,
            sort=False,
        )
        if sort:
            sort_keys = (
                SortKey(
                    ColumnRef(count_col.id),
                    SortDirection.ASCENDING if ascending else SortDirection.DESCENDING,
                    NullPlacement.LAST,
                ),
            )
            plan = SortPlan(plan, sort_keys, after_sort(metadata, sort_keys))
        result = self._session._executor.collect(plan)
        return result.iloc[:, 0]

    def nlargest(
        self, n: int = 5, *, keep: Literal["first", "last"] = "first"
    ) -> Series:
        """Return the largest ``n`` elements."""
        return self._top_n(n, largest=True, keep=keep)

    def nsmallest(
        self, n: int = 5, *, keep: Literal["first", "last"] = "first"
    ) -> Series:
        """Return the smallest ``n`` elements."""
        return self._top_n(n, largest=False, keep=keep)

    def _top_n(self, n: int, *, largest: bool, keep: str) -> Series:
        if n < 0:
            raise ValueError("n must be non-negative")
        if keep not in {"first", "last"}:
            raise ValueError("keep must be 'first' or 'last'")
        from duckpd.frame import DataFrame

        frame = DataFrame(self._session, self._plan)
        col_label = self.name or "__duckpd_topn__"
        if self.name is None:
            frame = frame.assign(__duckpd_topn__=self)
        result = frame._top_n(n, col_label, largest=largest, keep=keep)
        return Series(
            self._session,
            result._plan,
            ColumnRef(self._find_column_id(col_label)),
            self.name,
        )

    def _find_column_id(self, label: str) -> ColumnId:
        from duckpd._metadata import find_column

        return find_column(self._plan.metadata, label, include_hidden=True).id

    def drop_duplicates(
        self,
        *,
        keep: Literal["first", "last", False] = "first",
        inplace: bool = False,
    ) -> Series:
        """Return a Series with duplicate values removed."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        from duckpd.frame import DataFrame

        col_label = self.name or "__duckpd_dedup__"
        frame = DataFrame(self._session, self._plan)
        if self.name is None:
            frame = frame.assign(__duckpd_dedup__=self)
        result = frame.drop_duplicates(subset=[col_label], keep=keep)
        return Series(
            self._session,
            result._plan,
            ColumnRef(self._find_column_id(col_label)),
            self.name,
        )

    def groupby(
        self,
        by: str | Sequence[str] | Series | Sequence[Series],
        *,
        as_index: bool = True,
        sort: bool = True,
        dropna: bool = True,
        observed: bool = True,
    ) -> SeriesGroupBy:
        """Group Series using a mapper or by a Series/column of groups."""
        from duckpd.groupby import SeriesGroupBy

        return SeriesGroupBy(
            self,
            by=by,
            as_index=as_index,
            sort=sort,
            dropna=dropna,
            observed=observed,
        )

    def _binary(self, other: object, operator: BinaryOperator) -> Series:
        if (
            isinstance(other, Series)
            and other._plan is not self._plan
            and operator
            in {
                BinaryOperator.ADD,
                BinaryOperator.SUBTRACT,
                BinaryOperator.MULTIPLY,
                BinaryOperator.TRUE_DIVIDE,
                BinaryOperator.MODULO,
            }
        ):
            return self._aligned_binary(other, operator)
        right = self._coerce_other(other)
        return Series(
            self._session,
            self._plan,
            BinaryExpression(self._expression, operator, right),
            self.name,
        )

    def _rbinary(self, other: object, operator: BinaryOperator) -> Series:
        if isinstance(other, Series) and other._plan is not self._plan:
            return other._aligned_binary(self, operator)
        left = self._coerce_other(other)
        return Series(
            self._session,
            self._plan,
            BinaryExpression(left, operator, self._expression),
            self.name,
        )

    def _aligned_binary(
        self,
        other: Series,
        operator: BinaryOperator,
    ) -> Series:
        from duckpd._merging import plan_merge, validate_explicit_index_alignment

        left = self.to_frame(name="__duckpd_align_left__")
        right = other.to_frame(name="__duckpd_align_right__")
        validate_explicit_index_alignment(left, right)
        left_column = left._plan.metadata.visible_columns[0]
        right_column = right._plan.metadata.visible_columns[0]
        if not is_numeric_type(left_column.duckdb_type) or not is_numeric_type(
            right_column.duckdb_type
        ):
            raise UnsupportedOperationError(
                "Cross-frame Series arithmetic currently supports only numeric data"
            )
        plan = plan_merge(
            left,
            right,
            how="outer",
            left_index=True,
            right_index=True,
            sort=True,
            suffixes=("", ""),
            validate="1:1",
        )
        name = self.name if self.name == other.name else None
        return Series(
            self._session,
            plan,
            BinaryExpression(
                ColumnRef(left_column.id),
                operator,
                ColumnRef(right_column.id),
            ),
            name,
        )

    def _unary(self, operator: UnaryOperator) -> Series:
        return Series(
            self._session,
            self._plan,
            UnaryExpression(operator, self._expression),
            self.name,
        )

    def _call_function(self, name: str, *args: Expression) -> Series:
        """Construct a new Series by calling a function on this expression."""
        from duckpd._logical import FunctionCall

        return Series(
            self._session,
            self._plan,
            FunctionCall(name, (self._expression, *args)),
            self.name,
        )

    @property
    def str(self) -> StringMethods:
        """Vectorized string functions for Series."""
        from duckpd.accessors import StringMethods

        return StringMethods(self)

    @property
    def dt(self) -> DatetimeProperties:
        """Access datetime properties of Series values."""
        from duckpd.accessors import DatetimeProperties

        return DatetimeProperties(self)

    def _require_order(self) -> tuple[SortKey, ...]:
        """Validate that the plan has guaranteed ordering and return SortKeys."""
        ordering = self._plan.metadata.ordering
        if not ordering.keys:
            raise UnorderedOperationError("Operation requires a guaranteed row order")
        return tuple(
            SortKey(
                ColumnRef(k.column_id),
                k.direction,
                k.null_placement,
            )
            for k in ordering.keys
        )

    def cumsum(self, *, axis: int | str | None = None, skipna: bool = True) -> Series:
        """Return cumulative sum over a DataFrame or Series axis."""
        validate_axis(axis, series=True)
        order_keys = self._require_order()
        in_type = expression_type(self._plan, self._expression)
        op = (
            CastExpression(self._expression, "BIGINT")
            if in_type == "BOOLEAN"
            else self._expression
        )
        window: Expression = WindowExpression(
            function="sum",
            arguments=(op,),
            order_by=order_keys,
            frame_spec="ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
        )
        if in_type in {
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "BOOLEAN",
        }:
            window = CastExpression(window, "BIGINT")
        if skipna:
            expr = CaseWhen(
                FunctionCall("isnull", (self._expression,)),
                LiteralValue(None),
                window,
            )
        else:
            has_null = WindowExpression(
                function="bool_or",
                arguments=(FunctionCall("isnull", (self._expression,)),),
                order_by=order_keys,
                frame_spec="ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
            )
            expr = CaseWhen(
                has_null,
                LiteralValue(None),
                window,
            )
        return Series(self._session, self._plan, expr, self.name)

    def cummin(self, *, axis: int | str | None = None, skipna: bool = True) -> Series:
        """Return cumulative minimum over a DataFrame or Series axis."""
        validate_axis(axis, series=True)
        order_keys = self._require_order()
        window = WindowExpression(
            function="min",
            arguments=(self._expression,),
            order_by=order_keys,
            frame_spec="ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
        )
        if skipna:
            expr = CaseWhen(
                FunctionCall("isnull", (self._expression,)),
                LiteralValue(None),
                window,
            )
        else:
            has_null = WindowExpression(
                function="bool_or",
                arguments=(FunctionCall("isnull", (self._expression,)),),
                order_by=order_keys,
                frame_spec="ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
            )
            expr = CaseWhen(
                has_null,
                LiteralValue(None),
                window,
            )
        return Series(self._session, self._plan, expr, self.name)

    def cummax(self, *, axis: int | str | None = None, skipna: bool = True) -> Series:
        """Return cumulative maximum over a DataFrame or Series axis."""
        validate_axis(axis, series=True)
        order_keys = self._require_order()
        window = WindowExpression(
            function="max",
            arguments=(self._expression,),
            order_by=order_keys,
            frame_spec="ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
        )
        if skipna:
            expr = CaseWhen(
                FunctionCall("isnull", (self._expression,)),
                LiteralValue(None),
                window,
            )
        else:
            has_null = WindowExpression(
                function="bool_or",
                arguments=(FunctionCall("isnull", (self._expression,)),),
                order_by=order_keys,
                frame_spec="ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
            )
            expr = CaseWhen(
                has_null,
                LiteralValue(None),
                window,
            )
        return Series(self._session, self._plan, expr, self.name)

    def cumprod(self, *, axis: int | str | None = None, skipna: bool = True) -> Series:
        """Return cumulative product over a DataFrame or Series axis."""
        validate_axis(axis, series=True)
        order_keys = self._require_order()
        in_type = expression_type(self._plan, self._expression)
        op = (
            CastExpression(self._expression, "BIGINT")
            if in_type == "BOOLEAN"
            else self._expression
        )
        window: Expression = WindowExpression(
            function="product",
            arguments=(op,),
            order_by=order_keys,
            frame_spec="ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
        )
        if in_type in {
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "BOOLEAN",
        }:
            window = CastExpression(window, "BIGINT")
        if skipna:
            expr = CaseWhen(
                FunctionCall("isnull", (self._expression,)),
                LiteralValue(None),
                window,
            )
        else:
            has_null = WindowExpression(
                function="bool_or",
                arguments=(FunctionCall("isnull", (self._expression,)),),
                order_by=order_keys,
                frame_spec="ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
            )
            expr = CaseWhen(
                has_null,
                LiteralValue(None),
                window,
            )
        return Series(self._session, self._plan, expr, self.name)

    def shift(
        self,
        periods: int = 1,
        *,
        freq: object = None,
        axis: int | str | None = 0,
        fill_value: object = None,
    ) -> Series:
        """Shift index by desired number of periods."""
        validate_axis(axis, series=True)
        if freq is not None:
            raise UnsupportedOperationError("DuckPD does not support freq in shift")
        order_keys = self._require_order()
        if periods == 0:
            return self
        func_name = "lag" if periods > 0 else "lead"
        offset = abs(periods)
        window = WindowExpression(
            function=func_name,
            arguments=(self._expression, LiteralValue(offset)),
            order_by=order_keys,
        )
        if fill_value is not None:
            fill_expr = self._coerce_other(fill_value)
            expr = CaseWhen(
                FunctionCall("isnull", (window,)),
                fill_expr,
                window,
            )
        else:
            expr = window
        return Series(self._session, self._plan, expr, self.name)

    def diff(self, periods: int = 1, *, axis: int | str | None = 0) -> Series:
        """First discrete difference of element."""
        validate_axis(axis, series=True)
        shifted = self.shift(periods=periods, axis=axis)
        return self - shifted

    def pct_change(
        self,
        periods: int = 1,
        *,
        fill_method: object = None,
        limit: object = None,
        freq: object = None,
        axis: int | str | None = 0,
    ) -> Series:
        """Percentage change between the current and a prior element."""
        validate_axis(axis, series=True)
        if fill_method is not None:
            raise UnsupportedOperationError(
                "DuckPD does not support fill_method in pct_change"
            )
        if limit is not None:
            raise UnsupportedOperationError(
                "DuckPD does not support limit in pct_change"
            )
        if freq is not None:
            raise UnsupportedOperationError(
                "DuckPD does not support freq in pct_change"
            )
        shifted = self.shift(periods=periods, axis=axis)
        diff_val = self - shifted
        return diff_val / shifted

    def rank(
        self,
        axis: int | str | None = 0,
        method: Literal["average", "min", "max", "first", "dense"] = "average",
        numeric_only: bool = False,
        na_option: Literal["keep", "top", "bottom"] = "keep",
        ascending: bool = True,
        pct: bool = False,
    ) -> Series:
        """Compute numerical data ranks (1 through n) along axis."""
        validate_axis(axis, series=True)
        self._validate_numeric_only(numeric_only)
        if method not in {"average", "min", "max", "first", "dense"}:
            raise ValueError(
                "method must be 'average', 'min', 'max', 'first', or 'dense'"
            )
        if na_option not in {"keep", "top", "bottom"}:
            raise ValueError("na_option must be 'keep', 'top', or 'bottom'")

        direction = SortDirection.ASCENDING if ascending else SortDirection.DESCENDING
        if na_option == "keep":
            null_placement = NullPlacement.LAST
        elif na_option == "top":
            null_placement = NullPlacement.FIRST
        else:  # bottom
            null_placement = NullPlacement.LAST

        val_sort_key = SortKey(self._expression, direction, null_placement)
        order_by_keys: tuple[SortKey, ...]
        if method == "first":
            guaranteed_order = self._require_order()
            tiebreaker_keys = list(guaranteed_order)
            order_by_keys = (val_sort_key, *tiebreaker_keys)
        else:
            # other methods rank solely by value
            order_by_keys = (val_sort_key,)

        if method == "min":
            rank_expr: Expression = WindowExpression("rank", order_by=order_by_keys)
        elif method == "dense":
            rank_expr = WindowExpression("dense_rank", order_by=order_by_keys)
        elif method == "first":
            rank_expr = WindowExpression("row_number", order_by=order_by_keys)
        elif method == "max":
            rank_val = WindowExpression("rank", order_by=order_by_keys)
            count_val = WindowExpression(
                "count",
                arguments=(LiteralValue(1),),
                partition_by=(self._expression,),
            )
            # rank + count(*) - 1
            rank_expr = BinaryExpression(
                BinaryExpression(rank_val, BinaryOperator.ADD, count_val),
                BinaryOperator.SUBTRACT,
                LiteralValue(1),
            )
        else:  # average
            rank_val = WindowExpression("rank", order_by=order_by_keys)
            count_val = WindowExpression(
                "count",
                arguments=(LiteralValue(1),),
                partition_by=(self._expression,),
            )
            # rank + (count(*) - 1.0) / 2.0
            count_minus_one = BinaryExpression(
                CastExpression(count_val, "DOUBLE"),
                BinaryOperator.SUBTRACT,
                LiteralValue(1.0),
            )
            offset = BinaryExpression(
                count_minus_one,
                BinaryOperator.TRUE_DIVIDE,
                LiteralValue(2.0),
            )
            rank_expr = BinaryExpression(
                CastExpression(rank_val, "DOUBLE"),
                BinaryOperator.ADD,
                offset,
            )

        if pct:
            total_count: Expression
            if na_option == "keep":
                # Only count non-nulls for percentage
                total_count = CastExpression(
                    WindowExpression(
                        "count",
                        arguments=(self._expression,),
                    ),
                    "DOUBLE",
                )
            else:
                total_count = CastExpression(
                    WindowExpression(
                        "count",
                        arguments=(LiteralValue(1),),
                    ),
                    "DOUBLE",
                )
            rank_expr = BinaryExpression(
                CastExpression(rank_expr, "DOUBLE"),
                BinaryOperator.TRUE_DIVIDE,
                total_count,
            )
        else:
            if method == "average":
                rank_expr = CastExpression(rank_expr, "DOUBLE")
            elif method in {"min", "dense", "first", "max"}:
                if na_option == "keep":
                    rank_expr = CastExpression(rank_expr, "DOUBLE")
                else:
                    rank_expr = CastExpression(rank_expr, "BIGINT")

        if na_option == "keep":
            rank_expr = CaseWhen(
                FunctionCall("isnull", (self._expression,)),
                LiteralValue(None),
                rank_expr,
            )

        return Series(self._session, self._plan, rank_expr, self.name)

    def rolling(
        self,
        window: int,
        min_periods: int | None = None,
        *,
        center: bool = False,
    ) -> Rolling:
        """Provide rolling window calculations."""
        from duckpd.window import Rolling

        return Rolling(self, window, min_periods=min_periods, center=center)

    def expanding(
        self,
        min_periods: int = 1,
    ) -> Expanding:
        """Provide expanding window calculations."""
        from duckpd.window import Expanding

        return Expanding(self, min_periods=min_periods)

    def _coerce_other(self, other: object) -> Expression:
        if isinstance(other, Series):
            if other._session is not self._session or other._plan is not self._plan:
                raise AlignmentError(
                    "Series from different frames require explicit index alignment"
                )
            return other._expression
        if not is_scalar_value(other):
            raise TypeError("DuckPD does not support this scalar literal type")
        return LiteralValue(other)

    def _reduce(
        self,
        operator: AggregateOperator,
        *,
        skipna: bool = True,
        min_count: int = 0,
        ddof: int = 1,
        q: float = 0.5,
    ) -> object:
        input_type = expression_type(self._plan, self._expression)
        plan = aggregate_plan(
            self._plan,
            ((self.name or "__duckpd_reduction__", self._expression, input_type),),
            operator,
            skipna=skipna,
            min_count=min_count,
            ddof=ddof,
            q=q,
        )
        return self._session._executor.reduce_scalar(plan)

    @staticmethod
    def _validate_numeric_only(numeric_only: bool) -> None:
        if numeric_only:
            raise UnsupportedOperationError(
                "Series reductions do not support numeric_only=True"
            )
