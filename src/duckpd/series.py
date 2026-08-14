"""Lazy Series expression API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
    LiteralValue,
    NullPlacement,
    OrderColumn,
    SortDirection,
    SortKey,
    SortPlan,
    UnaryExpression,
    UnaryOperator,
)
from duckpd._metadata import after_aggregate, after_sort
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
from duckpd._typing import is_scalar_value, normalize_dtype
from duckpd.errors import AlignmentError, UnsupportedOperationError

if TYPE_CHECKING:
    from duckpd._logical import Expression, LogicalPlan
    from duckpd.accessors import DatetimeProperties, StringMethods
    from duckpd.session import Session


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
        filtered_plan = FilterPlan(self._plan, not_null_pred, self._plan.metadata)
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
        result = frame.sort_values(
            col_label,
            ascending=not largest,
        ).limit(n)
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

    def _binary(self, other: object, operator: BinaryOperator) -> Series:
        right = self._coerce_other(other)
        return Series(
            self._session,
            self._plan,
            BinaryExpression(self._expression, operator, right),
            self.name,
        )

    def _rbinary(self, other: object, operator: BinaryOperator) -> Series:
        left = self._coerce_other(other)
        return Series(
            self._session,
            self._plan,
            BinaryExpression(left, operator, self._expression),
            self.name,
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
