"""Lazy Series expression API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from duckpd._logical import (
    AggregateOperator,
    BinaryExpression,
    BinaryOperator,
    LiteralValue,
    UnaryExpression,
    UnaryOperator,
)
from duckpd._reductions import (
    aggregate_plan,
    expression_type,
    materialized_int,
    validate_axis,
    validate_min_count,
)
from duckpd._typing import is_scalar_value
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
    ) -> object:
        input_type = expression_type(self._plan, self._expression)
        plan = aggregate_plan(
            self._plan,
            ((self.name or "__duckpd_reduction__", self._expression, input_type),),
            operator,
            skipna=skipna,
            min_count=min_count,
        )
        return self._session._executor.reduce_scalar(plan)

    @staticmethod
    def _validate_numeric_only(numeric_only: bool) -> None:
        if numeric_only:
            raise UnsupportedOperationError(
                "Series reductions do not support numeric_only=True"
            )
