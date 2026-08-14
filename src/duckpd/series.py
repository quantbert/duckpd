"""Lazy Series expression API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from duckpd._logical import (
    BinaryExpression,
    BinaryOperator,
    LiteralValue,
    UnaryExpression,
    UnaryOperator,
)
from duckpd._typing import is_scalar_value
from duckpd.errors import AlignmentError

if TYPE_CHECKING:
    from duckpd._logical import Expression, LogicalPlan
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