"""Construction and validation of pandas-compatible global reductions."""

from __future__ import annotations

from numbers import Integral

from duckpd._logical import (
    AggregateExpression,
    AggregateOperator,
    AggregatePlan,
    Column,
    ColumnId,
    ColumnRef,
    Expression,
    LiteralValue,
    LogicalPlan,
    UnaryExpression,
)
from duckpd._metadata import after_aggregate
from duckpd.errors import UnsupportedOperationError

_NUMERIC_TYPES = frozenset(
    {
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
        "FLOAT",
        "DOUBLE",
        "BOOLEAN",
    }
)


def is_numeric_type(duckdb_type: str) -> bool:
    """Return whether the initial reduction API supports this DuckDB type."""
    return duckdb_type in _NUMERIC_TYPES or duckdb_type.startswith("DECIMAL(")


def expression_type(plan: LogicalPlan, expression: Expression) -> str:
    """Infer enough expression type information to validate basic reductions."""
    if isinstance(expression, ColumnRef):
        for column in plan.columns:
            if column.id == expression.column_id:
                return column.duckdb_type
        raise AssertionError(f"Expression column is absent: {expression.column_id}")
    if isinstance(expression, LiteralValue):
        value = expression.value
        if isinstance(value, bool):
            return "BOOLEAN"
        if isinstance(value, int):
            return "BIGINT"
        if isinstance(value, float):
            return "DOUBLE"
        return "UNKNOWN"
    if isinstance(expression, UnaryExpression):
        return expression_type(plan, expression.operand)
    left = expression_type(plan, expression.left)
    right = expression_type(plan, expression.right)
    if is_numeric_type(left) and is_numeric_type(right):
        return "DOUBLE" if "DOUBLE" in {left, right} else "BIGINT"
    return "UNKNOWN"


def validate_axis(axis: int | str | None, *, series: bool) -> None:
    """Accept only pandas' column-wise reduction axis."""
    valid: set[int | str | None] = {0, "index"}
    if series:
        valid.add(None)
    if axis not in valid:
        raise UnsupportedOperationError(
            "DuckPD reductions currently support only axis=0 or axis='index'"
        )


def validate_min_count(min_count: object) -> None:
    """Validate pandas' minimum-valid-value threshold."""
    if not isinstance(min_count, int):
        raise TypeError("min_count must be an integer")
    if min_count < 0:
        raise ValueError("min_count must be non-negative")


def materialized_int(value: object) -> int:
    """Validate and normalize an integral scalar produced by DuckDB."""
    if not isinstance(value, Integral):
        raise TypeError("DuckDB did not return an integral reduction result")
    return int(value)


def aggregate_plan(
    input_plan: LogicalPlan,
    requests: tuple[tuple[str, Expression | None, str | None], ...],
    operator: AggregateOperator,
    *,
    skipna: bool = True,
    min_count: int = 0,
) -> AggregatePlan:
    """Build one global aggregate plan with outputs in request order."""
    aggregates: list[AggregateExpression] = []
    for label, expression, input_type in requests:
        if operator not in {AggregateOperator.COUNT, AggregateOperator.SIZE} and (
            input_type is None or not is_numeric_type(input_type)
        ):
            raise UnsupportedOperationError(
                f"{operator.value} currently supports only numeric and boolean data; "
                f"column {label!r} has DuckDB type {input_type}"
            )
        output_type = (
            "BIGINT"
            if operator in {AggregateOperator.COUNT, AggregateOperator.SIZE}
            else "UNKNOWN"
        )
        output = Column(ColumnId.create(), label, output_type)
        aggregates.append(
            AggregateExpression(
                output,
                operator,
                expression,
                input_type,
                skipna,
                min_count,
            )
        )
    columns = tuple(aggregate.column for aggregate in aggregates)
    return AggregatePlan(
        input_plan,
        tuple(aggregates),
        after_aggregate(columns),
    )