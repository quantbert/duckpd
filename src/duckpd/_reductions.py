"""Construction and validation of pandas-compatible global reductions."""

from __future__ import annotations

from numbers import Integral

from duckpd._logical import (
    AggregateExpression,
    AggregateOperator,
    AggregatePlan,
    BinaryOperator,
    CaseWhen,
    CastExpression,
    Column,
    ColumnId,
    ColumnRef,
    Expression,
    FunctionCall,
    LiteralValue,
    LogicalPlan,
    Nullability,
    UnaryExpression,
    WindowExpression,
)
from duckpd._metadata import after_aggregate
from duckpd._typing import binary_numeric_type
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
        if value is None:
            return "UNKNOWN"
        if isinstance(value, bool):
            return "BOOLEAN"
        if isinstance(value, int):
            return "BIGINT"
        if isinstance(value, float):
            return "DOUBLE"
        if isinstance(value, str):
            return "VARCHAR"
        return "VARCHAR"
    if isinstance(expression, UnaryExpression):
        return expression_type(plan, expression.operand)
    if isinstance(expression, CastExpression):
        return expression.target_type
    if isinstance(expression, CaseWhen):
        value_type = expression_type(plan, expression.value)
        otherwise_type = expression_type(plan, expression.otherwise)
        if value_type == "UNKNOWN":
            return otherwise_type
        if otherwise_type == "UNKNOWN" or otherwise_type == value_type:
            return value_type
        if is_numeric_type(value_type) and is_numeric_type(otherwise_type):
            return binary_numeric_type(value_type, otherwise_type, "add")
        return "UNKNOWN"
    if isinstance(expression, FunctionCall):
        if expression.return_type is not None:
            return expression.return_type
        func_name = expression.name.lower()
        if func_name in {"length", "year", "month", "day", "hour", "minute", "second"}:
            return "BIGINT"
        if func_name in {"starts_with", "ends_with", "contains"}:
            return "BOOLEAN"
        if func_name in {"isnull", "notnull"}:
            return "BOOLEAN"
        if func_name in {"upper", "lower", "trim", "replace", "strftime"}:
            return "VARCHAR"
        return "UNKNOWN"
    if isinstance(expression, WindowExpression):
        func_name = expression.function.lower()
        if func_name in {"row_number", "rank", "dense_rank", "count"}:
            return "BIGINT"
        if func_name in {"avg"}:
            return "DOUBLE"
        if func_name in {"bool_or", "bool_and"}:
            return "BOOLEAN"
        if expression.arguments:
            return expression_type(plan, expression.arguments[0])
        return "UNKNOWN"
    if expression.operator in {
        BinaryOperator.EQUAL,
        BinaryOperator.NOT_EQUAL,
        BinaryOperator.LESS_THAN,
        BinaryOperator.LESS_EQUAL,
        BinaryOperator.GREATER_THAN,
        BinaryOperator.GREATER_EQUAL,
        BinaryOperator.AND,
        BinaryOperator.OR,
    }:
        return "BOOLEAN"
    left = expression_type(plan, expression.left)
    right = expression_type(plan, expression.right)
    return binary_numeric_type(left, right, expression.operator.value)


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


def validate_ddof(ddof: object) -> int:
    """Validate degrees of freedom for variance and standard deviation."""
    if not isinstance(ddof, int):
        raise TypeError("ddof must be an integer")
    if ddof not in {0, 1}:
        raise UnsupportedOperationError("DuckPD currently supports ddof=0 or ddof=1")
    return ddof


def validate_quantile(q: object) -> float:
    """Validate quantile parameter q."""
    if isinstance(q, (int, float)):
        val = float(q)
        if 0.0 <= val <= 1.0:
            return val
        raise ValueError("quantile must be between 0 and 1")
    raise UnsupportedOperationError("DuckPD currently supports only scalar float/int quantiles")


def aggregate_plan(
    input_plan: LogicalPlan,
    requests: tuple[tuple[str, Expression | None, str | None], ...],
    operator: AggregateOperator,
    *,
    skipna: bool = True,
    min_count: int = 0,
    ddof: int = 1,
    q: float = 0.5,
) -> AggregatePlan:
    """Build one global aggregate plan with outputs in request order."""
    aggregates: list[AggregateExpression] = []
    for label, expression, input_type in requests:
        if operator not in {
            AggregateOperator.COUNT,
            AggregateOperator.SIZE,
            AggregateOperator.NUNIQUE,
            AggregateOperator.ANY,
            AggregateOperator.ALL,
        } and (input_type is None or not is_numeric_type(input_type)):
            raise UnsupportedOperationError(
                f"{operator.value} currently supports only numeric and boolean data; "
                f"column {label!r} has DuckDB type {input_type}"
            )
        if operator in {
            AggregateOperator.COUNT,
            AggregateOperator.SIZE,
            AggregateOperator.NUNIQUE,
        }:
            output_type = "BIGINT"
        elif operator in {AggregateOperator.ANY, AggregateOperator.ALL}:
            output_type = "BOOLEAN"
        elif operator in {
            AggregateOperator.STD,
            AggregateOperator.VAR,
            AggregateOperator.MEDIAN,
            AggregateOperator.QUANTILE,
            AggregateOperator.MEAN,
        }:
            output_type = "DOUBLE"
        else:
            output_type = "UNKNOWN"

        non_null = operator in {
            AggregateOperator.COUNT,
            AggregateOperator.SIZE,
            AggregateOperator.NUNIQUE,
            AggregateOperator.ANY,
            AggregateOperator.ALL,
        } or (operator is AggregateOperator.SUM and skipna and min_count == 0)
        output = Column(
            ColumnId.create(),
            label,
            output_type,
            nullable=(Nullability.NON_NULL if non_null else Nullability.NULLABLE),
        )
        aggregates.append(
            AggregateExpression(
                output,
                operator,
                expression,
                input_type,
                skipna,
                min_count,
                ddof=ddof,
                q=q,
            )
        )
    columns = tuple(aggregate.column for aggregate in aggregates)
    return AggregatePlan(
        input_plan,
        tuple(aggregates),
        after_aggregate(columns),
    )
