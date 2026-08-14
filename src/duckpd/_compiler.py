"""Compilation of DuckPD logical plans into DuckDB relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb
import pandas as pd
import pyarrow as pa

from duckpd._logical import (
    AggregateExpression,
    AggregateOperator,
    AggregatePlan,
    ArrowSource,
    BinaryOperator,
    Column,
    ColumnId,
    ColumnRef,
    Expression,
    FilterPlan,
    LiteralValue,
    LogicalPlan,
    NullPlacement,
    PandasSource,
    ParquetSource,
    ProjectPlan,
    ScanPlan,
    SortDirection,
    SortKey,
    SortPlan,
    SqlSource,
    TableSource,
    UnaryExpression,
    UnaryOperator,
)
from duckpd._quoting import quote_identifier

if TYPE_CHECKING:
    from duckpd.session import Session


@dataclass(frozen=True)
class CompiledFrame:
    """A DuckDB relation and its logical-to-physical column bindings."""

    relation: duckdb.DuckDBPyRelation
    bindings: dict[ColumnId, str]


class DuckDBCompiler:
    """Compile typed DuckPD plans without triggering query output."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def inspect_source(
        self,
        source: ArrowSource | PandasSource | ParquetSource | SqlSource | TableSource,
    ) -> tuple[Column, ...]:
        relation = self._relation_for_source(source)
        labels = relation.columns
        if len(labels) != len(set(labels)):
            msg = "DuckPD does not yet support duplicate column labels"
            raise ValueError(msg)
        return tuple(
            Column(ColumnId.create(), label, str(duckdb_type))
            for label, duckdb_type in zip(labels, relation.types, strict=True)
        )

    def project_visible(
        self, compiled: CompiledFrame, plan: LogicalPlan
    ) -> CompiledFrame:
        """Drop hidden metadata columns at non-pandas output boundaries."""
        visible = plan.metadata.visible_columns
        expressions = tuple(
            duckdb.SQLExpression(quote_identifier(compiled.bindings[column.id])).alias(
                column.label
            )
            for column in visible
        )
        relation = compiled.relation.project(*expressions)
        return CompiledFrame(
            relation,
            {column.id: column.label for column in visible},
        )

    def compile(self, plan: LogicalPlan) -> CompiledFrame:
        self._session._ensure_open()
        if isinstance(plan, ScanPlan):
            relation = self._relation_for_source(plan.source)
            bindings = {column.id: column.label for column in plan.columns}
            return CompiledFrame(relation, bindings)

        compiled_input = self.compile(plan.input)
        if isinstance(plan, FilterPlan):
            predicate = self.compile_expression(plan.predicate, compiled_input.bindings)
            return CompiledFrame(
                compiled_input.relation.filter(predicate), compiled_input.bindings
            )
        if isinstance(plan, ProjectPlan):
            expressions = tuple(
                self.compile_expression(projection.expression, compiled_input.bindings)
                .alias(projection.column.label)
                for projection in plan.projections
            )
            relation = compiled_input.relation.project(*expressions)
            bindings = {
                projection.column.id: projection.column.label
                for projection in plan.projections
            }
            return CompiledFrame(relation, bindings)
        if isinstance(plan, AggregatePlan):
            input_rel = compiled_input.relation
            if plan.keys and plan.dropna:
                # filter out rows where any group key is null
                for key_id in plan.keys:
                    key_label = quote_identifier(compiled_input.bindings[key_id])
                    input_rel = input_rel.filter(
                        duckdb.SQLExpression(f"{key_label} IS NOT NULL")
                    )

            expressions = [
                self._compile_aggregate(aggregate, compiled_input.bindings).alias(
                    aggregate.column.label
                )
                for aggregate in plan.aggregates
            ]
            if plan.keys:
                key_labels = [
                    quote_identifier(compiled_input.bindings[key_id])
                    for key_id in plan.keys
                ]
                groups_spec = ", ".join(key_labels)
                relation = input_rel.aggregate(expressions, groups_spec)
                if plan.sort:
                    sort_keys = [
                        duckdb.SQLExpression(k).asc().nulls_last()
                        for k in key_labels
                    ]
                    relation = relation.sort(*sort_keys)
            else:
                relation = input_rel.aggregate(expressions)

            return CompiledFrame(
                relation,
                {
                    aggregate.column.id: aggregate.column.label
                    for aggregate in plan.aggregates
                },
            )
        if isinstance(plan, SortPlan):
            keys = tuple(
                self._compile_sort_key(key, compiled_input.bindings)
                for key in plan.keys
            )
            return CompiledFrame(
                compiled_input.relation.sort(*keys), compiled_input.bindings
            )
        relation = compiled_input.relation.limit(plan.count, offset=plan.offset)
        return CompiledFrame(relation, compiled_input.bindings)

    def compile_expression(
        self, expression: Expression, bindings: dict[ColumnId, str]
    ) -> duckdb.Expression:
        """Compile a typed expression against physical relation bindings."""
        if isinstance(expression, ColumnRef):
            try:
                label = bindings[expression.column_id]
            except KeyError as error:
                msg = (
                    f"Column {expression.column_id.value} is not available "
                    "in this plan"
                )
                raise KeyError(msg) from error
            return duckdb.SQLExpression(quote_identifier(label))
        if isinstance(expression, LiteralValue):
            return duckdb.ConstantExpression(expression.value)
        if isinstance(expression, UnaryExpression):
            operand = self.compile_expression(expression.operand, bindings)
            if expression.operator is UnaryOperator.INVERT:
                return ~operand
            if expression.operator is UnaryOperator.NEGATE:
                return -operand
            if expression.operator is UnaryOperator.POSITIVE:
                return operand
            raise AssertionError(f"Unknown unary operator: {expression.operator}")

        left = self.compile_expression(expression.left, bindings)
        right = self.compile_expression(expression.right, bindings)
        operator = expression.operator
        if operator is BinaryOperator.ADD:
            return left + right
        if operator is BinaryOperator.SUBTRACT:
            return left - right
        if operator is BinaryOperator.MULTIPLY:
            return left * right
        if operator is BinaryOperator.TRUE_DIVIDE:
            return left / right
        if operator is BinaryOperator.MODULO:
            return left % right
        if operator is BinaryOperator.EQUAL:
            return left == right
        if operator is BinaryOperator.NOT_EQUAL:
            return left != right
        if operator is BinaryOperator.LESS_THAN:
            return left < right
        if operator is BinaryOperator.LESS_EQUAL:
            return left <= right
        if operator is BinaryOperator.GREATER_THAN:
            return left > right
        if operator is BinaryOperator.GREATER_EQUAL:
            return left >= right
        if operator is BinaryOperator.AND:
            return left & right
        if operator is BinaryOperator.OR:
            return left | right
        raise AssertionError(f"Unknown binary operator: {operator}")

    def _relation_for_source(
        self,
        source: ArrowSource | PandasSource | ParquetSource | SqlSource | TableSource,
    ) -> duckdb.DuckDBPyRelation:
        self._session._ensure_open()
        if isinstance(source, PandasSource):
            value = self._session._get_registered_source(source.key)
            if not isinstance(value, pd.DataFrame):
                msg = f"Registered source {source.key!r} is not a pandas DataFrame"
                raise TypeError(msg)
            return self._session._connection.from_df(value)
        if isinstance(source, ArrowSource):
            value = self._session._get_registered_source(source.key)
            if not isinstance(value, (pa.Table, pa.RecordBatch)):
                msg = f"Registered source {source.key!r} is not an Arrow table or batch"
                raise TypeError(msg)
            return self._session._connection.from_arrow(value)
        if isinstance(source, TableSource):
            return self._session._connection.table(source.name)
        if isinstance(source, SqlSource):
            return self._session._connection.sql(source.query)

        paths: str | list[str] = (
            source.paths[0] if len(source.paths) == 1 else list(source.paths)
        )
        return self._session._connection.read_parquet(
            paths,
            hive_partitioning=source.hive_partitioning,
            union_by_name=source.union_by_name,
        )

    def _compile_sort_key(
        self, key: SortKey, bindings: dict[ColumnId, str]
    ) -> duckdb.Expression:
        result = self.compile_expression(key.expression, bindings)
        result = (
            result.asc()
            if key.direction is SortDirection.ASCENDING
            else result.desc()
        )
        return (
            result.nulls_first()
            if key.null_placement is NullPlacement.FIRST
            else result.nulls_last()
        )

    def _compile_aggregate(
        self,
        aggregate: AggregateExpression,
        bindings: dict[ColumnId, str],
    ) -> duckdb.Expression:
        if aggregate.operator is AggregateOperator.SIZE:
            return duckdb.SQLExpression("count(*)")
        if aggregate.expression is None:
            raise AssertionError("Only size aggregates may omit an expression")

        # Identity column pass-through for group keys in projection
        if aggregate.operator is None:
            return self.compile_expression(aggregate.expression, bindings)

        operand = self.compile_expression(aggregate.expression, bindings)
        non_null_count = duckdb.FunctionExpression("count", operand)
        if aggregate.operator is AggregateOperator.COUNT:
            return non_null_count

        function = {
            AggregateOperator.SUM: "sum",
            AggregateOperator.MEAN: "avg",
            AggregateOperator.MIN: "min",
            AggregateOperator.MAX: "max",
        }[aggregate.operator]
        aggregate_operand = operand
        if aggregate.input_duckdb_type == "BOOLEAN" and function in {"sum", "avg"}:
            aggregate_operand = operand.cast("BIGINT")
        value = duckdb.FunctionExpression(function, aggregate_operand)
        if aggregate.operator is AggregateOperator.SUM:
            value = duckdb.CaseExpression(
                non_null_count == duckdb.ConstantExpression(0),
                duckdb.ConstantExpression(0),
            ).otherwise(value)
            if aggregate.input_duckdb_type == "BOOLEAN" or (
                aggregate.input_duckdb_type is not None
                and aggregate.input_duckdb_type
                in {"TINYINT", "SMALLINT", "INTEGER", "BIGINT"}
            ):
                value = value.cast("BIGINT")
            elif aggregate.input_duckdb_type in {
                "UTINYINT",
                "USMALLINT",
                "UINTEGER",
                "UBIGINT",
            }:
                value = value.cast("UBIGINT")

        invalid = non_null_count < duckdb.ConstantExpression(aggregate.min_count)
        if not aggregate.skipna:
            row_count = duckdb.SQLExpression("count(*)")
            invalid = invalid | (non_null_count < row_count)
        return duckdb.CaseExpression(
            invalid,
            duckdb.ConstantExpression(None),
        ).otherwise(value)
