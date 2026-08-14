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
    CaseWhen,
    CastExpression,
    Column,
    ColumnId,
    ColumnRef,
    Expression,
    FilterPlan,
    FunctionCall,
    JoinPlan,
    JoinType,
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
from duckpd.errors import UnsupportedOperationError

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

        if isinstance(plan, JoinPlan):
            return self._compile_join(plan)

        compiled_input = self.compile(plan.input)
        if isinstance(plan, FilterPlan):
            predicate = self.compile_expression(plan.predicate, compiled_input.bindings)
            return CompiledFrame(
                compiled_input.relation.filter(predicate), compiled_input.bindings
            )
        if isinstance(plan, ProjectPlan):
            expressions = tuple(
                self.compile_expression(
                    projection.expression, compiled_input.bindings
                ).alias(projection.column.label)
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
                        duckdb.SQLExpression(k).asc().nulls_last() for k in key_labels
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
                    f"Column {expression.column_id.value} is not available in this plan"
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
        if isinstance(expression, CastExpression):
            operand = self.compile_expression(expression.operand, bindings)
            return operand.cast(expression.target_type)
        if isinstance(expression, CaseWhen):
            cond = self.compile_expression(expression.condition, bindings)
            val = self.compile_expression(expression.value, bindings)
            other = self.compile_expression(expression.otherwise, bindings)
            return duckdb.CaseExpression(cond, val).otherwise(other)
        if isinstance(expression, FunctionCall):
            compiled_args = [
                self.compile_expression(arg, bindings) for arg in expression.arguments
            ]
            name = expression.name.lower()
            if name == "coalesce" and len(compiled_args) == 2:
                return duckdb.CaseExpression(
                    compiled_args[0].isnull(), compiled_args[1]
                ).otherwise(compiled_args[0])
            if name == "isnull" and len(compiled_args) == 1:
                return compiled_args[0].isnull()
            if name == "notnull" and len(compiled_args) == 1:
                return compiled_args[0].isnotnull()
            return duckdb.FunctionExpression(expression.name, *compiled_args)

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
            result.asc() if key.direction is SortDirection.ASCENDING else result.desc()
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
        if aggregate.operator is AggregateOperator.NUNIQUE:
            # DuckDB's Python FunctionExpression doesn't support DISTINCT,
            # so use a SQL expression with count(distinct ...).
            # We need the SQL representation of the operand; for a simple
            # ColumnRef this is the quoted binding label.
            if isinstance(aggregate.expression, ColumnRef):
                label = bindings[aggregate.expression.column_id]
                return duckdb.SQLExpression(
                    f"count(DISTINCT {quote_identifier(label)})"
                )
            # For non-column expressions, fall back to a subquery-free approach:
            # cast the expression to a SQL fragment via the relation's SQL.
            # This is a limitation; for now only ColumnRef is supported.
            raise UnsupportedOperationError(
                "nunique currently supports only direct column references"
            )
        if aggregate.operator is AggregateOperator.ANY_VALUE:
            return duckdb.FunctionExpression("any_value", operand)

        if aggregate.operator in {AggregateOperator.ANY, AggregateOperator.ALL}:
            bool_op = (
                operand
                if aggregate.input_duckdb_type == "BOOLEAN"
                else operand.cast("BOOLEAN")
            )
            func_name = (
                "bool_or" if aggregate.operator is AggregateOperator.ANY else "bool_and"
            )
            raw_val = duckdb.FunctionExpression(func_name, bool_op)
            default_val = aggregate.operator is AggregateOperator.ALL
            # When skipna is True, all-null gives False for any, True for all.
            # When skipna is False, any gives False if all null, all gives False
            # if null present.
            if aggregate.operator is AggregateOperator.ALL and not aggregate.skipna:
                row_count = duckdb.SQLExpression("count(*)")
                value = duckdb.CaseExpression(
                    non_null_count < row_count,
                    duckdb.ConstantExpression(False),
                ).otherwise(
                    duckdb.CaseExpression(
                        raw_val.isnull(),
                        duckdb.ConstantExpression(default_val),
                    ).otherwise(raw_val)
                )
            else:
                value = duckdb.CaseExpression(
                    raw_val.isnull(),
                    duckdb.ConstantExpression(default_val),
                ).otherwise(raw_val)
            return value

        if aggregate.operator in {AggregateOperator.STD, AggregateOperator.VAR}:
            if aggregate.ddof not in {0, 1}:
                raise UnsupportedOperationError(
                    f"ddof={aggregate.ddof} is not supported; must be 0 or 1"
                )
            if aggregate.ddof == 0:
                func_name = (
                    "stddev_pop"
                    if aggregate.operator is AggregateOperator.STD
                    else "var_pop"
                )
            else:
                func_name = (
                    "stddev_samp"
                    if aggregate.operator is AggregateOperator.STD
                    else "var_samp"
                )
            agg_operand = operand
            if aggregate.input_duckdb_type == "BOOLEAN":
                agg_operand = operand.cast("DOUBLE")
            value = duckdb.FunctionExpression(func_name, agg_operand).cast("DOUBLE")
            # If ddof == 1 and non_null_count <= 1, result is NaN (DuckDB returns NULL)
            # If ddof == 0 and non_null_count == 0, result is NaN
            invalid = (
                non_null_count <= duckdb.ConstantExpression(1)
                if aggregate.ddof == 1
                else non_null_count == duckdb.ConstantExpression(0)
            )
            if not aggregate.skipna:
                row_count = duckdb.SQLExpression("count(*)")
                invalid = invalid | (non_null_count < row_count)
            return duckdb.CaseExpression(
                invalid,
                duckdb.ConstantExpression(None),
            ).otherwise(value)

        if aggregate.operator is AggregateOperator.MEDIAN:
            agg_operand = operand
            if aggregate.input_duckdb_type == "BOOLEAN":
                agg_operand = operand.cast("DOUBLE")
            value = duckdb.FunctionExpression("median", agg_operand).cast("DOUBLE")
            invalid = non_null_count == duckdb.ConstantExpression(0)
            if not aggregate.skipna:
                row_count = duckdb.SQLExpression("count(*)")
                invalid = invalid | (non_null_count < row_count)
            return duckdb.CaseExpression(
                invalid,
                duckdb.ConstantExpression(None),
            ).otherwise(value)

        if aggregate.operator is AggregateOperator.QUANTILE:
            agg_operand = operand
            if aggregate.input_duckdb_type == "BOOLEAN":
                agg_operand = operand.cast("DOUBLE")
            value = duckdb.FunctionExpression(
                "quantile_cont", agg_operand, duckdb.ConstantExpression(aggregate.q)
            ).cast("DOUBLE")
            invalid = non_null_count == duckdb.ConstantExpression(0)
            if not aggregate.skipna:
                row_count = duckdb.SQLExpression("count(*)")
                invalid = invalid | (non_null_count < row_count)
            return duckdb.CaseExpression(
                invalid,
                duckdb.ConstantExpression(None),
            ).otherwise(value)

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

    def _compile_join(self, plan: JoinPlan) -> CompiledFrame:
        left_compiled = self.compile(plan.left)
        right_compiled = self.compile(plan.right)

        lhs_alias = "lhs"
        rhs_alias = "rhs"

        # Explicitly project each side with unique physical column names
        left_proj: list[duckdb.Expression] = []
        left_temp_bindings: dict[ColumnId, str] = {}
        for col in plan.left.columns:
            temp_name = f"l_{col.id.value.hex[:8]}"
            left_proj.append(
                duckdb.SQLExpression(
                    quote_identifier(left_compiled.bindings[col.id])
                ).alias(temp_name)
            )
            left_temp_bindings[col.id] = temp_name

        right_proj: list[duckdb.Expression] = []
        right_temp_bindings: dict[ColumnId, str] = {}
        for col in plan.right.columns:
            temp_name = f"r_{col.id.value.hex[:8]}"
            right_proj.append(
                duckdb.SQLExpression(
                    quote_identifier(right_compiled.bindings[col.id])
                ).alias(temp_name)
            )
            right_temp_bindings[col.id] = temp_name

        lhs_rel = left_compiled.relation.project(*left_proj).set_alias(lhs_alias)
        rhs_rel = right_compiled.relation.project(*right_proj).set_alias(rhs_alias)

        if plan.how is JoinType.CROSS:
            join_rel = lhs_rel.cross(rhs_rel)
        elif plan.how is JoinType.INNER:
            join_cond_parts = [
                (
                    f"{lhs_alias}.{quote_identifier(left_temp_bindings[l_id])} "
                    f"IS NOT DISTINCT FROM "
                    f"{rhs_alias}.{quote_identifier(right_temp_bindings[r_id])}"
                )
                for l_id, r_id in zip(plan.left_keys, plan.right_keys, strict=True)
            ]
            join_rel = lhs_rel.join(rhs_rel, " AND ".join(join_cond_parts), how="inner")
        elif plan.how is JoinType.LEFT:
            join_cond_parts = [
                (
                    f"{lhs_alias}.{quote_identifier(left_temp_bindings[l_id])} "
                    f"IS NOT DISTINCT FROM "
                    f"{rhs_alias}.{quote_identifier(right_temp_bindings[r_id])}"
                )
                for l_id, r_id in zip(plan.left_keys, plan.right_keys, strict=True)
            ]
            join_rel = lhs_rel.join(rhs_rel, " AND ".join(join_cond_parts), how="left")
        elif plan.how is JoinType.RIGHT:
            join_cond_parts = [
                (
                    f"{lhs_alias}.{quote_identifier(left_temp_bindings[l_id])} "
                    f"IS NOT DISTINCT FROM "
                    f"{rhs_alias}.{quote_identifier(right_temp_bindings[r_id])}"
                )
                for l_id, r_id in zip(plan.left_keys, plan.right_keys, strict=True)
            ]
            join_rel = lhs_rel.join(rhs_rel, " AND ".join(join_cond_parts), how="right")
        elif plan.how is JoinType.OUTER:
            join_cond_parts = [
                (
                    f"{lhs_alias}.{quote_identifier(left_temp_bindings[l_id])} "
                    f"IS NOT DISTINCT FROM "
                    f"{rhs_alias}.{quote_identifier(right_temp_bindings[r_id])}"
                )
                for l_id, r_id in zip(plan.left_keys, plan.right_keys, strict=True)
            ]
            join_rel = lhs_rel.join(rhs_rel, " AND ".join(join_cond_parts), how="outer")
        else:
            raise AssertionError(f"Unknown JoinType: {plan.how}")

        # Output projection to match plan.metadata.columns
        final_proj: list[duckdb.Expression] = []
        final_bindings: dict[ColumnId, str] = {}
        for col in plan.metadata.columns:
            l_bind = left_temp_bindings.get(col.id)
            r_bind = None
            if col.id in plan.left_keys:
                idx = plan.left_keys.index(col.id)
                r_key_id = plan.right_keys[idx]
                r_bind = right_temp_bindings.get(r_key_id)

            if (
                l_bind is not None
                and r_bind is not None
                and plan.how in {JoinType.RIGHT, JoinType.OUTER}
            ):
                # Coalesce left and right key so right-only rows retain value
                l_col = f"{lhs_alias}.{quote_identifier(l_bind)}"
                r_col = f"{rhs_alias}.{quote_identifier(r_bind)}"
                source_col = f"COALESCE({l_col}, {r_col})"
            elif l_bind is not None:
                source_col = f"{lhs_alias}.{quote_identifier(l_bind)}"
            elif col.id in right_temp_bindings:
                source_col = (
                    f"{rhs_alias}.{quote_identifier(right_temp_bindings[col.id])}"
                )
            else:
                # Could happen if a key column was synthesized
                raise AssertionError(
                    f"Column {col.id} not found in join input bindings"
                )

            final_proj.append(duckdb.SQLExpression(source_col).alias(col.label))
            final_bindings[col.id] = col.label

        result_rel = join_rel.project(*final_proj)

        if plan.sort and plan.metadata.ordering.keys:
            sort_keys = [
                duckdb.SQLExpression(quote_identifier(final_bindings[k.column_id]))
                .asc()
                .nulls_last()
                if k.direction is SortDirection.ASCENDING
                else duckdb.SQLExpression(quote_identifier(final_bindings[k.column_id]))
                .desc()
                .nulls_last()
                for k in plan.metadata.ordering.keys
            ]
            result_rel = result_rel.sort(*sort_keys)

        return CompiledFrame(result_rel, final_bindings)
