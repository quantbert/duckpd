"""Compilation of DuckPD logical plans into DuckDB relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import duckdb
import pandas as pd
import pyarrow as pa

from duckpd._logical import (
    AggregateExpression,
    AggregateOperator,
    AggregatePlan,
    ArrowSource,
    BinaryExpression,
    BinaryOperator,
    CaseWhen,
    CastExpression,
    Column,
    ColumnId,
    ColumnRef,
    CsvSource,
    Expression,
    FilterPlan,
    FunctionCall,
    JoinPlan,
    JoinType,
    LiteralValue,
    LocIndexPlan,
    LogicalPlan,
    NullPlacement,
    OrderColumn,
    PandasSource,
    ParquetSource,
    ProjectPlan,
    SamplePlan,
    ScanPlan,
    SortDirection,
    SortKey,
    SortPlan,
    SqlSource,
    TableSource,
    UnaryExpression,
    UnaryOperator,
    UnionPlan,
    WindowExpression,
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
        source: (
            ArrowSource
            | CsvSource
            | PandasSource
            | ParquetSource
            | SqlSource
            | TableSource
        ),
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

        if isinstance(plan, UnionPlan):
            return self._compile_union(plan)

        if isinstance(plan, LocIndexPlan):
            return self._compile_loc_index(plan)

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
        if isinstance(plan, SamplePlan):
            clause = self._compile_sample_clause(plan)
            query = f"SELECT * FROM _subquery_to_sample USING SAMPLE {clause}"
            relation = compiled_input.relation.query("_subquery_to_sample", query)
            return CompiledFrame(relation, compiled_input.bindings)
        relation = compiled_input.relation.limit(plan.count, offset=plan.offset)
        return CompiledFrame(relation, compiled_input.bindings)

    def _compile_sample_clause(self, plan: SamplePlan) -> str:
        if plan.n is not None:
            clause = f"reservoir({plan.n} ROWS)"
        elif plan.frac is not None:
            pct = plan.frac * 100.0
            clause = f"reservoir({pct} PERCENT)"
        else:
            clause = "reservoir(1 ROWS)"

        if plan.seed is not None:
            clause += f" REPEATABLE ({plan.seed})"
        return clause

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
        if isinstance(expression, WindowExpression):
            sql = self._expression_to_sql(expression, bindings)
            return duckdb.SQLExpression(sql)

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

    def _expression_to_sql(
        self, expression: Expression, bindings: dict[ColumnId, str]
    ) -> str:
        """Convert an expression to an equivalent SQL fragment."""
        if isinstance(expression, ColumnRef):
            return quote_identifier(bindings[expression.column_id])
        if isinstance(expression, LiteralValue):
            val = expression.value
            if val is None:
                return "NULL"
            if isinstance(val, bool):
                return "TRUE" if val else "FALSE"
            if isinstance(val, (int, float)):
                return str(val)
            if isinstance(val, str):
                escaped = val.replace("'", "''")
                return f"'{escaped}'"
            return f"'{val}'"
        if isinstance(expression, UnaryExpression):
            operand_sql = self._expression_to_sql(expression.operand, bindings)
            if expression.operator is UnaryOperator.INVERT:
                return f"(NOT ({operand_sql}))"
            if expression.operator is UnaryOperator.NEGATE:
                return f"(-({operand_sql}))"
            if expression.operator is UnaryOperator.POSITIVE:
                return f"(+({operand_sql}))"
            raise AssertionError(f"Unknown unary operator: {expression.operator}")
        if isinstance(expression, CastExpression):
            operand_sql = self._expression_to_sql(expression.operand, bindings)
            return f"CAST(({operand_sql}) AS {expression.target_type})"
        if isinstance(expression, CaseWhen):
            cond_sql = self._expression_to_sql(expression.condition, bindings)
            val_sql = self._expression_to_sql(expression.value, bindings)
            other_sql = self._expression_to_sql(expression.otherwise, bindings)
            return f"(CASE WHEN {cond_sql} THEN {val_sql} ELSE {other_sql} END)"
        if isinstance(expression, FunctionCall):
            name = expression.name
            if name.lower() == "coalesce" and len(expression.arguments) == 2:
                arg0 = self._expression_to_sql(expression.arguments[0], bindings)
                arg1 = self._expression_to_sql(expression.arguments[1], bindings)
                return f"(CASE WHEN ({arg0}) IS NULL THEN ({arg1}) ELSE ({arg0}) END)"
            if name.lower() == "isnull" and len(expression.arguments) == 1:
                arg0 = self._expression_to_sql(expression.arguments[0], bindings)
                return f"(({arg0}) IS NULL)"
            if name.lower() == "notnull" and len(expression.arguments) == 1:
                arg0 = self._expression_to_sql(expression.arguments[0], bindings)
                return f"(({arg0}) IS NOT NULL)"
            args_sql = ", ".join(
                self._expression_to_sql(arg, bindings) for arg in expression.arguments
            )
            return f"{name}({args_sql})"
        if isinstance(expression, BinaryExpression):
            left_sql = self._expression_to_sql(expression.left, bindings)
            right_sql = self._expression_to_sql(expression.right, bindings)
            op_map = {
                BinaryOperator.ADD: "+",
                BinaryOperator.SUBTRACT: "-",
                BinaryOperator.MULTIPLY: "*",
                BinaryOperator.TRUE_DIVIDE: "/",
                BinaryOperator.MODULO: "%",
                BinaryOperator.EQUAL: "=",
                BinaryOperator.NOT_EQUAL: "!=",
                BinaryOperator.LESS_THAN: "<",
                BinaryOperator.LESS_EQUAL: "<=",
                BinaryOperator.GREATER_THAN: ">",
                BinaryOperator.GREATER_EQUAL: ">=",
                BinaryOperator.AND: "AND",
                BinaryOperator.OR: "OR",
            }
            op_str = op_map[expression.operator]
            return f"(({left_sql}) {op_str} ({right_sql}))"

        args_str = ", ".join(
            self._expression_to_sql(arg, bindings) for arg in expression.arguments
        )
        window_parts: list[str] = []
        if expression.partition_by:
            parts = ", ".join(
                self._expression_to_sql(p, bindings) for p in expression.partition_by
            )
            window_parts.append(f"PARTITION BY {parts}")
        if expression.order_by:
            order_strs: list[str] = []
            for k in expression.order_by:
                expr_sql = self._expression_to_sql(k.expression, bindings)
                dir_sql = "ASC" if k.direction is SortDirection.ASCENDING else "DESC"
                null_sql = (
                    "NULLS FIRST"
                    if k.null_placement is NullPlacement.FIRST
                    else "NULLS LAST"
                )
                order_strs.append(f"{expr_sql} {dir_sql} {null_sql}")
            window_parts.append(f"ORDER BY {', '.join(order_strs)}")
        if expression.frame_spec:
            window_parts.append(expression.frame_spec)
        over_clause = " ".join(window_parts)
        return f"{expression.function}({args_str}) OVER ({over_clause})"

    def _relation_for_source(
        self,
        source: (
            ArrowSource
            | CsvSource
            | PandasSource
            | ParquetSource
            | SqlSource
            | TableSource
        ),
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
        if isinstance(source, CsvSource):
            csv_paths: str | list[str] = (
                source.paths[0] if len(source.paths) == 1 else list(source.paths)
            )
            return self._session._connection.read_csv(
                cast("str", csv_paths),
                header=source.header,
                sep=source.delimiter,
                auto_detect=source.auto_detect,
            )

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

    @staticmethod
    def _order_column_to_sql(key: OrderColumn, bindings: dict[ColumnId, str]) -> str:
        direction = "ASC" if key.direction is SortDirection.ASCENDING else "DESC"
        nulls = (
            "NULLS FIRST" if key.null_placement is NullPlacement.FIRST else "NULLS LAST"
        )
        return f"{quote_identifier(bindings[key.column_id])} {direction} {nulls}"

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
            if aggregate.input_duckdb_type == "BOOLEAN":
                bool_op = operand
            elif aggregate.input_duckdb_type == "VARCHAR":
                bool_op = duckdb.FunctionExpression("length", operand) > 0
            else:
                bool_op = operand.cast("BOOLEAN")
            func_name = (
                "bool_or" if aggregate.operator is AggregateOperator.ANY else "bool_and"
            )
            raw_val = duckdb.FunctionExpression(func_name, bool_op)
            default_val = aggregate.operator is AggregateOperator.ALL
            # When skipna is True, all-null gives False for any, True for all.
            # When skipna is False, any gives False if all null, all gives False
            # if null present.
            null_override: bool | None = None
            if not aggregate.skipna:
                floating_input = aggregate.input_duckdb_type in {"FLOAT", "DOUBLE"}
                if aggregate.operator is AggregateOperator.ANY and floating_input:
                    null_override = True
                elif aggregate.operator is AggregateOperator.ALL and not floating_input:
                    null_override = False
            if null_override is not None:
                row_count = duckdb.SQLExpression("count(*)")
                value = duckdb.CaseExpression(
                    non_null_count < row_count,
                    duckdb.ConstantExpression(null_override),
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

        if plan.sort and plan.left_keys:
            sort_keys = [
                duckdb.SQLExpression(quote_identifier(final_bindings[column_id]))
                .asc()
                .nulls_last()
                for column_id in plan.left_keys
                if column_id in final_bindings
            ]
            if sort_keys:
                result_rel = result_rel.sort(*sort_keys)

        return CompiledFrame(result_rel, final_bindings)

    def _compile_union(self, plan: UnionPlan) -> CompiledFrame:
        if not plan.inputs:
            raise ValueError("UnionPlan requires at least one input plan")

        compiled_inputs = [self.compile(inp) for inp in plan.inputs]
        target_columns = plan.metadata.columns

        projected_relations: list[duckdb.DuckDBPyRelation] = []
        for input_index, (inp_plan, compiled) in enumerate(
            zip(plan.inputs, compiled_inputs, strict=True)
        ):
            input_col_map = {col.label: col for col in inp_plan.columns}
            projections: list[duckdb.Expression] = []
            input_order_sql = ", ".join(
                self._order_column_to_sql(key, compiled.bindings)
                for key in inp_plan.metadata.ordering.keys
            )

            for target_col in target_columns:
                target_temp_name = f"c_{target_col.id.value.hex[:8]}"
                if target_col.id == plan.source_order_id:
                    expr = (
                        duckdb.ConstantExpression(input_index)
                        .cast("UBIGINT")
                        .alias(target_temp_name)
                    )
                elif target_col.id == plan.source_row_id:
                    row_sql = f"row_number() OVER (ORDER BY {input_order_sql}) - 1"
                    expr = (
                        duckdb.SQLExpression(row_sql)
                        .cast("UBIGINT")
                        .alias(target_temp_name)
                    )
                elif target_col.label in input_col_map:
                    matching_input_col = input_col_map[target_col.label]
                    source_label = quote_identifier(
                        compiled.bindings[matching_input_col.id]
                    )
                    cast_sql = f"CAST({source_label} AS {target_col.duckdb_type})"
                    expr = duckdb.SQLExpression(cast_sql).alias(target_temp_name)
                else:
                    target_type = (
                        "VARCHAR"
                        if target_col.duckdb_type == "UNKNOWN"
                        else target_col.duckdb_type
                    )
                    cast_null_sql = f"CAST(NULL AS {target_type})"
                    expr = duckdb.SQLExpression(cast_null_sql).alias(target_temp_name)
                projections.append(expr)

            projected_rel = compiled.relation.project(*projections)
            projected_relations.append(projected_rel)

        result_rel = projected_relations[0]
        for next_rel in projected_relations[1:]:
            result_rel = result_rel.union(next_rel)

        # Output projection to rename temp column names back to original target labels
        final_proj = [
            duckdb.SQLExpression(quote_identifier(f"c_{col.id.value.hex[:8]}")).alias(
                col.label
            )
            for col in target_columns
        ]
        final_bindings = {col.id: col.label for col in target_columns}
        return CompiledFrame(result_rel.project(*final_proj), final_bindings)

    def _compile_loc_index(self, plan: LocIndexPlan) -> CompiledFrame:
        compiled_input = self.compile(plan.input)
        index_ids = plan.input.metadata.index.columns
        index_cols = [compiled_input.bindings[column_id] for column_id in index_ids]

        keys_df = cast(
            "pd.DataFrame", self._session._get_registered_source(plan.source_key)
        )
        keys_rel = self._session._connection.from_df(keys_df).set_alias(
            "__duckpd_loc_keys__"
        )

        input_alias = "__duckpd_loc_input__"
        matched_label = f"__duckpd_loc_matched_{plan.source_key}__"
        flagged_input = compiled_input.relation.project(
            f"*, 1 AS {quote_identifier(matched_label)}"
        ).set_alias(input_alias)

        conditions = [
            f"__duckpd_loc_keys__.{quote_identifier(key_label)} "
            f"IS NOT DISTINCT FROM {input_alias}.{quote_identifier(index_col)}"
            for key_label, index_col in zip(plan.key_labels, index_cols, strict=True)
        ]
        joined = keys_rel.join(flagged_input, " AND ".join(conditions), how="left")

        final_projection: list[duckdb.Expression] = []
        final_bindings: dict[ColumnId, str] = {}
        for column in plan.input.columns:
            bound_label = compiled_input.bindings[column.id]
            final_projection.append(
                duckdb.SQLExpression(quote_identifier(bound_label)).alias(column.label)
            )
            final_bindings[column.id] = column.label

        order_column = next(
            column
            for column in plan.metadata.columns
            if column.id == plan.order_column_id
        )
        final_projection.append(
            duckdb.SQLExpression(quote_identifier(plan.source_order_label)).alias(
                order_column.label
            )
        )
        final_bindings[plan.order_column_id] = order_column.label

        sort_keys: list[duckdb.Expression] = [
            duckdb.SQLExpression(quote_identifier(order_column.label))
            .asc()
            .nulls_last()
        ]
        for key in plan.input.metadata.ordering.keys:
            if key.column_id in final_bindings:
                sort_keys.append(
                    self._compile_sort_key(
                        SortKey(
                            ColumnRef(key.column_id),
                            key.direction,
                            key.null_placement,
                        ),
                        final_bindings,
                    )
                )
        result = joined.project(*final_projection).sort(*sort_keys)
        return CompiledFrame(result, final_bindings)
