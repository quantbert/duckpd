"""The only layer that triggers DuckDB result production."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Literal, cast

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from duckpd._logical import (
    AggregatePlan,
    ColumnId,
    ColumnRef,
    FilterPlan,
    JoinPlan,
    JoinType,
    LimitPlan,
    LocIndexPlan,
    PandasSource,
    ProjectPlan,
    ScanPlan,
    SortKey,
    SortPlan,
    UnionPlan,
)
from duckpd._quoting import quote_identifier
from duckpd._typing import ParquetCompression
from duckpd.errors import MaterializationError, MergeError

if TYPE_CHECKING:
    from duckpd._compiler import DuckDBCompiler
    from duckpd._logical import LogicalPlan
    from duckpd.session import Session


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return Decimal(str(value))


def _bytes_or_none(value: object) -> bytes | None:
    if value is None or value is pd.NA:
        return None
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"Expected a binary value, got {type(value).__name__}")
    return bytes(value)


class Executor:
    """Execute compiled plans and track observable execution boundaries."""

    def __init__(self, session: Session, compiler: DuckDBCompiler) -> None:
        self._session = session
        self._compiler = compiler

    def collect(self, plan: LogicalPlan) -> pd.DataFrame:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        rel = compiled.relation
        if plan.metadata.ordering.keys:
            order_exprs = [
                self._compiler._compile_sort_key(
                    SortKey(
                        ColumnRef(k.column_id),
                        k.direction,
                        k.null_placement,
                    ),
                    compiled.bindings,
                )
                for k in plan.metadata.ordering.keys
                if k.column_id in compiled.bindings
            ]
            if order_exprs:
                rel = rel.sort(*order_exprs)
        decimal_labels = {
            compiled.bindings[column.id]
            for column in plan.metadata.columns
            if column.duckdb_type.startswith("DECIMAL(")
            and column.id in compiled.bindings
        }
        if decimal_labels:
            rel = rel.project(
                *(
                    duckdb.SQLExpression(quote_identifier(label))
                    .cast("VARCHAR")
                    .alias(label)
                    if label in decimal_labels
                    else duckdb.SQLExpression(quote_identifier(label))
                    for label in compiled.bindings.values()
                )
            )
        result = rel.to_df()
        type_by_label = {
            compiled.bindings[column.id]: column.duckdb_type
            for column in plan.metadata.columns
            if column.id in compiled.bindings
        }
        for label, duckdb_type in type_by_label.items():
            if duckdb_type.startswith("DECIMAL("):
                result[label] = result[label].map(_decimal_or_none)
            elif duckdb_type == "BLOB":
                result[label] = result[label].map(_bytes_or_none)
            elif duckdb_type == "DATE":
                result[label] = result[label].map(
                    lambda value: (
                        None
                        if value is None or value is pd.NaT
                        else cast("pd.Timestamp", value).date()
                    )
                )
        preserved_ids = self._pandas_nullable_integer_ids(plan)
        preserved_labels = {
            compiled.bindings[column_id]
            for column_id in preserved_ids
            if column_id in compiled.bindings
        }
        for label in result.columns:
            dtype_name = str(result[label].dtype)
            if (
                label not in preserved_labels
                and dtype_name.startswith(("Int", "UInt"))
                and result[label].isna().any()
            ):
                result[label] = result[label].astype("float64")

        index_ids = plan.metadata.index.columns
        if index_ids:
            index_labels = [compiled.bindings[column_id] for column_id in index_ids]
            result = result.set_index(index_labels, drop=plan.metadata.index.drop)
        hidden_labels = [
            compiled.bindings[column.id]
            for column in plan.metadata.columns
            if column.hidden and column.id not in index_ids
        ]
        if hidden_labels:
            result = result.drop(columns=hidden_labels)
        return result

    def _pandas_nullable_integer_ids(self, plan: LogicalPlan) -> set[ColumnId]:
        if isinstance(plan, ScanPlan):
            if not isinstance(plan.source, PandasSource):
                return set()
            source = self._session._get_registered_source(plan.source.key)
            if not isinstance(source, pd.DataFrame):
                raise TypeError("Registered pandas source must be a DataFrame")
            return {
                column.id
                for column in plan.metadata.columns
                if str(source[column.label].dtype).startswith(("Int", "UInt"))
            }
        if isinstance(plan, JoinPlan):
            return self._pandas_nullable_integer_ids(
                plan.left
            ) | self._pandas_nullable_integer_ids(plan.right)
        if isinstance(plan, UnionPlan):
            nullable_labels: set[str] = set()
            for input_plan in plan.inputs:
                nullable_ids = self._pandas_nullable_integer_ids(input_plan)
                nullable_labels.update(
                    column.label
                    for column in input_plan.metadata.columns
                    if column.id in nullable_ids
                )
            pandas_nullable_types = {
                "TINYINT",
                "SMALLINT",
                "INTEGER",
                "BIGINT",
                "UTINYINT",
                "USMALLINT",
                "UINTEGER",
                "UBIGINT",
            }
            return {
                column.id
                for column in plan.metadata.columns
                if column.label in nullable_labels
                and column.duckdb_type in pandas_nullable_types
            }
        if isinstance(plan, LocIndexPlan):
            return self._pandas_nullable_integer_ids(plan.input)
        return self._pandas_nullable_integer_ids(plan.input)

    def to_arrow(self, plan: LogicalPlan) -> pa.Table:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        return self._compiler.project_visible(compiled, plan).relation.to_arrow_table()

    def to_arrow_batches(
        self, plan: LogicalPlan, *, batch_size: int
    ) -> pa.RecordBatchReader:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        return self._compiler.project_visible(compiled, plan).relation.to_arrow_reader(
            batch_size
        )

    def write_parquet(
        self,
        plan: LogicalPlan,
        path: str,
        *,
        compression: ParquetCompression,
        overwrite: bool,
    ) -> None:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        self._compiler.project_visible(compiled, plan).relation.write_parquet(
            path,
            compression=compression,
            overwrite=overwrite,
        )

    def write_csv(
        self,
        plan: LogicalPlan,
        path: str,
        *,
        sep: str = ",",
        header: bool = True,
    ) -> None:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        self._compiler.project_visible(compiled, plan).relation.write_csv(
            path,
            sep=sep,
            header=header,
        )

    def persist(
        self,
        plan: LogicalPlan,
        name: str,
    ) -> None:
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        compiled.relation.create(name)

    def explain(
        self,
        plan: LogicalPlan,
        *,
        mode: Literal["all", "logical", "sql", "physical"] = "all",
    ) -> str:
        compiled = self._compiler.compile(plan)
        relation = compiled.relation
        self._session._begin_execution()
        if mode == "logical":
            return f"DuckPD logical plan:\n{plan!r}"
        if mode == "sql":
            return f"DuckDB SQL:\n{relation.sql_query()}"
        if mode == "physical":
            return f"DuckDB physical plan:\n{relation.explain()}"
        if mode == "all":
            return (
                f"DuckPD logical plan:\n{plan!r}\n\n"
                f"DuckDB SQL:\n{relation.sql_query()}\n\n"
                f"DuckDB physical plan:\n{relation.explain()}"
            )
        msg = (
            f"Unknown explain mode: {mode!r}; "
            "expected 'all', 'logical', 'sql', or 'physical'"
        )
        raise ValueError(msg)

    def explain_write(
        self,
        plan: LogicalPlan,
        path: str,
        *,
        compression: ParquetCompression = "snappy",
    ) -> str:
        """Inspect write strategy and execution plan without writing rows."""
        compiled = self._compiler.compile(plan)
        visible_rel = self._compiler.project_visible(compiled, plan).relation
        self._session._begin_execution()
        return (
            f"Write target: {path}\n"
            f"Compression: {compression}\n"
            f"Output columns: {list(plan.metadata.visible_columns)}\n"
            f"DuckDB physical plan:\n{visible_rel.explain()}"
        )

    def reduce_scalar(self, plan: LogicalPlan) -> object:
        """Execute a one-column, one-row aggregate plan."""
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        if len(plan.metadata.visible_columns) != 1:
            raise MaterializationError("Scalar reduction requires one output column")
        self._session._begin_execution()
        result = compiled.relation.to_df()
        if result.shape != (1, 1):
            raise MaterializationError("Scalar reduction did not produce one value")
        value = cast("object", result.iloc[0, 0])
        return np.nan if value is None else value

    def reduce_columns(self, plan: LogicalPlan) -> pd.Series:
        """Execute a one-row aggregate plan as a label-indexed pandas Series."""
        self._validate_execution(plan)
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        result = compiled.relation.to_df()
        if result.shape != (1, len(plan.metadata.visible_columns)):
            raise MaterializationError("Column reduction did not produce one row")
        reduced = result.iloc[0]
        reduced.index = [column.label for column in plan.metadata.visible_columns]
        reduced.name = None
        if reduced.isna().all():
            return pd.Series(np.nan, index=reduced.index, dtype="float64")
        if reduced.isna().any():
            reduced = reduced.map(lambda value: np.nan if value is None else value)
        return reduced.infer_objects()

    def _validate_execution(self, plan: LogicalPlan) -> None:
        self._validate_cardinalities(plan)
        self._validate_loc_indices(plan)

    def _validate_cardinalities(self, plan: LogicalPlan) -> None:
        joins = self._find_join_plans(plan)
        for join in joins:
            if join.validate and join.validate not in {"m:m", "many_to_many"}:
                self._validate_join(join)

    def _validate_loc_indices(self, plan: LogicalPlan) -> None:
        locs = self._find_loc_plans(plan)
        for loc in locs:
            self._validate_loc_plan(loc)

    def _find_join_plans(self, plan: LogicalPlan) -> list[JoinPlan]:
        joins: list[JoinPlan] = []
        if isinstance(plan, JoinPlan):
            joins.extend(self._find_join_plans(plan.left))
            joins.extend(self._find_join_plans(plan.right))
            joins.append(plan)
        elif isinstance(plan, UnionPlan):
            for input_plan in plan.inputs:
                joins.extend(self._find_join_plans(input_plan))
        elif isinstance(
            plan,
            (
                FilterPlan,
                ProjectPlan,
                SortPlan,
                LimitPlan,
                AggregatePlan,
                LocIndexPlan,
            ),
        ):
            joins.extend(self._find_join_plans(plan.input))
        return joins

    def _find_loc_plans(self, plan: LogicalPlan) -> list[LocIndexPlan]:
        locs: list[LocIndexPlan] = []
        if isinstance(plan, LocIndexPlan):
            locs.extend(self._find_loc_plans(plan.input))
            locs.append(plan)
        elif isinstance(plan, JoinPlan):
            locs.extend(self._find_loc_plans(plan.left))
            locs.extend(self._find_loc_plans(plan.right))
        elif isinstance(plan, UnionPlan):
            for input_plan in plan.inputs:
                locs.extend(self._find_loc_plans(input_plan))
        elif isinstance(
            plan, (FilterPlan, ProjectPlan, SortPlan, LimitPlan, AggregatePlan)
        ):
            locs.extend(self._find_loc_plans(plan.input))
        return locs

    def _validate_loc_plan(self, plan: LocIndexPlan) -> None:
        compiled_input = self._compiler.compile(plan.input)
        index_ids = plan.input.metadata.index.columns
        index_cols = [compiled_input.bindings[cid] for cid in index_ids]

        keys_df = cast(
            "pd.DataFrame", self._session._get_registered_source(plan.source_key)
        )
        keys_col_map = {col: f"_loc_k_{i}" for i, col in enumerate(index_cols)}
        keys_df_renamed = keys_df.rename(columns=keys_col_map)
        keys_rel = self._session._connection.from_df(keys_df_renamed).set_alias(
            "__duckpd_loc_keys__"
        )

        input_alias = "__duckpd_loc_inp__"
        flagged_input = compiled_input.relation.project(
            "*, 1 AS __duckpd_matched__"
        ).set_alias(input_alias)

        cond_parts = [
            f"__duckpd_loc_keys__._loc_k_{i} IS NOT DISTINCT FROM "
            f"{input_alias}.{quote_identifier(col)}"
            for i, col in enumerate(index_cols)
        ]
        joined = keys_rel.join(flagged_input, " AND ".join(cond_parts), how="left")
        self._session._begin_execution()

        key_proj = ", ".join(f"_loc_k_{i}" for i in range(len(index_ids)))
        missing_rows = (
            joined.filter("__duckpd_matched__ IS NULL")
            .project(key_proj)
            .limit(1)
            .fetchall()
        )
        if missing_rows:
            row = missing_rows[0]
            missing_val = row[0] if len(row) == 1 else row
            raise KeyError(f"[{missing_val!r}] not in index")

    def _validate_join(self, join: JoinPlan) -> None:
        check_left = join.validate in {"1:1", "1:m", "one_to_one", "one_to_many"}
        check_right = join.validate in {"1:1", "m:1", "one_to_one", "many_to_one"}
        relationship = (
            "one-to-one"
            if join.validate in {"1:1", "one_to_one"}
            else "one-to-many"
            if join.validate in {"1:m", "one_to_many"}
            else "many-to-one"
        )

        if check_left:
            self._check_uniqueness(
                join.left,
                join.left_keys,
                side="left",
                relationship=relationship,
                is_cross=join.how is JoinType.CROSS,
            )
        if check_right:
            self._check_uniqueness(
                join.right,
                join.right_keys,
                side="right",
                relationship=relationship,
                is_cross=join.how is JoinType.CROSS,
            )

    def _check_uniqueness(
        self,
        input_plan: LogicalPlan,
        key_ids: tuple[ColumnId, ...],
        *,
        side: str,
        relationship: str,
        is_cross: bool,
    ) -> None:
        compiled = self._compiler.compile(input_plan)
        self._session._begin_execution()
        if is_cross:
            limit_rel = compiled.relation.limit(2)
            if len(limit_rel.fetchall()) > 1:
                raise MergeError(
                    f"Merge keys are not unique in {side} dataset; "
                    f"not a {relationship} merge"
                )
            return

        key_cols = [quote_identifier(compiled.bindings[k_id]) for k_id in key_ids]
        group_keys = ", ".join(key_cols)
        dup_check = (
            compiled.relation.aggregate("COUNT(*) AS __duckpd_dup_cnt__", group_keys)
            .filter("__duckpd_dup_cnt__ > 1")
            .limit(1)
        )
        if len(dup_check.fetchall()) > 0:
            raise MergeError(
                f"Merge keys are not unique in {side} dataset; "
                f"not a {relationship} merge"
            )
