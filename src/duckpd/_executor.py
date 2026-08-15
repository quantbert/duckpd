"""The only layer that triggers DuckDB result production."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from duckpd._logical import (
    ColumnId,
    ColumnRef,
    JoinPlan,
    PandasSource,
    ScanPlan,
    SortKey,
    UnionPlan,
)
from duckpd._typing import ParquetCompression
from duckpd.errors import MaterializationError

if TYPE_CHECKING:
    from duckpd._compiler import DuckDBCompiler
    from duckpd._logical import LogicalPlan
    from duckpd.session import Session


class Executor:
    """Execute compiled plans and track observable execution boundaries."""

    def __init__(self, session: Session, compiler: DuckDBCompiler) -> None:
        self._session = session
        self._compiler = compiler

    def collect(self, plan: LogicalPlan) -> pd.DataFrame:
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
        result = rel.to_df()
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
            preserved: set[ColumnId] = set()
            for input_plan in plan.inputs:
                preserved.update(self._pandas_nullable_integer_ids(input_plan))
            return preserved
        return self._pandas_nullable_integer_ids(plan.input)

    def to_arrow(self, plan: LogicalPlan) -> pa.Table:
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        return self._compiler.project_visible(compiled, plan).relation.to_arrow_table()

    def to_arrow_batches(
        self, plan: LogicalPlan, *, batch_size: int
    ) -> pa.RecordBatchReader:
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
        compiled = self._compiler.compile(plan)
        self._session._begin_execution()
        self._compiler.project_visible(compiled, plan).relation.create(name)

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
