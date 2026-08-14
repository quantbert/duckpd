"""The only layer that triggers DuckDB result production."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa

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
        result = compiled.relation.to_df()
        # Normalize DuckDB nullable integer columns to float64 if they contain nulls
        for col in result.columns:
            dtype_str = str(result[col].dtype)
            if (dtype_str.startswith("Int") or dtype_str.startswith("UInt")) and result[
                col
            ].isna().any():
                result[col] = result[col].astype("float64")

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
