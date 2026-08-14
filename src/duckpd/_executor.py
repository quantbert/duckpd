"""The only layer that triggers DuckDB result production."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from duckpd._typing import ParquetCompression

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
        self._session._begin_execution()
        compiled = self._compiler.compile(plan)
        result = compiled.relation.to_df()
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
        self._session._begin_execution()
        compiled = self._compiler.compile(plan)
        return self._compiler.project_visible(compiled, plan).relation.to_arrow_table()

    def to_arrow_batches(
        self, plan: LogicalPlan, *, batch_size: int
    ) -> pa.RecordBatchReader:
        self._session._begin_execution()
        compiled = self._compiler.compile(plan)
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
        self._session._begin_execution()
        compiled = self._compiler.compile(plan)
        self._compiler.project_visible(compiled, plan).relation.write_parquet(
            path,
            compression=compression,
            overwrite=overwrite,
        )

    def explain(self, plan: LogicalPlan) -> str:
        self._session._begin_execution()
        relation = self._compiler.compile(plan).relation
        return (
            f"DuckPD logical plan:\n{plan!r}\n\n"
            f"DuckDB SQL:\n{relation.sql_query()}\n\n"
            f"DuckDB physical plan:\n{relation.explain()}"
        )
