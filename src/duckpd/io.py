"""Module-level data source helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import ReferenceType, ref

import pandas as pd
import pyarrow as pa

from duckpd._logical import (
    Column,
    ColumnId,
    UnionPlan,
)
from duckpd._metadata import after_union, find_column
from duckpd.errors import AlignmentError, UnsupportedOperationError
from duckpd.session import Session

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckpd.frame import DataFrame


_implicit_session: ContextVar[ReferenceType[Session] | None] = ContextVar(
    "duckpd_implicit_session", default=None
)


def _get_implicit_session() -> Session:
    reference = _implicit_session.get()
    session = reference() if reference is not None else None
    if session is None or session.closed:
        session = Session()
        _implicit_session.set(ref(session))
    return session


def from_pandas(
    value: pd.DataFrame,
    *,
    session: Session | None = None,
    index: str | Sequence[str] | None = None,
    order_by: str | Sequence[str] | None = None,
) -> DataFrame:
    """Create a lazy DuckPD frame from a pandas DataFrame."""
    owner = session if session is not None else _get_implicit_session()
    return owner.from_pandas(value, index=index, order_by=order_by)


def from_arrow(
    value: pa.Table | pa.RecordBatch,
    *,
    session: Session | None = None,
    index: str | Sequence[str] | None = None,
    order_by: str | Sequence[str] | None = None,
) -> DataFrame:
    """Create a lazy DuckPD frame from an Arrow table or record batch."""
    owner = session if session is not None else _get_implicit_session()
    return owner.from_arrow(value, index=index, order_by=order_by)


def read_parquet(
    path: str | Path | Sequence[str | Path],
    *,
    session: Session | None = None,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    index: str | Sequence[str] | None = None,
    order_by: str | Sequence[str] | None = None,
) -> DataFrame:
    """Create a lazy DuckPD frame from one or more Parquet files."""
    owner = session if session is not None else _get_implicit_session()
    return owner.read_parquet(
        path,
        hive_partitioning=hive_partitioning,
        union_by_name=union_by_name,
        index=index,
        order_by=order_by,
    )


def read_csv(
    path: str | Path | Sequence[str | Path],
    *,
    session: Session | None = None,
    header: bool = True,
    delimiter: str = ",",
    auto_detect: bool = True,
    index: str | Sequence[str] | None = None,
    order_by: str | Sequence[str] | None = None,
) -> DataFrame:
    """Create a lazy DuckPD frame from one or more CSV files."""
    owner = session if session is not None else _get_implicit_session()
    return owner.read_csv(
        path,
        header=header,
        delimiter=delimiter,
        auto_detect=auto_detect,
        index=index,
        order_by=order_by,
    )


def concat(
    objs: Iterable[object],
    *,
    axis: int | str = 0,
    join: str = "outer",
    ignore_index: bool = False,
) -> DataFrame:
    """Concatenate DuckPD DataFrame or Series objects along axis 0."""
    from duckpd.frame import DataFrame
    from duckpd.series import Series

    if axis not in {0, "index"}:
        raise UnsupportedOperationError(
            "DuckPD concat currently supports only axis=0 or axis='index'"
        )
    if join != "outer":
        raise UnsupportedOperationError(
            "DuckPD concat currently supports only join='outer'"
        )

    obj_list = list(objs)
    if not obj_list:
        raise ValueError("No objects to concatenate")

    # Coerce any Series to single-column DataFrames
    frames: list[DataFrame] = []
    for item in obj_list:
        if isinstance(item, Series):
            col_name = item.name if item.name is not None else 0
            frames.append(DataFrame(item._session, item._plan)[[str(col_name)]])
        elif isinstance(item, DataFrame):
            frames.append(item)
        else:
            raise TypeError("Objects to concat must be DataFrame or Series")

    first_session = frames[0]._session
    for f in frames[1:]:
        if f._session is not first_session:
            raise AlignmentError("Cannot concat frames from different sessions")

    # Reconcile visible columns
    # Collect columns in appearance order
    seen_labels: dict[str, str] = {}  # label -> duckdb_type
    for f in frames:
        for col in f._plan.metadata.visible_columns:
            if col.label not in seen_labels:
                seen_labels[col.label] = col.duckdb_type
            else:
                # Type promotion if needed (e.g. integer + float -> DOUBLE)
                curr_type = seen_labels[col.label]
                if curr_type != col.duckdb_type:
                    numeric_set = {
                        "TINYINT",
                        "SMALLINT",
                        "INTEGER",
                        "BIGINT",
                        "HUGEINT",
                        "FLOAT",
                        "DOUBLE",
                    }
                    if curr_type in numeric_set and col.duckdb_type in numeric_set:
                        seen_labels[col.label] = "DOUBLE"
                    else:
                        seen_labels[col.label] = "VARCHAR"

    # Check index preservation across all frames
    # If not ignore_index and all frames have matching index structure:
    index_ids: list[ColumnId] = []
    hidden_index_cols: list[Column] = []
    if not ignore_index:
        first_index_labels = frames[0].index_names
        all_match_index = all(f.index_names == first_index_labels for f in frames)
        if all_match_index and first_index_labels:
            for idx_label in first_index_labels:
                col_type = find_column(
                    frames[0]._plan.metadata, idx_label, include_hidden=True
                ).duckdb_type
                idx_col = Column(ColumnId.create(), idx_label, col_type, hidden=True)
                hidden_index_cols.append(idx_col)
                index_ids.append(idx_col.id)

    # Build target output columns: visible columns + hidden index columns
    output_columns = [
        Column(ColumnId.create(), label, dtype, hidden=False)
        for label, dtype in seen_labels.items()
    ]
    output_columns.extend(hidden_index_cols)

    input_plans = tuple(f._plan for f in frames)
    metadata = after_union(tuple(output_columns), index_ids=tuple(index_ids))
    plan = UnionPlan(input_plans, metadata)
    return DataFrame(first_session, plan)
