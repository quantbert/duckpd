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
    NullPlacement,
    OrderColumn,
    SortDirection,
    UnionPlan,
)
from duckpd._metadata import after_union, find_column
from duckpd._typing import common_union_type
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

    types_by_label: dict[str, list[str]] = {}
    for frame in frames:
        for column in frame._plan.metadata.visible_columns:
            types_by_label.setdefault(column.label, []).append(column.duckdb_type)

    try:
        reconciled_types = {
            label: common_union_type(types) for label, types in types_by_label.items()
        }
    except TypeError as error:
        raise UnsupportedOperationError(str(error)) from error

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

    output_columns = [
        Column(ColumnId.create(), label, dtype, hidden=False)
        for label, dtype in reconciled_types.items()
    ]
    output_columns.extend(hidden_index_cols)

    input_plans = tuple(f._plan for f in frames)
    source_order_id: ColumnId | None = None
    source_row_id: ColumnId | None = None
    ordering_keys: tuple[OrderColumn, ...] = ()
    if all(plan.metadata.ordering.keys for plan in input_plans):
        source_order_id = ColumnId.create()
        source_row_id = ColumnId.create()
        output_columns.extend(
            (
                Column(
                    source_order_id,
                    f"__duckpd_union_source_{source_order_id.value.hex}__",
                    "UBIGINT",
                    hidden=True,
                    row_identity=True,
                ),
                Column(
                    source_row_id,
                    f"__duckpd_union_row_{source_row_id.value.hex}__",
                    "UBIGINT",
                    hidden=True,
                    row_identity=True,
                ),
            )
        )
        ordering_keys = (
            OrderColumn(source_order_id, SortDirection.ASCENDING, NullPlacement.LAST),
            OrderColumn(source_row_id, SortDirection.ASCENDING, NullPlacement.LAST),
        )

    metadata = after_union(
        tuple(output_columns),
        index_ids=tuple(index_ids),
        ordering_keys=ordering_keys,
    )
    plan = UnionPlan(
        input_plans,
        metadata,
        source_order_id=source_order_id,
        source_row_id=source_row_id,
    )
    return DataFrame(first_session, plan)
