"""Module-level data source helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from dataclasses import replace as replace_column
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from weakref import ReferenceType, ref

import pandas as pd
import pyarrow as pa

from duckpd._logical import (
    Column,
    ColumnId,
    ColumnRef,
    Expression,
    NamedExpression,
    NullPlacement,
    OrderColumn,
    ProjectPlan,
    SortDirection,
    UnionPlan,
)
from duckpd._metadata import (
    after_projection,
    after_union,
    find_column,
    projection_columns,
)
from duckpd._reductions import expression_type
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
    sort: bool = False,
) -> DataFrame:
    """Concatenate DuckPD DataFrame or Series objects along axis 0 or axis 1."""
    from duckpd.frame import DataFrame
    from duckpd.series import Series

    if axis not in {0, 1, "index", "columns"}:
        raise ValueError("axis must be 0, 1, 'index', or 'columns'")
    if join not in {"outer", "inner"}:
        raise ValueError(
            "Only can inner (intersect) or outer (union) join the other axis"
        )

    typed_objs: list[DataFrame | Series] = []
    for item in objs:
        if isinstance(item, (DataFrame, Series)):
            typed_objs.append(item)
        else:
            raise TypeError("Objects to concat must be DataFrame or Series")

    if not typed_objs:
        raise ValueError("No objects to concatenate")

    first_session = typed_objs[0]._session
    for item in typed_objs[1:]:
        if item._session is not first_session:
            raise AlignmentError("Cannot concat frames from different sessions")

    if axis in {1, "columns"}:
        first_plan = typed_objs[0]._plan
        all_same_plan = all(item._plan is first_plan for item in typed_objs)

        if all_same_plan:
            axis1_output_columns: list[tuple[str, Expression]] = []
            unnamed_idx = 0
            for item in typed_objs:
                if isinstance(item, Series):
                    lbl = str(item.name) if item.name is not None else str(unnamed_idx)
                    if item.name is None:
                        unnamed_idx += 1
                    axis1_output_columns.append((lbl, item._expression))
                else:
                    for col in item._plan.metadata.visible_columns:
                        axis1_output_columns.append((col.label, ColumnRef(col.id)))

            labels = [lbl for lbl, _ in axis1_output_columns]
            if not ignore_index and len(labels) != len(set(labels)):
                duplicates = sorted({x for x in labels if labels.count(x) > 1})
                raise ValueError(
                    "Duplicate column labels found across concatenated objects: "
                    f"{duplicates!r}"
                )

            if ignore_index:
                axis1_output_columns = [
                    (str(i), expr) for i, (_, expr) in enumerate(axis1_output_columns)
                ]
            out_cols = [
                Column(ColumnId.create(), lbl, expression_type(first_plan, expr))
                for lbl, expr in axis1_output_columns
            ]
            all_cols = projection_columns(first_plan.metadata, tuple(out_cols))
            out_col_map = {
                col.label: expr
                for col, (_, expr) in zip(out_cols, axis1_output_columns, strict=True)
            }
            projections: list[NamedExpression] = []
            for col in all_cols:
                if col.label in out_col_map:
                    projections.append(NamedExpression(col, out_col_map[col.label]))
                else:
                    projections.append(NamedExpression(col, ColumnRef(col.id)))
            metadata = after_projection(first_plan.metadata, all_cols)
            return DataFrame(
                first_session,
                ProjectPlan(first_plan, tuple(projections), metadata),
            )

        frames: list[DataFrame] = []
        unnamed_idx = 0
        for idx, item in enumerate(typed_objs):
            if isinstance(item, Series):
                col_name = str(item.name) if item.name is not None else str(unnamed_idx)
                if item.name is None:
                    unnamed_idx += 1
                f = item.to_frame(name=col_name)
            else:
                f = item
            if ignore_index:
                mapping = {
                    c.label: f"__duckpd_concat_{idx}_{c.label}__"
                    for c in f._plan.metadata.visible_columns
                }
                f = f.rename(columns=mapping)
            frames.append(f)

        for f in frames:
            if not f._plan.metadata.index.columns:
                raise AlignmentError(
                    "Cannot concat frames along axis=1 without an explicit index; "
                    "use set_index() first to align on keys"
                )

        if not ignore_index:
            all_visible_labels = [
                col.label for f in frames for col in f._plan.metadata.visible_columns
            ]
            if len(all_visible_labels) != len(set(all_visible_labels)):
                duplicates = sorted(
                    {x for x in all_visible_labels if all_visible_labels.count(x) > 1}
                )
                raise ValueError(
                    "Duplicate column labels found across concatenated frames: "
                    f"{duplicates!r}"
                )
        join_mode: Literal["outer", "inner"] = "inner" if join == "inner" else "outer"
        result = frames[0]
        for f in frames[1:]:
            result = result.join(f, how=join_mode, sort=sort)

        if ignore_index:
            new_cols: list[Column] = []
            counter = 0
            for col in result._plan.metadata.columns:
                if col.hidden:
                    new_cols.append(col)
                else:
                    new_cols.append(replace_column(col, label=str(counter)))
                    counter += 1
            new_meta = replace_column(
                result._plan.metadata,
                columns=tuple(new_cols),
            )
            result = result._identity_project(new_meta)

        return result

    if join != "outer":
        raise UnsupportedOperationError(
            "DuckPD concat along axis=0 currently supports only join='outer'"
        )

    # Axis 0 (row-wise union)
    frames = []
    for item in typed_objs:
        if isinstance(item, Series):
            col_name = item.name if item.name is not None else 0
            frames.append(DataFrame(item._session, item._plan)[[str(col_name)]])
        else:
            frames.append(item)
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
                ),
                Column(
                    source_row_id,
                    f"__duckpd_union_row_{source_row_id.value.hex}__",
                    "UBIGINT",
                    hidden=True,
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
        identity_ids=(
            (source_order_id, source_row_id)
            if source_order_id is not None and source_row_id is not None
            else ()
        ),
    )
    plan = UnionPlan(
        input_plans,
        metadata,
        source_order_id=source_order_id,
        source_row_id=source_row_id,
    )
    return DataFrame(first_session, plan)
