"""Metadata construction, validation, and plan transitions."""

from __future__ import annotations

from dataclasses import replace

from duckpd._logical import (
    Column,
    ColumnId,
    ColumnRef,
    FrameMetadata,
    IndexSpec,
    NullPlacement,
    OrderColumn,
    OrderSpec,
    SortDirection,
    SortKey,
)


def find_column(
    metadata: FrameMetadata, label: str, *, include_hidden: bool = False
) -> Column:
    """Find a column by displayed label."""
    for column in metadata.columns:
        if column.label == label and (include_hidden or not column.hidden):
            return column
    raise KeyError(label)


def source_metadata(
    columns: tuple[Column, ...],
    *,
    index_labels: tuple[str, ...] = (),
) -> FrameMetadata:
    """Create source metadata and hide explicit index columns."""
    base = FrameMetadata(columns)
    if not index_labels:
        return base
    index_columns = tuple(find_column(base, label) for label in index_labels)
    index_ids = tuple(column.id for column in index_columns)
    updated = tuple(
        replace(column, hidden=True) if column.id in index_ids else column
        for column in columns
    )
    return FrameMetadata(updated, IndexSpec(index_ids, drop=True), OrderSpec())


def projection_columns(
    metadata: FrameMetadata, selected: tuple[Column, ...]
) -> tuple[Column, ...]:
    """Append hidden columns required to preserve index and ordering metadata."""
    selected_ids = {column.id for column in selected}
    required_ids = set(metadata.index.columns)
    required_ids.update(key.column_id for key in metadata.ordering.keys)
    hidden = tuple(
        replace(column, hidden=True)
        for column in metadata.columns
        if column.id in required_ids and column.id not in selected_ids
    )
    return (*selected, *hidden)


def after_projection(
    metadata: FrameMetadata, columns: tuple[Column, ...]
) -> FrameMetadata:
    """Carry only metadata whose physical columns remain available."""
    available = {column.id for column in columns}
    index = (
        metadata.index
        if all(column_id in available for column_id in metadata.index.columns)
        else IndexSpec()
    )
    ordering = (
        metadata.ordering
        if all(key.column_id in available for key in metadata.ordering.keys)
        else OrderSpec()
    )
    result = FrameMetadata(columns, index, ordering)
    validate_metadata(result)
    return result


def after_sort(metadata: FrameMetadata, keys: tuple[SortKey, ...]) -> FrameMetadata:
    """Establish guaranteed ordering for simple column sort keys."""
    order_keys: list[OrderColumn] = []
    for key in keys:
        if not isinstance(key.expression, ColumnRef):
            return FrameMetadata(metadata.columns, metadata.index, OrderSpec())
        order_keys.append(
            OrderColumn(
                key.expression.column_id,
                key.direction,
                key.null_placement,
            )
        )
    result = FrameMetadata(
        metadata.columns,
        metadata.index,
        OrderSpec(tuple(order_keys)),
    )
    validate_metadata(result)
    return result


def after_aggregate(columns: tuple[Column, ...]) -> FrameMetadata:
    """Create metadata for a global aggregate with no inherited index or order."""
    result = FrameMetadata(columns)
    validate_metadata(result)
    return result


def set_index(
    metadata: FrameMetadata, columns: tuple[Column, ...], *, drop: bool
) -> FrameMetadata:
    """Replace the explicit index without requiring uniqueness."""
    index_ids = tuple(column.id for column in columns)
    updated = tuple(
        replace(column, hidden=True)
        if drop and column.id in index_ids
        else column
        for column in metadata.columns
    )
    result = FrameMetadata(
        updated,
        IndexSpec(index_ids, drop=drop),
        metadata.ordering,
    )
    validate_metadata(result)
    return result


def reset_index(metadata: FrameMetadata, *, drop: bool) -> FrameMetadata:
    """Remove the explicit index and optionally restore its hidden columns."""
    if not metadata.index.columns:
        raise ValueError("DataFrame has no explicit index to reset")
    index_ids = set(metadata.index.columns)
    index_columns = tuple(
        column for column in metadata.columns if column.id in index_ids
    )
    other_columns = tuple(
        column for column in metadata.columns if column.id not in index_ids
    )
    if drop:
        columns = tuple(
            column
            for column in metadata.columns
            if column.id not in index_ids or not column.hidden
        )
    else:
        if any(not column.hidden for column in index_columns):
            raise ValueError("cannot insert an index column that already exists")
        columns = (
            *(replace(column, hidden=False) for column in index_columns),
            *other_columns,
        )
    result = FrameMetadata(columns, IndexSpec(), metadata.ordering)
    validate_metadata(result)
    return result


def protected_column_ids(metadata: FrameMetadata) -> frozenset[ColumnId]:
    """Return columns whose replacement would invalidate frame metadata."""
    values = set(metadata.index.columns)
    values.update(key.column_id for key in metadata.ordering.keys)
    return frozenset(values)


def sort_keys_for_labels(
    metadata: FrameMetadata, labels: tuple[str, ...]
) -> tuple[SortKey, ...]:
    """Create ascending, nulls-last sort keys for source order declarations."""
    return tuple(
        SortKey(
            ColumnRef(find_column(metadata, label, include_hidden=True).id),
            SortDirection.ASCENDING,
            NullPlacement.LAST,
        )
        for label in labels
    )


def validate_metadata(metadata: FrameMetadata) -> None:
    """Reject metadata that references columns absent from the schema."""
    available = {column.id for column in metadata.columns}
    dangling = set(metadata.index.columns) - available
    dangling.update(
        key.column_id
        for key in metadata.ordering.keys
        if key.column_id not in available
    )
    if dangling:
        raise AssertionError(f"Metadata references missing columns: {dangling}")