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
    RowIdentity,
    SortDirection,
    SortKey,
    SourceProvenance,
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
    provenance: SourceProvenance | None = None,
) -> FrameMetadata:
    """Create source metadata and hide explicit index columns."""
    provenance = provenance or SourceProvenance()
    base = FrameMetadata(
        columns,
        provenance=provenance,
    )
    if not index_labels:
        return base
    index_columns = tuple(find_column(base, label) for label in index_labels)
    index_ids = tuple(column.id for column in index_columns)
    updated = tuple(
        replace(column, hidden=True) if column.id in index_ids else column
        for column in columns
    )
    return replace(base, columns=updated, index=IndexSpec(index_ids, drop=True))


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
    identity = (
        metadata.row_identity
        if all(column_id in available for column_id in metadata.row_identity.columns)
        else RowIdentity()
    )
    result = FrameMetadata(
        columns,
        index,
        ordering,
        identity,
        _after_transform(metadata.provenance, "project"),
    )
    validate_metadata(result)
    return result


def after_sort(metadata: FrameMetadata, keys: tuple[SortKey, ...]) -> FrameMetadata:
    """Establish guaranteed ordering for simple column sort keys."""
    order_keys: list[OrderColumn] = []
    for key in keys:
        if not isinstance(key.expression, ColumnRef):
            return replace(
                metadata,
                ordering=OrderSpec(),
                provenance=_after_transform(metadata.provenance, "sort"),
            )
        order_keys.append(
            OrderColumn(
                key.expression.column_id,
                key.direction,
                key.null_placement,
            )
        )
    result = replace(
        metadata,
        ordering=OrderSpec(tuple(order_keys)),
        provenance=_after_transform(metadata.provenance, "sort"),
    )
    validate_metadata(result)
    return result


def after_aggregate(
    columns: tuple[Column, ...],
    *,
    index_ids: tuple[ColumnId, ...] = (),
    ordering_keys: tuple[OrderColumn, ...] = (),
) -> FrameMetadata:
    """Create metadata for an aggregate plan with optional index and order."""
    index = IndexSpec(index_ids, drop=True) if index_ids else IndexSpec()
    ordering = OrderSpec(ordering_keys) if ordering_keys else OrderSpec()
    result = FrameMetadata(columns, index, ordering)
    validate_metadata(result)
    return result


def set_index(
    metadata: FrameMetadata, columns: tuple[Column, ...], *, drop: bool
) -> FrameMetadata:
    """Replace the explicit index without requiring uniqueness."""
    index_ids = tuple(column.id for column in columns)
    updated = tuple(
        replace(column, hidden=True) if drop and column.id in index_ids else column
        for column in metadata.columns
    )
    result = replace(
        metadata,
        columns=updated,
        index=IndexSpec(index_ids, drop=drop),
        provenance=_after_transform(metadata.provenance, "set_index"),
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
    available = {column.id for column in columns}
    ordering = (
        metadata.ordering
        if all(key.column_id in available for key in metadata.ordering.keys)
        else OrderSpec()
    )
    result = replace(
        metadata,
        columns=columns,
        index=IndexSpec(),
        ordering=ordering,
        provenance=_after_transform(metadata.provenance, "reset_index"),
    )
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


def after_join(
    columns: tuple[Column, ...],
    *,
    index_ids: tuple[ColumnId, ...] = (),
    ordering_keys: tuple[OrderColumn, ...] = (),
) -> FrameMetadata:
    """Create metadata for a join plan."""
    index = IndexSpec(index_ids, drop=True) if index_ids else IndexSpec()
    ordering = OrderSpec(ordering_keys) if ordering_keys else OrderSpec()
    result = FrameMetadata(columns, index, ordering)
    validate_metadata(result)
    return result


def after_union(
    columns: tuple[Column, ...],
    *,
    index_ids: tuple[ColumnId, ...] = (),
    index_names: tuple[str | None, ...] = (),
    ordering_keys: tuple[OrderColumn, ...] = (),
    identity_ids: tuple[ColumnId, ...] = (),
) -> FrameMetadata:
    """Create metadata for a union (concat) plan."""
    index = (
        IndexSpec(index_ids, drop=True, names=index_names) if index_ids else IndexSpec()
    )
    identity_ids = tuple(identity_ids)
    result = FrameMetadata(
        columns,
        index,
        OrderSpec(ordering_keys),
        RowIdentity(
            identity_ids,
            stable=bool(identity_ids),
            unique=bool(identity_ids),
        ),
    )
    validate_metadata(result)
    return result


def after_filter(metadata: FrameMetadata) -> FrameMetadata:
    """Preserve identity while recording that source rows were filtered."""
    return replace(
        metadata,
        provenance=_after_transform(
            metadata.provenance,
            "filter",
            row_preserving=False,
        ),
    )


def after_reindex(
    metadata: FrameMetadata,
    columns: tuple[Column, ...],
    ordering: OrderSpec,
    row_identity: RowIdentity,
) -> FrameMetadata:
    """Track stable request order without claiming duplicated rows are unique."""
    result = replace(
        metadata,
        columns=columns,
        ordering=ordering,
        row_identity=row_identity,
        provenance=_after_transform(
            metadata.provenance,
            "reindex",
            row_preserving=False,
        ),
    )
    validate_metadata(result)
    return result


def _after_transform(
    provenance: SourceProvenance,
    operation: str,
    *,
    row_preserving: bool = True,
) -> SourceProvenance:
    if not provenance.locations:
        return provenance
    return replace(
        provenance,
        row_preserving=provenance.row_preserving and row_preserving,
        transformations=(*provenance.transformations, operation),
    )


def validate_metadata(metadata: FrameMetadata) -> None:
    """Reject metadata that references columns absent from the schema."""
    available = {column.id for column in metadata.columns}
    if metadata.index.names and len(metadata.index.names) != len(
        metadata.index.columns
    ):
        raise AssertionError("Index names must match the number of index columns")
    dangling = set(metadata.index.columns) - available
    dangling.update(
        key.column_id
        for key in metadata.ordering.keys
        if key.column_id not in available
    )
    dangling.update(set(metadata.row_identity.columns) - available)
    if dangling:
        raise AssertionError(f"Metadata references missing columns: {dangling}")
