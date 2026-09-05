"""Construction and validation of pandas-compatible DataFrame merges and joins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from duckpd._logical import (
    Column,
    ColumnId,
    IndexUniqueness,
    JoinPlan,
    JoinType,
    Nullability,
)
from duckpd._metadata import after_join, find_column
from duckpd.errors import AlignmentError

if TYPE_CHECKING:
    from duckpd.frame import DataFrame


MergeHow = Literal["left", "right", "outer", "inner", "cross"]


def validate_explicit_index_alignment(
    left_frame: DataFrame,
    right_frame: DataFrame,
) -> None:
    """Reject cross-frame alignment that lacks identical explicit index metadata."""
    if left_frame._session is not right_frame._session:
        raise AlignmentError("Cannot align frames from different sessions")

    left_ids = left_frame._plan.metadata.index.columns
    right_ids = right_frame._plan.metadata.index.columns
    if not left_ids or not right_ids:
        raise AlignmentError(
            "Cross-frame arithmetic requires explicit index alignment on both "
            "operands; use set_index() first"
        )
    if (
        left_frame._plan.metadata.index.uniqueness is IndexUniqueness.NON_UNIQUE
        or right_frame._plan.metadata.index.uniqueness is IndexUniqueness.NON_UNIQUE
    ):
        raise AlignmentError("Cross-frame arithmetic requires unique explicit indexes")
    if len(left_ids) != len(right_ids):
        raise AlignmentError(
            "Cross-frame arithmetic requires indexes with the same number of levels"
        )

    left_columns = tuple(left_frame._column_by_id(column_id) for column_id in left_ids)
    right_columns = tuple(right_frame._column_by_id(column_id) for column_id in right_ids)
    left_names = tuple(column.label for column in left_columns)
    right_names = tuple(column.label for column in right_columns)
    if left_names != right_names:
        raise AlignmentError(
            "Cross-frame arithmetic requires matching index names; "
            f"found {left_names!r} and {right_names!r}"
        )
    left_types = tuple(column.duckdb_type for column in left_columns)
    right_types = tuple(column.duckdb_type for column in right_columns)
    if left_types != right_types:
        raise AlignmentError(
            "Cross-frame arithmetic requires matching index dtypes; "
            f"found {left_types!r} and {right_types!r}"
        )


def plan_merge(
    left_frame: DataFrame,
    right_frame: DataFrame,
    how: MergeHow = "inner",
    on: str | Sequence[str] | None = None,
    left_on: str | Sequence[str] | None = None,
    right_on: str | Sequence[str] | None = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: tuple[str | None, str | None] = ("_x", "_y"),
    validate: str | None = None,
) -> JoinPlan:
    """Build a typed JoinPlan following pandas merge semantics."""
    if how not in {"inner", "left", "right", "outer", "cross"}:
        msg = f"Invalid how={how!r}; must be 'inner', 'left', 'right', 'outer', or 'cross'"
        raise ValueError(msg)

    join_type = {
        "inner": JoinType.INNER,
        "left": JoinType.LEFT,
        "right": JoinType.RIGHT,
        "outer": JoinType.OUTER,
        "cross": JoinType.CROSS,
    }[how]

    if len(suffixes) != 2 or any(
        suffix is not None and type(suffix) is not str for suffix in suffixes
    ):
        raise ValueError("suffixes must contain two strings or None")
    lsuffix, rsuffix = suffixes

    _valid_validate_values = {
        "1:1",
        "1:m",
        "m:1",
        "m:m",
        "one_to_one",
        "one_to_many",
        "many_to_one",
        "many_to_many",
    }
    if validate is not None and validate not in _valid_validate_values:
        raise ValueError(
            f'"{validate}" is not a valid argument. Valid arguments are:\n'
            '- "1:1"\n'
            '- "1:m"\n'
            '- "m:1"\n'
            '- "m:m"\n'
            '- "one_to_one"\n'
            '- "one_to_many"\n'
            '- "many_to_one"\n'
            '- "many_to_many"'
        )

    # 1. Resolve key columns
    left_keys: list[Column] = []
    right_keys: list[Column] = []

    if join_type is JoinType.CROSS:
        if on or left_on or right_on or left_index or right_index:
            raise ValueError(
                "Can not pass on, left_on, right_on, left_index or right_index when how='cross'"
            )
    else:
        if on is not None:
            if left_on is not None or right_on is not None or left_index or right_index:
                raise ValueError("Cannot pass on with left_on, right_on, left_index or right_index")
            on_labels = (on,) if isinstance(on, str) else tuple(on)
            if not on_labels:
                raise ValueError("on must not be empty")
            for label in on_labels:
                left_keys.append(find_column(left_frame._plan.metadata, label))
                right_keys.append(find_column(right_frame._plan.metadata, label))
        else:
            if left_index:
                if not left_frame._plan.metadata.index.columns:
                    raise ValueError(
                        "left_index=True requires left frame to have an explicit index"
                    )
                left_keys = [
                    left_frame._column_by_id(c_id)
                    for c_id in left_frame._plan.metadata.index.columns
                ]
            elif left_on is not None:
                left_on_labels = (left_on,) if isinstance(left_on, str) else tuple(left_on)
                left_keys = [
                    find_column(left_frame._plan.metadata, key_label)
                    for key_label in left_on_labels
                ]
            else:
                # default: intersect visible column names
                common = [
                    c.label
                    for c in left_frame._plan.metadata.visible_columns
                    if any(r.label == c.label for r in right_frame._plan.metadata.visible_columns)
                ]
                if not common:
                    raise ValueError(
                        "No common columns to merge on; "
                        "specify on, left_on, right_on, or how='cross'"
                    )
                for label in common:
                    left_keys.append(find_column(left_frame._plan.metadata, label))
                    right_keys.append(find_column(right_frame._plan.metadata, label))

            if right_index:
                if not right_frame._plan.metadata.index.columns:
                    raise ValueError(
                        "right_index=True requires right frame to have an explicit index"
                    )
                right_keys = [
                    right_frame._column_by_id(c_id)
                    for c_id in right_frame._plan.metadata.index.columns
                ]
            elif right_on is not None:
                right_on_labels = (right_on,) if isinstance(right_on, str) else tuple(right_on)
                right_keys = [
                    find_column(right_frame._plan.metadata, key_label)
                    for key_label in right_on_labels
                ]
            elif not right_keys:
                raise ValueError(
                    "Must specify right_on or right_index when left_on/left_index is provided"
                )

        if len(left_keys) != len(right_keys):
            raise ValueError(
                f"len(left_on) ({len(left_keys)}) does not match len(right_on) ({len(right_keys)})"
            )

    # 2. Build output schema & handle name collisions and duplicate merge keys
    # When `on` is used, the right merge key columns are NOT duplicated in the output.
    # When `left_on` / `right_on` have different names, both columns appear.
    output_columns: list[Column] = []

    left_key_ids = {k.id for k in left_keys}
    right_key_ids = {k.id for k in right_keys}

    # If left_index=True and right_index=True, left index columns become index
    output_index_ids: list[ColumnId] = []
    if left_index and right_index:
        output_index_ids = list(left_frame._plan.metadata.index.columns)
        # Left columns include hidden index columns
        left_included = left_frame._plan.metadata.columns
        # Right columns include visible only (hidden index dropped)
        right_included = [
            c for c in right_frame._plan.metadata.visible_columns if c.id not in right_key_ids
        ]
    else:
        left_included = left_frame._plan.metadata.visible_columns
        right_included = [c for c in right_frame._plan.metadata.visible_columns]

    # Set of right key columns that are mapped to left key columns
    right_to_left_key_map: dict[ColumnId, ColumnId] = {}
    if on is not None or (
        not left_on
        and not right_on
        and not left_index
        and not right_index
        and join_type is not JoinType.CROSS
    ):
        for l_col, r_col in zip(left_keys, right_keys, strict=True):
            if l_col.label == r_col.label:
                right_to_left_key_map[r_col.id] = l_col.id

    # Check collisions between left visible and right visible
    left_labels = {c.label for c in left_included if not c.hidden}
    right_labels = {
        c.label for c in right_included if c.id not in right_to_left_key_map and not c.hidden
    }
    common_labels = left_labels.intersection(right_labels)
    if common_labels and not lsuffix and not rsuffix:
        raise ValueError(f"columns overlap but no suffix is specified: {sorted(common_labels)!r}")

    for c in left_included:
        is_collision = not c.hidden and c.label in common_labels
        if (is_collision and c.id not in left_key_ids) or (
            is_collision and c.id in left_key_ids and join_type is JoinType.CROSS
        ):
            output_columns.append(replace(c, label=f"{c.label}{lsuffix or ''}"))
        else:
            output_columns.append(c)

    for c in right_included:
        if c.id in right_to_left_key_map:
            continue
        is_collision = not c.hidden and c.label in common_labels
        if (is_collision and c.id not in right_key_ids) or (
            is_collision and c.id in right_key_ids and join_type is JoinType.CROSS
        ):
            output_columns.append(replace(c, label=f"{c.label}{rsuffix or ''}"))
        else:
            output_columns.append(c)

    nullable_ids: set[ColumnId] = set()
    if join_type in {JoinType.RIGHT, JoinType.OUTER}:
        nullable_ids.update(column.id for column in left_included)
    if join_type in {JoinType.LEFT, JoinType.OUTER}:
        nullable_ids.update(column.id for column in right_included)
    output_columns = [
        replace(column, nullable=Nullability.NULLABLE) if column.id in nullable_ids else column
        for column in output_columns
    ]

    output_labels = [column.label for column in output_columns]
    duplicate_labels = sorted(
        label for label in set(output_labels) if output_labels.count(label) > 1
    )
    if duplicate_labels:
        raise ValueError(f"suffixes produce duplicate columns: {duplicate_labels!r}")

    # SQL joins do not preserve a total row order. Even sort=True only orders
    # by merge keys, whose duplicates have no stable tie-breaker after a join.
    metadata = after_join(
        tuple(output_columns),
        index_ids=tuple(output_index_ids),
    )

    return JoinPlan(
        left=left_frame._plan,
        right=right_frame._plan,
        how=join_type,
        left_keys=tuple(k.id for k in left_keys),
        right_keys=tuple(k.id for k in right_keys),
        metadata=metadata,
        sort=sort,
        validate=validate,
    )
