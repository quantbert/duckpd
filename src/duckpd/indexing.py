"""Label-based and positional indexers for DuckPD DataFrames."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import pandas as pd

from duckpd._logical import (
    BinaryExpression,
    BinaryOperator,
    CaseWhen,
    Column,
    ColumnId,
    ColumnRef,
    Expression,
    FilterPlan,
    FunctionCall,
    LimitPlan,
    LiteralValue,
    LocIndexPlan,
    NamedExpression,
    NullPlacement,
    OrderColumn,
    OrderSpec,
    ProjectPlan,
    RowIdentity,
    SortDirection,
)
from duckpd._metadata import (
    after_filter,
    after_projection,
    after_reindex,
    protected_column_ids,
)
from duckpd._reductions import expression_type
from duckpd._typing import ScalarValue, is_scalar_value
from duckpd.errors import (
    UnorderedOperationError,
    UnsupportedOperationError,
)

if TYPE_CHECKING:
    from duckpd.frame import DataFrame
    from duckpd.series import Series


class LocIndexer:
    """Label-based indexing and masked assignment (.loc)."""

    def __init__(self, frame: DataFrame) -> None:
        self._frame = frame

    def __getitem__(self, key: object) -> DataFrame | Series:
        from duckpd.series import Series

        row_key, col_key = self._split_key(key)

        # Resolve rows
        filtered_df: DataFrame
        if isinstance(row_key, slice):
            slice_key = cast("slice", row_key)
            if (
                slice_key.start is not None
                or slice_key.stop is not None
                or slice_key.step is not None
            ):
                raise UnsupportedOperationError(
                    "Label-based row slicing is not supported; use boolean masks"
                )
            filtered_df = self._frame
        elif isinstance(row_key, Series):
            self._frame._require_same_plan(row_key)
            filtered_df = self._frame[row_key]
        elif isinstance(row_key, set):
            raise TypeError("Passing a set as an indexer is not supported. Use a list instead.")
        elif isinstance(row_key, list):
            if not row_key:
                filtered_df = self._frame.limit(0)
            else:
                index_ids = self._frame._plan.metadata.index.columns
                if not index_ids:
                    raise UnsupportedOperationError(
                        "DataFrame has no explicit index to match labels"
                    )
                filtered_df = self._build_loc_index_frame(cast("Sequence[object]", row_key))
        else:
            pred = self._index_predicate(row_key)
            from duckpd.frame import DataFrame as DataFrameClass

            filtered_df = DataFrameClass(
                self._frame._session,
                FilterPlan(self._frame._plan, pred, after_filter(self._frame._plan.metadata)),
            )

        # Resolve columns
        if col_key is None or self._is_full_slice(col_key):
            return filtered_df
        if isinstance(col_key, str):
            return filtered_df[col_key]
        if isinstance(col_key, Sequence):
            return filtered_df[list(cast("Sequence[str]", col_key))]
        raise UnsupportedOperationError(f"Unsupported column key in .loc: {col_key!r}")

    def _build_loc_index_frame(self, keys: Sequence[object]) -> DataFrame:
        from duckpd.frame import DataFrame as DataFrameClass

        index_ids = self._frame._plan.metadata.index.columns
        source_key = uuid4().hex
        key_labels = tuple(
            f"__duckpd_loc_key_{source_key}_{position}__" for position in range(len(index_ids))
        )
        source_order_label = f"__duckpd_loc_request_order_{source_key}__"
        order_col_id = ColumnId.create()
        order_label = f"__duckpd_loc_order_{order_col_id.value.hex}__"
        order_col = Column(
            order_col_id,
            order_label,
            "BIGINT",
            hidden=True,
        )

        records: list[dict[str, object]] = []
        for position, key in enumerate(keys):
            values: tuple[object, ...]
            if isinstance(key, tuple) and len(index_ids) > 1:
                values = cast("tuple[object, ...]", key)
            else:
                values = (cast("object", key),)
            if len(values) != len(index_ids):
                raise KeyError("Index key has the wrong number of levels")
            record = {label: value for label, value in zip(key_labels, values, strict=True)}
            record[source_order_label] = position
            records.append(record)

        keys_df = pd.DataFrame(records)
        self._frame._session._registered_sources[source_key] = keys_df

        columns = (*self._frame._plan.metadata.columns, order_col)
        input_ordering = self._frame._plan.metadata.ordering
        input_identity = self._frame._plan.metadata.row_identity
        has_total_input_order = (
            input_identity.stable
            and input_identity.unique
            and bool(input_identity.columns)
            and bool(input_ordering.keys)
        )
        if has_total_input_order:
            request_order = OrderColumn(
                order_col_id,
                SortDirection.ASCENDING,
                NullPlacement.LAST,
            )
            ordering = OrderSpec((request_order, *input_ordering.keys))
            row_identity = RowIdentity(
                (order_col_id, *input_identity.columns),
                stable=True,
                unique=True,
                source_key=input_identity.source_key,
            )
        else:
            ordering = OrderSpec()
            row_identity = RowIdentity()
        metadata = after_reindex(
            self._frame._plan.metadata,
            columns,
            ordering,
            row_identity,
        )
        plan = LocIndexPlan(
            input=self._frame._plan,
            metadata=metadata,
            order_column_id=order_col_id,
            source_key=source_key,
            key_labels=key_labels,
            source_order_label=source_order_label,
        )
        return DataFrameClass(self._frame._session, plan)

    def _split_key(self, key: object) -> tuple[object, object | None]:
        if not isinstance(key, tuple):
            return key, None
        tuple_key = cast("tuple[object, ...]", key)
        index_depth = len(self._frame._plan.metadata.index.columns)
        if len(tuple_key) == 2 and (
            isinstance(tuple_key[0], (tuple, list)) or self._is_full_slice(tuple_key[0])
        ):
            return cast("object", tuple_key[0]), tuple_key[1]
        if index_depth > 1 and len(tuple_key) <= index_depth:
            return tuple_key, None
        if index_depth == 1 and len(tuple_key) == 2:
            return tuple_key[0], tuple_key[1]
        raise UnsupportedOperationError("Ambiguous or invalid .loc key")

    def _index_predicate(self, key: object) -> Expression:
        index_ids = self._frame._plan.metadata.index.columns
        if not index_ids:
            raise UnsupportedOperationError("DataFrame has no explicit index to match labels")
        values: tuple[object, ...] = (
            cast("tuple[object, ...]", key) if isinstance(key, tuple) else (key,)
        )
        if not values or len(values) > len(index_ids):
            raise KeyError("Index key has the wrong number of levels")
        predicate: Expression = LiteralValue(True)
        for column_id, value in zip(index_ids, values, strict=False):
            if not is_scalar_value(value):
                raise TypeError(".loc index labels must be scalar values")
            comparison: Expression
            if value is None:
                comparison = FunctionCall("isnull", (ColumnRef(column_id),))
            else:
                comparison = BinaryExpression(
                    ColumnRef(column_id),
                    BinaryOperator.EQUAL,
                    LiteralValue(cast("ScalarValue", value)),
                )
            predicate = BinaryExpression(predicate, BinaryOperator.AND, comparison)
        return predicate

    @staticmethod
    def _is_full_slice(key: object) -> bool:
        return isinstance(key, slice) and key == slice(None)

    def __setitem__(self, key: object, value: object) -> None:
        """Masked assignment: df.loc[mask, col] = val."""
        from duckpd.series import Series

        row_key: object
        col_key: object = None
        if isinstance(key, tuple):
            tup_key = cast("tuple[object, ...]", key)
            if len(tup_key) == 2:
                row_key, col_key = tup_key[0], tup_key[1]
            else:
                raise UnsupportedOperationError("DuckPD .loc assignment supports (row_mask, col)")
        else:
            row_key = key

        if not isinstance(row_key, Series):
            raise UnsupportedOperationError(
                "DuckPD .loc assignment currently requires a boolean Series mask"
            )
        self._frame._require_same_plan(row_key)

        if col_key is None:
            # Masked assignment to all visible columns
            target_cols: list[str] = [c.label for c in self._frame._plan.metadata.visible_columns]
        elif isinstance(col_key, str):
            target_cols = [col_key]
        elif isinstance(col_key, Sequence):
            target_cols = list(cast("Sequence[str]", col_key))
        else:
            raise UnsupportedOperationError(f"Unsupported column target in .loc: {col_key!r}")

        targets = [self._frame._column(label) for label in target_cols]
        protected = protected_column_ids(self._frame._plan.metadata)
        if any(column.id in protected for column in targets):
            raise ValueError("Cannot assign to an index or ordering column")

        # For each target column, create a CaseWhen(mask, value, existing_col)
        val_expr = self._frame._coerce_expression(value)
        new_projections: list[NamedExpression] = []
        target_set = set(target_cols)

        for col in self._frame._plan.metadata.columns:
            if col.label in target_set and not col.hidden:
                case_expr = CaseWhen(
                    row_key._expression,
                    val_expr,
                    ColumnRef(col.id),
                )
                out_col = Column(
                    ColumnId.create(),
                    col.label,
                    expression_type(self._frame._plan, case_expr),
                )
                new_projections.append(NamedExpression(out_col, case_expr))
            else:
                new_projections.append(NamedExpression(col, ColumnRef(col.id)))

        output_columns = tuple(p.column for p in new_projections)
        metadata = after_projection(self._frame._plan.metadata, output_columns)
        self._frame._plan = ProjectPlan(self._frame._plan, tuple(new_projections), metadata)


class ILocIndexer:
    """Positional indexing and slicing (.iloc)."""

    def __init__(self, frame: DataFrame) -> None:
        self._frame = frame

    def __getitem__(self, key: object) -> DataFrame | Series:
        row_key: object = key
        column_key: object | None = None
        if isinstance(key, tuple):
            tuple_key = cast("tuple[object, ...]", key)
            if len(tuple_key) != 2:
                raise UnsupportedOperationError(
                    "DuckPD .iloc supports up to 2 dimensions [row, col]"
                )
            row_key, column_key = tuple_key

        result = self._select_rows(row_key)
        if column_key is None or (isinstance(column_key, slice) and column_key == slice(None)):
            return result
        visible = result._plan.metadata.visible_columns
        if isinstance(column_key, int):
            try:
                return result[visible[column_key].label]
            except IndexError as error:
                raise IndexError("single positional indexer is out-of-bounds") from error
        if isinstance(column_key, slice):
            labels = [column.label for column in visible[column_key]]
            return result[labels]
        if isinstance(column_key, Sequence) and not isinstance(column_key, str):
            positions = cast("Sequence[int]", column_key)
            try:
                labels = [visible[position].label for position in positions]
            except IndexError as error:
                raise IndexError("positional indexers are out-of-bounds") from error
            return result[labels]
        raise UnsupportedOperationError(f"Unsupported column key in .iloc: {column_key!r}")

    def _select_rows(self, key: object) -> DataFrame:
        if isinstance(key, slice):
            if key == slice(None):
                return self._frame
            ordering = self._frame._plan.metadata.ordering
            if not ordering.keys:
                raise UnorderedOperationError(
                    "Positional .iloc slicing requires a guaranteed row ordering. "
                    "Specify order_by when creating a SQL/table source or sort "
                    "first using .sort_values(...)"
                )
            slice_key = cast("slice[int | None, int | None, int | None]", key)
            start_val = slice_key.start
            stop_val = slice_key.stop
            step_val = slice_key.step

            start = start_val if start_val is not None else 0
            if step_val is not None and step_val != 1:
                raise UnsupportedOperationError("DuckPD .iloc does not support step != 1")
            if start < 0 or (stop_val is not None and stop_val < 0):
                raise UnsupportedOperationError(
                    "DuckPD .iloc does not support negative slice indices"
                )

            offset = start
            count = (stop_val - start) if stop_val is not None else 9223372036854775807
            if count < 0:
                count = 0

            from duckpd.frame import DataFrame as DataFrameClass

            plan = LimitPlan(
                self._frame._plan,
                count=count,
                offset=offset,
                metadata=self._frame._plan.metadata,
            )
            return DataFrameClass(self._frame._session, plan)

        raise UnsupportedOperationError(
            "DuckPD .iloc currently supports slice indexing [start:stop]"
        )
