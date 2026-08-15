"""Label-based and positional indexers for DuckPD DataFrames."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from duckpd._logical import (
    BinaryExpression,
    BinaryOperator,
    CaseWhen,
    Column,
    ColumnId,
    ColumnRef,
    Expression,
    FilterPlan,
    LimitPlan,
    LiteralValue,
    NamedExpression,
    ProjectPlan,
)
from duckpd._metadata import after_projection
from duckpd._reductions import expression_type
from duckpd._typing import ScalarValue
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

        # Case 1: tuple like (mask, columns) or (index_val, columns)
        row_key: object
        col_key: object = None
        if isinstance(key, tuple):
            tup_key = cast("tuple[object, ...]", key)
            if len(tup_key) == 2:
                row_key, col_key = tup_key[0], tup_key[1]
            else:
                raise UnsupportedOperationError(
                    "DuckPD .loc supports up to 2 dimensions [row, col]"
                )
        else:
            row_key = key

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
        elif isinstance(row_key, list | tuple | set):
            # Check for explicit index matching
            index_ids = self._frame._plan.metadata.index.columns
            if not index_ids:
                raise UnsupportedOperationError(
                    "DataFrame has no explicit index to match labels"
                )
            # If 1 index column, match value in row_key
            idx_col_id = index_ids[0]
            pred: Expression = LiteralValue(False)
            for item in cast("Sequence[object]", row_key):
                val_pred = BinaryExpression(
                    ColumnRef(idx_col_id),
                    BinaryOperator.EQUAL,
                    LiteralValue(cast("ScalarValue", item)),
                )
                pred = BinaryExpression(pred, BinaryOperator.OR, val_pred)
            from duckpd.frame import DataFrame as DataFrameClass

            filtered_df = DataFrameClass(
                self._frame._session,
                FilterPlan(self._frame._plan, pred, self._frame._plan.metadata),
            )
        else:
            # Single label scalar on explicit index
            index_ids = self._frame._plan.metadata.index.columns
            if not index_ids:
                raise UnsupportedOperationError(
                    "DataFrame has no explicit index to match labels"
                )
            idx_col_id = index_ids[0]
            pred = BinaryExpression(
                ColumnRef(idx_col_id),
                BinaryOperator.EQUAL,
                LiteralValue(cast("ScalarValue", row_key)),
            )
            from duckpd.frame import DataFrame as DataFrameClass

            filtered_df = DataFrameClass(
                self._frame._session,
                FilterPlan(self._frame._plan, pred, self._frame._plan.metadata),
            )

        # Resolve columns
        if col_key is None:
            return filtered_df
        if isinstance(col_key, str):
            return filtered_df[col_key]
        if isinstance(col_key, Sequence):
            return filtered_df[list(cast("Sequence[str]", col_key))]
        raise UnsupportedOperationError(f"Unsupported column key in .loc: {col_key!r}")

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
                raise UnsupportedOperationError(
                    "DuckPD .loc assignment supports (row_mask, col)"
                )
        else:
            row_key = key

        if not isinstance(row_key, Series):
            raise UnsupportedOperationError(
                "DuckPD .loc assignment currently requires a boolean Series mask"
            )
        self._frame._require_same_plan(row_key)

        if col_key is None:
            # Masked assignment to all visible columns
            target_cols: list[str] = [
                c.label for c in self._frame._plan.metadata.visible_columns
            ]
        elif isinstance(col_key, str):
            target_cols = [col_key]
        elif isinstance(col_key, Sequence):
            target_cols = list(cast("Sequence[str]", col_key))
        else:
            raise UnsupportedOperationError(
                f"Unsupported column target in .loc: {col_key!r}"
            )

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
        self._frame._plan = ProjectPlan(
            self._frame._plan, tuple(new_projections), metadata
        )


class ILocIndexer:
    """Positional indexing and slicing (.iloc)."""

    def __init__(self, frame: DataFrame) -> None:
        self._frame = frame

    def __getitem__(self, key: object) -> DataFrame:
        if isinstance(key, slice):
            ordering = self._frame._plan.metadata.ordering
            if not ordering.keys:
                raise UnorderedOperationError(
                    "Positional .iloc slicing requires a guaranteed row order"
                )
            slice_key = cast("slice", key)
            start_val = cast("int | None", slice_key.start)
            stop_val = cast("int | None", slice_key.stop)
            step_val = cast("int | None", slice_key.step)

            start = start_val if start_val is not None else 0
            if step_val is not None and step_val != 1:
                raise UnsupportedOperationError(
                    "DuckPD .iloc does not support step != 1"
                )
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
