"""Lazy DataFrame public API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace as dataclass_replace
from datetime import timedelta
from decimal import Decimal
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, overload
from uuid import uuid4

import pandas as pd
import pyarrow as pa

from duckpd._executor import CommitReport, ProfileResult
from duckpd._logical import (
    AggregateOperator,
    BinaryExpression,
    BinaryOperator,
    CaseWhen,
    CastExpression,
    Column,
    ColumnId,
    ColumnRef,
    FilterPlan,
    FrameMetadata,
    FunctionCall,
    IndexSpec,
    LimitPlan,
    LiteralValue,
    NamedExpression,
    NullPlacement,
    OrderSpec,
    ParquetSource,
    ProjectPlan,
    SamplePlan,
    ScanPlan,
    SortDirection,
    SortKey,
    SortPlan,
    SourceKind,
    SourceProvenance,
    TableSource,
    WindowExpression,
    expression_nullability,
)
from duckpd._metadata import (
    after_filter,
    after_projection,
    after_sort,
    find_column,
    projection_columns,
    protected_column_ids,
)
from duckpd._metadata import reset_index as reset_index_metadata
from duckpd._metadata import set_index as set_index_metadata
from duckpd._reductions import (
    aggregate_plan,
    expression_categorical,
    expression_timezone,
    expression_type,
    is_numeric_type,
    materialized_int,
    validate_axis,
    validate_ddof,
    validate_min_count,
    validate_quantile,
)
from duckpd._typing import (
    ParquetCompression,
    ScalarValue,
    is_scalar_value,
    normalize_dtype,
)
from duckpd.errors import (
    AlignmentError,
    UnorderedOperationError,
    UnsupportedOperationError,
)

if TYPE_CHECKING:
    from duckpd._logical import Expression, LogicalPlan
    from duckpd.groupby import DataFrameGroupBy
    from duckpd.indexing import ILocIndexer, LocIndexer
    from duckpd.series import Series
    from duckpd.session import Session
    from duckpd.window import Expanding, Rolling


class DataFrame:
    """A lazy pandas-shaped DataFrame backed by a DuckDB query plan."""

    def __init__(
        self,
        session: Session,
        plan: LogicalPlan,
        *,
        alignment_source: LogicalPlan | None = None,
        alignment_expressions: tuple[Expression, ...] = (),
    ) -> None:
        self._session = session
        self._plan = plan
        self._alignment_source = alignment_source
        self._alignment_expressions = alignment_expressions

    @property
    def columns(self) -> tuple[str, ...]:
        """Displayed column labels without executing the plan."""
        return tuple(column.label for column in self._plan.metadata.visible_columns)

    @property
    def index_names(self) -> tuple[str | None, ...]:
        """Names of explicit index columns without executing the plan."""
        if self._plan.metadata.index.names:
            return self._plan.metadata.index.names
        return tuple(
            self._column_by_id(column_id).label for column_id in self._plan.metadata.index.columns
        )

    @property
    def ordering(self) -> tuple[str, ...]:
        """Names of columns that guarantee the current row order."""
        identity_ids = set(self._plan.metadata.row_identity.columns)
        return tuple(
            self._column_by_id(key.column_id).label
            for key in self._plan.metadata.ordering.keys
            if key.column_id not in identity_ids
        )

    @property
    def loc(self) -> LocIndexer:
        """Label-based indexing and masked assignment."""
        from duckpd.indexing import LocIndexer

        return LocIndexer(self)

    @property
    def iloc(self) -> ILocIndexer:
        """Positional indexing and row slicing."""
        from duckpd.indexing import ILocIndexer

        return ILocIndexer(self)

    def collect(self) -> pd.DataFrame:
        """Execute the complete plan and return a pandas DataFrame."""
        return self._session._executor.collect(self._plan)

    def collect_small(self, max_bytes: int) -> pd.DataFrame:
        """Explicitly collect a local Parquet plan under a strict byte limit."""
        return self._session._executor.collect_small(
            self._plan,
            max_bytes=max_bytes,
        )

    def to_pandas(self) -> pd.DataFrame:
        """Alias for :meth:`collect`."""
        return self.collect()

    def to_arrow(self) -> pa.Table:
        """Execute the complete plan and return an Arrow table."""
        return self._session._executor.to_arrow(self._plan)

    def to_arrow_batches(self, batch_size: int = 1_000_000) -> pa.RecordBatchReader:
        """Execute the plan and stream its rows as Arrow record batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return self._session._executor.to_arrow_batches(self._plan, batch_size=batch_size)

    def write_parquet(
        self,
        path: str | Path,
        *,
        compression: ParquetCompression = "snappy",
        overwrite: bool = False,
    ) -> None:
        """Execute the plan directly into a Parquet file."""
        self._session._executor.write_parquet(
            self._plan,
            str(path),
            compression=compression,
            overwrite=overwrite,
        )

    def write_csv(
        self,
        path: str | Path,
        *,
        sep: str = ",",
        header: bool = True,
    ) -> None:
        """Execute the plan directly into a CSV file."""
        self._session._executor.write_csv(
            self._plan,
            str(path),
            sep=sep,
            header=header,
        )

    def to_csv(
        self,
        path: str | Path,
        *,
        sep: str = ",",
        header: bool = True,
    ) -> None:
        """Alias for :meth:`write_csv`."""
        self.write_csv(path, sep=sep, header=header)

    def persist(self, name: str | None = None) -> DataFrame:
        """Persist the complete plan while retaining index and order metadata."""
        table_name = name if name is not None else f"__duckpd_persist_{uuid4().hex}__"
        self._session._executor.persist(self._plan, table_name)
        metadata = dataclass_replace(
            self._plan.metadata,
            provenance=SourceProvenance(
                SourceKind.TABLE,
                (table_name,),
                writable=True,
                row_preserving=self._plan.metadata.provenance.row_preserving,
                transformations=(
                    *self._plan.metadata.provenance.transformations,
                    "persist",
                ),
            ),
        )
        return DataFrame(
            self._session,
            ScanPlan(TableSource(table_name), metadata),
        )

    def save_as_table(
        self,
        name: str,
        *,
        mode: Literal["error", "overwrite", "append"] = "error",
    ) -> None:
        """Save the DataFrame to a persistent DuckDB table in the current session.

        Parameters
        ----------
        name : str
            Table name to create or update.
        mode : {'error', 'overwrite', 'append'}, default 'error'
            - 'error': raise ValueError if table already exists.
            - 'overwrite': drop existing table and create new table.
            - 'append': append rows to existing table with schema validation.
        """
        self._session._executor.save_as_table(self._plan, name, mode=mode)

    def commit(
        self,
        *,
        compression: ParquetCompression = "snappy",
        retain_previous: bool = False,
    ) -> CommitReport:
        """Atomically commit transformations back to a single Parquet source.

        Parameters
        ----------
        compression : ParquetCompression, default 'snappy'
            Parquet compression codec for the committed file.
        retain_previous : bool, default False
            If True, preserve previous version as a backup file in source directory.

        Returns
        -------
        CommitReport
            Structured summary of the commit operation.
        """
        report = self._session._executor.commit(
            self._plan,
            compression=compression,
            retain_previous=retain_previous,
        )
        identity_columns = self._plan.metadata.row_identity.columns
        stable_order_label = (
            self._column_by_id(identity_columns[0]).label if len(identity_columns) == 1 else None
        )
        self._plan = ScanPlan(
            ParquetSource(
                (report.source_path,),
                stable_order_label=stable_order_label,
                native_order=stable_order_label is not None,
            ),
            self._plan.metadata,
        )
        return report

    def explain(
        self,
        mode: Literal["all", "logical", "optimized", "json", "sql", "physical", "analyze"] = "all",
    ) -> str:
        """Return a logical, optimized, SQL, physical, analyzed, or JSON plan."""
        return self._session._executor.explain(self._plan, mode=mode)

    def explain_write(
        self,
        path: str | Path,
        *,
        compression: ParquetCompression = "snappy",
    ) -> str:
        """Inspect write strategy and execution plan without writing rows."""
        return self._session._executor.explain_write(self._plan, str(path), compression=compression)

    def profile(self) -> ProfileResult:
        """Execute the plan with DuckDB profiling enabled and return metrics."""
        return self._session._executor.profile(self._plan)

    def __add__(self, other: object) -> DataFrame:
        return self._binary(other, BinaryOperator.ADD)

    def __radd__(self, other: object) -> DataFrame:
        return self._binary(other, BinaryOperator.ADD, reverse=True)

    def __sub__(self, other: object) -> DataFrame:
        return self._binary(other, BinaryOperator.SUBTRACT)

    def __rsub__(self, other: object) -> DataFrame:
        return self._binary(other, BinaryOperator.SUBTRACT, reverse=True)

    def __mul__(self, other: object) -> DataFrame:
        return self._binary(other, BinaryOperator.MULTIPLY)

    def __rmul__(self, other: object) -> DataFrame:
        return self._binary(other, BinaryOperator.MULTIPLY, reverse=True)

    def __truediv__(self, other: object) -> DataFrame:
        return self._binary(other, BinaryOperator.TRUE_DIVIDE)

    def __rtruediv__(self, other: object) -> DataFrame:
        return self._binary(other, BinaryOperator.TRUE_DIVIDE, reverse=True)

    def __mod__(self, other: object) -> DataFrame:
        return self._binary(other, BinaryOperator.MODULO)

    def __rmod__(self, other: object) -> DataFrame:
        return self._binary(other, BinaryOperator.MODULO, reverse=True)

    @property
    def size(self) -> int:
        """Return the number of visible elements in the frame."""
        plan = aggregate_plan(
            self._plan,
            (("__duckpd_size__", None, None),),
            AggregateOperator.SIZE,
        )
        rows = materialized_int(self._session._executor.reduce_scalar(plan))
        return rows * len(self._plan.metadata.visible_columns)

    def count(
        self,
        axis: int | str = 0,
        numeric_only: bool = False,
    ) -> pd.Series:
        """Return the number of non-null values in each supported column."""
        validate_axis(axis, series=False)
        columns = self._reduction_columns(numeric_only=numeric_only)
        return self._reduce_columns(AggregateOperator.COUNT, columns)

    def sum(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
    ) -> pd.Series:
        """Return column-wise sums for supported numeric and boolean columns."""
        validate_axis(axis, series=False)
        validate_min_count(min_count)
        columns = self._reduction_columns(
            numeric_only=numeric_only,
            require_numeric=True,
        )
        return self._reduce_columns(
            AggregateOperator.SUM,
            columns,
            skipna=skipna,
            min_count=min_count,
        )

    def mean(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> pd.Series:
        """Return column-wise means for supported numeric and boolean columns."""
        validate_axis(axis, series=False)
        columns = self._reduction_columns(
            numeric_only=numeric_only,
            require_numeric=True,
        )
        return self._reduce_columns(
            AggregateOperator.MEAN,
            columns,
            skipna=skipna,
        )

    def min(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> pd.Series:
        """Return column-wise minima for supported numeric and boolean columns."""
        validate_axis(axis, series=False)
        columns = self._reduction_columns(
            numeric_only=numeric_only,
            require_numeric=True,
        )
        return self._reduce_columns(
            AggregateOperator.MIN,
            columns,
            skipna=skipna,
        )

    def max(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> pd.Series:
        """Return column-wise maxima for supported numeric and boolean columns."""
        validate_axis(axis, series=False)
        columns = self._reduction_columns(
            numeric_only=numeric_only,
            require_numeric=True,
        )
        return self._reduce_columns(
            AggregateOperator.MAX,
            columns,
            skipna=skipna,
        )

    def std(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> pd.Series:
        """Return column-wise sample standard deviation."""
        validate_axis(axis, series=False)
        validate_ddof(ddof)
        columns = self._reduction_columns(
            numeric_only=numeric_only,
            require_numeric=True,
        )
        return self._reduce_columns(
            AggregateOperator.STD,
            columns,
            skipna=skipna,
            ddof=ddof,
        )

    def var(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> pd.Series:
        """Return column-wise sample variance."""
        validate_axis(axis, series=False)
        validate_ddof(ddof)
        columns = self._reduction_columns(
            numeric_only=numeric_only,
            require_numeric=True,
        )
        return self._reduce_columns(
            AggregateOperator.VAR,
            columns,
            skipna=skipna,
            ddof=ddof,
        )

    def median(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> pd.Series:
        """Return column-wise median values."""
        validate_axis(axis, series=False)
        columns = self._reduction_columns(
            numeric_only=numeric_only,
            require_numeric=True,
        )
        return self._reduce_columns(
            AggregateOperator.MEDIAN,
            columns,
            skipna=skipna,
        )

    def quantile(
        self,
        q: float = 0.5,
        *,
        axis: int | str | None = 0,
        numeric_only: bool = False,
        interpolation: str = "linear",
    ) -> pd.Series:
        """Return column-wise quantiles."""
        validate_axis(axis, series=False)
        if interpolation != "linear":
            raise UnsupportedOperationError(
                "DuckPD quantile currently supports only interpolation='linear'"
            )
        q_val = validate_quantile(q)
        columns = self._reduction_columns(
            numeric_only=numeric_only,
            require_numeric=True,
        )
        res = self._reduce_columns(
            AggregateOperator.QUANTILE,
            columns,
            q=q_val,
        )
        res.name = q
        return res

    def any(
        self,
        *,
        axis: int | str | None = 0,
        bool_only: bool = False,
        skipna: bool = True,
    ) -> pd.Series:
        """Return True if any element in each column is True."""
        validate_axis(axis, series=False)
        visible = self._plan.metadata.visible_columns
        if bool_only:
            columns = tuple(col for col in visible if col.duckdb_type == "BOOLEAN")
            if not columns:
                raise UnsupportedOperationError("No boolean columns found")
        else:
            columns = visible
        return self._reduce_columns(
            AggregateOperator.ANY,
            columns,
            skipna=skipna,
        )

    def all(
        self,
        *,
        axis: int | str | None = 0,
        bool_only: bool = False,
        skipna: bool = True,
    ) -> pd.Series:
        """Return True if all elements in each column are True."""
        validate_axis(axis, series=False)
        visible = self._plan.metadata.visible_columns
        if bool_only:
            columns = tuple(col for col in visible if col.duckdb_type == "BOOLEAN")
            if not columns:
                raise UnsupportedOperationError("No boolean columns found")
        else:
            columns = visible
        return self._reduce_columns(
            AggregateOperator.ALL,
            columns,
            skipna=skipna,
        )

    def isna(self) -> DataFrame:
        """Return a lazy boolean frame marking missing values."""
        return self._null_check("isnull")

    def notna(self) -> DataFrame:
        """Return a lazy boolean frame marking non-missing values."""
        return self._null_check("notnull")

    def isnull(self) -> DataFrame:
        """Alias for :meth:`isna`."""
        return self.isna()

    def notnull(self) -> DataFrame:
        """Alias for :meth:`notna`."""
        return self.notna()

    def astype(
        self,
        dtype: object,
        *,
        copy: bool = True,
        errors: Literal["raise", "ignore"] = "raise",
    ) -> DataFrame:
        """Cast DataFrame columns to specified dtype(s)."""
        if not copy:
            raise UnsupportedOperationError("DuckPD does not support copy=False")
        if errors not in {"raise", "ignore"}:
            raise ValueError("errors must be 'raise' or 'ignore'")

        visible = self._plan.metadata.visible_columns
        index_ids = set(self._plan.metadata.index.columns)
        index_columns = tuple(
            column
            for column in self._plan.metadata.columns
            if column.id in index_ids and column.hidden
        )

        from dataclasses import replace as replace_column

        if isinstance(dtype, dict):
            mapping = cast("dict[object, object]", dtype)
            projections: list[NamedExpression] = []
            for col in visible:
                if col.label in mapping:
                    target_spec = mapping[col.label]
                    try:
                        target_type = normalize_dtype(target_spec)
                        out_col = replace_column(col, duckdb_type=target_type)
                        projections.append(
                            NamedExpression(out_col, CastExpression(ColumnRef(col.id), target_type))
                        )
                    except (TypeError, ValueError):
                        if errors == "raise":
                            raise
                        projections.append(NamedExpression(col, ColumnRef(col.id)))
                else:
                    projections.append(NamedExpression(col, ColumnRef(col.id)))
        else:
            try:
                target_type = normalize_dtype(dtype)
                projections = [
                    NamedExpression(
                        replace_column(col, duckdb_type=target_type),
                        CastExpression(ColumnRef(col.id), target_type),
                    )
                    for col in visible
                ]
            except (TypeError, ValueError):
                if errors == "raise":
                    raise
                projections = [NamedExpression(col, ColumnRef(col.id)) for col in visible]

        for col in index_columns:
            projections.append(NamedExpression(col, ColumnRef(col.id)))

        projected_columns = tuple(p.column for p in projections)
        return DataFrame(
            self._session,
            ProjectPlan(
                self._plan,
                tuple(projections),
                after_projection(self._plan.metadata, projected_columns),
            ),
        )

    def fillna(
        self,
        value: object = None,
        *,
        axis: int | str | None = None,
        inplace: bool = False,
        limit: int | None = None,
    ) -> DataFrame:
        """Fill NA/NaN values using the specified value."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if limit is not None:
            raise UnsupportedOperationError(
                "DuckPD does not support limit in fillna without explicit windows"
            )
        if axis not in {0, "index", None}:
            raise UnsupportedOperationError(
                "DuckPD fillna currently supports only axis=0 or axis='index'"
            )
        if value is None:
            raise ValueError("Must specify a value to fill NA/NaN values with")

        visible = self._plan.metadata.visible_columns
        index_ids = set(self._plan.metadata.index.columns)
        index_columns = tuple(
            column
            for column in self._plan.metadata.columns
            if column.id in index_ids and column.hidden
        )

        from duckpd.series import Series

        if isinstance(value, dict):
            val_map = cast("dict[object, object]", value)
            projections: list[NamedExpression] = []
            for col in visible:
                if col.label in val_map:
                    fill_expr = self._coerce_expression(val_map[col.label])
                    projections.append(
                        NamedExpression(
                            col,
                            FunctionCall("coalesce", (ColumnRef(col.id), fill_expr)),
                        )
                    )
                else:
                    projections.append(NamedExpression(col, ColumnRef(col.id)))
        elif isinstance(value, Series):
            self._require_same_plan(value)
            projections = [
                NamedExpression(
                    col,
                    FunctionCall("coalesce", (ColumnRef(col.id), value._expression)),
                )
                for col in visible
            ]
        elif is_scalar_value(value):
            fill_expr = LiteralValue(value)
            projections = [
                NamedExpression(
                    col,
                    FunctionCall("coalesce", (ColumnRef(col.id), fill_expr)),
                )
                for col in visible
            ]
        else:
            raise TypeError(
                "value must be a scalar, Series, or dict mapping column names to values"
            )

        for col in index_columns:
            projections.append(NamedExpression(col, ColumnRef(col.id)))

        projected_columns = tuple(p.column for p in projections)
        return DataFrame(
            self._session,
            ProjectPlan(
                self._plan,
                tuple(projections),
                after_projection(self._plan.metadata, projected_columns),
            ),
        )

    def dropna(
        self,
        *,
        axis: int | str = 0,
        how: Literal["any", "all"] | None = None,
        thresh: int | None = None,
        subset: str | Sequence[str] | None = None,
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> DataFrame:
        """Remove missing values."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if ignore_index:
            raise UnsupportedOperationError("DuckPD does not support ignore_index=True in dropna")
        if how not in {None, "any", "all"}:
            raise ValueError("how must be 'any' or 'all'")
        if how is not None and thresh is not None:
            raise TypeError("cannot set both how and thresh")
        if thresh is not None:
            if type(thresh) is not int:
                raise TypeError("thresh must be an integer")
            if thresh < 0:
                raise ValueError("thresh must be non-negative")
        how = "any" if how is None else how

        if axis in {1, "columns"}:
            raise UnsupportedOperationError(
                "dropna(axis=1) is not supported because column dropping requires data inspection"
            )

        if axis not in {0, "index"}:
            raise ValueError("axis must be 0, 'index', 1, or 'columns'")

        # Resolve subset columns
        if subset is not None:
            sub_labels = (subset,) if isinstance(subset, str) else tuple(subset)
            target_columns = tuple(self._column(label) for label in sub_labels)
        else:
            target_columns = self._plan.metadata.visible_columns

        if not target_columns:
            return self

        # Build predicate
        if thresh is not None:
            if thresh <= 0:
                return self
            count_expr: Expression = CastExpression(
                FunctionCall("notnull", (ColumnRef(target_columns[0].id),)),
                "INTEGER",
            )
            for col in target_columns[1:]:
                count_expr = BinaryExpression(
                    count_expr,
                    BinaryOperator.ADD,
                    CastExpression(FunctionCall("notnull", (ColumnRef(col.id),)), "INTEGER"),
                )
            predicate: Expression = BinaryExpression(
                count_expr, BinaryOperator.GREATER_EQUAL, LiteralValue(thresh)
            )
        elif how == "any":
            predicate = FunctionCall("notnull", (ColumnRef(target_columns[0].id),))
            for col in target_columns[1:]:
                predicate = BinaryExpression(
                    predicate,
                    BinaryOperator.AND,
                    FunctionCall("notnull", (ColumnRef(col.id),)),
                )
        else:  # how == "all"
            predicate = FunctionCall("notnull", (ColumnRef(target_columns[0].id),))
            for col in target_columns[1:]:
                predicate = BinaryExpression(
                    predicate,
                    BinaryOperator.OR,
                    FunctionCall("notnull", (ColumnRef(col.id),)),
                )

        return DataFrame(
            self._session,
            FilterPlan(self._plan, predicate, after_filter(self._plan.metadata)),
        )

    def where(
        self,
        cond: object,
        other: object = None,
        *,
        inplace: bool = False,
        axis: int | str | None = None,
    ) -> DataFrame:
        """Replace values where the condition is False."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if axis not in {0, "index", None}:
            raise UnsupportedOperationError(
                "DuckPD where currently supports only axis=0 or axis='index'"
            )
        return self._where_mask(cond, other, invert=False)

    def mask(
        self,
        cond: object,
        other: object = None,
        *,
        inplace: bool = False,
        axis: int | str | None = None,
    ) -> DataFrame:
        """Replace values where the condition is True."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if axis not in {0, "index", None}:
            raise UnsupportedOperationError(
                "DuckPD mask currently supports only axis=0 or axis='index'"
            )
        return self._where_mask(cond, other, invert=True)

    def clip(
        self,
        lower: object = None,
        upper: object = None,
        *,
        axis: int | str | None = None,
        inplace: bool = False,
    ) -> DataFrame:
        """Trim values at input thresholds."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if axis not in {0, "index", None}:
            raise UnsupportedOperationError(
                "DuckPD clip currently supports only axis=0 or axis='index'"
            )
        if lower is None and upper is None:
            return self

        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and lower > upper:
            raise ValueError("Cannot set lower > upper")

        new_cols: list[Column] = []
        projections: list[NamedExpression] = []
        for col in self._plan.metadata.visible_columns:
            expr = ColumnRef(col.id)
            col_lower: object = (
                cast("dict[str, object]", lower).get(col.label, None)
                if isinstance(lower, dict)
                else lower
            )
            col_upper: object = (
                cast("dict[str, object]", upper).get(col.label, None)
                if isinstance(upper, dict)
                else upper
            )

            lower_val = (
                LiteralValue(cast("ScalarValue", col_lower)) if col_lower is not None else None
            )
            upper_val = (
                LiteralValue(cast("ScalarValue", col_upper)) if col_upper is not None else None
            )

            cond_lower = (
                BinaryExpression(expr, BinaryOperator.LESS_THAN, lower_val)
                if lower_val is not None
                else None
            )
            cond_upper = (
                BinaryExpression(expr, BinaryOperator.GREATER_THAN, upper_val)
                if upper_val is not None
                else None
            )

            if cond_lower is not None and cond_upper is not None:
                clipped: Expression = CaseWhen(
                    cond_lower,
                    lower_val,  # type: ignore[arg-type]
                    CaseWhen(cond_upper, upper_val, expr),  # type: ignore[arg-type]
                )
            elif cond_lower is not None:
                clipped = CaseWhen(cond_lower, lower_val, expr)  # type: ignore[arg-type]
            elif cond_upper is not None:
                clipped = CaseWhen(cond_upper, upper_val, expr)  # type: ignore[arg-type]
            else:
                clipped = expr

            new_col = Column(
                ColumnId.create(),
                col.label,
                col.duckdb_type,
                nullable=expression_nullability(clipped, self._plan.metadata),
                alias_of=col.id if isinstance(clipped, ColumnRef) else None,
            )
            new_cols.append(new_col)
            projections.append(NamedExpression(new_col, clipped))

        all_cols = projection_columns(self._plan.metadata, tuple(new_cols))
        full_projections: list[NamedExpression] = []
        proj_map = {col.id: p for col, p in zip(new_cols, projections, strict=True)}
        for col in all_cols:
            if col.id in proj_map:
                full_projections.append(proj_map[col.id])
            else:
                full_projections.append(NamedExpression(col, ColumnRef(col.id)))
        metadata = after_projection(self._plan.metadata, all_cols)
        return DataFrame(self._session, ProjectPlan(self._plan, tuple(full_projections), metadata))

    def replace(
        self,
        to_replace: object = None,
        value: object = None,
        *,
        inplace: bool = False,
        limit: int | None = None,
        regex: bool = False,
        method: str | None = None,
    ) -> DataFrame:
        """Replace values given in to_replace with value."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if regex:
            raise UnsupportedOperationError("DuckPD replace does not yet support regex=True")
        if limit is not None or method is not None:
            raise UnsupportedOperationError(
                "DuckPD replace does not support limit or method parameters"
            )
        if to_replace is None and value is None:
            return self

        def _is_replace_compatible(val: object, duckdb_type: str) -> bool:
            if val is None or val is pd.NA:
                return True
            if is_numeric_type(duckdb_type):
                return isinstance(val, (int, float, Decimal)) and not isinstance(val, bool)
            if duckdb_type in {"VARCHAR", "TEXT"}:
                return isinstance(val, str)
            if duckdb_type == "BOOLEAN":
                return isinstance(val, bool)
            return True

        is_column_dict = False
        if isinstance(to_replace, dict):
            visible_names = {c.label for c in self._plan.metadata.visible_columns}
            if any(k in visible_names for k in cast("dict[str, object]", to_replace)):
                is_column_dict = True

        new_cols: list[Column] = []
        projections: list[NamedExpression] = []

        for col in self._plan.metadata.visible_columns:
            expr = ColumnRef(col.id)
            pairs: list[tuple[object, object]] = []

            if is_column_dict:
                col_spec = cast("dict[str, object]", to_replace).get(col.label, None)
                if col_spec is not None:
                    if isinstance(col_spec, dict):
                        pairs = list(cast("dict[object, object]", col_spec).items())
                    else:
                        pairs = [(col_spec, value)]
            elif isinstance(to_replace, dict):
                pairs = list(cast("dict[object, object]", to_replace).items())
            elif isinstance(to_replace, (list, tuple)):
                replace_list = cast("Sequence[object]", to_replace)
                if isinstance(value, (list, tuple)):
                    val_list = cast("Sequence[object]", value)
                    if len(replace_list) != len(val_list):
                        raise ValueError(
                            "Replacement list lengths must match: "
                            f"len(to_replace)={len(replace_list)} vs "
                            f"len(value)={len(val_list)}"
                        )
                    pairs = list(zip(replace_list, val_list, strict=True))
                else:
                    pairs = [(old_val, value) for old_val in replace_list]
            else:
                pairs = [(to_replace, value)]

            cur_expr: Expression = expr
            applicable_pairs = [p for p in pairs if _is_replace_compatible(p[0], col.duckdb_type)]
            if applicable_pairs:
                for old_v, new_v in reversed(applicable_pairs):
                    is_null_val = (
                        old_v is None
                        or old_v is pd.NA
                        or (isinstance(old_v, float) and pd.isna(old_v))
                    )
                    if is_null_val:
                        cond: Expression = FunctionCall("isnull", (expr,))
                    else:
                        old_scalar = cast("ScalarValue", old_v)
                        cond = BinaryExpression(
                            expr, BinaryOperator.EQUAL, LiteralValue(old_scalar)
                        )
                    new_scalar = cast("ScalarValue", new_v)
                    cur_expr = CaseWhen(cond, LiteralValue(new_scalar), cur_expr)
            new_col = Column(
                ColumnId.create(),
                col.label,
                col.duckdb_type,
                nullable=expression_nullability(cur_expr, self._plan.metadata),
                alias_of=col.id if isinstance(cur_expr, ColumnRef) else None,
            )
            new_cols.append(new_col)
            projections.append(NamedExpression(new_col, cur_expr))
        all_cols = projection_columns(self._plan.metadata, tuple(new_cols))
        full_projections: list[NamedExpression] = []
        proj_map = {col.id: p for col, p in zip(new_cols, projections, strict=True)}
        for col in all_cols:
            if col.id in proj_map:
                full_projections.append(proj_map[col.id])
            else:
                full_projections.append(NamedExpression(col, ColumnRef(col.id)))
        metadata = after_projection(self._plan.metadata, all_cols)
        return DataFrame(self._session, ProjectPlan(self._plan, tuple(full_projections), metadata))

    def _where_mask(
        self,
        cond: object,
        other: object,
        *,
        invert: bool,
    ) -> DataFrame:
        from duckpd.series import Series

        visible = self._plan.metadata.visible_columns
        index_ids = set(self._plan.metadata.index.columns)
        index_columns = tuple(
            column
            for column in self._plan.metadata.columns
            if column.id in index_ids and column.hidden
        )

        # Handle cond: DataFrame, Series, or boolean literal
        if isinstance(cond, DataFrame):
            if cond._session is not self._session or cond._plan is not self._plan:
                raise AlignmentError(
                    "Condition DataFrame from different frame requires explicit index alignment"
                )
            # Match each column of cond to self
            cond_map = {col.label: ColumnRef(col.id) for col in cond._plan.metadata.visible_columns}
        elif isinstance(cond, Series):
            self._require_same_plan(cond)
            cond_map = {col.label: cond._expression for col in visible}
        elif isinstance(cond, bool):
            cond_map = {col.label: LiteralValue(cond) for col in visible}
        else:
            raise TypeError("cond must be a DataFrame, Series, or boolean")

        # Handle other: DataFrame, Series, scalar, or dict
        if isinstance(other, DataFrame):
            if other._session is not self._session or other._plan is not self._plan:
                raise AlignmentError(
                    "Other DataFrame from different frame requires explicit index alignment"
                )
            other_map = {
                col.label: ColumnRef(col.id) for col in other._plan.metadata.visible_columns
            }
        elif isinstance(other, Series):
            self._require_same_plan(other)
            other_map = {col.label: other._expression for col in visible}
        elif isinstance(other, dict):
            other_dict = cast("dict[object, object]", other)
            other_map = {
                col.label: self._coerce_expression(other_dict.get(col.label)) for col in visible
            }
        elif is_scalar_value(other):
            other_map = {col.label: LiteralValue(other) for col in visible}
        else:
            raise TypeError("other must be a scalar, Series, DataFrame, or dict")

        projections: list[NamedExpression] = []
        for col in visible:
            c_expr = cond_map.get(col.label, LiteralValue(True))
            o_expr = other_map.get(col.label, LiteralValue(None))
            val_if_true = ColumnRef(col.id) if not invert else o_expr
            val_if_false = o_expr if not invert else ColumnRef(col.id)
            expr = CaseWhen(c_expr, val_if_true, val_if_false)
            projections.append(NamedExpression(col, expr))

        for col in index_columns:
            projections.append(NamedExpression(col, ColumnRef(col.id)))

        projected_columns = tuple(p.column for p in projections)
        return DataFrame(
            self._session,
            ProjectPlan(
                self._plan,
                tuple(projections),
                after_projection(self._plan.metadata, projected_columns),
            ),
        )

    def _null_check(self, function_name: str) -> DataFrame:
        visible = self._plan.metadata.visible_columns
        index_ids = set(self._plan.metadata.index.columns)
        # Keep hidden index columns as identity pass-through so the index survives.
        index_columns = tuple(
            column
            for column in self._plan.metadata.columns
            if column.id in index_ids and column.hidden
        )
        null_projections = tuple(
            NamedExpression(
                Column(column.id, column.label, "BOOLEAN"),
                FunctionCall(function_name, (ColumnRef(column.id),)),
            )
            for column in visible
        )
        index_projections = tuple(
            NamedExpression(column, ColumnRef(column.id)) for column in index_columns
        )
        projections = (*null_projections, *index_projections)
        projected_columns = tuple(projection.column for projection in projections)
        return DataFrame(
            self._session,
            ProjectPlan(
                self._plan,
                projections,
                after_projection(self._plan.metadata, projected_columns),
            ),
        )

    @overload
    def __getitem__(self, key: str) -> Series: ...

    @overload
    def __getitem__(self, key: list[str] | tuple[str, ...]) -> DataFrame: ...

    @overload
    def __getitem__(self, key: Series) -> DataFrame: ...

    def __getitem__(self, key: str | Sequence[str] | Series) -> Series | DataFrame:
        from duckpd.series import Series

        if isinstance(key, str):
            column = self._column(key)
            return Series(self._session, self._plan, ColumnRef(column.id), key)
        if isinstance(key, Series):
            self._require_same_plan(key)
            return DataFrame(
                self._session,
                FilterPlan(self._plan, key._expression, after_filter(self._plan.metadata)),
            )

        labels = tuple(key)
        if not labels:
            raise ValueError("DuckPD does not yet support empty projections")
        if len(labels) != len(set(labels)):
            raise ValueError("DuckPD does not yet support duplicate column labels")
        projections = tuple(
            NamedExpression(column, ColumnRef(column.id))
            for column in (self._column(label) for label in labels)
        )
        selected = tuple(projection.column for projection in projections)
        columns = projection_columns(self._plan.metadata, selected)
        projections = tuple(NamedExpression(column, ColumnRef(column.id)) for column in columns)
        return DataFrame(
            self._session,
            ProjectPlan(
                self._plan,
                projections,
                after_projection(self._plan.metadata, columns),
            ),
        )

    def __setitem__(
        self,
        key: str | Sequence[str],
        value: object | Callable[[DataFrame], object],
    ) -> None:
        """Lazily assign or replace columns in this DataFrame handle."""
        if isinstance(key, str):
            updated = self.assign(**{key: value})
            self._plan = updated._plan
        else:
            labels = tuple(key)
            if not labels:
                raise ValueError("DuckPD does not support empty column keys")
            if isinstance(value, DataFrame):
                if value._session is not self._session:
                    raise AlignmentError(
                        "Assigned DataFrame from different frame requires explicit alignment"
                    )
                if value._plan is self._plan:
                    expressions = tuple(
                        ColumnRef(column.id) for column in value._plan.metadata.visible_columns
                    )
                elif value._alignment_source is self._plan:
                    expressions = value._alignment_expressions
                elif isinstance(value._plan, ProjectPlan) and value._plan.input is self._plan:
                    expressions = tuple(
                        projection.expression
                        for projection in value._plan.projections
                        if not projection.column.hidden
                    )
                else:
                    raise AlignmentError(
                        "Assigned DataFrame requires a direct same-frame projection"
                    )
                if len(labels) != len(expressions):
                    raise ValueError("Number of columns does not match number of labels")
                from duckpd.series import Series

                kwargs = {
                    label: Series(self._session, self._plan, expression, label)
                    for label, expression in zip(labels, expressions, strict=True)
                }
                updated = self.assign(**kwargs)
                self._plan = updated._plan
            elif isinstance(value, (tuple, list)):
                val_seq = list(cast("Sequence[object]", value))
                if len(labels) != len(val_seq):
                    raise ValueError("Number of values does not match number of labels")
                kwargs = dict(zip(labels, val_seq, strict=True))
                updated = self.assign(**kwargs)
                self._plan = updated._plan
            else:
                # Scalar broadcast across all specified labels
                kwargs = {label: value for label in labels}
                updated = self.assign(**kwargs)
                self._plan = updated._plan

    def assign(
        self,
        **columns: object | Callable[[DataFrame], object],
    ) -> DataFrame:
        """Assign columns sequentially while keeping the result lazy."""
        original_plan = self._plan
        frame = self
        for label, value in columns.items():
            resolved = value(frame) if callable(value) else value
            expression = frame._coerce_expression(resolved, alternate_plan=original_plan)
            try:
                existing = find_column(frame._plan.metadata, label, include_hidden=True)
            except KeyError:
                existing = None
            if existing is not None and existing.id in protected_column_ids(frame._plan.metadata):
                raise ValueError("Cannot replace an index or ordering column; reset metadata first")
            output = Column(
                ColumnId.create(),
                label,
                expression_type(frame._plan, expression),
                nullable=expression_nullability(expression, frame._plan.metadata),
                categorical=expression_categorical(frame._plan, expression),
                timezone=expression_timezone(frame._plan, expression),
                alias_of=(expression.column_id if isinstance(expression, ColumnRef) else None),
            )
            if existing is not None:
                # Replace column in-place to preserve original column order
                projections = tuple(
                    (
                        NamedExpression(output, expression)
                        if column.label == label
                        else NamedExpression(column, ColumnRef(column.id))
                    )
                    for column in frame._plan.columns
                )
            else:
                # Append new column at the end
                projections = (
                    *(
                        NamedExpression(column, ColumnRef(column.id))
                        for column in frame._plan.columns
                    ),
                    NamedExpression(output, expression),
                )
            projected_columns = tuple(projection.column for projection in projections)
            plan = ProjectPlan(
                frame._plan,
                projections,
                after_projection(frame._plan.metadata, projected_columns),
            )
            frame = DataFrame(frame._session, plan)
        return frame

    def sort_values(
        self,
        by: str | Sequence[str],
        *,
        ascending: bool | Sequence[bool] = True,
        na_position: Literal["first", "last"] = "last",
    ) -> DataFrame:
        """Return a lazy frame ordered by one or more columns."""
        labels = (by,) if isinstance(by, str) else tuple(by)
        if not labels:
            raise ValueError("sort_values requires at least one column")
        if na_position not in {"first", "last"}:
            raise ValueError("na_position must be 'first' or 'last'")

        directions = (ascending,) * len(labels) if isinstance(ascending, bool) else tuple(ascending)
        if len(directions) != len(labels):
            raise ValueError("Length of ascending must match length of by")
        categorical_columns = [self._column(label) for label in labels]
        unsupported_categories = [
            column.label
            for column in categorical_columns
            if column.categorical is not None and not column.duckdb_type.startswith("ENUM(")
        ]
        if unsupported_categories:
            raise UnsupportedOperationError(
                "Categorical sorting currently requires string categories; "
                f"unsupported columns: {unsupported_categories!r}"
            )

        null_placement = NullPlacement.FIRST if na_position == "first" else NullPlacement.LAST
        keys = tuple(
            SortKey(
                ColumnRef(self._column(label).id),
                SortDirection.ASCENDING if direction else SortDirection.DESCENDING,
                null_placement,
            )
            for label, direction in zip(labels, directions, strict=True)
        )
        selected_ids = {
            key.expression.column_id for key in keys if isinstance(key.expression, ColumnRef)
        }
        identity_ids = set(self._plan.metadata.row_identity.columns)
        stable_keys = tuple(
            SortKey(ColumnRef(key.column_id), key.direction, key.null_placement)
            for key in self._plan.metadata.ordering.keys
            if key.column_id not in selected_ids and key.column_id in identity_ids
        )
        keys = (*keys, *stable_keys)
        return DataFrame(
            self._session,
            SortPlan(self._plan, keys, after_sort(self._plan.metadata, keys)),
        )

    def groupby(
        self,
        by: str | Sequence[str],
        *,
        as_index: bool = True,
        sort: bool = True,
        dropna: bool = True,
        observed: bool = True,
    ) -> DataFrameGroupBy:
        """Group DataFrame using a mapper or by a Series of columns."""
        from duckpd.groupby import DataFrameGroupBy

        return DataFrameGroupBy(
            self,
            by=by,
            as_index=as_index,
            sort=sort,
            dropna=dropna,
            observed=observed,
        )

    def merge(
        self,
        right: DataFrame,
        how: Literal["left", "right", "outer", "inner", "cross"] = "inner",
        on: str | Sequence[str] | None = None,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = False,
        suffixes: tuple[str | None, str | None] = ("_x", "_y"),
        validate: str | None = None,
    ) -> DataFrame:
        """Merge DataFrame or named Series objects with a database-style join."""
        from duckpd._merging import plan_merge

        if self._session is not right._session:
            raise AlignmentError("Cannot merge frames from different sessions")

        plan = plan_merge(
            self,
            right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            validate=validate,
        )
        return DataFrame(self._session, plan)

    def join(
        self,
        other: DataFrame,
        on: str | Sequence[str] | None = None,
        how: Literal["left", "right", "outer", "inner"] = "left",
        lsuffix: str = "",
        rsuffix: str = "",
        sort: bool = False,
        validate: str | None = None,
    ) -> DataFrame:
        """Join columns of another DataFrame using index or a key column."""
        return self.merge(
            other,
            how=how,
            left_on=on,
            right_index=True,
            left_index=(on is None),
            sort=sort,
            suffixes=(lsuffix, rsuffix),
            validate=validate,
        )

    def limit(self, count: int, *, offset: int = 0) -> DataFrame:
        """Return a lazy frame containing at most ``count`` rows."""
        if count < 0:
            raise ValueError("count must be non-negative")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        return DataFrame(
            self._session,
            LimitPlan(self._plan, count, offset, self._plan.metadata),
        )

    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights: object = None,
        random_state: int | None = None,
        axis: int | str | None = None,
        ignore_index: bool = False,
    ) -> DataFrame:
        """Return a random sample of rows from the DataFrame.

        Builds an immutable lazy plan using DuckDB reservoir sampling.
        """
        if axis not in (0, "index", None):
            raise UnsupportedOperationError("DuckPD sample supports only axis=0 or axis='index'")
        if replace is not False:
            raise UnsupportedOperationError("DuckPD sample does not currently support replace=True")
        if weights is not None:
            raise UnsupportedOperationError("DuckPD sample does not currently support weights")
        if n is not None and frac is not None:
            raise ValueError("Only one of 'n' or 'frac' can be specified")
        if n is None and frac is None:
            n = 1
        if n is not None:
            if isinstance(n, bool) or not isinstance(cast("object", n), int):
                raise ValueError(f"'n' must be an integer, got {type(n).__name__}")
            if n < 0:
                raise ValueError("A negative number of rows was requested")
        if frac is not None:
            if isinstance(frac, bool) or not isinstance(cast("object", frac), (int, float)):
                raise ValueError(f"'frac' must be a float, got {type(frac).__name__}")
            if not isfinite(float(frac)):
                raise ValueError("'frac' must be finite")
            if frac < 0.0:
                raise ValueError("A negative fraction of rows was requested")
            if frac > 1.0:
                raise ValueError("Replace has to be set to True when frac > 1")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(cast("object", random_state), int)
        ):
            raise ValueError("random_state must be an integer seed or None")
        if random_state is not None and not 0 <= random_state <= 2_147_483_647:
            raise ValueError("random_state must be between 0 and 2**31 - 1")

        metadata = dataclass_replace(self._plan.metadata, ordering=OrderSpec())
        if ignore_index:
            if metadata.index.columns:
                metadata = reset_index_metadata(metadata, drop=True)
            else:
                metadata = dataclass_replace(
                    metadata,
                    index=IndexSpec(),
                    ordering=OrderSpec(),
                )

        plan = SamplePlan(
            input=self._plan,
            n=n,
            frac=float(frac) if frac is not None else None,
            seed=random_state,
            metadata=metadata,
        )
        return DataFrame(self._session, plan)

    def set_index(self, keys: str | Sequence[str], *, drop: bool = True) -> DataFrame:
        """Set one or more existing columns as an explicit lazy index."""
        labels = (keys,) if isinstance(keys, str) else tuple(keys)
        if not labels:
            raise ValueError("set_index requires at least one column")
        if len(labels) != len(set(labels)):
            raise ValueError("set_index keys must be unique")
        columns = tuple(self._column(label) for label in labels)
        metadata = set_index_metadata(self._plan.metadata, columns, drop=drop)
        return self._identity_project(metadata)

    def reset_index(self, *, drop: bool = False) -> DataFrame:
        """Remove the explicit index without executing the plan."""
        metadata = reset_index_metadata(self._plan.metadata, drop=drop)
        return self._identity_project(metadata)

    def rename(
        self,
        columns: dict[str, str] | None = None,
        *,
        mapper: object | None = None,
        axis: int | str = 0,
        index: object | None = None,
        copy: bool = True,
        inplace: bool = False,
        level: object | None = None,
        errors: Literal["ignore", "raise"] = "raise",
    ) -> DataFrame:
        """Rename columns or index labels lazily."""
        if not copy:
            raise UnsupportedOperationError("DuckPD does not support copy=False")
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if level is not None:
            raise UnsupportedOperationError("DuckPD does not support MultiIndex levels")
        if index is not None:
            raise UnsupportedOperationError("DuckPD does not yet support renaming index labels")
        if axis not in {0, 1, "index", "columns"}:
            raise ValueError("axis must be 0, 'index', 1, or 'columns'")
        if mapper is not None and columns is not None:
            raise TypeError("Cannot specify both mapper and columns")
        mapping: dict[str, str] = {}
        source = mapper if mapper is not None else columns
        if source is None:
            return self
        if not isinstance(source, dict):
            raise TypeError("rename currently supports only a dict mapping")
        if mapper is not None and axis in {0, "index"}:
            raise UnsupportedOperationError("DuckPD does not yet support renaming index labels")
        mapping: dict[str, str] = {}
        for old_item, new_item in cast("dict[object, object]", source).items():
            if not isinstance(old_item, str) or not isinstance(new_item, str):
                raise TypeError("rename mapping keys and values must be strings")
            mapping[old_item] = new_item
        if not mapping:
            return self
        if errors not in {"ignore", "raise"}:
            raise ValueError("errors must be 'ignore' or 'raise'")

        existing_labels = {column.label for column in self._plan.metadata.columns}
        for old in mapping:
            if old not in existing_labels and errors == "raise":
                raise KeyError(old)

        from dataclasses import replace as replace_column

        new_columns = tuple(
            replace_column(column, label=mapping[column.label])
            if column.label in mapping
            else column
            for column in self._plan.metadata.columns
        )
        new_labels = [column.label for column in new_columns]
        if len(new_labels) != len(set(new_labels)):
            raise ValueError("rename would create duplicate column labels")
        new_metadata = dataclass_replace(self._plan.metadata, columns=new_columns)
        return self._identity_project(new_metadata)

    def drop(
        self,
        labels: str | Sequence[str] | None = None,
        *,
        axis: int | str = 0,
        columns: str | Sequence[str] | None = None,
        index: object | None = None,
        level: object | None = None,
        inplace: bool = False,
        errors: Literal["ignore", "raise"] = "raise",
    ) -> DataFrame:
        """Drop columns or rows lazily."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if level is not None:
            raise UnsupportedOperationError("DuckPD does not support MultiIndex levels")
        if index is not None:
            raise UnsupportedOperationError("DuckPD does not yet support dropping rows by index")
        if axis not in {0, 1, "index", "columns"}:
            raise ValueError("axis must be 0, 'index', 1, or 'columns'")
        if errors not in {"ignore", "raise"}:
            raise ValueError("errors must be 'ignore' or 'raise'")

        if columns is not None:
            drop_labels = (columns,) if isinstance(columns, str) else tuple(columns)
        elif labels is None:
            raise TypeError("must specify either labels or columns")
        elif axis in {0, "index"}:
            raise UnsupportedOperationError("DuckPD does not yet support dropping rows by index")
        else:
            drop_labels = (labels,) if isinstance(labels, str) else tuple(labels)

        if not drop_labels:
            return self

        existing_labels = {column.label for column in self._plan.metadata.columns}
        for label in drop_labels:
            if label not in existing_labels and errors == "raise":
                raise KeyError(label)

        drop_set = set(drop_labels)
        kept_visible = tuple(
            column for column in self._plan.metadata.visible_columns if column.label not in drop_set
        )
        if not kept_visible:
            raise ValueError("DuckPD does not yet support empty projections")
        projections = tuple(
            NamedExpression(column, ColumnRef(column.id)) for column in kept_visible
        )
        selected = tuple(projection.column for projection in projections)
        columns_to_keep = projection_columns(self._plan.metadata, selected)
        projections = tuple(
            NamedExpression(column, ColumnRef(column.id)) for column in columns_to_keep
        )
        return DataFrame(
            self._session,
            ProjectPlan(
                self._plan,
                projections,
                after_projection(self._plan.metadata, columns_to_keep),
            ),
        )

    def cumsum(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame:
        """Return cumulative sum over a DataFrame axis."""
        validate_axis(axis, series=False)
        return self._cumulative_transform("cumsum", skipna=skipna, numeric_only=numeric_only)

    def cummin(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame:
        """Return cumulative minimum over a DataFrame axis."""
        validate_axis(axis, series=False)
        return self._cumulative_transform("cummin", skipna=skipna, numeric_only=numeric_only)

    def cummax(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame:
        """Return cumulative maximum over a DataFrame axis."""
        validate_axis(axis, series=False)
        return self._cumulative_transform("cummax", skipna=skipna, numeric_only=numeric_only)

    def cumprod(
        self,
        *,
        axis: int | str | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame:
        """Return cumulative product over a DataFrame axis."""
        validate_axis(axis, series=False)
        return self._cumulative_transform("cumprod", skipna=skipna, numeric_only=numeric_only)

    def _cumulative_transform(
        self,
        name: Literal["cumsum", "cummin", "cummax", "cumprod"],
        *,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame:
        from duckpd.series import Series

        visible = self._plan.metadata.visible_columns
        protected = [col for col in self._plan.metadata.columns if col.hidden]
        new_projections: list[NamedExpression] = []

        for col in visible:
            is_num = is_numeric_type(col.duckdb_type)
            if numeric_only and not is_num:
                continue
            s = Series(self._session, self._plan, ColumnRef(col.id), col.label)
            method = getattr(s, name)
            res_s: Series = method(skipna=skipna)
            out_col = Column(
                ColumnId.create(),
                col.label,
                expression_type(self._plan, res_s._expression),
            )
            new_projections.append(NamedExpression(out_col, res_s._expression))

        for col in protected:
            new_projections.append(NamedExpression(col, ColumnRef(col.id)))

        output_columns = tuple(p.column for p in new_projections)
        metadata = after_projection(self._plan.metadata, output_columns)
        return DataFrame(
            self._session,
            ProjectPlan(self._plan, tuple(new_projections), metadata),
        )

    def shift(
        self,
        periods: int = 1,
        *,
        freq: object = None,
        axis: int | str | None = 0,
        fill_value: object = None,
    ) -> DataFrame:
        """Shift index by desired number of periods."""
        validate_axis(axis, series=False)
        from duckpd.series import Series

        visible = self._plan.metadata.visible_columns
        protected = [col for col in self._plan.metadata.columns if col.hidden]
        new_projections: list[NamedExpression] = []

        for col in visible:
            s = Series(self._session, self._plan, ColumnRef(col.id), col.label)
            res_s = s.shift(periods=periods, freq=freq, fill_value=fill_value)
            out_col = Column(
                ColumnId.create(),
                col.label,
                expression_type(self._plan, res_s._expression),
            )
            new_projections.append(NamedExpression(out_col, res_s._expression))

        for col in protected:
            new_projections.append(NamedExpression(col, ColumnRef(col.id)))

        output_columns = tuple(p.column for p in new_projections)
        metadata = after_projection(self._plan.metadata, output_columns)
        return DataFrame(
            self._session,
            ProjectPlan(self._plan, tuple(new_projections), metadata),
        )

    def diff(
        self,
        periods: int = 1,
        *,
        axis: int | str | None = 0,
    ) -> DataFrame:
        """First discrete difference of element."""
        validate_axis(axis, series=False)
        from duckpd.series import Series

        visible = self._plan.metadata.visible_columns
        protected = [col for col in self._plan.metadata.columns if col.hidden]
        new_projections: list[NamedExpression] = []

        for col in visible:
            s = Series(self._session, self._plan, ColumnRef(col.id), col.label)
            res_s = s.diff(periods=periods)
            out_col = Column(
                ColumnId.create(),
                col.label,
                expression_type(self._plan, res_s._expression),
            )
            new_projections.append(NamedExpression(out_col, res_s._expression))

        for col in protected:
            new_projections.append(NamedExpression(col, ColumnRef(col.id)))

        output_columns = tuple(p.column for p in new_projections)
        metadata = after_projection(self._plan.metadata, output_columns)
        return DataFrame(
            self._session,
            ProjectPlan(self._plan, tuple(new_projections), metadata),
        )

    def pct_change(
        self,
        periods: int = 1,
        *,
        fill_method: object = None,
        limit: object = None,
        freq: object = None,
        axis: int | str | None = 0,
    ) -> DataFrame:
        """Percentage change between the current and a prior element."""
        validate_axis(axis, series=False)
        from duckpd.series import Series

        visible = self._plan.metadata.visible_columns
        protected = [col for col in self._plan.metadata.columns if col.hidden]
        new_projections: list[NamedExpression] = []

        for col in visible:
            s = Series(self._session, self._plan, ColumnRef(col.id), col.label)
            res_s = s.pct_change(
                periods=periods,
                fill_method=fill_method,
                limit=limit,
                freq=freq,
            )
            out_col = Column(
                ColumnId.create(),
                col.label,
                expression_type(self._plan, res_s._expression),
            )
            new_projections.append(NamedExpression(out_col, res_s._expression))

        for col in protected:
            new_projections.append(NamedExpression(col, ColumnRef(col.id)))

        output_columns = tuple(p.column for p in new_projections)
        metadata = after_projection(self._plan.metadata, output_columns)
        return DataFrame(
            self._session,
            ProjectPlan(self._plan, tuple(new_projections), metadata),
        )

    def rank(
        self,
        axis: int | str | None = 0,
        method: Literal["average", "min", "max", "first", "dense"] = "average",
        numeric_only: bool = False,
        na_option: Literal["keep", "top", "bottom"] = "keep",
        ascending: bool = True,
        pct: bool = False,
    ) -> DataFrame:
        """Compute numerical data ranks (1 through n) along axis."""
        validate_axis(axis, series=False)
        from duckpd.series import Series

        visible = self._plan.metadata.visible_columns
        protected = [col for col in self._plan.metadata.columns if col.hidden]
        new_projections: list[NamedExpression] = []

        for col in visible:
            is_num = is_numeric_type(col.duckdb_type)
            if numeric_only and not is_num:
                continue
            s = Series(self._session, self._plan, ColumnRef(col.id), col.label)
            res_s = s.rank(
                axis=axis,
                method=method,
                numeric_only=False,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            )
            out_col = Column(
                ColumnId.create(),
                col.label,
                expression_type(self._plan, res_s._expression),
            )
            new_projections.append(NamedExpression(out_col, res_s._expression))

        for col in protected:
            new_projections.append(NamedExpression(col, ColumnRef(col.id)))

        output_columns = tuple(p.column for p in new_projections)
        metadata = after_projection(self._plan.metadata, output_columns)
        return DataFrame(
            self._session,
            ProjectPlan(self._plan, tuple(new_projections), metadata),
        )

    def head(self, count: int = 5) -> pd.DataFrame:
        """Execute a bounded preview and return a pandas DataFrame."""
        if count < 0:
            raise ValueError("count must be non-negative")
        if not self._plan.metadata.ordering.keys:
            raise UnorderedOperationError(
                "head() requires a guaranteed row ordering. Specify "
                "order_by when creating a SQL/table source or sort first using "
                ".sort_values(...)"
            )
        return self.limit(count).collect()

    def nunique(
        self,
        *,
        axis: int | str = 0,
        dropna: bool = True,
    ) -> pd.Series:
        """Return the number of unique values per column."""
        if axis not in {0, "index"}:
            raise UnsupportedOperationError(
                "DuckPD nunique currently supports only axis=0 or axis='index'"
            )
        visible = self._plan.metadata.visible_columns
        if not visible:
            raise UnsupportedOperationError("No columns are available for nunique")
        requests = tuple(
            (column.label, ColumnRef(column.id), column.duckdb_type) for column in visible
        )
        plan = aggregate_plan(self._plan, requests, AggregateOperator.NUNIQUE)
        result = self._session._executor.reduce_columns(plan)
        if not dropna:
            raise UnsupportedOperationError("DuckPD nunique currently supports only dropna=True")
        return result

    def rolling(
        self,
        window: int | str | timedelta,
        min_periods: int | None = None,
        *,
        center: bool = False,
        on: str | None = None,
        closed: Literal["right", "left", "both", "neither"] | None = None,
    ) -> Rolling:
        """Provide row-count or fixed-duration rolling window calculations."""
        from duckpd.window import Rolling

        return Rolling(
            self,
            window,
            min_periods=min_periods,
            center=center,
            on=on,
            closed=closed,
        )

    def expanding(
        self,
        min_periods: int = 1,
    ) -> Expanding:
        """Provide expanding window calculations."""
        from duckpd.window import Expanding

        return Expanding(self, min_periods=min_periods)

    def drop_duplicates(
        self,
        subset: str | Sequence[str] | None = None,
        *,
        keep: Literal["first", "last", False] = "first",
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> DataFrame:
        """Return DataFrame with duplicate rows removed."""
        if inplace:
            raise UnsupportedOperationError("DuckPD does not support inplace=True")
        if ignore_index:
            raise UnsupportedOperationError(
                "DuckPD does not support ignore_index=True in drop_duplicates"
            )
        if keep not in {"first", "last", False}:
            raise ValueError("keep must be 'first', 'last', or False")

        if subset is None:
            subset_labels = tuple(self.columns)
        elif isinstance(subset, str):
            subset_labels = (subset,)
        else:
            subset_labels = tuple(subset)

        if not subset_labels:
            raise ValueError("subset must not be empty")

        subset_columns = tuple(self._column(label) for label in subset_labels)

        ordering_keys = self._plan.metadata.ordering.keys
        if keep in {"first", "last"} and not ordering_keys:
            raise UnorderedOperationError(
                "drop_duplicates keep='first'/'last' requires a guaranteed row "
                "ordering. Specify order_by when creating a SQL/table source or "
                "sort first using .sort_values(...)"
            )

        # Window-based deduplication
        partition_exprs = tuple(ColumnRef(col.id) for col in subset_columns)
        order_by_keys: tuple[SortKey, ...]
        if ordering_keys:
            order_by_keys = tuple(
                SortKey(
                    ColumnRef(k.column_id),
                    (
                        k.direction
                        if keep != "last"
                        else (
                            SortDirection.DESCENDING
                            if k.direction is SortDirection.ASCENDING
                            else SortDirection.ASCENDING
                        )
                    ),
                    (
                        k.null_placement
                        if keep != "last"
                        else (
                            NullPlacement.LAST
                            if k.null_placement is NullPlacement.FIRST
                            else NullPlacement.FIRST
                        )
                    ),
                )
                for k in ordering_keys
            )
        else:
            order_by_keys = ()

        if keep in {"first", "last"}:
            row_num = WindowExpression(
                function="row_number",
                partition_by=partition_exprs,
                order_by=order_by_keys,
            )
            dedup_col = Column(ColumnId.create(), "__duckpd_rn__", "BIGINT", hidden=True)
            proj_plan = ProjectPlan(
                self._plan,
                (
                    *(
                        NamedExpression(col, ColumnRef(col.id))
                        for col in self._plan.metadata.columns
                    ),
                    NamedExpression(dedup_col, row_num),
                ),
                after_projection(
                    self._plan.metadata,
                    (*self._plan.metadata.columns, dedup_col),
                ),
            )
            predicate: Expression = BinaryExpression(
                ColumnRef(dedup_col.id),
                BinaryOperator.EQUAL,
                LiteralValue(1),
            )
            filter_plan = FilterPlan(proj_plan, predicate, after_filter(proj_plan.metadata))
            return DataFrame(self._session, filter_plan)
        else:  # keep is False
            cnt = WindowExpression(
                function="count",
                arguments=(LiteralValue(1),),
                partition_by=partition_exprs,
            )
            dedup_col = Column(ColumnId.create(), "__duckpd_cnt__", "BIGINT", hidden=True)
            proj_plan = ProjectPlan(
                self._plan,
                (
                    *(
                        NamedExpression(col, ColumnRef(col.id))
                        for col in self._plan.metadata.columns
                    ),
                    NamedExpression(dedup_col, cnt),
                ),
                after_projection(
                    self._plan.metadata,
                    (*self._plan.metadata.columns, dedup_col),
                ),
            )
            predicate = BinaryExpression(
                ColumnRef(dedup_col.id),
                BinaryOperator.EQUAL,
                LiteralValue(1),
            )
            filter_plan = FilterPlan(proj_plan, predicate, after_filter(proj_plan.metadata))
            return DataFrame(self._session, filter_plan)

    def nlargest(
        self,
        n: int,
        columns: str | Sequence[str],
        *,
        keep: Literal["first", "last", "all"] = "first",
    ) -> DataFrame:
        """Return the first ``n`` rows ordered by ``columns`` in descending order."""
        return self._top_n(n, columns, largest=True, keep=keep)

    def nsmallest(
        self,
        n: int,
        columns: str | Sequence[str],
        *,
        keep: Literal["first", "last", "all"] = "first",
    ) -> DataFrame:
        """Return the first ``n`` rows ordered by ``columns`` in ascending order."""
        return self._top_n(n, columns, largest=False, keep=keep)

    def _top_n(
        self,
        n: int,
        columns: str | Sequence[str],
        *,
        largest: bool,
        keep: str,
    ) -> DataFrame:
        if n < 0:
            raise ValueError("n must be non-negative")
        if keep not in {"first", "last", "all"}:
            raise ValueError("keep must be 'first', 'last', or 'all'")
        if keep == "all":
            raise UnsupportedOperationError(
                "DuckPD does not yet support keep='all' in nlargest/nsmallest"
            )
        if not self._plan.metadata.ordering.keys:
            raise UnorderedOperationError(
                "nlargest/nsmallest tie handling requires a guaranteed row "
                "ordering. Specify order_by when creating a SQL/table source or "
                "sort first using .sort_values(...)"
            )
        labels = (columns,) if isinstance(columns, str) else tuple(columns)
        value_direction = SortDirection.DESCENDING if largest else SortDirection.ASCENDING
        keys = tuple(
            SortKey(
                ColumnRef(self._column(label).id),
                value_direction,
                NullPlacement.LAST,
            )
            for label in labels
        )
        value_ids = {self._column(label).id for label in labels}
        for ordering_key in self._plan.metadata.ordering.keys:
            if ordering_key.column_id in value_ids:
                continue
            direction = ordering_key.direction
            null_placement = ordering_key.null_placement
            if keep == "last":
                direction = (
                    SortDirection.DESCENDING
                    if direction is SortDirection.ASCENDING
                    else SortDirection.ASCENDING
                )
                null_placement = (
                    NullPlacement.LAST
                    if null_placement is NullPlacement.FIRST
                    else NullPlacement.FIRST
                )
            keys = (
                *keys,
                SortKey(ColumnRef(ordering_key.column_id), direction, null_placement),
            )
        sorted_frame = DataFrame(
            self._session,
            SortPlan(self._plan, keys, after_sort(self._plan.metadata, keys)),
        )
        return sorted_frame.limit(n)

    def _binary(
        self,
        other: object,
        operator: BinaryOperator,
        *,
        reverse: bool = False,
    ) -> DataFrame:
        if isinstance(other, DataFrame):
            return self._frame_binary(other, operator, reverse=reverse)
        if not is_scalar_value(other):
            raise TypeError("DuckPD DataFrame arithmetic requires a scalar or DataFrame operand")
        scalar = LiteralValue(other)
        outputs: list[tuple[Column, Expression]] = []
        for source in self._plan.metadata.visible_columns:
            if not is_numeric_type(source.duckdb_type):
                raise UnsupportedOperationError(
                    "DataFrame arithmetic currently supports only numeric columns"
                )
            expression = BinaryExpression(
                scalar if reverse else ColumnRef(source.id),
                operator,
                ColumnRef(source.id) if reverse else scalar,
            )
            outputs.append(
                (
                    Column(
                        ColumnId.create(),
                        source.label,
                        expression_type(self._plan, expression),
                    ),
                    expression,
                )
            )
        return self._project_binary_outputs(self._plan, outputs)

    def _frame_binary(
        self,
        other: DataFrame,
        operator: BinaryOperator,
        *,
        reverse: bool,
    ) -> DataFrame:
        from duckpd._merging import plan_merge, validate_explicit_index_alignment

        if other._session is not self._session:
            raise AlignmentError("Cannot align frames from different sessions")
        if other._plan is self._plan:
            plan = self._plan
        else:
            validate_explicit_index_alignment(self, other)
            plan = plan_merge(
                self,
                other,
                how="outer",
                left_index=True,
                right_index=True,
                sort=True,
                validate="1:1",
            )

        left_columns = {column.label: column for column in self._plan.metadata.visible_columns}
        right_columns = {column.label: column for column in other._plan.metadata.visible_columns}
        outputs: list[tuple[Column, Expression]] = []
        for label in sorted(left_columns.keys() | right_columns.keys()):
            left = left_columns.get(label)
            right = right_columns.get(label)
            if left is None or right is None:
                expression: Expression = CastExpression(LiteralValue(None), "DOUBLE")
                output_type = "DOUBLE"
            else:
                if not is_numeric_type(left.duckdb_type) or not is_numeric_type(right.duckdb_type):
                    raise UnsupportedOperationError(
                        "Cross-frame DataFrame arithmetic currently supports only "
                        "numeric columns shared by both operands"
                    )
                expression = BinaryExpression(
                    ColumnRef(right.id if reverse else left.id),
                    operator,
                    ColumnRef(left.id if reverse else right.id),
                )
                output_type = expression_type(plan, expression)
            outputs.append(
                (
                    Column(
                        ColumnId.create(),
                        label,
                        output_type,
                        nullable=expression_nullability(expression, plan.metadata),
                        alias_of=(
                            expression.column_id if isinstance(expression, ColumnRef) else None
                        ),
                    ),
                    expression,
                )
            )
        return self._project_binary_outputs(plan, outputs)

    def _project_binary_outputs(
        self,
        plan: LogicalPlan,
        outputs: Sequence[tuple[Column, Expression]],
    ) -> DataFrame:
        visible = tuple(column for column, _ in outputs)
        columns = projection_columns(plan.metadata, visible)
        expressions = {column.id: expression for column, expression in outputs}
        projections = tuple(
            NamedExpression(
                column,
                expressions.get(column.id, ColumnRef(column.id)),
            )
            for column in columns
        )
        metadata = after_projection(plan.metadata, columns)
        return DataFrame(
            self._session,
            ProjectPlan(plan, projections, metadata),
        )

    def __repr__(self) -> str:
        labels = ", ".join(repr(label) for label in self.columns)
        plan_name = type(self._plan).__name__
        return f"DuckPD DataFrame\nColumns: [{labels}]\nPlan: {plan_name}"

    def _column(self, label: str) -> Column:
        return find_column(self._plan.metadata, label)

    def _column_by_id(self, column_id: ColumnId) -> Column:
        for column in self._plan.columns:
            if column.id == column_id:
                return column
        raise AssertionError(f"Column ID is missing from metadata: {column_id}")

    def _identity_project(self, metadata: object) -> DataFrame:
        if not isinstance(metadata, FrameMetadata):
            raise TypeError("Expected FrameMetadata")
        projections = tuple(
            NamedExpression(column, ColumnRef(column.id)) for column in metadata.columns
        )
        return DataFrame(
            self._session,
            ProjectPlan(self._plan, projections, metadata),
        )

    def _require_same_plan(self, series: Series) -> None:
        if series._session is not self._session or series._plan is not self._plan:
            raise AlignmentError("Series from a different frame requires explicit index alignment")

    def _coerce_expression(
        self, value: object, *, alternate_plan: LogicalPlan | None = None
    ) -> Expression:
        from duckpd.series import Series

        if isinstance(value, Series):
            same_session = value._session is self._session
            if same_session and value._alignment_source is self._plan:
                if value._alignment_expression is None:
                    raise AssertionError("Aligned Series is missing its source expression")
                return value._alignment_expression
            compatible_plan = value._plan is self._plan or value._plan is alternate_plan
            if not same_session or not compatible_plan:
                raise AlignmentError(
                    "Series from a different frame requires explicit index alignment"
                )
            return value._expression
        if isinstance(value, DataFrame):
            raise TypeError("assign values must be scalar or Series expressions")
        if not is_scalar_value(value):
            raise TypeError("DuckPD does not support this scalar literal type")
        return LiteralValue(value)

    def _reduction_columns(
        self,
        *,
        numeric_only: bool,
        require_numeric: bool = False,
    ) -> tuple[Column, ...]:
        visible = self._plan.metadata.visible_columns
        numeric = tuple(column for column in visible if is_numeric_type(column.duckdb_type))
        if numeric_only:
            if not numeric:
                raise UnsupportedOperationError(
                    "No numeric columns are available for this reduction"
                )
            return numeric
        if require_numeric and len(numeric) != len(visible):
            unsupported = next(
                column for column in visible if not is_numeric_type(column.duckdb_type)
            )
            raise UnsupportedOperationError(
                "This reduction currently supports only numeric and boolean data; "
                f"column {unsupported.label!r} has DuckDB type "
                f"{unsupported.duckdb_type}. Pass numeric_only=True to exclude it."
            )
        return visible

    def _reduce_columns(
        self,
        operator: AggregateOperator,
        columns: tuple[Column, ...],
        *,
        skipna: bool = True,
        min_count: int = 0,
        ddof: int = 1,
        q: float = 0.5,
    ) -> pd.Series:
        requests = tuple(
            (column.label, ColumnRef(column.id), column.duckdb_type) for column in columns
        )
        plan = aggregate_plan(
            self._plan,
            requests,
            operator,
            skipna=skipna,
            min_count=min_count,
            ddof=ddof,
            q=q,
        )
        return self._session._executor.reduce_columns(plan)
