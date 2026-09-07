"""Typed lazy vector distance and nearest-neighbor operations."""

from __future__ import annotations

import re
from collections.abc import Iterable
from decimal import Decimal
from math import isfinite
from typing import TYPE_CHECKING, Literal, cast

from duckpd._logical import (
    Column,
    ColumnId,
    FrameMetadata,
    Nullability,
    SemanticSearchPlan,
    VectorDistanceExpression,
    VectorExecutionMode,
    VectorMetric,
    VectorQuery,
    VectorSearchPlan,
)
from duckpd._metadata import after_vector_search, find_column
from duckpd._reductions import expression_type
from duckpd.errors import UnsupportedOperationError

if TYPE_CHECKING:
    from duckpd.embeddings import EmbeddingModelSpec
    from duckpd.frame import DataFrame
    from duckpd.series import Series

VectorMetricName = Literal["cosine", "l2", "inner_product"]
VectorMode = Literal["exact", "approximate"]

_VECTOR_TYPE = re.compile(r"^(FLOAT|DOUBLE)\[(\d*)\]$")
_ORDERABLE_PREFIXES = (
    "BOOLEAN",
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL(",
    "VARCHAR",
    "DATE",
    "TIME",
    "TIMESTAMP",
)


def _vector_type(dtype: str) -> tuple[Literal["FLOAT", "DOUBLE"], int | None]:
    match = _VECTOR_TYPE.fullmatch(dtype.upper().replace(" ", ""))
    if match is None:
        raise UnsupportedOperationError(
            "Vector operations require a FLOAT[n], DOUBLE[n], FLOAT[], or DOUBLE[] column; "
            f"got {dtype}"
        )
    element_type = cast("Literal['FLOAT', 'DOUBLE']", match.group(1))
    dimension = int(match.group(2)) if match.group(2) else None
    if dimension == 0:
        raise UnsupportedOperationError("Vector columns must have a positive dimension")
    return element_type, dimension


def _query_vector(
    query: object,
    *,
    element_type: Literal["FLOAT", "DOUBLE"],
    expected_dimension: int | None,
) -> VectorQuery:
    if isinstance(query, (str, bytes, bytearray)):
        raise TypeError("query must be a one-dimensional numeric sequence")
    try:
        raw = tuple(cast("Iterable[object]", query))
    except TypeError:
        raise TypeError("query must be a one-dimensional numeric sequence") from None
    if not raw:
        raise ValueError("query vector must not be empty")
    values: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise TypeError("query vector elements must be numeric scalars")
        coerced = float(value)
        if not isfinite(coerced):
            raise ValueError("query vector elements must be finite")
        values.append(coerced)
    if expected_dimension is not None and len(values) != expected_dimension:
        raise ValueError(
            f"query vector dimension {len(values)} does not match column dimension "
            f"{expected_dimension}"
        )
    return VectorQuery(tuple(values), element_type)


def _validate_query_metric(query: VectorQuery, metric: VectorMetric) -> None:
    if metric is VectorMetric.COSINE and not any(value != 0.0 for value in query.values):
        raise ValueError("cosine distance requires a query vector with non-zero norm")


def _metric(metric: str) -> VectorMetric:
    try:
        return VectorMetric(metric)
    except ValueError:
        supported = ", ".join(item.value for item in VectorMetric)
        raise ValueError(
            f"Unsupported vector metric {metric!r}; expected one of {supported}"
        ) from None


class VectorMethods:
    """Lazy distance operations for a fixed-size numeric-array Series."""

    def __init__(self, series: Series) -> None:
        self._series = series

    def distance(
        self,
        query: object,
        *,
        metric: VectorMetricName = "cosine",
    ) -> Series:
        """Return the exact distance to one finite query vector lazily.

        ``cosine`` maps to DuckDB ``array_cosine_distance``; ``l2`` maps to
        ``array_distance``; ``inner_product`` maps to
        ``array_negative_inner_product`` so smaller values are always nearer.
        Null vectors, vectors with null elements, non-finite vectors, and
        runtime-sized vectors of the wrong dimension fail during the single
        execution query instead of being silently removed.
        """
        from duckpd.series import Series

        dtype = expression_type(self._series._plan, self._series._expression)
        element_type, dimension = _vector_type(dtype)
        vector_metric = _metric(metric)
        vector_query = _query_vector(
            query,
            element_type=element_type,
            expected_dimension=dimension,
        )
        _validate_query_metric(vector_query, vector_metric)
        return Series(
            self._series._session,
            self._series._plan,
            VectorDistanceExpression(
                self._series._expression,
                vector_query,
                vector_metric,
                dtype,
            ),
            self._series.name,
        )


def _search_metadata(
    frame: DataFrame,
    *,
    metric: str,
    k: int,
    distance_column: str,
    tie_breaker: str | None,
    distance_type: str = "FLOAT",
) -> tuple[VectorMetric, Column, Column | None, FrameMetadata]:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if not distance_column:
        raise ValueError("distance_column must be a non-empty string label")
    if distance_column in frame.columns:
        raise ValueError(f"distance column {distance_column!r} already exists")
    vector_metric = _metric(metric)
    tie_column = None
    if tie_breaker is not None:
        tie_column = find_column(frame._plan.metadata, tie_breaker)
        tie_type = tie_column.duckdb_type.upper()
        if "[" in tie_type or not tie_type.startswith(_ORDERABLE_PREFIXES):
            raise UnsupportedOperationError(
                f"tie_breaker column {tie_breaker!r} is not orderable: {tie_column.duckdb_type}"
            )
    distance = Column(
        ColumnId.create(),
        distance_column,
        distance_type,
        nullable=Nullability.NON_NULL,
    )
    metadata = after_vector_search(frame._plan.metadata, distance, tie_breaker=tie_column)
    return vector_metric, distance, tie_column, metadata


class VectorFrameMethods:
    """Lazy nearest-neighbor retrieval for a DataFrame."""

    def __init__(self, frame: DataFrame) -> None:
        self._frame = frame

    def search_text(
        self,
        query: str,
        *,
        column: str,
        model: EmbeddingModelSpec,
        metric: VectorMetricName = "cosine",
        k: int = 10,
        batch_size: int = 256,
        distance_column: str = "_distance",
        tie_breaker: str | None = None,
    ) -> DataFrame:
        """Lazily embed one text query against a verified embedding column."""
        from duckpd.embeddings import _embedding_settings
        from duckpd.frame import DataFrame

        if not query:
            raise ValueError("query must be a non-empty string")
        vector_column = find_column(self._frame._plan.metadata, column)
        _, dimension = _vector_type(vector_column.duckdb_type)
        if dimension is not None and dimension != model.dimension:
            raise ValueError(
                f"model dimension {model.dimension} does not match column dimension {dimension}"
            )
        if (
            vector_column.embedding is None
            or vector_column.embedding.fingerprint != model.fingerprint
        ):
            raise UnsupportedOperationError(
                "search_text requires embedding metadata matching the requested model"
            )
        settings = _embedding_settings(
            self._frame,
            model=model,
            batch_size=batch_size,
            separator="",
            null_policy="error",
            output_label=None,
        )
        query_key = self._frame._session._register_embedding_query(model, query)
        vector_metric, distance, tie_column, metadata = _search_metadata(
            self._frame,
            metric=metric,
            k=k,
            distance_column=distance_column,
            tie_breaker=tie_breaker,
        )
        plan = SemanticSearchPlan(
            self._frame._plan,
            (),
            vector_column.id,
            query_key,
            model,
            settings.batch_size,
            settings.separator,
            settings.null_policy,
            vector_metric,
            k,
            distance,
            tie_column.id if tie_column is not None else None,
            metadata,
        )
        return DataFrame(self._frame._session, plan)

    def search(
        self,
        query: object,
        *,
        column: str,
        metric: VectorMetricName = "cosine",
        k: int = 10,
        mode: VectorMode = "exact",
        distance_column: str = "_distance",
        tie_breaker: str | None = None,
    ) -> DataFrame:
        """Return the nearest rows as an ordinary lazy DuckPD DataFrame."""
        from duckpd.frame import DataFrame

        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        if not distance_column:
            raise ValueError("distance_column must be a non-empty string label")
        if distance_column in self._frame.columns:
            raise ValueError(f"distance column {distance_column!r} already exists")

        vector_column = find_column(self._frame._plan.metadata, column)
        element_type, dimension = _vector_type(vector_column.duckdb_type)
        vector_metric = _metric(metric)
        vector_query = _query_vector(
            query,
            element_type=element_type,
            expected_dimension=dimension,
        )
        _validate_query_metric(vector_query, vector_metric)
        try:
            execution_mode = VectorExecutionMode(mode)
        except ValueError:
            raise ValueError("mode must be 'exact' or 'approximate'") from None
        if execution_mode is VectorExecutionMode.APPROXIMATE and tie_breaker is not None:
            raise UnsupportedOperationError(
                "Approximate vector search does not support tie_breaker because "
                "DuckDB HNSW plan eligibility requires distance-only ordering"
            )

        tie_column = None
        if tie_breaker is not None:
            tie_column = find_column(self._frame._plan.metadata, tie_breaker)
            tie_type = tie_column.duckdb_type.upper()
            if "[" in tie_type or not tie_type.startswith(_ORDERABLE_PREFIXES):
                raise UnsupportedOperationError(
                    f"tie_breaker column {tie_breaker!r} is not orderable: {tie_column.duckdb_type}"
                )

        index_name = None
        if execution_mode is VectorExecutionMode.APPROXIMATE:
            index_name = self._frame._session._resolve_vector_index(
                self._frame._plan,
                vector_column,
                vector_metric,
            )

        distance = Column(
            ColumnId.create(),
            distance_column,
            element_type,
            nullable=Nullability.NON_NULL,
        )
        metadata = after_vector_search(
            self._frame._plan.metadata,
            distance,
            tie_breaker=tie_column,
        )
        plan = VectorSearchPlan(
            self._frame._plan,
            vector_column.id,
            vector_query,
            vector_metric,
            k,
            execution_mode,
            distance,
            tie_column.id if tie_column is not None else None,
            metadata,
            index_name,
        )
        return DataFrame(self._frame._session, plan)
