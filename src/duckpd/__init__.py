"""Lazy pandas-shaped DataFrames powered by DuckDB."""

from importlib.metadata import version

from duckpd._executor import CommitReport, MaterializationReport, ProfileResult
from duckpd._merging import merge_asof
from duckpd.embeddings import (
    EmbeddedQuery,
    EmbeddingModelSpec,
    FastEmbedProvider,
    PreparedModelInfo,
    TextEmbeddingProvider,
    TransformersEmbeddingProvider,
    embedding_model,
)
from duckpd.errors import ConcurrentModificationError, MergeError
from duckpd.featurestore import FeatureStore, SyncReport
from duckpd.frame import DataFrame
from duckpd.groupby import DataFrameGroupBy, SeriesGroupBy
from duckpd.io import concat, from_arrow, from_pandas, read_csv, read_parquet
from duckpd.series import Series
from duckpd.series_embeddings import (
    EmbeddedSeriesQuery,
    SeriesColumnSpec,
    SeriesEmbeddingModelSpec,
    SeriesRepresentationSpec,
    SeriesWindowSpec,
    series_embedding_model,
    series_representation,
)
from duckpd.session import (
    ArrowUDFSpec,
    AttachedDatabase,
    ObjectStoreSecret,
    Session,
    VectorIndexInfo,
    connect,
)

__version__ = version("duckpd")

__all__ = [
    "ArrowUDFSpec",
    "AttachedDatabase",
    "CommitReport",
    "ConcurrentModificationError",
    "DataFrame",
    "DataFrameGroupBy",
    "EmbeddedQuery",
    "EmbeddedSeriesQuery",
    "EmbeddingModelSpec",
    "FastEmbedProvider",
    "FeatureStore",
    "MaterializationReport",
    "MergeError",
    "ObjectStoreSecret",
    "PreparedModelInfo",
    "ProfileResult",
    "Series",
    "SeriesColumnSpec",
    "SeriesEmbeddingModelSpec",
    "SeriesGroupBy",
    "SeriesRepresentationSpec",
    "SeriesWindowSpec",
    "Session",
    "SyncReport",
    "TextEmbeddingProvider",
    "TransformersEmbeddingProvider",
    "VectorIndexInfo",
    "__version__",
    "concat",
    "connect",
    "embedding_model",
    "from_arrow",
    "from_pandas",
    "merge_asof",
    "read_csv",
    "read_parquet",
    "series_embedding_model",
    "series_representation",
]
