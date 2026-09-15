"""Lazy pandas-shaped DataFrames powered by DuckDB."""

from importlib.metadata import version

from duckpd._executor import CommitReport, MaterializationReport, ProfileResult
from duckpd._merging import merge_asof
from duckpd._ts2vec import TS2VecProvider, ts2vec_series_embedding_model
from duckpd._tspulse import TSPulseProvider, tspulse_series_embedding_model
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
    PreparedSeriesModelInfo,
    SeriesChannelRole,
    SeriesColumnSpec,
    SeriesEmbeddingModelSpec,
    SeriesEmbeddingProvider,
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
    "PreparedSeriesModelInfo",
    "ProfileResult",
    "Series",
    "SeriesChannelRole",
    "SeriesColumnSpec",
    "SeriesEmbeddingModelSpec",
    "SeriesEmbeddingProvider",
    "SeriesGroupBy",
    "SeriesRepresentationSpec",
    "SeriesWindowSpec",
    "Session",
    "SyncReport",
    "TS2VecProvider",
    "TSPulseProvider",
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
    "ts2vec_series_embedding_model",
    "tspulse_series_embedding_model",
]
