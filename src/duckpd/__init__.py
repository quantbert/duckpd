"""Lazy pandas-shaped DataFrames powered by DuckDB."""

from importlib.metadata import version

from duckpd._executor import CommitReport, MaterializationReport, ProfileResult
from duckpd._merging import merge_asof
from duckpd.embeddings import (
    EmbeddedQuery,
    EmbeddingModelSpec,
    EmbeddingProvider,
    FastEmbedProvider,
    PreparedModelInfo,
    SeriesCadenceSpec,
    SeriesCadenceUnit,
    SeriesChannelRole,
    SeriesEmbeddingInputSpec,
    SeriesFrequencyInputSpec,
    SeriesStaticInputSpec,
    SeriesTemporalInputSpec,
    SeriesTimeFeature,
    TextEmbeddingProvider,
    TransformersEmbeddingProvider,
    embedding_model,
    series_cadence,
    series_embedding_input,
    series_frequency_input,
    series_static_input,
    series_temporal_input,
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
    SeriesEmbeddingProvider,
    SeriesQueryInput,
    SeriesRepresentationSpec,
    SeriesWindowSpec,
    series_query,
    series_representation,
)
from duckpd.series_providers import (
    MomentEmbeddingProvider,
    TransformersSeriesEmbeddingProvider,
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
    "EmbeddingProvider",
    "FastEmbedProvider",
    "FeatureStore",
    "MaterializationReport",
    "MergeError",
    "MomentEmbeddingProvider",
    "ObjectStoreSecret",
    "PreparedModelInfo",
    "ProfileResult",
    "Series",
    "SeriesCadenceSpec",
    "SeriesCadenceUnit",
    "SeriesChannelRole",
    "SeriesColumnSpec",
    "SeriesEmbeddingInputSpec",
    "SeriesEmbeddingProvider",
    "SeriesFrequencyInputSpec",
    "SeriesGroupBy",
    "SeriesQueryInput",
    "SeriesRepresentationSpec",
    "SeriesStaticInputSpec",
    "SeriesTemporalInputSpec",
    "SeriesTimeFeature",
    "SeriesWindowSpec",
    "Session",
    "SyncReport",
    "TextEmbeddingProvider",
    "TransformersEmbeddingProvider",
    "TransformersSeriesEmbeddingProvider",
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
    "series_cadence",
    "series_embedding_input",
    "series_frequency_input",
    "series_query",
    "series_representation",
    "series_static_input",
    "series_temporal_input",
]
