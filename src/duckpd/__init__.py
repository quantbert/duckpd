"""Lazy pandas-shaped DataFrames powered by DuckDB."""

from importlib.metadata import version

from duckpd._executor import CommitReport, MaterializationReport, ProfileResult
from duckpd.errors import ConcurrentModificationError, MergeError
from duckpd.featurestore import FeatureStore, SyncReport
from duckpd.frame import DataFrame
from duckpd.groupby import DataFrameGroupBy, SeriesGroupBy
from duckpd.io import concat, from_arrow, from_pandas, read_csv, read_parquet
from duckpd.series import Series
from duckpd.session import (
    ArrowUDFSpec,
    AttachedDatabase,
    ObjectStoreSecret,
    Session,
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
    "FeatureStore",
    "MaterializationReport",
    "MergeError",
    "ObjectStoreSecret",
    "ProfileResult",
    "Series",
    "SeriesGroupBy",
    "Session",
    "SyncReport",
    "__version__",
    "concat",
    "connect",
    "from_arrow",
    "from_pandas",
    "read_csv",
    "read_parquet",
]
