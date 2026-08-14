"""Lazy pandas-shaped DataFrames powered by DuckDB."""

from importlib.metadata import version

from duckpd.frame import DataFrame
from duckpd.groupby import DataFrameGroupBy
from duckpd.io import from_arrow, from_pandas, read_parquet
from duckpd.series import Series
from duckpd.session import Session, connect

__version__ = version("duckpd")

__all__ = [
    "DataFrame",
    "DataFrameGroupBy",
    "Series",
    "Session",
    "__version__",
    "connect",
    "from_arrow",
    "from_pandas",
    "read_parquet",
]
