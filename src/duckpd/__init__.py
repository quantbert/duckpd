"""Lazy pandas-shaped DataFrames powered by DuckDB."""

from importlib.metadata import version

from duckpd.frame import DataFrame
from duckpd.groupby import DataFrameGroupBy, SeriesGroupBy
from duckpd.io import concat, from_arrow, from_pandas, read_parquet
from duckpd.series import Series
from duckpd.session import Session, connect

__version__ = version("duckpd")

__all__ = [
    "DataFrame",
    "DataFrameGroupBy",
    "Series",
    "SeriesGroupBy",
    "Session",
    "__version__",
    "concat",
    "connect",
    "from_arrow",
    "from_pandas",
    "read_parquet",
]
