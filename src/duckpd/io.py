"""Module-level data source helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from duckpd.session import Session

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckpd.frame import DataFrame


def from_pandas(
    value: pd.DataFrame,
    *,
    session: Session | None = None,
    index: str | Sequence[str] | None = None,
    order_by: str | Sequence[str] | None = None,
) -> DataFrame:
    """Create a lazy DuckPD frame from a pandas DataFrame."""
    owner = session if session is not None else Session()
    return owner.from_pandas(value, index=index, order_by=order_by)


def from_arrow(
    value: pa.Table | pa.RecordBatch,
    *,
    session: Session | None = None,
    index: str | Sequence[str] | None = None,
    order_by: str | Sequence[str] | None = None,
) -> DataFrame:
    """Create a lazy DuckPD frame from an Arrow table or record batch."""
    owner = session if session is not None else Session()
    return owner.from_arrow(value, index=index, order_by=order_by)


def read_parquet(
    path: str | Path | Sequence[str | Path],
    *,
    session: Session | None = None,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    index: str | Sequence[str] | None = None,
    order_by: str | Sequence[str] | None = None,
) -> DataFrame:
    """Create a lazy DuckPD frame from one or more Parquet files."""
    owner = session if session is not None else Session()
    return owner.read_parquet(
        path,
        hive_partitioning=hive_partitioning,
        union_by_name=union_by_name,
        index=index,
        order_by=order_by,
    )
