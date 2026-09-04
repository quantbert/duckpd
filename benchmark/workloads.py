"""Analytical workloads for benchmarking DuckPD against regular pandas."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import pandas as pd_orig
from pandas.testing import assert_frame_equal

import duckpd


@dataclass(frozen=True)
class Workload:
    """Benchmark workload specification."""

    name: str
    description: str
    run_duckpd: Callable[[str, int], pd_orig.DataFrame]
    run_pandas: Callable[[str], pd_orig.DataFrame]
    verify: Callable[[pd_orig.DataFrame, pd_orig.DataFrame], None]


# ---------------------------------------------------------------------------
# Workload 1: Filter + Derived Columns + GroupBy Aggregation
# ---------------------------------------------------------------------------


def _duckpd_filter_groupby_agg(
    parquet_path: str, threads: int, selected_ticker: str = "NVDA"
) -> pd_orig.DataFrame:
    with duckpd.connect(threads=threads) as session:
        lazy_df = session.read_parquet(parquet_path)
        aggregated = (
            lazy_df[lazy_df["ticker"] == selected_ticker]
            .assign(
                bar_return=lambda f: (f["close"] - f["open"]) / f["open"],
                bar_range=lambda f: f["high"] - f["low"],
            )
            .groupby("ticker", as_index=False)
            .agg(
                avg_return=("bar_return", "mean"),
                avg_range=("bar_range", "mean"),
                max_high=("high", "max"),
                min_low=("low", "min"),
                total_bars=("close", "count"),
            )
        )
        return aggregated.collect()


def _pandas_filter_groupby_agg(
    parquet_path: str, selected_ticker: str = "NVDA"
) -> pd_orig.DataFrame:
    raw_df = pd_orig.read_parquet(parquet_path)
    filtered = raw_df[raw_df["ticker"] == selected_ticker].copy()
    filtered["bar_return"] = (filtered["close"] - filtered["open"]) / filtered["open"]
    filtered["bar_range"] = filtered["high"] - filtered["low"]
    return filtered.groupby("ticker", as_index=False).agg(
        avg_return=("bar_return", "mean"),
        avg_range=("bar_range", "mean"),
        max_high=("high", "max"),
        min_low=("low", "min"),
        total_bars=("close", "count"),
    )


def _verify_filter_groupby_agg(
    duck_df: pd_orig.DataFrame, pandas_df: pd_orig.DataFrame
) -> None:
    d_sorted = duck_df.sort_values("ticker").reset_index(drop=True)
    p_sorted = pandas_df.sort_values("ticker").reset_index(drop=True)
    assert_frame_equal(d_sorted, p_sorted, check_dtype=False, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Workload 2: Full Scan Multi-Column GroupBy Aggregation
# ---------------------------------------------------------------------------


def _duckpd_full_scan_agg(parquet_path: str, threads: int) -> pd_orig.DataFrame:
    with duckpd.connect(threads=threads) as session:
        lazy_df = session.read_parquet(parquet_path)
        aggregated = lazy_df.groupby("ticker", as_index=False).agg(
            mean_open=("open", "mean"),
            max_high=("high", "max"),
            min_low=("low", "min"),
            mean_close=("close", "mean"),
            bar_count=("close", "count"),
        )
        return aggregated.collect()


def _pandas_full_scan_agg(parquet_path: str) -> pd_orig.DataFrame:
    raw_df = pd_orig.read_parquet(parquet_path)
    return raw_df.groupby("ticker", as_index=False).agg(
        mean_open=("open", "mean"),
        max_high=("high", "max"),
        min_low=("low", "min"),
        mean_close=("close", "mean"),
        bar_count=("close", "count"),
    )


def _verify_full_scan_agg(
    duck_df: pd_orig.DataFrame, pandas_df: pd_orig.DataFrame
) -> None:
    d_sorted = duck_df.sort_values("ticker").reset_index(drop=True)
    p_sorted = pandas_df.sort_values("ticker").reset_index(drop=True)
    assert_frame_equal(d_sorted, p_sorted, check_dtype=False, atol=1e-4, rtol=1e-4)


WORKLOADS: Final[dict[str, Workload]] = {
    "filter_groupby_agg": Workload(
        name="filter_groupby_agg",
        description="Predicate filter + derived columns + groupby aggregation",
        run_duckpd=_duckpd_filter_groupby_agg,
        run_pandas=_pandas_filter_groupby_agg,
        verify=_verify_filter_groupby_agg,
    ),
    "full_scan_agg": Workload(
        name="full_scan_agg",
        description="Full-scan multi-column groupby aggregation across all rows",
        run_duckpd=_duckpd_full_scan_agg,
        run_pandas=_pandas_full_scan_agg,
        verify=_verify_full_scan_agg,
    ),
}
