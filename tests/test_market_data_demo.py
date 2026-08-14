from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import duckpd
from demo.generate_market_data import (
    estimate_row_count,
    human_size,
    market_query,
    parse_args,
    write_market_data,
)


def test_market_query_produces_valid_ohlc_rows() -> None:
    with duckpd.connect() as session:
        result = session.sql(market_query(80)).collect()

    assert list(result.columns) == [
        "datetime",
        "ticker",
        "open",
        "high",
        "low",
        "close",
    ]
    assert result["ticker"].nunique() == 8
    assert result["datetime"].nunique() == 10
    assert (result["high"] >= result[["open", "close"]].max(axis=1)).all()
    assert (result["low"] <= result[["open", "close"]].min(axis=1)).all()


def test_market_query_rejects_non_positive_rows() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        market_query(0)


def test_row_estimate_scales_calibration() -> None:
    assert estimate_row_count(100, 1_000, 10_000) == 1_000
    with pytest.raises(ValueError, match="must be positive"):
        estimate_row_count(100, 0, 10_000)


def test_human_size_uses_decimal_units() -> None:
    assert human_size(5_000_000) == "5.00 MB"
    assert human_size(1_000_000_000) == "1.00 GB"


def test_cli_defaults_to_smoke_and_accepts_multiple_sizes() -> None:
    assert parse_args([]).sizes == ("smoke",)
    assert parse_args(["100mb", "1gb"]).sizes == ("100mb", "1gb")


def test_write_market_data_creates_readable_parquet(tmp_path: Path) -> None:
    path = tmp_path / "market.parquet"
    with duckpd.connect() as session:
        write_market_data(session, 1_000, path)

    result = pd.read_parquet(path)
    assert len(result) == 1_000
    assert result["ticker"].nunique() == 8