from __future__ import annotations

from pathlib import Path

import duckpd
from demo.market_data_demo import benchmark_file, human_size, parse_args


def test_human_size_formatting() -> None:
    assert human_size(500) == "500.00 B"
    assert human_size(50_000) == "50.00 KB"
    assert human_size(5_000_000) == "5.00 MB"
    assert human_size(5_000_000_000) == "5.00 GB"


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.preset == "100mb"
    assert args.ticker == "NVDA"
    assert args.threads == 4
    assert not args.skip_pandas


def test_benchmark_smoke_execution(tmp_path: Path) -> None:
    parquet_path = tmp_path / "smoke.parquet"
    with duckpd.connect() as session:
        # Generate 1,000 rows
        session.sql(
            """
            SELECT 
                TIMESTAMP '2020-01-01' + interval (i) second as datetime,
                CASE (i % 2) WHEN 0 THEN 'NVDA' ELSE 'AAPL' END as ticker,
                100.0::DOUBLE as open,
                105.0::DOUBLE as high,
                95.0::DOUBLE as low,
                102.0::DOUBLE as close
            FROM range(1000) t(i)
            """
        ).write_parquet(parquet_path)

    # Run benchmark_file on the smoke dataset
    benchmark_file(parquet_path, selected_ticker="NVDA", threads=2, skip_pandas=False)
