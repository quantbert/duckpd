"""Tests for the DuckPD benchmarking suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.generate import (
    CALIBRATION_ROWS,
    CANONICAL_PRESETS,
    PRESET_SIZES,
    calibrate,
    canonicalize_preset,
    ensure_dataset,
    estimate_row_count,
    generate_dataset,
    market_query,
)
from benchmark.metrics import (
    RunMetric,
    aggregate_runs,
    get_peak_rss_bytes,
    human_bytes,
    human_seconds,
    human_throughput,
    run_benchmark,
)
from benchmark.report import generate_markdown_report
from benchmark.runner import parse_args, resolve_sizes, resolve_workloads
from benchmark.tracks import results_json, run_tracks
from benchmark.workloads import WORKLOADS


def test_preset_canonicalization() -> None:
    assert canonicalize_preset("5mb") == "5mb"
    assert canonicalize_preset("50mb") == "50mb"
    assert canonicalize_preset("500m") == "500m"
    assert canonicalize_preset("500mb") == "500m"
    assert canonicalize_preset("5g") == "5g"
    assert canonicalize_preset("5gb") == "5g"
    assert canonicalize_preset("50g") == "50g"
    assert canonicalize_preset("50gb") == "50g"

    with pytest.raises(ValueError, match="Unknown preset size"):
        canonicalize_preset("10tb")


def test_preset_target_bytes() -> None:
    assert PRESET_SIZES["5mb"] == 5_000_000
    assert PRESET_SIZES["50mb"] == 50_000_000
    assert PRESET_SIZES["500m"] == 500_000_000
    assert PRESET_SIZES["5g"] == 5_000_000_000
    assert PRESET_SIZES["50g"] == 50_000_000_000
    assert CANONICAL_PRESETS == ("5mb", "50mb", "500m", "5g", "50g")


def test_estimate_row_count_validation() -> None:
    assert estimate_row_count(100, 1_000, 10_000) == 1_000
    with pytest.raises(ValueError, match="calibration_rows must be positive"):
        estimate_row_count(0, 1_000, 10_000)
    with pytest.raises(ValueError, match="calibration_bytes must be positive"):
        estimate_row_count(100, 0, 10_000)
    with pytest.raises(ValueError, match="target_bytes must be positive"):
        estimate_row_count(100, 1_000, 0)


def test_market_query_validation() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        market_query(0)
    query = market_query(100)
    assert "range(100)" in query
    assert "AAPL" in query


def test_human_formatters() -> None:
    assert human_bytes(500) == "500.00 B"
    assert human_bytes(5_000_000) == "5.00 MB"
    assert human_bytes(5_000_000_000) == "5.00 GB"

    assert human_seconds(0.0005) == "500.0 µs"
    assert human_seconds(0.05) == "50.00 ms"
    assert human_seconds(1.2345) == "1.2345 s"

    assert human_throughput(500, "rows/s") == "500.0 rows/s"
    assert human_throughput(50_000, "rows/s") == "50.00 K rows/s"
    assert human_throughput(5_000_000, "rows/s") == "5.00 M rows/s"
    assert human_throughput(5_000_000_000, "rows/s") == "5.00 B rows/s"


def test_get_peak_rss_bytes() -> None:
    rss = get_peak_rss_bytes()
    assert isinstance(rss, int)
    assert rss > 0


def test_aggregate_runs_empty_raises() -> None:
    with pytest.raises(ValueError, match="Cannot aggregate empty runs"):
        aggregate_runs([], 100, 10)


def test_aggregate_runs_success_and_failure() -> None:
    successful_runs = [
        RunMetric("DuckPD", 0.1, 100_000_000, 500_000, 10, True),
        RunMetric("DuckPD", 0.2, 110_000_000, 500_000, 10, True),
        RunMetric("DuckPD", 0.3, 120_000_000, 500_000, 10, True),
    ]
    summary = aggregate_runs(successful_runs, 10_000_000, 100_000)
    assert summary.engine == "DuckPD"
    assert summary.success is True
    assert summary.median_time == 0.2
    assert summary.min_time == 0.1
    assert summary.max_time == 0.3
    assert summary.median_rss_bytes == 110_000_000
    assert summary.throughput_mb_s > 0

    failed_runs = [
        RunMetric("pandas", 0.0, 0, 0, 0, False, "Out Of Memory"),
    ]
    failed_summary = aggregate_runs(failed_runs, 10_000_000, 100_000)
    assert failed_summary.success is False
    assert failed_summary.error_message == "Out Of Memory"


def test_calibration_and_generate_dataset(tmp_path: Path) -> None:
    # Calibrate in tmp dir
    calib_rows, calib_bytes = calibrate(tmp_path)
    assert calib_rows == CALIBRATION_ROWS
    assert calib_bytes > 0

    # Generate a small 5mb dataset in tmp dir
    dataset = generate_dataset("5mb", tmp_path, calibration=(calib_rows, calib_bytes))
    assert dataset.path.exists()
    assert dataset.file_size_bytes > 0
    assert dataset.row_count > 0

    # Ensure dataset reuses cached file
    cached = ensure_dataset("5mb", tmp_path)
    assert cached.path == dataset.path
    assert cached.file_size_bytes == dataset.file_size_bytes


def test_workload_execution_and_verification(tmp_path: Path) -> None:
    dataset = generate_dataset("5mb", tmp_path)

    # Test filter_groupby_agg workload
    w1 = WORKLOADS["filter_groupby_agg"]
    duck_df1 = w1.run_duckpd(str(dataset.path), 2)
    pan_df1 = w1.run_pandas(str(dataset.path))
    assert len(duck_df1) == 1
    assert len(pan_df1) == 1
    w1.verify(duck_df1, pan_df1)

    # Test full_scan_agg workload
    w2 = WORKLOADS["full_scan_agg"]
    duck_df2 = w2.run_duckpd(str(dataset.path), 2)
    pan_df2 = w2.run_pandas(str(dataset.path))
    assert len(duck_df2) == 8
    assert len(pan_df2) == 8
    w2.verify(duck_df2, pan_df2)


def test_run_benchmark_comparison(tmp_path: Path) -> None:
    dataset = generate_dataset("5mb", tmp_path)
    comp = run_benchmark(
        dataset,
        WORKLOADS["filter_groupby_agg"],
        repetitions=2,
        threads=2,
    )
    assert comp.duckpd.success is True
    assert comp.pandas is not None
    assert comp.pandas.success is True
    assert comp.verified is True
    assert comp.speedup is not None and comp.speedup > 0


def test_run_benchmark_skipping_pandas(tmp_path: Path) -> None:
    dataset = generate_dataset("5mb", tmp_path)
    comp = run_benchmark(
        dataset,
        WORKLOADS["filter_groupby_agg"],
        repetitions=1,
        threads=2,
        skip_pandas=True,
    )
    assert comp.duckpd.success is True
    assert comp.pandas is None
    assert comp.verification_notes == "pandas skipped by user flag"


def test_max_pandas_bytes_safety_guard(tmp_path: Path) -> None:
    dataset = generate_dataset("5mb", tmp_path)
    # Set max_pandas_bytes lower than the 5mb dataset size
    comp = run_benchmark(
        dataset,
        WORKLOADS["filter_groupby_agg"],
        repetitions=1,
        threads=2,
        max_pandas_bytes=100_000,  # 100 KB
    )
    assert comp.duckpd.success is True
    assert comp.pandas is None
    assert "safety limit" in comp.verification_notes


def test_report_generation(tmp_path: Path) -> None:
    dataset = generate_dataset("5mb", tmp_path)
    comp = run_benchmark(
        dataset,
        WORKLOADS["filter_groupby_agg"],
        repetitions=1,
        threads=2,
    )
    report_file = tmp_path / "REPORT.md"
    content = generate_markdown_report(
        [comp], threads=2, repetitions=1, output_path=report_file
    )
    assert "# DuckPD vs pandas Benchmark Report" in content
    assert "5mb" in content
    assert "filter_groupby_agg" in content
    assert report_file.exists()


def test_cli_parsing_and_resolvers() -> None:
    args = parse_args(["--sizes", "5mb", "50mb", "--repetitions", "2"])
    assert args.sizes == ["5mb", "50mb"]
    assert args.repetitions == 2

    assert resolve_sizes(["all"]) == list(CANONICAL_PRESETS)
    assert resolve_sizes(["500mb", "5gb"]) == ["500m", "5g"]

    assert resolve_workloads(["all"]) == list(WORKLOADS.keys())
    assert resolve_workloads(["full_scan_agg"]) == ["full_scan_agg"]


def test_validated_release_tracks(tmp_path: Path) -> None:
    results = run_tracks(1_000, tmp_path)

    assert [result.track for result in results] == [
        "tpch_q1",
        "db_groupby_join",
        "synthetic_ohlc",
    ]
    for result in results:
        engines = {engine.engine: engine for engine in result.engines}
        for name in ("duckdb_sql", "duckpd", "pandas"):
            assert engines[name].status == "success"
            assert engines[name].correct is True
            execution_seconds = engines[name].execution_seconds
            peak_rss_bytes = engines[name].peak_rss_bytes
            assert execution_seconds is not None
            assert execution_seconds >= 0
            assert peak_rss_bytes is not None
            assert peak_rss_bytes > 0
        assert engines["duckpd"].planning_seconds is not None
        assert engines["duckpd"].spill_bytes is not None
        assert engines["polars"].status in {"unavailable", "unsupported"}
        assert engines["fireducks"].status in {"unavailable", "unsupported"}

    encoded = results_json(results)
    assert '"track": "tpch_q1"' in encoded
    assert '"correct": true' in encoded
