"""Benchmark execution time and memory usage between DuckPD and pandas.

Runs analytical market data workflows on synthetic OHLC datasets.
"""

from __future__ import annotations

import argparse
import multiprocessing
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd_orig
from pandas.testing import assert_frame_equal

import duckpd as pd


def human_size(size: int) -> str:
    """Format a byte count for console output."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1000.0
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class BenchmarkResult:
    """Metrics from a single pipeline execution."""

    engine: str
    elapsed_seconds: float
    peak_python_heap_bytes: int
    result_rows: int


def _run_duckpd_pipeline(
    parquet_path: str,
    selected_ticker: str,
    threads: int,
    queue: multiprocessing.Queue[tuple[pd_orig.DataFrame, float, int]],
) -> None:
    import tracemalloc

    tracemalloc.start()
    t0 = time.perf_counter()

    with pd.connect(threads=threads) as session:
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
        result_df = aggregated.collect()

    t1 = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    queue.put((result_df, t1 - t0, peak_bytes))


def _run_pandas_pipeline(
    parquet_path: str,
    selected_ticker: str,
    queue: multiprocessing.Queue[tuple[pd_orig.DataFrame, float, int]],
) -> None:
    import tracemalloc

    tracemalloc.start()
    t0 = time.perf_counter()

    raw_df = pd_orig.read_parquet(parquet_path)
    filtered = raw_df[raw_df["ticker"] == selected_ticker].copy()
    filtered["bar_return"] = (filtered["close"] - filtered["open"]) / filtered["open"]
    filtered["bar_range"] = filtered["high"] - filtered["low"]
    result_df = filtered.groupby("ticker", as_index=False).agg(
        avg_return=("bar_return", "mean"),
        avg_range=("bar_range", "mean"),
        max_high=("high", "max"),
        min_low=("low", "min"),
        total_bars=("close", "count"),
    )

    t1 = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    queue.put((result_df, t1 - t0, peak_bytes))


def run_in_subprocess(
    target: object, *args: object
) -> tuple[pd_orig.DataFrame, float, int]:
    """Run pipeline in an isolated process to measure peak memory allocations."""
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue[tuple[pd_orig.DataFrame, float, int]] = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, queue))  # type: ignore[arg-type]
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"Subprocess failed with exit code {proc.exitcode}")
    res = queue.get()
    df, elapsed, peak_ram = res
    return df, elapsed, peak_ram


def benchmark_file(
    parquet_path: Path,
    selected_ticker: str = "NVDA",
    threads: int = 4,
    skip_pandas: bool = False,
    repetitions: int = 1,
) -> tuple[BenchmarkResult, BenchmarkResult | None]:
    """Benchmark DuckPD vs pandas on a specific dataset file."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    file_size_bytes = parquet_path.stat().st_size
    print("=" * 78)
    print(f"Dataset: {parquet_path.name} ({human_size(file_size_bytes)})")
    print(f"Target filter: ticker == {selected_ticker!r}")
    print("=" * 78)

    duck_runs: list[BenchmarkResult] = []
    pandas_runs: list[BenchmarkResult] = []
    print(f"Running {repetitions} isolated repetition(s), alternating engine order...")
    for repetition in range(repetitions):
        run_pandas_first = repetition % 2 == 1 and not skip_pandas
        engines = ("pandas", "duckpd") if run_pandas_first else ("duckpd", "pandas")
        outputs: dict[str, pd_orig.DataFrame] = {}
        for engine in engines:
            if engine == "pandas" and skip_pandas:
                continue
            if engine == "duckpd":
                result_df, elapsed, peak_heap = run_in_subprocess(
                    _run_duckpd_pipeline,
                    str(parquet_path),
                    selected_ticker,
                    threads,
                )
                duck_runs.append(
                    BenchmarkResult("DuckPD", elapsed, peak_heap, len(result_df))
                )
            else:
                result_df, elapsed, peak_heap = run_in_subprocess(
                    _run_pandas_pipeline, str(parquet_path), selected_ticker
                )
                pandas_runs.append(
                    BenchmarkResult("pandas", elapsed, peak_heap, len(result_df))
                )
            outputs[engine] = result_df
        if not skip_pandas:
            assert_frame_equal(outputs["duckpd"], outputs["pandas"])

    duck_summary = _summarize_runs(duck_runs)
    print(_format_summary(duck_summary, duck_runs))
    if skip_pandas:
        print("Skipping pandas run (requested).")
        return duck_summary, None

    pandas_summary = _summarize_runs(pandas_runs)
    print(_format_summary(pandas_summary, pandas_runs))
    print("[OK] Semantic verification passed for every repetition.")
    speedup = pandas_summary.elapsed_seconds / max(duck_summary.elapsed_seconds, 1e-6)
    heap_ratio = pandas_summary.peak_python_heap_bytes / max(
        duck_summary.peak_python_heap_bytes, 1
    )
    print(f"  * Median speedup:           {speedup:.2f}x")
    print(f"  * Traced Python heap ratio: {heap_ratio:.2f}x")
    return duck_summary, pandas_summary


def _summarize_runs(runs: list[BenchmarkResult]) -> BenchmarkResult:
    """Return a median summary for repeated benchmark observations."""
    if not runs:
        raise ValueError("At least one benchmark result is required")
    return BenchmarkResult(
        runs[0].engine,
        statistics.median(run.elapsed_seconds for run in runs),
        round(statistics.median(run.peak_python_heap_bytes for run in runs)),
        runs[0].result_rows,
    )


def _format_summary(summary: BenchmarkResult, runs: list[BenchmarkResult]) -> str:
    """Format median and observed timing range for console output."""
    elapsed = [run.elapsed_seconds for run in runs]
    return (
        f"{summary.engine}: median {summary.elapsed_seconds:.4f} s "
        f"(range {min(elapsed):.4f}-{max(elapsed):.4f} s), "
        f"median peak traced Python heap "
        f"{human_size(summary.peak_python_heap_bytes)}, "
        f"output rows {summary.result_rows}"
    )


def parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DuckPD vs pandas on OHLC market datasets."
    )
    parser.add_argument(
        "preset",
        nargs="?",
        default="100mb",
        choices=["smoke", "100mb", "1gb", "5gb", "all"],
        help="Dataset preset to benchmark (default: 100mb)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("demo/data"),
        help="Directory containing the market data files (default: demo/data)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="NVDA",
        help="Ticker to filter on (default: NVDA)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for DuckPD engine (default: 4)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of isolated repetitions per engine (default: 3)",
    )
    parser.add_argument(
        "--skip-pandas",
        action="store_true",
        help="Skip pandas execution (useful for very large datasets)",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args(sys.argv[1:])
    presets: list[str] = (
        ["smoke", "100mb", "1gb", "5gb"] if args.preset == "all" else [args.preset]
    )

    found_any = False
    for preset in presets:
        path = args.data_dir / f"market-data-{preset}.parquet"
        if not path.exists():
            print(
                f"\n[!] Dataset file not found: {path}\n"
                f"    Generate: uv run python demo/generate_market_data.py {preset}"
            )
            continue

        found_any = True
        benchmark_file(
            path,
            selected_ticker=args.ticker,
            threads=args.threads,
            skip_pandas=args.skip_pandas,
            repetitions=args.repetitions,
        )

    if not found_any:
        print(
            "\nNo dataset files found to benchmark. "
            "Please generate data files first using:\n"
            "  uv run python demo/generate_market_data.py smoke 100mb 1gb\n"
        )


if __name__ == "__main__":
    main()
