"""Benchmark execution time and memory usage between DuckPD and pandas.

Runs analytical market data workflows on synthetic OHLC datasets.
"""

from __future__ import annotations

import argparse
import multiprocessing
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
    peak_python_ram_bytes: int
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
    res = queue.get()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"Subprocess failed with exit code {proc.exitcode}")
    df, elapsed, peak_ram = res
    return df, elapsed, peak_ram


def benchmark_file(
    parquet_path: Path,
    selected_ticker: str = "NVDA",
    threads: int = 4,
    skip_pandas: bool = False,
) -> None:
    """Benchmark DuckPD vs pandas on a specific dataset file."""
    file_size_bytes = parquet_path.stat().st_size
    print("=" * 78)
    print(f"Dataset: {parquet_path.name} ({human_size(file_size_bytes)})")
    print(f"Target filter: ticker == {selected_ticker!r}")
    print("=" * 78)

    # 1. DuckPD Run
    print("Running DuckPD (lazy projection + predicate pushdown + SQL backend)...")
    duck_df, duck_time, duck_ram = run_in_subprocess(
        _run_duckpd_pipeline, str(parquet_path), selected_ticker, threads
    )

    print(f"  -> Execution time: {duck_time:.4f} s")
    print(f"  -> Peak Python RAM: {human_size(duck_ram)}")
    print(f"  -> Output rows:    {len(duck_df)}")

    # 2. Pandas Run
    if skip_pandas:
        print("\nSkipping pandas run (requested).")
        return

    print("\nRunning standard pandas (eager full-table read + in-memory execution)...")
    try:
        pandas_df, pandas_time, pandas_ram = run_in_subprocess(
            _run_pandas_pipeline, str(parquet_path), selected_ticker
        )
        print(f"  -> Execution time: {pandas_time:.4f} s")
        print(f"  -> Peak Python RAM: {human_size(pandas_ram)}")
        print(f"  -> Output rows:    {len(pandas_df)}")

        # Verify exact numerical / semantic equivalence
        assert_frame_equal(duck_df, pandas_df)
        print(
            "\n[OK] Semantic verification: DuckPD and pandas results match identically!"
        )

        speedup = pandas_time / max(duck_time, 1e-6)
        ram_saving = pandas_ram / max(duck_ram, 1)
        print(f"  * Speedup:    {speedup:.2f}x faster")
        print(f"  * RAM saving: {ram_saving:.2f}x less memory")

    except Exception as exc:
        print(f"  -> Pandas run failed or OOM: {exc}")


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
        )

    if not found_any:
        print(
            "\nNo dataset files found to benchmark. "
            "Please generate data files first using:\n"
            "  uv run python demo/generate_market_data.py smoke 100mb 1gb\n"
        )


if __name__ == "__main__":
    main()
