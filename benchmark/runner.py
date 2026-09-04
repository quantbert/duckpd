"""CLI entrypoint and orchestration for DuckPD benchmark suite."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmark.generate import (
    CANONICAL_PRESETS,
    canonicalize_preset,
    ensure_dataset,
)
from benchmark.metrics import (
    BenchmarkComparison,
    human_bytes,
    human_seconds,
    run_benchmark,
)
from benchmark.report import generate_markdown_report
from benchmark.workloads import WORKLOADS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse benchmark suite CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run DuckPD vs pandas benchmarks across multiple file sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["5mb", "50mb", "500m"],
        help=(
            "Dataset sizes to benchmark. Choices: 5mb, 50mb, 500m, 5g, 50g, "
            "or 'all'. (Default: 5mb 50mb 500m)"
        ),
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["filter_groupby_agg"],
        choices=[*list(WORKLOADS.keys()), "all"],
        help="Workloads to execute.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmark/data"),
        help="Directory to store or load Parquet benchmark datasets.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmark/REPORT.md"),
        help="Path where the Markdown report will be saved.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of isolated execution repetitions per engine.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Thread count for DuckPD execution engine.",
    )
    parser.add_argument(
        "--skip-pandas",
        action="store_true",
        help="Skip pandas execution across all datasets.",
    )
    parser.add_argument(
        "--max-pandas-bytes",
        type=int,
        default=15_000_000_000,
        help="Maximum dataset size in bytes for running pandas (safety guard).",
    )
    parser.add_argument(
        "--force-generate",
        action="store_true",
        help="Regenerate datasets even if they already exist on disk.",
    )
    return parser.parse_args(argv)


def resolve_sizes(raw_sizes: list[str]) -> list[str]:
    """Resolve preset size arguments into canonical preset names."""
    if "all" in [s.lower() for s in raw_sizes]:
        return list(CANONICAL_PRESETS)

    resolved: list[str] = []
    for s in raw_sizes:
        canon = canonicalize_preset(s)
        if canon not in resolved:
            resolved.append(canon)
    return resolved


def resolve_workloads(raw_workloads: list[str]) -> list[str]:
    """Resolve requested workloads into workload dictionary keys."""
    if "all" in [w.lower() for w in raw_workloads]:
        return list(WORKLOADS.keys())
    return [w for w in raw_workloads if w in WORKLOADS]


def run_suite(args: argparse.Namespace) -> list[BenchmarkComparison]:
    """Execute complete benchmark suite and generate report."""
    sizes = resolve_sizes(args.sizes)
    workload_names = resolve_workloads(args.workloads)

    print("=" * 80)
    print(" DuckPD Benchmark Suite")
    print("=" * 80)
    print(f"Target sizes:     {', '.join(sizes)}")
    print(f"Workloads:        {', '.join(workload_names)}")
    print(f"DuckPD threads:   {args.threads}")
    print(f"Repetitions:      {args.repetitions}")
    print(f"Data directory:   {args.data_dir}")
    print(f"Report path:      {args.report}")
    print("=" * 80)

    comparisons: list[BenchmarkComparison] = []

    for size_preset in sizes:
        print(f"\n[1/2] Preparing dataset for preset '{size_preset}'...")
        dataset = ensure_dataset(size_preset, args.data_dir, force=args.force_generate)
        print(
            f"  Dataset ready: {dataset.path.name} "
            f"({human_bytes(dataset.file_size_bytes)}, {dataset.row_count:,} rows)"
        )

        for w_name in workload_names:
            workload = WORKLOADS[w_name]
            print(f"\n[2/2] Benchmarking workload: {w_name}")
            print(f"  Description: {workload.description}")

            comp = run_benchmark(
                dataset,
                workload,
                repetitions=args.repetitions,
                threads=args.threads,
                skip_pandas=args.skip_pandas,
                max_pandas_bytes=args.max_pandas_bytes,
            )
            comparisons.append(comp)

            # Print brief live summary to terminal
            duck_t = human_seconds(comp.duckpd.median_time)
            duck_rss = human_bytes(comp.duckpd.median_rss_bytes)
            if comp.pandas is not None and comp.pandas.success:
                pan_t = human_seconds(comp.pandas.median_time)
                pan_rss = human_bytes(comp.pandas.median_rss_bytes)
                spd = f"{comp.speedup:.2f}x" if comp.speedup else "N/A"
                rss_red = f"{comp.rss_reduction:.2f}x" if comp.rss_reduction else "N/A"
                print(f"  DuckPD: {duck_t} (Peak RSS: {duck_rss})")
                print(f"  pandas: {pan_t} (Peak RSS: {pan_rss})")
                print(f"  --> Speedup: {spd} | RSS Memory Savings: {rss_red}")
            elif comp.pandas is not None and not comp.pandas.success:
                print(f"  DuckPD: {duck_t} (Peak RSS: {duck_rss})")
                print(f"  pandas: FAILED / OOM ({comp.pandas.error_message})")
                print("  --> DuckPD successfully ran out-of-core where pandas failed!")
            else:
                print(f"  DuckPD: {duck_t} (Peak RSS: {duck_rss}) [pandas skipped]")

            print(f"  Verification: {comp.verification_notes}")

    # Generate Markdown Report
    print(f"\nGenerating Markdown benchmark report at {args.report}...")
    generate_markdown_report(
        comparisons,
        threads=args.threads,
        repetitions=args.repetitions,
        output_path=args.report,
    )
    print(f"[OK] Report written successfully: {args.report.resolve()}")

    return comparisons


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv or sys.argv[1:])
    run_suite(args)


if __name__ == "__main__":
    main()
