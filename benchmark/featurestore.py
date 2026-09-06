"""Benchmark cold remote feature fetches against warm local-cache execution."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import duckpd


@dataclass(frozen=True)
class FeatureStoreBenchmarkResult:
    """Measured cold-fetch and warm-cache feature query performance."""

    source: str
    features: tuple[str, ...]
    start: str
    end: str
    alignment: str
    repetitions: int
    rows: int
    cold_seconds: float
    warm_seconds: tuple[float, ...]
    warm_median_seconds: float
    cold_to_warm_speedup: float
    cache_bytes: int

    def to_json(self) -> str:
        """Serialize the benchmark result for CI and release evidence."""
        return json.dumps(asdict(self), indent=2)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_featurestore_benchmark(
    *,
    source: str,
    cache: str | Path,
    features: Sequence[str],
    start: str,
    end: str,
    alignment: Literal["exact", "point_in_time"] = "exact",
    spine: str | None = None,
    repetitions: int = 3,
    token: str | None = None,
    catalog_path: str | Path | None = None,
) -> FeatureStoreBenchmarkResult:
    """Measure one cold remote execution and repeated warm-cache executions.

    The cache directory must be empty so the first measurement includes remote
    partition transfer and projection. The function never deletes caller data.
    """
    if not source.startswith(("hf://", "http://", "https://")):
        raise ValueError("featurestore benchmark source must be a remote URI")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not features:
        raise ValueError("at least one feature is required")

    cache_path = Path(cache).expanduser().resolve()
    if cache_path.exists() and any(cache_path.iterdir()):
        raise ValueError("featurestore benchmark cache directory must be empty")
    cache_path.mkdir(parents=True, exist_ok=True)

    cold_started = perf_counter()
    store = duckpd.FeatureStore(
        source=source,
        cache=cache_path,
        token=token,
        catalog_path=catalog_path,
    )
    frame = store.features(
        features=features,
        start=start,
        end=end,
        alignment=alignment,
        spine=spine,
    )
    cold_result = frame.collect()
    cold_seconds = perf_counter() - cold_started

    warm_seconds: list[float] = []
    for _ in range(repetitions):
        started = perf_counter()
        warm_result = frame.collect()
        warm_seconds.append(perf_counter() - started)
        if len(warm_result) != len(cold_result):
            raise RuntimeError(
                "cold and warm featurestore executions returned different row counts"
            )

    warm_median = statistics.median(warm_seconds)
    return FeatureStoreBenchmarkResult(
        source=source,
        features=tuple(features),
        start=start,
        end=end,
        alignment=alignment,
        repetitions=repetitions,
        rows=len(cold_result),
        cold_seconds=cold_seconds,
        warm_seconds=tuple(warm_seconds),
        warm_median_seconds=warm_median,
        cold_to_warm_speedup=cold_seconds / warm_median,
        cache_bytes=_directory_bytes(cache_path),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the standalone featurestore benchmark arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark cold remote feature fetches against warm local-cache execution."
    )
    parser.add_argument("--source", required=True, help="Remote hf://, http://, or https:// store")
    parser.add_argument("--cache", required=True, type=Path, help="Empty local cache directory")
    parser.add_argument("--features", required=True, nargs="+", help="Catalog feature references")
    parser.add_argument("--start", required=True, help="Inclusive ISO-8601 start timestamp")
    parser.add_argument("--end", required=True, help="Exclusive ISO-8601 end timestamp")
    parser.add_argument(
        "--alignment",
        choices=("exact", "point_in_time"),
        default="exact",
    )
    parser.add_argument("--spine", help="Required spine dataset for point-in-time alignment")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the featurestore benchmark CLI."""
    args = parse_args(argv or sys.argv[1:])
    token = os.environ.get("FEATURESTORE_TOKEN") or os.environ.get("HF_TOKEN")
    result = run_featurestore_benchmark(
        source=args.source,
        cache=args.cache,
        features=args.features,
        start=args.start,
        end=args.end,
        alignment=args.alignment,
        spine=args.spine,
        repetitions=args.repetitions,
        token=token,
        catalog_path=args.catalog_path,
    )
    payload = result.to_json() + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
