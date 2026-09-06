"""DuckPD Benchmarking Suite.

Provides multi-size dataset generation, multi-engine benchmarking (DuckPD vs pandas),
system metrics tracking (execution time, peak RSS, traced Python heap, throughput),
and automated Markdown report generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmark.featurestore import (
    FeatureStoreBenchmarkResult,
    run_featurestore_benchmark,
)
from benchmark.generate import PRESET_SIZES, ensure_dataset, generate_dataset
from benchmark.metrics import BenchmarkComparison, run_benchmark
from benchmark.report import generate_markdown_report
from benchmark.workloads import WORKLOADS

if TYPE_CHECKING:
    from benchmark.runner import run_suite

__all__ = [
    "PRESET_SIZES",
    "WORKLOADS",
    "BenchmarkComparison",
    "FeatureStoreBenchmarkResult",
    "ensure_dataset",
    "generate_dataset",
    "generate_markdown_report",
    "run_benchmark",
    "run_featurestore_benchmark",
    "run_suite",
]


def __getattr__(name: str) -> Any:
    if name == "run_suite":
        from benchmark.runner import run_suite

        return run_suite
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
