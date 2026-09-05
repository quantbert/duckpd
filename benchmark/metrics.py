"""Metrics collection, isolated execution, and comparison analysis."""

from __future__ import annotations

import multiprocessing
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd_orig

from benchmark.generate import DatasetMetadata
from benchmark.workloads import Workload


def get_peak_rss_bytes() -> int:
    """Return peak resident set size (RSS) in bytes for current process."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except OSError:
            pass

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return int(counters.PeakWorkingSetSize)
        except Exception:
            pass

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            return usage.ru_maxrss
        return usage.ru_maxrss * 1024
    except ImportError:
        pass

    try:
        from importlib import import_module

        psutil = import_module("psutil")
        process = cast("Any", psutil).Process()
        return int(process.memory_info().rss)
    except ImportError:
        pass

    return 0


def human_bytes(size: float | int) -> str:
    """Format byte count into human-readable representation."""
    val = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(val) < 1000.0 or unit == "TB":
            return f"{val:.2f} {unit}"
        val /= 1000.0
    return f"{val:.2f} TB"


def human_seconds(sec: float) -> str:
    """Format seconds into appropriate unit."""
    if sec < 0.001:
        return f"{sec * 1_000_000:.1f} µs"
    if sec < 1.0:
        return f"{sec * 1000.0:.2f} ms"
    return f"{sec:.4f} s"


def human_throughput(rate: float, unit: str = "rows/s") -> str:
    """Format throughput rate into human-readable representation."""
    if rate >= 1_000_000_000:
        return f"{rate / 1_000_000_000:.2f} B {unit}"
    if rate >= 1_000_000:
        return f"{rate / 1_000_000:.2f} M {unit}"
    if rate >= 1_000:
        return f"{rate / 1_000:.2f} K {unit}"
    return f"{rate:.1f} {unit}"


@dataclass(frozen=True)
class RunMetric:
    """Measurement from a single run in an isolated subprocess."""

    engine: str
    elapsed_seconds: float
    peak_rss_bytes: int
    peak_heap_bytes: int
    result_rows: int
    success: bool
    error_message: str | None = None


@dataclass(frozen=True)
class AggregatedMetric:
    """Summary metrics across repeated executions."""

    engine: str
    median_time: float
    min_time: float
    max_time: float
    median_rss_bytes: int
    median_heap_bytes: int
    throughput_mb_s: float
    throughput_rows_s: float
    result_rows: int
    success: bool
    error_message: str | None = None


@dataclass(frozen=True)
class BenchmarkComparison:
    """Comparative results between DuckPD and pandas for one preset & workload."""

    preset: str
    workload_name: str
    workload_description: str
    file_size_bytes: int
    row_count: int
    duckpd: AggregatedMetric
    pandas: AggregatedMetric | None
    speedup: float | None
    rss_reduction: float | None
    heap_reduction: float | None
    verified: bool
    verification_notes: str


def _worker_wrapper(
    func: Callable[..., pd_orig.DataFrame],
    args: tuple[Any, ...],
    queue: multiprocessing.Queue[
        tuple[pd_orig.DataFrame | None, float, int, int, int, bool, str | None]
    ],
) -> None:
    """Execute target function inside fresh child process to isolate memory."""
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        df = func(*args)
        t1 = time.perf_counter()
        _, peak_heap = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_rss = get_peak_rss_bytes()
        queue.put((df, t1 - t0, peak_rss, peak_heap, len(df), True, None))
    except Exception as exc:
        t1 = time.perf_counter()
        tracemalloc.stop()
        peak_rss = get_peak_rss_bytes()
        queue.put((None, t1 - t0, peak_rss, 0, 0, False, f"{type(exc).__name__}: {exc}"))


def _run_in_subprocess(
    func: Callable[..., pd_orig.DataFrame], *args: Any
) -> tuple[pd_orig.DataFrame | None, RunMetric]:
    """Run function in isolated spawn subprocess to measure memory accurately."""
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue[
        tuple[pd_orig.DataFrame | None, float, int, int, int, bool, str | None]
    ] = ctx.Queue()
    proc = ctx.Process(target=_worker_wrapper, args=(func, args, queue))
    proc.start()
    proc.join()

    engine_name = "DuckPD" if "duckpd" in func.__name__ else "pandas"

    if proc.exitcode != 0:
        # Exit code indicates subprocess crashed (e.g. killed by SIGKILL / OOM killer)
        error_msg = f"Subprocess crashed (exit code {proc.exitcode}, likely OOM)"
        return None, RunMetric(
            engine=engine_name,
            elapsed_seconds=0.0,
            peak_rss_bytes=0,
            peak_heap_bytes=0,
            result_rows=0,
            success=False,
            error_message=error_msg,
        )

    if queue.empty():
        return None, RunMetric(
            engine=engine_name,
            elapsed_seconds=0.0,
            peak_rss_bytes=0,
            peak_heap_bytes=0,
            result_rows=0,
            success=False,
            error_message="Subprocess produced no result",
        )

    df, elapsed, peak_rss, peak_heap, rows, success, err = queue.get()
    return df, RunMetric(
        engine=engine_name,
        elapsed_seconds=elapsed,
        peak_rss_bytes=peak_rss,
        peak_heap_bytes=peak_heap,
        result_rows=rows,
        success=success,
        error_message=err,
    )


def aggregate_runs(runs: list[RunMetric], file_size_bytes: int, row_count: int) -> AggregatedMetric:
    """Aggregate repeated benchmark runs into statistical summary."""
    if not runs:
        raise ValueError("Cannot aggregate empty runs")

    engine = runs[0].engine
    successful = [r for r in runs if r.success]

    if not successful:
        first_err = runs[0].error_message or "Execution failed"
        return AggregatedMetric(
            engine=engine,
            median_time=0.0,
            min_time=0.0,
            max_time=0.0,
            median_rss_bytes=0,
            median_heap_bytes=0,
            throughput_mb_s=0.0,
            throughput_rows_s=0.0,
            result_rows=0,
            success=False,
            error_message=first_err,
        )

    times = [r.elapsed_seconds for r in successful]
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    median_rss = int(statistics.median([r.peak_rss_bytes for r in successful]))
    median_heap = int(statistics.median([r.peak_heap_bytes for r in successful]))

    throughput_mb_s = (
        (file_size_bytes / (1000.0 * 1000.0)) / median_time if median_time > 0 else 0.0
    )
    throughput_rows_s = row_count / median_time if median_time > 0 else 0.0

    return AggregatedMetric(
        engine=engine,
        median_time=median_time,
        min_time=min_time,
        max_time=max_time,
        median_rss_bytes=median_rss,
        median_heap_bytes=median_heap,
        throughput_mb_s=throughput_mb_s,
        throughput_rows_s=throughput_rows_s,
        result_rows=successful[0].result_rows,
        success=True,
    )


def run_benchmark(
    dataset: DatasetMetadata,
    workload: Workload,
    *,
    repetitions: int = 3,
    threads: int = 4,
    skip_pandas: bool = False,
    max_pandas_bytes: int = 15_000_000_000,
) -> BenchmarkComparison:
    """Run benchmark comparison on a dataset and workload."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    path_str = str(dataset.path)
    duck_runs: list[RunMetric] = []
    pandas_runs: list[RunMetric] = []
    verified = False
    verification_notes = "Not verified"

    # Determine whether pandas should be skipped due to safety threshold (e.g. 50GB)
    pandas_auto_skipped = False
    if not skip_pandas and dataset.file_size_bytes > max_pandas_bytes:
        pandas_auto_skipped = True
        verification_notes = (
            f"pandas skipped (file size {human_bytes(dataset.file_size_bytes)} > "
            f"safety limit {human_bytes(max_pandas_bytes)})"
        )

    last_duck_df: pd_orig.DataFrame | None = None
    last_pandas_df: pd_orig.DataFrame | None = None

    for rep in range(repetitions):
        run_pandas_first = (rep % 2 == 1) and not skip_pandas and not pandas_auto_skipped
        engines = ("pandas", "duckpd") if run_pandas_first else ("duckpd", "pandas")
        for engine in engines:
            if engine == "duckpd":
                df, metric = _run_in_subprocess(workload.run_duckpd, path_str, threads)
                duck_runs.append(metric)
                if df is not None:
                    last_duck_df = df
            else:
                if skip_pandas or pandas_auto_skipped:
                    continue
                df, metric = _run_in_subprocess(workload.run_pandas, path_str)
                pandas_runs.append(metric)
                if df is not None:
                    last_pandas_df = df

    duck_summary = aggregate_runs(duck_runs, dataset.file_size_bytes, dataset.row_count)

    pandas_summary: AggregatedMetric | None = None
    if not skip_pandas and not pandas_auto_skipped:
        pandas_summary = aggregate_runs(pandas_runs, dataset.file_size_bytes, dataset.row_count)

    # Verification
    if (
        duck_summary.success
        and pandas_summary is not None
        and pandas_summary.success
        and last_duck_df is not None
        and last_pandas_df is not None
    ):
        try:
            workload.verify(last_duck_df, last_pandas_df)
            verified = True
            verification_notes = f"Verified identical ({repetitions}/{repetitions} runs)"
        except Exception as exc:
            verified = False
            verification_notes = f"Mismatch: {exc}"
    elif skip_pandas:
        verification_notes = "pandas skipped by user flag"
    elif pandas_summary is not None and not pandas_summary.success:
        verification_notes = f"pandas failed: {pandas_summary.error_message or 'unknown error'}"

    # Compute comparative ratios
    speedup: float | None = None
    rss_reduction: float | None = None
    heap_reduction: float | None = None

    if pandas_summary is not None and pandas_summary.success and duck_summary.success:
        if duck_summary.median_time > 0:
            speedup = pandas_summary.median_time / duck_summary.median_time
        if duck_summary.median_rss_bytes > 0:
            rss_reduction = pandas_summary.median_rss_bytes / duck_summary.median_rss_bytes
        if duck_summary.median_heap_bytes > 0:
            heap_reduction = pandas_summary.median_heap_bytes / duck_summary.median_heap_bytes

    return BenchmarkComparison(
        preset=dataset.preset,
        workload_name=workload.name,
        workload_description=workload.description,
        file_size_bytes=dataset.file_size_bytes,
        row_count=dataset.row_count,
        duckpd=duck_summary,
        pandas=pandas_summary,
        speedup=speedup,
        rss_reduction=rss_reduction,
        heap_reduction=heap_reduction,
        verified=verified,
        verification_notes=verification_notes,
    )
