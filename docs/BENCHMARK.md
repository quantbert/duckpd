# DuckPD vs pandas Benchmark Results

This document provides reproducible benchmarks comparing **DuckPD** and standard **pandas** on analytical time-series workloads over synthetic OHLC market datasets.

## Benchmark Overview

The benchmark runs an analytical workflow typical in quantitative finance and data science:
1. **Parquet Scan & Predicate Filter**: Filter by ticker (`ticker == 'NVDA'`) across millions of rows.
2. **Column Calculations**: Derive bar returns `(close - open) / open` and bar ranges `(high - low)`.
3. **GroupBy Aggregation**: Group by `ticker` and compute summary statistics:
   - `avg_return = ("bar_return", "mean")`
   - `avg_range = ("bar_range", "mean")`
   - `max_high = ("high", "max")`
   - `min_low = ("low", "min")`
   - `total_bars = ("close", "count")`
4. **Semantic Validation**: Both engines must produce numerically and structurally identical output (`assert_frame_equal`).

## Validated release tracks

`python -m benchmark.tracks` generates deterministic inputs and runs three
separate result-validated tracks:

- `tpch_q1`: the TPC-H Q1 grouping and pricing-expression shape;
- `db_groupby_join`: a db-benchmark-style customer/order join followed by
  regional aggregation;
- `synthetic_ohlc`: the existing market-data GroupBy workload.

Each DuckPD run records planning time, execution time, process peak RSS, spill
directory bytes, and result equality against direct DuckDB SQL. Direct DuckDB
SQL and pandas are timed baselines. Polars and FireDucks are recorded as
`unavailable` when not installed and `unsupported` when installed without a
validated adapter; unavailable, unsupported, failed, and OOM outcomes remain in
the JSON instead of being removed.

```bash
make benchmark-tracks
make optimizer-gate
```

The first command writes `benchmark/TRACKS.json`. The second warms and
alternates optimized/unoptimized plans, checks Arrow result equality, measures
each optimizer pass by ablation, and fails if optimized median execution exceeds
the configured regression ratio.


### Why this workload favors DuckPD

- **Predicate & Projection Pushdown**: DuckPD compiles the lazy plan directly to DuckDB's vectorized query engine. Rather than deserializing all columns and rows into Python heap memory, DuckDB scans only the requested columns (`ticker`, `open`, `close`, `high`, `low`) and pushes filtering into the Parquet reader.
- **Multithreading & Vectorized Execution**: DuckDB processes batches in parallel across CPU cores using SIMD instructions and cache-friendly column vectors.
- **Small Python heap footprint**: traced Python allocations remain near 358 KB
    because intermediate calculations stay in DuckDB until the small aggregate is
    collected. This is not a measurement of DuckDB native memory or total
    process RSS.

---

## Environment & Hardware

| Parameter | Value |
|---|---|
| **OS** | Linux (Ubuntu, x86_64, Kernel 7.0.0-27-generic) |
| **CPU** | AMD Ryzen AI 9 HX 370, 12 cores / 24 logical CPUs |
| **Python Version** | 3.12.13 |
| **DuckDB Version** | 1.5.5 |
| **pandas Version** | 3.0.5 |
| **PyArrow Version** | 25.0.1 |
| **Package Version Reported** | 0.0.7 |
| **Repository State** | Commit `d67dca2` (development state; later documentation-only changes excluded) |
| **DuckPD Threads** | 4 |
| **Measurement Method** | Three isolated worker subprocesses per engine (`spawn`), alternating engine order; median and observed range from `time.perf_counter()`; peak traced Python heap from `tracemalloc` |

---

## Benchmark Results

### Current results: 2026-09-04

| Dataset | Parquet Size | Rows | DuckPD Median (Range) | pandas Median (Range) | Median Speedup | DuckPD Traced Heap | pandas Traced Heap | Verification |
|---|---|---|---|---|---|---|---|---|
| **Smoke** | 4.99 MB | 323,618 | **0.0365 s** (0.0342-0.0428) | 0.0589 s (0.0586-0.0599) | **1.61x** | **357.96 KB** | 6.77 MB | 3/3 identical |
| **100 MB** | 99.72 MB | 6,472,353 | **0.0993 s** (0.0827-0.1130) | 0.2096 s (0.2035-0.2288) | **2.11x** | **357.88 KB** | 99.84 MB | 3/3 identical |
| **1 GB** | 997.18 MB | 64,723,528 | **0.4965 s** (0.4890-0.5005) | 1.5497 s (1.4535-1.7710) | **3.12x** | **358.00 KB** | 981.49 MB | 3/3 identical |
| **5 GB** | 4.99 GB | 323,617,641 | **2.3876 s** (2.3177-2.6055) | 9.4497 s (7.6746-12.9861) | **3.96x** | **357.88 KB** | 4.90 GB | 3/3 identical |

Peak RSS and spill bytes are process-level observations recorded by the current
benchmark harness. They are not guarantees for other data distributions,
operators, filesystems, or DuckDB versions.

### Historical comparison & trends

Comparing the previous measured development baseline against the current
development state reporting package version 0.0.7:

| Dataset | Previous Baseline | Current Development State | DuckPD Delta | pandas Median | Current Speedup |
|---|---:|---:|---:|---:|---:|
| Smoke | 0.0334 s | 0.0365 s | +0.0031 s | 0.0589 s | **1.61x** |
| 100 MB | 0.0787 s | 0.0993 s | +0.0206 s | 0.2096 s | **2.11x** |
| 1 GB | 0.4784 s | 0.4965 s | +0.0181 s | 1.5497 s | **3.12x** |
| 5 GB | 2.3277 s | 2.3876 s | +0.0599 s | 9.4497 s | **3.96x** |

Execution times scale sub-linearly relative to input size: scanning and aggregating 5 GB takes 2.39 s vs 0.49 s for 1 GB, while pandas scales to ~9.45 s (increasing DuckPD's advantage from 1.6x on small files to ~4x on multi-gigabyte datasets).

### Memory scaling & Python heap invariance

DuckPD's peak traced Python heap remains flat at **~358 KB** across all dataset sizes, from 4.99 MB up to 4.99 GB (over **13,600x** lower than pandas on the 5 GB dataset).

- **Why Python heap stays flat**: DuckPD compiles lazy relational pipelines directly to DuckDB. Intermediate allocations—scanning columns, computing bar return/range expressions, grouping, and reducing—occur within DuckDB's C++ engine. Python materialization is deferred until the 1-row summary DataFrame is collected.
- **Traced heap vs process RSS**: `tracemalloc` observes only Python object allocations and does not reflect DuckDB native memory or OS-level page cache. `tests/test_execution_limits.py` exercises strict DuckDB memory settings, but does not measure RSS, spill files, or spill bytes.

### Dataset provenance

| Dataset | SHA-256 |
|---|---|
| Smoke | `c2e035ee0051fa692c6ae68b1c804f993f26a2d6e74e3a9dc935648322b096b8` |
| 100 MB | `810ed2244bff0088f6170cd89b7224f8b1373253eb74aa0d09f960c7e98feff9` |
| 1 GB | `35a26bf158fcbb8a1aae1f33ce06e43f7c497abcdf6b8e1636c4b18ea3d7106a` |
| 5 GB | `d20db6194f2399231343016ff9977ccc88d937bba628cee07e7a0b96122c5f8f` |

---

## Reproduction Instructions

Run the following commands from the repository root.

### 1. Generate the Synthetic Datasets

Generate the calibrated OHLC datasets using the generator utility in `demo/`:

```bash
# Generate smoke (~5 MB) and 100 MB datasets
uv run python demo/generate_market_data.py smoke 100mb

# Generate 1 GB and 5 GB datasets
uv run python demo/generate_market_data.py 1gb 5gb
```

Generated files are saved in `demo/data/`:
- `market-data-smoke.parquet` (~5 MB, 323k rows)
- `market-data-100mb.parquet` (~100 MB, 6.47M rows)
- `market-data-1gb.parquet` (~1 GB, 64.7M rows)
- `market-data-5gb.parquet` (~5 GB, 323.6M rows)

### 2. Run the Benchmark

Execute `demo/market_data_demo.py` for individual presets or all datasets:

```bash
# Run benchmark on a specific preset
uv run python demo/market_data_demo.py 100mb
uv run python demo/market_data_demo.py 1gb
uv run python demo/market_data_demo.py 5gb

# Run benchmark across all available datasets
uv run python demo/market_data_demo.py all
```

The default is three repetitions per engine. Every repetition runs in a fresh
subprocess, engine order alternates, and any semantic mismatch fails the run.

### 3. Optional Benchmark Options

```bash
# Filter on a different ticker (e.g. AAPL, MSFT, TSLA, JPM)
uv run python demo/market_data_demo.py 1gb --ticker AAPL

# Specify custom worker thread count for DuckPD
uv run python demo/market_data_demo.py 1gb --threads 8

# Increase repetitions for a more stable local comparison
uv run python demo/market_data_demo.py 1gb --repetitions 7

# Skip pandas run (e.g. if memory is constrained on large files)
uv run python demo/market_data_demo.py 5gb --skip-pandas
```

DuckPD's thread count is controlled by `--threads`; pandas/PyArrow uses its
runtime default. Results are therefore representative of the default user
experience, not a matched-thread microbenchmark. Filesystem cache state and CPU
frequency are also uncontrolled. Use the observed ranges and repeat on the
target deployment hardware before drawing operational conclusions.

---

## Code Example Comparison

### DuckPD Pipeline

```python
import duckpd as pd

with pd.connect(threads=4) as session:
    df = session.read_parquet("demo/data/market-data-1gb.parquet")
    result = (
        df[df["ticker"] == "NVDA"]
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
        .collect()
    )
```

### Equivalent pandas Pipeline

```python
import pandas as pd

df = pd.read_parquet("demo/data/market-data-1gb.parquet")
filtered = df[df["ticker"] == "NVDA"].copy()
filtered["bar_return"] = (filtered["close"] - filtered["open"]) / filtered["open"]
filtered["bar_range"] = filtered["high"] - filtered["low"]

result = filtered.groupby("ticker", as_index=False).agg(
    avg_return=("bar_return", "mean"),
    avg_range=("bar_range", "mean"),
    max_high=("high", "max"),
    min_low=("low", "min"),
    total_bars=("close", "count"),
)
```
