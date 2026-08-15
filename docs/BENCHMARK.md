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
| **DuckPD Version** | 0.0.5 |
| **Repository State** | Commit `721bd03` plus the working-tree changes described in this document |
| **DuckPD Threads** | 4 |
| **Measurement Method** | Three isolated worker subprocesses per engine (`spawn`), alternating engine order; median and observed range from `time.perf_counter()`; peak traced Python heap from `tracemalloc` |

---

## Benchmark Results

### Current results: 2026-08-15

| Dataset | Parquet Size | Rows | DuckPD Median (Range) | pandas Median (Range) | Median Speedup | DuckPD Traced Heap | pandas Traced Heap | Verification |
|---|---|---|---|---|---|---|---|---|
| **Smoke** | 4.99 MB | 323,618 | **0.0334 s** (0.0294-0.0340) | 0.0601 s (0.0560-0.0603) | **1.80x** | **357.45 KB** | 6.77 MB | 3/3 identical |
| **100 MB** | 99.72 MB | 6,472,353 | **0.0787 s** (0.0765-0.0997) | 0.1994 s (0.1991-0.2059) | **2.53x** | **357.78 KB** | 99.84 MB | 3/3 identical |
| **1 GB** | 997.18 MB | 64,723,528 | **0.4784 s** (0.4714-0.4845) | 1.4342 s (1.4270-1.4626) | **3.00x** | **357.70 KB** | 981.48 MB | 3/3 identical |
| **5 GB** | 4.99 GB | 323,617,641 | **2.3277 s** (2.2823-2.3672) | 6.8508 s (6.7869-6.8691) | **2.94x** | **357.88 KB** | 4.90 GB | 3/3 identical |

`tracemalloc` does not observe DuckDB or Arrow native allocations. The heap
columns must not be interpreted as total RAM usage or larger-than-memory proof.
Peak RSS, spill bytes, and bytes read remain future benchmark metrics.

### Regression review

The previous table contained one observation per engine. Compared with those
historical values, the current DuckPD medians changed by:

| Dataset | Previous DuckPD | Current Median | Change | pandas Change |
|---|---:|---:|---:|---:|
| Smoke | 0.032 s | 0.0334 s | +4.4% | +0.2% |
| 100 MB | 0.070 s | 0.0787 s | +12.4% | -8.1% |
| 1 GB | 0.465 s | 0.4784 s | +2.9% | -0.1% |
| 5 GB | 2.233 s | 2.3277 s | +4.2% | +0.3% |

The 100 MB result is a regression signal worth tracking; the larger scans show
only a 3-4% shift. This workload reads Parquet and does not exercise the new
hidden row identity used by pandas/Arrow snapshots, so there is no evidence
that row identity caused a scaling regression. The old single observations and
new medians are not statistically equivalent baselines, and no claim below
10% should be treated as conclusive until a persistent benchmark history with
more repetitions and controlled cache state exists.

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
