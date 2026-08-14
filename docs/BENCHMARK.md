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

### Why DuckPD Outperforms pandas

- **Predicate & Projection Pushdown**: DuckPD compiles the lazy plan directly to DuckDB's vectorized query engine. Rather than deserializing all columns and rows into Python heap memory, DuckDB scans only the requested columns (`ticker`, `open`, `close`, `high`, `low`) and pushes filtering into the Parquet reader.
- **Multithreading & Vectorized Execution**: DuckDB processes batches in parallel across CPU cores using SIMD instructions and cache-friendly column vectors.
- **Constant Memory Footprint**: Python heap allocation remains minimal (~350 KB) because intermediate calculations never materialize into pandas/Python object instances until the small aggregated summary is collected.

---

## Environment & Hardware

| Parameter | Value |
|---|---|
| **OS** | Linux (Ubuntu, x86_64, Kernel 7.0.0-27-generic) |
| **CPU Cores** | 24 cores |
| **Python Version** | 3.12.13 |
| **DuckDB Version** | 1.5.5 |
| **pandas Version** | 3.0.5 |
| **PyArrow Version** | 25.0.1 |
| **DuckPD Version** | 0.0.1.dev0 |
| **Measurement Method** | Isolated worker subprocess (`multiprocessing.get_context("spawn")`), wall-time via `time.perf_counter()`, peak Python heap via `tracemalloc`. |

---

## Benchmark Results

### Summary Table

| Dataset Size | Parquet Size | Total Rows | DuckPD Time | pandas Time | Speedup | DuckPD Peak RAM | pandas Peak RAM | RAM Reduction | Output Verification |
|---|---|---|---|---|---|---|---|---|---|
| **Smoke** | 4.99 MB | 323,618 | **0.032 s** | 0.060 s | **1.85x** | **357 KB** | 6.77 MB | **19.0x less** | Identical |
| **100 MB** | 99.72 MB | 6,472,357 | **0.070 s** | 0.217 s | **3.10x** | **358 KB** | 99.83 MB | **279.1x less** | Identical |
| **1 GB** | 997.18 MB | 64,723,568 | **0.465 s** | 1.436 s | **3.09x** | **357 KB** | 981.48 MB | **2,746.8x less** | Identical |
| **5 GB** | 4.99 GB | 323,617,840 | **2.233 s** | 6.829 s | **3.06x** | **357 KB** | 4.90 GB | **13,715.2x less** | Identical |

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

### 3. Optional Benchmark Options

```bash
# Filter on a different ticker (e.g. AAPL, MSFT, TSLA, JPM)
uv run python demo/market_data_demo.py 1gb --ticker AAPL

# Specify custom worker thread count for DuckPD
uv run python demo/market_data_demo.py 1gb --threads 8

# Skip pandas run (e.g. if memory is constrained on large files)
uv run python demo/market_data_demo.py 5gb --skip-pandas
```

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
