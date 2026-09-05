# DuckPD vs pandas Benchmark Report

**Generated:** 2026-09-05 11:31:54 UTC  
**DuckPD:** `0.1.3` | **pandas:** `3.0.5` | **DuckDB:** `1.5.5`

## 1. Executive Summary

- **Execution Speed:** DuckPD delivers up to **2.94x speedup** (avg **2.17x**) over pandas across workloads.
- **Process Memory (RSS):** DuckPD reduces peak physical memory by up to **179.63x** through vectorized query pushdowns.
- **Python Heap Footprint:** DuckPD uses up to **12,250x lower Python heap** as intermediate states stay within DuckDB.
- **Out-of-Core Scalability:** DuckPD executes large queries with bounded memory, avoiding pandas OOM crashes on large files.
- **Mathematical Equivalence:** 100% of tested runs produce verified identical results against pandas (`assert_frame_equal`).

---

## 2. Environment & System Specifications

| Parameter | Specification |
|---|---|
| **Operating System** | Linux (7.0.0-27-generic, x86_64) |
| **Processor (CPU)** | AMD Ryzen AI 9 HX 370 w/ Radeon 890M |
| **Logical Cores** | 24 logical CPUs |
| **Total System RAM** | 94.19 GB |
| **Python Version** | `3.12.13` |
| **DuckPD Version** | `0.1.3` |
| **DuckDB Version** | `1.5.5` |
| **pandas Version** | `3.0.5` |
| **PyArrow Version** | `25.0.1` |
| **Benchmark Config** | 4 worker threads, 3 isolated repetitions |

---

## 3. High-Level Performance Comparison

| Dataset | Size | Rows | Workload | DuckPD Time | pandas Time | Speedup | DuckPD Peak RSS | pandas Peak RSS | RSS Reduction | Verification |
|---|---|---|---|---|---|---|---|---|---|---|
| **5mb** | 5.03 MB | 326,394 | `filter_groupby_agg` | **45.93 ms** | 48.95 ms | **1.07x** | **174.13 MB** | 290.87 MB | **1.67x** | Passed |
| **50mb** | 50.29 MB | 3,263,944 | `filter_groupby_agg` | **70.18 ms** | 131.62 ms | **1.88x** | **184.07 MB** | 795.02 MB | **4.32x** | Passed |
| **500m** | 502.87 MB | 32,639,441 | `filter_groupby_agg` | **288.79 ms** | 803.03 ms | **2.78x** | **189.18 MB** | 4.88 GB | **25.79x** | Passed |
| **5g** | 5.03 GB | 326,394,406 | `filter_groupby_agg` | **2.4566 s** | 7.2339 s | **2.94x** | **212.30 MB** | 38.14 GB | **179.63x** | Passed |

---

## 4. Detailed Metrics Breakdown

| Preset | Workload | Engine | Time (Median) | Time (Min-Max) | Peak RSS | Peak Python Heap | Throughput (MB/s) | Throughput (Rows/s) | Status |
|---|---|---|---|---|---|---|---|---|---|
| **5mb** | `filter_groupby_agg` | **DuckPD** | **45.93 ms** | 43.60 ms - 56.49 ms | **174.13 MB** | **402.83 KB** | 109.46 MB/s | 7.11 M rows/s | SUCCESS |
| **5mb** | `filter_groupby_agg` | pandas | 48.95 ms | 47.60 ms - 50.77 ms | 290.87 MB | 5.71 MB | 102.71 MB/s | 6.67 M rows/s | SUCCESS |
| **50mb** | `filter_groupby_agg` | **DuckPD** | **70.18 ms** | 68.37 ms - 82.15 ms | **184.07 MB** | **402.89 KB** | 716.61 MB/s | 46.51 M rows/s | SUCCESS |
| **50mb** | `filter_groupby_agg` | pandas | 131.62 ms | 125.18 ms - 163.13 ms | 795.02 MB | 50.17 MB | 382.09 MB/s | 24.80 M rows/s | SUCCESS |
| **500m** | `filter_groupby_agg` | **DuckPD** | **288.79 ms** | 283.54 ms - 294.05 ms | **189.18 MB** | **402.92 KB** | 1741.31 MB/s | 113.02 M rows/s | SUCCESS |
| **500m** | `filter_groupby_agg` | pandas | 803.03 ms | 783.56 ms - 814.99 ms | 4.88 GB | 494.78 MB | 626.22 MB/s | 40.65 M rows/s | SUCCESS |
| **5g** | `filter_groupby_agg` | **DuckPD** | **2.4566 s** | 2.4308 s - 2.4593 s | **212.30 MB** | **403.33 KB** | 2047.08 MB/s | 132.87 M rows/s | SUCCESS |
| **5g** | `filter_groupby_agg` | pandas | 7.2339 s | 7.1006 s - 7.2495 s | 38.14 GB | 4.94 GB | 695.17 MB/s | 45.12 M rows/s | SUCCESS |

---

## 5. Architectural Advantages & Memory Analysis

### Why DuckPD Outperforms Standard pandas

1. **Lazy Execution & Relational Query Plan**: DuckPD compiles operations into DuckDB's logical plan. Calculations are not performed eagerly.
2. **Predicate & Projection Pushdown**: Filters pass directly to the storage layer. Unneeded columns and row groups are skipped at I/O time.
3. **Vectorized Multithreaded Engine**: DuckDB's C++ execution engine processes columnar vectors in parallel using SIMD instructions.
4. **Out-of-Core Scalability (Handling 5GB and 50GB)**: pandas loads all uncompressed data into RAM (which for 50 GB Parquet can exceed 200 GB, causing fatal OOM). DuckPD streams in chunks with bounded memory.

---

## 6. How to Reproduce

To run the benchmarks locally and regenerate this report:

```bash
# Run standard benchmarks (5MB, 50MB, 500MB)
make benchmark

# Run all benchmark presets including multi-gigabyte files (up to 50GB)
make benchmark SIZES="5mb 50mb 500m 5g 50g"

# Direct CLI invocation with custom parameters
uv run python -m benchmark.runner --sizes 5mb 50mb 500m --repetitions 3 --threads 4 --report benchmark/REPORT.md
```
