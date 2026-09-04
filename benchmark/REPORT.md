# DuckPD vs pandas Benchmark Report

**Generated:** 2026-09-04 08:19:04 UTC  
**DuckPD:** `0.0.7` | **pandas:** `3.0.5` | **DuckDB:** `1.5.5`

## 1. Executive Summary

- **Execution Speed:** DuckPD delivers up to **4.31x speedup** (avg **2.60x**) over pandas across workloads.
- **Process Memory (RSS):** DuckPD reduces peak physical memory by up to **186.83x** through vectorized query pushdowns.
- **Python Heap Footprint:** DuckPD uses up to **13,805x lower Python heap** as intermediate states stay within DuckDB.
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
| **DuckPD Version** | `0.0.7` |
| **DuckDB Version** | `1.5.5` |
| **pandas Version** | `3.0.5` |
| **PyArrow Version** | `25.0.1` |
| **Benchmark Config** | 4 worker threads, 2 isolated repetitions |

---

## 3. High-Level Performance Comparison

| Dataset | Size | Rows | Workload | DuckPD Time | pandas Time | Speedup | DuckPD Peak RSS | pandas Peak RSS | RSS Reduction | Verification |
|---|---|---|---|---|---|---|---|---|---|---|
| **5mb** | 5.03 MB | 326,394 | `filter_groupby_agg` | **40.44 ms** | 61.26 ms | **1.51x** | **164.43 MB** | 286.25 MB | **1.74x** | Passed |
| **50mb** | 50.29 MB | 3,263,944 | `filter_groupby_agg` | **72.39 ms** | 145.69 ms | **2.01x** | **174.06 MB** | 773.08 MB | **4.44x** | Passed |
| **500m** | 502.87 MB | 32,639,441 | `filter_groupby_agg` | **320.80 ms** | 827.60 ms | **2.58x** | **179.04 MB** | 4.88 GB | **27.27x** | Passed |
| **5g** | 5.03 GB | 326,394,406 | `filter_groupby_agg` | **2.8512 s** | 12.2891 s | **4.31x** | **203.36 MB** | 37.99 GB | **186.83x** | Passed |

---

## 4. Detailed Metrics Breakdown

| Preset | Workload | Engine | Time (Median) | Time (Min-Max) | Peak RSS | Peak Python Heap | Throughput (MB/s) | Throughput (Rows/s) | Status |
|---|---|---|---|---|---|---|---|---|---|
| **5mb** | `filter_groupby_agg` | **DuckPD** | **40.44 ms** | 36.20 ms - 44.68 ms | **164.43 MB** | **358.14 KB** | 124.32 MB/s | 8.07 M rows/s | SUCCESS |
| **5mb** | `filter_groupby_agg` | pandas | 61.26 ms | 58.86 ms - 63.66 ms | 286.25 MB | 6.82 MB | 82.06 MB/s | 5.33 M rows/s | SUCCESS |
| **50mb** | `filter_groupby_agg` | **DuckPD** | **72.39 ms** | 69.43 ms - 75.34 ms | **174.06 MB** | **358.18 KB** | 694.78 MB/s | 45.09 M rows/s | SUCCESS |
| **50mb** | `filter_groupby_agg` | pandas | 145.69 ms | 144.95 ms - 146.43 ms | 773.08 MB | 51.28 MB | 345.20 MB/s | 22.40 M rows/s | SUCCESS |
| **500m** | `filter_groupby_agg` | **DuckPD** | **320.80 ms** | 279.66 ms - 361.93 ms | **179.04 MB** | **357.72 KB** | 1567.58 MB/s | 101.75 M rows/s | SUCCESS |
| **500m** | `filter_groupby_agg` | pandas | 827.60 ms | 824.47 ms - 830.74 ms | 4.88 GB | 495.88 MB | 607.63 MB/s | 39.44 M rows/s | SUCCESS |
| **5g** | `filter_groupby_agg` | **DuckPD** | **2.8512 s** | 2.4137 s - 3.2887 s | **203.36 MB** | **358.00 KB** | 1763.73 MB/s | 114.48 M rows/s | SUCCESS |
| **5g** | `filter_groupby_agg` | pandas | 12.2891 s | 9.0124 s - 15.5659 s | 37.99 GB | 4.94 GB | 409.20 MB/s | 26.56 M rows/s | SUCCESS |

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
