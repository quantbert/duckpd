<p align="center">
  <img src="duckpd.png" alt="DuckPD mascot - a duck dressed as a panda" width="280">
</p>

# DuckPD 🦆❤️🐼

**DuckPD is DuckDB dressed as a pandas DataFrame.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DuckDB Powered](https://img.shields.io/badge/powered%20by-DuckDB%201.5-yellowgreen.svg)](https://duckdb.org/)
[![Narwhals Compliant](https://img.shields.io/badge/Narwhals-Lazy%20Backend-brightgreen.svg)](docs/NARWHALS_COMPATIBILITY.md)

DuckPD is a lazy, out-of-core DataFrame library with a familiar pandas-shaped API and DuckDB as its high-performance analytical execution engine. Write intuitive pandas code; execute at DuckDB speed across 100 GB+ datasets with bounded memory and zero OOM crashes.

---

## ⚡ Why DuckPD? The Core Advantages

* **🚫 Never OOM on Large Datasets:** Stream, filter, join, and aggregate multi-gigabyte or multi-terabyte datasets within a bounded memory footprint (e.g. 2 GB RAM budget) using DuckDB's vectorized execution and disk spillover.
* **📈 Zero-Materialization Grouped Rolling Windows:** Compute grouped, multi-entity rolling/expanding statistics (e.g., 20-day vs 50-day moving average crossovers across thousands of stock tickers) and assign them directly back to your dataframe without materializing intermediate tables.
* **🌐 Federated Remote Queries:** Query HTTP/S3/GCS Parquet and attach PostgreSQL, MySQL, and SQLite databases directly into lazy pandas pipelines—with full credential redaction and scan guardrails.
* **🛡️ Zero Silent Fallbacks:** If an operation is unsupported or ordering is ambiguous, DuckPD fails explicitly before query execution. Your dataset will never be silently materialized into in-memory pandas.
* **🔌 Native Narwhals Lazy Backend:** Drop DuckPD directly into modern visualization and machine learning libraries (Plotly, Altair, etc.) via `nw.from_native(df)` for zero-copy, lazy DuckDB execution.
* **🔍 Deep Observability:** Inspect physical plans, optimizer pushdown, operator timings, peak RSS, and DuckDB spill metrics with `df.explain()`, `df.explain_write()`, and `df.profile()`.

---

## 🚀 Quickstart & Example Workflows

### 1. Complex Feature Engineering & Aggregations Per Ticker

Compute rolling indicators and aggregate summary statistics per group/ticker without materializing intermediate tables:

```python
import duckpd as pd

# Lazy Parquet scan with explicit multi-column ordering
prices = pd.read_parquet(
    "price-data/*.parquet",
    order_by=["date", "ticker"],
)

# 1. Grouped rolling indicators computed per ticker (aligned to source rows)
features = prices.assign(
    return_pct=lambda df: df.groupby("ticker")["close"].pct_change(),
    fast_ma=lambda df: df.groupby("ticker")["close"].rolling(20).mean(),
    slow_ma=lambda df: df.groupby("ticker")["close"].rolling(50).mean(),
).assign(ma_cross=lambda df: df["fast_ma"] > df["slow_ma"])

# 2. Groupby aggregations per ticker
ticker_summary = (
    features.groupby("ticker", as_index=False)
    .agg(
        avg_daily_return=("return_pct", "mean"),
        max_high=("high", "max"),
        min_low=("low", "min"),
        total_volume=("volume", "sum"),
        crossover_signals=("ma_cross", "sum"),
    )
    .sort_values("total_volume", ascending=False)
)

# Inspect query plan without executing
print(ticker_summary.explain("optimized"))

# Materialize a bounded preview in memory or stream full result straight to Parquet
preview = ticker_summary.head(10)  # Bounded pandas DataFrame
features.write_parquet("price-features.parquet")  # Direct zero-copy DuckDB sink
```

---

### 2. Federated Cloud Parquet & Remote Databases

Join remote cloud Parquet datasets with live PostgreSQL/MySQL reporting tables in a single lazy pipeline:

```python
import os
import duckpd as pd

with pd.connect(memory_limit="2GB") as session:
    # Read private S3 Parquet using scoped temporary secrets
    session.create_s3_secret(
        "analytics",
        key_id=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        region="us-east-1",
        scope="s3://company-warehouse/",
    )
    events = pd.read_parquet("s3://company-warehouse/clickstream/*.parquet", session=session)

    # Attach PostgreSQL read-only without leaking credentials in logs or reprs
    warehouse = session.attach_postgres(
        "warehouse",
        host=os.environ["PGHOST"],
        database=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        unbounded_scan="allow",
    )
    users = warehouse.table("users", order_by="user_id")

    # Lazy relational join and aggregation
    active_user_metrics = (
        events.merge(users, on="user_id")
        .groupby(["country", "subscription_tier"], as_index=False)
        .agg(total_events=("event_id", "count"), total_revenue=("amount", "sum"))
        .sort_values("total_revenue", ascending=False)
    )

    # Collect result into pandas or export to CSV
    top_metrics = active_user_metrics.head(50)
```

---

### 3. Native Narwhals Lazy Interoperability

Use DuckPD seamlessly inside libraries that support Narwhals without collecting to pandas:

```python
import narwhals as nw
import duckpd as pd

df = pd.read_parquet("data.parquet", order_by="id")
lazy_df = nw.from_native(df)

# Narwhals transforms execute lazily inside DuckDB
transformed = lazy_df.with_columns(
    z_score=(nw.col("value") - nw.col("value").mean()) / nw.col("value").std()
).filter(nw.col("z_score") > 2.0)

# Collect cleanly as Arrow or pandas when ready
arrow_table = nw.to_native(transformed).to_arrow()
```

---

## 📊 Benchmarks: DuckPD vs. pandas

DuckPD is benchmarked across dataset sizes from **5 MB to 50 GB** on standard single-node machines.

| Dataset Size | Pandas Execution Time | DuckPD Execution Time | Pandas Peak RSS | DuckPD Peak RSS | DuckPD Heap Traced | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **500 MB** | 1.84s | **0.28s (6.5x faster)** | ~4.8 GB | **~180 MB** | ~358 KB | Complete |
| **5 GB** | 19.6s | **2.10s (9.3x faster)** | ~38 GB | **~850 MB** | ~358 KB | Complete |
| **50 GB** | 💥 **OOM Crash** (>250 GB req.) | **18.4s (Out-of-Core)** | 💥 Out of Memory | **< 2 GB (Spill Bounded)** | ~358 KB | **Zero OOM** |

Run benchmarks locally:
```bash
make benchmark          # Fast everyday suite (5mb, 50mb, 500mb)
make benchmark-all      # Exhaustive stress test (up to 50gb out-of-core)
```
See the complete [Benchmark Report](benchmark/REPORT.md) and [Detailed Benchmarks](docs/BENCHMARK.md).

---

## 🧭 Project Architecture & Contracts

DuckPD's core design philosophy is rooted in **correctness, transparency, and relational rigor**:

* **Explicit Execution Boundaries:** Transformations build typed immutable logical plans. Execution occurs strictly at intentional boundaries: `collect()`, `head(n)`, `to_arrow()`, `to_arrow_batches()`, `write_parquet()`, `write_csv()`, `save_as_table()`, and `commit()`.
* **Honest Ordering & Relational Row Identity:** Pandas assumes physical in-memory row order. DuckPD tracks hidden relational identity metadata so positional `.iloc`, MultiIndex `.loc`, rank ties, and window operations are completely deterministic—failing with `UnorderedOperationError` only when source order is truly undefined.
* **Transactional Parquet Commits:** Update local Parquet files in-place with `df.commit()`, featuring automated schema validation, conflict detection, and atomic staging replacement (`os.replace`).

For in-depth architectural specifications and design decisions:
* [Core Project Directives & Relational Architecture](docs/decisions/0003-directives-and-architecture.md)
* [Ordering, Indexing & Session Contracts](docs/decisions/0002-order-index-session-contract.md)
* [API Compatibility & Semantic Guide](docs/COMPATIBILITY.md)
* [Narwhals Lazy-Frame Compliance Matrix](docs/NARWHALS_COMPATIBILITY.md)

---

## 📚 Demos & Walkthroughs

Check out the runnable tutorials and interactive notebooks in [demo/](demo/README.md):

* 📘 [`demo/DuckPD_Quickstart.ipynb`](demo/DuckPD_Quickstart.ipynb) — 5-minute interactive introduction.
* 📈 [`demo/DuckPD_Features_Walkthrough.ipynb`](demo/DuckPD_Features_Walkthrough.ipynb) — Deep-dive across 3.9M rows of AlphaDojo stock news data.
* 🔬 [`demo/DuckPD_Order_Index_Window_Workflows.ipynb`](demo/DuckPD_Order_Index_Window_Workflows.ipynb) — Differential walkthrough of rolling windows, `.loc`/`.iloc` mechanics, and persistence.

Runnable pipelines:
```bash
uv run python demo/basic_pipeline.py
uv run python demo/parquet_pipeline.py
uv run python demo/reduction_pipeline.py
uv run python demo/generate_market_data.py
uv run python demo/market_data_demo.py smoke
```

---

## 🛠️ Development & Quality Gate

DuckPD uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environment management.

```bash
# Set up dependencies
uv sync --frozen --group dev

# Run quality gate
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv build
```

Or using Make:
```bash
make check
make build
```

---

## 📄 License & Documentation

* **License:** [MIT License](LICENSE)
* **Documentation Index:** [docs/README.md](docs/README.md)
* **Release Notes:** [CHANGELOG.md](docs/CHANGELOG.md)
* **Roadmap:** [docs/roadmap.md](docs/roadmap.md)
* **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)