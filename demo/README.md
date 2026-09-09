# DuckPD demos

Run these programs from the repository root after installing the development
environment:

```bash
uv sync --frozen --group dev
uv run python demo/basic_pipeline.py
uv run python demo/parquet_pipeline.py
uv run python demo/reduction_pipeline.py
uv run python demo/generate_market_data.py
uv run python demo/market_data_demo.py smoke
```

The embedding tutorials use an explicit PyTorch GPU provider. First create the
accelerator environment described in
[`generate_data/README.md`](generate_data/README.md#amd-rocm-gpu-embeddings).
Then run the script directly from that environment:

```bash
demo/generate_data/.venv-rocm/bin/python demo/vector_search.py
```

The ROCm setup registers a **DuckPD ROCm 7.2.4** Jupyter kernel. In VS Code,
open `demo/DuckPD_Vector_Search.ipynb` or
`demo/featurestore_demo/DuckPD_FeatureStore_Walkthrough.ipynb`, choose
**Select Kernel** in the upper-right, and select that kernel. Restart the
notebook kernel after switching. The FeatureStore notebook's non-embedding
sections also run in the normal development environment; its semantic-search
sections require Transformers and PyTorch. The normal `.venv` intentionally
lacks those accelerator-specific packages.

To use JupyterLab instead, install it into the ROCm environment and launch it
with the same interpreter:

```bash
uv pip install \
  --python demo/generate_data/.venv-rocm/bin/python \
  jupyterlab
demo/generate_data/.venv-rocm/bin/python -m jupyter lab demo
```

PyTorch uses the `cuda` device API for both NVIDIA CUDA and AMD ROCm. The demos
request that device explicitly and fail rather than falling back to CPU.


- `basic_pipeline.py` builds a lazy frame from pandas, sets an explicit index,
  filters rows, calculates columns, tracks ordering, projects, previews, and
  collects the result.
- `parquet_pipeline.py` creates a small Parquet input, scans it lazily, displays
  the execution plan, and writes the transformed result directly to Parquet.
- `reduction_pipeline.py` demonstrates eager `count`, `size`, `sum`, `mean`,
  `min`, and `max` execution over a lazy frame. It covers DataFrame
  `numeric_only`, Series null handling, `skipna`, `min_count`, hidden indexes,
  expression reductions, and the session execution counter.
- `vector_search.py` prepares a pinned Transformers model on an explicitly
  selected PyTorch GPU. On its first run it lazily scans the AlphaDojo
  stock-news Parquet archive over HTTPS, embeds NVIDIA candidates in bounded
  Arrow and model batches, and writes
  `nvidia-news-embedded-transformers.parquet`. Later runs load that local
  dataset directly, perform exact `vector.search_text()` retrieval, and report
  the selected PyTorch runtime plus model-preparation and query-to-response
  timings.
- `generate_market_data.py` calibrates compressed bytes per row, then streams a
  deterministic OHLC time-series dataset directly to Parquet. The safe default
  creates an approximately 5 MB smoke file under `demo/data/`.
- `market_data_demo.py` benchmarks and compares execution time and memory usage
  between DuckPD and standard pandas across OHLC market datasets. Running with
  `smoke` executes in ~3 seconds on a 4.99 MB file.
- `generate_data/` contains the deterministic Nasdaq Stockholm feature-store
  generator. It writes yearly OHLCV and simple-moving-average Parquet partitions,
  reference tables, catalog metadata, and a Hugging Face dataset card.
- `DuckPD_Quickstart.ipynb` is a 5-minute interactive Jupyter Notebook
  demonstrating remote data loading, column reductions, string transformations,
  `groupby` aggregations, query plans, and Parquet exports on the Goodreads
  Books dataset.
- `DuckPD_Features_Walkthrough.ipynb` is a comprehensive interactive walkthrough
  showcasing recent additions (remote cloud parquet, relational joins with
  cardinality validation, `.str` and `.dt` accessors, lossless `duckpd.concat`,
  statistical/quantile reductions, multi-column groupbys, rolling/expanding
  windows, shifts, and persistence) using the AlphaDojo stock news dataset
  (~3.9M rows).
- `DuckPD_Order_Index_Window_Workflows.ipynb` is a self-contained offline
  tutorial covering stable snapshot order, deterministic duplicate/top-N ties,
  MultiIndex and label-list `.loc[[...]]`, two-dimensional `.iloc`, cumulative
  and row windows, automatic file order and explicit relational ordering,
  masked assignment, persistence, and direct DuckDB CSV/Parquet output with
  pandas differential assertions.
- `DuckPD_Temporal_and_Categorical_Semantics.ipynb` is an interactive tutorial
  demonstrating fixed-duration rolling windows on timestamps, `.dt` floor,
  ceil, round, and timezone conversions, timestamp/duration arithmetic, lazy
  `.cat` metadata accessors, ordered comparisons, and `groupby(observed=False)`
  unused-category expansion.
- `DuckPD_Vector_Search.ipynb` is an interactive GPU tutorial demonstrating
  explicit `TransformersEmbeddingProvider` registration, pinned PyTorch model
  preparation, lazy remote Parquet streaming, in-engine batch embedding via
  `.embed_text()`, persisted sidecar metadata, model-omitting
  `.vector.search_text()`, and parity with an explicit fingerprint assertion.
  Ordinary sidecar-backed frames still require explicit model preparation.
  Catalog-scoped automatic preparation is demonstrated separately in the
  FeatureStore walkthrough.

- `featurestore_demo/DuckPD_FeatureStore_Walkthrough.ipynb` covers remote
  catalog inspection, exact and point-in-time alignment, reference tables,
  Arrow batch streaming, and catalog-inferred semantic search. Its embedding
  workflow inspects the catalog model, plans without side effects, demonstrates
  bounded automatic preparation, compares cold and warm profile metrics, and
  shows explicit prewarming with automatic preparation disabled.

## Generate Feature-Store Data

Generate a small local feature store from the repository root:

```bash
make -C demo/generate_data generate \
  START_DATE=2024-01-01 \
  END_DATE=2025-01-01 \
  TICKER_COUNT=10
```

To build the metadata and preview a Hugging Face upload, configure `HF_TOKEN` and
`HF_DESTINATION` in `demo/generate_data/.env`, then run:

```bash
make -C demo/generate_data data TICKER_COUNT=10 DRY_RUN=true
```

The generated values are synthetic and must not be used for trading or investment
decisions. See [generate_data/README.md](generate_data/README.md) for configuration,
individual commands, upload behavior, and tests.

Run market data benchmarks:

```bash
uv run python demo/market_data_demo.py smoke
uv run python demo/market_data_demo.py 100mb
uv run python demo/market_data_demo.py 1gb
uv run python demo/market_data_demo.py 5gb
uv run python demo/market_data_demo.py all
```

The benchmark defaults to three isolated repetitions per engine and verifies
semantic equality on every repetition. Use `--repetitions 7` for a more stable
local median. Reported memory is peak Python heap traced by `tracemalloc`, not
total process RSS or DuckDB native memory.

Generate individual benchmark files:

```bash
uv run python demo/generate_market_data.py 100mb
uv run python demo/generate_market_data.py 1gb
uv run python demo/generate_market_data.py 5gb
```

Generate all three in one run:

```bash
uv run python demo/generate_market_data.py 100mb 1gb 5gb
```

The preset names use decimal target sizes. Actual Zstandard-compressed size is
estimated from a calibration file and may differ slightly. The 1 GB and 5 GB
presets can take several minutes and require enough free disk space. Existing
files are skipped unless `--force` is supplied.