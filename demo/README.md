# DuckPD demos

The demo tree separates interactive tutorials from executable data and benchmark
utilities:

```text
demo/
├── notebooks/       # Primary user-facing walkthroughs
├── benchmarks/      # Reproducible market-data generator and benchmark
├── generate_data/   # Feature-store dataset production tools
└── .artifacts/      # Ignored notebook outputs, created on demand
```

Run commands from the repository root after installing the development
environment:

```bash
uv sync --frozen --group dev
```

## Notebooks

Notebooks are the canonical tutorials. Generated outputs are not checked in;
notebooks write reusable files under `demo/.artifacts/`.

| Notebook | Scope | Runtime |
| --- | --- | --- |
| [`quickstart.ipynb`](notebooks/quickstart.ipynb) | Lazy loading, transformations, reductions, grouping, plans, and export | Standard development environment |
| [`feature_walkthrough.ipynb`](notebooks/feature_walkthrough.ipynb) | Remote Parquet, joins, concatenation, accessors, reductions, windows, and persistence | Network access |
| [`order_index_window_workflows.ipynb`](notebooks/order_index_window_workflows.ipynb) | Stable order, indexing, window semantics, assignment, and persistence | Offline |
| [`temporal_categorical_semantics.ipynb`](notebooks/temporal_categorical_semantics.ipynb) | Temporal rolling, datetime arithmetic, categoricals, and Arrow interchange | Offline |
| [`time_series_embeddings.ipynb`](notebooks/time_series_embeddings.ipynb) | Native multi-channel series representations and exact retrieval | Offline |
| [`moment_time_series_embeddings.ipynb`](notebooks/moment_time_series_embeddings.ipynb) | Built-in MOMENT provider, bounded GPU inference, persistence, and exact retrieval | Network access and PyTorch GPU environment |
| [`transformers_series_embeddings.ipynb`](notebooks/transformers_series_embeddings.ipynb) | Pinned bare PatchTSMixer inference, series-query parity, and rich input contracts | Network access; CPU or PyTorch GPU environment |
| [`event_windows_exact_fusion.ipynb`](notebooks/event_windows_exact_fusion.ipynb) | Event grids, availability, reaction retrieval, and exact late fusion | Offline |
| [`vector_search.ipynb`](notebooks/vector_search.ipynb) | GPU text embedding, persisted model identity, and semantic search | Network access and PyTorch GPU environment |
| [`feature_store.ipynb`](notebooks/feature_store.ipynb) | Catalog inspection, point-in-time alignment, caching, series search, and semantic search | Network access; GPU environment for embedding sections |

The GPU notebooks use the **DuckPD ROCm 7.2.4** kernel configured in
[`generate_data/README.md`](generate_data/README.md#amd-rocm-generation). The
normal development environment intentionally excludes accelerator-specific
PyTorch and Transformers packages.

To use JupyterLab, install it into the selected environment and launch the
notebook directory:

```bash
uv pip install jupyterlab
uv run python -m jupyter lab demo/notebooks
```

## Market-data benchmark

The benchmark utilities remain Python programs because they provide reusable
CLI entry points, isolated subprocess measurement, and testable data generation
rather than duplicating a tutorial.

Generate the approximately 5 MB smoke dataset and run one validated comparison:

```bash
uv run python demo/benchmarks/generate_market_data.py smoke
uv run python demo/benchmarks/market_data.py smoke --repetitions 1
```

Generated files live under ignored `demo/benchmarks/data/`. Larger presets are
available for throughput and memory exercises:

```bash
uv run python demo/benchmarks/generate_market_data.py 100mb 1gb 5gb
uv run python demo/benchmarks/market_data.py all
```

The benchmark validates DuckPD and pandas result equality on every repetition.
Reported memory is peak Python heap traced by `tracemalloc`, not total process
RSS or DuckDB native memory.

## Feature-store data generation

`generate_data/` contains the deterministic Nasdaq Stockholm feature-store
producer. It writes UTC-day-partitioned OHLCV, moving-average, and news datasets,
reference tables, catalog metadata, and a Hugging Face dataset card.

Generate a bounded local store:

```bash
make -C demo/generate_data generate \
  START_DATE=2024-01-01 \
  END_DATE=2025-01-01 \
  TICKER_COUNT=10
```

Build metadata and preview an upload after configuring
`demo/generate_data/.env`:

```bash
make -C demo/generate_data data TICKER_COUNT=10 DRY_RUN=true
```

See [`generate_data/README.md`](generate_data/README.md) for source provenance,
GPU generation, individual commands, upload behavior, and focused tests.

Synthetic market and feature-store values are demonstration data only and must
not be used for trading or investment decisions.