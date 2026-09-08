# Sample Data Generation

This directory contains tools that generate sample data. The tools generate
deterministic Nasdaq Stockholm minute data. The tools calculate simple moving averages.
The tools build feature-store metadata. The tools upload the complete store to Hugging
Face.

The generated values are not real trades or prices. Do not use these values for live
trading or investment decisions.

## End-To-End Run

Set `HF_TOKEN` and `HF_DESTINATION` in the ignored `.env` file. Then, go to this
directory and run the pipeline:

```bash
cd demo/generate_data
make data
```

The `data` target runs the `generate` and `upload` targets. They do these steps:

1. Generates yearly OHLCV Parquet partitions.
2. Generates SMA10, SMA20, SMA50, and SMA200 partitions.
3. Generates monthly news partitions with BGE embeddings.
4. Builds catalog version 1, dataset metadata, and the Hugging Face dataset card.
5. Creates or updates the configured private Hugging Face bucket or dataset.

By default, generation does not replace existing yearly files. It generates the same
data for the same date range, ticker range, and seed.

## Configuration

The Makefile passes the ignored `.env` file to commands that need Hugging Face
configuration. Set these values there:

| Setting | Purpose |
| --- | --- |
| `HF_DESTINATION` | Target `hf://buckets/OWNER/NAME` or `hf://datasets/OWNER/NAME` URI. |
| `HF_TOKEN` | Hugging Face token with write access. Never commit this value. |

Generation and run controls are Make variables with defaults:

| Setting | Purpose |
| --- | --- |
| `DATA_DIRECTORY` | Generated store directory, relative to this directory. |
| `START_DATE` | Inclusive ISO 8601 coverage start date. |
| `END_DATE` | Exclusive ISO 8601 coverage end date. |
| `TICKER_START` | First numeric ticker identifier. |
| `TICKER_COUNT` | Number of sequential tickers to generate. |
| `SEED` | Seed used for deterministic generation. |
| `OVERWRITE` | Controls replacement of existing yearly partitions. |
| `DRY_RUN` | Prevents the Hugging Face upload when set to `true`. |
| `NEWS_SOURCE` | Local pinned AlphaDojo news Parquet source. |
| `EMBEDDING_BATCH_SIZE` | Number of articles embedded per model call. |
| `EMBEDDING_BACKEND` | `fastembed` for default CPU inference or `transformers` for PyTorch. |
| `EMBEDDING_DEVICE` | PyTorch device; `cpu` by default and `cuda` for NVIDIA CUDA or AMD ROCm. |
| `TRANSFORMER_BATCH_SIZE` | Internal PyTorch inference batch size; defaults to the qualified value 64. |
| `NEWS_UV_RUN` | Override the `uv run` prefix when using a separately prepared accelerator environment. |

Override settings on the command line. For example, this generates ten tickers and
previews the upload without writing to the remote store:

```bash
make data TICKER_COUNT=10 DRY_RUN=true
```

Use `make generate OVERWRITE=true` to replace existing partitions. Use `make upload`
to rebuild metadata and upload an already generated store.

## Embedded News Stress Data

The ignored `source_data/dojo_stock_news.parquet` file is the pinned source for a
large derived stress dataset. Generate only the news family with:

```bash
make generate-news
```

The target downloads and verifies the pinned source when it is not already present.
News rows are assigned evenly across the configured synthetic tickers and XSTO
trading minutes. The original symbol and publication date are retained as provenance;
their relationship to the generated ticker and timestamp is intentionally artificial.

The generator embeds all 3,951,636 article titles and descriptions with
`BAAI/bge-small-en-v1.5`. The vectors alone require about 5.65 GiB before Parquet
encoding, and staging plus final output requires additional free disk space. The
operation can take a long time on CPU. Work is staged in bounded, restartable chunks.

Catalog-driven model inference is part of the feature-store embedding design and must
be implemented in the DuckPD runtime before `search_text()` can omit its explicit
`model=` argument. Generation and raw-vector search do not depend on that inference.

### AMD ROCm generation

GPU generation is explicit. The normal `duckpd[embeddings]` installation and
`make generate-news` remain CPU-only. DuckPD does not install or replace
accelerator-specific PyTorch builds.

The following isolated environment matches ROCm 7.2.4 and Python 3.12 on
`gfx1150`. Create it from this directory:

```bash
uv venv --python 3.12 .venv-rocm
uv pip install --python .venv-rocm/bin/python \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torch-2.9.1%2Brocm7.2.4.lw.git39497456-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/triton-3.5.1%2Brocm7.2.4.gita272dfa8-cp312-cp312-linux_x86_64.whl' \
  'transformers>=4.50,<5' \
  'exchange-calendars>=4.11,<5' \
  'ipykernel>=7,<8' \
  'ipywidgets>=8,<9' \
  -e ../..
```

Register that interpreter as a Jupyter kernel:

```bash
.venv-rocm/bin/python -m ipykernel install --user \
  --name duckpd-rocm \
  --display-name "DuckPD ROCm 7.2.4"
```

In VS Code, open `DuckPD_Vector_Search.ipynb`, choose **Select Kernel** in the
upper-right, and select **DuckPD ROCm 7.2.4**. Restart the notebook kernel after
changing it. The repository's normal `.venv` does not contain the
accelerator-specific PyTorch and Transformers packages.

Verify that PyTorch sees the GPU before generation:

```bash
.venv-rocm/bin/python -c \
  "import torch; print(torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Generate with the verified Transformers/ROCm path:

```bash
make generate-news \
  NEWS_UV_RUN='UV_PROJECT_ENVIRONMENT=.venv-rocm uv run --no-sync' \
  EMBEDDING_BACKEND=transformers \
  EMBEDDING_DEVICE=cuda \
  TRANSFORMER_BATCH_SIZE=64
```

The model backend is part of the embedding fingerprint and staging identity.
CPU FastEmbed chunks and PyTorch Transformers chunks cannot be mixed or resumed
into each other. Use a fresh output/staging directory when changing backends.
An explicitly requested GPU fails preparation instead of silently falling back
to CPU.

News rows are assigned round-robin to synthetic tickers and spread over the same
Nasdaq Stockholm trading-minute range as OHLCV. These ticker and timestamp
associations are artificial. Original symbols and publication-date text are retained
for provenance.

## Individual Commands

Use this command to generate a small local test store. The store contains UTC OHLCV and
SMA time series. It also contains a keyed symbology table and a markets table without a
key.

```bash
uv run --group generation python gendata.py \
	--output data/ohlcv \
	--start 2024-01-01 \
	--end 2025-01-01 \
	--ticker-count 10 \
	--generate-sma \
	--sma-output data/sma \
	--symbology-output data/symbols/data.parquet \
	--markets-output data/markets/data.parquet
```

Use `--overwrite` to replace partitions that an older version generated. Catalog
version 1 accepts only Parquet time columns with the Arrow timezone `UTC`.

Use this command to build catalog metadata and read Parquet partition statistics:

```bash
uv run python build_catalog.py --data data \
	--store-name OWNER/NAME \
	--source hf://buckets/OWNER/NAME
```

This command writes catalog version 1 to `catalog.json`. It writes `metadata.json` for
each dataset. It also writes the generated store `README.md` in the configured data
directory. The `DATASETS` definition in `build_catalog.py` is the source for dataset and
feature semantics. Time-series datasets declare `time_column` and `series_keys`. Table
datasets can declare a composite `primary_key`. They can also omit `primary_key`.

Use this command to examine an upload without a remote write:

```bash
uv run --env-file .env python hfupload.py \
	--data data \
	--destination hf://buckets/OWNER/NAME \
	--dry-run
```

The uploader supports Hugging Face buckets and dataset repositories. The uploader
rebuilds the catalog and the generated dataset card before it transfers files.

## Library Demo

Run the remote cache and query example from this directory:

```bash
uv run --env-file .env python demo.py
```

The example reads `HF_DESTINATION` and `HF_TOKEN` from the environment. It stores
downloaded fragments in `demo/feature_cache`. The launch directory does not change
the cache location.

## Testing

Run the generation tool tests from this directory:

```bash
uv run --group dev --group generation python -m pytest -q test_generate_data.py --no-cov
```
