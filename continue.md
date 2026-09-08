# Continue: News Dataset Generation

## Goal

Continue with the complete feature-store demo dataset in `demo/generate_data`.
GPU support is implemented and verified; use the explicit ROCm path below for
news generation while keeping FastEmbed CPU as the default.

## GPU Implementation

DuckPD now exposes `TransformersEmbeddingProvider` for explicit PyTorch CPU,
NVIDIA CUDA, or AMD ROCm inference. The provider:

- requires an embedding specification with `backend="transformers"` and
  explicit `cls` or `mean` pooling;
- keeps model revision, pooling, prefixes, normalization, and backend in the
  model fingerprint;
- uses immutable, digest-verified local model caches;
- performs bounded internal tokenizer/model batches;
- reports `PyTorchCPU`, `PyTorchCUDA`, or `PyTorchROCm`;
- raises when a requested GPU is unavailable instead of silently using CPU.

The host-qualified environment uses ROCm 7.2.4 PyTorch 2.9.1 and Triton 3.5.1
from AMD's `gfx1150` wheels. Reproduce it with the commands in
`demo/generate_data/README.md`; accelerator packages remain application-owned
and are not added to DuckPD's portable dependency lock.

FastEmbed remains the default CPU provider. The generator selects GPU execution
only when passed `--embedding-backend transformers --embedding-device cuda`, or
the equivalent Make variables.

## Current Implementation

The full news-generation pipeline is implemented and validated. It:

- downloads and verifies pinned `AlphaDojo/dojo_stock_news` source data;
- maps all 3,951,636 rows deterministically onto synthetic XSTO ticker/minute keys;
- embeds title/description text with `BAAI/bge-small-en-v1.5` (384 dimensions);
- writes resumable staging chunks and monthly final Parquet partitions;
- writes `_SUCCESS.json` only after successful finalization;
- selects either the default FastEmbed CPU model or the distinct Transformers
  GPU model and records that model fingerprint in completion/catalog metadata;
- adds model and news metadata to catalog version 1;
- rejects incomplete news output during catalog generation;
- uploads the complete feature store only after generation.

The verified source already exists at:

```text
demo/generate_data/source_data/dojo_stock_news.parquet
```

The generated `demo/generate_data/data/` directory does not yet exist.

The working tree also contains the user's broader in-progress dataset and
feature-store changes. Do not discard them; re-read current files before editing.

## GPU Qualification

The controlling implementation is `TransformersEmbeddingProvider` in
`src/duckpd/embeddings.py`. The generator exposes:

```text
EMBEDDING_BACKEND=transformers
EMBEDDING_DEVICE=cuda
TRANSFORMER_BATCH_SIZE=64
```

On this AMD `gfx1150` host, the implemented provider produced normalized
`float32[384]` vectors through `PyTorchROCm`. CPU/GPU parity over the same pinned
`BAAI/bge-small-en-v1.5` Transformers model measured maximum absolute error
`1.7881393432617188e-07` and minimum cosine similarity
`0.9999999783689986`. An earlier 64-row GPU benchmark processed about 138 rows/s
at roughly 294 MiB peak allocated device memory, versus about 15 rows/s through
the existing CPU FastEmbed path.

FastEmbed through MIGraphX is not used: its optimized graph rejected the
attention padding required by this model, and the unfused attempt produced
invalid zero output. PyTorch Transformers is the verified correctness path.

## Default Dataset Size

The Makefile covers `[2010-01-01, 2025-01-01)` for 100 tickers (`000`-`099`):

| Family | Rows | Expected Parquet files |
| --- | ---: | ---: |
| OHLCV | 192,270,000 | 15 yearly |
| SMA | 192,270,000 | 15 yearly |
| News | 3,951,636 | about 180 monthly |
| Symbols | 100 | 1 |
| Markets | 10 | 1 |

There are 192,270,000 unique ticker/minute slots, so news assignment capacity is safe.
News vectors alone require about 5.65 GiB before Parquet encoding. Staging and final
news files coexist until completion. Approximately 1.42 TiB of disk was free when last
checked.

Use `TRANSFORMER_BATCH_SIZE=64` for the qualified GPU run. The outer
`EMBEDDING_BATCH_SIZE=1024` still controls DuckPD's Arrow execution batches.

## Validation

Use `uv` for all Python commands.

```bash
uv run --group dev ruff check src/duckpd/embeddings.py tests/test_embeddings.py \
  demo/generate_data pyproject.toml
uv run --group dev python -m pytest -q tests/test_embeddings.py --no-cov
uv run --group dev --group generation python -m pytest -q \
  demo/generate_data/test_generate_data.py --no-cov
git diff --check
make -n -C demo/generate_data data DRY_RUN=true
```

Last validation for GPU support:

- focused embedding tests: 19 passed;
- generator tests: 14 passed;
- Ruff and format checks passed for every changed Python file;
- strict Pyright passed for changed library/embedding test files;
- default and ROCm Make commands expanded with the expected backend/device
  arguments;
- the real provider executed on AMD ROCm and matched its CPU reference at the
  parity values recorded above;
- generated compatibility documentation and `git diff --check` passed.

The full-project Pyright command still reports pre-existing PyArrow stub and
untyped `exchange_calendars` diagnostics in generator files.

## Full Run Safety

A plain `make data` performs a real upload because `DRY_RUN=false`. The configured
private destination was `hf://datasets/hifinab/fdb`; never print or copy `HF_TOKEN`.
Generate and validate locally first:

```bash
cd demo/generate_data
make data DRY_RUN=true \
  NEWS_UV_RUN='UV_PROJECT_ENVIRONMENT=.venv-rocm uv run --no-sync' \
  EMBEDDING_BACKEND=transformers \
  EMBEDDING_DEVICE=cuda \
  TRANSFORMER_BATCH_SIZE=64
```

After inspecting the local store and `data/news/_SUCCESS.json`, publish separately:

```bash
make upload
```

Do not use `make -j data`: generation and upload are sibling prerequisites and may
race under parallel Make. News staging is resumable with identical settings.
`OVERWRITE=true` deletes news output and staging, so avoid it when resuming.

Run the full generation in detached `tmux` with an unbuffered persistent log. Monitor
`rocm-smi`, disk usage, batch throughput, and the
`Embedded X/3,951,636 news rows` messages.
