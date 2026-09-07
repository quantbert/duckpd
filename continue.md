# Continue: News Dataset Generation

## Goal

Continue with the complete feature-store demo dataset in `demo/generate_data`.
Separately, explore possible AMD ROCm acceleration later. GPU support needs deeper
research before choosing an implementation or dependency strategy.

## Future GPU Research

This machine exposes an AMD GPU through ROCm:

```text
Card Series: AMD Radeon Graphics
GFX version: gfx1150
Reported VRAM: 4 GiB
```

`rocm-smi` works. `nvidia-smi` is irrelevant on this host.

The current `duckpd[embeddings]` environment reports these ONNX Runtime providers:

```text
['AzureExecutionProvider', 'CPUExecutionProvider']
```

The installed FastEmbed/ONNX Runtime stack currently uses CPU. This does not establish
whether FastEmbed can or cannot run efficiently on this AMD GPU with a different
runtime or integration.

Explore GPU support later with a deeper search of current FastEmbed, ONNX Runtime,
ROCm, MIGraphX, and `gfx1150` documentation and compatibility. Do not presume a
provider name, package, or architecture before that research. Keep CPU behavior as the
default if GPU support is eventually added.

## Current Implementation

The full news-generation pipeline is implemented and validated. It:

- downloads and verifies pinned `AlphaDojo/dojo_stock_news` source data;
- maps all 3,951,636 rows deterministically onto synthetic XSTO ticker/minute keys;
- embeds title/description text with `BAAI/bge-small-en-v1.5` (384 dimensions);
- writes resumable staging chunks and monthly final Parquet partitions;
- writes `_SUCCESS.json` only after successful finalization;
- adds model and news metadata to catalog version 1;
- rejects incomplete news output during catalog generation;
- uploads the complete feature store only after generation.

The verified source already exists at:

```text
demo/generate_data/source_data/dojo_stock_news.parquet
```

The generated `demo/generate_data/data/` directory does not yet exist.

Current uncommitted work was last observed as:

```text
 M .gitignore
 M demo/README.md
 M docs/design/featurestore-embeddings.md
 M pyproject.toml
 M uv.lock
?? demo/generate_data/
```

Do not discard these changes. Re-read current files before editing.

## GPU Research Anchors

The controlling implementation is `src/duckpd/embeddings.py`.
`FastEmbedProvider` currently:

- describes itself as CPU-only;
- passes `providers=["CPUExecutionProvider"]` in both `TextEmbedding(...)` paths;
- reports `("CPUExecutionProvider",)` in `PreparedModelInfo`.

`demo/generate_data/gendata.py` creates `pd.FastEmbedProvider(NEWS_MODEL)` without
provider configuration. `tests/test_embeddings.py` currently asserts CPU-only
behavior.

Tomorrow's research should establish:

1. Whether current FastEmbed supports AMD ROCm directly or through another backend.
2. Which runtime and execution provider support this host's ROCm version and `gfx1150`.
3. Whether suitable prebuilt Python wheels exist for the project's Python version.
4. Whether 4 GiB reported VRAM can run `BAAI/bge-small-en-v1.5` at a useful batch size.
5. What minimal DuckPD API would expose acceleration without changing CPU defaults.

Only after those facts are verified should code or dependency changes be designed.

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

If GPU support is later proven, determine a stable `EMBEDDING_BATCH_SIZE` with small
experiments before running the complete archive.

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

Last known validation:

- Ruff passed.
- Generator tests: 14 passed.
- CPU FastEmbed smoke test returned fixed-size `float[384]` vectors.
- Source checksum and all required source fields were verified.
- Production generator files had no editor diagnostics.

PyArrow stub `reportUnknown*` diagnostics remain in the generator tests and are
third-party typing limitations, not runtime failures.

## Full Run Safety

A plain `make data` performs a real upload because `DRY_RUN=false`. The configured
private destination was `hf://datasets/hifinab/fdb`; never print or copy `HF_TOKEN`.
Generate and validate locally first:

```bash
cd demo/generate_data
make data DRY_RUN=true
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
