# DuckPD demos

Run these programs from the repository root after installing the development
environment:

```bash
uv sync --group dev
uv run python demo/basic_pipeline.py
uv run python demo/parquet_pipeline.py
uv run python demo/reduction_pipeline.py
uv run python demo/generate_market_data.py
```

- `basic_pipeline.py` builds a lazy frame from pandas, sets an explicit index,
  filters rows, calculates columns, tracks ordering, projects, previews, and
  collects the result.
- `parquet_pipeline.py` creates a small Parquet input, scans it lazily, displays
  the execution plan, and writes the transformed result directly to Parquet.
- `reduction_pipeline.py` demonstrates eager `count`, `size`, `sum`, `mean`,
  `min`, and `max` execution over a lazy frame. It covers DataFrame
  `numeric_only`, Series null handling, `skipna`, `min_count`, hidden indexes,
  expression reductions, and the session execution counter.
- `generate_market_data.py` calibrates compressed bytes per row, then streams a
  deterministic OHLC time-series dataset directly to Parquet. The safe default
  creates an approximately 5 MB smoke file under `demo/data/`.

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