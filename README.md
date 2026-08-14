# duckpd

<p align="center">
  <img src="duckpd.png" alt="DuckPD mascot - a duck dressed as a panda" width="280">
</p>

DuckPD is an experimental lazy DataFrame library with a pandas-shaped frontend
and DuckDB as its execution engine.

> [!WARNING]
> **DuckPD is a work in progress and is not yet recommended for
> production-critical workloads.** The API and supported pandas semantics may
> change between `0.x` releases, and many pandas operations are intentionally
> unsupported. Validate results and resource behavior for each intended
> workload before adopting it.

DuckPD intentionally supports a small, explicit subset of pandas rather than
silently falling back to materializing a complete pandas DataFrame. See the
[release policy](docs/RELEASES.md) for the pre-`1.0` stability policy.

## Current capabilities

- Lazy pandas, Arrow, Parquet, DuckDB table, and read-only SQL sources.
- Column selection, boolean filtering, arithmetic expressions, `assign`,
  `sort_values`, `limit`, and distinct/drop_duplicates deduplication.
- Relational DataFrame joins (`merge`) supporting `inner`, `left`, `right`,
  `outer`, and `cross` with column collision suffix management.
- Multi-DataFrame row-wise concatenation (`duckpd.concat`) with schema alignment
  and null-padding.
- Vectorized `.str` (e.g. `upper`, `lower`, `strip`, `len`, `contains`, `replace`)
  and `.dt` (e.g. `year`, `month`, `day`, `hour`, `minute`, `second`, `strftime`,
  `to_period`) accessor pipelines.
- Multi-column `groupby()` supporting eager and lazy `agg()`, `sum()`, `mean()`,
  `min()`, `max()`, `std()`, `var()`, and `count()`.
- Eager DataFrame and Series reductions: `count`, `size`, `sum`, `mean`, `min`,
  `max`, `std`, `var`, `median`, `quantile`, `any`, and `all` over numeric and
  boolean data, including `skipna`, `min_count`, and DataFrame `numeric_only` support.
- Explicit lazy indexes with `set_index()`/`reset_index()` and source
	`index=`/`order_by=` declarations.
- Explicit pandas collection, bounded `head`, Arrow tables and record batches,
  physical plan inspection (`explain`), and direct zero-copy Parquet writes.
- Session-level memory, spill-directory, temporary-size, and thread settings.
- Rejection of ambiguous cross-frame alignment and mutating SQL.

## Example

```python
import duckpd as pd

orders = pd.read_parquet("orders/*.parquet")

result = (
    orders[orders["status"] == "paid"]
    .assign(net=lambda frame: frame["amount"] - frame["refund_amount"])
    .sort_values("net", ascending=False)[["order_id", "net"]]
    .limit(100)
)

print(result.explain())
preview = result.head(10)
result.write_parquet("largest-paid-orders.parquet")
pandas_result = result.collect()
```

Transformations above are lazy. `explain()`, `head()`, `collect()`, Arrow output,
and file output are explicit execution boundaries. `limit()` stays lazy while
`head()` returns a bounded pandas preview.

## Demos

Interactive notebooks and small runnable programs are available in [demo/](demo/README.md):

- `demo/DuckPD_Quickstart.ipynb` — 5-minute quickstart on the Goodreads Books dataset.
- `demo/DuckPD_Features_Walkthrough.ipynb` — Deep dive into recent additions (remote cloud parquet, multi-table joins, `.str`/`.dt` accessors, `duckpd.concat`, statistical reductions, and multi-column groupbys) using the AlphaDojo stock news dataset (~3.9M rows).

```bash
uv run python demo/basic_pipeline.py
uv run python demo/parquet_pipeline.py
uv run python demo/reduction_pipeline.py
uv run python demo/generate_market_data.py
uv run python demo/market_data_demo.py
```

See the [benchmark results](docs/BENCHMARK.md) for performance and memory
comparisons between DuckPD and pandas across 100 MB, 1 GB, and 5 GB datasets.

## Development

```bash
uv sync --frozen --group dev
make check
make build
```

GNU Make is optional. The equivalent commands are:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv build
```

See the [documentation index](docs/README.md) for the implementation roadmap,
architecture decisions, benchmarks, research, and changelog.