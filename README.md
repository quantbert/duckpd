> [!WARNING]
> **DuckPD is a work in progress and is not yet recommended for
> production-critical workloads.** The API and supported pandas semantics may
> change between `0.x` releases. Validate results and resource behavior for
> each intended workload before adopting it.

<p align="center">
  <img src="duckpd.png" alt="DuckPD mascot - a duck dressed as a panda" width="280">
</p>

# DuckPD 🦆❤️🐼

**DuckPD is DuckDB dressed as a pandas DataFrame.**

DuckPD is a lazy DataFrame library with a pandas-shaped API and DuckDB as its execution engine. The goal is to make working with DuckDB feel familiar to pandas users while preserving the performance, scalability, and query-optimization advantages of DuckDB.

Where practical, DuckPD aims to match pandas APIs and semantics closely enough that existing pandas knowledge — and eventually a large amount of pandas-oriented code — transfers naturally. It does **not**, however, aim to reproduce pandas by sacrificing the properties that make DuckDB valuable.

## Project directives

These principles define the direction of DuckPD and should guide API and implementation decisions:

1. **Pandas-shaped, DuckDB-native.**
   The public API should feel like pandas, but operations should map naturally onto DuckDB's relational and vectorized execution model.

2. **Stay lazy by default.**
   Transformations should build a query plan rather than execute immediately. Execution should happen only at clear and intentional boundaries such as `collect()`, `head()`, Arrow conversion, or file output.

3. **Never silently fall back to pandas.**
   Unsupported operations should fail explicitly rather than unexpectedly materializing an entire dataset into memory. Users should always be able to reason about where computation happens.

4. **Push work into DuckDB.**
   Filtering, projection, joins, aggregation, sorting, expressions, and other supported operations should be translated into DuckDB operations whenever possible so DuckDB can optimize the complete query.

5. **Preserve pandas semantics where we claim compatibility.**
   API similarity alone is not enough. Supported operations should match pandas behavior as closely as practical, including edge cases around nulls, indexes, dtypes, grouping, and column behavior.

6. **Correctness before coverage.**
   It is better to support a smaller pandas surface correctly than to advertise broad compatibility backed by incomplete semantics, hidden fallbacks, or surprising execution behavior.

7. **Make execution visible and predictable.**
   Users should be able to understand when data is scanned, materialized, transferred, or written. Laziness must be a useful property, not hidden magic.

8. **Exploit the ecosystem boundaries.**
   DuckPD should interoperate cleanly with pandas, Arrow, Parquet, SQL, and DuckDB itself. Crossing those boundaries should be explicit and inexpensive wherever the underlying systems allow it.

The long-term ambition is broad pandas API coverage **where those APIs can be implemented without violating these directives**. Compatibility is the interface; DuckDB-native execution is the foundation.

## Current capabilities

- Lazy Parquet, CSV, pandas, Arrow, DuckDB table, and read-only SQL sources.
- Column selection, boolean filtering, arithmetic expressions, `assign`,
  `sort_values`, `limit`, and distinct/drop_duplicates deduplication.
- Relational DataFrame joins (`merge`, `join`) supporting `inner`, `left`, `right`,
  `outer`, and `cross` with suffix collision handling and cardinality validation
  (`validate="1:1"`, `"1:m"`, `"m:1"`, `"m:m"`).
- Multi-DataFrame row-wise concatenation (`duckpd.concat`) with schema reconciliation,
  null-padding, lossless numeric coercion, exact nullable integer preservation,
  and stable sequence order synthesis.
- Vectorized `.str` (e.g. `upper`, `lower`, `strip`, `len`, `contains`, `replace`)
  and `.dt` (e.g. `year`, `month`, `day`, `hour`, `minute`, `second`, `strftime`,
  `to_period`) accessor pipelines.
- Multi-column `groupby()` supporting eager and lazy `agg()`, `sum()`, `mean()`,
  `min()`, `max()`, `std()`, `var()`, and `count()`.
- Eager DataFrame and Series reductions: `count`, `size`, `sum`, `mean`, `min`,
  `max`, `std`, `var`, `median`, `quantile`, `any`, and `all` over numeric and
  boolean data, including `skipna`, `min_count`, and DataFrame `numeric_only` support.
- Explicit lazy indexes with `set_index()`/`reset_index()` and source
  `index=`/`order_by=` declarations, MultiIndex exact/prefix matching, and
  honest ordered label-list `.loc[[...]]` selection.
- Positional row slicing via `df.iloc[start:stop]`.
- Stable snapshot order for pandas and Arrow inputs, synthesized order for
  ordered concatenation, and metadata-preserving `persist()`.
- Context-local implicit sessions, allowing frames created by separate
  module-level helpers to participate in the same lazy plan.
- Explicit execution boundaries: pandas collection (`collect`, `to_pandas`),
  bounded `head`, Arrow tables and streaming record batches (`to_arrow_batches`),
  plan inspection (`explain`, `explain_write`), and direct zero-copy Parquet
  (`write_parquet`) and CSV (`write_csv`, `to_csv`) writes.
## Supported pandas API Coverage

DuckPD maps pandas semantics directly to DuckDB's vectorized analytical engine:

| API Category | Supported Methods & Operations | Execution Model |
| :--- | :--- | :--- |
| **I/O & Data Loading** | `read_parquet()`, `read_csv()`, `read_sql()`, `from_pandas()`, `from_arrow()`, `sql()`, `connect()` | **Lazy** (scans metadata / registers source) |
| **Transformations & Projections** | `df[cols]`, `df[bool_filter]`, `assign()`, `sort_values()`, `limit()`, `drop_duplicates()`, `set_index()`, `reset_index()`, `df.loc[]`, `df.iloc[]` | **Lazy** (appends to logical query graph) |
| **Joins & Merges** | `merge()`, `join()` (`inner`, `left`, `right`, `outer`, `cross`, custom suffixes, `validate=`) | **Lazy** (relational hash join, pre-flight cardinality check) |
| **Concatenation** | `duckpd.concat()` (multi-frame row union, schema alignment, null padding, lossless numeric coercion, stable order synthesis) | **Lazy** (union with projection padding) |
| **String Accessor (`.str`)** | `upper()`, `lower()`, `strip()`, `len()`, `startswith()`, `endswith()`, `contains()`, `replace()` | **Lazy** (DuckDB SQL functions) |
| **Datetime Accessor (`.dt`)** | `year`, `month`, `day`, `hour`, `minute`, `second`, `strftime()`, `to_period()` | **Lazy** (DuckDB timestamp extractors) |
| **GroupBy Aggregations** | `groupby().agg()`, `.sum()`, `.mean()`, `.min()`, `.max()`, `.std()`, `.var()`, `.count()` (`as_index=True/False`) | **Lazy** for `.agg()`, **Eager** for reductions |
| **Statistical Reductions** | `sum()`, `mean()`, `min()`, `max()`, `count()`, `size`, `std()`, `var()`, `median()`, `quantile()`, `any()`, `all()`, `nunique()` | **Eager** (single aggregate SQL pushdown) |
| **Collection, Output & State** | `collect()`, `to_pandas()`, `head(n)`, `explain()`, `explain_write()`, `write_parquet()`, `write_csv()`, `to_csv()`, `to_arrow_table()`, `to_arrow_batches()`, `persist()` | **Explicit Execution Boundary** |

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

## Ordering, indexing, and sessions

Pandas and Arrow inputs are snapshots with a stable source row order. DuckPD
tracks that order with hidden relational metadata so operations such as
`.iloc`, `drop_duplicates(keep=...)`, `rank(method="first")`, and top-N tie
selection remain deterministic without exposing a synthetic pandas index.

Parquet, CSV, SQL, and DuckDB table scans remain unordered unless `order_by=`
is provided. Ordering-sensitive operations fail with
`UnorderedOperationError` rather than relying on accidental scan order.

Row-wise `concat` preserves input sequence and each input's guaranteed order
when every input is ordered; one unordered input makes the result unordered.
Persistence retains explicit indexes and ordering metadata. SQL joins never
claim a total order because duplicate join keys lack a stable relational
tie-breaker, even when `merge(sort=True)` sorts by the merge keys; follow-up
positional work must sort by enough columns to break ties.

Label selections remain lazy and therefore return DuckPD `DataFrame` or
`Series` handles. Exact pandas return-type switching for `df.loc[label]`
depends on runtime index uniqueness and is intentionally deferred to a bounded
eager scalar/row API. MultiIndex exact and prefix keys and honest ordered
label-list selections are supported (preserving duplicates and requested order,
raising KeyError on missing labels); cross-frame assignment alignment remains
unsupported.

Module-level readers reuse a context-local implicit session, so independently
created helper frames can be combined. Explicit `Session` context managers are
still recommended when resource limits, database lifetime, or deterministic
cleanup matter.

## Demos

Interactive notebooks and small runnable programs are available in [demo/](demo/README.md):

- `demo/DuckPD_Quickstart.ipynb` — 5-minute quickstart on the Goodreads Books dataset.
- `demo/DuckPD_Features_Walkthrough.ipynb` — Deep dive into recent additions (remote cloud parquet, multi-table joins, `.str`/`.dt` accessors, `duckpd.concat`, statistical reductions, and multi-column groupbys) using the AlphaDojo stock news dataset (~3.9M rows).
- `demo/DuckPD_Order_Index_Window_Workflows.ipynb` — Offline, differential
  walkthrough of stable row order, deterministic ties, MultiIndex `.loc`, 2D
  `.iloc`, cumulative/rolling/expanding windows, masked assignment,
  persistence, and direct outputs.

```bash
uv run python demo/basic_pipeline.py
uv run python demo/parquet_pipeline.py
uv run python demo/reduction_pipeline.py
uv run python demo/generate_market_data.py
uv run python demo/market_data_demo.py smoke
```

Full multi-gigabyte benchmarks can be run across `100mb`, `1gb`, `5gb`, or `all`:

```bash
uv run python demo/market_data_demo.py all
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