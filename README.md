# duckpd

DuckPD is an experimental lazy DataFrame library with a pandas-shaped frontend
and DuckDB as its execution engine.

The project is in early development. It intentionally supports a small,
explicit subset of pandas rather than silently falling back to materializing a
complete pandas DataFrame.

## Current capabilities

- Lazy pandas, Arrow, Parquet, DuckDB table, and read-only SQL sources.
- Column selection, boolean filtering, arithmetic expressions, `assign`,
  `sort_values`, and `limit`.
- Eager DataFrame and Series `count`, `size`, `sum`, `mean`, `min`, and `max`
	reductions over numeric and boolean data, including `skipna`, `min_count`,
	and DataFrame `numeric_only` support.
- Explicit lazy indexes with `set_index()`/`reset_index()` and source
	`index=`/`order_by=` declarations.
- Explicit pandas collection, bounded `head`, Arrow tables and record batches,
  physical plan inspection, and direct Parquet writes.
- Session-level memory, spill-directory, temporary-size, and thread settings.
- Rejection of ambiguous cross-frame alignment and mutating SQL.

## Example

```python
import duckpd as pd

orders = pd.read_parquet("orders/*.parquet")

result = (
	orders[orders["status"] == "paid"]
	.assign(net=lambda frame: frame["amount"] - frame["refund_amount"])
	.sort_values("net", ascending=False)
	[["order_id", "net"]]
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

Small runnable programs are available in [demo/](demo/README.md):

```bash
uv run python demo/basic_pipeline.py
uv run python demo/parquet_pipeline.py
uv run python demo/reduction_pipeline.py
uv run python demo/generate_market_data.py
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run pyright
uv build
```

See [todo.md](todo.md) for the implementation roadmap and compatibility scope.