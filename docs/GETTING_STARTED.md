# Getting started

DuckPD is a lazy, pandas-shaped DataFrame library backed by DuckDB. This guide
runs the supported acceptance workflow and makes each execution boundary
explicit.

## Install

```bash
python -m pip install "duckpd==0.1.0a1"
```

DuckPD requires Python 3.11 or newer. Pre-1.0 releases target evaluation on
Linux; they are not production-stability releases.

## Build one lazy query

```python
import duckpd as pd

orders = pd.read_parquet(
    "orders/*.parquet",
    index="order_id",
    order_by=["created_at", "order_id"],
)

monthly = (
    orders[orders["status"] == "paid"]
    .assign(
        month=lambda frame: frame["created_at"].dt.to_period("M"),
        net_amount=lambda frame: frame["amount"] - frame["refund_amount"],
    )
    .groupby(["month", "customer_id"], as_index=False)
    .agg(
        revenue=("net_amount", "sum"),
        order_count=("order_id", "size"),
    )
    .sort_values(["month", "revenue"], ascending=[True, False])
)
```

The reader inspects source schema, and every transformation builds an immutable
logical plan. No source rows are returned while constructing `monthly`.

## Inspect, preview, collect, or write

```python
print(monthly.explain("optimized"))  # plan inspection; no result collection
preview = monthly.head(20)  # executes with a bounded LIMIT
result = monthly.collect()  # executes and returns pandas.DataFrame
monthly.write_parquet("monthly.parquet")  # direct DuckDB sink; no pandas frame
```

`limit(20)` differs from `head(20)`: `limit` remains lazy and returns a DuckPD
DataFrame, while `head` is an eager bounded preview returning pandas.
`to_arrow()`, `to_arrow_batches()`, `profile()`, table/file sinks, `persist()`,
and `commit()` are also explicit execution or materialization boundaries.

For a deliberately small result, `collect_small(max_bytes)` accepts only
non-expanding local-Parquet plans with fixed-width output types. It proves a
conservative row/type memory upper bound before execution, checks measured
pandas memory afterward, and exposes both through
`session.last_materialization_report`. Joins, unions, Python UDFs, and
variable-width output reject before any scan. Arbitrary Python work is available only
through `session.register_arrow_udf(...)` plus `series.map_arrow(name)`, with
declared DuckDB types and an explicit batch-independence contract. Explain and
profile output report each Arrow UDF boundary. DuckPD never invokes DuckDB
relation `map()` as a hidden fallback.

## Ordering and indexes

Parquet, CSV, SQL, and table scans have no stable row order unless `order_by=`
is declared. Positional, cumulative, ranking, and rolling operations reject an
unordered plan with `UnorderedOperationError`. A later join clears total-order
guarantees; call `sort_values()` with deterministic tie-breakers before another
order-dependent operation.

Cross-frame alignment requires compatible explicit indexes. DuckPD rejects
implicit positional alignment instead of guessing from scan order.

## Resource-bounded sessions

```python
import duckpd as pd

with pd.connect(
    memory_limit="2GB",
    temp_directory="/tmp/duckpd-spill",
    max_temp_directory_size="20GB",
    threads=4,
) as session:
    frame = pd.read_parquet("orders/*.parquet", session=session)
    frame.write_parquet("orders-copy.parquet")
```

Direct sinks and Arrow batch readers avoid constructing an intermediate pandas
DataFrame. Individual DuckDB operators may still require blocking state; use
`explain_write()` and `profile()` to inspect the plan and observed resource use.

## Unsupported behavior

DuckPD never silently falls back to pandas. Unsupported methods, dtypes, or
argument combinations raise before result execution. The authoritative method
and argument inventory is the [API compatibility guide](COMPATIBILITY.md); the
Narwhals adapter has a separate generated
[compatibility matrix](NARWHALS_COMPATIBILITY.md).
