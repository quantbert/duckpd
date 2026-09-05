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

## Read-only remote databases

DuckPD uses DuckDB's `postgres` and `mysql` extensions rather than loading
remote tables through pandas. Each attachment call ensures that the
corresponding extension is installed and loaded, then creates the attachment
with `READ_ONLY`.

### Structured connection parameters

Pass credentials as structured values. DuckPD binds them into a temporary
DuckDB secret; they are never embedded in table names, logical plans,
`explain()` output, exceptions, or attachment reprs.

```python
import os

import duckpd as pd

with pd.connect() as session:
    postgres = session.attach_postgres(
        "warehouse",
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", "5432")),
        database=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        sslmode="require",
        unbounded_scan="allow",
    )

    orders = postgres.table(
        "orders",
        schema="reporting",
        order_by="order_id",
    )
    open_orders = orders[orders["status"] == "open"][
        ["order_id", "customer_id", "total"]
    ]

    # No rows are fetched until an explicit execution boundary.
    print(open_orders.explain("json"))
    result = open_orders.collect()
```

Use `attach_mysql()` with the same `host`, `port`, `database`, `user`, and
`password` structure:

```python
with pd.connect() as session:
    mysql = session.attach_mysql(
        "catalog",
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        database=os.environ["MYSQL_DATABASE"],
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
    )
    products = mysql.table("products")
```

### Existing DuckDB secrets

An existing persistent DuckDB secret avoids passing credentials to DuckPD.
Provide only its name:

```python
with pd.connect() as session:
    warehouse = session.attach_postgres(
        "warehouse",
        secret="production_postgres_readonly",
        schema="reporting",
    )
```

The caller owns an existing secret, so detaching or closing the session does
not delete it. DuckPD deletes temporary secrets that it creates from structured
parameters.

### Freshness and schema changes

A remote `DataFrame` is a lazy query, not a snapshot. Repeated execution of the
same frame sees data committed before each execution:

```python
first = orders.collect()
# Another transaction commits remote inserts or updates.
second = orders.collect()

snapshot = orders.persist()
```

`snapshot` is DuckDB-owned and does not change with the source. DuckDB caches
remote schemas separately. After remote DDL, clear that cache and construct a
new frame:

```python
warehouse.refresh_schema()
orders_with_new_schema = warehouse.table("orders", schema="reporting")
```

### Network-scan policy and explain output

`unbounded_scan` controls execution when DuckPD cannot prove a bound on data
transferred from the remote server:

| Value | Behavior |
| :--- | :--- |
| `"warn"` | Default. Emit `RemoteScanWarning`, then execute. |
| `"error"` | Raise `MaterializationError` before DuckDB executes the plan. |
| `"allow"` | Execute without the unbounded-transfer warning. |

The guard is intentionally conservative. A local output `limit()` does not
prove a network-transfer bound unless the remote extension can establish that
the limit executes remotely. Set the policy on the attachment, or override it
for one table:

```python
customers = warehouse.table("customers", unbounded_scan="error")
```

`explain("json")` reports the sanitized remote source, backend, scan policy,
unknown transfer estimate, and statically known projection/filter pushdown
capabilities. It never reports credentials.

### Cleanup

`attachment.detach()` releases one attachment early. Closing its owning
`Session` detaches every remaining remote database and removes every temporary
secret created by DuckPD. Frames belonging to a detached attachment cannot be
executed or used to construct new table scans.

## Unsupported behavior

DuckPD never silently falls back to pandas. Unsupported methods, dtypes, or
argument combinations raise before result execution. The authoritative method
and argument inventory is the [API compatibility guide](COMPATIBILITY.md); the
Narwhals adapter has a separate generated
[compatibility matrix](NARWHALS_COMPATIBILITY.md).
