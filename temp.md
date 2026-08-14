## Verdict

**DuckPD is highly viable as a lazy, pandas-shaped analytical DataFrame.** It is much less viable as a completely transparent replacement for every pandas workflow.

| Goal                                                                  |                                            Viability |
| --------------------------------------------------------------------- | ---------------------------------------------------: |
| Familiar pandas syntax over Parquet, DuckDB, Arrow and object storage |                                             **9/10** |
| Common analytical pandas operations over larger-than-memory data      |                                             **8/10** |
| Drop-in compatibility for arbitrary existing pandas programs          |                                             **4/10** |
| Perfect pandas semantics, including indexing, ordering and mutation   |                                             **2/10** |
| Universal `pd.connect()` to any datastore                             | **5/10**, unless source-side pushdown is designed in |

DuckDB is an unusually good engine for this. Its Python relations are symbolic and lazily evaluated, its execution engine can spill joins, grouping, sorting and window operations to disk, and results can be streamed as Arrow record batches rather than collected into one in-memory DataFrame. Some complex queries and aggregates can still exceed memory, but the fundamental larger-than-memory machinery is already present. ([DuckDB][1])

There is also direct precedent. Ponder previously exposed pandas through Modin while executing operations as SQL on DuckDB, Snowflake and BigQuery; Snowflake subsequently announced its acquisition of Ponder. That strongly validates both the technical concept and the demand for it. ([Google Colab][2])

## The product should be defined carefully

The winning definition would be:

> **A lazy relational DataFrame with pandas syntax and explicitly documented pandas compatibility.**

Not:

> A pandas DataFrame that happens to store its data in DuckDB.

That distinction matters because a pandas DataFrame and a SQL relation have fundamentally different semantics. Pandas has labeled axes and automatically aligns objects by row and column labels. SQL relations do not have a pandas index, and row order is not generally defined without an explicit ordering. Ibis encounters the same boundary: it provides lazy relational expressions over DuckDB but deliberately does not preserve a pandas index and notes that SQL does not guarantee row order. ([Pandas][3])

| Pandas behavior                         | DuckDB/SQL behavior                                    | DuckPD requirement                                 |
| --------------------------------------- | ------------------------------------------------------ | -------------------------------------------------- |
| Every frame has an index                | Tables have no implicit index                          | Track index columns as metadata                    |
| Row order is observable                 | Order is undefined without `ORDER BY`                  | Track an explicit order descriptor                 |
| Arithmetic aligns by index              | SQL combines rows through joins                        | Compile alignment into joins                       |
| Frames appear mutable                   | Query plans are immutable                              | Mutate the Python wrapper, not the source          |
| Arbitrary Python functions are accepted | Engine understands SQL expressions and registered UDFs | Translate, run Arrow UDFs, or reject               |
| `df` represents materialized data       | Lazy frame represents a query                          | Add explicit execution boundaries                  |
| Duplicate labels are allowed            | SQL column names generally need unique identities      | Separate internal column IDs from displayed labels |

The engine is therefore not the hard part. **The pandas semantic layer is the hard part.**

## A suitable user experience

```python
import duckpd as pd

con = pd.connect(
    "postgresql://analytics@host/database",
    read_only=True,
)

orders = con.table(
    "public.orders",
    index="order_id",
    order_by=["created_at", "order_id"],
)

paid = orders[orders["status"] == "paid"]

monthly = (
    paid
    .assign(
        month=lambda df: df["created_at"].dt.to_period("M"),
        net_amount=lambda df: df["amount"] - df["refund_amount"],
    )
    .groupby(["month", "customer_id"], as_index=False)
    .agg(
        revenue=("net_amount", "sum"),
        order_count=("order_id", "size"),
    )
    .sort_values(
        ["month", "revenue"],
        ascending=[True, False],
    )
)

monthly.explain()                       # Show generated plan and SQL
monthly.head(20)                        # Execute with LIMIT 20
monthly.write_parquet("monthly.parquet") # Never enters pandas memory
result = monthly.collect()              # Return a real pandas DataFrame
```

The important addition is that **materialization is explicit**:

* `collect()` or `to_pandas()` loads the complete result.
* `head()` executes only a limited query.
* `to_batches()` streams pandas or Arrow batches.
* `write_parquet()` and `save_as_table()` execute directly into a durable sink.
* `persist()` materializes an intermediate result into a temporary or persistent DuckDB table.
* `explain()` shows where computation happens and whether a source table will be scanned remotely.

Calling `collect()` on an enormous result will still be impossible because the destination is pandas. The value is that users can reduce, aggregate, join and export that enormous dataset without collecting it.

## Recommended internal architecture

```text
DuckPD DataFrame / Series / GroupBy API
                  |
                  v
       Pandas-semantic logical IR
     + index, order and dtype metadata
                  |
                  v
              Planner
        /                       \
Source-side SQL fragment      DuckDB fragment
        \                       /
                  v
       DuckDB execution / federation
                  |
                  v
 Arrow batches | DuckDB table | Parquet | pandas
```

### 1. DataFrame and Series objects

A `DuckDataFrame` should hold:

```python
@dataclass(frozen=True)
class FrameState:
    plan: LogicalPlan
    columns: tuple[ColumnMetadata, ...]
    index_columns: tuple[ColumnId, ...]
    ordering: tuple[OrderKey, ...]
    connection: Session
```

A `DuckSeries` should normally be an expression bound to a frame plan, not a materialized vector.

That makes this expression natural:

```python
df["revenue"] - df["cost"]
```

It creates an expression node rather than reading either column.

### 2. A typed logical plan

Do not construct nested SQL strings inside every public method. Introduce a small relational algebra:

```text
Scan
Project
Filter
Aggregate
Join
Sort
Window
Limit
Union
Materialize
```

And a typed expression algebra:

```text
Column
Literal
BinaryExpression
UnaryExpression
Function
Case
Cast
AggregateExpression
WindowExpression
```

DuckDB’s relational API could support an initial proof of concept because its relations already represent lazy symbolic queries. A dedicated IR becomes important once you implement pandas-specific rewrites, source pushdown and index alignment. ([DuckDB][4])

### 3. Index and order metadata

This is the most important architectural decision.

A frame should carry:

* Zero or more index columns.
* Whether the index is unique.
* An optional stable row identity.
* An explicit ordering.
* Whether that ordering is guaranteed or merely incidental.

Operations then transform that metadata:

* Filtering preserves order.
* Projection preserves order even when order columns become hidden.
* `sort_values()` establishes a new order.
* Grouping destroys row order unless pandas-compatible group ordering is explicitly recreated.
* Joining must carry left and right row identities when pandas join ordering matters.
* `drop_duplicates(keep="first")` needs a row-number expression based on the previous order.
* `iloc`, `shift`, `diff`, rolling calculations and positional slicing require stable order.

For remote tables, requiring an `index=` and `order_by=` declaration is preferable to silently pretending that the source has a natural row order.

### 4. Pandas-semantic rewrites

Many methods map easily to SQL, but not always to the obvious SQL:

* `groupby(..., dropna=True)` may need null group keys filtered out.
* `sum()` over all-null data differs from SQL `SUM`, depending on `min_count`.
* `std()` and `var()` need the correct sample/population behavior.
* `drop_duplicates(keep="last")` needs a window function.
* `rank(method="average")` requires more than plain SQL `RANK()`.
* `merge(validate="one_to_one")` requires cardinality validation.
* `concat(axis=1)` is index alignment, not `UNION`.
* Arithmetic between unrelated frames may require a full index join.
* `groupby(sort=False)` may require tracking first-seen group order.

These details are what distinguish DuckPD from a thin wrapper that merely translates method names into SQL.

### 5. Controlled Python execution

Element-wise Python functions can be registered as DuckDB UDFs. DuckDB supports both native scalar UDFs and Arrow-based UDFs, with Arrow UDFs operating in batches and generally being more efficient. ([DuckDB][5])

A useful policy would be:

```python
pd.options.fallback = "error"
# alternatives:
# "collect_small"
# "arrow_batches"
```

* `error`: unsupported operations fail before executing.
* `collect_small`: collect only when a cardinality estimate is below a configured threshold.
* `arrow_batches`: allow operations proven to be independently batchable.

Arbitrary `apply(axis=1)` should not silently collect a 500 GB table. Modin’s standard fallback converts the complete Modin frame to pandas, executes the operation, and converts it back; that behavior directly conflicts with DuckPD’s larger-than-memory guarantee. ([modin.readthedocs.io][6])

## The datastore question is more complicated than the DataFrame question

DuckDB supports a broad collection of file systems, table formats and database extensions, including Parquet, S3-compatible storage, Delta, Iceberg, PostgreSQL, MySQL, SQLite and ODBC. Parquet scans can use projection and predicate pushdown, row-group pruning and partial reads, making object-storage datasets an especially strong use case. ([DuckDB][7])

However, `pd.connect("...")` introduces a second problem: **where does the computation execute?**

### Local DuckDB and object storage: excellent

For:

* Local DuckDB databases
* Local Parquet/CSV/Arrow
* S3, GCS or compatible object storage
* Delta and Iceberg tables
* Arrow datasets and scanners

DuckDB can be the actual execution engine. This is the cleanest version of DuckPD.

### PostgreSQL and MySQL: viable, with data-movement risk

DuckDB can attach PostgreSQL and query its tables directly. But PostgreSQL filter pushdown is currently described as experimental; it should not be assumed that a complete group-by, join or window pipeline will execute inside PostgreSQL. A query could therefore pull a large volume of source data across the network and process it locally in DuckDB. ([DuckDB][8])

Out-of-core execution solves local RAM pressure. It does **not** solve unnecessary network transfer.

### Generic ODBC and cloud warehouses: requires a split planner

The ODBC scanner is useful for connectivity, but DuckDB documents it as single-threaded and not a high-performance data-transfer API. ([DuckDB][9])

For arbitrary databases and warehouses, DuckPD should have a source capability model:

```python
SourceCapabilities(
    projection_pushdown=True,
    filter_pushdown=True,
    aggregation_pushdown=True,
    join_pushdown=False,
    window_pushdown=False,
)
```

The planner can then split a pipeline:

1. Compile supported operations into source-native SQL.
2. Stream the reduced result through Arrow.
3. Execute unsupported or cross-source work in DuckDB.
4. Write or collect only the final result.

DuckDB can consume Arrow datasets, scanners and record-batch readers, so Arrow is a natural boundary between remote execution and local DuckDB execution. ([DuckDB][10])

If universal connectivity is a primary objective, the IR should therefore be backend-neutral from the beginning—even though DuckDB remains the default local engine.

## Build on Modin or start separately?

Modin’s architecture is a strong conceptual template:

```text
pandas API
    ↓
query compiler
    ↓
dataframe algebra
    ↓
execution/storage backend
```

Its current architecture explicitly separates the pandas-facing API, query compiler, logical planning, execution and storage formats. ([modin.readthedocs.io][11])

There are two reasonable routes.

### A standalone DuckPD implementation

Best when the goal is:

* A well-defined analytical subset.
* Clean lazy semantics.
* No dependency on Ray or Dask.
* Explicit unsupported-operation errors.
* Tight control over index and order handling.

This is the cleaner starting point.

### A Modin DuckDB query compiler

Best when the goal is:

* Very broad pandas API coverage.
* Reusing Modin’s public API wrappers.
* Eventually offering `import duckpd as pd` with high compatibility.
* Sharing compatibility tests and pandas version adaptation.

Ponder demonstrated that Modin can be extended in this direction. A long-standing Modin discussion also specifically considered DuckDB as an SQL backend and later referenced Ponder’s DuckDB implementation. ([GitHub][12])

I would **not** model DuckDB as another partitioned Ray/Dask engine. I would implement a relation-backed query compiler where one logical DataFrame corresponds to one query plan.

## Sensible first compatibility target

The first useful release should support:

* Reading DuckDB tables, Parquet, Arrow and attached PostgreSQL tables.
* Column selection and boolean filtering.
* Arithmetic, comparisons and boolean expressions.
* `assign`, `rename`, `drop`, `astype`, `fillna`, `dropna`.
* String and datetime accessors.
* `groupby().agg()`.
* `merge`, `join` and row-wise `concat`.
* `sort_values`, `head`, `nlargest`, `value_counts`.
* Basic window operations: `rank`, `cumsum`, `shift`, rolling aggregates.
* `collect`, `to_batches`, `write_parquet`, `save_as_table`, `persist`.
* `sql()` and `explain()` inspection methods.

Initially exclude or strictly limit:

* Arbitrary row-wise `apply`.
* MultiIndex.
* Duplicate column labels.
* Positional operations without declared order.
* Automatic full-frame fallback to pandas.
* In-place updates of source database tables.
* Operations whose result schema is data-dependent, such as unrestricted `get_dummies` or dynamic pivots.
* Complex object-dtype behavior.

Compatibility should be tested differentially against pandas on small randomized frames, including nulls, duplicate index values, unusual dtypes and operation sequences. Separately, execution tests should verify that large pipelines compile into one query and remain within configured memory limits.

## Existing adjacent projects

The space is not empty, which is encouraging rather than discouraging:

* DuckDB itself already exposes lazy relational transformations and pandas/Arrow conversion. ([GitHub][13])
* Ibis provides a mature lazy expression system with DuckDB as its recommended default local backend, but intentionally uses relational rather than pandas semantics. ([Ibis][14])
* Ponder demonstrated pandas-on-DuckDB and pandas-on-warehouse execution through Modin. ([Google Colab][2])
* A small open-source project called `lazy-pandas` also wraps a subset of pandas-like operations around DuckDB, but it appears limited in scope. ([GitHub][15])

There does not appear to be a dominant, mature open-source package delivering strict pandas-style lazy execution over DuckDB with explicit index/order semantics and safe larger-than-memory behavior.

## Bottom line

**DuckPD is worth building.**

The compelling version is:

> Familiar pandas expressions compiled into lazy DuckDB queries, with explicit ordering, explicit materialization, streaming outputs and no dangerous silent fallback.

The weak version is:

> Pretend every pandas operation works, then unexpectedly collect the entire dataset when translation fails.

A strong proof of concept needs only eight or nine logical plan nodes, a typed expression tree, index/order metadata, and roughly two dozen high-value pandas operations. That would be enough to prove the central experience on datasets far larger than RAM while exposing exactly where the deeper compatibility work lies.

[1]: https://duckdb.org/docs/current/clients/python/relational_api?utm_source=chatgpt.com "Relational API – DuckDB"
[2]: https://colab.research.google.com/github/ponder-org/ponder-notebooks/blob/main/duckdb/example/Ponder_Quickstart.ipynb?utm_source=chatgpt.com "Ponder_Quickstart.ipynb - Colab"
[3]: https://pandas.pydata.org/docs/dev/reference/api/pandas.DataFrame.html?utm_source=chatgpt.com "pandas.DataFrame — pandas 3.1.0.dev0+1386.gcb2086a1a4 documentation"
[4]: https://duckdb.org/docs/lts/clients/python/relational_api?utm_source=chatgpt.com "Relational API – DuckDB"
[5]: https://duckdb.org/docs/current/clients/python/function?utm_source=chatgpt.com "Python Function API – DuckDB"
[6]: https://modin.readthedocs.io/en/latest/supported_apis/defaulting_to_pandas.html?utm_source=chatgpt.com "Defaulting to pandas — Modin 0.37.0+3.g7ca200b documentation"
[7]: https://duckdb.org/docs/current/core_extensions/overview?utm_source=chatgpt.com "Core Extensions – DuckDB"
[8]: https://duckdb.org/docs/current/core_extensions/postgres/overview?utm_source=chatgpt.com "PostgreSQL Extension – DuckDB"
[9]: https://duckdb.org/docs/current/core_extensions/odbc/overview?utm_source=chatgpt.com "ODBC Extension – DuckDB"
[10]: https://duckdb.org/docs/current/guides/python/sql_on_arrow?utm_source=chatgpt.com "SQL on Apache Arrow – DuckDB"
[11]: https://modin.readthedocs.io/en/latest/development/architecture.html?utm_source=chatgpt.com "System Architecture — Modin 0.37.0+3.g7ca200b documentation"
[12]: https://github.com/modin-project/modin/issues/2589?utm_source=chatgpt.com "Steps for new SQL Engine: DuckDB · Issue #2589 · modin-project/modin"
[13]: https://github.com/duckdb/duckdb-web/blob/main/docs/current/guides/python/relational_api_pandas.md?utm_source=chatgpt.com "duckdb-web/docs/current/guides/python/relational_api_pandas.md at main · duckdb/duckdb-web · GitHub"
[14]: https://ibis-project.org/tutorials/basics.html?utm_source=chatgpt.com "10 minutes to Ibis – Ibis"
[15]: https://github.com/mariotaddeucci/lazy-pandas?utm_source=chatgpt.com "GitHub - mariotaddeucci/lazy-pandas: The power of duckdb with the ease of pandas · GitHub"

----

The key distinction is **memory cost versus rewrite cost**.

A 100 GB Parquet file should not consume 100 GB of RAM. But modifying a plain Parquet file will usually require reading and writing the affected data again.

## Proposal: mutations modify the query plan, not the file

```python
df = duckpd.read_parquet("orders.parquet")

df.loc[df["status"] == "pending", "status"] = "processed"
df["net_amount"] = df["gross_amount"] - df["tax_amount"]
```

Neither assignment touches the file. Internally, DuckPD changes an immutable logical plan:

```text
ParquetScan("orders.parquet")
    ↓
Project(
    status = CASE
        WHEN status = 'pending' THEN 'processed'
        ELSE status
    END,
    net_amount = gross_amount - tax_amount,
    ...
)
```

The Python `DataFrame` object appears mutable, but it merely points to a new plan.

Only an explicit sink executes it:

```python
df.to_parquet("orders_updated.parquet")
```

or, for replacing the source:

```python
df.commit()
```

## Execution should go directly from DuckDB to the file

`to_parquet()` must never internally call `collect()` or create pandas objects.

It should compile approximately to:

```sql
SET memory_limit = '8GB';
SET temp_directory = '/fast-ssd/duckpd-spill';

COPY (
    SELECT
        * REPLACE (
            CASE
                WHEN status = 'pending' THEN 'processed'
                ELSE status
            END AS status
        ),
        gross_amount - tax_amount AS net_amount
    FROM read_parquet('orders.parquet')
)
TO 'orders.parquet.duckpd-staging-7f28.parquet'
(
    FORMAT parquet,
    COMPRESSION zstd
);
```

DuckDB supports writing the result of a query directly with `COPY (query) TO ...`; there is no need to materialize the result in pandas. ([DuckDB][1])

For filters, projections, casts, arithmetic and similar operations, the execution pipeline is conceptually:

```text
read a batch
    → transform the batch
    → compress it
    → write it
    → release the batch
```

Memory usage is therefore related to active batches and operator state, not the complete 100 GB dataset.

## Blocking operations spill to disk

Operations such as these cannot always be executed as a simple stream:

* `sort_values`
* large joins
* `groupby`
* window calculations
* `drop_duplicates`
* index alignment

DuckDB supports spilling grouping, joins, sorting and window operations to a temporary directory for larger-than-memory execution. There are still edge cases involving several blocking operators or particular aggregate functions, so DuckPD should sometimes break a complicated plan into persisted stages. ([DuckDB][2])

Configuration could be exposed through DuckPD:

```python
session = duckpd.connect(
    memory_budget="8GB",
    spill_directory="/mnt/nvme/duckpd-spill",
    max_spill_size="300GB",
)
```

The DuckDB `memory_limit` setting controls its buffer manager rather than guaranteeing an exact process-RSS limit, so DuckPD should describe it as a working-memory budget rather than a perfect hard cap. ([DuckDB][3])

## Replacing the original file safely

DuckPD should never simultaneously read from and overwrite the same physical file.

For:

```python
df = duckpd.read_parquet("orders.parquet")
df["amount"] *= 1.25
df.commit()
```

the commit sequence should be:

1. Record the source file fingerprint: path, size, modification time and optionally hash.
2. Write the result to a unique staging file.
3. Validate schema, row count and file statistics.
4. Confirm that the source has not changed since execution began.
5. Atomically replace the source with the staging file.
6. Remove the previous version according to the retention policy.

DuckDB can return file and column statistics from `COPY`, which gives DuckPD useful commit-validation information. ([DuckDB][4])

A failure before step 5 leaves the original file untouched.

```python
report = df.commit(
    atomic=True,
    retain_previous=True,
)

print(report)
# rows_written=1_482_223_991
# bytes_written=91_381_001_284
# old_version="..."
# new_version="..."
```

## The unavoidable cost of a single Parquet file

For a single 100 GB Parquet file:

```python
df.loc[df["customer_id"] == 42, "status"] = "blocked"
df.commit()
```

Even though only one row changes, the safe baseline implementation performs approximately:

```text
read:   100 GB
write:  approximately 100 GB
RAM:    perhaps 2–8 GB
disk:   original + new file + possible spill
```

So it is memory-efficient but I/O-expensive.

I would not promise “in-place Parquet updates.” DuckPD should treat bare Parquet files as immutable objects and expose replacement as a copy-on-write operation.

## Partitioned datasets improve this considerably

Instead of:

```text
orders.parquet                 100 GB
```

prefer:

```text
orders/
    order_date=2026-08-01/
        part-000.parquet
    order_date=2026-08-02/
        part-000.parquet
    ...
```

Then:

```python
df.loc[
    df["order_date"] == "2026-08-02",
    "status"
] = "archived"

df.commit()
```

can rewrite only the `order_date=2026-08-02` partition, provided:

* the predicate identifies the affected partitions;
* the partition column itself is not changed;
* there are no transformations requiring rows from other partitions.

DuckDB supports partitioned Parquet writes and configurable file sizing. However, overwriting partition directories is not supported uniformly on remote file systems, so remote datasets should use versioned paths plus a catalog or manifest rather than destructive directory replacement. ([DuckDB][5])

## Three storage strategies

| Source                        | Modification strategy                 | Cost                                         |
| ----------------------------- | ------------------------------------- | -------------------------------------------- |
| Single Parquet file           | Full streaming rewrite                | Full file read/write                         |
| Partitioned Parquet dataset   | Rewrite affected partitions           | Proportional to touched partitions           |
| DuckDB/Iceberg/DuckLake table | Transactional update, merge or delete | Proportional to changed data and maintenance |

For frequent row-level updates, bare Parquet is the wrong storage abstraction. DuckPD should recommend converting it into a managed table:

```python
df = duckpd.read_parquet("orders/*.parquet")

editable = df.persist(
    "warehouse.duckdb",
    table="orders",
)

editable.loc[editable["customer_id"] == 42, "status"] = "blocked"
editable.commit()
```

Or:

```python
editable = duckpd.open_table(
    "iceberg://catalog/analytics/orders"
)
```

Current DuckDB Iceberg support includes `UPDATE`, `DELETE`, and `MERGE INTO`; updates can use merge-on-read semantics with positional delete files rather than rewriting the complete table immediately. ([DuckDB][6])

## Suggested API semantics

```python
# Create a new file
df.to_parquet("output.parquet")

# Replace the original source safely
df.commit()

# Append rows without rewriting existing files
df.append_to("orders/", partition_by=["order_date"])

# Upsert by key; only supported by managed datasets
df.merge_to(
    "iceberg://catalog/orders",
    key="order_id",
)

# See the expected physical work
df.explain_write()
```

Example output:

```text
Write strategy: full copy-on-write

Input:
  orders.parquet
  Estimated scan: 100.2 GB

Output:
  orders.parquet
  Estimated output: 82–110 GB

Execution:
  Projection: streaming
  Conditional update: streaming
  Sort: none
  Spill expected: no

Peak additional disk:
  82–110 GB

Memory budget:
  8 GB
```

## Recommended first implementation

For the initial DuckPD version, I would implement:

1. Lazy mutations represented as `Project`, `Filter`, `Case` and related plan nodes.
2. Direct `COPY (query) TO` sinks.
3. Full copy-on-write for single files.
4. Atomic local-file replacement.
5. Explicit spill-directory and memory-budget configuration.
6. Partition-aware replacement as the next optimization.
7. Managed DuckDB/Iceberg/DuckLake targets for true recurring updates.

I would not initially build a proprietary base-file-plus-delta-log format. That becomes a miniature lakehouse implementation involving manifests, snapshots, delete records, compaction and concurrency control. DuckPD should delegate that responsibility to DuckDB tables or an existing open table format.

[1]: https://duckdb.org/docs/current/sql/statements/copy "COPY Statement – DuckDB"
[2]: https://duckdb.org/docs/current/guides/performance/how_to_tune_workloads?utm_source=chatgpt.com "Tuning Workloads – DuckDB"
[3]: https://duckdb.org/docs/current/operations_manual/limits?utm_source=chatgpt.com "Limits – DuckDB"
[4]: https://duckdb.org/docs/current/sql/statements/copy?utm_source=chatgpt.com "COPY Statement – DuckDB"
[5]: https://duckdb.org/docs/current/data/partitioning/partitioned_writes?utm_source=chatgpt.com "Partitioned Writes – DuckDB"
[6]: https://duckdb.org/docs/current/core_extensions/iceberg/writing?utm_source=chatgpt.com "Writing to Iceberg – DuckDB"
