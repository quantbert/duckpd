# DuckPD API Compatibility & Semantic Guide

This document is the public API reference and compatibility guide for **DuckPD**. It defines the core execution and metadata contract, details **DuckPD-exclusive extensions**, documents **intentional semantic deviations**, and provides a classified inventory of supported operations against **pandas 3.0**.

---

## 1. Classification Scheme & Core Invariants

DuckPD is a **lazy relational DataFrame library** powered by DuckDB with a pandas-shaped API. Because DuckPD translates operations into symbolic relational plans rather than manipulating in-memory array buffers, every API falls into one of three classifications:

1. **`[DuckPD Extension]`**:
   Methods or functions that do not exist on standard pandas objects. They provide explicit control over plan compilation, materialization boundaries, profiling, native DuckDB file exports, and streaming Arrow readers.
2. **`[Intentional Deviation]`**:
   Methods that share a standard pandas name, but intentionally diverge in execution timing, return types, preconditions, or parameter constraints to preserve relational correctness and avoid silent resource exhaustion.
3. **`[Pandas-API Subset]`**:
   Methods that share a standard pandas name and target pandas 3.0 semantics for their documented parameter subset, but operate **lazily** (returning a plan-backed `duckpd.DataFrame` or `duckpd.Series`) and reject unsupported pandas arguments before execution.

### Invariants Enforced Across All Methods
* **Plan-Backed Handles**:
  `DataFrame` is a mutable Python handle backed by an immutable `LogicalPlan` (`self._plan`). Transformations return new plan-backed handles. Assignment (`df[col] = ...` or `df.loc[mask, col] = val`) rebinds the handle's internal plan reference to a new `ProjectPlan`, mutating handle state without query execution.
* **Rejection of `inplace=True`**:
  Methods accepting `inplace=True` in pandas (e.g. `drop`, `rename`, `fillna`, `clip`, `replace`) raise `UnsupportedOperationError("DuckPD does not support inplace=True")`.
* **No Silent Full-Frame Fallback**:
  Unsupported operations or argument combinations raise `UnsupportedOperationError` immediately before plan compilation. DuckPD will never silently download, copy, or convert an out-of-core dataset into an in-memory pandas DataFrame.
* **Strict Column Label Uniqueness**:
  Every visible column in a DuckPD DataFrame must have a unique string label. Operations that would produce duplicate column labels (such as `merge` without suffixes or `concat` with overlapping names when `ignore_index=False`) raise `ValueError`.
* **Ordering Observability**:
  CSV and Parquet scans plus pandas and Arrow snapshots carry hidden source
  order. SQL/table relations and joins are unordered unless order is declared
  or established. Positional and window operations require a guaranteed
  `OrderSpec`, raising `UnorderedOperationError` when ordering is unknown.

---

## 2. DuckPD Extensions (APIs Absent in pandas)

These functions and methods exist exclusively in DuckPD to control query planning, inspect physical execution graphs, profile performance, enforce memory limits, and stream data directly from DuckDB.

| DuckPD Extension | Return Type | Nearest pandas Workflow | Description & Semantic Purpose |
| :--- | :--- | :--- | :--- |
| **`df.collect()`** / **`s.collect()`** | `pd.DataFrame` / `pd.Series` | *(Already in memory)* | Compiles and executes the relational query plan, returning the final result as a real in-memory pandas object. |
| **`df.collect_small(max_bytes)`** | `pd.DataFrame` | `df.memory_usage(deep=True)` followed by a size check | Explicitly opts into pandas materialization only when a local-Parquet, non-expanding plan with fixed-width output types has a conservative row/type memory upper bound below the strict byte limit. Joins, unions, Python UDFs, strings, binary, nested, and other variable-width output reject before execution. The measured result must also fit. `session.last_materialization_report` records reason, upper bound, actual bytes, and limit. |
| **`df.limit(count, offset=0)`** | `duckpd.DataFrame` | `df.iloc[offset:offset+count]` | Appends a lazy `LIMIT count OFFSET offset` node to the query plan without reading rows or executing. |
| **`df.persist(name=None)`** | `duckpd.DataFrame` | `df.copy()` *(in Python heap)* | Materializes the plan into a temporary (or named) DuckDB table. Caches intermediate results in complex DAGs without transferring rows into Python heap memory. |
| **`df.profile()`** | `ProfileResult` | `%timeit` / `cProfile` | Executes the plan with native DuckDB JSON profiling enabled (`PRAGMA enable_profiling = 'json'`). Returns a structured `ProfileResult` containing execution latency, CPU time, rows scanned/returned, bytes read/written, peak buffer memory, peak temp spill size, and physical operator timings. |
| **`df.explain(mode="all"\|"logical"\|"optimized"\|"json"\|"sql"\|"physical"\|"analyze")`** | `str` | *(Not available)* | Logical, optimized, and JSON modes do not execute. SQL/physical compile DuckDB plans. Analyze executes `EXPLAIN ANALYZE`, obeys remote scan guards, and exposes runtime pruning evidence without collecting rows into pandas. |
| **`session.register_arrow_udf(...)`** / **`s.map_arrow(name)`** | `ArrowUDFSpec` / `duckpd.Series` | `Series.map()` | Registers declared input/output types, null and exception handling, determinism, side-effect metadata, and the required batch-independence contract. Execution remains inside DuckDB's Arrow UDF batches; no DuckDB relation `map()` or implicit pandas fallback is used. Explain/profile output exposes the boundary and unknown engine-batched transfer estimate. |
| **`df.explain_write(path, compression=...)`** | `str` | *(Not available)* | Inspects the direct-write strategy, target, compression, output schema, blocking operators, ordering, and spill configuration without writing rows. Any sizes are explicitly estimates. |
| **`df.write_parquet(path, ...)`** | `None` | `df.to_parquet(path)` | Executes a direct DuckDB `COPY (query) TO ... (FORMAT PARQUET)` sink. Streams directly from DuckDB's C++ engine to disk without constructing an intermediate pandas DataFrame. |
| **`df.write_csv(path, ...)`** | `None` | `df.to_csv(path)` | Direct DuckDB `COPY ... TO ... (FORMAT CSV)` export without pandas intermediate heap allocation. |
| **`df.to_arrow()`** | `pa.Table` | `pa.Table.from_pandas(df)` | Executes plan and returns an in-memory PyArrow `Table` directly from DuckDB. |
| **`df.to_arrow_batches(batch_size=1_000_000)`** | `pa.RecordBatchReader` | `pd.read_csv(..., chunksize=N)` | Streams results as PyArrow `RecordBatch` chunks directly from DuckDB's query reader, enabling bounded-memory chunk processing on larger-than-RAM datasets. `batch_size` must be a positive integer. |
| **`duckpd.connect(...)`** | `Session` | *(OS / cgroups)* | Configures an isolated DuckDB execution session with strict resource limits: `memory_limit`, `temp_directory`, `max_temp_directory_size`, `threads`, and `read_only`. `fallback` is fixed to `"error"`; any other value is rejected before a connection or scan. |
| **`session.sql(query)`** | `duckpd.DataFrame` | `pd.read_sql(query, con)` | Compiles a raw SQL `SELECT` statement into a lazy DuckPD `DataFrame`. |
| **`session.table(name)`** | `duckpd.DataFrame` | `pd.read_sql_table(name, con)` | Scans an existing DuckDB catalog table as a lazy DuckPD `DataFrame`. |
| **`from_pandas(df, ...)`** | `duckpd.DataFrame` | `pd.DataFrame(...)` | Copies a snapshot into an isolated session, appending a hidden stable row ordinal column (`__duckpd_row_ordinal_...__`) to track original row sequence. |
| **`from_arrow(table, ...)`** | `duckpd.DataFrame` | `pd.DataFrame(...)` | Retains an Arrow table or RecordBatch snapshot, appending a hidden stable row ordinal column (`__duckpd_row_ordinal_...__`) to preserve source sequence. |
| **`order_by=`** *(reader parameter)* | Metadata | `df.sort_values(...)` | Overrides automatic file/snapshot order with domain sort keys, or establishes order for SQL and table sources, so downstream positional and window operations are valid. |
| **`duckpd.FeatureStore(source, ...)`** / **`session.feature_store(...)`** | `FeatureStore` | *(Not available)* | Creates an embedded feature store connecting to local or remote (`hf://`, `http://`, or `https://`) Parquet feature datasets with yearly or monthly partition templates. Supports exact multi-family equi-joins and point-in-time (`ASOF LEFT JOIN`) alignment with `availability_delay` lookahead bias protection. Returns lazy `duckpd.DataFrame` objects. |
| **`store.table(name)`** | `duckpd.DataFrame` | `pd.read_parquet(...)` | Returns a static reference or dimension table from the feature store catalog as a lazy `duckpd.DataFrame`. |
| **`store.features(...)`** | `duckpd.DataFrame` | *(Not available)* | Returns aligned feature selections as a lazy `duckpd.DataFrame` supporting exact equi-joins or point-in-time alignment with an explicit time spine. |
| **`store.feature_batches(window=...)`** | `Iterator[duckpd.DataFrame]` | *(Not available)* | Yields consecutive time-windowed lazy `duckpd.DataFrame` instances for chunked streaming without materializing the full dataset into memory. |
| **`store.sync(...)`** | `SyncReport` | *(Not available)* | Headless partition pre-fetching and column projection for batch pipelines and cluster workers. |

---

## 3. Intentional Semantic Deviations from pandas 3.0

These methods share standard pandas names, but deviate in execution timing, preconditions, return types, or parameter constraints to preserve relational invariants.

| Method / Feature | pandas 3.0 Behavior | DuckPD Behavior | Rationale & Guidance |
| :--- | :--- | :--- | :--- |
| **`head(count=5)`** | Eager in-memory slice of already-materialized rows. | **Eager execution boundary with bounded materialization**: compiles `LIMIT count` and calls `collect()`, returning at most `count` rows as an eager pandas DataFrame preview. | Bounded output prevents memory exhaustion during interactive notebook inspection. *(Note: plans containing upstream blocking operations such as `sort_values` or `groupby` may still scan upstream rows within DuckDB before truncating).* Use `df.limit(count)` for lazy truncation. |
| **`repr(df)` / Display** | Eagerly scans and formats data rows into a text table. | **Plan-focused summary**: outputs visible column labels and the logical plan class name (e.g. `DuckPD DataFrame\nColumns: ['a', 'b']\nPlan: ScanPlan`). | Inspecting a lazy frame must never trigger an accidental multi-gigabyte table scan or network fetch. Use `df.explain()` for complete plan trees. |
| **`concat(objs, axis=1)`** | Aligns on implicit `RangeIndex(0, n)` if unindexed. Permits duplicate column labels in output. | **Requires an explicit index** for multi-frame joins (`AlignmentError` if absent). Same-plan Series optimize into a **single projection** (no join). Rejects duplicate column labels when `ignore_index=False`. | Relational tables have no stable natural row numbers across files. Implicit positional alignment across separate frames is rejected to prevent silent data corruption. |
| **Cross-frame arithmetic** (`df1 + df2`, `s1 + s2`, and analogous numeric operators) | Aligns both axes, including implicit and duplicate indexes. | **Requires matching, unique explicit indexes** on separate plans and rejects different sessions, index level counts, names, or DuckDB key types with `AlignmentError`. Unknown uniqueness is validated lazily as one-to-one before result production; duplicate indexes raise `MergeError` rather than Cartesian-expanding. DataFrame columns use pandas-style sorted union alignment. | Positional and duplicate-key alignment across lazy relations is ambiguous without materializing index sequences. Explicit unique-index contracts prevent silent row multiplication or mismatches. |
| **`to_csv(path, ...)`** | Supports `path_or_buf=None` to return a CSV string in memory; accepts many formatting options. | **Direct file export only**: requires a destination `path`, executes via DuckDB `COPY`, and never constructs a pandas DataFrame. Does not support returning a CSV string. | Avoids materializing entire datasets in Python heap memory. For small string serialization, use `df.collect().to_csv()`. |
| **`df.loc[key]`** | Returns a `Series` if key matches 1 row; `DataFrame` if duplicate keys exist. | **Always returns a lazy `DataFrame`**. Ordered label-list reindexing (`df.loc[[...]]`) requires guaranteed input order. | Keeps plan return types deterministic before execution. |
| **`df.iloc[start:stop]`** | Slices by physical memory row offset on any frame. | **Requires a guaranteed `OrderSpec`** (`UnorderedOperationError` if unordered). CSV, Parquet, pandas, and Arrow sources provide one automatically; SQL and table scans require `order_by=`. | Positional slicing is non-deterministic when the source provides no row sequence. |
| **Window & Cumulative Functions** (`cumsum`, `shift`, `diff`, `pct_change`, `rank`, `rolling`, `expanding`) | Assumes physical row order in memory. | **Requires a guaranteed `OrderSpec`**. Raises `UnorderedOperationError` before query execution if ordering is unknown. | Relational window functions require explicit `OVER (ORDER BY ...)` clauses to avoid non-deterministic output. |
| **`merge()` & `join()` Ordering** | Preserves left/right order arbitrarily; `sort=True` produces an ordered frame. | **Explicitly clears total ordering guarantees** (`OrderSpec()`). | SQL joins lack deterministic tie-breakers for duplicate join keys. Downstream order-dependent operations must explicitly sort. |
| **`df.replace()` / `s.replace()`** | Allows global replacement dictionaries with arbitrary heterogeneous types across columns. | Validates **type compatibility per column** (`_is_replace_compatible`). Incompatible types (e.g. strings on numeric columns) are skipped. | DuckDB SQL requires uniform return types across branches in a `CASE WHEN` expression; avoids runtime binder type mismatches. |
| **Null Joins** | In SQL `NULL = NULL` is unknown (does not match). Pandas matches null with null. | **Matches pandas**: joins compile with `IS NOT DISTINCT FROM` so null keys join with null keys as in pandas. | Preserves pandas relational semantics rather than SQL ternary logic. |
| **Collection dtypes and nulls** | Dtypes originate from in-memory arrays and pandas extension metadata. | Preserves supported pandas nullable integer, boolean, string, datetime/time-zone, and duration dtypes through identity-preserving plans. SQL nullable integers collect as `float64`; nullable SQL booleans use pandas `boolean`; decimal, binary, date, and string values use exact Python-object representations. Nested DuckDB types fail at source inspection. | Makes `pd.NA`, `NaN`, `NaT`, and SQL `NULL` conversions explicit instead of relying on DuckDB conversion defaults. |

---

## 4. Top-Level Functions & I/O (`duckpd`)

| Function | Classification | Supported Parameters | Returns | Notes / Status |
| :--- | :--- | :--- | :--- | :--- |
| `duckpd.connect(...)` | **`[DuckPD Extension]`** | `memory_limit`, `threads`, `temp_directory`, `max_temp_directory_size`, `read_only` | `Session` | Configures isolated DuckDB connection & resource limits. |
| `Session.attach_postgres(alias, ...)` | **`[DuckPD Extension]`** | Structured `host`, `database`, `user`, `password`, optional `port`, `schema`, `sslmode`; or a caller-managed DuckDB `secret`; `unbounded_scan` | `AttachedDatabase` | Installs/loads DuckDB's PostgreSQL extension and creates a `READ_ONLY` attachment. Credentials are excluded from frame plans, explanations, errors, and reprs. |
| `Session.attach_mysql(alias, ...)` | **`[DuckPD Extension]`** | Structured `host`, `database`, `user`, `password`, optional `port`; or a caller-managed DuckDB `secret`; `unbounded_scan` | `AttachedDatabase` | Installs/loads DuckDB's MySQL extension and creates a `READ_ONLY` attachment. |
| `Session.attach_sqlite(alias, path, ...)` | **`[DuckPD Extension]`** | Existing local `path`; `unbounded_scan` | `AttachedDatabase` | Installs/loads DuckDB's SQLite extension and attaches the file with `READ_ONLY`. Schema refresh detaches and reattaches the same file. |
| `Session.create_s3_secret(name, ...)` | **`[DuckPD Extension]`** | Scoped static `key_id`/`secret`, or `credential_chain=True`; optional `region`, `endpoint`, `scope` | `ObjectStoreSecret` | Creates a session-owned temporary DuckDB secret. Credential values never enter DuckPD plans or diagnostics. |
| `Session.create_gcs_secret(name, ...)` | **`[DuckPD Extension]`** | HMAC `key_id`/`secret`; optional `scope` | `ObjectStoreSecret` | Creates a session-owned temporary GCS secret for `gcs://` or `gs://` objects. |
| `AttachedDatabase.table(name, ...)` | **`[DuckPD Extension]`** | `schema`, `index`, `order_by`, `unbounded_scan` | `DataFrame` | Creates a lazy remote-table frame. Each execution sees data committed before that execution; `persist()` is the explicit snapshot boundary. |
| `AttachedDatabase.refresh_schema()` / `.detach()` | **`[DuckPD Extension]`** | None | `None` | Clears a server extension schema cache, reattaches SQLite, or explicitly releases an attachment. Session shutdown also detaches and removes DuckPD-owned temporary secrets. |
| `duckpd.read_parquet(path, ...)` | **`[Pandas-API Subset]`** | `path`, `session`, `hive_partitioning`, `union_by_name`, `index`, `order_by` | `DataFrame` | Lazy local or credential-free HTTP/HTTPS/S3/GCS/GS Parquet scan with automatic hidden file-order identity. URL credentials/query parameters are rejected; use a scoped temporary secret. |
| `duckpd.read_csv(path, ...)` | **`[Pandas-API Subset]`** | `path`, `session`, `header`, `delimiter`, `auto_detect`, `index`, `order_by` | `DataFrame` | Lazy CSV scan via DuckDB reader with automatic hidden file-order identity. Supports explicit `order_by` and `index` declarations. |
| `duckpd.from_pandas(df, ...)` | **`[DuckPD Extension]`** | `value`, `session`, `index`, `order_by` | `DataFrame` | Copies a snapshot into a session, tracking hidden source row identity. |
| `duckpd.from_arrow(table, ...)` | **`[DuckPD Extension]`** | `value`, `session`, `index`, `order_by` | `DataFrame` | Retains an Arrow snapshot with an appended hidden stable row identity column. |
| `duckpd.concat(objs, ...)` | **`[Intentional Deviation]`** | `objs`, `axis=0\|1`, `join='outer'\|'inner'`, `ignore_index=False`, `sort=False` | `DataFrame` | Supports `axis=0` (row-wise union) and `axis=1` (column-wise concatenation). Axis 1 optimizes same-plan Series into a single projection and aligns multi-frame inputs via explicit index joins. Rejects duplicate column labels when `ignore_index=False`. |

---

## 5. DataFrame Methods & Indexers (`duckpd.DataFrame`)

| Method / Property | Classification | Supported Parameters | Description / Semantics |
| :--- | :--- | :--- | :--- |
| `df[col]`, `df[cols]` | **`[Pandas-API Subset]`** | Single column label or sequence | Lazy column projection. |
| `df[mask]` | **`[Pandas-API Subset]`** | Boolean `Series` | Lazy row filtering. |
| `df[col] = value` | **`[Pandas-API Subset]`** | Label & scalar/Series/DataFrame | Lazy column assignment mutating handle state. |
| `df.assign(**kwargs)` | **`[Pandas-API Subset]`** | Callables or expressions | Sequential lazy column assignment. |
| `df.loc[mask, col] = val` | **`[Pandas-API Subset]`** | Boolean mask and column | Masked assignment compiled to `CASE WHEN`. |
| `df.loc[key]` | **`[Intentional Deviation]`** | Scalar, MultiIndex tuple/prefix, list of labels, or mask | Lazy label filtering and relational reindexing. Always returns a lazy `DataFrame`. Lists preserve requested keys and order; raises `KeyError` on missing labels. |
| `df.iloc[start:stop, columns]` | **`[Intentional Deviation]`** | Row slice plus integer/slice/list column selector | Lazy positional slicing. Requires guaranteed ordering (`OrderSpec`). |
| `df.set_index(keys, ...)` | **`[Pandas-API Subset]`** | `keys`, `drop=True` | Declares existing columns as an explicit lazy index without materialization. |
| `df.reset_index(...)` | **`[Pandas-API Subset]`** | `drop=False` | Removes the explicit index lazily without executing the plan. |
| `df.rename(columns=...)` | **`[Pandas-API Subset]`** | `columns`, `errors='raise'\|'ignore'` | Renames columns lazily, preserving metadata. |
| `df.drop(columns=...)` | **`[Pandas-API Subset]`** | `labels`, `columns`, `errors` | Drops columns lazily, preserving index/order keys. |
| `df.astype(dtype)` | **`[Pandas-API Subset]`** | Scalar dtype or dict mapping | Casts columns lazily across DuckDB/pandas types. |
| `df.fillna(value=...)` | **`[Pandas-API Subset]`** | Scalar or column dict | Imputes missing values lazily via `COALESCE`. |
| `df.dropna(...)` | **`[Pandas-API Subset]`** | `how='any'\|'all'`, `subset`, `thresh` | Filters rows with null values lazily. |
| `df.where(cond, other)` | **`[Pandas-API Subset]`** | Boolean mask / scalar / dict | Replaces values where condition is False. |
| `df.mask(cond, other)` | **`[Pandas-API Subset]`** | Boolean mask / scalar / dict | Replaces values where condition is True. |
| `df.clip(lower, upper)` | **`[Pandas-API Subset]`** | `lower=None`, `upper=None`, `axis=0` | Trims values at input thresholds lazily via `CASE WHEN` compilation for scalars or column dictionaries. |
| `df.replace(to_replace, val)` | **`[Intentional Deviation]`** | `to_replace`, `value=None` | Replaces values lazily via typed `CASE WHEN` branches. Filters replacement rules per column to enforce SQL type compatibility. |
| `df.isna()`, `df.notna()` | **`[Pandas-API Subset]`** | None | Returns lazy boolean DataFrame. |
| `df.sort_values(by, ...)` | **`[Pandas-API Subset]`** | `by`, `ascending`, `na_position` | Returns lazy sorted DataFrame. |
| `df.sample(...)` | **`[Pandas-API Subset]`** | `n=None`, `frac=None`, `random_state=None`, `ignore_index=False` | Lazy exact-count sampling: reservoir sampling for `n`, and deterministic hash/random ordering plus pandas-compatible round-to-even sizing for `frac`. |
| `df.limit(count, offset)` | **`[DuckPD Extension]`** | `count`, `offset` | Lazy limit plan node (`LIMIT count OFFSET offset`). |
| `df.head(count=5)` | **`[Intentional Deviation]`** | `count` | Bounded eager preview (`limit(count).collect()`). |
| `df.drop_duplicates(...)` | **`[Pandas-API Subset]`** | `subset`, `keep='first'\|'last'\|False` | Deduplicates rows (aggregate or window-based). |
| `df.nlargest(n, cols)` | **`[Pandas-API Subset]`** | `n`, `columns`, `keep` | Top `n` rows in descending order. |
| `df.nsmallest(n, cols)` | **`[Pandas-API Subset]`** | `n`, `columns`, `keep` | Bottom `n` rows in ascending order. |
| `df.cumsum()`, `cummin()`, `cummax()`, `cumprod()` | **`[Intentional Deviation]`** | `axis=0`, `skipna=True\|False`, `numeric_only` | Cumulative transforms (requires explicit `OrderSpec`). |
| `df.shift(periods, ...)` | **`[Intentional Deviation]`** | `periods`, `fill_value`, `axis=0` | Positional lag/lead shifts (requires explicit `OrderSpec`). |
| `df.diff(periods=1)` | **`[Intentional Deviation]`** | `periods`, `axis=0` | Discrete difference between current and prior row (requires `OrderSpec`). |
| `df.pct_change(periods=1)` | **`[Intentional Deviation]`** | `periods`, `axis=0` | Percentage change between rows (requires `OrderSpec`). |
| `df.rank(...)` | **`[Intentional Deviation]`** | `method`, `na_option`, `ascending`, `pct` | Numerical ranking (average, min, max, first, dense) (requires `OrderSpec`). |
| `df.rolling(window, ...)` | **`[Pandas-API Subset]`** | `window`, `min_periods`, `center=False`, `on`, `closed` | Row-count windows accept positive integers. Fixed-duration windows accept strings or `datetime.timedelta`, default `min_periods` to 1, require `on=` for DataFrames, and use the explicit datetime index for Series. Supports `closed='right'\|'left'\|'both'\|'neither'` and all rolling reductions listed above. |
| `df.expanding(...)` | **`[Intentional Deviation]`** | `min_periods` | Expanding window object (`sum`, `mean`, `min`, `max`, `std`, `var`, `count`) (requires `OrderSpec`). |
| `df.groupby(by, ...)` | **`[Pandas-API Subset]`** | `by`, `as_index`, `sort`, `dropna` | Creates `DataFrameGroupBy` builder. |
| `df.groupby(...).rolling(window, ...)` | **`[Pandas-API Subset]`** | `window`, `min_periods`, `center=False`, `on`, `closed` | Row-count and fixed-duration `DataFrameGroupBy` and `SeriesGroupBy` windows (`sum`, `mean`, `min`, `max`, `std`, `var`, `count`). Group keys compile to window partitions; source `OrderSpec` defines order within each group. DataFrame duration windows use `on=`; Series duration windows use the explicit datetime index. Results preserve pandas grouped index layout, and direct assignment to the originating frame uses row-preserving lazy alignment. |
| `df.merge(right, ...)` | **`[Intentional Deviation]`** | `how`, `on`, `left_on`, `right_on`, `left_index`, `right_index`, `suffixes`, `sort`, `validate` | Relational join with pandas null-key semantics (`IS NOT DISTINCT FROM`) and lazy cardinality validation. Clears total ordering guarantees. |
| `df.join(other, ...)` | **`[Intentional Deviation]`** | `how`, `lsuffix`, `rsuffix`, `sort`, `validate` | Index-based join convenience method supporting cardinality validation. Clears total ordering guarantees. |
| `df.collect()` / `to_pandas()` | **`[DuckPD Extension]`** | None | Executes plan and returns pandas DataFrame. |
| `df.to_arrow()` | **`[DuckPD Extension]`** | None | Executes plan and returns PyArrow Table directly from DuckDB. |
| `df.to_arrow_batches(batch_size=1_000_000)` | **`[DuckPD Extension]`** | `batch_size` | Streams execution results as `pa.RecordBatchReader` directly from DuckDB. `batch_size` must be positive integer. |
| `df.write_parquet(path, ...)` | **`[DuckPD Extension]`** | `path`, `compression`, `overwrite` | Direct DuckDB export to Parquet bypassing pandas DataFrame materialization. |
| `df.write_csv(path, ...)` | **`[DuckPD Extension]`** | `path`, `sep`, `header` | Direct DuckDB export to CSV bypassing pandas DataFrame materialization. |
| `df.to_csv(path, ...)` | **`[Intentional Deviation]`** | `path`, `sep`, `header` | Direct DuckDB CSV export. Requires a destination `path` (does not return a CSV string). |
| `df.persist(name=None)` | **`[DuckPD Extension]`** | `name` | Materializes plan into temporary table for DAG reuse. |
| `df.explain(mode=...)` | **`[DuckPD Extension]`** | `mode='all'\|'logical'\|'optimized'\|'json'\|'sql'\|'physical'\|'analyze'` | Detailed plan inspection; `analyze` is an execution boundary and all other modes avoid row execution. |
| `df.explain_write(path, ...)` | **`[DuckPD Extension]`** | `path`, `compression` | Write execution strategy inspection. |
| `df.profile()` | **`[DuckPD Extension]`** | None | Executes plan with DuckDB profiling enabled and returns structured `ProfileResult` metrics. |
| `df.save_as_table(name, ...)` | **`[DuckPD Extension]`** | `name`, `mode='error'\|'overwrite'\|'append'` | Direct DuckDB table persistence with schema validation and transactional failure rollback. |
| `df.commit(...)` | **`[DuckPD Extension]`** | `compression='snappy'`, `retain_previous=False` | Atomic in-place commit to one canonical local Parquet source with row count, DuckDB logical types, and Arrow schema/pandas metadata preservation. POSIX mode and available extended attributes are copied; Windows replacement metadata is preserved by `ReplaceFileW`. Owner/group, Parquet encodings, and physical layout are not guaranteed. Single-writer only; unrelated writers are not locked. |

Grouped rolling collection is a narrow exception to DuckPD's general
MultiIndex non-goal: `as_index=True` returns pandas' group-key-prefixed
MultiIndex, while `as_index=False` keeps group keys as columns and retains the
source index. Assignment does not align through that collected index; it keeps
the uncollected window expression attached to source row identity, so duplicate
source indexes cannot cause a join or Cartesian expansion.

Fixed-duration rolling requires guaranteed ascending timestamp order after group
keys. Null timestamps and duplicate timestamps within one group fail at the
execution boundary: DuckDB `RANGE` frames treat equal timestamp peers as one
frame, while pandas advances through duplicate rows sequentially. Rejecting
that ambiguous case prevents silently different results.

---

## 6. Series-Specific Methods (`duckpd.Series`)

`Series` objects represent typed expressions bound to one immutable frame plan:

| Method / Property | Classification | Supported Parameters | Description / Semantics |
| :--- | :--- | :--- | :--- |
| `s.name` | **`[Pandas-API Subset]`** | Property | Name label of the Series expression. |
| `s.rename(name, ...)` | **`[Pandas-API Subset]`** | `index=None` | Alters Series name metadata lazily without execution. |
| `s.to_frame(name=None)` | **`[Pandas-API Subset]`** | `name=None` | Converts Series expression to a single-column `DataFrame`, preserving index and order metadata. |
| `s.unique()` | **`[Pandas-API Subset]`** | None | Eagerly returns unique non-null values as a pandas Series. |
| `s.value_counts(...)` | **`[Pandas-API Subset]`** | `sort=True`, `ascending=False`, `dropna=True` | Returns a lazy Series of value frequency counts. |
| `s.nlargest(n)` | **`[Pandas-API Subset]`** | `n=5`, `keep='first'` | Returns top `n` values lazily. |
| `s.nsmallest(n)` | **`[Pandas-API Subset]`** | `n=5`, `keep='first'` | Returns bottom `n` values lazily. |
| `s.collect()` / `to_pandas()` | **`[DuckPD Extension]`** | None | Executes plan and returns pandas Series. |

All transformation, cumulative, window, and reduction methods listed for DataFrames (`astype`, `fillna`, `dropna`, `where`, `mask`, `clip`, `replace`, `sample`, `isna`, `notna`, `cumsum`, `shift`, `diff`, `pct_change`, `rank`, `rolling`, `expanding`, `groupby`) are also supported on `Series`.

---

## 7. Reductions (Eager Column Aggregations)

Available on both `DataFrame` and `Series`:

| Method | Classification | Parameters | Execution | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `count()` | **`[Pandas-API Subset]`** | None | **Eager** | Non-null count. |
| `size` | **`[Pandas-API Subset]`** | Property | **Eager** | Total count including nulls. |
| `sum()` | **`[Pandas-API Subset]`** | `skipna`, `min_count`, `numeric_only` | **Eager** | Sum of values (matches pandas empty/null rules). |
| `mean()` | **`[Pandas-API Subset]`** | `skipna`, `numeric_only` | **Eager** | Arithmetic mean. |
| `min()`, `max()` | **`[Pandas-API Subset]`** | `skipna`, `numeric_only` | **Eager** | Minimum and maximum values. |
| `std()`, `var()` | **`[Pandas-API Subset]`** | `ddof=1\|0`, `skipna`, `numeric_only` | **Eager** | Standard deviation and variance. |
| `median()` | **`[Pandas-API Subset]`** | `skipna`, `numeric_only` | **Eager** | 50th percentile / median value. |
| `quantile(q)` | **`[Pandas-API Subset]`** | `q=0.5`, `skipna`, `numeric_only` | **Eager** | Continuous quantile estimation (`quantile_cont`). |
| `any()`, `all()` | **`[Pandas-API Subset]`** | `bool_only`, `skipna` | **Eager** | Boolean logical reduction. |
| `nunique()` | **`[Pandas-API Subset]`** | `dropna=True` | **Eager** | Distinct non-null count per column. |

---

## 8. Vectorized Accessors (`Series.str` & `Series.dt`)

Methods operate lazily on plan-backed Series, compiling directly into DuckDB string and temporal functions.

### `Series.str` (`[Pandas-API Subset]`)
* `upper()`, `lower()`, `strip()`, `len()`
* `startswith(pat, na=False)`, `endswith(pat, na=False)`, `contains(pat, na=False)`
* `replace(pat, repl)`

### `Series.dt` (`[Pandas-API Subset]`)
* `year`, `month`, `day`, `hour`, `minute`, `second`, `date`
* `strftime(date_format)`
* `to_period(freq='Y'\|'M'\|'D')`

## 9. Narwhals Interoperability

DuckPD ships an experimental
[Narwhals plugin](https://narwhals-dev.github.io/narwhals/extending/).
It wraps a `duckpd.DataFrame` as a Narwhals `LazyFrame` without collecting and
keeps supported transformations in DuckPD/DuckDB. The machine-readable contract
and generated method table are in
[`narwhals-compatibility.json`](narwhals-compatibility.json) and
[`NARWHALS_COMPATIBILITY.md`](NARWHALS_COMPATIBILITY.md).

### Why this integration matters

- **Downstream libraries need one adapter.** A library written against Narwhals
  can accept DuckPD without adding a DuckPD-specific conversion path.
- **Compatibility does not require pandas materialization.** Supported Narwhals
  transformations build DuckPD logical plans; DuckDB remains the execution
  engine and large inputs remain lazy.
- **Applications can stay backend-neutral.** The same application boundary can
  accept DuckPD and other Narwhals backends while preserving each native frame.
- **Round-tripping preserves ownership.** `to_native()` returns the underlying
  `duckpd.DataFrame`, not a pandas or Polars substitute.
- **The protocol creates a testable contract.** Schema mapping, error classes,
  supported arguments, ordering behavior, and execution boundaries are recorded
  in the generated compatibility matrix.

For example, every operation below is lazy:

```python
import narwhals as nw

lazy = nw.from_native(duckpd_frame)
result = (
    lazy.with_columns(
        (nw.col("amount") * 2).alias("gross"),
        nw.col("customer").str.strip_chars().str.to_uppercase().alias("customer"),
        nw.col("created_at").dt.year().alias("year"),
    )
    .filter((nw.col("gross") > nw.lit(100)) & nw.col("customer").str.starts_with("A"))
    .select("order_id", "customer", "year", "gross")
    .sort("gross")
    .head(100)
)
native = result.to_native()  # duckpd.DataFrame; still not executed
```

Narwhals is an interoperability layer, not a new execution engine. It does not
increase pandas compatibility, make unsupported DuckPD operations available,
or guarantee that every Narwhals consumer works with the current adapter.
The adapter supports only operations marked supported in the generated table,
including the documented expression, relational, equi/cross join, schema,
collection, and sink subsets. As-of/semi/anti joins, reshape operations, nested
types, arbitrary Python `map_batches`, and public `backend="duckpd"` scans are
explicit exclusions. For the scan workaround, use a DuckPD reader and pass the
result to `narwhals.from_native()`; this remains lazy.


---

## 10. Ordering and Resource Contract

* **Hidden Row Identity**: CSV and Parquet scans plus pandas and Arrow snapshots carry a hidden stable row identity. It is never exposed in columns, indexes, Arrow output, or file sinks.
* **Deterministic Tie-Breaking**: User sorts append row identity only as a final tie-breaker. SQL and table scans remain unordered unless callers establish an order.
* **Stable Identity Operations**: `drop_duplicates`, `rank(method="first")`, top-N ties, `groupby(sort=False)`, and grouped rolling alignment use stable row identity where available.
* **Join Ordering Destruction**: Joins do not claim a total order, including with `sort=True`, because duplicate merge keys lack a stable tie-breaker. Ordering-sensitive follow-up operations must explicitly sort by enough columns to break ties.
* **Explicit Session Isolation**: Module-level helpers share a weak context-local implicit session. Explicit sessions created via `duckpd.connect(...)` remain isolated, configurable, and authoritative for resource management and cleanup.

---

## 11. Intentional exclusions

The following are unsupported contracts, not implicit future behavior:

- arbitrary row-wise Python `apply` and invisible pandas or DuckDB `map()`
  fallback;
- `GroupBy.apply`, categorical grouping, categorical metadata, and `.cat`;
- calendar-offset rolling windows such as months or years, timezone
  transformations, and temporal floor/ceil/round;
- nested list, array, struct, map, union, and enum collection;
- duplicate displayed column labels and implicit positional cross-frame
  alignment;
- unbounded eager collection except through an explicit collection API;
- partition-aware or remote in-place commit and multi-writer commit locking;
- Narwhals eager-frame protocols, arbitrary `map_batches`, as-of/semi/anti
  joins, unpivot/explode, nested schemas, and public plugin-dispatched scans.

Unsupported methods and argument combinations raise
`UnsupportedOperationError`, `UnorderedOperationError`, `AlignmentError`, or a
more specific validation error before result execution. DuckPD does not silently
materialize a pandas object as a fallback.
