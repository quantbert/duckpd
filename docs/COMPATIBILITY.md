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
  DuckDB relations are unordered multiset tables unless an explicit sort is declared. Positional and window operations strictly require a guaranteed `OrderSpec`, raising `UnorderedOperationError` before execution when ordering is unknown.

---

## 2. DuckPD Extensions (APIs Absent in pandas)

These functions and methods exist exclusively in DuckPD to control query planning, inspect physical execution graphs, profile performance, enforce memory limits, and stream data directly from DuckDB.

| DuckPD Extension | Return Type | Nearest pandas Workflow | Description & Semantic Purpose |
| :--- | :--- | :--- | :--- |
| **`df.collect()`** / **`s.collect()`** | `pd.DataFrame` / `pd.Series` | *(Already in memory)* | Compiles and executes the relational query plan, returning the final result as a real in-memory pandas object. |
| **`df.limit(count, offset=0)`** | `duckpd.DataFrame` | `df.iloc[offset:offset+count]` | Appends a lazy `LIMIT count OFFSET offset` node to the query plan without reading rows or executing. |
| **`df.persist(name=None)`** | `duckpd.DataFrame` | `df.copy()` *(in Python heap)* | Materializes the plan into a temporary (or named) DuckDB table. Caches intermediate results in complex DAGs without transferring rows into Python heap memory. |
| **`df.profile()`** | `ProfileResult` | `%timeit` / `cProfile` | Executes the plan with native DuckDB JSON profiling enabled (`PRAGMA enable_profiling = 'json'`). Returns a structured `ProfileResult` containing execution latency, CPU time, rows scanned/returned, bytes read/written, peak buffer memory, peak temp spill size, and physical operator timings. |
| **`df.explain(mode="all"\|"logical"\|"sql"\|"physical")`** | `str` | *(Not available)* | Inspects the query plan without scanning or reading data. Displays DuckPD's typed logical plan, the generated DuckDB SQL, and DuckDB's physical execution plan. |
| **`df.explain_write(path, compression=...)`** | `str` | *(Not available)* | Inspects the physical write plan, target path, compression codec, and output schema for direct file sinks without writing rows to disk. |
| **`df.write_parquet(path, ...)`** | `None` | `df.to_parquet(path)` | Executes a direct DuckDB `COPY (query) TO ... (FORMAT PARQUET)` sink. Streams directly from DuckDB's C++ engine to disk without constructing an intermediate pandas DataFrame. |
| **`df.write_csv(path, ...)`** | `None` | `df.to_csv(path)` | Direct DuckDB `COPY ... TO ... (FORMAT CSV)` export without pandas intermediate heap allocation. |
| **`df.to_arrow()`** | `pa.Table` | `pa.Table.from_pandas(df)` | Executes plan and returns an in-memory PyArrow `Table` directly from DuckDB. |
| **`df.to_arrow_batches(batch_size=1_000_000)`** | `pa.RecordBatchReader` | `pd.read_csv(..., chunksize=N)` | Streams results as PyArrow `RecordBatch` chunks directly from DuckDB's query reader, enabling bounded-memory chunk processing on larger-than-RAM datasets. `batch_size` must be a positive integer. |
| **`duckpd.connect(...)`** | `Session` | *(OS / cgroups)* | Configures an isolated DuckDB execution session with strict resource limits: `memory_limit`, `temp_directory`, `max_temp_directory_size`, `threads`, and `read_only`. |
| **`session.sql(query)`** | `duckpd.DataFrame` | `pd.read_sql(query, con)` | Compiles a raw SQL `SELECT` statement into a lazy DuckPD `DataFrame`. |
| **`session.table(name)`** | `duckpd.DataFrame` | `pd.read_sql_table(name, con)` | Scans an existing DuckDB catalog table as a lazy DuckPD `DataFrame`. |
| **`from_pandas(df, ...)`** | `duckpd.DataFrame` | `pd.DataFrame(...)` | Copies a snapshot into an isolated session, appending a hidden stable row ordinal column (`__duckpd_row_ordinal_...__`) to track original row sequence. |
| **`from_arrow(table, ...)`** | `duckpd.DataFrame` | `pd.DataFrame(...)` | Retains an Arrow table or RecordBatch snapshot, appending a hidden stable row ordinal column (`__duckpd_row_ordinal_...__`) to preserve source sequence. |
| **`order_by=`** *(reader parameter)* | Metadata | `df.sort_values(...)` | Declares guaranteed physical sort keys at data-source boundaries (`read_parquet`, `read_csv`, `from_pandas`, `from_arrow`) so downstream positional and window operations are valid. |

---

## 3. Intentional Semantic Deviations from pandas 3.0

These methods share standard pandas names, but deviate in execution timing, preconditions, return types, or parameter constraints to preserve relational invariants.

| Method / Feature | pandas 3.0 Behavior | DuckPD Behavior | Rationale & Guidance |
| :--- | :--- | :--- | :--- |
| **`head(count=5)`** | Eager in-memory slice of already-materialized rows. | **Eager execution boundary with bounded materialization**: compiles `LIMIT count` and calls `collect()`, returning at most `count` rows as an eager pandas DataFrame preview. | Bounded output prevents memory exhaustion during interactive notebook inspection. *(Note: plans containing upstream blocking operations such as `sort_values` or `groupby` may still scan upstream rows within DuckDB before truncating).* Use `df.limit(count)` for lazy truncation. |
| **`repr(df)` / Display** | Eagerly scans and formats data rows into a text table. | **Plan-focused summary**: outputs visible column labels and the logical plan class name (e.g. `DuckPD DataFrame\nColumns: ['a', 'b']\nPlan: ScanPlan`). | Inspecting a lazy frame must never trigger an accidental multi-gigabyte table scan or network fetch. Use `df.explain()` for complete plan trees. |
| **`concat(objs, axis=1)`** | Aligns on implicit `RangeIndex(0, n)` if unindexed. Permits duplicate column labels in output. | **Requires an explicit index** for multi-frame joins (`AlignmentError` if absent). Same-plan Series optimize into a **single projection** (no join). Rejects duplicate column labels when `ignore_index=False`. | Relational tables have no stable natural row numbers across files. Implicit positional alignment across separate frames is rejected to prevent silent data corruption. |
| **`to_csv(path, ...)`** | Supports `path_or_buf=None` to return a CSV string in memory; accepts many formatting options. | **Direct file export only**: requires a destination `path`, executes via DuckDB `COPY`, and never constructs a pandas DataFrame. Does not support returning a CSV string. | Avoids materializing entire datasets in Python heap memory. For small string serialization, use `df.collect().to_csv()`. |
| **`df.loc[key]`** | Returns a `Series` if key matches 1 row; `DataFrame` if duplicate keys exist. | **Always returns a lazy `DataFrame`**. Ordered label-list reindexing (`df.loc[[...]]`) requires guaranteed input order. | Keeps plan return types deterministic before execution. |
| **`df.iloc[start:stop]`** | Slices by physical memory row offset on any frame. | **Requires a guaranteed `OrderSpec`** (`UnorderedOperationError` if unordered). External scans require `order_by=`. | DuckDB relations are unordered multiset tables; positional slicing is non-deterministic without an explicit sort key. |
| **Window & Cumulative Functions** (`cumsum`, `shift`, `diff`, `pct_change`, `rank`, `rolling`, `expanding`) | Assumes physical row order in memory. | **Requires a guaranteed `OrderSpec`**. Raises `UnorderedOperationError` before query execution if ordering is unknown. | Relational window functions require explicit `OVER (ORDER BY ...)` clauses to avoid non-deterministic output. |
| **`merge()` & `join()` Ordering** | Preserves left/right order arbitrarily; `sort=True` produces an ordered frame. | **Explicitly clears total ordering guarantees** (`OrderSpec()`). | SQL joins lack deterministic tie-breakers for duplicate join keys. Downstream order-dependent operations must explicitly sort. |
| **`df.replace()` / `s.replace()`** | Allows global replacement dictionaries with arbitrary heterogeneous types across columns. | Validates **type compatibility per column** (`_is_replace_compatible`). Incompatible types (e.g. strings on numeric columns) are skipped. | DuckDB SQL requires uniform return types across branches in a `CASE WHEN` expression; avoids runtime binder type mismatches. |
| **Null Joins** | In SQL `NULL = NULL` is unknown (does not match). Pandas matches null with null. | **Matches pandas**: joins compile with `IS NOT DISTINCT FROM` so null keys join with null keys as in pandas. | Preserves pandas relational semantics rather than SQL ternary logic. |

---

## 4. Top-Level Functions & I/O (`duckpd`)

| Function | Classification | Supported Parameters | Returns | Notes / Status |
| :--- | :--- | :--- | :--- | :--- |
| `duckpd.connect(...)` | **`[DuckPD Extension]`** | `memory_limit`, `threads`, `temp_directory`, `max_temp_directory_size`, `read_only` | `Session` | Configures isolated DuckDB connection & resource limits. |
| `duckpd.read_parquet(path, ...)` | **`[Pandas-API Subset]`** | `path`, `session`, `hive_partitioning`, `union_by_name`, `index`, `order_by` | `DataFrame` | Lazy Parquet scan. Supports `order_by` and `index` declarations. |
| `duckpd.read_csv(path, ...)` | **`[Pandas-API Subset]`** | `path`, `session`, `header`, `delimiter`, `auto_detect`, `index`, `order_by` | `DataFrame` | Lazy CSV scan via DuckDB reader. Supports `order_by` and `index` declarations. |
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
| `df.rolling(window, ...)` | **`[Intentional Deviation]`** | `window`, `min_periods`, `center=False` | Rolling window object (`sum`, `mean`, `min`, `max`, `std`, `var`, `count`) (requires `OrderSpec`). |
| `df.expanding(...)` | **`[Intentional Deviation]`** | `min_periods` | Expanding window object (`sum`, `mean`, `min`, `max`, `std`, `var`, `count`) (requires `OrderSpec`). |
| `df.groupby(by, ...)` | **`[Pandas-API Subset]`** | `by`, `as_index`, `sort`, `dropna` | Creates `DataFrameGroupBy` builder. |
| `df.merge(right, ...)` | **`[Intentional Deviation]`** | `how`, `on`, `left_on`, `right_on`, `left_index`, `right_index`, `suffixes`, `sort`, `validate` | Relational join with pandas null-key semantics (`IS NOT DISTINCT FROM`) and lazy cardinality validation. Clears total ordering guarantees. |
| `df.join(other, ...)` | **`[Intentional Deviation]`** | `how`, `lsuffix`, `rsuffix`, `sort`, `validate` | Index-based join convenience method supporting cardinality validation. Clears total ordering guarantees. |
| `df.collect()` / `to_pandas()` | **`[DuckPD Extension]`** | None | Executes plan and returns pandas DataFrame. |
| `df.to_arrow()` | **`[DuckPD Extension]`** | None | Executes plan and returns PyArrow Table directly from DuckDB. |
| `df.to_arrow_batches(batch_size=1_000_000)` | **`[DuckPD Extension]`** | `batch_size` | Streams execution results as `pa.RecordBatchReader` directly from DuckDB. `batch_size` must be positive integer. |
| `df.write_parquet(path, ...)` | **`[DuckPD Extension]`** | `path`, `compression`, `overwrite` | Direct DuckDB export to Parquet bypassing pandas DataFrame materialization. |
| `df.write_csv(path, ...)` | **`[DuckPD Extension]`** | `path`, `sep`, `header` | Direct DuckDB export to CSV bypassing pandas DataFrame materialization. |
| `df.to_csv(path, ...)` | **`[Intentional Deviation]`** | `path`, `sep`, `header` | Direct DuckDB CSV export. Requires a destination `path` (does not return a CSV string). |
| `df.persist(name=None)` | **`[DuckPD Extension]`** | `name` | Materializes plan into temporary table for DAG reuse. |
| `df.explain(mode=...)` | **`[DuckPD Extension]`** | `mode='all'\|'logical'\|'sql'\|'physical'` | Detailed plan inspection without reading rows. |
| `df.explain_write(path, ...)` | **`[DuckPD Extension]`** | `path`, `compression` | Write execution strategy inspection. |
| `df.profile()` | **`[DuckPD Extension]`** | None | Executes plan with DuckDB profiling enabled and returns structured `ProfileResult` metrics. |
| `df.save_as_table(name, ...)` | **`[DuckPD Extension]`** | `name`, `mode='error'\|'overwrite'\|'append'` | Direct DuckDB table persistence with schema validation and transactional failure rollback. |
| `df.commit(...)` | **`[DuckPD Extension]`** | `compression='snappy'`, `retain_previous=False` | Atomic in-place commit to one canonical local Parquet source with row count, DuckDB logical types, and Arrow schema/pandas metadata preservation. POSIX mode and available extended attributes are copied; Windows replacement metadata is preserved by `ReplaceFileW`. Owner/group, Parquet encodings, and physical layout are not guaranteed. Single-writer only; unrelated writers are not locked. |

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

---

## 9. Ordering and Resource Contract

* **Hidden Row Identity**: Pandas and Arrow snapshots carry a hidden stable row identity. It is never exposed in columns, indexes, Arrow output, or file sinks.
* **Deterministic Tie-Breaking**: User sorts append row identity only as a final tie-breaker. External scans do not acquire an artificial order.
* **Stable Identity Operations**: `drop_duplicates`, `rank(method="first")`, top-N ties, and `groupby(sort=False)` use stable row identity where available.
* **Join Ordering Destruction**: Joins do not claim a total order, including with `sort=True`, because duplicate merge keys lack a stable tie-breaker. Ordering-sensitive follow-up operations must explicitly sort by enough columns to break ties.
* **Explicit Session Isolation**: Module-level helpers share a weak context-local implicit session. Explicit sessions created via `duckpd.connect(...)` remain isolated, configurable, and authoritative for resource management and cleanup.
