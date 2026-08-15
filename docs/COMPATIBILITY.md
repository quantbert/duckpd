# DuckPD API Compatibility Matrix

This document provides a comprehensive overview of the public API implemented in DuckPD, along with parameter support and behavior compared to pandas 3.0.

## Core Design Principles

1. **Strict Laziness**: Public transformations return lazy handles (`DataFrame`, `Series`, `GroupBy`) without reading source rows.
2. **Explicit Execution Boundaries**: Materialization occurs only when calling `collect()`, `to_pandas()`, `head(n)`, streaming readers (`to_arrow()`, `to_arrow_batches()`), direct sinks (`write_parquet()`, `write_csv()`), or scalar reductions.
3. **No Silent Fallbacks**: Unsupported argument combinations or unsupported operations fail before a query is sent to DuckDB.
4. **Order and Index Observability**: Guarantees are tracked explicitly in metadata; positional or window operations enforce ordering via `UnorderedOperationError`.

---

## 1. Top-Level Functions & I/O (`duckpd`)

| Function | Supported Parameters | Returns | Notes / Status |
| :--- | :--- | :--- | :--- |
| `duckpd.connect(...)` | `memory_limit`, `threads`, `temp_directory`, `max_temp_directory_size`, `read_only` | `Session` | Configures isolated DuckDB connection & limits. |
| `duckpd.read_parquet(path, ...)` | `path`, `session`, `hive_partitioning`, `union_by_name`, `index`, `order_by` | `DataFrame` | Lazy Parquet scan over local/remote files. |
| `duckpd.read_csv(path, ...)` | `path`, `session`, `header`, `delimiter`, `auto_detect`, `index`, `order_by` | `DataFrame` | Lazy CSV scan via DuckDB reader. |
| `duckpd.from_pandas(df, ...)` | `value`, `session`, `index`, `order_by` | `DataFrame` | Copies a stable snapshot and tracks hidden source row identity. |
| `duckpd.from_arrow(table, ...)` | `value`, `session`, `index`, `order_by` | `DataFrame` | Retains an Arrow snapshot with hidden stable row identity. |
| `duckpd.concat(objs, ...)` | `objs`, `axis=0`, `join='outer'` | `DataFrame` | Row-wise union with schema reconciliation and null-padding. |

---

## 2. DataFrame Methods (`duckpd.DataFrame`)

| Method / Property | Supported Parameters | Description / Semantics |
| :--- | :--- | :--- |
| `df[col]`, `df[cols]` | Single column label or sequence | Lazy column projection. |
| `df[mask]` | Boolean `Series` | Lazy row filtering. |
| `df[col] = value` | Label & scalar/Series/DataFrame | Lazy column assignment mutating handle state. |
| `df.assign(**kwargs)` | Callables or expressions | Sequential lazy column assignment. |
| `df.loc[mask, col] = val` | Boolean mask and column | Masked assignment compiled to `CASE WHEN`. |
| `df.loc[key]` | Scalar, MultiIndex tuple/prefix, list of labels, or mask | Lazy label filtering. Row selections return lazy frames; label-list request order is not yet guaranteed. |
| `df.iloc[start:stop, columns]` | Row slice plus integer/slice/list column selector | Lazy positional slicing. Stable pandas/Arrow snapshots qualify; external scans require `order_by`. |
| `df.rename(columns=...)` | `columns`, `errors='raise'|'ignore'` | Renames columns lazily, preserving metadata. |
| `df.drop(columns=...)` | `labels`, `columns`, `errors` | Drops columns lazily, preserving index/order keys. |
| `df.astype(dtype)` | Scalar dtype or dict mapping | Casts columns lazily across DuckDB/pandas types. |
| `df.fillna(value=...)` | Scalar or column dict | Imputes missing values lazily via `COALESCE`. |
| `df.dropna(...)` | `how='any'|'all'`, `subset`, `thresh` | Filters rows with null values lazily. |
| `df.where(cond, other)` | Boolean mask / scalar / dict | Replaces values where condition is False. |
| `df.mask(cond, other)` | Boolean mask / scalar / dict | Replaces values where condition is True. |
| `df.isna()`, `df.notna()` | None | Returns lazy boolean DataFrame. |
| `df.sort_values(by, ...)` | `by`, `ascending`, `na_position` | Returns lazy sorted DataFrame. |
| `df.limit(count, offset)` | `count`, `offset` | Lazy limit plan node (`LIMIT count OFFSET offset`). |
| `df.head(count=5)` | `count` | Bounded eager preview (`limit(count).collect()`). |
| `df.drop_duplicates(...)` | `subset`, `keep='first'|'last'|False` | Deduplicates rows (aggregate or window-based). |
| `df.nlargest(n, cols)` | `n`, `columns`, `keep` | Top `n` rows in descending order. |
| `df.nsmallest(n, cols)` | `n`, `columns`, `keep` | Bottom `n` rows in ascending order. |
| `df.cumsum()`, `df.cummin()`, `df.cummax()`, `df.cumprod()` | `axis=0`, `skipna=True|False`, `numeric_only` | Cumulative transforms (requires ordering). |
| `df.shift(periods, ...)` | `periods`, `fill_value`, `axis=0` | Positional lag/lead shifts (requires ordering). |
| `df.diff(periods=1)` | `periods`, `axis=0` | Discrete difference between current and prior row. |
| `df.pct_change(periods=1)` | `periods`, `axis=0` | Percentage change between rows. |
| `df.rank(...)` | `method`, `na_option`, `ascending`, `pct` | Numerical ranking (average, min, max, first, dense). |
| `df.rolling(window, ...)` | `window`, `min_periods`, `center=False` | Rolling window object (`sum`, `mean`, `min`, `max`, `std`, `var`, `count`). |
| `df.expanding(...)` | `min_periods` | Expanding window object (`sum`, `mean`, `min`, `max`, `std`, `var`, `count`). |
| `df.groupby(by, ...)` | `by`, `as_index`, `sort`, `dropna` | Creates `DataFrameGroupBy` builder. |
| `df.merge(right, ...)` | `how`, `on`, `left_on`, `right_on`, `left_index`, `right_index`, `suffixes`, `sort` | Relational join with pandas null-key semantics (`IS NOT DISTINCT FROM`). |
| `df.join(other, ...)` | `how`, `lsuffix`, `rsuffix`, `sort` | Index-based join convenience method. |
| `df.collect()` / `df.to_pandas()` | None | Executes plan and returns pandas DataFrame. |
| `df.to_arrow()` | None | Executes plan and returns Arrow Table. |
| `df.to_arrow_batches(batch_size)` | `batch_size` | Streams execution results as Arrow RecordBatches. |
| `df.write_parquet(path, ...)` | `path`, `compression`, `overwrite` | Direct zero-copy export to Parquet. |
| `df.write_csv(path, ...)` / `to_csv` | `path`, `sep`, `header` | Direct zero-copy export to CSV. |
| `df.persist(name=None)` | `name` | Materializes plan into temporary table for DAG reuse. |
| `df.explain(mode=...)` | `mode='all'|'logical'|'sql'|'physical'` | Detailed plan inspection without reading rows. |
| `df.explain_write(path, ...)` | `path`, `compression` | Write execution strategy inspection. |

---

## 3. Reductions (Eager Column Aggregations)

Available on both `DataFrame` and `Series`:

| Method | Parameters | Notes |
| :--- | :--- | :--- |
| `count()` | None | Non-null count. |
| `size` | Property | Total count including nulls. |
| `sum()` | `skipna`, `min_count`, `numeric_only` | Sum of values (matches pandas empty/null rules). |
| `mean()` | `skipna`, `numeric_only` | Arithmetic mean. |
| `min()`, `max()` | `skipna`, `numeric_only` | Minimum and maximum values. |
| `std()`, `var()` | `ddof=1|0`, `skipna`, `numeric_only` | Standard deviation and variance. |
| `median()` | `skipna`, `numeric_only` | 50th percentile / median value. |
| `quantile(q)` | `q=0.5`, `skipna`, `numeric_only` | Continuous quantile estimation (`quantile_cont`). |
| `any()`, `all()` | `bool_only`, `skipna` | Boolean logical reduction. |
| `nunique()` | `dropna=True` | Distinct non-null count per column. |

---

## 4. Vectorized Accessors (`Series.str` & `Series.dt`)

### `Series.str`
* `upper()`, `lower()`, `strip()`, `len()`
* `startswith(pat, na=False)`, `endswith(pat, na=False)`, `contains(pat, na=False)`
* `replace(pat, repl)`

### `Series.dt`
* `year`, `month`, `day`, `hour`, `minute`, `second`, `date`
* `strftime(date_format)`
* `to_period(freq='Y'|'M'|'D')`

---

## Ordering and resource contract

- Pandas and Arrow sources carry a hidden stable row identity. It is never
	exposed in columns, indexes, Arrow output, or file sinks.
- User sorts append row identity only as a final tie-breaker. External scans do
	not acquire an artificial order.
- `drop_duplicates`, `rank(method="first")`, top-N ties, and
	`groupby(sort=False)` use stable identity where available.
- Join and concat row identity propagation is not yet complete; code requiring
	pandas-exact output order after those operations should add `sort_values`.
- Module-level helpers share a weak context-local implicit session. Explicit
	sessions remain isolated and are preferred for configured or long-lived
	workloads.
