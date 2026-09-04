# Changelog

All notable changes to DuckPD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
DuckPD uses the Semantic Versioning-inspired, PEP 440-compatible policy in
the [release policy](RELEASES.md).

## Unreleased

### Added

- Cardinality validation (`validate="1:1"`, `"1:m"`, `"m:1"`, `"m:m"`, and verbose aliases) for `DataFrame.merge` and `DataFrame.join`, executed as bounded pre-flight relational checks raising `MergeError` before result production.
- Relational label-list selection for `df.loc[[...]]`: preserves requested keys, duplicate requests, and key-group order; raises `KeyError` on missing labels at execution boundaries; and rejects sets with `TypeError`. It establishes a total order only when the input already has guaranteed order.
- Direct-sink tests ensure `DataFrame.write_csv` and `DataFrame.to_csv` avoid pandas materialization.
- Streaming tests ensure `DataFrame.to_arrow_batches` yields bounded record batches without constructing pandas DataFrames.
- A constrained-memory smoke test exercises sorting, aggregation, and Parquet output under strict DuckDB resource settings; it does not measure RSS or prove that spill occurred.
- Hypothesis differential tests cover selected nullable-integer and float round trips, reductions, string accessors, missing-value transformations, and concatenation against pandas 3.0.

### Changed

- `duckpd.concat` now uses defined numeric reconciliation, preserves nullable integer values through union collection, rejects decimal/float precision loss, and rejects incompatible heterogeneous columns instead of coercing them to strings.
- Pandas collection preserves exact decimal, binary, and date values rather than accepting DuckDB's lossy/default pandas conversions.
- Row-wise concat now preserves input sequence and ordered-input row identity; persistence retains index and order metadata. Joins explicitly clear total ordering guarantees so positional and window operations fail early until an explicit stable sort is applied.

## Untagged development milestones

The repository has no `v<version>` tags. The snapshots below were reconstructed
from package-version commits and are historical development milestones, not
formal release records. Versions without attributable release notes are omitted.

### Package version 0.0.7 snapshot — 2026-08-15

#### Added

- Window expressions (`WindowExpression`) and compiler translation to DuckDB `OVER (...)` window clauses.
- Lazy cumulative operations on `DataFrame` and `Series`: `cumsum`, `cummin`, `cummax`, and `cumprod` with `skipna` support and explicit `OrderSpec` validation.
- Lazy positional shifts and differences: `Series.shift` and `DataFrame.shift` (with `fill_value`), `Series.diff` and `DataFrame.diff`, and `Series.pct_change` and `DataFrame.pct_change`.
- Numerical ranking on `Series.rank` and `DataFrame.rank` supporting methods (`average`, `min`, `max`, `first`, `dense`), `na_option` (`keep`, `top`, `bottom`), `ascending`, and `pct`.
- Extended `DataFrame.drop_duplicates` to support `keep='last'` and `keep=False` via window row numbers and count filtering.
- MultiIndex exact and prefix selection via `df.loc[key]`.
- Positional row slicing via `df.iloc[start:stop]`.
- Intermediate materialization via `DataFrame.persist(name)`.
- Write strategy inspection via `DataFrame.explain_write(path)`.
- Calibrated synthetic OHLC Parquet benchmark generator and demo notebook.

### Package version 0.0.5 snapshot — 2026-08-14

#### Added

- Relational DataFrame joins (`DataFrame.merge`) supporting `inner`, `left`, `right`, `outer`, and `cross` joins with column collision suffix management and pandas null-key semantics (`IS NOT DISTINCT FROM`).
- Index-based join convenience method (`DataFrame.join`).
- Multi-DataFrame row-wise concatenation (`duckpd.concat`) with schema alignment and null-padding.
- Extended `DataFrameGroupBy` supporting dictionary aggregations (`agg({"a": "sum", "b": "mean"})`), string function aggregations (`agg("sum")`), and column selection indexing (`g["col"]`, `g[["col1", "col2"]]`).
- `Series.groupby` returning lazy `SeriesGroupBy` supporting `agg`, `sum`, `mean`, `min`, `max`, `count`, `size`, `std`, `var`, and `median`.
- Eager DataFrame and Series reductions: `sum`, `mean`, `min`, `max`, `count`, `size`, `std`, `var`, `median`, `quantile`, `any`, `all`, and `nunique`.
- Vectorized string accessors (`Series.str`) and datetime accessors (`Series.dt`).
- Missing-value transformations: `fillna`, `dropna`, `where`, `mask`, `isna`, and `notna`.

### Package version 0.0.1 snapshot — 2026-08-14

#### Added

- Initial walking vertical slice: `Session`, `DataFrame`, and `Series` wrappers.
- Lazy data sources: Parquet, CSV, pandas, Arrow, DuckDB table, and read-only SQL scans.
- Immutable logical plan IR: `ScanPlan`, `ProjectPlan`, `FilterPlan`, `SortPlan`, and `LimitPlan`.
- Recursive DuckDB compilation and typed expression translation.
- Explicit execution boundaries: `collect()`, `head()`, `to_arrow()`, `to_arrow_batches()`, `explain()`, and direct `write_parquet()`.
- Isolated session configuration: memory limit, temp directory, threads, and read-only mode.
- Context-local implicit session for standalone module-level readers.
- Initial development toolchain: `uv`, `pytest`, `ruff`, `pyright`, and Make targets.
