# Changelog

All notable changes to DuckPD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and [PEP 440](https://peps.python.org/pep-0440/).

## Unreleased

### Added

- Cardinality validation (`validate="1:1"`, `"1:m"`, `"m:1"`, `"m:m"`, and verbose aliases) for `DataFrame.merge` and `DataFrame.join`, executed as bounded pre-flight relational checks raising `MergeError` before result production.
- Honest relational label-list selection for `df.loc[[...]]`: preserves exact requested key order, retains duplicate requested keys, raises `KeyError` on missing labels (at execution boundaries), rejects sets with `TypeError`, and establishes a guaranteed row order for subsequent positional and window operations.
- Direct-sink invariant tests ensuring `DataFrame.write_csv` and `DataFrame.to_csv` avoid pandas materialization.
- Streaming verification asserting `DataFrame.to_arrow_batches` streams chunks incrementally without accumulating pandas DataFrames.
- Memory-constrained out-of-core sorting and aggregation test verifying DuckDB's `temp_directory` spill path under strict buffer limits without process OOM.
- Hypothesis property-based differential test suite (`tests/test_property_dtypes.py`) verifying round-trip fidelity and reductions across numeric, boolean, string, missing-value, and concat transformations against pandas 3.0.

### Changed

- `duckpd.concat` now uses lossless numeric dtype reconciliation, preserves nullable integer values through union collection, and rejects incompatible heterogeneous columns before execution instead of coercing them to strings.
- Pandas collection preserves exact decimal, binary, and date values rather than accepting DuckDB's lossy/default pandas conversions.
- Row-wise concat now preserves input sequence and ordered-input row identity; persistence retains index and order metadata. Joins explicitly clear total ordering guarantees so positional and window operations fail early until an explicit stable sort is applied.

## [0.0.7] - 2026-08-15

### Added

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

## [0.0.5] - 2026-08-14

### Added

- Relational DataFrame joins (`DataFrame.merge`) supporting `inner`, `left`, `right`, `outer`, and `cross` joins with column collision suffix management and pandas null-key semantics (`IS NOT DISTINCT FROM`).
- Index-based join convenience method (`DataFrame.join`).
- Multi-DataFrame row-wise concatenation (`duckpd.concat`) with schema alignment and null-padding.
- Extended `DataFrameGroupBy` supporting dictionary aggregations (`agg({"a": "sum", "b": "mean"})`), string function aggregations (`agg("sum")`), and column selection indexing (`g["col"]`, `g[["col1", "col2"]]`).
- `Series.groupby` returning lazy `SeriesGroupBy` supporting `agg`, `sum`, `mean`, `min`, `max`, `count`, `size`, `std`, `var`, and `median`.
- Eager DataFrame and Series reductions: `sum`, `mean`, `min`, `max`, `count`, `size`, `std`, `var`, `median`, `quantile`, `any`, `all`, and `nunique`.
- Vectorized string accessors (`Series.str`) and datetime accessors (`Series.dt`).
- Missing-value transformations: `fillna`, `dropna`, `where`, `mask`, `isna`, and `notna`.

## [0.0.1] - 2026-08-14

### Added

- Initial walking vertical slice: `Session`, `DataFrame`, and `Series` wrappers.
- Lazy data sources: Parquet, CSV, pandas, Arrow, DuckDB table, and read-only SQL scans.
- Immutable logical plan IR: `ScanPlan`, `ProjectPlan`, `FilterPlan`, `SortPlan`, and `LimitPlan`.
- Recursive DuckDB compilation and typed expression translation.
- Explicit execution boundaries: `collect()`, `head()`, `to_arrow()`, `to_arrow_batches()`, `explain()`, and direct `write_parquet()`.
- Isolated session configuration: memory limit, temp directory, threads, and read-only mode.
- Context-local implicit session for standalone module-level readers.
- Initial development toolchain: `uv`, `pytest`, `ruff`, `pyright`, and Make targets.
