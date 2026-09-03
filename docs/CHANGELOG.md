# Changelog

All notable changes to DuckPD will be documented in this file.

## Unreleased

### Changed

- `duckpd.concat` now uses lossless numeric dtype reconciliation, preserves
  nullable integer values through union collection, and rejects incompatible
  heterogeneous columns before execution instead of coercing them to strings.
- Pandas collection preserves exact decimal, binary, and date values rather
  than accepting DuckDB's lossy/default pandas conversions.
- Row-wise concat now preserves input sequence and ordered-input row identity;
  persistence retains index and order metadata. Joins explicitly clear total
  ordering guarantees so positional and window operations fail early until a
  sort with sufficient tie-breakers is applied.

### Added

- Window expressions (`WindowExpression`) and compiler translation to DuckDB `OVER (...)` window clauses.
- Lazy cumulative operations on `DataFrame` and `Series`: `cumsum`, `cummin`, `cummax`, and `cumprod` with `skipna` support and explicit `OrderSpec` validation.
- Lazy positional shifts and differences: `Series.shift` and `DataFrame.shift` (with `fill_value`), `Series.diff` and `DataFrame.diff`, and `Series.pct_change` and `DataFrame.pct_change`.
- Numerical ranking on `Series.rank` and `DataFrame.rank` supporting methods (`average`, `min`, `max`, `first`, `dense`), `na_option` (`keep`, `top`, `bottom`), `ascending`, and `pct`.
- Extended `DataFrame.drop_duplicates` to support `keep='last'` and `keep=False` via window row numbers and count filtering.
- Added `Series.to_frame()` and `Series.collect()` for lazy Series evaluation and materialization.
- Extended `DataFrameGroupBy` supporting dictionary aggregations (`agg({"a": "sum", "b": "mean"})`),
  string function aggregations (`agg("sum")`), convenience methods (`sum`, `mean`, `min`, `max`,
  `count`, `size`), and column selection indexing (`g["col"]`, `g[["col1", "col2"]]`).
- `Series.groupby` returning lazy `SeriesGroupBy` supporting `by` as column name string,
  Series, or sequences, with `agg`, `sum`, `mean`, `min`, `max`, `count`, `size`,
  `std`, `var`, and `median` aggregations.
- `duckpd.concat` for lazy row-wise concatenation (`axis=0`) of multiple
  `DataFrame` and `Series` objects with schema reconciliation, type promotion,
  null padding for disjoint columns, and explicit index preservation.
- `DataFrame.std`, `DataFrame.var`, `DataFrame.median`, `DataFrame.quantile`,
  `DataFrame.any`, `DataFrame.all`, and their `Series` counterparts for eager
  reductions matching pandas 3.0 semantics with `ddof`, `skipna`, `bool_only`,
  and `numeric_only` support.
- `DataFrame.nunique` and `Series.nunique` return the count of distinct non-null
  values per column using DuckDB `COUNT(DISTINCT ...)`.
- `Series.unique` returns distinct non-null values as a pandas Series via a
  `DISTINCT` aggregate plan.
- `Series.value_counts` returns counts of unique values with `sort`,
  `ascending`, and `dropna` support, compiled as a grouped `SIZE` aggregate
  with an optional `SortPlan`.
- `DataFrame.drop_duplicates` removes duplicate rows using `subset` column
  selection and `keep="first"` semantics via grouped `any_value()` aggregates.
- `DataFrame.nlargest`, `DataFrame.nsmallest`, `Series.nlargest`, and
  `Series.nsmallest` return the top/bottom `n` rows via `sort_values` + `limit`
  composition.
- `DataFrame.astype` and `Series.astype` for lazy type casts across standard
  integer, float, boolean, string, date, timestamp, and decimal representations;
  supports `errors="ignore"` and dictionary column-mapping specifications.
- `DataFrame.fillna` and `Series.fillna` for lazy missing-value imputation via
  DuckDB `COALESCE`, supporting scalar replacements and column-mapping dicts.
- `DataFrame.dropna` and `Series.dropna` for lazy row filtering of missing values
  with `how="any"`, `how="all"`, `subset`, and `thresh` threshold parameters.
- `DataFrame.where`, `DataFrame.mask`, `Series.where`, and `Series.mask` for lazy
  conditional replacement using typed SQL `CASE WHEN` expressions.
- `DataFrame.isna`, `DataFrame.notna`, `Series.isna`, and `Series.notna`
  (plus `isnull`/`notnull` aliases) build lazy boolean frames and series backed
  by DuckDB `IS NULL` / `IS NOT NULL` predicates.
- `DataFrame.rename` lazily renames column labels via a dict mapping, preserving
  index and ordering metadata; `errors="ignore"`, `mapper`, and `axis` are
  supported while `inplace`, `copy=False`, `level`, and index renaming are
  rejected before execution.
- `DataFrame.drop` lazily drops columns by label or via the `columns` keyword,
  preserving index and ordering metadata; `errors="ignore"` is supported while
  `inplace`, `level`, and row dropping are rejected before execution.
- Initial `uv`-managed Python package and quality toolchain.
- Lazy pandas, Arrow, Parquet, DuckDB table, and read-only SQL sources.
- Typed scan, filter, project, sort, and limit logical plans.
- Lazy Series arithmetic, comparisons, boolean expressions, filtering,
  assignment, projection, sorting, and limiting.
- Explicit pandas and Arrow materialization, bounded previews, plan inspection,
  and direct Parquet output.
- Session resource configuration, source retention, execution counting, and
  clear closed-session errors.
- Immutable schema/index/order metadata with hidden-column preservation across
  projections and output boundaries.
- Lazy `set_index()` and `reset_index()`, plus source `index=` and `order_by=`
  declarations.
- Expression metadata for scalar, elementwise, and length-preserving behavior.
- Calibrated synthetic OHLC Parquet generator with smoke, 100 MB, 1 GB, and
  5 GB target-size presets.
- Self-documenting Make targets for development checks, builds, demo smoke
  runs, release validation, and cleanup.
- Package classifiers, project URLs, and PEP 561 typed-package metadata.
- Central documentation index with roadmap, design, decision, benchmark,
  changelog, and research sections.
- Explicit work-in-progress warning and a documented pre-`1.0` release policy.
- Make target for tested patch-version builds and manual PyPI publication.