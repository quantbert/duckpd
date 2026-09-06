# Changelog

All notable changes to DuckPD will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
DuckPD uses the Semantic Versioning-inspired, PEP 440-compatible policy in
the [release policy](RELEASES.md).

## Unreleased

### Added

- Fixed-duration `DataFrame.rolling(..., on=...)`,
  `DataFrameGroupBy.rolling(..., on=...)`, and datetime-indexed Series rolling
  windows with pandas-compatible `min_periods` defaults and
  `closed='right'|'left'|'both'|'neither'` boundaries.
- Typed row and range window frames in the logical plan. Duration windows
  compile to nanosecond `RANGE` frames and preserve timezone-aware timestamps.
- Timestamp/duration arithmetic, fixed-duration `Series.dt.floor()`, `ceil()`,
  and `round()`, plus timezone conversion, UTC localization, and timezone
  removal with pandas-compatible collected dtypes.
- Pandas categorical metadata propagation, lazy `Series.cat` metadata/codes
  accessors, ordered comparisons, and unused-category expansion for supported
  `groupby(observed=False)` aggregations.

### Changed

- Fixed-duration rolling now rejects unordered timestamps, null timestamps,
  and duplicate timestamps within a group instead of returning
  engine-dependent peer results.
- Replaced the stale feature-store implementation roadmap with an architecture
  description of the shipped native DuckPD implementation and removed obsolete
  external-project comparisons.

## 0.1.4 - 2026-09-06

### Added

- Native `duckpd.FeatureStore` and `Session.feature_store()` APIs for building
  lazy feature frames from a versioned `catalog.json`.
- Exact multi-family alignment and point-in-time alignment backed by DuckDB's
  native `ASOF LEFT JOIN`. Catalog-declared `availability_delay` values prevent
  features from becoming visible before they would have been known, including
  sparse predecessors from the full declared dataset history.
- Local, Hugging Face (`hf://`), and HTTP(S) feature stores with yearly and
  monthly partition templates.
- Execution-time partition caching that prunes irrelevant time partitions,
  projects only required columns, and preserves the cumulative requested
  column set as subsequent queries expand a cached partition.
- Shared-cache safety through validated root-relative catalog paths,
  per-partition process coordination, unique staging files, and atomic
  replacement.
- Lazy reference-table access through `store.table()`, time-windowed lazy
  frames through `store.feature_batches()`, and headless cache pre-warming
  through `store.sync()` and `SyncReport`.
- Typed `FeatureParquetSource` and `AsOfJoinPlan` logical nodes, keeping remote
  materialization and point-in-time joins visible to planning, explain, and
  execution rather than hiding them behind eager preprocessing.
- A standalone `python -m benchmark.featurestore` harness reporting cold
  remote execution, repeated warm-cache execution, row counts, cache bytes,
  and cold-to-warm speedup as JSON.
- An end-to-end FeatureStore notebook and README documentation covering the
  feature-store workflow and every supported local, remote, database, and
  in-memory source feeding DuckPD's lazy DataFrame engine.

### Fixed

- `feature_batches(frame=...)` now resolves the time column from the supplied
  frame instead of depending on catalog dataset order.
- Exact alignment now supports multiple output aliases for the same physical
  feature.
- Remote timeseries datasets now honor path templates declared in
  `metadata.json`.

## 0.1.3 - 2026-09-05

### Changed

- CSV and Parquet readers now preserve file scan order automatically with a
  hidden stable row identity. Positional, cumulative, ranking, rolling, and
  first/last-sensitive operations no longer require `order_by=` for file-backed
  frames; SQL and table relations remain strict.
- Explicit `order_by=` on file readers now uses the hidden source identity as a
  final tie-breaker, preserving deterministic file order for duplicate sort
  keys. Single-file Parquet scans use DuckDB's native row-number metadata so
  predicate and projection pushdown remain available.

### Fixed

- Parquet scans fall back to a generated source ordinal when a physical column
  is named `file_row_number`, avoiding a collision with DuckDB's virtual
  metadata column.
- `DataFrame.commit()` now preserves automatic Parquet row identity after file
  replacement, while schema validation ignores only DuckPD's generated
  identity and continues to validate user-defined hidden index columns.

## 0.1.2 - 2026-09-05

### Added

- Pandas-compatible row-based `DataFrameGroupBy.rolling()` and
  `SeriesGroupBy.rolling()` compile group keys into DuckDB window partitions,
  preserve grouped result indexes and ordering, and support lazy assignment
  back to the originating frame without materialization.
- Safe, read-only `Session.attach_postgres()` and `Session.attach_mysql()` APIs
  with lazy table scans (`AttachedDatabase.table()`), refresh-on-execution
  semantics, schema cache invalidation (`refresh_schema()`), structured
  credential parameters or DuckDB secret references, and full credential redaction
  from logical plans, `explain()`, and exceptions.
- Credential-safe HTTP/S3/GCS Parquet scans, scoped temporary S3/GCS secrets
  (`Session.create_s3_secret()`, `Session.create_gcs_secret()`), and read-only
  SQLite attachments (`Session.attach_sqlite()`) with session-owned cleanup.
- Unbounded remote scan guardrails (`unbounded_scan="warn"|"error"|"allow"`)
  raising `UnboundedRemoteScanError` or emitting warnings when remote transfer
  cannot be proven bounded.
- Backend-neutral source-fragment planning, conservative pushdown-candidate and
  required-local-work reporting, cross-source movement plans, and source-I/O
  profile metrics.
- Executing `explain("analyze")` output for projection, predicate, and Parquet
  row-group pruning evidence, guarded by the remote scan policy.
- Redacted remote schema inspection errors preventing connection string and
  credential leakage during attachment and table planning failures.
- Release checks now validate project/changelog metadata before publishing an
  already-versioned immutable release; validated benchmark tracks report source
  bytes and preserve unknown remote-transfer metrics as `null`.

## 0.1.0a1 - 2026-09-04

### Added

- Cardinality validation (`validate="1:1"`, `"1:m"`, `"m:1"`, `"m:m"`, and verbose aliases) for `DataFrame.merge` and `DataFrame.join`, executed as bounded pre-flight relational checks raising `MergeError` before result production.
- Relational label-list selection for `df.loc[[...]]`: preserves requested keys, duplicate requests, and key-group order; raises `KeyError` on missing labels at execution boundaries; and rejects sets with `TypeError`. It establishes a total order only when the input already has guaranteed order.
- Direct-sink tests ensure `DataFrame.write_csv` and `DataFrame.to_csv` avoid pandas materialization.
- Streaming tests ensure `DataFrame.to_arrow_batches` yields bounded record batches without constructing pandas DataFrames.
- A constrained-memory smoke test exercises sorting, aggregation, and Parquet output under strict DuckDB resource settings; it does not measure RSS or prove that spill occurred.
- Hypothesis differential tests cover selected nullable-integer and float round trips, reductions, string accessors, missing-value transformations, and concatenation against pandas 3.0.
- Column-wise concatenation (`duckpd.concat(..., axis=1)` or `axis='columns'`): optimizes same-lineage Series and DataFrames into a single lazy projection with zero joins, aligns multi-frame inputs via explicit index outer/inner joins, supports `ignore_index=True` renumbering, and rejects colliding column labels when `ignore_index=False`.
- Explicit-index cross-frame arithmetic for DataFrames and Series: numeric operators align lazily through one-to-one outer joins, DataFrame columns use pandas-style union alignment, and absent, incompatible, or duplicate index contracts fail without Cartesian expansion.
- Numerical boundary trimming (`Series.clip` and `DataFrame.clip`): compiles thresholds lazily to bounded `CASE WHEN` SQL expressions supporting scalar, Series, and per-column dictionary bounds with NULL preservation.
- Value replacement (`Series.replace` and `DataFrame.replace`): compiles replacements lazily via typed `CASE WHEN` branches supporting scalars, lists, mappings, and column-specific dictionaries with type-compatibility filtering and NULL detection.
- Series renaming (`Series.rename`): alters Series name metadata lazily without execution.
- Query profiling (`DataFrame.profile` and `ProfileResult`): executes queries with DuckDB structured JSON profiling enabled and returns typed performance metrics (execution latency, CPU time, rows scanned, rows returned, bytes read/written, peak buffer memory, peak spill directory size, JSON serialization, and human-readable summary).
- Deterministic row sampling (`DataFrame.sample` and `Series.sample`): uses reservoir sampling for unseeded fixed-size requests, stable content-hash ordering for seeded requests, and exact pandas-compatible round-to-even sizing for `frac`, with `ignore_index` RangeIndex resetting.
- Persistent table sinks (`DataFrame.save_as_table`): persists visible columns to DuckDB tables with `"error"`, `"overwrite"`, and schema-validated `"append"` modes, wrapped in transactional rollback to ensure table integrity on failure.
- Local Parquet atomic commit (`DataFrame.commit` and `CommitReport`): canonicalizes local source paths at frame creation, streams lazy modifications through a sibling staging file, preserves hidden index columns and Arrow schema/pandas metadata, validates DuckDB logical types and row-count preservation, supports optional backup retention, and performs atomic replacement. POSIX mode and available extended attributes are copied; Windows replacement metadata is preserved by `ReplaceFileW`. Owner/group, Parquet encodings, and physical layout may change. The initial implementation is single-writer and does not lock unrelated processes.
- Cross-platform process RSS sampling (`get_peak_rss_bytes` in `benchmark.metrics`): unifies Linux `/proc/self/status` VmHWM, macOS `getrusage`, and Windows `ctypes` peak working set sampling without unconditional `resource` imports.
- Experimental Narwhals plugin: `nw.from_native()` wraps DuckPD DataFrames as lazy frames; `nw.col`/`nw.lit` expressions support aliases, scalar broadcasting, arithmetic, comparisons, boolean composition, casts, null predicates, documented string/datetime accessors, global reductions, and lazy grouped aggregation. Supported transformations build DuckPD plans without collection.
- Narwhals lazy coverage now includes numeric transforms, ordered cumulative/rank/row-window expressions, relational transforms, equi/cross joins, scalar temporal/decimal schemas, explicit Arrow/pandas collection, and direct Parquet sinks. Undefined reshape, join, nested-type, fallback, and public plugin-scan paths reject before execution.
- Typed semantic metadata now includes expression aliases and nullability, explicit `RowIdentity` stability/uniqueness/source keys, and `SourceProvenance` with sanitized canonical locations, fingerprints, write capability, and transformation history.
- A named idempotent logical optimizer adds safe predicate pushdown, required-column/source liveness, combined limits, top-k plans, redundant project/sort removal, and common-subplan persist recommendations.
- `explain("optimized")` and `explain("json")` expose optimized plans and per-pass before/after snapshots. Profiles separate DuckPD planning and execution timings; `scripts/benchmark_optimizer.py` warms and alternates variants, asserts Arrow-result equality, reports median/range statistics, and ablates each retained pass on Linux.
- Release artifact verification: generated Narwhals compatibility documentation, wheel/sdist content checks, and clean-environment wheel installation smoke tests are configured for the CI Python 3.11–3.14 matrix; local Linux verification passes on all four Python versions.
- Controlled Python escape hatches: sessions enforce `fallback="error"`; typed, batch-independent Arrow UDFs are registered explicitly and applied with `Series.map_arrow()`; `DataFrame.collect_small()` accepts only non-expanding local-Parquet plans with fixed-width output and a proven conservative memory upper bound, rejecting expanding or variable-width plans before execution; materialization reports and explain/profile output expose their boundaries.
- Read-only PostgreSQL and MySQL attachments through DuckDB extensions: structured parameters or caller-managed secrets, credential-redacted diagnostics, lazy table frames with refresh-on-execution data visibility, explicit schema-cache refresh and detach lifecycle, source capability reporting, configurable unbounded-network-scan guards, and live PostgreSQL 17/MySQL 8.4 CI coverage.

### Changed

- `duckpd.concat` now uses defined numeric reconciliation, preserves nullable integer values through union collection, rejects decimal/float precision loss, and rejects incompatible heterogeneous columns instead of coercing them to strings.
- Pandas collection preserves exact decimal, binary, and date values rather than accepting DuckDB's lossy/default pandas conversions.
- Pandas collection now preserves nullable integer, boolean, string, datetime/time-zone, and duration source dtypes through identity-preserving plans; SQL null output policy is explicit, outer joins safely promote plain integer/boolean payloads, and nested DuckDB types are rejected before execution.
- Row-wise concat now preserves input sequence and ordered-input row identity; persistence retains index and order metadata. Joins explicitly clear total ordering guarantees so positional and window operations fail early until an explicit stable sort is applied.
- Compiler output is checked against declared logical schemas; unknown types remain conservative rather than suppressing known mismatches.
- Stable-order requirements now cover first-tie ranking and composite `.loc` request/source identities; unordered operations reject before execution.

## Pre-tag development milestones

The snapshots below predate the repository's first version tag. They were
reconstructed from package-version commits and are historical development
milestones, not formal release records. Versions without attributable release
notes are omitted.

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
