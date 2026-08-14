# DuckPD competitive landscape and implementation references

This document is a map of ideas worth reusing, behavior worth testing, and
sources worth revisiting. It is not evidence that two libraries have identical
semantics. Re-check linked documentation and pin benchmark versions before
making implementation claims.

## What it means to beat FireDucks

DuckPD should not define success as universal pandas compatibility or winning
every microbenchmark. The target is a stronger overall engineering proposition
for the documented API subset:

| Dimension | DuckPD target |
| --- | --- |
| Correctness | Match pinned pandas behavior for every supported argument, including dtype, null, index, and ordering semantics. |
| Safety | Never silently collect an unbounded frame or fall back to pandas. Reject unsupported behavior before scanning data. |
| Scale | Complete representative workloads larger than memory through DuckDB spill, with limitations reported clearly. |
| Observability | Expose logical rewrites, SQL, physical operators, compile and execution time, cardinalities, spill, I/O, and materialization boundaries. |
| Performance | Be competitive on supported db-benchmark, TPC-H, and OHLC workflows; publish reproducible latency, peak RSS, and I/O results. |
| Portability | Support Linux, macOS, and Windows on the documented Python matrix. |
| Interoperability | Preserve explicit pandas semantics while supporting pandas, Arrow, DuckDB, and eventually Narwhals without hidden copies. |
| Openness | Keep the implementation, compatibility matrix, benchmark harness, and known limitations inspectable. |

FireDucks is the primary pandas-acceleration competitor. DuckDB is DuckPD's
execution engine and optimizer. pandas is the semantic oracle. Ibis, Narwhals,
and Modin are architectural and interoperability references rather than
semantic authorities.

## Competitive snapshot

| Project | Useful strength | Important boundary | DuckPD response |
| --- | --- | --- | --- |
| FireDucks | Lazy DataFrame IR, runtime optimization, multithreaded Arrow backend, broad pandas-shaped API, import hook, tracing | Binary implementation; silent pandas fallback unless warnings are enabled; merge order and some pandas details intentionally differ; limited platforms | Reuse optimization and diagnostics ideas, but require explicit materialization and publish a narrower tested contract |
| DuckDB | Mature vectorized SQL engine, optimizer, Parquet pushdown, Arrow interchange, spill, profiling | SQL null/order/index semantics are not pandas semantics | Keep a DuckPD semantic and metadata layer above DuckDB; delegate relational execution rather than rebuilding kernels |
| Ibis | Mature typed relational expressions, SQL compilation, backend abstraction, windows, joins, streaming and sinks | Relational semantics do not supply pandas index, alignment, mutation, or exact dtype rules | Spike it as an optional compiler substrate while retaining DuckPD plans and pandas correction rules |
| Narwhals | Lightweight cross-DataFrame expression protocol and expression metadata | Polars-shaped semantics; its DuckDB equality helper does not establish pandas row-order semantics | Borrow metadata concepts and add optional native-frame interoperability later |
| Modin | API/query-compiler/executor layering, API coverage tracking, engine movement costs | Distributed partition machinery is excessive for a single embedded DuckDB plan; default-to-pandas can be expensive | Keep the layers, compatibility inventory, and future movement-cost idea; avoid partition infrastructure and silent fallback |
| pandas | Canonical public behavior and a deep regression suite | Eager, in-memory execution is the behavior being accelerated | Differentially test a declared subset; mine upstream tests by semantic area rather than copying private internals |

## FireDucks: adopt, adapt, avoid

### Adopt

- A DataFrame-specific immutable IR accumulated until an explicit result is
  required. FireDucks documents file writes, display, and `_evaluate()` as
  execution points. DuckPD should retain clearer public boundaries such as
  `collect()`, `head()`, `persist()`, and direct sinks.
- Required-column analysis and projection pushdown before scans, filters,
  sorts, joins, and aggregates.
- Predicate pushdown across semantically safe boundaries, especially before
  joins. Null and outer-join rules must be proven by tests before each rewrite.
- Pattern rewrites such as eliminating redundant sorts and folding a following
  `reset_index(drop=True)` into an operation that can directly produce a range
  index.
- Liveness and common-subplan analysis to decide whether an intermediate should
  be recomputed, cached, persisted, or eliminated.
- Runtime evidence for physical choices. FireDucks estimates GroupBy key
  cardinality to choose an algorithm. DuckPD should first expose DuckDB's
  estimates and physical choices, adding DuckPD sampling only for decisions
  DuckDB cannot make.
- Fallback and operator telemetry. FireDucks offers `-Wfallback`, kernel timing
  summaries, and Chrome trace JSON. DuckPD should report all materialization,
  transfer, spill, and explicit fallback boundaries by default in profiling.
- A release-note-derived compatibility corpus. FireDucks' history is a compact
  catalogue of difficult pandas behavior: empty/all-null reductions, result
  dtypes, nullable comparisons, GroupBy ordering, categorical `observed`, join
  keys and indexes, suffix collisions, rolling boundaries, duplicate labels,
  and non-string labels.

### Adapt

- FireDucks' benchmark mode eagerly evaluates every method but disables useful
  optimization. Prefer a DuckPD benchmark helper that separately records plan
  construction, compile, execution, and output conversion while preserving the
  normal optimized plan.
- FireDucks can transparently replace pandas imports. Defer a DuckPD import hook
  until API coverage is broad and machine-readable. Start with explicit
  `import duckpd as pd`; an import hook magnifies every unsupported edge case.
- FireDucks falls back by converting the whole object to pandas and back.
  DuckPD may eventually offer explicitly requested, size-bounded collection or
  Arrow batch UDFs, with estimates and transfer cost visible in `explain()`.
- FireDucks uses an Arrow CPU backend and owns parallel kernels. DuckPD should
  first exploit DuckDB's vectorized parallel engine and avoid duplicating it.
- FireDucks does not promise exact merge row order. DuckPD should either
  preserve pandas-required order using hidden row identities or declare the
  result unordered and require an explicit sort. Never inherit incidental SQL
  order.

### Avoid

- Do not claim drop-in pandas compatibility while excluding behavior only in
  prose. Generate compatibility docs from executable data.
- Do not silently run unsupported methods in pandas. Conversion can dominate
  latency and memory and can violate the larger-than-memory promise.
- Do not make display an unbounded execution boundary. A representation should
  show a plan or use a documented bounded preview.
- Do not copy pandas bugs, undefined chained-assignment behavior, or private
  APIs merely to increase an API count.
- Do not optimize from benchmark timing alone. Each rewrite needs semantic
  tests, plan assertions, and adversarial null/order/index cases.

## Compatibility corpus to mine

FireDucks release notes should be reviewed whenever the corresponding DuckPD
area is implemented. Convert relevant cases into original, minimal
differential tests against the pinned pandas version.

| Area | Cases to capture early |
| --- | --- |
| Dtypes and nulls | result-width normalization, unsigned scalars, boolean reductions, `NaN` versus SQL `NULL`, `NaT`, empty and all-null inputs, categorical fill |
| GroupBy | multiple/null keys, `dropna`, `observed`, first-seen order, empty groups, selector projection, rank methods, transform, result dtypes |
| Merge/alignment | null keys, index-name combinations, differing key types, suffix collisions, empty result index dtype, categorical payloads, duplicate keys, output order |
| Indexing | duplicate columns, non-default and MultiIndex inputs, boolean masks, negative positions, callable selectors, non-string labels |
| Windows | nulls in keys/values, consecutive equal values, `min_periods`, nonnumeric windows, tie methods, `ddof` |
| Datetime | units other than nanoseconds, timezone transitions, `date`/`time` scalars, locale formatting, out-of-range values |
| Optimizer | projection through join/concat/groupby, predicate through joins, empty projections, redundant sort removal, hidden-index preservation |
| Lifecycle | repeated metadata calls, cache invalidation, delayed memory deallocation, conversion and sink boundaries |

Start from pandas' public API and test folders for the authoritative current
behavior. FireDucks release notes identify risky combinations; they are not the
oracle.

## Optimizer backlog

DuckDB already performs many SQL-level optimizations. DuckPD optimization is
valuable where pandas semantics or the wrapper's plan shape hide opportunities
from DuckDB:

1. Required-column and hidden-metadata liveness analysis.
2. Projection pushdown into Parquet/table scans and through safe plan nodes.
3. Predicate pushdown through projects and into the preserved side of joins.
4. Limit and top-k pushdown, including `sort_values(...).head(n)`.
5. Redundant project, sort, reset-index, and materialization elimination.
6. Common-subplan detection with explicit `persist()` recommendations before
   any automatic caching.
7. Blocking-operator and spill-risk annotation.
8. Source-capability and transfer-cost planning for future remote execution.

Every pass should expose before/after logical plans, have idempotence tests, and
prove that schema, index, order, row identity, and null semantics are unchanged.

## Observability backlog

Build on DuckDB profiling rather than creating a parallel profiler:

- `explain()` should present DuckPD logical plan, optimized logical plan,
  generated SQL, DuckDB physical plan, and semantic metadata.
- `profile()` should execute once and return structured metrics for planning,
  compilation, operators, rows, bytes, peak memory where available, spill,
  remote requests, output conversion, and total wall time.
- Export DuckDB JSON profiles and optionally a Chrome Trace Event JSON view.
- Label execution and materialization boundaries in summaries. There is no
  implicit fallback to summarize; any explicit escape hatch must include its
  reason and estimated/actual bytes.
- Provide a benchmark context that forces completion without changing the
  optimizer, and distinguish cold/warm runs and I/O-included/excluded runs.

DuckDB 1.5 exposes `EXPLAIN (FORMAT json|html|graphviz|mermaid)`, `EXPLAIN
ANALYZE`, JSON profiling, detailed planner/optimizer metrics, and query-graph
rendering. Prefer these structured surfaces over parsing human-readable plans.

## Benchmark program

Use three complementary tracks and pin all hardware, software, data seeds, and
settings:

1. **db-benchmark:** GroupBy and join across cardinality, null, sortedness, and
   0.5/5/50 GB scales. Force evaluation and validate answers.
2. **TPC-H:** end-to-end relational pipelines at SF1 for CI, SF10 for regular
   comparison, and a larger spill scale. Report I/O separately from query time.
3. **OHLC workflows:** DuckPD's deterministic 100 MB, 1 GB, and 5 GB Parquet
   data; include filtering, resampling/grouping, rolling indicators, as-of
   joins, top-k, Arrow streaming, and direct Parquet output.

Compare pandas, DuckPD, direct DuckDB SQL, Polars, and FireDucks where the
platform supports it. Add Ibis-on-DuckDB during the substrate spike. Report:

- plan-build and compile latency;
- cold and warm execution wall time;
- peak RSS and configured DuckDB memory;
- bytes read, written, transferred, and spilled;
- result materialization/conversion time;
- result checksum and semantic validation status;
- unsupported, fallback, timeout, and out-of-memory outcomes explicitly.

Performance gates should initially prevent major regressions rather than claim
absolute supremacy. Publish a FireDucks comparison only for equivalent code,
validated results, normal optimized execution, and identical hardware/data.

## Ibis substrate spike

Before manually expanding DuckPD plans for GroupBy, join, and windows, build a
bounded prototype that lowers the existing DuckPD logical nodes to both the
current DuckDB compiler and Ibis expressions. Compare:

- generated SQL and compile latency;
- type/null behavior and error timing;
- preservation of hidden index/order/row-identity columns;
- joins, reductions, windows, Arrow batch streaming, and direct sinks;
- access to DuckDB-specific plans and profiling;
- dependency size, version coupling, and escape hatches.

Adopt Ibis only if it removes substantial compiler work without forcing pandas
semantics into Ibis or blocking DuckDB-specific capabilities. DuckPD's public
IR and semantic metadata must remain independently owned so the compiler can be
replaced.

## Primary references

### FireDucks

- [Introduction](https://fireducks-dev.github.io/docs/user-guide/01-intro/)
- [Execution model and benchmark mode](https://fireducks-dev.github.io/docs/user-guide/02-exec-model/)
- [Compiler optimization and multithreaded backend](https://fireducks-dev.github.io/docs/user-guide/03-acceleration/)
- [pandas compatibility boundaries](https://fireducks-dev.github.io/docs/user-guide/04-compatibility/)
- [Conversion and explicit evaluation APIs](https://fireducks-dev.github.io/docs/user-guide/05-api/)
- [Fallback and profiling tips](https://fireducks-dev.github.io/docs/user-guide/tips/)
- [Projection and predicate pushdown case study](https://fireducks-dev.github.io/posts/data_flow_optimization/)
- [Liveness analysis](https://fireducks-dev.github.io/posts/20241217_liveness_analysis/)
- [Caching intermediates](https://fireducks-dev.github.io/posts/efficient_caching/)
- [Tracing](https://fireducks-dev.github.io/posts/2024-12-20-trace/)
- [Lazy timing pitfalls](https://fireducks-dev.github.io/posts/2024-12-26-time-pitfalls/)
- [pandas dtype, null, and merge compatibility](https://fireducks-dev.github.io/posts/2024-12-19-araki-en/)
- [Cardinality-driven GroupBy selection](https://fireducks-dev.github.io/posts/est/)
- [Import hook design and limitations](https://fireducks-dev.github.io/posts/importhook/)
- [Release notes and regression catalogue](https://fireducks-dev.github.io/docs/release-note/)
- [FireDucks db-benchmark fork](https://github.com/fireducks-dev/db-benchmark)
- [FireDucks TPC-H fork](https://github.com/fireducks-dev/polars-tpch/tree/fireducks_20250204)

### DuckDB and pandas

- [DuckDB profiling](https://duckdb.org/docs/current/dev/profiling.html)
- [DuckDB workload tuning and spill limits](https://duckdb.org/docs/current/guides/performance/how_to_tune_workloads.html)
- [DuckDB order preservation](https://duckdb.org/docs/current/sql/dialect/order_preservation.html)
- [DuckDB Parquet overview](https://duckdb.org/docs/current/data/parquet/overview.html)
- [db-benchmark upstream](https://github.com/duckdblabs/db-benchmark)
- [pandas public API](https://pandas.pydata.org/docs/reference/index.html)
- [pandas test suite](https://github.com/pandas-dev/pandas/tree/main/pandas/tests)
- [pandas testing guidance](https://pandas.pydata.org/docs/dev/development/contributing_codebase.html)

### Ibis, Narwhals, and Modin

- [Ibis DuckDB backend](https://ibis-project.org/backends/duckdb)
- [Ibis expression API](https://ibis-project.org/reference/expression-generic)
- [Ibis datatype API](https://ibis-project.org/reference/datatypes)
- [Ibis pandas tutorial](https://ibis-project.org/tutorials/coming-from/pandas)
- [Narwhals internals](https://narwhals-dev.github.io/narwhals/how_it_works/)
- [Extending Narwhals](https://narwhals-dev.github.io/narwhals/extending/)
- [Narwhals testing helpers](https://narwhals-dev.github.io/narwhals/api-reference/testing/)
- [Modin architecture](https://modin.readthedocs.io/en/stable/development/architecture.html)
- [Modin default-to-pandas behavior](https://modin.readthedocs.io/en/stable/supported_apis/defaulting_to_pandas.html)