# 0004: Retain the Native DuckDB Compiler

Status: accepted

## Context

The roadmap required a bounded evaluation of Ibis as an optional lowering substrate before further manual compiler expansion. DuckPD must continue to own pandas-specific validation and its typed metadata for index, ordering, row identity, provenance, nullability, execution boundaries, resource policy, and credential redaction.

The spike compared DuckPD 0.1.4 with Ibis 12.0.0 on the same in-memory data and DuckDB execution engine. It covered projection plus filtering, grouped aggregation, left join, and a row-based rolling window. Each pair produced equal normalized pandas results. Compilation was measured as the median of 100 warm calls.

| Case | DuckPD compiler | Ibis SQL compiler |
|---|---:|---:|
| Projection and filter | 2,139 µs | 956 µs |
| Grouped aggregate | 1,998 µs | 786 µs |
| Left join | 3,889 µs | 382 µs |
| Row window | 1,596 µs | 700 µs |

Ibis generated substantially shorter SQL. DuckPD's SQL retained hidden stable row identities, deterministic ordering tie-breakers, pandas empty-sum corrections, minimum-period window guards, and logical-to-physical column bindings. Those additions are required semantics rather than incidental compiler complexity.

The installed `ibis-framework` distribution occupied approximately 6.85 MB and declared seven unconditional dependencies: `atpublic`, `parsy`, `python-dateutil`, `sqlglot`, `toolz`, `typing-extensions`, and `tzdata`. Its DuckDB extra adds dependencies already present in DuckPD's environment, but Ibis would still add a second relational IR, another type system, SQLGlot coupling, and a release compatibility surface.

## Decision

Retain DuckPD's native DuckDB compiler. Do not add Ibis as a runtime or development dependency.

DuckPD's typed logical plan remains the only public semantic IR. The compiler may continue using DuckDB relations, expressions, and narrowly generated SQL where DuckDB's relation API cannot express an operation safely.

Ibis remains a design and benchmark reference. Reconsider it only if a future backend requirement needs multi-engine SQL generation or measured compiler maintenance materially exceeds the cost of adapting DuckPD metadata and execution contracts across two IRs.

## Rationale

- Ibis does not model DuckPD's index, stable row identity, source provenance, pandas collection rules, or fail-before-execution contract.
- Lowering through Ibis would still require DuckPD-owned validation and metadata transitions before lowering, plus DuckPD-owned profiling, direct-sink, commit, resource, and redaction adapters afterward.
- The spike showed lower Ibis compile latency, but compilation is not the dominant cost for the analytical workloads DuckPD targets. Adding a second IR to obtain that reduction would increase correctness risk and version coupling.
- Native DuckDB compilation preserves direct access to physical plans, structured profiling, extensions, ASOF joins, Arrow streaming, and direct sinks without another abstraction boundary.

## Consequences

- New logical operations require explicit native compiler support and semantic tests.
- Compiler latency remains a benchmark dimension; large regressions should be addressed within the native compiler.
- DuckPD should continue to delegate physical optimization and execution to DuckDB rather than duplicate DuckDB optimizer work.
- The temporary spike program is not retained because the decision records its reproducible corpus, versions, measurements, and acceptance criteria without adding an optional dependency to project tooling.
