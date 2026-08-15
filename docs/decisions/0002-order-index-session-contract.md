# 0002: Ordering, indexing, and implicit session contract

Status: accepted

## Context

Pandas derives positional and tie-breaking behavior from materialized row
order. DuckDB relations are unordered unless ordering is represented in the
plan. Exact pandas `.loc` return types can also depend on runtime index
uniqueness, while module-level lazy readers need a session that outlives each
individual helper call.

## Decision

- Pandas and Arrow inputs are stable snapshots. DuckPD adds a hidden ordinal
  column and records it as row identity and internal ordering metadata.
- Hidden row identity is a relational implementation detail, not a pandas
  index. It is excluded from every public schema and sink.
- Declared sorts use row identity only as a final stable tie-breaker.
- External Parquet, CSV, SQL, and table scans remain unordered unless callers
  provide `order_by` or a future source adapter can prove a stable key.
- Ordering-sensitive APIs fail before execution when no guarantee exists.
- Lazy `.loc` row selections return lazy DuckPD frames. Exact pandas dynamic
  Series/DataFrame switching is deferred until bounded eager lookup and index
  uniqueness semantics are implemented.
- MultiIndex tuple and prefix matching is lazy. Ordered label-list reindexing,
  missing-label validation, and cross-frame alignment require dedicated
  relational operators and are not emulated with pandas.
- Module-level readers share a weak context-local implicit session. Explicit
  sessions remain isolated, configurable, and authoritative for cleanup.

## Consequences

- Snapshot-backed workflows gain deterministic `.iloc`, windows, duplicate
  retention, first-ranked ties, top-N ties, and first-seen group order.
- External scans retain honest relational semantics and may require explicit
  `order_by` where pandas code assumes physical order.
- Join and union identity propagation must be implemented before DuckPD claims
  pandas-exact output order for those operations.
- Scalar index access will be an explicit bounded execution boundary rather
  than a hidden full-frame materialization.