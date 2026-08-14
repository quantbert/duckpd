# 0001: Core execution contract

Status: accepted

## Context

Pandas exposes mutable, indexed, ordered, materialized objects. DuckDB exposes
lazy relational plans whose rows are unordered unless ordering is explicit.
Treating those models as identical would create incorrect alignment, ordering,
and memory behavior.

## Decision

- DuckPD is a standalone library, not a Modin execution backend.
- Public `DataFrame` and `Series` objects build a typed immutable logical plan.
- DuckDB relations and expressions are compiler output, not DuckPD's semantic
  state.
- Transformations are lazy. The executor alone may trigger DuckDB output.
- `limit()` is lazy and `head()` is an eager, bounded pandas preview.
- `collect()` and `to_pandas()` return a real pandas DataFrame.
- Module-level readers create an implicit session retained by the returned
  frame. Explicit `Session` objects are preferred for shared resources.
- `repr` is plan-focused and does not scan data.
- Unsupported behavior raises before execution. There is no automatic pandas
  fallback.
- The first compatibility oracle is pandas 3.0 on Python 3.11+ with DuckDB 1.5.
- PyArrow is required because streaming output is part of the core contract.
- Absent indexes may become a pandas `RangeIndex` at collection but are not an
  internal alignment key. Explicit index and order metadata will be added
  before positional or cross-frame alignment APIs.

## Consequences

- Some pandas code will require explicit collection or ordering declarations.
- Cross-frame Series operations are rejected until index alignment exists.
- DuckPD can preserve bounded-memory execution and inspect complete plans.
- Compatibility is added method by method with differential tests.