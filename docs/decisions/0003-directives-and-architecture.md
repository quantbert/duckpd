# 0003: Core Project Directives and Relational Architecture

Status: accepted

## Context

DuckPD is designed as a lazy relational DataFrame library with a pandas-shaped API and DuckDB as its execution engine. To prevent semantic drift and maintain a strict separation between in-memory pandas emulation and database-native relational query optimization, the project adheres to eight core architectural directives.

## Project Directives

1. **Pandas-shaped, DuckDB-native.**
   The public API feels like pandas, but operations map naturally onto DuckDB's relational and vectorized execution model.
2. **Stay lazy by default.**
   Transformations build a query plan rather than executing immediately. Execution happens only at explicit and intentional boundaries such as `collect()`, `head()`, Arrow conversion, or direct file/table sinks.
3. **Never silently fall back to pandas.**
   Unsupported operations fail explicitly with descriptive, actionable errors rather than unexpectedly materializing an entire dataset into Python heap memory. Error messages guide users with concrete remediation steps (such as specifying `order_by` at the data source or using `.sort_values(...)`) without ambiguous parameter suggestions. Users can always reason about where computation happens.
4. **Push work into DuckDB.**
   Filtering, projection, joins, aggregation, windowing, sorting, and expressions are translated into DuckDB operations so DuckDB can optimize the complete end-to-end query.
5. **Preserve pandas semantics where compatibility is claimed.**
   API similarity alone is not enough. Supported operations match pandas behavior as closely as practical, including edge cases around null handling, indexes, dtypes, grouping, and column behavior.
6. **Correctness before coverage.**
   It is better to support a smaller pandas surface correctly than to advertise broad compatibility backed by incomplete semantics, hidden fallbacks, or surprising execution behavior.
7. **Make execution visible and predictable.**
   Users can inspect when data is scanned, materialized, transferred, or written via `explain()`, `explain_write()`, and `profile()`. Laziness is a transparent property, not hidden magic.
8. **Exploit ecosystem boundaries.**
   DuckPD interoperates cleanly with pandas, Arrow, Parquet, SQL, Narwhals, and DuckDB itself. Crossing those boundaries is explicit and inexpensive.

## Architecture and Execution Model

DuckPD structures query planning and execution across clear layered boundaries:

```text
duckpd public API (DataFrame / Series / GroupBy / indexers / readers)
                    │
                    ▼
pandas-semantic layer (validation + rewrites + metadata transitions)
                    │
                    ▼
typed immutable logical plan and expression IR
                    │
                    ▼
DuckDB compiler (DuckDB relation & SQL AST with safe parameterization)
                    │
                    ▼
Session and executor (collect | Arrow streaming | Parquet/CSV sinks | persist)
```

### Ordering and Indexing Contracts
- Pandas and Arrow inputs are treated as stable snapshots with hidden relational row identity.
- CSV and Parquet scans carry hidden physical scan identity by default. SQL and
   table scans remain unordered unless `order_by=` is declared.
- Positional, ranking, cumulative, and window operations require a guaranteed `OrderSpec` and raise `UnorderedOperationError` otherwise.
- SQL joins clear total ordering guarantees due to non-deterministic duplicate join key ties.
- Label selections (`.loc`) return lazy handles; dynamic scalar/frame return-type switching is deferred to bounded eager APIs.
