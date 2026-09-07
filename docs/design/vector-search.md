# Vector Search and Analytical Retrieval

**Status: implemented exact search; experimental approximate mode is guarded by verified DuckDB HNSW plans.**

## Product thesis

DuckPD should not become a vector database server. Its opportunity is broader:
make retrieval an ordinary, composable DataFrame operation.

A vector search should return the same lazy `duckpd.DataFrame` as a Parquet
scan, SQL query, or feature-store selection. Users can then filter, join,
aggregate, apply windows, inspect execution, stream Arrow batches, or write the
result without crossing into a separate retrieval framework.

This supports DuckPD's long-term position as:

> **A pandas-shaped analytical runtime for data that is too large, too
> distributed, or too temporally complex for pandas.**

Vector search is the first multimodal retrieval primitive, not a new identity
for the project. Text, hybrid, spatial, and temporal retrieval may eventually
follow the same design.

## Why this belongs in DuckPD

DuckDB already provides the execution building blocks:

- built-in array distance functions for exhaustive nearest-neighbor search;
- the optional `vss` extension and HNSW indexes for approximate search;
- relational filtering, joins, aggregation, windows, and direct sinks.

DuckPD adds the layer that is missing from raw SQL:

- a familiar pandas-shaped API;
- typed lazy plans and early validation;
- explicit ordering, execution, and materialization contracts;
- point-in-time semantics for leakage-sensitive research;
- one observable pipeline across local, remote, and attached data.

The differentiated workflow is not merely finding similar rows. It is finding
eligible rows and immediately using them in a larger analytical pipeline.

```python
import duckpd as pd

with pd.connect(memory_limit="2GB") as session:
    documents = pd.read_parquet(
        "s3://research/news/*.parquet",
        session=session,
    )
    prices = pd.read_parquet(
        "s3://market/prices/*.parquet",
        session=session,
        order_by=["timestamp", "ticker"],
    )

    eligible = documents[
        (documents["available_at"] <= observation_time) & (documents["language"] == "en")
    ]

    matches = eligible.vector.search(
        query=query_embedding,
        column="embedding",
        metric="cosine",
        k=100,
        mode="exact",
        tie_breaker="document_id",
    )

    result = matches.merge(prices, on="ticker")
    result.write_parquet("similar-events.parquet")
```

The value is the composition: semantic retrieval, temporal eligibility,
relational enrichment, feature engineering, and output remain in one lazy
DuckDB-backed plan.

## Public API

### Distance expressions

Distance is useful independently of top-k retrieval, so it belongs on a Series
accessor and returns an ordinary lazy `Series`.

```python
scored = documents.assign(
    distance=documents["embedding"].vector.distance(
        query_embedding,
        metric="cosine",
    )
)
```

For a fixed-size float array this can compile to a DuckDB expression such as:

```sql
array_cosine_distance(embedding, $query::FLOAT[384])
```

Initial metrics should be limited to those DuckDB can execute consistently:
`cosine`, `l2`, and `inner_product`. The implementation must map each public
metric to one exact DuckDB function and one compatible index metric.

### Nearest-neighbor search

Top-k retrieval belongs on a DataFrame accessor because it changes row
cardinality and ordering.

```python
matches = documents.vector.search(
    query=query_embedding,
    column="embedding",
    metric="cosine",
    k=20,
    mode="exact",
    distance_column="_distance",
    tie_breaker="document_id",
)
```

It returns a lazy DataFrame containing the input columns plus the distance
column, ordered by distance and then by the optional tie-break key.

The initial API should require:

- a one-dimensional, finite numeric query vector;
- a supported fixed-size numeric array column;
- matching query and column dimensions;
- a positive integer `k`;
- a distance-column name that does not collide with an existing label;
- an existing, orderable tie-break column when deterministic ties are needed.

Null vectors and vectors containing non-finite values need an explicit policy.
The safest initial contract is to reject them before result production rather
than silently change the candidate population.

### Physical index management

Index creation is a physical, eager side effect and does not belong on a lazy
DataFrame transformation.

```python
documents.save_as_table("documents")

session.create_vector_index(
    table="documents",
    column="embedding",
    name="documents_embedding_idx",
    metric="cosine",
)
```

`create_vector_index()` is an eager, explicit side effect for a physical
main-schema DuckDB table with a fixed-size `FLOAT[n]` column. Inspection and
deletion are explicit; searches never create indexes.

## Search modes

Search strategy is part of result semantics and must not be hidden behind an
optimizer choice.

| Mode | Contract |
| --- | --- |
| `exact` | Exhaustively score the eligible input. This is the default. |
| `approximate` | Require a compatible physical index and verified indexed plan; fail if unavailable. |
| `auto` | Permit either strategy and report the selected strategy in `explain()` and `profile()`. |

Adding an HNSW index must not silently change an existing exact research
pipeline. DuckPD should therefore ship exact search first and add approximate
search only after it can verify index eligibility and report the chosen plan.

## Filtering is a semantic boundary

These pipelines answer different questions:

```python
# Nearest 20 among eligible documents.
eligible = documents[documents["available_at"] <= cutoff]
matches = eligible.vector.search(query=query, column="embedding", k=20)
```

```python
# Nearest 20 globally, followed by removal of future documents.
matches = documents.vector.search(query=query, column="embedding", k=20)
matches = matches[matches["available_at"] <= cutoff]
```

The second pipeline may return fewer than 20 rows and may omit eligible
neighbors. DuckPD's optimizer must preserve this distinction.

A typed `VectorSearchPlan` should own the input plan, query vector, vector
column, metric, `k`, mode, distance label, and tie-break metadata. An upstream
`FilterPlan` remains upstream for exact search. A backend may push that filter
into its retrieval operator only when the backend guarantees equivalent
prefilter semantics. Unsupported pushdown must fail or remain outside the
indexed path; it must never be silently interpreted as post-filtering.

## Ordering and determinism

Distance establishes a partial order. Equal distances do not establish stable
row order.

- Without `tie_breaker=`, the result is ordered by distance but ties are
  unspecified.
- With `tie_breaker=`, the result has a deterministic total order only when
  the key is itself sufficient to break ties.
- A downstream filter preserves the search order for retained rows.
- Joins follow DuckPD's existing rule and clear total ordering guarantees.

Approximate search may return a different candidate set across engine or index
versions. Reproducible workflows must record the DuckDB version, extension
version, index configuration, source fingerprint, metric, and search mode.

## Point-in-time retrieval

Temporal correctness is where vector retrieval can become unusually valuable
for DuckPD. A historical search must distinguish event time from availability
time: a document can describe an earlier event but only become knowable later.

A reproducible point-in-time retrieval contract needs:

- the document or revision availability timestamp;
- a cutoff for every query observation;
- source-dataset and embedding-model versions;
- a defined treatment of revisions and deletions;
- prefilter-before-top-k semantics.

The first vector release should support a scalar cutoff through ordinary
DataFrame filtering. Batched per-row cutoffs are a later feature and likely
need a dedicated temporal retrieval plan rather than a Python loop around
`vector.search()`.

## Backend strategy

The public API should describe semantics, not storage technology. Backends are
physical execution choices.

### Native DuckDB exact search

This is the first implementation target. It works on ordinary DuckDB tables,
Parquet-backed frames, and other compatible DuckPD sources without an optional
index or a new storage system. It provides the reference result against which
accelerated backends can be tested.

### DuckDB `vss`

The optional `vss` extension can accelerate eligible top-k query shapes with
HNSW indexes. Before DuckPD promises approximate mode it must qualify:

- compatibility with the supported DuckDB version;
- index eligibility for generated plans;
- index persistence and lifecycle behavior;
- filtering support and candidate semantics;
- memory use, because index memory may not follow ordinary spill guarantees;
- recall, latency, and reproducibility under documented configurations.

Responsibilities should remain separate:

| Component | Responsibility |
| --- | --- |
| DuckPD | DataFrame API, semantic validation, typed plans, ordering, observability |
| DuckDB | Relational execution, joins, aggregation, windows, sinks |
| DuckDB `vss` | Optional HNSW acceleration for DuckDB tables |

## Observability

Vector operations should strengthen DuckPD's existing execution visibility.
`explain()` and `profile()` should report:

- exact, approximate, or automatically selected execution;
- distance metric and vector dimension;
- candidate filter placement;
- requested `k` and tie-break ordering;
- physical index name and type when used;
- extension installation and loading boundaries;
- source and embedding-version provenance when available;
- peak memory and spill metrics without extending guarantees to index memory.

For `mode="approximate"`, inability to demonstrate a compatible indexed plan
is an error, not a performance warning.

### DuckDB 1.5.5 `vss` qualification

DuckPD installs and loads `vss` only from the explicit
`Session.create_vector_index()` boundary. The supported indexed subset is
in-memory, main-schema `FLOAT[n]` tables using `cosine` or `l2`. Index creation
rejects null, null-element, non-finite, and zero-norm cosine vectors.
`inner_product`, prefiltered inputs, tie-break ordering, persistent indexes, and
`mode="auto"` are rejected before retrieval.

Every approximate compilation checks the generated DuckDB physical plan for
both `HNSW_INDEX_SCAN` and the expected session-owned index name. Catalog
metadata alone is insufficient. `inspect_vector_indexes()` reports only indexes
created and still present in the current session.

The reproducible command
`uv run python -m benchmark.vector --rows 1000 --dimensions 16 --k 10 --vss`
was run on DuckDB 1.5.5. The deterministic smoke case reported exact/direct-SQL
parity, `1.0` recall@10, identical repeated approximate IDs, a verified
`HNSW_INDEX_SCAN`, 6.29 ms DuckPD exact latency, 1.73 ms direct DuckDB exact
latency, 57.14 ms index creation, 4.23 ms approximate latency, 187,793,408-byte
process peak RSS, and no exact-search spill. These are smoke observations, not
larger-than-memory or performance guarantees.

DuckDB documents `vss` as experimental. HNSW memory is outside DuckDB's
`memory_limit`; persistent custom indexes require an experimental setting and
have WAL recovery risks. DuckPD therefore does not enable persistence and makes
no bounded-memory, cross-version recall, or reproducibility promise for
approximate retrieval.

## Non-goals

The initial design does not include:

- automatic embedding generation or model downloads;
- hosted vector-database serving;
- implicit network requests during a DataFrame transformation;
- automatic index creation;
- silent exact-to-approximate substitution;
- claims that extension or index memory is bounded by DuckDB's spill settings;
- Python row-wise similarity functions;
- one API that hides materially different filter semantics across backends.

Users should supply precomputed embeddings initially. Embedding generation has
separate concerns around model selection, batching, devices, network access,
cost, versioning, and provenance.

## Delivery sequence

1. Add fixed-size vector type validation and lazy Series distance expressions.
2. Add exact `DataFrame.vector.search()` with deterministic tie handling and
   plan/explain coverage.
3. Test filtering order, nulls, dimensions, metrics, remote scans, direct
   sinks, and larger-than-memory exhaustive search.
4. Qualify DuckDB `vss`, then add explicit index lifecycle APIs and
   `mode="approximate"`.
5. Explore full-text, hybrid, and batched point-in-time retrieval only after
   vector semantics are stable.

## Acceptance gates

The first implementation is ready when:

- exact results match a trusted brute-force reference across metrics, ties,
  null cases, and filtered candidate sets;
- search remains lazy until an existing DuckPD execution boundary;
- filtering before and after search produces intentionally distinct plans;
- `explain()` identifies the strategy and all semantic inputs;
- downstream joins, grouping, Arrow streaming, and direct sinks work without
  special result types;
- approximate mode cannot run unless an eligible index is verified;
- memory and performance claims are backed by the benchmark harness.

The guiding rule is simple: retrieval belongs in DuckPD when it behaves like a
well-specified DataFrame operation and remains composable with the rest of the
analytical plan.
