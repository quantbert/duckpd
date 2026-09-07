# Vector Search and Text Embeddings

DuckPD provides three related retrieval workflows:

1. Search a vector column that already exists.
2. Embed text once, persist the vectors, and search them repeatedly.
3. Embed and search a filtered text corpus without first storing vectors.

All search operations return an ordinary lazy DuckPD `DataFrame`. You can
project, filter, join, aggregate, inspect, stream, or write the result using the
same APIs as any other DuckPD frame.

## Install

Vector search over existing vectors is part of the core package. Generating
text embeddings with the built-in local backend requires the optional
FastEmbed/ONNX dependency:

```bash
uv add "duckpd[embeddings]"
```

The built-in backend runs locally on `CPUExecutionProvider`. Preparing a model
may download model artifacts, but defining a model or building a lazy plan does
not.

## Five-minute example

Define a model with an immutable revision and its output dimension, prepare it
explicitly, embed the corpus, and write the result:

```python
import duckpd as pd

model = pd.embedding_model(
    "BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dimension=384,
)

with pd.connect() as session:
    session.prepare_embedding_model(model)
    news = session.read_parquet("news.parquet")

    embedded = news.embed_text(
        columns=["title", "description"],
        into="embedding",
        model=model,
        null_policy="empty",
    )
    embedded.write_parquet("news-embedded.parquet")
    stored = session.read_parquet("news-embedded.parquet")

    matches = stored.vector.search_text(
        "AI chip demand and revenue growth",
        column="embedding",
        model=model,
        k=10,
        tie_breaker="article_id",
    )

    print(matches[["article_id", "title", "_distance"]].collect())
```

`embed_text()` and `search_text()` are lazy. In this example,
`write_parquet()` and `collect()` are the execution boundaries. Calling
`session.prepare_embedding_model()` is eager and may perform network and disk
I/O.

## Choose a workflow

| Starting point and goal | API | Main tradeoff |
| --- | --- | --- |
| You already have vectors | `frame.vector.search()` | No model runtime is needed. |
| You need a distance for every row | `series.vector.distance()` | Scores all rows; it does not apply top-k. |
| You will search the same text corpus repeatedly | `frame.embed_text()` then `frame.vector.search_text()` | Pay document inference and storage cost once. |
| You need one search over a bounded candidate set | `frame.semantic.search()` | No stored vector column, but every execution re-embeds every candidate. |
| You need HNSW acceleration on a local table | `session.create_vector_index()` then `mode="approximate"` | Experimental and more restricted than exact search. |

For repeated queries, persisted embeddings are normally the right default.
Transient semantic search is most useful after a selective upstream filter.

## Search an existing vector column

### Exact top-k search

`DataFrame.vector.search()` ranks rows by distance and returns at most `k`
rows. Exact mode is the default and works with compatible table, Arrow, and
Parquet sources.

```python
matches = documents.vector.search(
    [0.12, -0.08, 0.44],
    column="embedding",
    metric="cosine",
    k=20,
    mode="exact",
    distance_column="_distance",
    tie_breaker="document_id",
)
```

The result contains all input columns plus `distance_column`, ordered from
nearest to farthest. Equal distances have unspecified order unless an adequate,
orderable `tie_breaker` is supplied.

Supported metrics are:

| Metric | Meaning | DuckDB operation |
| --- | --- | --- |
| `cosine` | Cosine distance; smaller is nearer | `array_cosine_distance` |
| `l2` | Euclidean distance; smaller is nearer | `array_distance` |
| `inner_product` | Negative inner product; smaller is nearer | `array_negative_inner_product` |

The query must be a non-empty, one-dimensional sequence of finite numeric
values. Boolean values are not accepted. Its dimension must match a fixed-size
column such as `FLOAT[384]` or `DOUBLE[384]`. Runtime-sized `FLOAT[]` and
`DOUBLE[]` lists can be searched exactly, but their dimensions are checked
during execution.

### Compute distance without top-k

Use `Series.vector.distance()` when the score belongs in a larger expression:

```python
scored = documents.assign(
    semantic_distance=documents["embedding"].vector.distance(
        [0.12, -0.08, 0.44],
        metric="l2",
    )
)
```

This remains lazy and preserves every row. Sorting or filtering the score is a
separate operation.

## Define and prepare an embedding model

An `EmbeddingModelSpec` describes an embedding space without loading a model:

```python
model = pd.embedding_model(
    "BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dimension=384,
    backend="fastembed",
    normalize=True,
    pooling="model-default",
    document_prefix="",
    query_prefix="",
)
```

The model name, revision, dimension, backend, normalization, pooling, and text
prefixes contribute to `model.fingerprint`. Matching dimensions alone do not
make vectors compatible: two 384-dimensional models can represent different
embedding spaces.

The revision must be non-empty and immutable. Mutable names such as `main`,
`master`, `latest`, and `head` are rejected. The dimension must be a positive
integer.

Preparation is explicit and session-scoped:

```python
prepared = session.prepare_embedding_model(model, cache_dir=".model-cache")

print(prepared.backend)
print(prepared.cache_path)
print(prepared.artifact_digest)
print(prepared.execution_providers)
```

DuckPD verifies the prepared artifact cache and refuses to use artifacts that
have changed since cache promotion. Inspect prepared models with:

```python
prepared_models = session.inspect_prepared_embedding_models()
```

Execution that needs an unprepared model fails with an instruction to call
`prepare_embedding_model()`.

## Create reusable corpus embeddings

`DataFrame.embed_text()` appends one non-nullable fixed-size `FLOAT[n]` column:

```python
embedded = documents.embed_text(
    columns=["title", "body"],
    into="embedding",
    model=model,
    batch_size=256,
    separator="\n\n",
    null_policy="error",
)
```

The selected columns must be `VARCHAR` columns. Their values are joined in the
given order using `separator`. The output name must be non-empty and must not
already exist.

Embedding is row-preserving and keeps the input row identity and ordering.
Inference is performed in bounded Arrow batches when the frame is executed;
DuckPD does not first collect the corpus into pandas. A materialization displays
embedding progress.

Null handling is explicit:

- `null_policy="error"` aborts if any selected component is null.
- `null_policy="empty"` replaces each null component with an empty string.
- Dropping rows as part of embedding is not supported.

For a reusable corpus, write the result directly instead of collecting it:

```python
embedded.write_parquet("documents-embedded.parquet")
embedded.save_as_table("documents_embedded")
```

DuckPD preserves embedding model metadata through supported projection,
renaming, row-wise concatenation, local Parquet round trips, and session-owned
table round trips. Replacing or calculating the vector column clears that
identity because DuckPD can no longer prove that the values belong to the same
embedding space.

### Search persisted embeddings with text

`DataFrame.vector.search_text()` embeds one query and performs exact top-k
search against a verified embedding column:

```python
stored = session.read_parquet("documents-embedded.parquet")

matches = stored.vector.search_text(
    "renewable energy investment",
    column="embedding",
    model=model,
    metric="cosine",
    k=10,
    distance_column="_distance",
    tie_breaker="document_id",
)
```

The column must carry embedding metadata with the same fingerprint as `model`.
DuckPD rejects missing or mismatched metadata instead of assuming that equal
dimensions imply compatibility. `search_text()` is exact; it does not expose an
approximate mode.

### Embed a query explicitly

For lower-level control, embed one query eagerly and pass its values to ordinary
vector search:

```python
query = session.embed_query("renewable energy investment", model=model)

matches = stored.vector.search(
    query.values,
    column="embedding",
    metric="cosine",
    k=10,
)
```

`EmbeddedQuery` retains `model_fingerprint` for application-level checks. The
lower-level `vector.search()` accepts numeric values and validates dimension,
but does not itself verify their model identity.

## Search text without storing embeddings

`DataFrame.semantic.search()` embeds the query and every eligible document,
then retains an exact top-k result:

```python
eligible = documents[
    (documents["language"] == "en")
    & (documents["available_at"] <= observation_time)
]

matches = eligible.semantic.search(
    "renewable energy investment",
    columns=["title", "body"],
    model=model,
    metric="cosine",
    k=10,
    batch_size=256,
    separator="\n\n",
    null_policy="empty",
    distance_column="_distance",
    tie_breaker="document_id",
)
```

This operation is lazy and streams documents in bounded batches. The query
embedding is cached for that plan in the session, but every execution re-embeds
every candidate document. Approximate search is unavailable because there is no
persistent indexed corpus.

## Filter placement changes the question

A filter before top-k limits the candidate set. A filter after top-k only
removes rows from the already selected result:

```python
# Find the nearest 20 among documents available at the observation time.
eligible = documents[documents["available_at"] <= observation_time]
correct = eligible.vector.search(query.values, column="embedding", k=20)

# Find the global nearest 20, then discard unavailable documents.
# This can return fewer than 20 rows and omit eligible neighbors.
global_top_k = documents.vector.search(query.values, column="embedding", k=20)
different = global_top_k[global_top_k["available_at"] <= observation_time]
```

The same distinction applies to `semantic.search()`. DuckPD preserves this plan
boundary, which is especially important for point-in-time and leakage-sensitive
workflows.

## Experimental approximate search

Approximate search uses DuckDB's experimental `vss` extension and an HNSW
index. Index creation is an eager, explicit session operation:

```python
embedded.save_as_table("documents_embedded")

index = session.create_vector_index(
    table="documents_embedded",
    column="embedding",
    name="documents_embedding_hnsw",
    metric="cosine",
)

table = session.table("documents_embedded")
matches = table.vector.search(
    query.values,
    column="embedding",
    metric="cosine",
    k=20,
    mode="approximate",
)
```

DuckPD verifies that the physical plan contains `HNSW_INDEX_SCAN` and names the
expected session-owned index. It does not silently fall back to exact search.

The current approximate path requires all of the following:

- an unfiltered local DuckDB table in the main schema;
- a fixed-size `FLOAT[n]` column;
- exactly one compatible index registered in the current session;
- `cosine` or `l2`, not `inner_product`;
- no `tie_breaker`;
- non-null, finite vectors, and non-zero norms for cosine.

There is no `mode="auto"`. Persistent HNSW indexes and prefiltered approximate
search are not supported. HNSW memory is outside DuckDB's `memory_limit`, and
DuckPD does not promise bounded memory, exact recall, or cross-version
reproducibility for this experimental path.

Inspect and remove indexes explicitly:

```python
indexes = session.inspect_vector_indexes()
session.drop_vector_index(index.name)
```

## Custom embedding providers

Applications can supply a session-owned provider for internal models or other
runtimes. Set `backend="custom"`, implement `TextEmbeddingProvider`, register
the provider, and then prepare it:

```python
custom_model = pd.embedding_model(
    "internal/legal-search",
    revision="sha256:immutable-artifact-digest",
    dimension=768,
    backend="custom",
)

provider = MyEmbeddingProvider(custom_model)
session.register_embedding_provider(custom_model, provider)
session.prepare_embedding_model(custom_model)
```

A provider exposes its exact `specification`, `prepare()`,
`embed_documents(texts)`, and `embed_query(text)` methods. Document output must
be a `pyarrow.FixedSizeListArray` of `float32` values with the declared row
count and dimension. DuckPD validates every batch for nulls, non-finite values,
shape, and requested normalization. Query output must carry the matching model
fingerprint.

Provider objects belong to a session and are not serialized into logical plans.
Hosted providers, credentials, retries, billing, and remote privacy guarantees
are not built into the current qualified backend.

## Validation and failure behavior

DuckPD validates metadata-level errors before source execution whenever
possible. Data-dependent failures happen inside the execution query and abort
the operation rather than silently removing invalid rows.

Important constraints include:

- Vector source columns must be `FLOAT[n]`, `DOUBLE[n]`, `FLOAT[]`, or
  `DOUBLE[]`.
- Query values and source vectors must be finite; cosine vectors must have
  non-zero norm.
- Null vectors, null vector elements, wrong dimensions, and invalid provider
  batches fail materialization.
- `k` and embedding `batch_size` must be positive integers.
- A distance-column label must be non-empty and cannot replace an existing
  column.
- A tie-breaker must exist and have an orderable scalar type.
- Query text, source text, and credentials are redacted from explain output.
- A failed direct Parquet embedding write does not leave a partial output or
  replace a valid previous file.

## Explain and profile

Inspect plans before execution:

```python
print(matches.explain("logical"))
print(matches.explain("physical"))
print(matches.explain("json"))
```

Vector explain output includes the metric, dimension, `k`, strategy, filter
placement, tie-breaker, and physical index use. Embedding explain output includes
the model fingerprint, columns, batch size, null policy, preparation state, and
whether search is persisted or transient. Raw query text is redacted.

Execute with profiling when inference or retrieval cost matters:

```python
profile = matches.profile()
```

Embedding profile metrics include source and document rows, text bytes, batch
count, preparation and inference timing, throughput, prepared-model count, and
provider retries where available.

## Execution summary

| Operation | Eager or lazy | Repeats document inference? |
| --- | --- | --- |
| `pd.embedding_model()` | Planning only | No |
| `session.prepare_embedding_model()` | Eager | No document inference |
| `session.embed_query()` | Eager | Query only |
| `frame.embed_text()` | Lazy | On each materialization |
| `series.vector.distance()` | Lazy | No |
| `frame.vector.search()` | Lazy | No |
| `frame.vector.search_text()` | Lazy | Query once per plan, then cached in the session |
| `frame.semantic.search()` | Lazy | Yes, every candidate on each materialization |
| `session.create_vector_index()` | Eager | No |

## Current scope

DuckPD currently supports local FastEmbed/ONNX CPU inference, custom providers,
dense float32 embeddings, exact vector search, exact text search, and guarded
in-memory HNSW search. It does not currently provide GPU execution, a qualified
hosted provider, automatic model selection, corpus caching or refresh,
reranking, hybrid lexical/vector retrieval, sparse or multi-vector embeddings,
quantized storage, distributed inference, or batched per-row temporal queries.

For the implementation rationale and deeper execution contracts, see
[Text Embeddings and Semantic Search](../design/text-embeddings.md) and
[Vector Search and Analytical Retrieval](../design/vector-search.md).