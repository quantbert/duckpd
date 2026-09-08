# Text Embeddings and Semantic Search

**Status: implemented in Phase 14.**

## Product decision

Text embedding should be a first-class DuckPD API, not SQL assembled by users and
not an implicit behavior of vector search. DuckPD should support two explicit
workflows:

1. stream a text corpus through an embedding model and persist a reusable
   fixed-size vector column;
2. stream a filtered candidate set through the same model and retain only an
   exact top-k result for one-off exploration.

The first workflow is the default for repeated retrieval. The second is useful
for bounded candidate sets and demos, but recomputes every document embedding on
every search. Streaming bounds memory; it does not reduce inference work.

Embedding execution is a hybrid boundary between DuckDB and a model runtime.
DuckPD remains responsible for lazy planning, Arrow batching, backpressure,
metadata, resource reporting, direct sinks, and vector retrieval. The model
provider remains responsible for tokenization and inference.

## Goals

- Replace hand-written embedding SQL with typed DuckPD operations.
- Keep planning side-effect free: no model download, network request, or
  inference before execution.
- Stream source rows and generated vectors in bounded Arrow batches without a
  pandas materialization.
- Produce ordinary fixed-size `FLOAT[n]` columns accepted by
  `Series.vector.distance()`, `DataFrame.vector.search()`, and DuckDB HNSW
  indexes.
- Make model identity and query/document encoding compatibility verifiable.
- Support local CPU inference without adding a required model-runtime
  dependency to core DuckPD.
- Keep transient semantic search explicit about repeated inference cost.
- Preserve DuckPD filter placement, ordering, row identity, error, and direct
  sink contracts.

## Non-goals for the first release

- Choosing a model automatically from the input text or query.
- Downloading model weights implicitly during planning or execution.
- Training, fine-tuning, reranking, or evaluating embedding models.
- Silently embedding an arbitrary string passed to `vector.search()`.
- Automatic corpus caches, cache invalidation, or incremental refresh.
- Approximate search over transient, unpersisted embeddings.
- Distributed inference or a general-purpose Python UDF framework.
- Float16, integer-quantized, sparse, or multi-vector document storage.
- Full-text, lexical, or hybrid score fusion.

## Public API

### Model specification

A model is an immutable, side-effect-free specification:

```python
import duckpd as pd

model = pd.embedding_model(
    "BAAI/bge-small-en-v1.5",
    backend="fastembed",
    revision="immutable-model-revision",
    dimension=384,
    normalize=True,
    document_prefix="passage: ",
    query_prefix="query: ",
)
```

The initial constructor should require an immutable revision and a declared
positive dimension. DuckPD may provide reviewed presets whose dimension,
prefixes, pooling, and normalization are known, but must not infer these values
by downloading model metadata during planning.

Local model acquisition is an explicit eager operation:

```python
session.prepare_embedding_model(model)
```

Preparation may download and validate weights. It must report its cache path,
content digest, backend, and execution providers. Execution fails before the
source scan if required local artifacts are unavailable or do not match the
model specification.

### Reusable corpus embeddings

```python
embedded = news.embed_text(
    columns=["title", "description"],
    into="embedding",
    model=model,
    batch_size=256,
    separator="\n\n",
    null_policy="error",
)
```

`embed_text()` returns a lazy DataFrame with the input columns plus a
non-nullable `FLOAT[dimension]` column. It performs no inference until an eager
consumer runs.

The primary execution path is a direct sink:

```python
embedded.write_parquet("stock-news-embedded.parquet")
```

The source scan, text assembly, inference, and Parquet writer operate with
bounded batches. DuckPD must not concatenate the corpus into one Arrow table.

The first release supports an ordered sequence of string columns. For each row,
DuckPD applies the declared null policy and joins values with `separator`. An
arbitrary Python formatting callback is intentionally unsupported because it
would create another opaque execution boundary.

### Query text over persisted embeddings

```python
matches = embedded.vector.search_text(
    "AI chip demand and revenue growth",
    column="embedding",
    model=model,
    metric="cosine",
    k=10,
    tie_breaker="id",
)
```

`vector.search_text()` embeds exactly one query, validates the resulting vector
against the column's embedding metadata, and executes the existing exact
DuckDB distance/top-k path. Text search intentionally has no approximate or
automatic mode in this release.

The lower-level form remains available:

```python
query_vector = session.embed_query(
    "AI chip demand and revenue growth",
    model=model,
)
matches = embedded.vector.search(query_vector.values, column="embedding", k=10)
```

`Session.embed_query()` is eager because it performs model inference. It returns
a typed immutable vector carrying the model fingerprint, not a bare list that
loses compatibility metadata.

### Transient streaming semantic search

```python
matches = news.semantic.search(
    "AI chip demand and revenue growth",
    columns=["title", "description"],
    model=model,
    metric="cosine",
    k=10,
    batch_size=256,
    tie_breaker="id",
)
```

This operation is lazy and returns an ordinary DuckPD DataFrame. At execution,
DuckPD embeds the query once per plan, streams eligible documents through a
bounded Arrow UDF, and lets DuckDB's exact top-k operator retain the result.
The ordinary result remains composable with downstream relational operations.

Transient search never selects approximate mode: no persistent corpus or
compatible physical index exists. Its explain output states that every
candidate document will be embedded again on the next execution.

Filtering before and after transient search retains the same semantic
distinction as vector search:

```python
# Embed and rank only historically eligible articles.
eligible = news[news["available_at"] <= cutoff]
matches = eligible.semantic.search(query, columns=["title", "body"], model=model, k=20)

# Embed and rank globally, then remove future articles. This may return fewer rows.
matches = news.semantic.search(query, columns=["title", "body"], model=model, k=20)
matches = matches[matches["available_at"] <= cutoff]
```

The optimizer must not move either filter across the semantic top-k boundary.

## Provider contract

Core DuckPD should define a small provider protocol using Arrow-compatible
values. Provider packages may depend on ONNX Runtime, FastEmbed,
SentenceTransformers, a hosted API, or an internal inference service.

```python
class TextEmbeddingProvider(Protocol):
    @property
    def specification(self) -> EmbeddingModelSpec: ...

    def prepare(self) -> PreparedModelInfo: ...

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> pyarrow.FixedSizeListArray: ...

    def embed_query(self, text: str) -> EmbeddedQuery: ...
```

Provider output is accepted only when it has the declared row count, dimension,
`FLOAT` element type, finite values, and required normalization. DuckPD validates
every batch before it enters a sink or distance calculation.

The provider is configured on the session or model specification; arbitrary
provider objects are not serialized inside logical plans. Plans carry an
immutable provider/model identifier resolved by the owning session.

Embedding execution owns a `tqdm` progress bar. `DataFrame.embed_text()` remains
lazy; every materialization derives the input row count, reports documents only
after validated provider calls, and closes the bar on success or failure.
Applications never decorate or replace providers to obtain standard progress.

### Local backends

The default optional backend remains FastEmbed using ONNX Runtime on
`CPUExecutionProvider`:

```bash
uv add "duckpd[embeddings]"
```

`TransformersEmbeddingProvider` adds explicit PyTorch CPU, NVIDIA CUDA, and AMD
ROCm execution. Accelerator-specific PyTorch builds remain application-owned:
DuckPD neither installs nor replaces them, and an explicitly requested GPU
never falls back to CPU. The Transformers model identity requires explicit
`cls` or `mean` pooling and has a distinct backend fingerprint from FastEmbed.

Core DuckPD remains importable and fully functional without either optional
runtime. A missing backend raises an actionable unsupported-operation error
before source execution.

A remote provider requires separate qualification for credentials, retries,
partial batches, rate limits, billing, privacy, and reproducibility. Remote text
transmission must never be inferred from a model name or enabled implicitly.

## Logical plans and execution boundaries

### `EmbeddingPlan`

A typed embedding plan carries:

- input plan;
- ordered text column IDs;
- output column metadata;
- immutable model fingerprint;
- batch size and separator;
- null and invalid-output policies;
- document encoding mode.

It is row preserving. It preserves input order and row identity, appends a
non-nullable `FLOAT[n]` column, and records an `embed_text` provenance
transformation. DuckDB invokes the session-owned provider at an explicit Arrow
UDF boundary; the provider subdivides every DuckDB chunk into the declared
maximum inference batch size.

### `SemanticSearchPlan`

A typed transient search plan carries:

- input plan and text column IDs;
- opaque session query key (raw query text is not serialized);
- model fingerprint;
- metric and `k`;
- batch size;
- distance label;
- optional deterministic tie-break column.

It is not row preserving. Its output ordering and metadata follow
`VectorSearchPlan`. Exact top-k selection must be differential-tested against
persisting the same embeddings and calling `vector.search()`.

### Backpressure and cancellation

The executor must own one bounded pipeline:

```mermaid
flowchart LR
    D[DuckDB Arrow reader] --> T[Text assembly]
    T --> M[Model batch inference]
    M --> V[Validated FLOAT n batch]
    V --> S[Parquet sink or exact top-k]
```

The inference batch size is bounded by the plan even when DuckDB supplies a
larger Arrow chunk. Closing an Arrow reader cancels the owning DuckDB query;
provider failures abort execution. Parquet output is written to a same-directory
staging file and atomically promoted only after successful execution.

## Model identity and persisted metadata

Dimension equality is insufficient. Two models can both produce 384-element
vectors in incompatible spaces. DuckPD therefore computes a stable embedding
fingerprint covering at least:

- model identifier and immutable revision or artifact digest;
- tokenizer identifier and revision;
- pooling strategy;
- output dimension and element type;
- normalization;
- document and query prefixes;
- provider and format compatibility versions.

`FrameMetadata` should associate this specification with the embedding
`ColumnId`. Projection and renaming preserve it; arithmetic or replacement
clears it.

Single-file Parquet output persists the specification in an atomically written
DuckPD sidecar manifest. DuckDB table output uses a session catalog entry.
`read_parquet()` recovers a local file's sidecar across sessions and `table()`
recovers table metadata in the owning session.

`vector.search_text()` rejects missing or mismatched model metadata by default.
A deliberate `allow_unverified_model=True` escape hatch may be considered only
if the plan and profile expose the loss of verification.

## Nulls, errors, and determinism

Initial policies should be narrow:

- `null_policy="error"` is the default and aborts on a null selected text value;
- `null_policy="empty"` replaces null components with empty strings without
  dropping rows;
- dropping rows during `embed_text()` is unsupported because it would violate
  the row-preserving contract;
- empty assembled documents are valid only if the provider returns a finite,
  non-zero vector for the selected metric;
- invalid provider shape, nulls, NaNs, infinities, and zero cosine norms abort
  before result production;
- a model fingerprint and identical input text must produce reproducible output
  within the provider's documented tolerance;
- deterministic search still requires an adequate tie-break key.

The initial local backend should use deterministic inference settings. Hosted
providers that cannot promise reproducibility must say so in model metadata and
profile output.

## Explain and profile

`explain()` must report, without exposing text or credentials:

- operation: corpus embedding or transient semantic search;
- backend and model fingerprint;
- dimension, normalization, and text column labels;
- batch size and null policy;
- materialization boundary;
- filter placement;
- exact strategy and top-k for transient search;
- whether corpus embeddings are persisted, transient, or metadata-unverified;
- whether model preparation is complete.

`profile()` additionally records:

- source rows and text bytes;
- token count when exposed cheaply by the provider;
- inference batches and rows;
- model preparation time separately from inference time;
- inference wall time and rows/second;
- peak process RSS and provider/device memory when measurable;
- sink bytes or retained top-k size;
- provider retries and remote requests, if applicable.

Raw document text, query text, access tokens, hosted endpoint credentials, and
local cache secrets must be redacted.

## Performance expectations

Embedding 2.52 million 384-dimensional float32 vectors produces approximately
3.87 GB of raw vector values before Parquet encoding and other columns:

```text
2,520,000 rows * 384 values * 4 bytes = 3,870,720,000 bytes
```

Batching can keep memory bounded but cannot make that inference interactive.
The implementation must benchmark representative short and long text across
batch sizes before choosing defaults or making throughput claims.

Transient semantic search is appropriate when an upstream filter bounds the
candidate set or when a one-off scan is acceptable. Repeated search should
persist embeddings once. Explain output should warn when estimated or known
candidate counts exceed a qualified transient-search threshold, but the first
release should not silently change strategy.

## Security and supply chain

- Model revisions or artifact digests are mandatory; mutable tags are rejected.
- Model preparation verifies downloaded content before cache promotion.
- Cache writes are atomic and do not follow unsafe symlinks.
- Hosted providers require explicit construction and credential-safe explain
  output.
- Sending source text to a remote provider requires an explicit remote model
  specification; DuckPD never falls back from local to hosted inference.
- Provider code executes with the Python process's privileges and is therefore
  trusted application code, not a sandbox boundary.

## Testing and qualification

Permanent tests should cover observable contracts:

- planning triggers no preparation, source execution, or inference;
- batch output matches direct provider output and preserves row order;
- bounded batches work across empty input, final partial batches, null policy,
  cancellation, and provider failure;
- fixed-size vector metadata survives projection, rename, Parquet round trips,
  and table persistence;
- model mismatches fail before vector retrieval;
- transient top-k matches persisted-embedding exact search for every metric,
  ties, filters, and empty input;
- downstream projection, join, grouping, Arrow streaming, and direct sinks
  compose without pandas materialization;
- provider and query text secrets are absent from explain/profile output;
- optional embedding dependencies are absent from the core installation.

Qualification benchmarks should measure local CPU inference first, then any GPU
or hosted backend separately. Results must include model revision, hardware,
text-length distribution, rows, tokens, batch size, wall time, peak RSS, output
size, and direct-provider parity.

## Delivered qualification

Phase 14 delivered the model, provider, prepared-artifact, query, column
metadata, `EmbeddingPlan`, and `SemanticSearchPlan` contracts; atomic Parquet
and transactional table sinks; eager query embedding; persisted and transient
exact search; redacted explain output; profile metrics; differential tests; a
local CPU benchmark; and the AlphaDojo remote-news demo.

The default qualified backend is FastEmbed 0.7.x with ONNX Runtime
`CPUExecutionProvider`, model `BAAI/bge-small-en-v1.5`, immutable revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, and dimension 384. The same model
is qualified through PyTorch Transformers on AMD ROCm with explicit CLS
pooling, bounded batches, and normalized float32 output. Qualification records
the downloaded artifact digest rather than treating the model name as identity.

Hosted providers, automatic corpus caches and refresh, multi-file dataset
writers, hybrid lexical/vector retrieval, reranking, quantization, sparse or
multi-vector storage, and distributed inference remain explicitly deferred.
Repeated searches should persist embeddings once; the transient API
deliberately exposes its repeated document-inference cost.
