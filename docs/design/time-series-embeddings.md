# Time-Series Embeddings and Event Similarity

**Status: proposed.**

This document proposes new behavior; it does not describe an
implemented time-series embedding API.

**Intended location:** `docs/design/time-series-embeddings.md`.

Companion designs: [Text Embeddings and Semantic Search][design-text],
[Vector Search][design-vector], [Feature Store Architecture][design-store], and
[Feature Store Embedding Metadata and Automatic Model Preparation][design-catalog].
The last document is itself a proposal. Its catalog-version-1 text embedding
declarations and automatic preparation behavior must not be treated as shipped dependencies.

## Product decision

Time-series similarity should be an extension of DuckPD's typed DataFrame and
vector operations, not a separate database, a forecasting subsystem, or a
training framework.

DuckPD should support three explicit steps:

1. Construct aligned, ordered, fixed-length windows from numeric series.
2. Transform each window into an ordinary `FLOAT[n]` representation, using either
   deterministic DuckDB expressions or an explicitly selected optional encoder.
3. Retrieve comparable representations through the existing vector distance and
   exact top-k machinery.

An event can attach text, return-pattern, and volatility representations to the
same stable event key. These representations support historical analogy and
pattern analysis. Forecasting accuracy, trading signals, and predictive alpha
are not the product objective.

Independent text and time-series embeddings remain separate spaces. Event-level
retrieval can combine their **scores**, but it must not compare their vector
coordinates. Direct text-to-series retrieval requires a separately qualified,
jointly aligned model and is a later extension.

The first useful release must work without downloading any model. Ordered return
windows, explicitly normalized when appropriate, already provide interpretable
pattern retrieval. Learned encoders are an optional improvement to evaluate, not
a prerequisite for the feature.

## Goals

- Find historical windows with similar price-return shape, momentum direction,
  volatility shape, or volatility level, with the meaning of similarity declared.
- Support news-to-event, reaction-to-event, and combined event similarity without
  conflating those workflows with shared-space multimodal retrieval.
- Keep window construction relational and model inference Arrow-batched, without
  pandas fallback or whole-corpus Python materialization.
- Produce reusable fixed-size numeric columns compatible with existing vector
  operations and, where supported, existing physical HNSW indexes.
- Preserve grouping, ordering, row identity, laziness, filter placement, and
  direct-sink contracts.
- Record enough representation metadata to reject incompatible queries even
  when vector dimensions happen to match.
- Make feature-family catalog integration declarative and versioned.
- Distinguish retrospective event analysis from retrieval constrained by what was
  available at a historical instant.
- Make implementation boundaries, unsupported cases, tests, and rollout gates
  explicit enough for independent implementation work.

## Non-goals

The initial scope does not include model training, forecasting, backtesting,
news causality attribution, market-calendar inference, automatic resampling,
unbounded dynamic-time-warping search, arbitrary Python window callbacks,
distributed inference, automatic feature refresh, or a new vector database.

It also does not include automatic model selection, embedding numeric values
with a general text model, implicit model downloads, automatic HNSW creation,
ANN search over arbitrary filtered DataFrames, or a general multimodal fusion
accessor. Ordinary relational expressions are sufficient for initial late
fusion.

## Existing implementation and integration baseline

The following are source observations, not proposed additions.

| Area | Current behavior | Consequence for this design |
| --- | --- | --- |
| Architecture | Immutable logical plans and metadata; compilation and execution are separate. No silent pandas fallback. | Add typed nodes and expressions, not SQL stored on public objects. |
| Windows | Integer and fixed-duration rolling aggregates exist; fixed-count `Rolling.to_array()` and `GroupedRolling.to_array()` emit typed, nullable `FLOAT[n]` windows. | Reuse the series-window metadata and grouped alignment path when implementing representations. |
| Text embeddings | `EmbeddingModelSpec`, providers, explicit preparation, Arrow inference, `embed_text()`, and fingerprint-checked `search_text()` exist. | Reuse the lifecycle pattern, not text-specific preprocessing or fingerprints. |
| Numeric vectors | Exact cosine, L2, and negative-inner-product ranking exist. Numeric queries are currently plain sequences. | Reuse distance/top-k lowering and add typed-query acceptance explicitly. |
| Approximate search | Requires a compatible session-owned HNSW index and a plain local table scan. Upstream filters and tie-breakers are rejected. | Do not promise filtered event ANN retrieval or a new automatic search mode. |
| Metadata | A `Column` currently has text-specific `embedding` metadata. | Introduce additive series metadata without rehashing legacy text models. |
| Persistence | Local Parquet embedding sidecars and session-managed table embedding metadata exist. | Extend versioned persistence; do not assume metadata survives every external tool or session reopen. |
| Feature store | Catalog parser accepts version 1. Timeseries feature planning initially uses `UNKNOWN` column types. | Catalog series search needs declared logical vector types plus physical validation. |
| Catalog embedding design | Catalog-version-1 text embedding declarations and automatic preparation are proposed. | Series catalog work has an explicit dependency, not an assumed capability. |

These observations are grounded in the architecture decisions and source for
[windows][src-window], [embeddings][src-embeddings], [vectors][src-vector],
[logical plans][src-logical], [session/persistence][src-session], and
[feature-store planning][src-store]. The design follows the existing
[architecture directives][adr-architecture].

Source review also identified integration details that require deliberate work:
`assign()` constructs replacement columns; feature-store alias and ASOF payload
construction also reconstruct columns. Their metadata propagation cannot be
assumed to preserve new representation fields. The current optimizer recursively
handles named binary plans and otherwise assumes a unary `input`; an event-window
node therefore needs explicit traversal support. The current embedding progress
path counts its input before processing it; that is unsuitable as an automatic
extra scan of an expensive series-window pipeline. These are implementation
requirements below, not claims that they have already been fixed.
[Sources: frame][src-frame], [feature store][src-store],
[optimizer][src-optimizer], [executor][src-executor].

## Terminology and data model

A **window** is an ordered sequence of `L` numeric observations for each of `C`
channels. Channel order and temporal order are part of its meaning. Its logical
shape is `(C, L)`; it is represented in a DataFrame as `C` separate fixed-size
array columns, not as an untyped Python object.

A **representation** is one `FLOAT[D]` vector derived from a complete window.
For the native encoder, `D = C * L`. For a learned encoder, `D` is declared in the
model specification and validated against actual output.

A **representation space** is the complete encoding contract: input semantics,
sampling, channel order, length, preprocessing, encoder identity, pooling, and
output normalization. Equal dimension alone does not imply equal space.

A **feature family** is a catalog organization, not an encoding algorithm.
`returns:shape_60`, `returns:momentum_60`, and `volatility:level_60` can be separate
features in their respective families. They need not share an encoder or space.

An **event observation** identifies one event revision affecting one entity. An
application should provide a stable `event_observation_id`, or another explicit
unique key, for that grain. A news item affecting three securities produces three
event observations if each has a different market reaction.

A **reaction window** is a window attached to an event observation with explicit
start, end, and data-availability timestamps. It describes an association with
an event; it is not evidence that the event caused the observed move.

The resulting architecture is:

```text
ordered numeric rows ---> rolling fixed-count arrays ---+
                                                       |
events + numeric rows ---> event-aligned arrays --------+--> representation
                                                       |    native SQL or
                                                       |    optional encoder
                                                       v
                                                 FLOAT[D] column
                                                       |
                                          existing vector distance/top-k

news ---> existing text encoder ---> text vector ------+
                                                       |
event key + text/reaction/volatility scores ------------+--> late-fused ranking
```

## Similarity semantics

### Shape is not level, direction is not volatility

Normalization must never be inferred from a family name or applied universally.
The recipe must answer what should remain distinguishable.

| Intended comparison | Suitable initial representation | Information intentionally retained or removed |
| --- | --- | --- |
| Correlated return fluctuations | Center each return window and unit-normalize it | Removes mean return and magnitude; preserves signed shape |
| Momentum path | Uncentered cumulative returns, prepared upstream, with no automatic scale normalization | Retains direction and path magnitude |
| Relative volatility shape | Centered/unit-normalized absolute-return or realized-volatility trajectory | Retains relative bursts; removes absolute volatility level |
| Volatility level and evolution | Uncentered volatility trajectory with fixed units; use L2 | Retains amplitude; distance depends on the declared units |
| Structural pattern | Learned encoder with a qualified recipe | Meaning is empirical and must be evaluated |

For two nonconstant, equally sized vectors, cosine similarity after mean
centering equals Pearson correlation. Unit normalization is convenient but not
required for this identity:

```text
xc = x - mean(x)
yc = y - mean(y)
cosine_similarity(xc, yc) = sum(xc * yc) / (norm(xc) * norm(yc)) = corr(x, y)
```

Thus cosine **distance** is `1 - correlation` for this specific representation.
It is not a general interpretation of learned-embedding distance. Constant
windows have undefined centered cosine similarity and require explicit handling.
Negative correlation is not treated as a match by default; taking an absolute
correlation would deliberately make opposite-direction patterns similar.

Absolute returns are a volatility proxy, not interchangeable with every
realized-volatility estimator. Definitions, units, return convention, adjustment
policy, and any upstream clipping or transforms belong in a versioned input
contract. DuckPD must not infer them from a column label.

Keep scalar descriptors such as cumulative return, realized volatility, duration,
and liquidity alongside embeddings. They provide explicit filters and help
interpret neighbors. A shape vector should not be forced to carry information
its normalization deliberately removes.

## Public API

All APIs introduced in this section are **proposed**. Existing operations such
as `read_parquet`, `assign`, `groupby`, `sort_values`, `write_parquet`, and raw
numeric `vector.search` retain their existing contracts.

### Fixed-count rolling arrays

Add `to_array()` to fixed-count rolling operations:

```python
import duckpd as pd

session = pd.connect()
prices = session.read_parquet(
    "prepared-returns.parquet",
    order_by=["ticker", "timestamp", "observation_id"],
)
windows = prices.assign(
    return_window=lambda df: (
        df.groupby("ticker")["return_1m"].rolling(60, min_periods=60).to_array()
    ),
)
```

`return_window` has logical type `FLOAT[60]` and is nullable during warm-up or when
any required observation is null. Each non-null array contains exactly 60 finite
values, oldest first. The operation does not normalize, pad, resample, or create
an embedding-space claim.

Proposed signature:

```text
Rolling.to_array() -> Series | DataFrame
GroupedRolling.to_array() -> Series | DataFrame
```

The result follows the receiver: a selected Series returns a Series; an explicit
selection of numeric DataFrame columns returns one array column per selected
column. Grouped standalone results retain the existing grouped index contract;
assignment back to the same source uses the existing grouped alignment path.
Implementing only an inherited ungrouped method is insufficient.

Initial constraints:

- `window` must be a positive integer and `min_periods` must equal `window`.
  Weaker `min_periods` values are rejected for `to_array()`, not reinterpreted.
- `center=True`, time-range windows, variable-length windows, and implicit
  padding are unsupported. Existing scalar rolling behavior is unchanged.
- Selected values must be numeric and convertible to finite float32 values.
  Boolean, string, struct, and nested-array inputs are rejected.
- A whole null window is valid output; a non-null array with null children is not.
  Nonfinite observations and float32 overflow raise an execution error.
- Ordering is the receiver's explicit guaranteed order. This is an observation
  window, not an assertion that rows are equally spaced in wall-clock time.
- Null group keys follow the existing `groupby(dropna=...)` and assignment
  semantics. No window may cross a group boundary.

Use an explicit chronological sort/order declaration for time-series work.
Physical file order is not proof of chronology. Equal timestamps require an
application tie-breaker such as `observation_id`; otherwise the ordering contract
must explicitly permit ties rather than promise reproducible windows.

A return transform consumes its own history. Sixty valid one-period returns
usually require sixty-one source prices. The window operator must not silently
synthesize the first return or discard a missing observation and shift the rest.

### Representation specification

Add an immutable specification for both native and learned representations:

```python
return_shape = pd.series_representation(
    window=60,
    channels=("simple_return",),
    sampling="observations",
    step="PT1M",
    data_contract="market/simple-return/split-adjusted/v1",
    normalization="center",
    unit_norm=True,
    zero_scale="null",
)
```

Proposed constructor:

```text
series_representation(
    *,
    window: int,
    channels: tuple[str, ...],
    sampling: Literal["observations", "fixed_grid"],
    data_contract: str,
    step: str | None = None,
    normalization: Literal["none", "center", "zscore"] = "none",
    unit_norm: bool = False,
    zero_scale: Literal["error", "null"] = "error",
    encoder: SeriesEmbeddingModelSpec | None = None,
) -> SeriesRepresentationSpec
```

`encoder=None` selects the native deterministic encoder. No model runtime is
needed. The specification exposes read-only `dimension` and `fingerprint`
properties and versioned `to_dict()` / `from_dict()` methods. Both
specification factories reject boolean values where integer lengths/dimensions
are required. API examples are fragments of an application-owned session; close
that session after its eager consumers finish.

`channels` contains unique semantic channel names, not physical DataFrame column
labels. `data_contract` is a nonempty, versioned identifier for the upstream
numeric meaning. Neither field independently verifies data correctness.

For `sampling="observations"`, `step` is an optional nominal cadence declaration.
It does not claim that a 60-observation window spans exactly 60 minutes. For
`sampling="fixed_grid"`, a positive fixed duration is required and all positions
must lie on that grid. Calendar months and exchange-session offsets are not
accepted as fixed durations. Persisted or explicitly attached window contracts
must distinguish these sampling modes.

The series-window metadata produced by `event_windows()` can verify a grid
claim. Bare arrays or rolling observation arrays cannot independently prove one;
assigning a representation specification to external arrays is an explicit
publisher assertion, recorded as such in provenance.

The initial release has no general non-event fixed-grid window constructor.
Verified `fixed_grid` metadata is produced only by `event_windows()`. Existing
non-event arrays may use `with_series_representation()` as an explicit
application assertion, but DuckPD then records `application_asserted` rather
than claiming that it verified grid construction. A future regular-grid rolling
API must define gap handling and provenance before it can produce verified
`fixed_grid` metadata.

The normalization order is fixed:

1. Validate channel mapping, shape, numeric values, and window completeness.
2. Apply per-channel `none`, `center`, or population `zscore` preprocessing.
3. Flatten native inputs in channel-major, then oldest-to-newest order; or call
   the selected learned encoder with the declared channel order.
4. If `unit_norm=True`, L2-normalize the final vector.
5. Validate finite float32 output of exactly the declared dimension.

Statistics are computed independently for each complete window, never across a
batch. Native means, variances, and norms use double-precision intermediate
arithmetic. `zscore` uses `ddof=0`. Implement stable reductions; do not subtract
large nearly equal moments without a numerical-stability test.

`zero_scale` applies when z-scoring a constant channel or unit-normalizing a
zero vector. With `"null"`, corpus output is one whole null vector; with
`"error"`, execution fails. A query that becomes null always raises, because
there is no meaningful query vector. No undocumented epsilon, clipping, or
zero-vector substitution is permitted. Center-only output can be zero and is
valid for L2, but not for cosine retrieval.

### Row-preserving representation construction

```python
embedded = windows.embed_series(
    columns={"simple_return": "return_window"},
    into="return_shape",
    representation=return_shape,
    batch_size=256,
    null_policy="propagate",
)
valid = embedded[embedded["return_shape"].notna()]
valid.write_parquet("return-patterns.parquet")
```

Proposed signature:

```text
DataFrame.embed_series(
    *,
    columns: Mapping[str, str],
    into: str,
    representation: SeriesRepresentationSpec,
    batch_size: int = 256,
    null_policy: Literal["propagate", "error"] = "propagate",
) -> DataFrame
```

The mapping keys must exactly match `representation.channels`. Mapping insertion
order is irrelevant: the specification determines channel order. Mapping values
identify same-row array columns. Each must have the declared length. Duplicate
physical inputs are allowed only when the caller deliberately maps them to
different declared channels; there is no implicit channel duplication.

The operation appends a column and preserves input rows, index, identity, and
ordering. A conflicting or empty `into` name is an eager error. `batch_size`
must be positive; it controls learned inference only. Native recipes compile
entirely to DuckDB expressions.

With `null_policy="propagate"`, a null input channel produces one whole null
output without calling a provider for that row. This does not permit NaN,
infinity, incorrect lengths, or partially null child arrays. Those are malformed
inputs and fail. Under `"error"`, a null channel also fails. Filtering incomplete
windows is explicit and normally occurs after constructing the windows.

A scalar numeric column is not implicitly interpreted as either a whole series
or a one-element channel. Users must construct arrays first. The initial API does
not accept arbitrary Python callbacks or row-dependent encoder configuration.

### Model specification and preparation

A learned encoder is a component of a representation, not the complete space:

```python
# Illustrative custom checkpoint. Replace the revision and digest placeholders
# with values from a qualified, immutable checkpoint manifest.
encoder = pd.series_embedding_model(
    "research/return-window-encoder",
    revision="immutable-checkpoint-revision",
    artifact_sha256="<64-lowercase-hex-digest>",
    backend="custom",
    dimension=128,
    input_length=60,
    input_channels=("simple_return",),
    input_normalization="none",
    pooling="mean-valid-v1",
    adapter_revision="return-encoder-adapter-v1",
)
learned_returns = pd.series_representation(
    window=60,
    channels=("simple_return",),
    sampling="observations",
    step="PT1M",
    data_contract="market/simple-return/split-adjusted/v1",
    normalization="none",
    unit_norm=True,
    zero_scale="error",
    encoder=encoder,
)
```

Proposed `SeriesEmbeddingModelSpec` fields are exactly those in the example.
The constructor is `pd.series_embedding_model(model, *, ...)`; all fields shown
are required. `dimension` and `input_length` are positive integers,
`input_channels` is a nonempty tuple of unique names, and `artifact_sha256` is a
validated SHA-256 digest of a canonical artifact manifest. The manifest hashes
all numerical assets and configuration required by the adapter, not merely one
arbitrarily selected weights file. The adapter revision identifies its inference
contract. Runtime package versions and resolved asset provenance are also
reported during preparation.

Initially, `backend="custom"` supports explicitly registered providers. A
reviewed `"moment"` backend is added only when its adapter and checkpoint pass
the qualification gates below. A string naming a backend does not authorize
importing arbitrary code from a catalog or model repository.

Model preparation is explicit and eager:

```python
# provider is an application-created object implementing the protocol below.
session.register_series_embedding_provider(encoder, provider)
info = session.prepare_series_embedding_model(encoder)

learned = windows.embed_series(
    columns={"simple_return": "return_window"},
    into="return_embedding",
    representation=learned_returns,
    batch_size=256,
)
```

Preparation verifies the actual resolved revision, artifact manifest digest,
input contract, pooling implementation, and output dimension. It must not merely
hash a declared revision while loading the provider's default checkpoint.
Unprepared or mismatched encoders fail before data scanning or remote feature
partition acquisition attributable to the execution.

`SeriesEmbeddingModelSpec` also exposes canonical `to_dict()` / `from_dict()` and
a computed `fingerprint`. `PreparedSeriesModelInfo` is immutable and records
`model_fingerprint`, `resolved_revision`, `artifact_sha256`, `backend`,
`adapter_revision`, `input_length`, `input_channels`, `input_normalization`,
`pooling`, `dimension`, `cache_path`, `execution_providers`, `runtime_versions`,
and `preparation_seconds`. Preparation validates all declared fields against the
loaded adapter rather than trusting a provider's reported dimension alone.

No implicit truncation, interpolation, padding, or context-length conversion is
allowed. The initial adapter contract requires `representation.window ==
encoder.input_length` and identical declared channels. Supporting a 60-point
query with a checkpoint requiring a different context length needs an explicitly
versioned adapter/specification extension, including mask semantics. It is not a
silent convenience behavior.

`input_normalization` documents normalization performed inside the encoder in
addition to representation preprocessing. Supported values are adapter-defined
and validated, not free-form options that change inference without changing the
fingerprint. A model with internal normalization is not assumed to preserve
absolute volatility simply because the outer recipe uses `"none"`.

### Query representation and exact retrieval

A raw series query is a mapping of channel names to finite sequences. Its values
are input observations, not an already embedded vector:

```python
matches = valid.vector.search_series(
    {"simple_return": query_returns},
    column="return_shape",
    k=50,
    metric="cosine",
    tie_breaker="observation_id",
)
```

Proposed signature:

```text
DataFrame.vector.search_series(
    query: Mapping[str, Sequence[float]],
    *,
    column: str,
    representation: SeriesRepresentationSpec | None = None,
    metric: Literal["cosine", "l2", "inner_product"] = "cosine",
    k: int = 10,
    distance_column: str = "_distance",
    tie_breaker: str | None = None,
) -> DataFrame
```

The operation resolves the representation from column metadata. An explicit
`representation` must match that metadata; it is an assertion, not an override.
Missing metadata raises rather than guessing a space from shape. Query values
are copied into an immutable session-owned snapshot during planning, validated
for shape, and referenced by an opaque query key. Mutating the caller's list
later must not alter the plan.

Planning does not run normalization or a model. Execution represents the query
once for that execution, validates it, and uses existing exact distance/top-k
semantics. Deterministic cached query encodings may be reused only under the
complete representation fingerprint and immutable query content; caches are
session-local and bounded. There is no `mode` argument in the initial high-level
API and no automatic approximate fallback.

The eager reusable form is:

```python
query_vector = session.embed_series_query(
    {"simple_return": query_returns},
    representation=return_shape,
)
matches = valid.vector.search(
    query_vector,
    column="return_shape",
    metric="cosine",
    k=50,
    tie_breaker="observation_id",
)
```

`Session.embed_series_query()` returns a frozen `EmbeddedSeriesQuery(values,
representation_fingerprint)`. It applies exactly the corpus recipe, except that
invalid/null output is an error rather than a missing corpus row.

As an additive compatibility improvement, extend `Series.vector.distance()` and
`DataFrame.vector.search()` to accept both this new type and the existing text
`EmbeddedQuery`. A typed query requires matching column metadata and dimension.
Existing plain numeric sequences keep working as the intentionally unverified
low-level path. Passing `.values` strips the semantic compatibility guarantee.
The existing text fingerprint algorithm and query type do not change.

Null corpus vectors are not silently skipped by search. Users filter them before
ranking; otherwise the existing invalid-vector failure policy applies. Ties use
the existing optional tie-breaker. Without a unique tie-breaker, results among
equal-distance rows need not be reproducible. Inner-product mode remains
negative inner product so all distances sort ascending.

## Event-aligned windows

Event alignment is a separate, later implementation phase built on the same
array and representation contracts. It is included here because pairing the
wrong windows with news would invalidate the main use case.

### Proposed API

```python
reactions = prices.event_windows(
    events,
    on="bar_start",
    bar_label="start",
    event_on="published_at",
    by="ticker",
    event_id="event_observation_id",
    columns={
        "return_window": "return_1m",
        "volatility_window": "abs_return_1m",
    },
    window=(0, 60),
    step="PT1M",
    anchor="ceil",
    available_at="bar_available_at",
    event_available_at="news_available_at",
    incomplete="null",
    metadata_prefix="reaction",
)
```

Proposed signature:

```text
DataFrame.event_windows(
    events: DataFrame,
    *,
    on: str,
    bar_label: Literal["start"],
    event_on: str,
    by: str | Sequence[str],
    event_id: str | Sequence[str],
    columns: Mapping[str, str],
    window: tuple[int, int],
    step: str,
    anchor: Literal["floor", "ceil"],
    available_at: str,
    event_available_at: str,
    incomplete: Literal["null", "error"] = "null",
    metadata_prefix: str = "window",
) -> DataFrame
```

The receiver is the numeric observation source. The result contains the event
columns, one array per mapping entry, and prefixed window metadata. Both frames
must belong to the same session. Column mapping keys are new output names and
values are source numeric columns. All collisions are rejected eagerly.

`event_id` specifies a non-null unique key of the **event input**, including
entity/revision when needed. `by` specifies identically named entity columns in
both inputs. The source must be unique on `(by, on)` within the relevant domain.
Null keys, null times, and duplicates are errors, not arbitrary first matches.
Event-row order/identity can be preserved only to the extent the event input
already guarantees them; an unordered event table does not become ordered just
because arrays inside each row are ordered.

`bar_label="start"` is a required publisher assertion, not an inference from the
column name or timestamp values. It is stored in `EventWindowPlan` and resulting
`SeriesWindowSpec` provenance. No other value is accepted initially. End-labeled
bars must be converted to start timestamps before this call; adding end-labeled
input later requires a separately specified conversion and availability rule.

### Fixed-grid semantics

The initial version accepts UTC, microsecond-representable timestamps and fixed
positive durations. Source `on` timestamps label the **start of a complete bar**.
A dataset labeling bar ends must be converted explicitly upstream. Events and
observations with incompatible timestamp semantics are rejected or require an
explicit conversion; no timezone or bar-label inference is performed.

The grid is anchored to the Unix epoch in UTC. For step `s`, event time `t`, and
integer offsets `[a, b)` with `a < b`:

```text
anchor_time = floor(t / s) * s       when anchor = "floor"
anchor_time = ceil(t / s) * s        when anchor = "ceil"
expected timestamps = anchor_time + j * s, for j in [a, b)
L = b - a
window_start = anchor_time + a * s
window_end = anchor_time + b * s     # exclusive
```

A post-event `(0, 60)` window with `anchor="ceil"` at `10:00:30` selects bar starts
`10:01` through `11:00`; its exclusive end is `11:01`. A pre-event `(-60, 0)`
window with `anchor="floor"` ends at `10:00` and excludes the bar containing the
event. At an exact grid boundary, `ceil` and `floor` both return that boundary.
These choices avoid silently including partially pre-event or post-event bars.

This is not "the next 60 available bars." A missing bar, exchange closure,
halt, or null required value makes the window incomplete. There is no forward
fill, nearest-time join, skip-to-next-session behavior, or compressed array.
Trading-time windows can be introduced later as a separate sampling contract,
not as a reinterpretation of `fixed_grid`. Source bar starts within the bounded
extraction domain must be aligned to the declared epoch grid;
off-grid starts are errors, not observations silently dropped as missing slots.
Event timestamps themselves need not lie on that grid.

The bounded extraction domain is the union, per requested entity, of the
half-open intervals `[window_start, window_end)` generated for the event rows in
this execution. Source rows outside that union are irrelevant and are not
validated for grid alignment. Within the union, every source row for a requested
entity is validated, including rows whose timestamp is not one of the exact
slots after overlapping intervals are coalesced. This definition bounds remote
work and prevents an off-grid row inside a requested interval from being
silently ignored.

The result adds:

| Column suffix | Meaning |
| --- | --- |
| `_window_start` | First expected bar start |
| `_window_end` | Exclusive end of the last expected bar |
| `_window_count` | Number of positions with one finite observation in every selected channel |
| `_window_complete` | Whether all `L` positions are complete |
| `_window_available_at` | Complete-window data availability, or null for incomplete windows |

For `metadata_prefix="reaction"`, these are named
`reaction_window_start`, `reaction_window_end`, and so on. With
`incomplete="null"`, an incomplete row remains present and all requested arrays
are whole nulls. With `"error"`, execution fails. Invalid nonfinite numeric values,
duplicates, and invalid availability timestamps are errors under either policy.

### Availability and event revisions

For a complete window:

```text
window_available_at = max(
    event_available_at,
    max(available_at for all required source bars),
)
```

A completed-bar observation cannot be available before its declared bar end;
validate this for the initial bar-start contract. For pre-event windows, including
`event_available_at` means the resulting **event association** is not available
before the event is known, even if its market observations were known earlier.
An execution/materialization timestamp is separate: it describes when this
particular artifact was produced, not when the underlying observations became
available. An online system claiming actual serving availability must additionally
account for encoding and publication latency.

Retain both event occurrence/publication time and first-known/ingestion time.
News corrections, price corrections, and revised bars must be versioned upstream
when historical-knowledge reconstruction is required. A latest-revised table
cannot retrospectively prove what was known earlier. The API does not invent a
bitemporal history that the input lacks.

### Relational lowering

Use a typed binary `EventWindowPlan`. A correct initial lowering constructs the
required event grid with bounded integer offsets, joins observations on entity
and exact expected timestamp, then aggregates in offset order. It must retain
unmatched event rows, detect duplicate observation matches, and validate counts
before casting arrays.

Do not join every price row to every event for an entity and filter afterward.
Push safe entity and outer time bounds into supported sources; use range/offset
relations only over the requested event windows. Cardinality before aggregation
is on the order of `events * window_length`, and explain/profile must expose this
workload. No Python loop may collect one DataFrame per event.

Filtering event candidates before extraction is allowed and usually desirable.
Filtering source bars changes window completeness and is semantically different.
The optimizer must never move an output eligibility predicate into the price
history in a way that truncates required slots.

## Event retrieval and score fusion

### Three supported workflows

**Text first:** search news embeddings, retain event identifiers, and inspect or
rerank their reaction vectors. **Reaction first:** search return/volatility
representations and join the corresponding news. **Joint event similarity:**
compute multiple distances for every eligible event and combine those distances
under an explicit scoring rule.

Use the event observation key for these joins. Joining news and prices only by
ticker, or assuming an ASOF price join identifies the right news event, is not
an event-pairing contract. News about multiple entities and multiple events in
the same minute require explicit application decisions.

### Exact joint scoring

The following is a proposed-API usage example. `q_text` and `q_reaction` are typed
queries produced by the existing text and proposed series query encoders:

```python
q_text = session.embed_query(query_news, model=text_model)
q_reaction = session.embed_series_query(
    {"simple_return": query_returns},
    representation=event_return_shape,
)

eligible = event_bank[
    (event_bank["reaction_window_available_at"] <= cutoff)
    & event_bank["news_embedding"].notna()
    & event_bank["reaction_embedding"].notna()
    & (event_bank["event_observation_id"] != query_event_id)
]
scored = eligible.assign(
    text_distance=lambda df: df["news_embedding"].vector.distance(q_text, metric="cosine"),
    reaction_distance=lambda df: df["reaction_embedding"].vector.distance(
        q_reaction, metric="cosine"
    ),
)
scored = scored.assign(
    combined_distance=lambda df: 0.4 * df["text_distance"] + 0.6 * df["reaction_distance"],
)
matches = scored.sort_values("event_observation_id").nsmallest(
    50, ["combined_distance", "event_observation_id"]
)
```

This uses the proposed additive typed-query support in `vector.distance()`.
`event_return_shape` must describe the same fixed-grid reaction space as the
stored column, not the earlier observation-count example. The explicit sort also
satisfies DuckPD's current `nsmallest()` ordering requirement.
[Source: DataFrame ordering/top-k implementation][src-frame].

The weights are an example scoring rule, not calibrated probabilities or a
recommended universal setting. Text and reaction distances can have different
distributions despite both using cosine. Applications may fit frozen calibration
functions on an appropriate reference set or choose rank-based fusion. Record
that calibration, its data provenance, missing-modality policy, weights, and
version with the analysis. Do not estimate a different score normalization from
each arbitrary candidate batch.

Exactness is defined **relative to the stated score over the entire eligible
set**. Retrieving the top 100 text matches and reranking their reactions is
candidate-limited, not globally exact joint retrieval. Taking the union of the
top candidates from each modality still does not guarantee the global weighted
top-k. The API and examples must label that distinction.

Missing modalities are excluded in this example. Alternatives such as separate
rankings or renormalized weights require an explicit policy; a missing modality
must not become distance zero or an unexplained advantage.

### Eligibility, overlap, and correlation analysis

Apply historical cutoffs, entity/universe restrictions, duplicate-event
exclusions, and forbidden overlap conditions **before** top-k. Filtering after
search can return fewer than `k` rows and is a different query.

Overlapping rolling windows are often trivial neighbors. Applications should
exclude the query event and, when appropriate, overlapping or nearby windows of
the same entity. Embargo widths must include upstream transform context, not
only the final array span. Do not universally ban same-entity matches: sometimes
those are exactly the desired comparisons.

A price pattern matching a news-linked reaction does not establish news
causality. For analytical comparison between news similarity and reaction
similarity, retain pair-level distances/correlations and stratify by relevant
entity, regime, horizon, and data conventions. Raw embedding coordinates from
different spaces are not interpretable as matched financial factors. Statistical
testing and research conclusions remain outside DuckPD core.

### True cross-modal retrieval is a later extension

A future jointly trained pair of encoders could map news and reactions into a
shared space. It requires an explicit shared-space contract covering both tower
revisions, projection heads, preprocessing, dimension, normalization, training
alignment identity, and permitted query/document modalities.

Equal dimension or a user-supplied shared label is insufficient. Do not allow a
caller to declare an arbitrary BGE text vector compatible with a MOMENT vector.
The first release exposes no `multimodal.search()` API. A later design can add
cross-modal typed queries once qualified aligned models exist for the intended
data. TRACE is a relevant research reference, not a ready-made guarantee of
financial event alignment; its public workflow includes separate pretraining
and context-alignment stages and weather/TimeMMD data.
[Source: official TRACE implementation][model-trace].

## Representation metadata and compatibility

### Separate representation identity from provenance

A representation fingerprint covers the canonical, fully resolved specification:

```text
series representation schema version
window length
ordered semantic channel names
sampling mode and canonical fixed step, if declared
data contract identifier
outer normalization and zero-scale policy
native channel-major layout version, or complete encoder specification
final unit-normalization policy
output type and dimension
```

For a learned encoder, the resolved specification includes actual artifact
identity, adapter revision, input normalization, pooling, and input/output shape.
Registry aliases are expanded before hashing. A renamed catalog registry key
must not create a new space if the resolved specification is identical.

The fingerprint excludes physical column labels, `ColumnId` values, entity IDs,
event IDs, file paths, batch size, cache location, and device name. Those describe
provenance or execution, not the mathematical contract. Dtype/quantization or a
backend change that materially changes the encoding function requires a distinct
qualified adapter/specification. Cross-device numerical equivalence is tested
within published tolerances, not assumed bit-for-bit.

Canonicalization must use a versioned JSON-compatible schema with sorted object
keys, preserved tuple/list order, canonical durations, no nonfinite numbers, and
explicit defaults. Fingerprints are computed, not accepted as arbitrary catalog
strings. Add golden fixtures before publishing the first schema version.

A `SeriesColumnSpec` should contain the immutable representation, its computed
fingerprint, and a separate provenance/trust descriptor. A `SeriesWindowSpec`
should describe raw window length, verified sampling information when available,
and the typed references used to establish order/partitioning. Both are separate
from legacy `EmbeddingColumnSpec`. Runtime `ColumnId` references in
window provenance are rebound to physical labels/new column IDs when persisted
and restored; they are not stable cross-session identifiers or fingerprint input.

Do not add new fields to the serialized legacy `EmbeddingModelSpec` and thereby
change all existing text fingerprints. A future generic representation wrapper
may unify access internally, but it must retain old text hashes and serialized
formats exactly. [Source: current text fingerprint construction][src-embeddings].

### Propagation rules

| Operation | Required behavior |
| --- | --- |
| Filter, sort, limit, direct column selection | Preserve the representation attached to retained values |
| Rename or direct-column alias | Preserve the same specification and fingerprint |
| `assign` of a direct `ColumnRef` | Copy representation metadata explicitly |
| `assign` of arithmetic, arbitrary UDF, or incompatible cast | Remove representation metadata from the transformed output |
| Join/ASOF payload projection | Preserve each copied vector's metadata; adjust nullability separately |
| Compatible concatenation | Preserve metadata only when every contributing column has the same specification |
| Conflicting concatenation | Raise for explicitly typed representation columns; never choose the first specification |
| Group aggregation | Drop representation metadata unless a specific operation proves preservation |
| External file lacking metadata | Keep numeric arrays usable through raw vector APIs; do not infer a representation |

A direct alias should not require recomputing embeddings. A transformed vector
must not retain a stale space declaration just because its dimension is unchanged.
Apply these rules to `Series.to_frame()`, DataFrame projection/rename/assignment,
merge output columns, and feature-store aliases, not only `embed_series()`.

Add an explicit external-data assertion path:

```python
annotated = external_vectors.with_series_representation(
    column="embedding",
    representation=known_representation,
)
```

This proposed row-preserving metadata operation performs no encoding. It validates
known shape/type eagerly and physical values at execution. It records
`provenance="application_asserted"`. It must reject a conflicting existing
specification rather than silently overwrite it. An absent producer history
remains absent: neither this method nor a catalog proves that the values were
actually produced by the stated encoder.

The operation creates a `SeriesRepresentationAssertionPlan`; changing only
`FrameMetadata` is insufficient. Its eager construction validates the known
logical type and dimension. Execution validates every consumed non-null value
for fixed length, numeric float32 convertibility, null children, and finiteness,
then passes rows through unchanged. Whole null vectors remain valid when the
column is nullable. The plan is a conservative optimizer boundary until a rewrite
can prove that validation is preserved, and all plan walkers, explain modes, and
direct sinks must handle it explicitly. As with other lazy validation, an
early-stopped consumer certifies only rows it consumed; a complete sink validates
the complete consumed input.

### Compatibility checks

The high-level series search path and typed numeric-query path must reject:

- A different window length, sampling mode, cadence, channel order, input contract,
  normalization, encoder artifact, adapter, pooling rule, or output dimension.
- A query carrying a text-only fingerprint against a series-only representation,
  even if both vectors have the same length.
- An explicit search specification that conflicts with persisted/catalog metadata.
- A query with missing/extra channels, wrong lengths, nonnumeric entries, nulls,
  infinities, or an undefined cosine norm.

A plain numeric vector remains the expert escape hatch for numerical comparison
without semantic checks. Its use should be visible in explain output as
`compatibility_check: "dimension_only"`, not reported as verified space equality.

## Persistence

### Parquet

Parquet sidecar manifest versions are independent of the feature-store
`catalog_version`, which remains 1. Keep reading the existing
`<path>.duckpd-embeddings.json` sidecar manifest version 1 format for legacy
text-only artifacts. New writes that contain only text metadata may retain that
format. Add a separate, versioned `<path>.duckpd-series.json` companion for
series window and representation metadata. A file containing both modalities
has both companions and uses the generation-bound envelopes below; the
serialized legacy `EmbeddingModelSpec` objects and their fingerprints remain
unchanged.

The new series sidecar envelope contains:

```text
version: 1
artifact:
    size_bytes
    sha256
    generation
columns:
    physical output label:
        window: serialized SeriesWindowSpec, when applicable
        representation: resolved SeriesRepresentationSpec, when applicable
        provenance: producer/declaration information
```

For a mixed-modality file, write `<path>.duckpd-embeddings.json` sidecar
manifest version 2 with the same `artifact` object before its existing `columns`
mapping. Both sidecars must contain identical `size_bytes`, `sha256`, and a
newly generated opaque `generation` value. A series-aware reader attaches
neither text nor series metadata unless both required sidecars exist, both
envelopes agree, and the data file matches the bound size and digest. It must
never restore valid-looking text metadata independently from an invalid series
sidecar on a mixed artifact. Sidecar manifest version 2 changes only envelope
binding; text model objects and fingerprinting are byte-for-byte compatible
with their version 1 definitions.

The digest binds metadata to the actual data file, not only to its schema. For
the initial local implementation, compute it by streaming the completed file;
verify it when the file is first executed with attached series metadata. Planning
may read the small sidecar and schema but must not read the full file to verify
the digest. The integrity pass adds real I/O and must be separately reported.
A later footer-embedded generation binding can replace this cost only after an
explicit, tested design; size and modification time alone are insufficient.

Write the data and all required sidecars to temporary paths and compute their
shared binding only after the data sink succeeds. For replacement, first make
old DuckPD sidecars unavailable, then replace the data, then publish every new
sidecar. This deliberately permits a crash to leave usable raw data without
verified metadata, but never leaves an old DuckPD sidecar beside newly published
data. Readers still verify the binding because independent filesystem operations
are not a transaction and concurrent observers can see intermediate states.
Obsolete companion metadata is removed on replacement. Failure cleanup removes
only staging artifacts created by that attempt and never unrelated pre-existing
files. Restoring metadata availability after a failed replacement is allowed
only when its digest is verified against the currently published data.

Every file in a multi-file read must declare compatible metadata for a verified
series column. Missing sidecars mean that column cannot acquire verified metadata
for the combined scan; conflicting or invalid supplied sidecars raise. Do not
silently retain the first file's declaration. Array/list physical round-trips
must be checked against declared length at source binding/execution, because
external Parquet tooling may expose lists rather than fixed-size arrays.

Remote raw Parquet reads do not acquire model trust by guessing sidecar URLs.
Portable remote metadata comes through an explicit feature-store catalog or
application assertion. This proposal adds no general remote artifact writer or
partitioned output API; the current single-file sink remains the first target.

### Local tables

Extend existing session-managed table metadata registration to series specs.
`persist()` and `save_as_table()` preserve metadata within the session, and
append validates compatible representation columns before accepting rows.
Overwriting a table must replace or remove stale metadata.

Do not promise automatic restoration after reopening a database: current text
metadata registration is session-managed. A catalog or explicit metadata assertion
can restore a series declaration. A durable database metadata registry would be
a separate persistence enhancement, not a hidden prerequisite.
[Source: session metadata and sink integration][src-session].

## Feature-store integration

### Relationship to the existing catalog proposal

Core window/representation/search functionality must ship independently of the
catalog work. DuckPD has one catalog schema, identified by `catalog_version: 1`.
The text-embedding and series-representation proposals both extend that schema
in place; they do not introduce catalog versions 2 or 3.

The series extension adds `series_embedding_models` and
`series_representations` registries plus `series_representation` references on
feature or table-column declarations. Existing text `embedding_models` and
`embedding_model` fields remain unchanged. The parser, generator, examples, and
both proposals must adopt the complete catalog version 1 schema atomically.

A deterministic example is:

```json
{
  "catalog_version": 1,
  "series_embedding_models": {},
  "series_representations": {
    "return_shape_60_1m": {
      "version": 1,
      "window": 60,
      "channels": ["simple_return"],
      "sampling": "observations",
      "step": "PT1M",
      "data_contract": "market/simple-return/split-adjusted/v1",
      "normalization": "center",
      "unit_norm": true,
      "zero_scale": "null",
      "encoder": null
    },
    "volatility_level_60_1m": {
      "version": 1,
      "window": 60,
      "channels": ["absolute_simple_return"],
      "sampling": "observations",
      "step": "PT1M",
      "data_contract": "market/absolute-simple-return/split-adjusted/v1",
      "normalization": "none",
      "unit_norm": false,
      "zero_scale": "error",
      "encoder": null
    }
  },
  "datasets": [
    {
      "name": "returns",
      "kind": "timeseries",
      "time_column": "available_at",
      "series_keys": ["ticker"],
      "partitioning": {
        "column": "available_at",
        "timezone": "UTC",
        "unit": "day"
      }
    },
    {
      "name": "volatility",
      "kind": "timeseries",
      "time_column": "available_at",
      "series_keys": ["ticker"],
      "partitioning": {
        "column": "available_at",
        "timezone": "UTC",
        "unit": "day"
      }
    }
  ],
  "features": {
    "returns:shape_60": {
      "dataset": "returns",
      "name": "shape_60",
      "availability_delay": "PT0S",
      "lookahead_safe": true,
      "series_representation": "return_shape_60_1m"
    },
    "volatility:level_60": {
      "dataset": "volatility",
      "name": "level_60",
      "availability_delay": "PT0S",
      "lookahead_safe": true,
      "series_representation": "volatility_level_60_1m"
    }
  }
}
```

This is the semantic catalog portion; deployment-specific source prefixes
remain catalog-defined. Published time-series datasets follow the feature-store
UTC-day contract: `year=YYYY/month=MM/day=DD/part.parquet`. The
`lookahead_safe` claims are valid only if the publisher actually timestamps
each complete representation at its availability and preserves the required
data history. They are not made true by the JSON field.

A learned recipe sets `encoder` to a registry key in `series_embedding_models`.
Canonical `SeriesRepresentationSpec.to_dict()` instead embeds the resolved model
object; the catalog loader resolves aliases before constructing that canonical
object or its fingerprint. Registry definitions match the model fields above.
Unknown fields inside either specification are rejected. Human descriptions and
producer audit metadata belong in separately defined metadata fields, not
unrecognized encoding options.

A table dataset may declare the same reference under
`datasets[].columns.<column>.series_representation`. A column cannot simultaneously
claim unrelated text and series spaces. A future aligned-space declaration needs
its own schema, not both fields attached opportunistically.

### Typed planning and source validation

The current feature-store timeseries planner constructs columns with `UNKNOWN`
types. Attaching a representation while leaving that type unchanged is not
sufficient for vector accessors. Resolve `FLOAT[D]` from the validated catalog
specification into logical metadata during planning, and validate physical type,
length, nullability, and values when binding/reading the actual partitions.
[Source: timeseries source construction][src-store].

This is a declared logical type, not proof about unseen Parquet data. Wrong
physical types or lengths raise; there is no silent projection/truncation to make
them fit. Catalog identity and any available producer sidecar must agree. A
conflicting declaration is an error, not an override policy.

Preserve representation metadata through requested aliases, exact alignment,
point-in-time payload projection, and nullable unmatched ASOF rows. Joining
feature families must not compare or merge their spaces simply because their
array dimensions match. Exact alignment still requires the existing compatible
time/key contracts; unrelated news and market families are not automatically
joined into events.

Proposed lookup and usage:

```python
store = pd.FeatureStore(
    source="/data/market-feature-catalog",
    session=session,
)
features = store.features(
    features={"returns:shape_60": "pattern"},
    start="2025-01-01T00:00:00Z",
    end="2026-01-01T00:00:00Z",
    alignment="exact",
)
recipe = store.series_representation("return_shape_60_1m")
valid = features[features["pattern"].notna()]
matches = valid.vector.search_series(
    {"simple_return": query_returns},
    column="pattern",
    representation=recipe,
    k=50,
)
```

`FeatureStore.series_representation(name)` is a new metadata-only lookup. For
learned registries, also add `FeatureStore.series_embedding_model(name)` so an
application can prepare the exact declared encoder explicitly.

### Model acquisition policy

The companion text proposal permits automatic preparation of catalog-declared
local text models. That permission does not automatically extend to potentially
large series runtimes or custom code. This proposal keeps learned-series
preparation **explicit by default**, including catalog-loaded columns. Native
representations need no preparation.

Portable catalogs initially accept native recipes. A built-in learned backend
may be allow-listed only after qualification. An explicitly registered custom
provider may satisfy a trusted local application catalog, but a catalog alone
cannot import a module, execute remote code, install packages, load arbitrary
pickles, obtain credentials, or select a hosted endpoint. No fallback from local
to hosted inference is allowed.

### Availability and point-in-time behavior

Existing point-in-time alignment supports a feature timestamp plus a static
`availability_delay`; it is not a general per-row event-completion engine.
[Sources: catalog delay parsing][src-catalog] and [ASOF feature planning][src-store].

For event reactions, prefer storing the representation's actual
`window_available_at` as the dataset time column, retaining `event_time`,
`window_start`, and `window_end` as separate payload. Zero additional delay is
valid only under that publisher contract. A same-entity availability timeline
also needs deterministic revision/key handling; duplicate effective times must
not create ambiguous ASOF matches.

Alternatively, keep an ordinary event table and explicitly filter
`window_available_at <= cutoff` before search. This is often the clearest
historical-event analysis path. Do not set `lookahead_safe=true` on an event-time
feature containing a later reaction merely because the window has a nominal
60-minute duration. Missing bars, closures, ingest delays, and revisions can make
a static 60-minute delay incorrect.

A window-local encoder may use every observation inside a completed window
without looking outside it. That does not establish that the encoder itself
existed at a historical cutoff. Retrospective similarity analysis may deliberately
use a newer model; strict historical-knowledge claims must also account for
checkpoint availability, training-data provenance, and data revisions. A
`causal=True` encoder option alone does not make per-window normalization or a
later-trained model historically available.

The feature store retrieves already produced features. It does not construct
reaction windows, train encoders, refresh embeddings, or publish revised catalogs
as part of a search call. Those remain explicit producer workflows.

## Provider contract and execution lifecycle

### Arrow provider interface

The proposed core protocol uses existing optional-provider conventions:

```python
class SeriesEmbeddingProvider(Protocol):
    @property
    def specification(self) -> SeriesEmbeddingModelSpec: ...

    @property
    def thread_safe(self) -> bool: ...

    def prepare(self) -> PreparedSeriesModelInfo: ...

    def embed_windows(
        self,
        batch: pyarrow.RecordBatch,
    ) -> pyarrow.FixedSizeListArray: ...
```

`batch` has one field per semantic channel in specification order. Each field is
a non-null `FixedSizeList<float32, L>` for the valid rows in that call. DuckPD has
already applied outer recipe normalization and excluded rows whose declared
null policy propagates. The provider performs its declared internal preprocessing
and encoding only; it must not reinterpret grouping or build windows across
Arrow batches.

The provider returns exactly `batch.num_rows` non-null vectors of the declared
float32 dimension. DuckPD validates shape, row count, null children, and finite
values, applies requested final unit normalization, then scatters results back
to their original row positions. Invalid provider output is an error, not an
occasion to silently cast arbitrary dimensions or drop rows.

Providers may pack tensors as `(B, C, L)` or `(B, L, C)` internally. That adaptation
belongs to the provider, not public SQL or a serialized opaque callable. Empty
valid batches do not call inference. Custom providers are registered objects in
the session, not executable configuration stored inside a logical plan.

`thread_safe` is an immutable capability for the lifetime of a registration.
DuckPD serializes calls when it is false. `prepare()` is the provider's attestation
boundary for custom backends: its result must report the resolved revision,
canonical artifact-manifest digest, adapter revision, input contract, pooling,
dimension, runtime versions, and execution providers. Core compares every
reported semantic field with the registered specification and rejects omissions
or mismatches. Core does not pretend it can independently discover arbitrary
custom model assets; a provider that cannot produce and verify the declared
manifest digest cannot be prepared. Built-in providers additionally perform
backend-specific independent artifact inspection before returning this report.

A qualified provider must use evaluation/inference mode, no stochastic dropout,
no training-state updates, and no statistics shared between corpus rows. Batch
size/order must not materially change output. Session close
releases retained model handles and bounded query caches.

### Execution sequence

At an eager consumer, the executor should perform these stages in order:

1. Walk the full plan, resolve representation/provider requirements, and validate
   immutable query snapshots and compatible metadata.
2. Verify required prepared model handles/artifacts. Report unavailable providers
   before initiating the data work attributable to this execution.
3. Bind sources and validate catalog/sidecar declarations and required source
   constraints, respecting remote scan policies and integrity checks.
4. Execute native windows and preprocessing in DuckDB.
5. For learned nodes, feed complete-window Arrow chunks into bounded provider
   calls and validate/scatter output.
6. Run ordinary projection/filter/distance/top-k and the requested eager sink.

DuckDB source schema inspection or catalog loading may already have occurred
while creating the source DataFrame. "No model work during planning" does not
promise that all existing source constructors perform zero metadata I/O.

Constructing `embed_series()` or `search_series()` must not perform inference,
model downloads, dependency installation, corpus scans, or preparation. Explicit
`prepare_series_embedding_model()` and `embed_series_query()` are eager by design.

### Resource behavior

Bounded Arrow batches constrain the inference handoff, not the entire query's
memory use. Sorts, ordered list windows, event-grid joins, DuckDB operators, model
weights/activations, and HNSW indexes have their own resource behavior. Do not
claim that `batch_size=256` bounds all memory or that DuckDB's memory limit governs
a GPU runtime.

Native rolling arrays require work and storage proportional to the number of
windows times their length/channels. Learned embeddings of overlapping windows
are not free: independently encoding every window repeats context computation.
A future cached-state implementation must prove identical window-local semantics
before replacing independent encoding.

For scale illustration, one million `FLOAT[60]` vectors contain 240 MB of raw
float payload; ten million `FLOAT[512]` vectors contain 20.48 GB before metadata,
indexes, or storage compression. These are dimensional arithmetic, not measured
DuckPD memory estimates.

Persist reusable representations before repeated retrieval. A transient
`embed_series(...).vector.search_series(...)` recomputes eligible corpus
embeddings on each execution. No automatic materialization, persistent cache, or
incremental refresh is introduced. Endpoint sampling can reduce stored windows,
but a row filter applied before rolling changes the underlying series. Sample
completed windows or select event endpoints explicitly.

Do not copy the current text progress implementation's input pre-count into
series execution. Use known cardinality where available or an indeterminate
processed-row counter. Full extra scans for progress are not permitted; any
explicit integrity pass or requested count must be separately visible.

## Logical plans, compilation, and optimizer rules

### New typed objects

Use the following initial internal boundaries:

| Object | Responsibility |
| --- | --- |
| `WindowArrayExpression` | Fixed-count ordered window packing, completeness, numeric validation, static array type |
| `SeriesRepresentationPlan` | Row-preserving native/learned representation with channel `ColumnId` mapping and output metadata |
| `SeriesSearchPlan` | Lazy raw-window query representation followed by exact vector ranking |
| `EventWindowPlan` | Binary event/observation alignment and ordered fixed-grid array aggregation |
| `SeriesRepresentationAssertionPlan` | Row-preserving execution validation for application-asserted external vectors |
| `SeriesWindowSpec` | Raw array shape/sampling provenance, without an embedding-space claim |
| `SeriesColumnSpec` | Complete representation identity and separate declaration/producer provenance |
| `EmbeddedSeriesQuery` | Immutable pre-encoded values plus representation fingerprint |

Plans contain immutable specifications, logical inputs, column references, and
opaque query/provider keys. They do not retain mutable tensors, DataFrames,
model instances, credentials, or SQL fragments as their semantic definition.

`SeriesRepresentationPlan` is intentionally separate from window construction.
For a native recipe the compiler lowers it to projections and list/array
expressions. For a learned recipe it lowers to the bounded Arrow provider
boundary. Both produce the same ordinary fixed-size vector column contract.
A common numerical distance compiler can serve existing vector search and the
new series search; do not duplicate the metric definitions.

### Native lowering requirements

Window construction must first produce an ordered list and count/validity
information in a window projection. A subsequent projection checks completeness
and casts to `FLOAT[L]`. Normalization belongs in a later projection or the
representation node. Do not generate nested window functions while combining
`pct_change()` or other lagged expressions with rolling arrays.

Conceptually, for `L=3`:

```sql
-- Illustrative lowering, not a replacement for typed compiler expressions.
WITH packed AS (
    SELECT
        *,
        list(CAST(return_1m AS DOUBLE)
             ORDER BY timestamp, observation_id) OVER w AS raw_window,
        count(*) OVER w AS window_rows,
        count(return_1m) OVER w AS valid_rows
    FROM input_relation
    WINDOW w AS (
        PARTITION BY ticker
        ORDER BY timestamp, observation_id
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )
)
SELECT
    CASE
        WHEN window_rows < 3 OR valid_rows < 3 THEN NULL::FLOAT[3]
        ELSE CAST(raw_window AS FLOAT[3])
    END AS return_window
FROM packed;
```

The actual lowering additionally checks finite source and post-cast values,
retains required hidden identity/order columns, and uses compiler-owned quoted
identifiers. The example only illustrates frame and packing semantics. DuckDB's
ordered window aggregates and array distance functions provide the underlying
building blocks; API-level invariants still require DuckPD validation.
[Sources: DuckDB window functions][duckdb-windows] and
[array functions][duckdb-arrays].

For multichannel native encoding, normalize each channel separately and then
concatenate in the specification's channel-major order. The query path must use
the same normalization/layout implementation or a tested numerically equivalent
implementation, not a subtly different Python formula.

### Required optimizer behavior

1. **Do not move filters across window construction** unless preserving the
   exact contributing rows is proved. A date filter on output endpoints does not
   permit dropping the required preceding observations or partitions.
2. **Do not move filters across top-k.** Candidate eligibility and post-ranking
   filtering are distinct operations.
3. **Do not prune hidden dependencies.** Liveness includes window operands,
   partition keys, ordering/tie-breaker columns, event keys/times/availability,
   provider input channels, and vector tie-breakers.
4. **Treat provider calls as explicit boundaries.** No automatic common-subplan
   materialization or reordering across inference. Initially keep conservative
   barriers even for potentially safe row-local rewrites, because error behavior
   and repeated inference costs are observable.
5. **Handle binary event plans explicitly.** Update recursive rewrite, execution,
   validation, source discovery, serialization, and profiling walkers. An
   `EventWindowPlan` does not have the generic unary `input` assumed by current
   optimizer recursion.
6. **Retain immutability and idempotence.** Rewrites must produce valid metadata,
   stable fingerprints, and fixed-point behavior under the existing passes.

Safe source bounds can be pushed only where the optimizer knows their dependency
closure. For fixed-grid event windows that closure follows from the event bounds
and offsets. For observation-count rolling windows it is not generally a fixed
wall-clock interval. Until implemented, the caller must request sufficient input
history; selecting a narrow feature-store date range then applying rolling does
not automatically fetch missing predecessors.

The source review for these requirements covers the current
[compiler][src-compiler], [optimizer][src-optimizer], [logical expression
schema][src-logical], and [metadata transition helpers][src-metadata].

### Source/module implementation map

| File | Planned changes |
| --- | --- |
| `src/duckpd/series_embeddings.py` (new) | Specs, canonicalization, typed queries, provider protocol, validation, public constructors |
| `src/duckpd/event_windows.py` (new) | Event-window API validation and plan construction; no model dependency |
| `src/duckpd/window.py` | Fixed-count `to_array()` for selected/grouped receivers; preserve grouped assignment alignment |
| `src/duckpd/_logical.py` | Array-window expression, four new plans, additive column metadata, expression/plan unions |
| `src/duckpd/_metadata.py` | Row-preserving representation transitions and explicit metadata propagation helpers |
| `src/duckpd/_compiler.py` | Native array/normalization lowering, event grid, provider call, shared search lowering |
| `src/duckpd/_executor.py` | Preflight, new plan traversal/validation, progress and resource metrics, failure cleanup |
| `src/duckpd/_optimizer.py` | Binary-plan traversal, liveness/expression mapping, conservative barriers and explain serialization |
| `src/duckpd/session.py` | Provider registration/preparation, immutable query storage, bounded cache, sidecar/table metadata lifecycle |
| `src/duckpd/frame.py` | `embed_series()`, `event_windows()`, metadata assertion, alias/sink integration |
| `src/duckpd/series.py` | Typed-query distance integration points and representation propagation through Series projections |
| `src/duckpd/vector.py` | `search_series()`, typed numeric/text query compatibility validation, existing exact/ANN rules retained |
| `src/duckpd/featurestore.py` | Registry lookup, typed declared columns, alias/exact/ASOF metadata propagation |
| `src/duckpd/_feature_catalog.py` | Strict catalog-version-1 series registry validation |
| `src/duckpd/__init__.py` | Export new public specifications, constructors, and typed query |
| Optional provider module/package | Reviewed adapter and dependency isolation; no import-time model/runtime loading in core |
| `tests/` and `docs/COMPATIBILITY.md` | New contract tests and truthful support matrix; preserve existing text/vector tests |

This map describes responsibilities rather than prescribing a large up-front
refactor. Reuse small existing helpers where their semantics actually match.
In particular, do not reuse a text-specific metadata transition that records
`embed_text` for a series operation without making the operation tag explicit.

## Explain and profiling

Non-executing explain modes for the new operations must not download, prepare,
or run models. Logical/optimized/JSON output uses the immutable plan. SQL and
physical previews must use safe typed placeholders/stubs where query inference
would otherwise be constant-folded during compilation, and label that preview
rather than pretending a model was executed.

`explain("analyze")`, `profile()`, and eager sinks execute work under their
existing contracts; they are not covered by the non-executing promise.
[Source: current explain/consumer entry points][src-frame].

Add observable fields without exposing raw series/query contents:

```text
series_operations:
    operation: window | representation | search | event_window
    representation_fingerprint
    provenance: generated | catalog_declared | application_asserted | sidecar_bound
    encoder_backend: native | qualified backend
    input_channels, input_length, output_dimension
    sampling, step, outer_normalization, encoder_normalization, unit_norm
    compatibility_check: representation | dimension_only
    null_policy, zero_scale_policy
    inference_boundary: none | arrow
    batch_size
    query_encoding: deferred | supplied_typed_vector
    strategy: exact | existing_explicit_approximate_path
    filter_placement
    model_prepared
    recomputes_corpus_on_execution
```

Execution metrics include input/complete/incomplete windows, propagated nulls,
provider calls and valid rows, query/corpus inference time, batch sizes, model
preparation time when explicitly requested, source transfer, sidecar integrity
bytes/time, output bytes where available, event-grid cardinality, and vector
strategy. Report actual index use only after verification through existing index
mechanisms. Do not infer HNSW use from a requested flag.

Explain/reporting must redact query values, news text, credentials, sensitive
source paths under the existing sanitization policy, and model-cache contents.
Model identifiers, fingerprints, shapes, and aggregate counts are sufficient for
diagnosis.

## Errors

Follow existing DuckPD exception conventions: eager type/value/unsupported
operation errors for known-invalid plans and `MaterializationError` wrapping
runtime data/provider failures. Add stable diagnostic prefixes only where useful;
do not introduce a parallel exception hierarchy solely for this feature.

| Condition | Required timing and message content |
| --- | --- |
| Unsupported rolling mode or weak `min_periods` | Eager; explain fixed-count/full-window requirement |
| Missing order for a rolling operation | Eager where known; preserve the existing unordered-operation behavior |
| Wrong channel mapping, dimension, or specification | Eager when metadata proves it; identify conflicting field names |
| Unprepared/unregistered learned provider | Execution preflight; identify exact model and explicit preparation operation |
| Missing or mismatched representation metadata | Before query inference; explain assertion versus raw-vector escape hatch |
| Null/invalid source array under declared policy | Execution; include column and bounded identifying context, not whole data payloads |
| Constant normalized query or zero cosine query | Query encoding/validation; do not return arbitrary neighbors |
| Provider row-count/type/finite-output failure | Execution; identify provider/specification and expected versus actual shape |
| Event duplicate keys or off-grid/missing slots | Execution; duplicates are errors; missing slots follow `incomplete` |
| Catalog schema or unsupported backend | Catalog load/plan construction; never start installing dependencies |
| Sidecar mismatch or incompatible partitions | Before verified metadata is used; no silent first-file selection |
| Unsupported approximate filtered search | Retain existing explicit rejection; do not widen candidates silently |

An early-stopped streaming consumer may not examine every later input row.
Validation claims must distinguish metadata checks, whole-input key checks where
required, and per-batch value checks. A successful complete sink validates all
rows it consumed; it does not certify an unconsumed upstream dataset.

## Model candidates and qualification

The encoder interface is model-agnostic. The following findings inform adapter
priorities; they are not retrieval-quality benchmarks on financial data.

| Candidate | Verified interface or characteristic | Proposed treatment |
| --- | --- | --- |
| Native ordered vectors | Deterministic recipe defined here; no pretrained model | Required baseline and first release |
| MOMENT | Explicit `embed()` returns representations; current implementation normalizes inputs internally and can average channels/patches | First built-in learned candidate, subject to context, masks, channel preservation, licensing, and task qualification |
| TS2Vec | Explicit `encode()` supports full-series and other pooled representations | Support through the custom-provider contract; suitable for a separately trained/checkpointed encoder |
| PatchTST | Official self-supervised implementation exposes an encoder architecture | Experimental adapter needs an exact checkpoint, extraction location, channel policy, and pooling contract |
| Original Chronos-T5 | `ChronosPipeline.embed()` explicitly returns encoder states and tokenizer state | Valid experimental representation candidate; pool states with explicit padding/EOS handling and scale policy |
| TimesFM 3.0 | Current official documentation is forecasting-first; released weights have a non-commercial license | Not the default production encoder; a qualified embedding adapter and compatible usage rights are separate requirements |
| TRACE | Explicit time-series/text alignment and retrieval research workflow | Later shared-space research, not a baseline finance-ready provider |

Sources: [MOMENT embedding implementation][model-moment], [TS2Vec encoder][model-ts2vec],
[PatchTST self-supervised source][model-patchtst], [Chronos embedding implementation][model-chronos],
[TimesFM official README][model-timesfm], [TimesFM 3.0 model card][model-timesfm-card],
and [TRACE implementation][model-trace]. Model-source review date is 2026-09-07;
provider releases must pin their own immutable revisions rather than depend on
these moving reference branches.

Important distinctions:

- A TS2Vec implementation is not automatically a pretrained finance checkpoint.
  Its full-series pooling and context choices must be fixed by the adapter.
- MOMENT's default channel averaging may discard distinctions important to a
  multichannel feature family. A channel-preserving alternative changes the
  representation and potentially its dimension; specify it rather than assuming
  every `embed()` output is interchangeable.
- Chronos-T5's explicit embedding method is different from accessing an arbitrary
  forecasting model's hidden states. Its output is still a sequence of states,
  not a prequalified fixed-length financial retrieval vector. Do not generalize
  this particular method contract to Chronos-Bolt or Chronos-2 without inspecting
  their own implementations.
- Forecasting benchmarks are not evidence of nearest-neighbor relevance for
  price-shape or news-reaction matching. No "best model" claim is made here.

Every shipped learned adapter must publish a qualification record containing the
exact checkpoint and artifact digest, source/package versions, source-code and
weights license identifiers, supported Python/platform/runtime matrix, required
input length/channels, all internal normalization, mask/padding semantics, pooling,
output dimension, CPU/memory behavior, determinism tolerances, and retrieval
benchmark results against native baselines.

The first built-in MOMENT adapter should support only a context length actually
qualified with its checkpoint. Do not copy a guessed embedding dimension or
assume an arbitrary 60-sample input is supported because the desired user window
has that length. Until qualified, the stable release can contain only native
encoding plus a custom-provider protocol.

## Test plan

Core tests must use deterministic fake providers and synthetic time series, with
no network or heavyweight optional dependency. Optional adapter tests run in a
separate dependency matrix and are not required merely to import DuckPD.

### Native windows and normalization

Cover lengths 1 and greater, warm-up, empty inputs, short groups, interleaved
entities, group null-key policies, timestamps with explicit ties, selected
DataFrame columns, standalone grouped output, and same-source assignment.
Verify oldest-to-newest order and no cross-group windows.

Test missing observations without compression, null versus nonfinite values,
float32 overflow, weaker `min_periods`, time-range rejection, and centered-window
rejection. Test history spanning files and Arrow batches. A physical batch
boundary must never reset a logical rolling window.

For each native recipe, compare corpus and query encodings against a small
independent reference implementation. Test Pearson/cosine equivalence for
nonconstant centered windows, positive rescaling invariance, direction reversal,
constant series, zero norm, huge/small magnitudes, per-channel ordering, and
z-score `ddof=0`. Explicitly demonstrate that mean-centering can erase momentum
and that unit normalization can erase volatility amplitude.

### Provider and lifecycle tests

Use a recording fake provider to assert no preparation or inference during
planning, bounded calls, no calls for propagated-null rows, correct scatter-back,
query encoding once per execution, immutable query snapshots, and bounded cache
behavior. Exercise different batch sizes and input permutations. Verify that
calls to a provider declaring `thread_safe=False` never overlap, while a
thread-safe fake follows the executor's documented concurrency policy.

Reject wrong row counts, output dimensions, null children, NaN/infinite output,
mismatched artifact digests, incomplete or mismatched preparation attestations,
unprepared providers, and unsupported model lengths. Assert preparation fails
before executing source work in the new path. Verify that repeated eager
consumers recompute an unpersisted corpus rather than silently creating a hidden
cache.

### Metadata, serialization, and compatibility

Add golden canonicalization/hash fixtures and prove legacy text fingerprints are
unchanged. Test rename/alias/assignment/Series projections, joins, ASOF payloads,
compatible and conflicting concatenation, and metadata invalidation after value
transforms. Exercise `SeriesRepresentationAssertionPlan` through every consumer,
including an early limit and a complete sink, and verify that only consumed rows
are claimed as validated. Include hostile column labels to exercise compiler
quoting.

Test local Parquet series sidecars, mixed text/series files, corrupt manifests,
wrong file binding, interrupted writes at every publication step, stale sidecar
removal, and mismatched generation values. Assert that a mixed artifact never
restores only one modality. Also test missing/conflicting partition metadata,
list/fixed-array round-trips, table append/overwrite, and session reopen without
falsely claiming persisted registry state.

### Search and event correctness

Compare exact search to brute-force distance ranking under every supported
metric. Test ties, unique tie-breakers, null candidates, metadata mismatch,
plain-vector compatibility bypass, and exclusion filters before versus after
ranking. Retain existing ANN rejection and verified-index tests.

For event grids, test events exactly on a boundary and between boundaries,
pre/post windows, duplicate source/event keys, multiple entities, multiple events
at one timestamp, missing bars, market closures, all-incomplete windows, empty
events, negative epoch timestamps, UTC conversions, timestamp precision, and
non-default metadata prefixes. Reject an omitted or unsupported `bar_label`.
Assert that off-grid rows inside the bounded extraction union fail while rows
outside it are not scanned or validated. Assert array positions, count,
start/end, and availability independently. Test revisions and late-arriving bars
with explicit input histories; do not synthesize unavailable revisions inside
the test helper.

Verify that exact fused ranking scores the complete eligible set. Include a
counterexample where an item outside either modality's small top-k is the best
weighted combination, proving why candidate-limited reranking is not exact.

### Feature-store and optimizer regression

Test the complete catalog version 1 schema, registry resolution, unknown fields,
unsupported backends, alias metadata, `FLOAT[D]` logical typing before partition
reads, physical schema mismatch, exact and point-in-time alignment, and no model
preparation from metadata-only access. Preserve existing remote partition and
scan-guard tests.

Run new plans through every optimizer pass and all explain modes. Test expression
liveness, binary-plan traversal, hidden order/key dependencies, filter boundaries,
required history before a selected date range, and fixed-point rewrites. Assert
that progress reporting does not introduce a full count scan.

### Retrieval-quality evaluation

Build a small reviewed benchmark around the actual product question, not
forecasting loss. Include trend continuation/reversal, impulses and recovery,
volatility bursts/decay, step changes, flat periods, and real event-linked windows.
Use controlled perturbations for scale, offset, noise, time shift, sign reversal,
and missingness; the expected neighbors depend on the declared family semantics.

Report precision/recall or nDCG on labeled neighbors, neighborhood stability,
representation/inference cost, storage, and exact query latency. Evaluate shape,
volatility-level, and event-fusion tasks separately. Use held-out events/entities
and chronology-aware splits where evaluating generalization; exclude overlapping
windows and duplicate news clusters that would inflate retrieval scores.

A learned adapter is promoted only if its measured benefit on a named retrieval
task justifies its cost. Native recipes remain supported even if a learned model
wins another task. Matching prices alone does not validate the text encoder or a
shared text/time-series space.

## Rollout and acceptance gates

### Phase 0: contracts and compatibility foundations

Add immutable specs, canonicalization, typed queries, metadata propagation
helpers, and no-network fake providers. Record the unchanged legacy text hash
fixtures, define the new persistence formats and catalog-version-1 registry
additions, and land any necessary alias/ASOF metadata fixes with regression tests.

**Exit gate:** specifications serialize deterministically; typed-query mismatches
fail; all existing text/vector/feature-store tests remain valid. No model backend
is required.

### Phase 1: native sequence representations

Implement fixed-count `to_array()`, native `embed_series()`, high-level exact
`search_series()`, typed numeric distance/search, and local persistence. Document
shape versus level and the required warm-up/filter semantics.

**Exit gate:** ordered-window and brute-force retrieval oracles pass; source/query
native vectors agree within defined float tolerances; direct sinks operate
without pandas materialization. This phase is independently useful and shippable.

### Phase 2: event windows and event similarity

Implement fixed-grid `event_windows()`, explicit availability metadata, exact
late-fusion examples, and event-join/overlap tests. No joint model is introduced.

**Exit gate:** boundary, missing-bar, duplicate-key, and availability cases are
correct; fused ranking is distinguished from candidate reranking. This completes
the core news-plus-market-reaction exploration workflow without learned models.

### Phase 3: feature-store catalog declarations

Implement registries and typed metadata propagation in catalog version 1 after
the companion text registry contract is resolved. Support deterministic catalogs
first; add qualified portable model backends under explicit preparation policy.

**Exit gate:** catalog-version-1 fixtures, alias/ASOF metadata preservation,
physical validation, offline behavior, and model trust-policy tests pass. No
catalog call generates or refreshes corpus embeddings.

### Phase 4: optional learned inference

Implement the provider lifecycle and Arrow execution path with custom providers,
then qualify one built-in adapter, initially MOMENT if it passes. Include exact
asset pinning, runtime isolation, mask/context/channel semantics, and benchmark
reports. Add TS2Vec or other adapters without changing the DataFrame/search API.

**Exit gate:** bounded provider calls, deterministic query/corpus agreement,
accurate resource reporting, no hidden normalization, and a documented retrieval
use case where the adapter is useful. Do not block native release on this gate.

### Later research: aligned multimodal spaces

Evaluate a finance-specific aligned encoder pair, potentially informed by TRACE,
and design a shared-space metadata/query contract separately. Require evidence
for text-to-reaction and reaction-to-text retrieval on held-out paired events.
No public cross-modal API is promised before that work.

## Decisions deliberately deferred

The architecture above is decided for this proposal. The following require
implementation experiments rather than invented defaults: the first qualified
learned checkpoint and output dimension; tolerances across provider devices;
practical batch defaults by adapter; event-grid workload warning/budget thresholds;
and whether a proven Parquet footer-generation mechanism should replace the
initial full-file sidecar integrity pass.

Automatic resampling, variable-length masked inputs, trading-calendar windows,
DTW reranking, learned score calibration, ANN with eligibility filters, durable
cross-session metadata registries, and incremental feature refresh each need a
separate extension. They must not enter the initial implementation as undocumented
special cases.

## Review and validation record

The repository review covered the relevant public API, current design documents,
logical/metadata model, rolling and grouping integration, vector/text embedding
paths, source/compiler/executor/optimizer behavior, persistence, feature-store
planning/catalog parsing, and associated embedding/vector tests. Repository
references below are pinned to the reviewed commit.

This is a design artifact, not an implementation or a claim that the full DuckPD
test suite or any learned provider was executed. Proposed Python blocks are API
usage fragments; model revision/digest placeholders require real qualified
artifacts. The illustrative SQL is not represented as a tested patch. Document
structure, reference completeness, JSON/Python fragment syntax, and independent
normalization/event-grid reference checks are validated separately during drafting.

## References

### Reviewed DuckPD snapshot

The [reviewed commit][repo-snapshot] fixes the implementation baseline. Start with
[architecture directives][adr-architecture], [text embeddings][design-text],
[vector search][design-vector], and the [feature-store catalog proposal][design-catalog];
implementation-specific links are attached to their corresponding sections above.

[repo-snapshot]: https://github.com/quantbert/duckpd/commit/fc1fce3aca0fc8028f685188875d6ff93338fa66
[design-text]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/docs/design/text-embeddings.md
[design-vector]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/docs/design/vector-search.md
[design-store]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/docs/design/featurestore-architecture.md
[design-catalog]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/docs/design/featurestore-embeddings.md
[adr-architecture]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/docs/decisions/0003-directives-and-architecture.md
[src-window]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/src/duckpd/window.py
[src-embeddings]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/src/duckpd/embeddings.py
[src-vector]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/src/duckpd/vector.py
[src-logical]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/src/duckpd/_logical.py
[src-metadata]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/src/duckpd/_metadata.py
[src-session]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/src/duckpd/session.py
[src-frame]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/src/duckpd/frame.py
[src-compiler]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/src/duckpd/_compiler.py
[src-executor]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/src/duckpd/_executor.py
[src-optimizer]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/src/duckpd/_optimizer.py
[src-store]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/src/duckpd/featurestore.py
[src-catalog]: https://github.com/quantbert/duckpd/blob/fc1fce3aca0fc8028f685188875d6ff93338fa66/src/duckpd/_feature_catalog.py

### Model and database sources

Primary implementation references are [MOMENT][model-moment], [TS2Vec][model-ts2vec],
[PatchTST][model-patchtst], [Chronos][model-chronos], [TimesFM][model-timesfm], and
[TRACE][model-trace]. The [TimesFM model card][model-timesfm-card] records its
checkpoint license. Native lowering relies on documented [window][duckdb-windows]
and [array][duckdb-arrays] operations; actual adapter/runtime revisions must be
pinned during implementation qualification.

[model-moment]: https://github.com/moment-timeseries-foundation-model/moment/blob/main/momentfm/models/moment.py
[model-ts2vec]: https://github.com/zhihanyue/ts2vec/blob/main/ts2vec.py
[model-patchtst]: https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_self_supervised/src/models/patchTST.py
[model-chronos]: https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos.py
[model-timesfm]: https://github.com/google-research/timesfm/blob/master/README.md
[model-timesfm-card]: https://huggingface.co/google/timesfm-3.0-pytorch
[model-trace]: https://github.com/Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder
[duckdb-windows]: https://duckdb.org/docs/stable/sql/functions/window_functions.html
[duckdb-arrays]: https://duckdb.org/docs/stable/sql/functions/array.html
