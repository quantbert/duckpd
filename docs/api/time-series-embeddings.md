# Time-Series Representations and Similarity Search

DuckPD supports model-free time-series similarity by turning ordered,
fixed-length numeric windows into typed vectors. These vectors can be searched
with the same exact vector engine used for other numeric data.

The current implementation ships deterministic native encoding and an explicitly
registered custom-provider boundary. DuckPD does not ship or recommend a learned
time-series model. Native and application-owned learned paths use the same window,
representation, metadata, and retrieval contracts described here.

The native rolling-window walkthrough is
[`DuckPD_Time_Series_Embeddings.ipynb`](../../demo/DuckPD_Time_Series_Embeddings.ipynb);
the dedicated event workflow is
[`DuckPD_Event_Windows_and_Exact_Fusion.ipynb`](../../demo/DuckPD_Event_Windows_and_Exact_Fusion.ipynb).
The lower-level architecture and proposed extensions are documented in the
[time-series embedding design](../design/time-series-embeddings.md).

## Current implementation at a glance

The shipped workflow has three stages:

1. Construct explicitly ordered, fixed-size windows for one or more numeric
   channels.
2. Apply a declared deterministic transformation to produce a `FLOAT[n]`
   representation.
3. Use exact vector distance or top-k retrieval to find similar windows.

For example, an eight-observation representation with two channels has 16
values:

```text
ordered rows
    |
    +-- bar_return ------> FLOAT[8] rolling window --+
    |                                                |
    +-- intrabar_range --> FLOAT[8] rolling window --+--> normalize
                                                         and flatten
                                                              |
                                                              v
                                                          FLOAT[16]
                                                              |
                                                              v
                                                       exact vector search
```

There is no neural-network inference in this path. `embed_series()` compiles
the transformation into DuckDB list and array expressions and remains lazy
until an execution boundary such as `collect()` or `write_parquet()`.

## Why this is called an embedding

An embedding is a representation of an object in a vector space where distance
has a declared meaning. It does not have to be produced by a learned model.

Text generally needs a language model because strings have no directly useful
continuous geometry: token IDs or character codes do not make semantically
similar documents nearby. A numeric time-series window already has an ordered
numeric structure. With an explicit choice of channels, preprocessing, and
distance metric, it can serve directly as an interpretable vector
representation.

DuckPD therefore uses **representation** as the more precise general term:

- A **native representation** is produced by deterministic DuckDB expressions.
- A **learned representation** is produced by an explicitly selected and
  prepared application-owned encoder.
- Both are embeddings when their vectors are used as coordinates for
  similarity retrieval.

The native representation is not a placeholder for a model. It is a useful
baseline with properties that a learned encoder must measurably improve for a
specific retrieval task.

## End-to-end example

The following example represents rolling return and intrabar-range patterns:

```python
import duckpd as pd

WINDOW = 8

with pd.connect() as session:
    prices = session.read_parquet(
        "market-data.parquet",
        order_by=["ticker", "timestamp"],
    )

    features = prices.assign(
        bar_return=lambda frame: (frame["close"] - frame["open"]) / frame["open"],
        intrabar_range=lambda frame: (frame["high"] - frame["low"]) / frame["open"],
    )

    windows = features.assign(
        return_window=lambda frame: (
            frame.groupby("ticker")["bar_return"].rolling(WINDOW).to_array()
        ),
        range_window=lambda frame: (
            frame.groupby("ticker")["intrabar_range"].rolling(WINDOW).to_array()
        ),
    )

    representation = pd.series_representation(
        window=WINDOW,
        channels=("bar_return", "intrabar_range"),
        sampling="observations",
        data_contract="market/ohlc-shape/v1",
        normalization="center",
        unit_norm=True,
        zero_scale="null",
    )

    embedded = windows.embed_series(
        columns={
            "bar_return": "return_window",
            "intrabar_range": "range_window",
        },
        into="market_shape",
        representation=representation,
    )

    candidates = embedded[embedded["market_shape"].notna()]
    query_row = candidates[["return_window", "range_window"]].head(1)
    query = {
        "bar_return": tuple(query_row.iloc[0]["return_window"]),
        "intrabar_range": tuple(query_row.iloc[0]["range_window"]),
    }

    matches = candidates.vector.search_series(
        query,
        column="market_shape",
        representation=representation,
        metric="l2",
        k=10,
        tie_breaker="timestamp",
    )
    print(matches[["ticker", "timestamp", "_distance"]].collect())
```

Constructing `windows`, `embedded`, and `matches` builds lazy plans. `head(1)`
executes once to obtain the example query, and the final `collect()` executes
the corpus search.

## How the native representation is calculated

Let a window contain `C` channels and `L` observations per channel. Channel
`c` is the ordered sequence

```text
x_c = [x_c,1, x_c,2, ..., x_c,L]
```

DuckPD validates complete finite inputs and then applies the representation
contract in this order.

### 1. Normalize each channel independently

`normalization="none"` retains the channel exactly:

```text
z_c,i = x_c,i
```

`normalization="center"` removes the within-window channel mean:

```text
z_c,i = x_c,i - mean(x_c)
```

`normalization="zscore"` also divides by the population standard deviation:

```text
z_c,i = (x_c,i - mean(x_c)) / stddev_pop(x_c)
```

Statistics are calculated separately for every complete channel window using
double-precision intermediates. They are not estimated across tickers, rows, or
an inference batch.

### 2. Flatten in a stable layout

Native representations use channel-major, oldest-first layout. For channels
`("bar_return", "intrabar_range")`, DuckPD produces:

```text
[
  return_oldest, ..., return_newest,
  range_oldest,  ..., range_newest,
]
```

`representation.channels` controls this order. The insertion order of the
`columns={...}` mapping does not change the vector layout.

For a native representation, the output dimension is:

```text
dimension = window * number_of_channels
```

### 3. Optionally normalize the complete vector

With `unit_norm=True`, DuckPD applies L2 normalization after flattening:

```text
v_normalized = v / sqrt(sum(v_i * v_i))
```

This removes overall vector magnitude while retaining direction and relative
contributions between channels.

### 4. Produce a typed vector

Every valid output is a fixed-size `FLOAT[dimension]`. The implementation
rejects nonfinite values, partial nulls, incompatible window widths, and invalid
output values instead of silently repairing them.

## Interpreting distance

The meaning of a nearest neighbor comes from the representation and metric
together. It is not supplied by the word "embedding."

### Centered shape similarity

For two nonconstant vectors, cosine similarity after mean centering equals
Pearson correlation:

```text
cosine(x - mean(x), y - mean(y)) = correlation(x, y)
```

Cosine distance is therefore `1 - correlation` for this particular recipe.
Opposite-direction patterns have negative correlation and are not considered
close. Taking an absolute correlation would define a different representation
semantics.

### L2 and cosine for unit vectors

For unit-normalized vectors `x` and `y`:

```text
L2(x, y)^2 = 2 - 2 * cosine_similarity(x, y)
```

L2 and cosine consequently produce the same ordering in the notebook, although
their reported distance values differ.

### Inner product

Inner product is sensitive to vector magnitude unless vectors are normalized.
DuckPD reports negative inner product so that all supported metrics retain the
same convention: smaller values are nearer.

## Choose a representation by meaning

Normalization decides which information retrieval preserves or discards.
There is no universally correct time-series embedding.

| Retrieval question | Initial representation | Preserved | Removed |
| --- | --- | --- | --- |
| Which return windows have a similar fluctuation shape? | Centered returns, unit norm, cosine or L2 | Signed temporal shape | Mean and overall magnitude |
| Which paths have similar direction and cumulative movement? | Upstream cumulative returns, no centering, L2 | Direction and path magnitude | Nothing automatically |
| Which volatility windows have similar burst shape? | Centered absolute-return or volatility channel, unit norm | Relative peaks and decay | Absolute volatility level |
| Which windows have similar volatility level? | Uncentered volatility in fixed units, L2 | Level and evolution | Nothing automatically |
| Which multivariate states have similar raw values? | Uncentered channels with deliberate scaling, L2 | Declared levels | Nothing automatically |
| Which windows share higher-level structural motifs? | Qualified learned encoder | Depends on training and adapter | Must be measured and documented |

Keep scalar descriptors such as cumulative return, realized volatility,
liquidity, duration, and event category beside the representation. They remain
available for explicit filtering and make retrieved neighbors easier to
interpret. A shape vector should not be forced to retain information that its
normalization intentionally removes.

## Ordering, grouping, and sampling

Time-series vectors are meaningful only when their input order is meaningful.

### Declare row order

Use `order_by` when loading or constructing the frame. Physical Parquet or CSV
row order is not a chronological contract. If timestamps can tie, include a
stable application-level tie-breaker.

```python
prices = session.read_parquet(
    "prices/*.parquet",
    order_by=["ticker", "timestamp", "observation_id"],
)
```

### Keep entities isolated

Grouped rolling windows prevent one entity's history from entering another
entity's representation:

```python
frame.groupby("ticker")["return"].rolling(60).to_array()
```

The first `window - 1` rows in each group are null because DuckPD does not pad
or synthesize missing history.

### Understand observation sampling

`sampling="observations"` means the window contains the last `L` rows. It does
not prove those observations are equally spaced in wall-clock time. A nominal
`step` documents expected cadence but does not fill gaps or validate a grid.

Use `event_windows()` when elapsed-time comparability must be enforced rather
than assumed. It validates a UTC epoch grid and retains missing slots as an
incomplete window instead of compressing the next available observations.

### Build event-aligned fixed-grid windows

`DataFrame.event_windows()` pairs an observation frame with event rows through
explicit entity and event-observation keys:

```python
reactions = prices.event_windows(
    news,
    on="bar_start",
    bar_label="start",
    event_on="published_at",
    by="ticker",
    event_id=("event_observation_id", "revision"),
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

The receiver supplies numeric observations. The result retains every event
column, appends one `FLOAT[60]` array per mapping entry, and adds
`reaction_window_start`, `reaction_window_end`, `reaction_window_count`,
`reaction_window_complete`, and `reaction_window_available_at`.

The grid is anchored to the Unix epoch in UTC. For offsets `[a, b)` and cadence
`s`, DuckPD selects the exact timestamps `anchor_time + j * s` for every integer
`j` in that interval. `floor` anchors at or before the event; `ceil` anchors at
or after it, with both choosing the event timestamp when it is exactly on-grid.
`bar_label="start"` is currently required, and each bar's availability must be
at or after its complete bar end.

Missing bars never cause a nearest-time match, forward fill, or compressed
array. With `incomplete="null"`, the event row remains but every requested array
and its window availability are null. With `"error"`, collection fails. Null or
nonfinite values, off-grid rows inside the bounded event intervals, duplicate
`(by, on)` observations, null keys, and duplicate `event_id` values are errors.
Use a composite `event_id` when revisions are distinct observations.

For a complete window, availability is:

```text
max(event_available_at, available_at for every required bar)
```

Filtering event rows before extraction is allowed. Filtering observation rows
before extraction changes completeness. Output eligibility filters remain above
the event-window optimizer barrier, so an availability cutoff cannot truncate
the source history needed to build a window.

The result's fixed-grid metadata composes directly with `embed_series()` by
using a representation with the same window length and `step`:

```python
reaction_space = pd.series_representation(
    window=60,
    channels=("simple_return", "volatility"),
    sampling="fixed_grid",
    step="PT1M",
    data_contract="market/reaction/v1",
    normalization="center",
    unit_norm=True,
)

event_bank = reactions.embed_series(
    columns={
        "simple_return": "return_window",
        "volatility": "volatility_window",
    },
    into="reaction_vector",
    representation=reaction_space,
)
```

Text-first retrieval searches news before inspecting reactions. Reaction-first
retrieval searches `reaction_vector` and keeps the event key for the news join.
For exact late fusion, filter the full eligible event bank, compute both
distances with `vector.distance()`, combine them under an explicit scoring rule,
and only then apply `nsmallest()`. Searching one modality first and reranking
that bounded result is candidate-limited; it is not an exact ranking of the
combined score over the full eligible population. Exclude the query event and
application-defined overlapping observations explicitly.

### Place filters relative to history and retrieval

A filter before `rolling(...).to_array()` changes the observations available to
each window. A filter after `embed_series()` but before `search_series()` keeps
the established history and restricts the eligible candidate endpoints. A
filter after `search_series()` filters the bounded top-k result and is not
equivalent to filtering candidates before ranking. DuckPD does not move filters
across window construction or exact top-k boundaries.

## Representation identity and safety

Vector width alone cannot establish compatibility. Two `FLOAT[16]` columns may
represent different channels, sampling rules, units, normalization, or models.

`SeriesRepresentationSpec` includes:

- Window length and ordered semantic channel names.
- Observation or fixed-grid sampling and optional cadence.
- A versioned application data contract.
- Native normalization, unit normalization, and zero-scale policy.
- The complete encoder identity when a learned encoder is used in the future.
- A versioned native vector layout.

DuckPD serializes this contract canonically and hashes it into a SHA-256
fingerprint. `search_series()` resolves the contract from verified column
metadata and represents raw channel windows only when the search executes. An
explicit representation is a compatibility assertion, not an override.
`Session.embed_series_query()` provides an eager reusable `EmbeddedSeriesQuery`.
Both typed paths reject incompatible columns during planning, even when their
dimensions match. Plain numeric `vector.search()` remains the explicitly
unverified low-level path.

The `data_contract` identifies upstream semantics such as return convention,
price adjustment, clipping, units, and feature definitions. DuckPD preserves
and compares the identifier; the application remains responsible for ensuring
that the actual source data follows it.

## Nulls, constant windows, and invalid values

Rolling warm-up rows are null until a complete window exists.

- `null_policy="propagate"` makes any null input window produce one whole null
  output vector.
- `null_policy="error"` aborts execution when a null window is encountered.
- NaN, infinity, float32 overflow, partial null children, and wrong dimensions
  are malformed inputs and fail under either policy.

Centering a constant channel produces zeros. Z-scoring a constant channel is
undefined, and unit-normalizing an all-zero vector is also undefined.

- `zero_scale="null"` turns such a corpus representation into a whole null
  vector.
- `zero_scale="error"` aborts execution.

DuckPD does not insert an undocumented epsilon, silently clip values, or
substitute arbitrary zero vectors. A zero query is invalid for cosine distance.

## Strengths of the native approach

### Interpretability

Every coordinate has a known origin. Distance behavior follows directly from
the declared normalization and metric, making surprising neighbors easier to
investigate.

### Reproducibility

The result does not depend on remote model revisions, hardware kernels, learned
weights, or inference-library versions. The representation fingerprint records
the complete declared space.

### Operational simplicity

There is no model download, GPU requirement, Python row callback, or inference
service. Native transformations remain in DuckDB and can flow directly into a
Parquet sink without pandas materialization.

### A strong evaluation baseline

Many shape-retrieval tasks are already well represented by normalized windows.
A learned encoder should be compared against this baseline rather than assumed
to be better because it is more complex.

### Strict contracts

DuckPD validates ordering provenance, window width, channel compatibility,
finite values, fixed output type, and representation identity. Errors surface
instead of changing the representation implicitly.

## Feature-store catalog vectors

Catalog version 1 can publish native series vectors without making callers
repeat their representation:

```json
{
  "catalog_version": 1,
  "series_embedding_models": {},
  "series_representations": {
    "return-shape-8": {
      "version": 1,
      "window": 8,
      "channels": ["simple_return"],
      "sampling": "fixed_grid",
      "step": "PT1M",
      "data_contract": "simple-return-left-zero-padded/v1",
      "normalization": "none",
      "unit_norm": false,
      "zero_scale": "null",
      "encoder": null
    }
  },
  "features": {
    "returns:shape_8": {
      "dataset": "returns",
      "name": "shape_8",
      "availability_delay": "PT0S",
      "lookahead_safe": true,
      "series_representation": "return-shape-8"
    }
  }
}
```

Reference tables use the same registry key under
`datasets[].columns.<column>.series_representation`. The referenced Parquet
column must be a fixed-size `float32[D]` list matching the resolved
representation. Null list children and nonfinite values fail at execution;
whole-vector nulls are retained. If a DuckPD sidecar also declares the column,
its canonical representation must agree with the catalog.

```python
store = pd.FeatureStore("/data/features")
features = store.features(
    {"pattern": "returns:shape_8"},
    start="2025-01-01T00:00:00Z",
    end="2025-02-01T00:00:00Z",
    alignment="exact",
)
matches = features[features["pattern"].notna()].vector.search_series(
    {"simple_return": query_returns},
    column="pattern",
    metric="l2",
    k=20,
)
```

`FeatureStore.series_representation(name)` and
`FeatureStore.series_embedding_model(name)` are metadata-only lookups.
Constructing the store, selecting features, calling `search_series()`, and
explaining the plan do not prepare a learned encoder or run inference. Executing
a learned search requires the exact provider to have been explicitly registered
and prepared; corpus vectors already stored in the feature partition are not
regenerated.

## Limitations and common mistakes

### The representation only knows declared features

Native vectors cannot discover nonlinear motifs, latent regimes, phase
invariance, or domain semantics not encoded in their channels and transforms.

### Normalization discards information

Centering removes local mean direction. Unit normalization removes total
magnitude. Z-scoring removes per-channel level and scale. This is useful for
shape retrieval but wrong for questions where momentum or volatility amplitude
matters.

### Channel weighting is part of the geometry

Concatenated channels share one distance calculation. Their scales, dimensions,
and preprocessing determine their influence. Add deliberate upstream scaling or
use separate representations and combine scores when channels express different
retrieval concepts.

### Overlapping windows can dominate results

Adjacent rolling windows share most observations, so a query's nearest results
may simply be nearby timestamps, including the query itself. Depending on the
analysis, exclude the source interval, impose a temporal embargo, retrieve from
other entities, or deduplicate overlapping candidates.

### Exact search has a scaling cost

Current retrieval scores the eligible candidate set exactly. This is simple and
correct, but query cost grows with corpus size and vector dimension. Persisting
representations avoids recomputing them, but it does not make exact ranking
sublinear.

### Observation windows can conceal gaps

The last 60 observations are not necessarily the last 60 minutes. Missing bars,
market closures, and irregular feeds need explicit preparation when elapsed
time is semantically important.

### Similarity is not prediction

Nearest-neighbor results are historical analogies. They do not establish
causality, forecasting accuracy, a trading signal, or predictive alpha. Any
predictive use requires chronology-aware evaluation, leakage controls,
transaction-cost assumptions, and a separately defined target.

## Practical workflow recommendations

1. Start from the retrieval question, then define channels and normalization.
2. Give the upstream data semantics a versioned `data_contract` identifier.
3. Declare deterministic row order and isolate entities with grouped windows.
4. Inspect raw windows and transformed vectors on a small known fixture.
5. Preserve scalar descriptors beside vectors for filters and interpretation.
6. Persist representations when searching the same corpus repeatedly.
7. Use a stable tie-breaker for reproducible top-k results.
8. Exclude the query and heavily overlapping windows when evaluating retrieval.
9. Compare results with an independent brute-force oracle on a bounded sample.
10. Treat any learned encoder as a candidate to benchmark against the native
    representation, not as an automatic replacement.

## Current API boundaries

The following behavior is implemented now:

- Fixed-count `Rolling.to_array()` and `GroupedRolling.to_array()` windows.
- Exact UTC fixed-grid `DataFrame.event_windows()` with explicit entity,
  event-identity, availability, duplicate, and missing-slot contracts.
- Native and learned `DataFrame.embed_series()` with `none`, `center`, or
  population `zscore` outer normalization and optional final unit normalization.
- `SeriesRepresentationSpec`, ordered learned channel roles, representation
  fingerprints, `SeriesEmbeddingProvider`, and `EmbeddedSeriesQuery`.
- Explicit session-owned learned-model registration and preparation with full
  artifact, adapter, input, pooling, runtime, and execution-provider attestation.
- Bounded complete-row Arrow inference with internal validity masks, null-row
  scatter-back, strict provider output validation, and serialized calls for
  providers declaring `thread_safe=False`.
- Typed-query compatibility checks through `vector.distance()` and
  `vector.search()`.
- Raw-window `vector.search_series()` and eager
  `Session.embed_series_query()` using the same native or learned recipe.
- Exact cosine, L2, and negative-inner-product retrieval.
- Representation metadata preservation for supported projections, joins,
  local Parquet sidecars, and session-owned tables.
- Strict catalog-version-1 series registries, typed feature/reference-table
  columns, alias and alignment propagation, and inferred native series search.
- Exact text-first, reaction-first, and full eligible-set late fusion through
  ordinary filters, typed distances, joins, and deterministic `nsmallest()`.

The following behavior is not implemented:

- Variable-length learned inputs, implicit masks/padding, or context conversion.
- General verified fixed-grid rolling windows outside event alignment.
- Automatic or general approximate series search.
- A joint multimodal model or shared text-to-series representation space.


`series_embedding_model()` creates a side-effect-free learned-model
specification. Planning and catalog lookup never import a runtime, prepare a
model, download an artifact, or run inference.

## Learned encoders

Learned encoders are an optional extension, not a replacement for native
representations. The public custom-provider workflow is:

```python
encoder = pd.series_embedding_model(
    "research/return-window-encoder",
    revision="immutable-checkpoint-revision",
    artifact_sha256="<canonical-artifact-manifest-sha256>",
    backend="custom",
    dimension=128,
    input_length=60,
    input_channels=("simple_return",),
    input_roles=("target",),
    input_normalization="checkpoint-defined-v1",
    pooling="mean-valid-v1",
    adapter_revision="return-encoder-adapter-v1",
)

representation = pd.series_representation(
    window=60,
    channels=("simple_return",),
    sampling="observations",
    data_contract="market/simple-return/split-adjusted/v1",
    normalization="none",
    unit_norm=True,
    zero_scale="error",
    encoder=encoder,
)

# Registration has no preparation or inference side effect.
session.register_series_embedding_provider(encoder, provider)
prepared = session.prepare_series_embedding_model(encoder)
learned = windows.embed_series(
    columns={"simple_return": "return_window"},
    into="return_embedding",
    representation=representation,
    batch_size=256,
)
```

A learned encoder is only one component of the representation space. Input
features, sampling, outer normalization, internal model normalization, channel
order, context length, masks, pooling, output normalization, checkpoint, and
adapter behavior all affect compatibility and must participate in identity.

The provider exposes `specification`, `thread_safe`, `prepare()`, and
`embed_windows(batch)`. `prepare()` returns immutable
`PreparedSeriesModelInfo`: the resolved revision, artifact digest, adapter and
backend revisions, input length/channels/roles, internal normalization, pooling,
dimension, cache path, execution providers, and runtime versions must all match
the declaration before the session promotes it. `embed_windows()` receives a
`pyarrow.RecordBatch` whose fields follow `input_channels` order and have type
`FixedSizeList<float32, input_length>`. Field metadata records each role.
DuckPD's validity mask removes outer-null and zero-scale rows before the call;
the provider therefore receives complete non-null rows only. Variable-length
padding masks are not part of this first contract.

### Targets, variates, and covariates

Some learned encoders distinguish roles that the native representation does
not need:

- **Target variates** are the series the forecasting model would predict.
- **Past-only covariates** are observed through the representation endpoint but
  are not known afterward.
- **Known-future covariates** are available for both the context and forecast
  horizon, such as a calendar or a published schedule.
- **Static covariates** describe the entity rather than varying by timestamp.

For native DuckPD representations, all numeric channels are simply ordered
inputs to one deterministic vector. Learned specifications pair every
`input_channels` entry with an `input_roles` entry: `target`,
`past_covariate`, or `known_future_covariate`. Inputs are numeric float32 fixed
windows. Static and categorical channels, future-horizon values, and
variable-length masks require a later versioned contract rather than an
implicit adapter convention.

This distinction also prevents leakage. A historical window representation may
only consume values that were available at its endpoint. A known-future
calendar value can be valid if it was genuinely known then; a revised economic
observation or future market value cannot be included merely because it exists
in today's dataset. Corpus and query adapters must apply the same availability
contract.

### Execution model

The implementation follows the text-embedding lifecycle while retaining
series-specific contracts:

- Planning remains side-effect-free and never downloads or imports a model.
- Preparation is explicit, eager, session-owned, and fully attested.
- Checkpoints and all numerical assets are pinned by immutable revision and a
  canonical artifact-manifest digest.
- Corpus inference uses bounded Arrow batches rather than pandas rows or whole
  corpus materialization.
- DuckPD applies declared outer normalization before the provider and final
  unit normalization afterward. Internal normalization is provider-declared.
- Channel order, roles, input length, complete-row masks, normalization,
  pooling, output dimension, row count, float32 type, nulls, and finite values
  are validated.
- Null rows are propagated without calling the provider when policy permits.
- Query and corpus encoding share preprocessing, provider validation, and the
  representation fingerprint. A raw query is encoded once per execution.
- Calls to non-thread-safe providers are serialized per model.
- No hidden interpolation, truncation, padding, device fallback, or model
  download is allowed.
- Profile metrics separate preparation, corpus/query inference, input and
  propagated-null rows, provider calls/rows/max batch size, Arrow bytes and peak
  batch bytes, query-cache state, cache bytes, and execution-provider count.

### What learned encoders may add

A qualified model may provide:

- Nonlinear representations of recurring motifs.
- Robustness to local noise or small temporal perturbations.
- Compression from long multichannel windows to a fixed lower dimension.
- Transfer from large, diverse time-series pretraining corpora.
- Task-specific invariances learned from contrastive or self-supervised data.
- Eventually, separately qualified alignment between event text and market
  reactions.

These are hypotheses to test. A model can also erase channel identity, normalize
away important amplitude, encode irrelevant pretraining biases, or perform well
for forecasting while producing poor retrieval neighborhoods.

### First-party provider policy

DuckPD ships no learned time-series model or model-specific runtime. Applications
own model selection, dependencies, artifact acquisition, preprocessing, and
provider implementation. DuckPD owns the stable execution boundary: immutable
model and representation identity, explicit registration and preparation,
bounded Arrow batches, output validation, metadata propagation, and exact
retrieval.

TSPulse and TS2Vec were evaluated as first-party candidates and removed. TSPulse
was restricted to one 512-point channel and excluded Python 3.14. TS2Vec required
a DuckPD-maintained training and runtime stack. Neither demonstrated material
held-out retrieval value over native or compact deterministic baselines. Their
model-specific complexity therefore did not justify a permanent public API.

A future built-in provider requires evidence on a named task that it materially
beats the best relevant native, PCA, or statistical baseline across held-out
entities and chronology while meeting runtime, memory, portability, licensing,
and determinism requirements. Until then, the custom provider boundary is the
only learned-series integration surface.

### Qualification requirements

Before a learned adapter is treated as supported, its qualification record
should include:

- Exact checkpoint revision and digest of every numerical artifact.
- Source, package, Python, platform, accelerator, and runtime versions.
- Source-code and weights license identifiers and deployment constraints.
- Supported input lengths, channels, dtypes, masks, and missing-value behavior.
- Outer and internal normalization, clipping, scaling, tokenization, and
  interpolation behavior.
- Pooling location and rule, output dimension, and unit-normalization policy.
- CPU and accelerator memory, throughput, batch-size, and cold-start results.
- Determinism tolerances across supported execution providers.
- Retrieval-quality results against declared native baselines.

The qualification benchmark must measure the actual product question rather
than forecasting loss. Useful suites include trend and reversal patterns,
impulses and recovery, volatility bursts and decay, level shifts, flat periods,
and event-linked reaction windows. Controlled perturbations should cover scale,
offset, noise, time shift, sign reversal, and missingness.

Report retrieval metrics such as recall, precision, or nDCG on labeled
neighbors; neighborhood stability; embedding cost; storage; and exact query
latency. Use held-out entities and chronology-aware splits where generalization
matters. Exclude overlapping windows and duplicate events that would inflate
scores.

A learned adapter should be promoted only when its improvement on a named task
justifies its complexity and operating cost. Native representations remain
supported even when a model wins a particular benchmark.

## Text and time-series representations

Text embeddings and time-series embeddings are independent vector spaces. A
text-embedding coordinate cannot be compared directly with a native return
coordinate or an application model's coordinate merely because dimensions match.

Near-term multimodal analysis should join representations by a stable event key
and combine normalized **scores**:

```text
text query --> text encoder --> text similarity --------+
                                                       +--> weighted ranking
event reaction --> series representation --> similarity +
```

The candidate set and score calibration determine whether this ranking is
exact. Combining only the small top-k from each modality can miss the best
combined result. Score fusion should therefore be treated as a deliberate
retrieval workflow, not an implicit vector operation.

Direct text-to-series search requires a jointly trained and independently
qualified aligned model. That is later research, with TRACE serving as one
reference rather than a production commitment.

## Further reading

- [Time-series embeddings and event similarity design](../design/time-series-embeddings.md)
- [Implementation roadmap](../roadmap.md#phase-16--priority-1-native-time-series-representations)
- [Event-window implementation roadmap](../roadmap.md#phase-17--priority-2-event-windows-and-exact-event-similarity)
- [Vector search and text embeddings](vector-search-and-embeddings.md)
- [API compatibility and semantic guide](../COMPATIBILITY.md#11-native-time-series-representations)
- [Event-window and exact-fusion notebook](../../demo/DuckPD_Event_Windows_and_Exact_Fusion.ipynb)
- [Time-series notebook](../../demo/DuckPD_Time_Series_Embeddings.ipynb)
- [Scripted time-series example](../../demo/time_series_embeddings.py)
