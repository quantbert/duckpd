# Time-Series Representations and Similarity Search

DuckPD supports model-free time-series similarity by turning ordered,
fixed-length numeric windows into typed vectors. These vectors can be searched
with the same exact vector engine used for other numeric data.

The current implementation is intentionally deterministic: it does not download
or run a learned model. Optional learned encoders are planned, but they will use
the same window, representation, metadata, and retrieval contracts described in
this guide.

The complete runnable walkthrough is
[`DuckPD_Time_Series_Embeddings.ipynb`](../../demo/DuckPD_Time_Series_Embeddings.ipynb).
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
- A **learned representation** will be produced by an explicitly selected
  encoder.
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
        bar_return=lambda frame: (
            (frame["close"] - frame["open"]) / frame["open"]
        ),
        intrabar_range=lambda frame: (
            (frame["high"] - frame["low"]) / frame["open"]
        ),
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
    query_row = candidates[["timestamp", "market_shape"]].head(1)
    query = pd.EmbeddedSeriesQuery(
        tuple(float(value) for value in query_row.iloc[0]["market_shape"]),
        representation.fingerprint,
    )

    matches = candidates.vector.search(
        query,
        column="market_shape",
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

Use upstream data preparation to establish a regular grid when elapsed-time
comparability matters. Fully verified fixed-grid and event-window construction
is part of the future time-series roadmap.

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
fingerprint. `EmbeddedSeriesQuery` carries that fingerprint with its values.
Searching a typed query against an incompatible column fails during planning,
even if their dimensions match.

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
- Native `DataFrame.embed_series()` with `none`, `center`, or population
  `zscore` normalization and optional final unit normalization.
- `SeriesRepresentationSpec`, representation fingerprints, and
  `EmbeddedSeriesQuery`.
- Typed-query compatibility checks through `vector.distance()` and
  `vector.search()`.
- Exact cosine, L2, and negative-inner-product retrieval.
- Representation metadata preservation for supported projections, joins,
  local Parquet sidecars, and session-owned tables.

The following behavior is not implemented yet:

- Learned series-model preparation or inference.
- A public `SeriesEmbeddingProvider` lifecycle.
- A high-level `search_series()` that accepts raw channel windows and encodes
  the query automatically.
- General verified fixed-grid rolling windows and event-aligned windows.
- Feature-store declarations for series representations.
- Automatic or general approximate series search.
- Joint text-to-time-series retrieval.

`series_embedding_model()` already creates a side-effect-free learned-model
specification so representation identity can stabilize before runtime support.
Passing a representation with an encoder to `embed_series()` currently raises
an unsupported-operation error.

## Future learned encoders

Learned encoders are planned as an optional extension, not a replacement for
native representations. The intended public workflow is:

```python
# Illustrative future API; model preparation and inference are not shipped yet.
encoder = pd.series_embedding_model(
    "research/return-window-encoder",
    revision="immutable-checkpoint-revision",
    artifact_sha256="<canonical-artifact-manifest-sha256>",
    backend="custom",
    dimension=128,
    input_length=60,
    input_channels=("simple_return",),
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

# Planned lifecycle, not currently available:
session.register_series_embedding_provider(encoder, provider)
session.prepare_series_embedding_model(encoder)
learned = windows.embed_series(
    columns={"simple_return": "return_window"},
    into="return_embedding",
    representation=representation,
)
```

A learned encoder is only one component of the representation space. Input
features, sampling, outer normalization, internal model normalization, channel
order, context length, masks, pooling, output normalization, checkpoint, and
adapter behavior all affect compatibility and must participate in identity.

### Planned execution model

The intended implementation follows the proven text-embedding lifecycle while
retaining series-specific contracts:

- Planning remains side-effect-free and never downloads a model.
- Preparation is explicit, eager, and session-owned.
- Checkpoints and all numerical assets are pinned by immutable revision and a
  canonical artifact-manifest digest.
- Corpus inference uses bounded Arrow batches rather than pandas rows or whole
  corpus materialization.
- Channel order, input length, masks, padding, normalization, pooling, output
  dimension, and finite values are validated.
- Null rows are propagated without calling the provider when policy permits.
- Query and corpus encoding use the same adapter and representation fingerprint.
- No hidden interpolation, truncation, padding, device fallback, or model
  download is allowed.

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

### Candidate model families

The following projects are useful implementation references and evaluation
candidates. They are not endorsements, bundled dependencies, or claims of
financial retrieval quality.

| Candidate | Relevant capability | Main qualification questions |
| --- | --- | --- |
| [MOMENT][moment] | Exposes an embedding path for pretrained time-series models | Does internal normalization match the task? Are channels averaged? Which context lengths, masks, and checkpoint license are supported? |
| [TS2Vec][ts2vec] | Contrastive encoder with full-series and temporal representations | Which training checkpoint is used? How are variable lengths pooled? Does a generic checkpoint transfer to the target domain? |
| [PatchTST][patchtst] | Patch-based self-supervised encoder architecture | Which hidden layer and pooling define the vector? How are channels represented? Is the exact checkpoint reproducible? |
| [Chronos-T5][chronos] | Original pipeline exposes encoder states through an embedding method | How are tokenizer scale, padding, EOS state, and sequence pooling fixed? Does forecasting pretraining help retrieval? |
| [TimesFM][timesfm] | Large pretrained forecasting model family | Is there a supported embedding contract, and are model license and deployment terms suitable? Forecasting quality alone is insufficient. |
| [TRACE][trace] | Research on aligned text and time-series representations | Can alignment generalize to held-out financial events without leakage? What shared-space metadata and evaluation are required? |

The initial built-in candidate is expected to be MOMENT only if a pinned
checkpoint and reviewed adapter pass qualification. Custom-provider support can
allow TS2Vec, PatchTST, or application-trained encoders without coupling the
DataFrame API to one model family.

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
BGE text coordinate cannot be compared directly with a native return coordinate
or a MOMENT coordinate merely because dimensions happen to match.

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
- [Vector search and text embeddings](vector-search-and-embeddings.md)
- [API compatibility and semantic guide](../COMPATIBILITY.md#11-native-time-series-representations)
- [Time-series notebook](../../demo/DuckPD_Time_Series_Embeddings.ipynb)
- [Scripted time-series example](../../demo/time_series_embeddings.py)

[moment]: https://github.com/moment-timeseries-foundation-model/moment
[ts2vec]: https://github.com/zhihanyue/ts2vec
[patchtst]: https://github.com/yuqinie98/PatchTST
[chronos]: https://github.com/amazon-science/chronos-forecasting
[timesfm]: https://github.com/google-research/timesfm
[trace]: https://github.com/Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder