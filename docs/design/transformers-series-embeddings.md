# Transformers Series Embedding Provider

**Status: implementation-ready specification.**

This document defines how DuckPD must add a first-party
`TransformersSeriesEmbeddingProvider` without weakening the existing lazy,
fingerprinted, bounded-batch series representation contract. It covers the
native Transformers time-series backbones currently relevant to DuckPD:
PatchTST, PatchTSMixer, TimesFM 1.0/2.0, TimesFM 2.5, Time Series Transformer,
Informer, and Autoformer.

The provider is an integration mechanism, not evidence that a forecasting
checkpoint produces useful retrieval embeddings. Each checkpoint still needs
independent task-specific qualification against deterministic baselines.

## Decision summary

1. Reuse `backend="transformers"`. `EmbeddingModelSpec.input` distinguishes text
   from series models; a second backend name would encode modality twice.
2. Load bare backbones with `AutoModel`, never task heads. Reject an unrecognized
   `config.model_type` and never enable `trust_remote_code`.
3. Dispatch through a private, explicit adapter allowlist. Do not infer support
   from a callable signature or the presence of `last_hidden_state`.
4. Add versioned frequency, temporal, static-feature, and query-time semantics
   to the immutable series input contract. Preserve existing serialized
   specifications and fingerprints when those fields are absent.
5. Generate timestamp-derived features in DuckPD's shared series boundary, not
   independently inside each model adapter. Corpus and query encoding must call
   the same implementation.
6. Treat TimesFM frequency and calendar time features as different concepts.
   TimesFM 1.0/2.0 receives one categorical frequency index; encoder-decoder
   models receive an ordered time-feature matrix.
7. Embed only past context. Time Series Transformer, Informer, and Autoformer use
   their encoder state; DuckPD must not execute the forecasting decoder or
   synthesize a future horizon.
8. Ship one exact, versioned pooling rule per model family before adding pooling
   choices.
9. Keep complete fixed-width windows as the first supported execution contract.
   No hidden resampling, imputation, truncation, or padding.

## Fresh-session implementation handoff

This document is the normative contract for this feature. A fresh implementation
session may be given only this instruction:

> Implement `docs/design/transformers-series-embeddings.md` completely. Treat
> every `must`, validation rule, phase exit gate, and definition-of-done item as
> required. Preserve all existing series behavior and fingerprints.

The reviewed source baseline is commit
`6b26c1105233fc3e2e8b8785ea2afa8121794ffd`. Before editing, inspect the current
versions of every file in [Source changes](#source-changes); later repository
changes win where they preserve this contract. The existing
`docs/design/time-series-embeddings.md` and
`docs/api/time-series-embeddings.md` remain authoritative for behavior this
document does not explicitly change.

This is one end-to-end change, not permission to stop after an early phase.
Completion requires all seven allowlisted profiles, corpus and rich-query
execution, catalog version 2, persistence, exports, shipped API documentation,
the changelog entry, and the permanent tests listed below. There are no
placeholder adapters, deferred public APIs, compatibility aliases, or
remote-code fallbacks.

Large public checkpoints and retrieval-quality benchmarking are qualification,
not merge gates. The implementation must provide a repeatable opt-in
qualification command. Permanent offline tests use faithful fake runtimes, and
the integration matrix must also exercise every adapter against a tiny locally
saved model under the reviewed Transformers releases. No checkpoint preset,
recommendation, or notebook is added until a separate qualification task
supplies a dataset and acceptance threshold.

### Resolved implementation decisions

- The public backend remains exactly `"transformers"`; dispatch is by input
  kind.
- The provider ABI is exactly `"transformers-series-v1"`.
- Frequency and timestamp-derived features are separate optional contracts.
  TimesFM 1.0/2.0 declares `frequency`; encoder-decoder models declare
  `temporal` exactly when configured time features are nonzero and may declare
  `static`; patch models and TimesFM 2.5 declare neither.
- A rich query uses the field and keyword `time`, not `end_at`, because
  `temporal.anchor` decides whether that instant is the final observation or the
  exclusive end.
- No new required dependency or default Hub download is added. Unit tests use
  fake runtimes or tiny locally constructed models and run offline.
- Runtime support is capability-detected, never selected by parsing a version
  string. Transformers 4.57.6 is the reviewed floor for the six older profiles;
  TimesFM 2.5 requires a release exposing `TimesFm2_5Model` through `AutoModel`
  (reviewed in Transformers 5.17.0).
- A version-1 catalog is read-only with respect to these extensions. It remains
  readable, is never rewritten implicitly, and rejects declarations containing
  version-2 nested series fields. New declarations using extensions require
  `catalog_version: 2`.
- The model fingerprint is still the SHA-256 of canonical model-spec JSON.
  Legacy input serialization is byte-for-byte unchanged; all new semantic
  fields are included before hashing.
- Model preparation uses `ignore_mismatched_sizes=False` and
  `output_loading_info=True`. `missing_keys`, `mismatched_keys`, and
  `error_msgs` must be empty. `unexpected_keys` may contain unused task-head
  weights; record their sorted names and digest in the manifest, but never load
  or execute that head.

### Definition of done

The feature is complete only when all of the following are true:

1. Existing MOMENT and temporal-free series tests pass without fingerprint,
   Arrow-schema, query, or catalog changes.
2. Every profile in [Supported model profiles](#supported-model-profiles) has an
   explicit adapter, exact prepare-time validation, tensor-shape checks, and a
   corpus/query parity test.
3. All temporal, frequency, static, and query inputs serialize canonically,
   survive sidecar/table/catalog round trips, and participate in cache identity.
4. Planning is model-runtime-free and network-free; preparation is explicit and
   artifact-verified; execution is bounded and deterministic.
5. The focused verification commands and the repository-wide `make check` and
   `make build` commands in [Verification commands](#verification-commands)
   succeed.
6. Public docs describe only behavior exercised by the implementation. The
   opt-in qualification command emits the record defined below and is not run
   by the default test suite.

## Goals

- Support the seven listed Transformers model families through one public
  provider and explicit private adapters.
- Preserve side-effect-free planning and explicit eager model preparation.
- Preserve bounded Arrow execution and fixed-size `float32` vector output.
- Make cadence, timestamp encoding, feature order, static inputs, role mapping,
  internal preprocessing, hidden-state selection, pooling, and output
  normalization part of canonical identity.
- Produce identical query and corpus embeddings for identical semantic inputs.
- Reject incompatible checkpoints during preparation, before scanning corpus
  rows.
- Keep PyTorch and Transformers optional and application-installed.

## Non-goals

- Claiming that every Hub repository tagged `transformers` is supported.
- Loading custom Hub Python with `trust_remote_code=True`.
- Treating a forecasting head's logits or predictions as an embedding.
- Guessing checkpoint time-feature semantics from `num_time_features`.
- Automatically resampling irregular observations.
- Supporting partially observed, variable-length, or padded DuckPD windows in
  the first implementation.
- Adding future decoder values, forecasting, fine-tuning, or training APIs.
- Supporting arbitrary static Python objects or row-dependent model settings.
- Recommending a checkpoint without held-out retrieval evidence.

## Existing integration baseline

The current implementation already provides the load-bearing pieces:

- `EmbeddingModelSpec` fingerprints model revision, backend, dimension,
  normalization, pooling, and optional `SeriesEmbeddingInputSpec`.
- `SeriesRepresentationSpec` fingerprints window, channels, sampling, step,
  data contract, outer normalization, final unit normalization, and encoder.
- `SeriesEmbeddingProvider` owns explicit `prepare()` and bounded
  `embed_windows(RecordBatch)` calls.
- `_make_series_provider_batch()` currently emits one non-null
  `FixedSizeList<float32, L>` field per channel.
- `Session` normalizes complete corpus/query windows before the provider,
  validates provider output, applies final unit normalization, and preserves
  representation identity through persistence and search.
- `MomentEmbeddingProvider` establishes the first-party series-provider cache,
  artifact attestation, tensor conversion, and device lifecycle pattern.

The missing data is temporal context. Corpus UDF calls currently receive only
channel arrays, while raw search queries contain only channel-to-value mappings.
A temporal model therefore cannot reconstruct the timestamp of each value or
checkpoint-compatible static features.

## Supported model profiles

Support is defined by `config.model_type` plus validation predicates, not by a
list of repository names.

| Profile | `config.model_type` | Required bare class | Hidden state | Initial pooling |
| --- | --- | --- | --- | --- |
| PatchTST | `patchtst` | `PatchTSTModel` | `(B, C, P, D)` | mean channels and patches |
| PatchTSMixer | `patchtsmixer` | `PatchTSMixerModel` | `(B, C, P, D)` | mean channels and patches |
| TimesFM 1/2 | `timesfm` | `TimesFmModel` | `(B, P, D)` | mean valid patches |
| TimesFM 2.5 | `timesfm2_5` | `TimesFm2_5Model` | `(B, P, D)` | mean valid patches |
| Time Series Transformer | `time_series_transformer` | `TimeSeriesTransformerModel` | `(B, context_length, D)` encoder state | mean encoder steps |
| Informer | `informer` | `InformerModel` | `(B, context_length, D)` encoder state | mean encoder steps |
| Autoformer | `autoformer` | `AutoformerModel` | `(B, context_length, D)` encoder state | mean encoder steps |

TimesFM 2.5 is feature-gated until the installed Transformers release exposes
`TimesFm2_5Model` and its `AutoModel` mapping. Absence is an actionable
preparation error; it must not trigger remote-code loading.

Task-head checkpoints may be accepted only when `AutoModel.from_pretrained()`
loads the allowlisted bare backbone and all backbone weights load cleanly.
DuckPD never invokes `ForPrediction`, `ForPretraining`, `ForRegression`, or
`ForClassification` heads for embeddings.

## Public immutable contracts

### Provider ABI

Extend `SeriesEmbeddingInputSpec` with optional fields. Existing fields and
existing serialization remain unchanged:

```python
@dataclass(frozen=True)
class SeriesEmbeddingInputSpec:
    length: int
    channels: tuple[str, ...]
    roles: tuple[SeriesChannelRole, ...]
    normalization: str
    provider_abi: str | None = None
    frequency: SeriesFrequencyInputSpec | None = None
    temporal: SeriesTemporalInputSpec | None = None
    static: tuple[SeriesStaticInputSpec, ...] = ()
```

New Transformers series specifications require:

```text
provider_abi = "transformers-series-v1"
```

`provider_abi` prevents a future tensor-layout or adapter change from silently
reusing an old embedding fingerprint. `to_dict()` must omit `provider_abi`,
`frequency`, `temporal`, and `static` when they have their legacy defaults so
existing model fingerprints do not change.

A new specification containing any extension field serializes its nested input
with `schema_version: 2`. The parser continues to accept the existing unversioned
shape as version 1. Catalog declarations using version-2 series input fields
require `catalog_version: 2`; catalog version 1 remains strict and unchanged.

Extension fields are valid only when the containing model uses
`backend="transformers"` and `provider_abi="transformers-series-v1"`.
`frequency` is mutually exclusive with both `temporal` and `static`. Static
features may be declared without temporal features for an encoder configured
with `num_time_features == 0`. MOMENT and custom providers retain the legacy
four-field contract; richer inputs in another provider require a future ABI.

### Canonical serialization and public constructors

The public constructors have these exact new keyword-only parameters:

```python
def series_embedding_input(
    *,
    length: int,
    channels: tuple[str, ...],
    roles: tuple[SeriesChannelRole, ...],
    normalization: str,
    provider_abi: str | None = None,
    frequency: SeriesFrequencyInputSpec | None = None,
    temporal: SeriesTemporalInputSpec | None = None,
    static: tuple[SeriesStaticInputSpec, ...] = (),
) -> SeriesEmbeddingInputSpec: ...

def series_cadence(
    unit: SeriesCadenceUnit,
    *,
    multiple: int = 1,
    mode: Literal["elapsed", "civil"] = "elapsed",
) -> SeriesCadenceSpec: ...

def series_frequency_input(
    *,
    cadence: SeriesCadenceSpec,
    timesfm_frequency: Literal["auto", 0, 1, 2] = "auto",
) -> SeriesFrequencyInputSpec: ...

def series_temporal_input(
    *,
    cadence: SeriesCadenceSpec,
    timezone: str,
    anchor: Literal["last", "end_exclusive"],
    recipe: Literal["gluonts-calendar-v1", "calendar-fourier-v1"],
    features: tuple[SeriesTimeFeature, ...],
) -> SeriesTemporalInputSpec: ...

def series_static_input(
    name: str,
    *,
    kind: Literal["real", "categorical"],
    cardinality: int | None = None,
    normalization: Literal["none"] = "none",
) -> SeriesStaticInputSpec: ...
```

Every exported dataclass validates the same invariants in `__post_init__`; the
helpers are conveniences, not validation bypasses.

Define `SeriesCadenceUnit`, `SeriesTimeFeature`, `SeriesCadenceSpec`,
`SeriesFrequencyInputSpec`, `SeriesTemporalInputSpec`,
`SeriesStaticInputSpec`, their five helpers, and the extended
`SeriesEmbeddingInputSpec` in `embeddings.py`. Define `SeriesQueryInput` and
`series_query()` in `series_embeddings.py`. Define
`TransformersSeriesEmbeddingProvider` in `series_providers.py`. Re-export every
non-private name in this paragraph from `duckpd.__init__`; adapters and temporal
generation functions remain private.

The existing four-field input serializes exactly as before, with no
`schema_version`. A version-2 input serializes to the following canonical shape;
optional `frequency`, `temporal`, and `static` keys are omitted when absent, but
`provider_abi` and `schema_version` are required:

```json
{
  "kind": "series",
  "schema_version": 2,
  "length": 61,
  "channels": ["target", "promotion"],
  "roles": ["target", "known_future_covariate"],
  "normalization": "hf-time-series-scaler-v1",
  "provider_abi": "transformers-series-v1",
  "temporal": {
    "cadence": {"unit": "month", "multiple": 1, "mode": "civil"},
    "timezone": "UTC",
    "anchor": "last",
    "recipe": "gluonts-calendar-v1",
    "features": ["month_of_year", "age_log10"]
  },
  "static": [
    {
      "name": "series_id",
      "kind": "categorical",
      "cardinality": 366,
      "normalization": "none"
    }
  ]
}
```

Canonical JSON uses UTF-8, sorted object keys, compact separators, and preserves
array order. Parsers reject booleans where integers are required, unknown keys,
missing required keys, duplicate names/features, and a `schema_version` other
than `2`. They never coerce strings to numbers. `EmbeddingModelSpec.to_dict()`
embeds this object unchanged; sidecars and table metadata therefore need no
second representation of these fields.

`frequency` serializes as
`{"cadence":{"mode":"elapsed","multiple":1,"unit":"hour"},"timesfm_frequency":"auto"}`
under canonical key sorting. Catalog version 2 has the same top-level fields and
the same `series_representations[].version == 1` contract as version 1; its only
schema extension is accepting nested series input schema version 2 in
`embedding_models`. Version 2 also accepts legacy text and series model entries.
Parsing preserves the declared catalog version, and no read, write, or lookup
implicitly migrates a version-1 catalog.

The existing series sidecar envelope version, table-metadata envelope, and
`SeriesRepresentationSpec` serialization version remain unchanged because the
embedded model input is self-versioned. New readers accept both nested input
versions. An older reader is expected to reject the unknown version-2 input
rather than partially interpret it.

### Cadence

Timestamp generation needs a semantic cadence, not a duration guessed from row
differences:

```python
SeriesCadenceUnit = Literal[
    "second", "minute", "hour", "day",
    "week", "month", "quarter", "year",
]

@dataclass(frozen=True)
class SeriesCadenceSpec:
    unit: SeriesCadenceUnit
    multiple: int = 1
    mode: Literal["elapsed", "civil"] = "elapsed"
```

Validation rules:

- `multiple` is a positive integer.
- `second`, `minute`, and `hour` require `mode="elapsed"`.
- `week`, `month`, `quarter`, and `year` require `mode="civil"`.
- `day` may be elapsed 24-hour time or a civil day; the distinction is
  fingerprinted because daylight-saving transitions differ.
- Exchange-session calendars are not silently represented as civil time. They
  remain unsupported until a separate calendar contract is implemented.

For a fixed-grid representation, only elapsed cadence is valid and it must equal
the existing canonical `SeriesRepresentationSpec.step`; the current UTC grid
does not claim civil-calendar semantics. An observation-sampled representation
may declare cadence only as an application assertion. DuckPD never infers
regularity from a window's observed deltas.

For window index `j` in `[0, L)`, reconstruct timestamps from the same supplied
anchor, never by repeatedly stepping from the prior generated timestamp:

```text
anchor="last":          t[j] = shift(anchor, j - (L - 1))
anchor="end_exclusive": t[j] = shift(anchor, j - L)
```

`shift(x, n)` applies `n * cadence.multiple` units. Elapsed units use exact
seconds (`minute=60`, `hour=3600`, `day=86400`) on the instant timeline. Civil
day and week shifts preserve local wall-clock fields while changing the local
date. Civil month, quarter, and year shifts compute the target year/month
directly from the original `x`; the day is `min(original_day,
last_day_of_target_month)`. Computing every point from the anchor preserves
month-end sequences and prevents iterative clipping drift. Any generated
ambiguous/nonexistent local time, overflow, non-increasing instant sequence, or
misalignment fails.

For age, the series start must be no later than the first generated timestamp.
Compute the nonnegative integer `i` directly for the cadence unit, then require
`shift(series_start, i) == timestamp`; otherwise the timestamps are not aligned
and fail.

### TimesFM frequency

TimesFM 1.0/2.0 frequency is categorical, not a timestamp-derived feature:

```python
@dataclass(frozen=True)
class SeriesFrequencyInputSpec:
    cadence: SeriesCadenceSpec
    timesfm_frequency: Literal["auto", 0, 1, 2] = "auto"
```

When `timesfm_frequency="auto"`, resolve from semantic cadence:

| Cadence unit | Index |
| --- | ---: |
| second, minute, hour, day | 0 |
| week, month | 1 |
| quarter, year | 2 |

An application may explicitly select `0`, `1`, or `2`, as permitted by TimesFM.
The explicit value or the `"auto"` rule, cadence, and representation sampling
all participate in the model and complete representation fingerprints. Do not
classify from observed gaps. A frequency declaration requires no timestamp
column and emits no per-step feature matrix. Every profile other than TimesFM
1.0/2.0 rejects `frequency`.

### Temporal preprocessing

```python
SeriesTimeFeature = Literal[
    "second_of_minute",
    "minute_of_hour",
    "hour_of_day",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "month_of_year",
    "age_log10",
]

@dataclass(frozen=True)
class SeriesTemporalInputSpec:
    cadence: SeriesCadenceSpec
    timezone: str
    anchor: Literal["last", "end_exclusive"]
    recipe: Literal[
        "gluonts-calendar-v1",
        "calendar-fourier-v1",
    ]
    features: tuple[SeriesTimeFeature, ...]
```

`features` is nonempty and contains no duplicates. `timezone` must be a key
accepted by `zoneinfo.ZoneInfo` and is stored exactly as supplied; aliases are
not normalized and therefore have distinct fingerprints. A timestamp represents
an instant; DuckPD converts it to the declared zone before extracting civil
fields. Naive timestamps are interpreted in that zone. Ambiguous or nonexistent
local times fail rather than selecting a fold silently.

`anchor="last"` means the supplied timestamp belongs to the final value in the
window. `anchor="end_exclusive"` means the supplied timestamp is one cadence
after the final value, matching fixed-grid event-window end boundaries.

Recipe semantics are exact:

- `gluonts-calendar-v1` reproduces the scalar calendar and age feature formulas
  used by the reviewed Hugging Face/GluonTS preprocessing recipe. Feature order
  is exactly `features` order. The implementation is DuckPD-owned and tested
  against a pinned GluonTS reference; GluonTS is not a runtime dependency.
- `calendar-fourier-v1` emits sine then cosine for each periodic calendar
  feature in `features` order. `age_log10` is scalar and therefore emits one
  value. This recipe is valid only for checkpoints trained with the same recipe;
  it is not a substitute for a GluonTS-trained checkpoint's preprocessing.

For a periodic component `x` with zero-based period `P`, Fourier output is:

```text
sin(2*pi*x/P), cos(2*pi*x/P)
```

`gluonts-calendar-v1` uses zero-based `x` and
`x / (P - 1) - 0.5`, matching the reviewed GluonTS implementation:

| Feature | `P` |
| --- | ---: |
| `second_of_minute` | 60 |
| `minute_of_hour` | 60 |
| `hour_of_day` | 24 |
| `day_of_week` | 7 |
| `day_of_month` | 31 |
| `day_of_year` | 366 |
| `week_of_year` | 53 |
| `month_of_year` | 12 |

The zero-based components are exact: `second`, `minute`, and `hour` use their
integer civil fields; `day_of_week` uses Monday `0` through Sunday `6`;
`day_of_month` is `day - 1`; `day_of_year` is `tm_yday - 1`;
`week_of_year` is ISO week minus one; and `month_of_year` is `month - 1`.
Subsecond timestamp components do not affect these features.

For `gluonts-calendar-v1`, each declared feature contributes width one. For
`calendar-fourier-v1`, each periodic feature contributes width two and
`age_log10` contributes width one. Preparation validates the resulting width
against `config.num_time_features`.

`age_log10` requires a series-start timestamp for both corpus and raw query
encoding. For zero-based absolute series index `i`, both recipes emit
`log10(2 + i)`, matching `AddAgeFeature(log_scale=True)`. A missing start
timestamp is an error; DuckPD must not redefine age relative to each retrieval
window. DuckPD computes `i` as the exact number of declared cadence applications
from `series_start` to each generated timestamp. A start/anchor pair that is not
aligned to the cadence fails instead of rounding an elapsed-duration ratio.

### Static features

Encoder-decoder checkpoints may require static real or categorical features:

```python
@dataclass(frozen=True)
class SeriesStaticInputSpec:
    name: str
    kind: Literal["real", "categorical"]
    cardinality: int | None = None
    normalization: Literal["none"] = "none"
```

Names are unique and ordered. This ABI supports only `normalization="none"`.
Categorical features require a positive `cardinality`; real features prohibit
it. Corpus categorical values are integer IDs in `[0, cardinality)`. Query and
corpus use the same IDs. DuckPD does not create category vocabularies during
embedding.

Preparation validates, in order:

```text
number of real specs        == config.num_static_real_features
number of categorical specs == config.num_static_categorical_features
categorical cardinalities   == config.cardinality
```

### Channel roles

Adapters interpret the existing ordered channel roles as follows:

| Family | `target` | `past_covariate` | `known_future_covariate` |
| --- | --- | --- | --- |
| PatchTST/PatchTSMixer | value channel | value channel | value channel |
| TimesFM | exactly one required | rejected | rejected |
| Encoder-decoder families | `past_values` channel | dynamic real time-feature channel | dynamic real time-feature channel |

For encoder-only embeddings, both covariate roles contribute only their values
inside the historical window. The role distinction remains important to data
availability and leakage checks. Encoder-decoder preparation validates:

```text
number of target channels    == config.input_size
number of non-target channels == config.num_dynamic_real_features
```

Generated calendar features precede declared dynamic real channels in
`past_time_features`. This ordering is part of `provider_abi`.

## Intended model specifications

Patch backbones declare no temporal or static inputs:

```python
patchtst = pd.embedding_model(
    "ibm-granite/granite-timeseries-patchtst",
    revision="7fe295d8bc8fbac8041b60ab351882634165517f",
    backend="transformers",
    dimension=128,
    normalize=True,
    pooling="mean-channels-patches-v1",
    input=pd.series_embedding_input(
        length=512,
        channels=("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"),
        roles=("target",) * 7,
        normalization="patchtst-config-scaling-v1",
        provider_abi="transformers-series-v1",
    ),
)
```

TimesFM 1.0/2.0 declares categorical frequency semantics:

```python
timesfm = pd.embedding_model(
    "google/timesfm-2.0-500m-pytorch",
    revision="dc2443792ce5516872b89b37cf1bc058c3bf0c10",
    backend="transformers",
    dimension=1280,
    normalize=True,
    pooling="mean-valid-patches-v1",
    input=pd.series_embedding_input(
        length=2048,
        channels=("target",),
        roles=("target",),
        normalization="timesfm-masked-mean-std-v1",
        provider_abi="transformers-series-v1",
        frequency=pd.series_frequency_input(
            cadence=pd.series_cadence("minute", multiple=5),
            timesfm_frequency="auto",
        ),
    ),
)
```

TimesFM 2.5 uses the same univariate value contract but omits both `frequency`
and `temporal`; its architecture no longer consumes the frequency indicator.

An encoder-decoder checkpoint declares the complete lag-history window,
calendar recipe, dynamic covariates, and static features:

```python
transformer = pd.embedding_model(
    "huggingface/time-series-transformer-tourism-monthly",
    revision="2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35",
    backend="transformers",
    dimension=26,
    normalize=True,
    pooling="mean-encoder-time-v1",
    input=pd.series_embedding_input(
        # config.context_length + max(config.lags_sequence)
        length=61,
        channels=("tourists",),
        roles=("target",),
        normalization="hf-time-series-scaler-v1",
        provider_abi="transformers-series-v1",
        temporal=pd.series_temporal_input(
            cadence=pd.series_cadence("month", mode="civil"),
            timezone="UTC",
            anchor="last",
            recipe="gluonts-calendar-v1",
            features=("month_of_year", "age_log10"),
        ),
        static=(
            pd.series_static_input("static_real_0", kind="real"),
            pd.series_static_input(
                "series_id",
                kind="categorical",
                cardinality=366,
            ),
        ),
    ),
)
```

These examples are validated configuration-shape illustrations, not qualified
retrieval presets. Their application semantics, artifact digest, licenses, and
retrieval quality still require the separate qualification record below.

## Corpus and query temporal coordinates

### Corpus API

Extend learned `embed_series()` calls with explicitly mapped context columns:

The public helper keeps every existing parameter and adds:

```python
def embed_series(
    frame: DataFrame,
    *,
    columns: Mapping[str, str],
    into: str,
    representation: SeriesRepresentationSpec,
    batch_size: int = 256,
    null_policy: SeriesNullPolicy = "propagate",
    time: str | None = None,
    series_start: str | None = None,
    static_columns: Mapping[str, str] | None = None,
) -> DataFrame: ...
```

`time` and `series_start` corpus columns must have DuckDB `TIMESTAMP` or
`TIMESTAMPTZ` logical type; dates, strings, and numeric epochs are rejected.
Static real columns must be numeric, and static categorical columns must be
integral. Null temporal/static scalars, booleans, nonfinite real values, and
out-of-range categorical IDs follow the batch-construction error boundary below;
they never become model inputs.

```python
embedded = windows.embed_series(
    columns={"target": "target_window"},
    into="embedding",
    representation=representation,
    time="window_end",
    series_start="series_start",
    static_columns={"series_id": "series_id"},
)
```

- `time` identifies a scalar timestamp column interpreted according to
  `temporal.anchor`; it is required exactly when `temporal` is declared and
  rejected otherwise.
- `series_start` is required exactly when `age_log10` is requested and rejected
  otherwise.
- `static_columns` keys must exactly match the declared static feature names.
- A `frequency` declaration needs no corpus column; its categorical value is
  constant model metadata derived from the immutable specification.
- Omitting a required mapping or supplying an undeclared mapping is an eager
  error.

Add the corresponding optional column IDs to `SeriesRepresentationPlan` so they
survive planning, projection, compiler binding, explain output, and source
validation. The physical UDF argument order is: the existing homogeneous packed
channel arrays; `time` when temporal input is declared; `series_start` when age
is declared; then one scalar static column per declaration in exact declaration
order. Register the UDF with the exact argument list and DuckDB timestamp/numeric
types for that plan; do not pass placeholder null arguments.
`_make_series_provider_batch()` groups the static scalars into the two reserved
typed arrays below. Never coerce timestamps or mixed static values through the
channel `list_value(DOUBLE[L], ...)` payload.

### Query API

A mapping alone remains valid for representations without temporal or static
inputs. Add this public immutable query value and constructor:

```python
@dataclass(frozen=True)
class SeriesQueryInput:
    values: tuple[tuple[str, tuple[float, ...]], ...]
    time: datetime | None = None
    series_start: datetime | None = None
    static: tuple[tuple[str, float | int], ...] = ()

def series_query(
    values: Mapping[str, Sequence[float]],
    *,
    time: datetime | None = None,
    series_start: datetime | None = None,
    static: Mapping[str, float | int] | None = None,
) -> SeriesQueryInput: ...
```

```python
query = pd.series_query(
    values={"target": values},
    time=datetime(2026, 8, 1, tzinfo=timezone.utc),
    series_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
    static={"series_id": 17},
)

result = embedded.vector.search_series(query, column="embedding")
```

`series_query()` validates and snapshots mappings into insertion-ordered tuples,
converts each value sequence to a tuple of finite floats, rejects booleans, and
requires `datetime` objects for temporal coordinates. Representation-specific
validation still happens in the session because channel/static order and
timestamp interpretation belong to the representation. The session reorders
names into specification order and uses resolved UTC nanoseconds in query-cache
identity.

Extend `Session.embed_series_query()` and `VectorNamespace.search_series()` to
accept `Mapping[str, Sequence[float]] | SeriesQueryInput`. A plain mapping fails
with an actionable message when the representation requires temporal or static
inputs.

Replace the tuple alias `SeriesQuerySnapshot` with an immutable internal
structure containing ordered channels, optional anchor timestamp, optional
series-start timestamp, and ordered static values. Corpus and query rows then
pass through the same normalization, temporal generation, Arrow construction,
adapter, pooling, and finalization functions.

## Canonical Arrow provider batch

Keep semantic channels as their existing fields. Add reserved fields only when
required:

```text
<channel name>                         FixedSizeList<float32, L>
__duckpd_past_time_features            FixedSizeList<float32, L*F>
__duckpd_static_real                   FixedSizeList<float32, R>
__duckpd_static_categorical            FixedSizeList<int64, K>
```

Fields are ordered as declared channels, optional past-time features, optional
static real values, then optional static categorical values. The
`__duckpd_past_time_features` payload contains generated temporal features only,
flattened time-major as
`[t0_feature0, ..., t0_featureF-1, t1_feature0, ...]`; adapters append ordered
non-target channel values after those features. Static arrays preserve the
relative declaration order within each kind. An absent or zero-width group is
omitted rather than encoded as a zero-length list.

Schema metadata records UTF-8 byte values under these exact keys:

```text
duckpd.model_fingerprint
duckpd.representation_fingerprint
duckpd.provider_abi
duckpd.mask_semantics = complete_rows_only-v1
duckpd.time_feature_shape = L,F          # temporal input only
duckpd.timesfm_frequency = 0|1|2         # TimesFM 1/2 only
```

Reserved fields are generated by DuckPD after outer normalization and before
provider invocation. They are not user-visible DataFrame columns. Providers
reject unexpected reserved fields, missing fields, field reordering, wrong
metadata, and representation/model fingerprint mismatches.

The existing MOMENT batch remains byte-for-byte unchanged because legacy input
specifications declare no extension fields.

## Provider architecture

`TransformersSeriesEmbeddingProvider` lives in `series_providers.py` and
implements the existing `SeriesEmbeddingProvider` protocol. It owns:

- optional dependency imports;
- pinned snapshot acquisition and atomic verified-cache promotion;
- `AutoConfig` and `AutoModel` loading with `trust_remote_code=False`;
- adapter selection and configuration validation;
- device placement and inference mode;
- Arrow-to-tensor conversion;
- output shape/type validation before Arrow conversion;
- retained model lifecycle and `close()`.

Its public constructor matches the existing series-provider lifecycle:

```python
def __init__(
    self,
    specification: EmbeddingModelSpec,
    *,
    device: Literal["cpu", "cuda"] = "cpu",
    cache_dir: str | Path | None = None,
    prepare_timeout_seconds: float | None = None,
    max_download_bytes: int | None = None,
) -> None: ...
```

On ROCm, PyTorch still uses the `"cuda"` device spelling. The constructor
performs only local specification/settings validation and path construction; it
does not import model dependencies, inspect devices, read the Hub, or create
cache directories.


Use a private adapter protocol:

```python
class _TransformersSeriesAdapter(Protocol):
    def validate(self, config, specification) -> None: ...
    def embed(self, model, batch, torch, device) -> Tensor: ...
```

The allowlist is a literal mapping:

```python
_ADAPTERS = {
    "patchtst": _PatchBackboneAdapter(...),
    "patchtsmixer": _PatchBackboneAdapter(...),
    "timesfm": _TimesFmAdapter(...),
    "timesfm2_5": _TimesFm25Adapter(...),
    "time_series_transformer": _EncoderDecoderAdapter(...),
    "informer": _EncoderDecoderAdapter(...),
    "autoformer": _AutoformerEncoderAdapter(...),
}
```

Do not expose adapters publicly. Similar signatures are not a stable semantic
contract.

The text and series Transformers providers may share small private artifact
verification helpers. Do not introduce a public provider base class or combine
text tokenization with series tensor conversion.

## Preparation and artifact policy

Preparation follows the current first-party cache lifecycle:

1. Validate provider settings without importing optional dependencies.
2. Import PyTorch, Transformers, and `huggingface_hub` only at explicit
   preparation.
3. Verify requested CPU/CUDA/ROCm availability; never fall back devices.
4. Download only configuration and numerical weight files for the immutable
   revision into a sibling staging directory.
5. Load `AutoConfig`, select the allowlisted adapter, and run all static config
   checks.
6. Load `AutoModel` with `trust_remote_code=False`, `local_files_only=True`,
   `ignore_mismatched_sizes=False`, and `output_loading_info=True`; require the
   exact allowlisted bare class for the selected `model_type`.
7. Require `missing_keys == []`, `mismatched_keys == []`, and
   `error_msgs == []`. Unused task-head weights may appear only in
   `unexpected_keys`; sort and record those names and their SHA-256 digest in the
   manifest. Never suppress a mismatch or retry with a task-head class.
8. Hash the promoted artifact directory and write the DuckPD manifest.
9. Record model type, resolved class, adapter ABI, pooling contract,
   configuration digest, Transformers version, execution provider, model
   specification, and loading-information digest in the manifest.
10. Move the model to the selected device, call `eval()`, and retain it in the
    session-owned provider.

Preparation validates model-specific dimensions before corpus execution:

```text
Patch families:       specification.dimension == config.d_model
TimesFM families:     specification.dimension == config.hidden_size
Encoder-decoder:      specification.dimension == config.d_model
```

The provider must not mutate checkpoint configuration to force compatibility.
In particular, it does not change input channel count, context length, lag list,
scaler, masking policy, or attention mode.

## Adapter contracts

`SeriesEmbeddingInputSpec.normalization` identifies model-internal
preprocessing, not the outer `SeriesRepresentationSpec.normalization`. Adapters
require the exact family string below and let the pinned bare model execute its
own scaler; DuckPD neither duplicates nor disables it:

| Profile | Required input normalization |
| --- | --- |
| PatchTST | `patchtst-config-scaling-v1` |
| PatchTSMixer | `patchtsmixer-config-scaling-v1` |
| TimesFM 1.0/2.0 | `timesfm-masked-mean-std-v1` |
| TimesFM 2.5 | `timesfm2_5-config-normalization-v1` |
| Time Series Transformer, Informer, Autoformer | `hf-time-series-scaler-v1` |


### PatchTST and PatchTSMixer

Validation:

```text
input.length == config.context_length
len(input.channels) == config.num_input_channels
input.static is empty
input.frequency is absent
input.temporal is absent
pooling == "mean-channels-patches-v1"
```

PatchTST additionally requires `config.do_mask_input is not True` and legacy
`config.mask_input is not True`. A masked-pretraining checkpoint must expose an
unmasked bare-backbone configuration; the provider does not mutate masking
fields or permit stochastic input masking during embedding.

Execution:

1. Stack ordered Arrow channels as contiguous float32 `(B, C, L)`.
2. Transpose once to `(B, L, C)` without a second materialized copy where the
   selected PyTorch operation accepts the view.
3. Call the bare model with `past_values`; omit the optional observed mask so
   the model uses all-observed semantics.
4. Require `last_hidden_state` rank four and shape `(B, C, P, D)`.
5. Mean over channel and patch dimensions in float32, yielding `(B, D)`.

PatchTST documentation has historically disagreed about the final hidden axis;
runtime validation must require `D == config.d_model` rather than trusting the
docstring label.

### TimesFM 1.0/2.0

Validation:

```text
exactly one target channel
no covariate or static inputs
input.frequency is present
input.temporal is absent
input.length <= config.context_length
input.length % config.patch_length == 0
pooling == "mean-valid-patches-v1"
```

Execution:

1. Materialize `(B, L)` float32 values.
2. Create an all-zero `(B, L)` integer padding mask; zero means valid.
3. Read the resolved frequency index from canonical batch metadata and create
   `(B,)` integers.
4. Call bare `TimesFmModel(past_values, past_values_padding, freq)`.
5. Require `(B, P, D)` hidden state and mean only valid patch tokens.

The first implementation does not left-pad arbitrary lengths. Requiring a patch
multiple keeps padding semantics explicit and preserves the current complete
window contract.

### TimesFM 2.5

Validation matches TimesFM 1/2 except:

- both `frequency` and `temporal` must be absent;
- no frequency metadata is accepted;
- the installed Transformers runtime must expose `TimesFm2_5Model` through
  `AutoModel`.

Call the bare model with `(B, L)` values and an all-zero padding mask. Require
`(B, P, D)` hidden state and apply `mean-valid-patches-v1`. Do not execute the
forecasting head, flip-invariance forecasting path, quantile head, or positivity
clamping.

### Time Series Transformer and Informer

Validation:

```text
input.frequency is absent
input.temporal is present iff config.num_time_features > 0
input.length == config.context_length + max(config.lags_sequence)
target channel count == config.input_size
covariate channel count == config.num_dynamic_real_features
generated feature width (zero when temporal is absent) == config.num_time_features
static fields match config counts/cardinalities
pooling == "mean-encoder-time-v1"
```

Treat absent `num_dynamic_real_features`, `num_static_real_features`, and
`num_static_categorical_features` configuration attributes as zero. When static
categorical features are declared, `config.cardinality` must be a sequence of
positive integers in the same order and with the same length; no broadcasting
or singleton expansion is allowed.

Execution uses the bare model's own preprocessing to preserve its scaler and lag
construction:

1. Stack target roles into `past_values` with shape `(B, L)` for univariate or
   `(B, L, input_size)` for multivariate input.
2. Concatenate generated time features and ordered non-target channel values as
   `past_time_features` with shape `(B, L, F)`.
3. Supply an all-one `past_observed_mask` matching `past_values`.
4. Supply ordered static categorical and real tensors when declared.
5. Call `model.create_network_inputs(...)` with no future values or future time
   features.
6. Slice the first `config.context_length` network steps, matching the upstream
   model implementation.
7. Call `model.get_encoder()(inputs_embeds=..., return_dict=True)` directly.
8. Require `(B, context_length, D)` and mean over encoder time.

Calling the full model and reading `last_hidden_state` is incorrect because that
field belongs to the decoder. DuckPD explicitly pools the encoder output.

### Autoformer

Validation is the same as the other encoder-decoder adapters. Its
`create_network_inputs()` returns lagged values and temporal/static features
separately. Reproduce the model's encoder path exactly:

1. Call `create_network_inputs(...)` without future inputs.
2. Concatenate its lagged input and temporal feature outputs for the first
   `context_length` steps.
3. Call `model.get_encoder()` directly.
4. Pool the encoder's `(B, context_length, D)` state with
   `mean-encoder-time-v1`.

Do not run the decomposition/decoder forecast path and do not pool Autoformer's
trend output.

## Session dispatch

Permit series inputs with `backend="transformers"` in
`EmbeddingModelSpec.__post_init__()`. Dispatch by input kind:

```text
backend="transformers", input is None   -> TransformersEmbeddingProvider
backend="transformers", input is series -> TransformersSeriesEmbeddingProvider
```

Catalog pooling validation must also branch by input kind. Text retains `cls` or
`mean`; series accepts only the three versioned family-specific pooling strings.
A custom registered provider remains valid only for `backend="custom"`.

Automatic preparation follows the same catalog/session policy as existing
built-in providers. Planning never imports Transformers, reads Hub metadata, or
selects an adapter.

## Error boundary

Fail during specification construction when the error is local:

- malformed cadence, timezone, recipe, feature order, static specification, or
  provider ABI;
- duplicate names or unsupported combinations;
- temporal/static mappings supplied to a representation that does not declare
  them.

Fail during `prepare()` before a source scan when checkpoint metadata is needed:

- unsupported `model_type` or unavailable Transformers architecture;
- remote-code requirement;
- context, patch, channel, role, feature-width, static-feature, lag, pooling, or
  output-dimension mismatch;
- missing or incompatible backbone weights.

Fail during batch construction before provider inference for row data problems:

- null/invalid time anchor or series start;
- ambiguous/nonexistent local timestamp;
- invalid static categorical ID or nonfinite static real value;
- timestamp/cadence overflow.

Fail after model execution for output problems:

- missing hidden state;
- wrong rank, row count, token count, or hidden dimension;
- null, NaN, or infinite output;
- zero norm under the representation's declared policy.

Every error names the model, adapter, conflicting field, expected value, and
actual value without including credentials or unbounded row data.

## Determinism and numerical behavior

- Always call `eval()` and use `torch.inference_mode()`.
- Do not enable stochastic checkpoint masking for embedding inference.
- Pool in float32 even when model execution uses an explicitly supported lower
  precision later.
- Initial support executes model inputs as float32. Mixed precision needs a
  separately fingerprinted provider ABI and qualification record.
- Output is contiguous CPU float32 before Arrow construction.
- Batch size, batch boundaries, and row order must not materially change results.
- Corpus and query temporal generation use the same pure functions.
- Device-specific tolerance is a qualification property, not permission to
  change fingerprints or silently repair results.

## Security and dependency policy

- Core DuckPD gains no required model dependency.
- Applications install a suitable PyTorch build, Transformers, and
  `huggingface_hub`; DuckPD does not choose a CPU/CUDA/ROCm wheel.
- Never execute Hub Python or enable `trust_remote_code`.
- Require immutable revisions and verified cache manifests.
- Download allowlists include configuration and numerical weights only.
- Cache paths and manifests contain no credentials or access tokens.
- Model source and weight licenses are recorded during qualification; they are
  not inferred to be compatible merely because loading succeeds.

## Source changes

| File | Required change |
| --- | --- |
| `src/duckpd/embeddings.py` | Define cadence/frequency/temporal/static public contracts and helpers; add version-2 input fields; allow Transformers series inputs; preserve legacy serialization/fingerprints. |
| `src/duckpd/_temporal.py` | Add pure timestamp reconstruction, timezone conversion, TimesFM frequency resolution, and versioned calendar feature generation. |
| `src/duckpd/series_embeddings.py` | Add rich query input/snapshot, shared feature generation, and extended Arrow batch construction. |
| `src/duckpd/series_providers.py` | Add `TransformersSeriesEmbeddingProvider` and private allowlisted adapters. |
| `src/duckpd/_logical.py` | Carry optional time, series-start, and static column IDs on `SeriesRepresentationPlan`. |
| `src/duckpd/_compiler.py` | Bind mixed typed UDF arguments instead of packing only homogeneous channel arrays. |
| `src/duckpd/session.py` | Dispatch the provider, extend corpus/query UDF boundaries and caches, verify preparation, and close retained providers. |
| `src/duckpd/vector.py` | Accept rich `SeriesQueryInput` while preserving mapping queries for temporal-free representations. |
| `src/duckpd/_feature_catalog.py` | Parse version-2 nested specs and validate pooling/backend by input kind. |
| `src/duckpd/__init__.py` | Export the new public specification helpers, query type, and provider. |
| `tests/test_series_embeddings.py` | Cover canonical identity, temporal/query parity, plan metadata, persistence, and failures. |
| `tests/test_series_providers.py` | Cover adapter validation, tensor layout, pooling, optional dependencies, caching, and deterministic inference. |
| `tests/test_featurestore.py` | Cover catalog-version-2 round trips and strict version-1 rejection. |
| `scripts/qualify_transformers_series.py` | Add an opt-in offline-input/public-checkpoint qualification runner; never invoke it from the default test suite. |
| `scripts/generate_compatibility.py` and `docs/COMPATIBILITY.md` | Regenerate the public signature/behavior matrix. |
| `docs/api/time-series-embeddings.md` | Document the shipped API after implementation. |
| `docs/design/time-series-embeddings.md` | Cross-reference the implemented provider and catalog-version-2 extension without rewriting its historical baseline. |
| `docs/CHANGELOG.md` | Record the feature when behavior ships. |

## Implementation sequence

### Phase 1: immutable semantics and temporal generation

- Add cadence, frequency, temporal, static, and rich query dataclasses.
- Implement canonical serialization with legacy fingerprint preservation.
- Add categorical frequency resolution, timestamp-anchor reconstruction, and
  the two versioned calendar recipes.
- Extend representation validation and query snapshots.
- Extend logical/compiler/UDF inputs for time and static values.

Exit gate: pure temporal fixtures prove exact values, DST behavior, corpus/query
parity, old fingerprint stability, and strict malformed-input rejection without
loading a model.

### Phase 2: provider and patch backbones

- Add `TransformersSeriesEmbeddingProvider`, artifact lifecycle, and adapter
  allowlist.
- Implement PatchTST and PatchTSMixer adapters.
- Add session dispatch and close behavior.

Exit gate: tiny local backbones prove direct-model parity, bounded batches,
`(B,C,P,D)` validation, pooling, query/corpus equality, and no planning imports or
network work.

### Phase 3: TimesFM

- Implement TimesFM 1/2 frequency resolution and bare-backbone adapter.
- Implement feature-detected TimesFM 2.5 adapter.
- Reject nonmultiples of patch length and all unsupported covariates.

Exit gate: tiny local configurations prove categorical frequency changes the
embedding for 1/2, 2.5 receives no frequency, masks/pooling are correct, and an
older Transformers runtime fails before corpus execution.

### Phase 4: encoder-decoder backbones

- Implement role-based target/dynamic input packing.
- Implement static real/categorical packing.
- Implement shared Time Series Transformer/Informer encoder path.
- Implement Autoformer's separate encoder-input concatenation.

Exit gate: direct calls to tiny local models match provider results; tests prove
that decoder state, trend output, missing lag history, wrong feature recipes, and
static-cardinality mismatches are rejected.

### Phase 5: catalog, documentation, and qualification tooling

- Add catalog version 2 support while retaining strict version 1 parsing.
- Update the API documentation and changelog for behavior that now ships.
- Add `scripts/qualify_transformers_series.py` with the command contract below.
- Do not add a notebook, checkpoint preset, or model recommendation in this
  change; those require a separate semantic/retrieval qualification.

Exit gate: sidecar/table/catalog round trips preserve exact representation
identity; package installation remains model-runtime-free; the opt-in runner
produces a complete machine-readable artifact/runtime record from pinned inputs.

## Permanent test matrix

### Contracts

- Existing MOMENT and custom model dictionaries round-trip with unchanged
  fingerprints.
- Every new nested field changes the appropriate model or representation
  fingerprint.
- Canonical timezone/cadence representations are stable.
- Catalog version 1 rejects version-2 fields; version 2 accepts and preserves
  them.
- Frequency is mutually exclusive with temporal and static declarations.
- Extension fields on non-Transformers models and wrong provider ABI values fail
  during model specification construction.

### Temporal generation

- Elapsed and civil day behavior across DST boundaries.
- `last` and `end_exclusive` anchors.
- Monthly and quarterly boundary transitions.
- Every TimesFM automatic category and explicit override.
- Every scalar GluonTS-compatible feature against pinned reference vectors.
- Fourier feature order, range, and period boundaries.
- Age generation from series start and rejection when start is absent.
- Corpus and query byte-level Arrow feature parity.

### Adapters

- Correct channel/role layout for univariate and multivariate inputs.
- Correct hidden-state selection and exact pooling axes.
- Config mismatches fail during preparation.
- Unsupported model types and remote-code checkpoints fail closed.
- Empty input does not call the model; final partial batches work.
- Batch splitting and row permutation preserve per-row output within qualified
  tolerance.
- NaN, infinity, wrong rank, wrong dimension, and zero-norm policies fail at the
  correct boundary.

### End-to-end behavior

- Rolling and event windows retain time/static bindings through projections.
- Learned vectors survive rename, Parquet sidecar, table, and catalog round trips.
- Raw rich queries and pre-embedded queries produce identical exact-search
  rankings.
- Representation mismatch fails before query inference.
- Explain/profile output contains no timestamp payloads, static category values,
  credentials, or model secrets.
- Direct sinks remain bounded and do not materialize pandas.

Network checkpoint downloads do not belong in the default unit suite. Keep tiny
local configurations as permanent behavioral tests and run pinned public
checkpoint smoke tests in an explicit qualification job.

## Public adapter smoke targets

These immutable repositories are load-and-shape targets, not endorsed retrieval
models. They resolve all model-family ambiguity for the optional network job.
Synthetic data proves only adapter compatibility.

| Profile | Pinned repository revision | Configuration contract to assert |
| --- | --- | --- |
| PatchTST | `ibm-granite/granite-timeseries-patchtst@7fe295d8bc8fbac8041b60ab351882634165517f` | `L=512`, `C=7`, `D=128`, patch `12/12` |
| PatchTSMixer | `ibm-granite/granite-timeseries-patchtsmixer@90dc5a88d45f032b7dceefb5d814ca2af54f2ff9` | `L=512`, `C=7`, `D=48`, patch `16/16` |
| TimesFM 1/2 | `google/timesfm-2.0-500m-pytorch@dc2443792ce5516872b89b37cf1bc058c3bf0c10` | context `2048`, patch `32`, `D=1280`, one target |
| TimesFM 2.5 | `google/timesfm-2.5-200m-transformers@5a9806b9b291fad9233b5249d88263f1846304d3` | context `16384`, patch `32`, `D=1280`, one target |
| Time Series Transformer | `huggingface/time-series-transformer-tourism-monthly@2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35` | context `24`, max lag `37`, `L=61`, `D=26`, time `2`, dynamic `0`, static real `1`, static categorical `1` with `[366]` |
| Informer | `huggingface/informer-tourism-monthly@da30269a25658b3626dda20a94c9bf6b765f1b13` | context `24`, max lag `37`, `L=61`, `D=32`, time `2`, dynamic `0`, static real `0`, static categorical `1` with `[366]` |
| Autoformer | `huggingface/autoformer-tourism-monthly@3801324c8213b225c5ee93ee53f7bfd8094ae7e2` | context `24`, max lag `37`, `L=61`, `D=64`, time `2`, dynamic `0`, static real `0`, static categorical `1` with `[366]` |

Run TimesFM 2.5 with Transformers 5.17.0 or later. The six older targets must
also be covered by the 4.57.6 capability matrix. A network smoke failure does
not justify changing a configured count, tensor layout, or checkpoint config;
record the exact upstream incompatibility.

## Qualification command contract

The runner accepts one canonical `EmbeddingModelSpec` JSON file and newline
delimited rich queries:

```text
scripts/qualify_transformers_series.py
  --spec MODEL_SPEC.json
  --queries QUERIES.jsonl
  --device cpu|cuda
  --batch-sizes 1,8
  --output QUALIFICATION.json
```

Each query line has `values`, optional RFC 3339 `time` and `series_start`, and
optional `static` fields matching `series_query()`. The runner rejects mutable
revisions, overwrites, unknown fields, and inputs that do not match the model
specification. It prepares once, runs each requested batch size twice, compares
row outputs, and writes the [Qualification record](#qualification-record) plus
loading information and exact maximum absolute/relative differences. It exits
nonzero on any preparation, validation, nonfinite-output, shape, or determinism
failure. It never uploads artifacts and never logs access tokens.

Example invocation in an explicitly provisioned environment:

```bash
uv run --with "transformers[torch]==5.17.0" --with huggingface_hub \
  python scripts/qualify_transformers_series.py \
  --spec /absolute/path/model-spec.json \
  --queries /absolute/path/queries.jsonl \
  --device cpu \
  --batch-sizes 1,8 \
  --output /absolute/path/qualification.json
```

## Verification commands

Run focused checks while implementing, then both repository-wide gates:

```bash
uv run pytest \
  tests/test_series_embeddings.py \
  tests/test_series_providers.py \
  tests/test_featurestore.py \
  -o addopts='--strict-config --strict-markers'
uv run --with "transformers[torch]==4.57.6" --with huggingface_hub \
  pytest tests/test_series_providers.py \
  -o addopts='--strict-config --strict-markers'
uv run --with "transformers[torch]==5.17.0" --with huggingface_hub \
  pytest tests/test_series_providers.py \
  -o addopts='--strict-config --strict-markers'
uv run ruff check src/duckpd tests scripts/qualify_transformers_series.py
uv run ruff format --check src/duckpd tests scripts/qualify_transformers_series.py
uv run pyright
make check
make build
```

The focused pytest commands intentionally disable the repository coverage gate;
`make check` is the required full-suite coverage proof. The 4.57.6 run exercises
the six older adapters and the deterministic TimesFM 2.5 unavailable error; the
5.17.0 run exercises all seven against models built from tiny configs, saved
locally, and reloaded through `AutoModel`. Public checkpoint qualification is
opt-in and must not be smuggled into any test command.


## Qualification record

For each checkpoint approved for examples or production guidance, record:

- repository, immutable revision, artifact digest, architecture, bare class, and
  configuration digest;
- source and weight licenses;
- Transformers, PyTorch, Python, operating system, and accelerator versions;
- input channels and their real semantics, roles, units, cadence, timezone,
  temporal recipe, static vocabularies, and outer/internal normalization;
- pooling rule, output dimension, and final unit normalization;
- cold preparation time, cache bytes, peak RSS/VRAM, throughput, and supported
  batch sizes;
- CPU/accelerator determinism tolerance;
- retrieval metrics against named deterministic and PCA baselines on held-out
  entities and chronology.

A checkpoint trained on seven ETTh1 channels is not qualified for seven OHLCV
channels merely because the tensor shape matches. Configuration compatibility is
a runtime condition; semantic compatibility is a data/model qualification
condition.

## Primary references

- [DuckPD time-series embedding design](time-series-embeddings.md)
- [DuckPD time-series embedding API](../api/time-series-embeddings.md)
- [Hugging Face PatchTST documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/patchtst)
- [Hugging Face PatchTSMixer documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/patchtsmixer)
- [Hugging Face TimesFM documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/timesfm)
- [Hugging Face TimesFM 2.5 documentation](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/timesfm2_5)
- [Hugging Face Time Series Transformer documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/time_series_transformer)
- [Hugging Face Informer documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/informer)
- [Hugging Face Autoformer documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/autoformer)
- [Hugging Face time-series preprocessing walkthrough](https://huggingface.co/blog/time-series-transformers)
- [Pinned GluonTS calendar feature source](https://github.com/awslabs/gluonts/blob/889a3df86a89a365880b4bc1488bcf4c039f265e/src/gluonts/time_feature/_base.py)
- [Pinned GluonTS age feature source](https://github.com/awslabs/gluonts/blob/889a3df86a89a365880b4bc1488bcf4c039f265e/src/gluonts/transform/feature.py)
- [Google Research TimesFM repository](https://github.com/google-research/timesfm)
- [Archived TimesFM 1.0/2.0 frequency contract](https://github.com/google-research/timesfm/blob/master/v1/README.md)
