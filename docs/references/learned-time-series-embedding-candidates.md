# Draft proposal: learned time-series embedding candidates

**Status:** TSPulse and TS2Vec providers implemented as unqualified controls.

**Research date:** 2026-09-11

**Independent review date:** 2026-09-14

**Related contracts:** [time-series embedding API][api-series],
[time-series embedding design][design-series], and
[Phase 19 roadmap][roadmap-phase-19].

## Executive decision

DuckPD will focus first-party learned-series work on exactly two providers:
**TSPulse** and **TS2Vec**. The fixed multivariate evaluation schema should
combine semantically ordered channels such as returns, volume surprise,
intrabar range, liquidity or spread, market return, and sector return. Training
belongs in an external producer/research workflow; DuckPD owns only frozen,
attested inference through the existing provider boundary.

The implementation order is:

1. Implement the immutable TSPulse search checkpoint as a narrow univariate
   control: 512 observations, one target channel, internal RevIN only,
   decoder/register extraction, 240 float32 values, and CPU execution.
2. Train TS2Vec outside DuckPD on eligible domain data and export a frozen
   artifact with ordered channels, preprocessing, pooling, weights, and
   provenance.
3. Verify that artifact through the built-in local-bundle provider.
4. Compare both against native, train-fitted PCA, compact statistical, and
   bounded multivariate-DTW baselines.

The first TSPulse provider intentionally excludes multivariate adaptation,
independent channel concatenation, fitted outer scaling, alternate readouts,
interpolation, padding, and accelerators. Each would create a different
representation or support contract and requires separate evidence.

No reviewed evidence yet identifies a winner on financial retrieval or proves
incremental predictive utility. The benchmark must evaluate retrieval geometry
and predictive usefulness separately.

## Focused provider set

| Candidate | Artifact/training path | Principal value | Main unresolved risk | Disposition |
| --- | --- | --- | --- | --- |
| TSPulse published search checkpoint | Immutable public checkpoint and pinned runtime | Immediate real-model control for the provider and artifact lifecycle | Published evidence is univariate and non-financial; outer-scaling recipe is ambiguous | Implement first with internal RevIN only |
| TS2Vec trained on DuckPD-domain data | Train and attest an application-owned checkpoint | Compact joint multivariate learned baseline | Pooling may erase event position; training labels and negatives may be economically wrong | External producer and frozen inference implemented; qualification pending |

Other learned architectures remain outside the planned provider surface. They
may be reconsidered only after TSPulse and TS2Vec have measured failure modes
that justify additional maintenance and runtime burden.

## Independent review disposition

The independent review confirmed:

- the TSPulse revision, weight-file digest and size, package pin, Python bound,
  preprocessing description, decoder/register dimension, and reported
  retrieval-table transcription;
- TS2Vec's MIT-licensed reviewed revision and genuine joint-channel input path;
- TSPulse univariate qualification is a recipe reproduction, not evidence of
  joint-channel behavior;
- external training is in scope while training inside DuckPD is not;
- distance metric belongs to a benchmark/search recipe, while output
  normalization remains part of representation identity; and
- Python 3.14 support is a promotion gate, not a gate for isolated research.

Claims about financial quality, adapted-model behavior, and predictive utility
remain unproven.

## Scope and decision standard

DuckPD's product question is historical similarity retrieval over ordered,
fixed-length time-series windows. It is not forecasting quality, anomaly score,
classification accuracy, or linear-probe performance in isolation. A viable
learned adapter must produce one deterministic `FLOAT[D]` value per input row
and must preserve the representation-space identity already required by
DuckPD.

The existing qualification contract requires all of the following before a
built-in adapter can be treated as supported:

- exact checkpoint revision and digest of every numerical artifact;
- source, package, Python, operating-system, accelerator, and runtime versions;
- source and weights licenses, including deployment restrictions;
- accepted input lengths, channel counts, dtypes, masks, and missing values;
- every outer and internal normalization, scaling, clipping, tokenization,
  padding, and interpolation operation;
- extraction point, pooling rule, output dimension, and output normalization;
- CPU and accelerator memory, throughput, batch-size, and cold-start results;
- determinism tolerances across every supported provider; and
- retrieval results against declared DuckPD native baselines.

The review therefore distinguishes three evidence classes:

1. **Observed:** directly stated by a model card, source file, package metadata,
   immutable artifact manifest, or paper.
2. **Derived:** mechanically calculated from observed configuration or source.
3. **Proposed:** a DuckPD adapter or benchmark choice that remains to be tested.

## Search method

The review favored models that advertise representation learning, diagnostic
analysis, tokenization, semantic similarity, or retrieval. Forecasting models
were retained only when they exposed a plausible encoder state. Candidates
were screened against:

- checkpoint availability and immutability;
- source and weights licensing;
- a callable, inspectable embedding path;
- fixed-vector semantics;
- preprocessing completeness;
- runtime/package compatibility;
- direct retrieval evidence; and
- compatibility with DuckPD's current provider protocol.

Primary sources were used for material findings: model repositories, immutable
artifact APIs, package indexes, official notebooks, source files, and papers.
The final section lists the exact sources a verifier should revisit.

## TSPulse: published control and adapted joint candidate

### Why TSPulse changes the shortlist

TSPulse targets diagnostic tasks rather than using a forecasting model as an
incidental encoder. Its model card explicitly names similarity search, and the
paper evaluates zero-shot semantic embeddings for retrieval under temporal
shift, magnitude scaling, and additive noise. Those properties align with
DuckPD's historical-pattern retrieval problem substantially better than a
forecasting score does.

The model remains unqualified for DuckPD because the published benchmark is not
financial or event-window retrieval, does not compare against DuckPD's native
representations, and does not establish DuckPD runtime or determinism behavior.
It is nevertheless the best-supported off-the-shelf retrieval control found.

### Exact artifact record

The proposed checkpoint is the search/imputation specialization, not the
repository's default anomaly-detection branch.

| Field | Observed value |
| --- | --- |
| Model repository | `ibm-granite/granite-timeseries-tspulse-r1` |
| Requested revision name | `tspulse-hybrid-dualhead-512-p8-r1` |
| Resolved immutable model revision | `b12164578f7b893ada0028c00d292ba10383d25a` |
| Weights file | `model.safetensors` |
| Weights size | 4,305,624 bytes |
| Weights SHA-256 | `b9332ae796ec7c313f991ed32dbec62c29a8e673281decb7308f955bdda7aae0` |
| Hugging Face metadata parameter count | 1,084,330 float32 parameters |
| Independently counted stored tensor elements | 1,068,958 float32 elements across 224 tensors |
| Config Git blob ID | `97bfe54c22c81913dc49e6efba0b198045ace32e` |
| Model license metadata | Apache-2.0 |
| Source package | `granite-tsfm` |
| Proposed package pin | `granite-tsfm==0.3.9` |
| Source tag commit for `v0.3.9` | `fe7a35697723e2a2f5246ae979474bfc554e26c0` |
| `0.3.9` wheel SHA-256 | `07ca9c503cfa9f7e34d6ee5616d1a37ae301cabc6daed0546b249ddbc2c8f272` |
| Source license | Apache-2.0 |
| Package Python requirement | `>=3.11,<3.14` |

The Hugging Face LFS digest covers `model.safetensors`; it is not yet the
DuckPD artifact-manifest digest. A prepared DuckPD model would need a canonical
manifest covering at least the weights, config, package/source revision,
adapter source, and every auxiliary numerical artifact. The resulting manifest
digest must be computed during preparation rather than copied from this table.

### Architecture and embedding contract

The search checkpoint configuration states:

| Configuration | Value |
| --- | ---: |
| Context length | 512 |
| Input channels in checkpoint config | 1 |
| Patch length | 8 |
| Patch stride | 8 |
| Configured patch count | 128 |
| Backbone width | 24 |
| Decoder width | 24 |
| Register tokens | 10 |
| Internal scaling | RevIN |
| RevIN affine parameters | Enabled |
| Minimum scale | 0.001 |
| Model dtype | float32 |

The public `get_embeddings()` helper accepts:

```text
past_values:        [batch, length, channels]
past_observed_mask: [batch, length, channels] or None
component:           "backbone" or "decoder"
mode:                "time", "fft", "register", or "full"
```

Its documented result has shape `[batch, channels, D]`. For
`component="decoder"` and `mode="register"`, the helper returns the final
`patch_register_tokens * decoder_d_model` values. The fixed dimension per
channel is therefore:

```text
10 register tokens * 24 decoder features = 240 values
```

This 240-dimensional value is the representation used by the official search
path. The official notebook:

- loads the search-specific revision;
- overrides `num_input_channels=1`;
- sets `mask_type="user"`;
- batches without shuffle;
- runs under `torch.no_grad()`;
- calls `get_embeddings(model, past_values)` with its default decoder/register
  extraction;
- removes the one-channel axis; and
- indexes the resulting vectors using L2 distance.

This is materially stronger than relying on an undocumented hidden state.
Extraction still needs to be pinned in DuckPD's `adapter_revision`; future
changes to the helper's default arguments must not silently create a different
representation space.

### Preprocessing ambiguity that must be resolved

The checkpoint applies affine RevIN internally. The official cookbook search
notebook also creates a `TimeSeriesClassificationPreprocessor` with
`scaling=True`. At package defaults, this selects standard scaling, fits a
scaler on the training/index data, and reuses that scaler for the query split.
The published recipe therefore appears to include **both** corpus-fitted outer
standard scaling and model-internal RevIN.

That outer scaler is not a harmless implementation detail:

- it is fitted state and must be versioned with the representation;
- it can leak information if fitted across a chronological evaluation boundary;
- query-only use requires the exact index-fitted scaler;
- it conflicts with a claim that the adapter is stateless; and
- omitting it may or may not change retrieval because internal RevIN also
  rescales each sample.

The first benchmark must compare at least:

1. exact cookbook preprocessing: index-fitted standard scaling plus internal
   RevIN;
2. internal RevIN only; and
3. DuckPD-declared outer normalization plus internal RevIN, only if that outer
   normalization represents a meaningful product space.

The chosen path must be encoded in the representation fingerprint. DuckPD must
not silently reproduce notebook preprocessing without recording the fitted
scaler artifact and split boundary.

### Channel semantics

TSPulse accepts tensors with a channel dimension, but its paper states that
pretraining used `c=1`, treating channels independently. Cross-channel mixing
is activated during task-specific fine-tuning. The public search notebook also
uses univariate data and squeezes the one-channel axis.

Consequences:

- The zero-shot evidence supports one 240-dimensional vector **per channel**.
- It does not establish a jointly learned target/covariate embedding.
- Concatenating channel vectors would yield `240 * C` dimensions but would be a
  DuckPD-defined representation with no published benchmark.
- Averaging channel vectors would discard channel identity and semantic role.
- Using target, past covariate, and known-future covariate channels together
  would overstate the checkpoint's qualified semantics.

The published-recipe reproduction should therefore be univariate and
target-only. The primary product experiment must instead train and attest a
multichannel TSPulse adaptation.

### Proposed multivariate adaptation

The paper and source expose a distinct adaptation path: retain pretrained
weights, enable identity-initialized decoder channel mixing, and train on the
target multivariate schema. DuckPD should compare four separately fingerprinted
variants:

1. original univariate search recipe;
2. independent per-channel register vectors with ordered concatenation;
3. trained decoder channel mixing plus a compact joint readout; and
4. trained mixing/readout plus an auxiliary predictive head.

The first trained readout should produce one 256-dimensional vector from
role-ordered decoder states. Compare register-only pooling with a timing-aware
readout that can retain temporal states. Log every newly initialized or
shape-mismatched parameter when the channel count changes; a successful load
must not conceal an untrained multivariate path.

This adaptation creates a new model artifact and representation. It does not
inherit the published checkpoint's retrieval claims. Backbone unfreezing,
retrieval supervision, scale-retaining side features, and predictive auxiliary
losses should be explicit ablations rather than bundled into the first result.

### Missing values and masks

The helper exposes `past_observed_mask` with the same shape as the input. The
checkpoint's search path is nevertheless documented for complete 512-point
examples. DuckPD's current provider contract supplies complete float32 windows
and does not support padding masks. The narrow experiment should preserve that
contract:

- reject incomplete windows before inference;
- require exactly 512 observations;
- provide an all-observed mask or `None`, whichever is proven equivalent;
- prohibit hidden padding or interpolation; and
- reject NaN and infinity before model invocation.

TSPulse's imputation capability is not evidence that two differently missing
windows inhabit a stable retrieval space. Missing-window retrieval requires an
independent contract and benchmark.

### External retrieval evidence

The TSPulse paper evaluates semantic register embeddings on synthetic and UCR
corpora. The index contains 1,680 synthetic segments and 5,960 real segments.
Queries are derived from indexed examples using:

- random temporal shifts within plus or minus 20 percent;
- random magnitude scaling within plus or minus 20 percent; and
- Gaussian noise with standard deviation equal to 10 percent of scaled signal
  magnitude.

It defines a coarse family-match task and a fine-grained task, uses Euclidean
distance, and reports top-three precision, reciprocal rank, average precision,
and nDCG. The reported averages are:

| Task and metric | TSPulse | MOMENT | Chronos |
| --- | ---: | ---: | ---: |
| Family PREC@3 | **0.678** | 0.532 | 0.234 |
| Family MRR@3 | **0.784** | 0.585 | 0.252 |
| Family AP@3 | **0.664** | 0.519 | 0.226 |
| Family NDCG@3 | **0.697** | 0.539 | 0.235 |
| Fine-grained PREC@3 | **0.584** | 0.415 | 0.176 |
| Fine-grained MRR@3 | **0.723** | 0.477 | 0.195 |
| Fine-grained AP@3 | **0.570** | 0.404 | 0.170 |
| Fine-grained NDCG@3 | **0.610** | 0.425 | 0.178 |

The real-data results show a larger separation than the synthetic results. For
example, real family-match PREC@3 is 0.645 for TSPulse, 0.389 for MOMENT, and
0.116 for Chronos. This makes TSPulse a credible retrieval candidate, but the
ground truth is still based on UCR dataset/class identity rather than financial
motifs, event reactions, or application labels.

The paper's ablation also supports using register embeddings rather than an
arbitrary model segment. On its reported retrieval setting, removing register
embeddings and using time embeddings reduces PREC@3 from 0.645 to 0.314; using
FFT embeddings reduces it to 0.201. The exact decoder/register extraction is
therefore part of the candidate, not an adapter preference that can be changed
without requalification.

### External efficiency evidence

The paper reports a computational comparison on an NVIDIA A100 system with 16
CPU cores and 256 GB RAM, using an input tensor shaped
`[batch=32, length=512, channels=5]`:

| Model | Parameters (millions) | GPU inference | CPU inference | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: |
| TSPulse | 1.06 | 7.16 ms | 0.06 s | 0.39 GB |
| MOMENT-small | 35.34 | 32.57 ms | 2.74 s | 0.56 GB |
| VQShape | 37.09 | 65.37 ms | 1.15 s | 6.88 GB |

These are author-reported numbers for their environment and workload. They do
not establish DuckPD's throughput, Arrow conversion cost, cold-start time,
process RSS, or behavior on one-channel windows. DuckPD must remeasure the
entire provider path, not cite these figures as its own performance.

### Runtime and maintenance risks

- `granite-tsfm==0.3.9` requires Python below 3.14, while DuckPD declares support
  for Python 3.14.
- The package introduces PyTorch and Transformers into an otherwise lightweight
  optional inference surface.
- IBM's repository disclosure says the code is an open-source project rather
  than an IBM product and makes no maintenance commitment.
- Search examples found during review use both `granite-tsfm==0.3.3` and source
  notebooks on moving branches. DuckPD must pin one package/source pairing and
  test it rather than follow `main`.
- The model's training set includes Bitcoin, but that does not establish
  retrieval quality across securities, frequencies, event types, or market
  regimes.
- CPU determinism, cross-platform tolerance, and accelerator equivalence have
  not been measured for DuckPD.

### Proposed published-recipe control contract

The following is a benchmark proposal, not an approved public model spec:

| Contract field | Proposed first value |
| --- | --- |
| Model | `ibm-granite/granite-timeseries-tspulse-r1` |
| Model revision | `b12164578f7b893ada0028c00d292ba10383d25a` |
| Source/package | `granite-tsfm==0.3.9`, source tag commit `fe7a35697723e2a2f5246ae979474bfc554e26c0` |
| Input length | Exactly 512 |
| Channels | Exactly one |
| Channel role | `target` |
| Input dtype | Finite float32 |
| Missingness | Unsupported; complete windows only |
| Model mask mode | `mask_type="user"`; prove all-observed mask versus `None` |
| Extraction | `component="decoder"`, `mode="register"` |
| Output | Flatten `[batch, 1, 240]` to `[batch, 240]` |
| Search recipe | Raw L2 first, matching published evaluation; metric is not representation identity |
| Unit normalization | Disabled for exact-recipe comparison; benchmark cosine/unit-normalized as a separate representation |
| Outer scaling | Unresolved; benchmark cookbook scaler and internal-only paths separately |
| Internal scaling | Affine RevIN, minimum scale 0.001 |
| Provider | Existing custom series provider until qualification passes |
| Device | CPU first; accelerator is a separate provider/runtime record |

The representation fingerprint must distinguish model, preprocessing,
extraction, pooling, and output-normalization variations. A raw vector searched
with L2 or cosine retains one representation fingerprint; the benchmark record
must identify the metric as part of its search recipe. Unit-normalizing the
stored vector changes the representation and therefore its fingerprint. Fitted
outer-scaler state must be covered by artifact identity.

### TSPulse promotion gates

Any TSPulse adapter should become built-in only if all of these pass:

1. Immutable preparation verifies every downloaded byte before loading.
2. A canonical DuckPD artifact manifest is computed and persisted.
3. The adapter produces the declared number of finite values for every accepted
   input.
4. Batch partitioning does not change output beyond a declared tolerance.
5. Repeated CPU runs and process restarts satisfy the determinism tolerance.
6. The adapted multivariate model improves at least one named joint retrieval
   task over its best native and trained baseline by a predeclared material
   threshold.
7. The gain survives entity-held-out and chronology-held-out evaluation.
8. The result survives realistic perturbations without destroying sensitivity
   to economically meaningful level, direction, dependency, or timing.
9. Cold start, peak RSS, throughput, and storage remain within declared budgets.
10. Python 3.14 is supported upstream or DuckPD explicitly documents and tests a
    narrower optional-adapter matrix.

## Joint TS2Vec trained on DuckPD-domain data

### Verified architecture and license

At source revision `b0088e14a99706c05451316dc6db8d3da9351163`,
TS2Vec accepts `[batch, time, channels]`. Its first learned operation is
`Linear(input_dims, hidden_dims)`, so channel values are mixed before the
dilated temporal convolution stack. This is genuine joint input processing,
not independent encoding followed by pooling.

The standard inference path uses the stochastic-weight-averaged network in
evaluation mode. `encoding_window="full_series"` max-pools timestamp states
to one vector per example. The reviewed source is MIT-licensed.

### Why it is the first trained baseline

TS2Vec answers the central experiment with relatively little machinery:

- fixed channel count and order map directly to DuckPD's model specification;
- output dimension is chosen at training time;
- one full-series vector already exists;
- temporal states remain available in the producer workflow; and
- no public checkpoint is needed because DuckPD can attest the exported model.

The proposed first variants use 128- and 256-dimensional output and train on
the same eligible windows used for TSPulse adaptation. Reproduce the standard
hierarchical contrastive objective before altering it. Compare standard max
pooling with one timing-aware or temporal-pyramid readout.

### Risks and controls

- Full-series max pooling may erase event position and order.
- Random negatives can push economically similar or synchronized market
  windows apart.
- Highly overlapping windows can exaggerate the effective training sample.
- The upstream encoder modifies input tensors in place while replacing missing
  values and applying masks. A provider must own or copy mutable input memory;
  it must not assume a tensor sharing Arrow memory is safe.
- The historical training environment is not the required inference
  environment. Modernize only the frozen inference component and prove output
  parity.
- Export whether the averaged or raw network is used. That choice belongs in
  artifact and adapter identity.

Training must use a streaming or sharded producer outside DuckPD rather than
materializing an unbounded corpus through the upstream in-memory `fit()` API.

## Chronos-2-small pretrained multivariate challenger

### Recorded artifact metadata and verified interface

| Field | Observed value |
| --- | --- |
| Model | `autogluon/chronos-2-small` |
| Immutable revision | `ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a` |
| Weights SHA-256 | `492290ae82bb89f9769e3479ce90b3179de1f33e600c34daa0352531538b23cd` |
| Weights size | 111,749,048 bytes |
| Hugging Face safetensors metadata total | 27,934,624 float32 elements |
| License | Apache-2.0 |
| Pipeline source reviewed | `4dbf163c2734c089cdf7da2b86fde48862ff9c6f` |

The public `Chronos2Pipeline.embed()` accepts `[batch, variates, history]`.
Its documentation states that information is shared among variates within each
example. It returns one tensor per example shaped
`[variates, num_patches + 2, d_model]` plus per-series location and scale. The
extra tokens represent a register and a masked output-patch token.

This is a documented multivariate extraction surface, not an undocumented
hidden-state hack. It still does not define a database-ready search vector.

### Proposed experiment and risks

Compare a fixed role-ordered readout of observed-history states with a learned
role-aware readout. Treat register-only extraction as an ablation. Evaluate
whether returned location/scale values should enter a separate state-aware
readout.

Chronos grouping must keep channels from one logical window together and
separate unrelated rows. Prove that a query embedded alone agrees with the same
query under different provider batch partitioning. Do not confuse the older
Chronos baseline in the TSPulse paper with this distinct Chronos-2-small model.
Its forecasting origin provides no financial retrieval guarantee.

## MantisV2 compact representation challenger

### Recorded artifact metadata and verified architecture

| Field | Observed value |
| --- | --- |
| Model | `paris-noah/MantisV2` |
| Immutable revision | `8f6ca35cb54ab14b120618943c6fca5ddf5a76a6` |
| Weights SHA-256 | `49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1` |
| Weights size | 16,771,648 bytes |
| Hugging Face safetensors metadata total | 4,188,690 float32 elements |
| License | Apache-2.0 |
| Source revision reviewed | `9018b98b4c1e093d2fa618338695cd57146d3cd0` |

MantisV2's base encoder accepts one channel. It derives original-signal,
difference, mean, and standard-deviation features, then exposes CLS, mean, or
combined token extraction. Default hidden width is 256; combined extraction is
512 values per independently encoded channel.

The package also provides a trainable `LinearChannelCombiner` that projects the
channel axis before base-model encoding. This is a real learned joint adapter,
though less expressive than unrestricted time-by-channel attention.

### Proposed experiment and packaging blocker

Run frozen ordered per-channel concatenation as a control, then train the
channel combiner and a compact joint readout. Compare CLS and combined
extraction only on development data.

`mantis-tsfm==1.1.0` permits Python 3.14 but requires `pandas<3.0`; DuckPD
requires `pandas>=3.0,<3.1`. The standard packages cannot share one supported
environment. Run Mantis in a separate benchmark environment until upstream
publishes compatible dependencies or an inference-only integration is
deliberately maintained.

## MOMENT-1-small historical learned control

### Role in the proposal

MOMENT exposes an explicit embedding mode and is compared directly with
TSPulse in the published retrieval paper. Retain it to reproduce that historical
comparison, not as the primary joint candidate: it is much larger, loses the
reported retrieval comparison, averages channels by default, and has a
problematic published package surface.

### Exact artifact record

| Field | Observed value |
| --- | --- |
| Model repository | `AutonLab/MOMENT-1-small` |
| Immutable model revision | `411e288267f82cce86296dbe4d6c8bc533cc162f` |
| Safetensors file size | 151,615,328 bytes |
| Safetensors SHA-256 | `785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877` |
| Safetensors parameter count | 37,901,512 float32 parameters |
| Model/source license metadata | MIT |
| Sequence length | 512 |
| Patch length and stride | 8 and 8 |
| Encoder width | 512 |
| Backbone | `google/flan-t5-small` configuration embedded in the model config |
| Published package | `momentfm==0.1.4` |
| Published package requirement | Python `>=3.10` |
| Current source commit reviewed | `38f7310ad594100747ca2a8357e9c7ca7d323e0e` |

### Embedding behavior

The current `MOMENT.embed()` source accepts `x_enc` shaped
`[batch, channels, sequence]` and a mask shaped `[batch, sequence]`. It:

1. applies RevIN;
2. converts NaN and infinities to zero;
3. creates length-eight patches;
4. encodes channels independently through the T5 encoder; and
5. returns either unreduced patch states or a default mean reduction.

For `reduction="mean"`, the implementation first averages across channels and
then computes a mask-weighted average across patches. The result is one
512-dimensional vector per input example. For `reduction="none"`, the output
retains `[batch, channels, patches, 512]` and requires a DuckPD-owned pooling
rule.

Default mean reduction is convenient but semantically lossy for DuckPD:

- target and covariate identities disappear;
- channel order no longer affects the vector;
- a channel added twice can alter weighting without adding information; and
- missing masks are shared across channels.

The first MOMENT control should therefore also be univariate. A multichannel
MOMENT adapter requires a separate, explicit channel-role aggregation contract.

### Packaging risk

PyPI currently publishes `momentfm==0.1.4` with exact pins:

- `huggingface-hub==0.24.0`;
- `numpy==1.25.2`;
- `torch~=2.0`; and
- `transformers==4.33.3`.

The repository's current source identifies itself as 0.1.5 and relaxes these
dependencies, but PyPI metadata reviewed here still lists 0.1.4 as latest. A
DuckPD adapter must not quietly mix the immutable model with an unversioned
`main` checkout. The options are to wait for a compatible release, pin an exact
source commit as an independently attested runtime, or run MOMENT in a separate
benchmark environment. The third option is safest for a control and does not
create a production dependency promise.

### Disposition

Use MOMENT-1-small as a benchmark control with:

- one complete 512-point float32 target channel;
- explicit `task_name="embedding"`;
- explicit `reduction="mean"`;
- exact model revision and safetensors digest;
- recorded package/source commit; and
- raw and unit-normalized distance variants kept as separate spaces.

Do not promote it unless it independently beats native baselines and its
package/runtime story is resolved.

## VQShape lower-priority experimental control

### Why retain it

VQShape learns an interpretable codebook of reusable shapes. That makes it a
useful challenger for motif retrieval even though its published evaluation is
classification-oriented rather than retrieval-oriented. Its histogram output
could also provide an interpretable representation of which learned shapes
occur in a window.

### Observed artifact and interface facts

| Field | Observed value |
| --- | --- |
| Repository | `YunshiWen/VQShape` |
| Current source revision reviewed | `24fc507461e17b4b8485d5f097aa91eaef98ae67` |
| Repository license metadata | MIT |
| Checkpoint release | `v0.1.0-cls` |
| Proposed checkpoint asset | `uea_dim256_codebook512.zip` |
| Asset size | 104,649,910 bytes |
| Release immutability | `false` in GitHub API metadata |
| Published asset digest | None |
| Embedding dimension | 256 |
| Codebook size | 512 |
| Parameter count | 9.5 million |
| Reported mean token classification accuracy | 0.731 |
| Reported mean histogram classification accuracy | 0.711 |
| Documented Python | 3.11 |

The official example interpolates each sequence to 512 points, reshapes
multichannel data into independent univariate sequences, and calls the model in
`mode="tokenize"`. It returns token and histogram representations. The exact
shape, ordering, pooling, histogram normalization, and multi-channel
reassembly required for DuckPD still need source-level verification.

### Qualification problems

- The release is mutable and the chosen asset has no published digest. DuckPD
  would need to download the exact asset, compute its digest, inspect its
  contents, and create its own immutable manifest.
- Official preprocessing uses linear interpolation to 512 points. Hidden
  interpolation is incompatible with DuckPD's current complete fixed-window
  contract. The initial benchmark should require already aligned 512-point
  windows instead.
- Channels are flattened into independent univariate series in the documented
  tokenization path. A row-level multichannel vector requires new aggregation
  semantics.
- The reported accuracy is classification accuracy, not nearest-neighbor
  retrieval quality.
- The code uses a research PyTorch Lightning environment rather than a compact
  inference package.

### Disposition

Retain only as a lower-priority experimental candidate behind MantisV2. Before
running it, pin the source commit, hash the chosen release archive and all
contained checkpoints, inspect the exact token/histogram tensor shapes, and
declare one deterministic row-level vector. Do not implement hidden
interpolation.

## Screened-out candidates

### TimeSiam

TimeSiam learns representations for temporally separated subseries and exposes
multiple lineage embeddings. The repository documents pretraining and
fine-tuning experiments and links datasets, but this review did not find a
standalone, immutable generic inference checkpoint or a fixed-vector search
contract. Choosing a lineage and pooling rule would become DuckPD-owned model
semantics. Defer unless a portable checkpoint and retrieval evaluation appear.

### T-Rep

T-Rep provides an encoder after per-dataset training, but the repository states
that it is not compatible with PyTorch 2.0 and recommends a dedicated
environment with specific old dependencies. It does not supply a canonical
portable zero-shot checkpoint for DuckPD. Reject as a built-in; it remains a
possible application-trained custom provider.

### TOTEM

TOTEM is conceptually relevant because it learns tokenized time-series
representations. The reviewed GitHub repository reports no detected license,
and its artifact distribution does not provide the immutable, checksummed,
package-addressable model record required by DuckPD. Artifact provenance and
license failures are sufficient to reject it for built-in use without assessing
retrieval quality.

### UniTS

UniTS has MIT-licensed source, explicit sequence and variable attention, and an
own-data training path. Its reviewed `ckpt` release is mutable, publishes no
asset digests, and consists of task-specific `.pth` files rather than one
stable zero-shot search artifact.

Those facts reject an off-the-shelf built-in adapter, not the architecture as a
research candidate. Place UniTS in the second round if simpler joint encoders
plateau. Train or adapt one explicit channel schema and include dataset prompts,
task configuration, readout weights, and every numerical file in artifact
identity.

### TRACE

TRACE is directly interesting for aligned text/time-series retrieval. That is a
different product contract from DuckPD's current independent text and series
spaces. The reviewed repository also reports no detected license. Until license,
artifact provenance, supported domains, and aligned-space metadata are resolved,
TRACE should remain research for a future direct text-to-series feature rather
than a current series adapter.

### Other forecasting-first models

TimesFM, TinyTimeMixer, PatchTST, and related forecasting models remain
architectural references. Their hidden states may be useful, but most require
DuckPD to invent a patch/variate pooling rule, and forecasting accuracy does not
establish historical retrieval quality. Chronos-2-small is handled separately
above because its public `embed()` is multivariate. TimesFM 3's noncommercial
weights still fail the intended production-deployment gate.

## Proposed DuckPD qualification benchmark

### Research questions

The benchmark must answer two questions independently:

1. Which representation best retrieves held-out windows under a declared
   multivariate financial relevance definition?
2. Do learned search vectors, richer encoder states, or neighbor-derived
   features add out-of-sample predictive information beyond strong conventional
   and native baselines?

A model may win one question and lose the other. Neither forecasting loss nor
attractive example neighbors answer both.

### Fixed multivariate data contract

Start with one instrument/window per example and one fixed, ordered feature
schema. A reasonable initial schema is:

1. instrument return;
2. volume surprise;
3. intrabar range;
4. spread or another timestamp-trustworthy liquidity measure;
5. market return; and
6. sector return.

Use only channels available with trustworthy timestamps. Define units,
adjustment policy, sampling interval, session handling, and availability for
every channel. Keep instrument identifiers out of the numerical vector unless
identity dependence is intentional. A changing universe of instruments as
channels is a different masked-membership problem and is out of scope.

Do not stretch a trading session to 512 observations to satisfy a checkpoint.
Use naturally defined 512-observation windows for matched TSPulse experiments,
and test shorter natural windows separately for trainable models.

### Distinguish three channel capabilities

Report these as separate baselines rather than labeling all of them
"multivariate":

| Capability | Information path | Interpretation |
| --- | --- | --- |
| Independent encoding plus mean pooling | Each channel is encoded separately, then averaged | Channel identity can disappear; inadequate as the sole product foundation |
| Independent encoding plus ordered concatenation | Each channel retains a separate vector block | Preserves roles, but learns no dependency or lag relationship |
| Joint representation learning | Input projection, attention, mixer, or trained fusion combines channels | Required candidate class for the primary product experiment |

Ordered concatenation remains useful. Under squared L2 it adds per-channel
distances; it does not learn which cross-channel relationships matter. If an
independent encoder discarded timing, concatenation cannot recover it.

### Three evaluation suites

| Suite | Question | Essential design |
| --- | --- | --- |
| Controlled multivariate fixtures | Does the representation preserve or ignore the intended dependency, lag, role, sign, and scale distinctions? | Hold channel marginals approximately fixed while changing their relationships |
| Historical market and event retrieval | Do neighbors satisfy an independent financial relevance rubric? | Deduplicated events, held-out chronology and instruments, graded relevance |
| Predictive usefulness | Does the frozen representation add out-of-sample information? | Walk-forward probes and neighbor features with explicit label availability |

Controlled fixtures are behavior tests, not market-generalization evidence.
Include same-event-family/different-reaction and
different-event-family/similar-reaction pairs.

### Controlled joint-channel tests

Every trained finalist must be tested with:

1. target-only input;
2. full multichannel input;
3. joint mixing disabled where the architecture supports it;
4. one channel shuffled within controlled strata, preserving its marginal
   distribution while disrupting relationships;
5. removal of each informative covariate group; and
6. correct versus incorrect channel-role assignment.

Fixtures should cover:

- the same price path with an earlier versus later volume shock;
- matching marginals with positive versus negative channel dependence;
- market-wide versus instrument-specific movement after controlling for market
  return;
- continuation versus reversal within one event family; and
- identical numerical values assigned to different semantic roles.

Swapping `(identity, values)` pairs can legitimately leave a set-based model
unchanged. Swapping only values while identities remain fixed should generally
change a role-sensitive representation.

### Representations and baselines

At minimum compare:

1. raw multichannel fixed windows;
2. centered, z-scored, and unit-normalized native variants where their semantics
   fit the task;
3. train-fitted PCA of the native multichannel vector at the same output budget
   as learned candidates;
4. compact relational/statistical features covering return distribution,
   volatility profile, drawdown/recovery, volume surprise, contemporaneous
   channel correlation, and selected lag correlations;
5. bounded multivariate DTW on an evaluation subset with a declared warping
   constraint;
6. TS2Vec joint 128- and 256-dimensional models;
7. the original univariate TSPulse decoder/register representation; and
8. a multivariate TSPulse adaptation only after the narrow published control
   and external training path are independently verified.

Evaluate a common output-size budget and each model's natural output. A compact
learned vector must not receive credit merely because the native baseline was
left at thousands of coordinates without PCA.

Each preprocessing or output-normalization variation is a separate
representation space and fingerprint. Distance metric and top-k belong to the
search-recipe record, not the current representation fingerprint.

### Retrieval tasks and invariance policy

Build independently labeled suites for:

- trend continuation and reversal;
- impulse, overshoot, and recovery;
- volatility burst and decay;
- level shift versus transient spike;
- periodic and repeated motifs;
- market-wide versus idiosyncratic movement;
- aligned price/volume/liquidity reactions;
- event-linked immediate and delayed reaction; and
- same event family with materially different direction or magnitude.

Declare two useful meanings separately:

- **Shape similarity:** selected scale or level changes may be nuisances.
- **State similarity:** amplitude, volatility, volume surprise, liquidity, and
  market context contribute to relevance.

Do not make scale retention conditional on a query. Train separate readouts or
store separate representation columns if both meanings are needed.

Test controlled positive scaling, additive offset where meaningful, Gaussian
and heavy-tailed noise, time shift, local warp, sign reversal, and
jointly-aligned crop. Independently shifting volume, reversing sign, or applying
large event-relative warps may change the label and must not be treated as an
automatic positive augmentation.

### Training, development, index, and final-query separation

Use at least four logical roles:

1. encoder training;
2. finite development/model selection;
3. untouched final queries; and
4. the historical candidate pool eligible for each query.

These roles need not be four simple date blocks. For each fold:

- fit scalers, PCA, codebooks, seasonal baselines, encoder weights, and readouts
  only on eligible training data;
- lock model and search recipes before final evaluation;
- purge or group overlapping windows and forward-label intervals;
- group duplicate events, revisions, and vendor copies under stable IDs;
- evaluate later chronology, unseen instruments, and their intersection;
- use block- or event-group resampling for uncertainty; and
- report unique time coverage and event count, not only the number of highly
  overlapping windows.

Candidate observations may enter the historical index over time while the
encoder remains frozen. For predictive neighbor features at cutoff `t`, a
neighbor's input and outcome label must both be available. A completed input
with an unfinished future outcome is eligible for retrospective shape search,
not outcome aggregation.

Match information content between predictive queries and historical examples.
A five-minute post-event query must compare with the same relative prefix, not
with an encoded sixty-minute completed reaction. Aligning partial inputs with
full outcomes requires a separate supervised asymmetric encoder.

### Training objectives

Begin with disciplined, separately ablated objectives:

- masked reconstruction of aligned time blocks;
- reconstruction of selected channel groups from other channels;
- contrastive agreement under mild label-preserving views; and
- future-target or later-latent prediction using only eligible training labels.

Prevent trivial reconstruction through redundant OHLC-derived channels by
masking related groups together. For contrastive learning, define treatment of
overlapping windows, duplicate events, and synchronized market moves; every
other batch row is not automatically a useful negative.

If hard-negative sampling becomes a demonstrated bottleneck, compare
minibatch- or sampled-neighbor soft targets and a JEPA/self-distillation
objective. Keep every teacher feature and neighbor sample inside the training
partition. Record embedding variance, effective rank, and neighbor concentration
to detect collapse. The reviewed SoftCLT reference code has no confirmed license,
so study the paper without incorporating its source until rights are resolved.

Keep a shared encoder's search readout and predictive readout distinct. The
search projection may intentionally discard information required by prediction.
Retain temporal states in the producer workflow and test them separately.

### Quality and predictive metrics

For retrieval, predeclare nDCG@10 or Recall@10 as the primary metric. Also
report Precision@k, MRR, AP, graded-relevance results, class/regime slices,
direction and lag failures, perturbation stability, neighbor concentration,
and block-resampled confidence intervals.

For predictive usefulness, compare:

| Input | Purpose |
| --- | --- |
| Conventional price, volume, volatility, and market features | Practical baseline |
| Native or PCA window features | Test whether recent history is sufficient |
| Learned search vector | Test the compact representation |
| Frozen temporal encoder states/readout | Test information lost by search compression |
| Conventional plus learned features | Measure incremental value |
| Conventional plus historical-neighbor summaries | Test the retrieval service itself |

Start with linear or ridge probes and one matched strong tabular model. Use
predeclared return, volatility, and direction horizons. Report error and rank
correlation for returns, declared target error for volatility, and calibration
plus discrimination for probabilities. Full trading-strategy evaluation
belongs downstream.

### Runtime, storage, and determinism

Measure preparation, cold import, first inference, warm throughput,
Arrow-to-tensor conversion, model compute, tensor-to-Arrow conversion, peak
RSS, device memory, persistence, exact index construction, eligible-population
filtering, and exact query latency. The current provider uses bounded Python row
snapshots rather than a zero-copy tensor handoff; this overhead may dominate a
small model and must be reported before optimization.

For 10 million float32 vectors with six channels, arithmetic storage before
metadata, indexing, compression, or replicas is:

| Representation | Dimension | Storage |
| --- | ---: | ---: |
| Native 512 by 6 | 3,072 | 122.88 GB |
| Independent TSPulse registers concatenated | 1,440 | 57.60 GB |
| Mantis combined outputs concatenated | 3,072 | 122.88 GB |
| Joint compact readout | 256 | 10.24 GB |

Repeat identical batches within and across processes, vary batch partitioning
and thread counts, test partial batches, verify fixed finite dimensions, and
confirm provider inputs are not mutated. Query/corpus agreement and
batch-partition invariance are mandatory. CPU and accelerator equivalence use a
declared tolerance rather than presumed bit identity.

### Staged decision

| Stage | Work | Exit criterion |
| --- | --- | --- |
| A | Implement and attest the narrow immutable TSPulse provider | **Implemented:** real 512-point univariate corpus/query encoding completed through the DuckPD path; model quality remains unqualified |
| B | Fix channel schema, relevance labels, splits, native/PCA/statistical/DTW controls | Fixtures distinguish desired invariance from meaningful change |
| C | Train joint TS2Vec externally and export a frozen artifact | **Implemented on synthetic data:** reproducible averaged checkpoint with complete preprocessing, pooling, and training provenance; real-data training remains |
| D | Compare TSPulse and TS2Vec under matched eligible data and output budgets | Determine whether either learned representation beats strong conventional controls |
| E | Adapt TSPulse only if the narrow control and TS2Vec results justify training | Attribute any gain to adaptation rather than hidden preprocessing or leakage |

Choose material-improvement and allowable direction/lag-regression thresholds
from baseline variance and product value before final evaluation. If neither
learned candidate passes, retain native representations and the custom-provider
protocol without recommending a built-in model.

## Implementation status and sequence

Completed:

1. `backend="tspulse"` resolves through the existing provider lifecycle.
2. Preparation verifies the immutable TSPulse config and weights before loading.
3. CPU execution, internal RevIN, an all-observed mask, decoder/register
   extraction, and raw 240-dimensional output are pinned.
4. Corpus and query paths execute through bounded Arrow batches with the real
   `granite-tsfm==0.3.9` runtime.
5. A pinned real-runtime golden fixture and repeated-process smoke runs produce
   the same 240-dimensional float32 output.
6. The bounded TS2Vec producer exports deterministic averaged safetensors from
   the generated market dataset with chronological embargoed splits,
   training-only standardization, and complete manifest provenance.
7. `backend="ts2vec"` verifies and runs that multivariate bundle through the
   same bounded corpus/query provider path.

Next:

1. Build native, PCA, statistical, and bounded-DTW evaluation controls.
2. Train the same attested TS2Vec recipe on eligible real domain data.
3. Run held-out retrieval, predictive, runtime, golden-output, and determinism
   suites.

The experiments require no new DataFrame operation, training API, or
variable-length provider protocol.

Provider `prepare()` must verify the complete manifest and bytes itself. The
session validates agreement between the returned attestation and requested
specification; it cannot independently prove arbitrary custom-provider files
were rehashed. Training cutoff, artifact publication/preparation time, and each
window's availability time are distinct provenance values.

Retraining, changing a readout, adding a predictive objective, or changing
fitted preprocessing creates a new representation fingerprint and requires
corpus re-encoding. Same dimensionality does not make successive checkpoints
compatible.

Automatic interpolation, imputation, training inside execution, text alignment,
automatic download during planning, and ANN-specific behavior remain separate
contracts.

## Verification record and remaining checks

### Independently confirmed

- [x] TSPulse search revision resolves to
  `b12164578f7b893ada0028c00d292ba10383d25a`.
- [x] Downloaded `model.safetensors` is 4,305,624 bytes with SHA-256
  `b9332ae796ec7c313f991ed32dbec62c29a8e673281decb7308f955bdda7aae0`.
- [x] The safetensors header contains 224 stored tensors totaling 1,068,958
  float32 elements; Hugging Face separately reports 1,084,330 parameters.
- [x] `granite-tsfm` tag `v0.3.9` resolves to
  `fe7a35697723e2a2f5246ae979474bfc554e26c0`.
- [x] Package metadata reports `granite-tsfm==0.3.9` and Python
  `>=3.11,<3.14`; wheel digest matches the proposal.
- [x] Decoder/register extraction is 240 values per channel for the pinned
  TSPulse checkpoint.
- [x] TSPulse uses univariate pretraining; decoder channel mixing requires
  adaptation.
- [x] Cookbook preprocessing fits an outer standard scaler on the index/training
  data and reuses it for queries.
- [x] TSPulse Tables 28–29 aggregate values were transcribed correctly.
- [x] TS2Vec source revision
  `b0088e14a99706c05451316dc6db8d3da9351163` is MIT-licensed, mixes input
  channels before temporal convolutions, and exposes full-series max pooling.
- [x] DuckPD's current representation fingerprint does not include search
  distance; distance is an explicit search argument.

### Still required before experiments or promotion

- [ ] Reproduce the official TSPulse retrieval benchmark, including exact
  comparator checkpoints and output normalization.
- [ ] Prove evaluation mode disables stochastic masking/dropout and determine
  whether a missing mask and all-observed mask are identical.
- [ ] Record all applicable source, model, and redistribution notice terms.
- [ ] Pin the TS2Vec training and inference environments.
- [ ] Define one immutable ordered financial channel schema.
- [ ] Verify query-alone, query-in-batch, and changed-batch-partition output
  agreement.
- [ ] Resolve Python 3.14 before a built-in TSPulse support promise; isolated
  research may use the recorded Python 3.11–3.13 matrix.
- [ ] Export and independently hash the trained TS2Vec bundle.
- [ ] Run all three evaluation suites; no source review substitutes for measured
  financial results.

## Open questions

1. Does joint TS2Vec beat native, PCA, statistical, and bounded-DTW baselines
   when trained on the same eligible windows?
2. Does the narrow TSPulse control add useful retrieval geometry before any
   multivariate adaptation?
3. Does a later TSPulse adaptation retain a pretrained advantage over TS2Vec,
   or only add complexity?
4. Does raw L2, cosine on raw vectors, or explicit unit normalization best match
   each declared relevance rubric?
5. Does RevIN erase economically useful amplitude, volatility, liquidity, or
   volume state?
6. Which TS2Vec pooling preserves event timing at a practical storage budget?
7. Do learned vectors or historical-neighbor summaries add predictive value
   beyond conventional features and native/PCA windows?
8. What material gain justifies PyTorch, training, artifact lifecycle,
   re-encoding, cold start, and long-term support?

## Sources

### DuckPD contracts

- [DuckPD time-series representation and similarity API][api-series]
- [DuckPD time-series embedding design][design-series]
- [DuckPD implementation roadmap][roadmap-phase-19]

### TSPulse

- [TSPulse model card][tspulse-card]
- [Immutable search-revision artifact metadata][tspulse-artifact]
- [Pinned search checkpoint configuration][tspulse-config]
- [TSPulse paper, arXiv v3][tspulse-paper]
- [Pinned `get_embeddings()` source][tspulse-helper]
- [Official cookbook search notebook][tspulse-cookbook]
- [Official core-repository search notebook][tspulse-search-notebook]
- [`granite-tsfm` PyPI metadata][granite-pypi]
- [`granite-tsfm` source repository][granite-source]
- [`granite-tsfm` Apache-2.0 license][granite-license]

### Joint candidates and multivariate challengers

- [Pinned TS2Vec encoder source][ts2vec-encoder]
- [Pinned TS2Vec full-series pooling source][ts2vec-pooling]
- [Pinned TS2Vec MIT license][ts2vec-license]
- [Chronos-2-small immutable artifact metadata][chronos-small-artifact]
- [Pinned Chronos-2 multivariate embedding pipeline][chronos-pipeline]
- [MantisV2 immutable artifact metadata][mantis-artifact]
- [Pinned MantisV2 architecture][mantis-source]
- [Pinned Mantis channel combiner][mantis-adapter]
- [Pinned Mantis package metadata][mantis-package]
- [Joint multivariate transformer reference implementation][joint-transformer]
- [SoftCLT paper][softclt-paper]

### Controls and screened candidates

- [MOMENT-1-small model card][moment-card]
- [MOMENT-1-small immutable artifact metadata][moment-artifact]
- [Pinned MOMENT config][moment-config]
- [MOMENT embedding implementation][moment-source]
- [`momentfm` PyPI metadata][moment-pypi]
- [MOMENT representation-learning tutorial][moment-tutorial]
- [VQShape repository and usage][vqshape]
- [VQShape checkpoint release][vqshape-release]
- [TimeSiam repository and usage][timesiam]
- [T-Rep repository and usage][trep]
- [TOTEM repository][totem]
- [UniTS repository][units]
- [UniTS checkpoint release][units-release]
- [TRACE repository][trace]

[api-series]: ../api/time-series-embeddings.md
[design-series]: ../design/time-series-embeddings.md
[roadmap-phase-19]: ../roadmap.md#phase-19--priority-4-optional-learned-series-encoders
[tspulse-card]: https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1
[tspulse-artifact]: https://huggingface.co/api/models/ibm-granite/granite-timeseries-tspulse-r1/revision/tspulse-hybrid-dualhead-512-p8-r1?blobs=true
[tspulse-config]: https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1/resolve/tspulse-hybrid-dualhead-512-p8-r1/config.json
[tspulse-paper]: https://arxiv.org/html/2505.13033
[tspulse-helper]: https://github.com/ibm-granite/granite-tsfm/blob/v0.3.9/tsfm_public/models/tspulse/utils/helpers.py
[tspulse-cookbook]: https://github.com/ibm-granite-community/granite-timeseries-cookbook/blob/main/recipes/Search/Getting_Started_with_TSPulse_Search.ipynb
[tspulse-search-notebook]: https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/tspulse_search_simple_example.ipynb
[granite-pypi]: https://pypi.org/project/granite-tsfm/
[granite-source]: https://github.com/ibm-granite/granite-tsfm/tree/v0.3.9
[granite-license]: https://github.com/ibm-granite/granite-tsfm/blob/v0.3.9/LICENSE
[moment-card]: https://huggingface.co/AutonLab/MOMENT-1-small
[moment-artifact]: https://huggingface.co/api/models/AutonLab/MOMENT-1-small?blobs=true
[moment-config]: https://huggingface.co/AutonLab/MOMENT-1-small/resolve/411e288267f82cce86296dbe4d6c8bc533cc162f/config.json
[moment-source]: https://github.com/moment-timeseries-foundation-model/moment/blob/main/momentfm/models/moment.py
[moment-pypi]: https://pypi.org/project/momentfm/
[moment-tutorial]: https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/representation_learning.ipynb
[vqshape]: https://github.com/YunshiWen/VQShape
[vqshape-release]: https://github.com/YunshiWen/VQShape/releases/tag/v0.1.0-cls
[timesiam]: https://github.com/thuml/TimeSiam
[trep]: https://github.com/Let-it-Care/T-Rep
[totem]: https://github.com/SaberaTalukder/TOTEM
[units]: https://github.com/mims-harvard/UniTS
[units-release]: https://github.com/mims-harvard/UniTS/releases/tag/ckpt
[trace]: https://github.com/Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder
[ts2vec-encoder]: https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/models/encoder.py
[ts2vec-pooling]: https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/ts2vec.py
[ts2vec-license]: https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/LICENSE
[chronos-small-artifact]: https://huggingface.co/api/models/autogluon/chronos-2-small/revision/ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a?blobs=true
[chronos-pipeline]: https://github.com/amazon-science/chronos-forecasting/blob/4dbf163c2734c089cdf7da2b86fde48862ff9c6f/src/chronos/chronos2/pipeline.py
[mantis-artifact]: https://huggingface.co/api/models/paris-noah/MantisV2/revision/8f6ca35cb54ab14b120618943c6fca5ddf5a76a6?blobs=true
[mantis-source]: https://github.com/vfeofanov/mantis/blob/9018b98b4c1e093d2fa618338695cd57146d3cd0/src/mantis/architecture/version2.py
[mantis-adapter]: https://github.com/vfeofanov/mantis/blob/9018b98b4c1e093d2fa618338695cd57146d3cd0/src/mantis/adapters/diff_adapter.py
[mantis-package]: https://github.com/vfeofanov/mantis/blob/9018b98b4c1e093d2fa618338695cd57146d3cd0/pyproject.toml
[joint-transformer]: https://github.com/gzerveas/mvts_transformer
[softclt-paper]: https://arxiv.org/html/2312.16424v4
