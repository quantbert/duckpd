# Draft proposal: learned time-series embedding candidates

**Status:** research draft for independent verification; no adapter is approved by this document.

**Research date:** 2026-09-11

**Related contracts:** [time-series embedding API][api-series],
[time-series embedding design][design-series], and
[Phase 19 roadmap][roadmap-phase-19].

## Executive decision

DuckPD should qualify **IBM TSPulse** before spending more implementation effort
on forecasting-oriented foundation models. TSPulse is the strongest candidate
found in this review because it combines:

- a checkpoint specialized for semantic time-series search;
- a public embedding extraction helper;
- a fixed, compact register representation;
- an immutable Hugging Face revision and checksummed weights;
- Apache-2.0 source and weights;
- explicit CPU deployment support; and
- published zero-shot retrieval results against MOMENT and Chronos.

The proposed evaluation order is:

1. **TSPulse** search checkpoint as the primary learned candidate.
2. **MOMENT-1-small** as a learned control.
3. **VQShape** as an experimental shape-token challenger.
4. DuckPD's shipped native representations as mandatory baselines.

This is a proposal to run a qualification benchmark, not a proposal to promote
TSPulse immediately. No reviewed evidence yet shows that TSPulse improves
DuckPD's named financial, market-pattern, or event-reaction retrieval tasks.
The external TSPulse benchmark establishes relevance and priority, not product
qualification.

## Review outcome at a glance

| Candidate | Search-oriented checkpoint | Reproducible weights | Stable fixed-vector path | Permissive license | Current DuckPD fit | Proposed disposition |
| --- | --- | --- | --- | --- | --- | --- |
| TSPulse search revision | Yes | Strong: immutable revision and LFS SHA-256 | Strong for one channel: 240-dimensional register vector | Apache-2.0 | Good on Python 3.11–3.13; Python 3.14 unavailable; multichannel retrieval semantics unqualified | Implement first as a custom-provider experiment |
| MOMENT-1-small | General representation mode | Strong: immutable revision and LFS SHA-256 | Strong: default 512-dimensional pooled vector | MIT | Package dependency pins are stale; default pooling erases channel identity | Use as learned benchmark control |
| VQShape 256/512 | Shape tokenization, not retrieval-specific | Weak until DuckPD attests the mutable GitHub release asset | Potentially usable token or histogram vector, but exact pooling must be pinned | MIT | Requires explicit 512-length policy; official path treats channels independently | Experimental third candidate |
| TRACE | Yes, cross-modal retrieval | Research checkpoint exists | Different aligned text/series contract | No declared repository license found | Does not match the current single-modality provider contract | Reject for current adapter; revisit for aligned retrieval research |
| TS2Vec | Representation learning | No canonical universal checkpoint | Full-series vector API exists | BSD-style repository license should be rechecked at a pinned revision | Requires application-specific training and old dependencies | Custom trained provider only |
| TimeSiam | Representation learning | No standalone general checkpoint found | Requires choosing lineage and pooling behavior | Repository license must be verified | Training and fine-tuning workflow rather than portable inference artifact | Defer |
| T-Rep | Representation learning | No canonical universal checkpoint | Encoder API exists after training | License must be verified | Explicitly incompatible with PyTorch 2.0 and expects dataset training | Reject for built-in use |
| TOTEM | Tokenized representations | Model zoo is not distributed as an immutable, checksummed package | Requires adapter-defined vectorization | No declared repository license found | Artifact and licensing gates fail | Reject for built-in use |
| UniTS | General time-series model | Task-specific release assets, no published digests | No stable semantic retrieval vector API | MIT source | Released files are task-specific and release is mutable | Defer |
| Forecasting models | Usually yes | Varies | Usually exposes patch/variate states, not one declared vector | Varies | Requires DuckPD-owned extraction and pooling with no direct retrieval evidence | Lower priority than TSPulse |

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

## Candidate 1: IBM TSPulse

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
It is nevertheless the best-supported candidate to test first.

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
| Safetensors parameter count | 1,084,330 float32 parameters |
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

The first DuckPD experiment should therefore be univariate and target-only.
Multichannel handling must remain unsupported until a separate pooling or
fine-tuning contract is benchmarked.

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

### Proposed narrow adapter contract

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
| Distance | Raw L2 first, matching published evaluation |
| Unit normalization | Disabled for exact-recipe comparison; benchmark cosine/unit-normalized as a separate representation |
| Outer scaling | Unresolved; benchmark cookbook scaler and internal-only paths separately |
| Internal scaling | Affine RevIN, minimum scale 0.001 |
| Provider | Existing custom series provider until qualification passes |
| Device | CPU first; accelerator is a separate provider/runtime record |

The representation fingerprint must distinguish every variation above. In
particular, raw L2 and unit-normalized cosine results must not share a
fingerprint, and fitted outer-scaler state must be part of artifact identity.

### TSPulse promotion gates

TSPulse should become a built-in optional adapter only if all of these pass:

1. Immutable preparation verifies every downloaded byte before loading.
2. A canonical DuckPD artifact manifest is computed and persisted.
3. The exact adapter produces 240 finite values for every accepted input.
4. Batch partitioning does not change output beyond a declared tolerance.
5. Repeated CPU runs and process restarts satisfy the determinism tolerance.
6. The model improves at least one named retrieval task over its best native
   baseline by a predeclared material threshold.
7. The gain survives entity-held-out and chronology-held-out evaluation.
8. The result survives realistic perturbations without destroying sensitivity
   to economically meaningful level, direction, or timing distinctions.
9. Cold start, peak RSS, throughput, and storage remain within declared budgets.
10. Python 3.14 is supported upstream or DuckPD explicitly documents and tests a
    narrower optional-adapter matrix.

## Candidate 2: MOMENT-1-small

### Role in the proposal

MOMENT is the best learned control because it exposes an explicit embedding
mode and is already compared directly with TSPulse in the TSPulse retrieval
paper. It should not be the first integration because it is much larger, loses
that external retrieval comparison, and has a problematic published package
surface.

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

## Candidate 3: VQShape

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

Retain only as a third, experimental candidate. Before running it, pin the
source commit, hash the chosen release archive and all contained checkpoints,
inspect the exact token/histogram tensor shapes, and declare one deterministic
row-level vector. Do not implement hidden interpolation.

## Screened-out candidates

### TS2Vec

TS2Vec has a clean full-series encoding concept: after training,
`encode(..., encoding_window="full_series")` returns one vector per series.
However, the official workflow trains a new model on each dataset, writes its
checkpoint into a run directory, and recommends Python 3.8 with PyTorch 1.8.1,
NumPy 1.19.2, and pandas 1.0.1. No canonical universal pretrained checkpoint
was found in the reviewed repository. TS2Vec remains suitable for an
application-owned trained custom provider, not a DuckPD-owned built-in artifact.

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

UniTS has MIT-licensed source and downloadable release assets, but the reviewed
`ckpt` release is mutable, publishes no asset digests, and consists of
fine-tuned task-specific `.pth` files. No stable zero-shot semantic embedding
interface was found. Revisit only if an immutable general checkpoint and fixed
retrieval representation are published.

### TRACE

TRACE is directly interesting for aligned text/time-series retrieval. That is a
different product contract from DuckPD's current independent text and series
spaces. The reviewed repository also reports no detected license. Until license,
artifact provenance, supported domains, and aligned-space metadata are resolved,
TRACE should remain research for a future direct text-to-series feature rather
than a current series adapter.

### Forecasting-first models

Chronos, TimesFM, TinyTimeMixer, PatchTST, and related forecasting models remain
architectural references. Their hidden states may be useful, but most require
DuckPD to invent a patch/variate pooling rule, and forecasting accuracy does not
establish historical retrieval quality. Chronos can remain a comparison if
benchmark capacity permits. TimesFM 3's noncommercial weights fail the intended
production-deployment gate. These candidates should not displace a direct
retrieval model from the first qualification run.

## Proposed DuckPD qualification benchmark

### Research question

For which declared retrieval tasks, if any, does a pinned learned encoder
produce materially better held-out neighbors than DuckPD's native
representations after accounting for latency, memory, storage, operational
complexity, and stability?

The benchmark must not collapse this into one global winner. A learned model may
win shift-tolerant motif retrieval while native centered returns remain better
for exact event-reaction shape or direction-sensitive retrieval.

### Representations

At minimum compare:

1. raw fixed return window;
2. centered return window;
3. z-score-normalized return window;
4. unit-normalized native window;
5. TSPulse decoder/register embedding with internal RevIN only;
6. TSPulse decoder/register embedding with the exact safely fitted cookbook
   outer scaler;
7. MOMENT-1-small default univariate mean embedding; and
8. VQShape's selected token or histogram vector if its contract is resolved.

Every preprocessing or output-normalization variation is a separate
representation space with a separate fingerprint. Do not tune the query
representation independently from the indexed representation.

### Retrieval tasks

Build labeled suites for distinct product meanings:

- trend continuation and trend reversal;
- impulse, overshoot, and recovery;
- volatility burst and decay;
- level shift versus transient spike;
- flat or low-information periods;
- periodic and repeated motifs;
- event-linked immediate reaction;
- event-linked delayed reaction; and
- same event family with materially different reaction direction.

The labels must encode what counts as relevant rather than deriving relevance
from whichever representation is under test.

### Splits and leakage controls

- Hold out entities so near-duplicate windows from one instrument do not appear
  in both index and query sets.
- Hold out later chronology where the intended product is forward deployment.
- Fit any corpus scaler on the index/training split only.
- Exclude overlapping windows across query and index boundaries.
- Deduplicate event observations and vendor revisions.
- Prevent the same underlying synthetic seed from appearing in both sides under
  only a trivial augmentation.
- Report in-domain and out-of-domain results separately.
- Record whether Bitcoin or any other candidate pretraining source overlaps an
  evaluation domain; do not treat model-card training-data lists as complete
  contamination evidence.

### Perturbations

Test controlled changes independently and in combinations:

- positive scaling;
- additive offset where meaningful;
- Gaussian and heavy-tailed noise;
- small and large time shifts;
- local temporal warp;
- sign reversal;
- missing prefix, suffix, block, and isolated values in a separately declared
  missingness experiment; and
- benign resampling changes performed before fixed-window construction.

Invariance is not universally desirable. For example, sign reversal may
preserve shape family but invert economic meaning. Report coarse family and
fine-grained direction-sensitive results separately.

### Quality metrics

For each task and representation, report:

- Precision@k and Recall@k;
- MRR@k;
- AP@k;
- nDCG@k;
- neighborhood overlap and rank correlation across perturbations;
- class-conditional and regime-conditional results;
- failure slices, not only macro averages; and
- confidence intervals across seeds or resampled query sets.

Use at least `k in {1, 3, 10}` where candidate-set size supports it. Predeclare
the primary metric for each task before tuning preprocessing.

### Runtime metrics

Measure the complete DuckPD path:

- model preparation download and verification time;
- cold import and first-inference time;
- warm batches per second and rows per second;
- Arrow-to-tensor and tensor-to-Arrow conversion time;
- model compute time separately;
- peak process RSS;
- accelerator memory where applicable;
- output bytes per row;
- exact index-build time;
- exact query latency at realistic corpus sizes; and
- end-to-end materialization or sink time.

Use batch sizes that include 1 and the expected production range. Test partial
final batches. Keep CPU qualification independent from accelerator results.

### Determinism and numerical checks

- Repeat the same batch in one process and across fresh processes.
- Change batch partitioning while preserving row order.
- Change thread counts.
- Compare supported CPU architectures and operating systems.
- Compare CPU and accelerator outputs only under an explicitly declared
  tolerance; do not assume bit identity.
- Verify every accepted output is finite and exactly dimensioned.
- Confirm constant, near-constant, maximum-magnitude, and minimum-scale windows
  do not produce unstable values.
- Confirm input arrays are not mutated by the provider.

### Proposed promotion threshold

Before running the benchmark, select a primary task and declare a material gain
threshold. A defensible initial rule is:

- statistically supported improvement on the primary held-out nDCG@10 or
  Recall@10 metric;
- no material regression on the direction-sensitive safety slice;
- acceptable stability under declared benign perturbations;
- bounded memory and cold-start behavior; and
- no unresolved artifact, license, runtime, or Python-matrix gate.

The exact numerical threshold should be chosen from application value and
native-baseline variance, not invented after results are known. If no learned
candidate clears it, retain the custom-provider protocol and promote nothing.

## Proposed implementation sequence after verification

1. Reverify every TSPulse identifier and license against immutable sources.
2. Build a throwaway provider outside the public built-in registry.
3. Prepare and attest the model under DuckPD's existing artifact lifecycle.
4. Prove one complete 512-point target channel maps to exactly 240 finite
   decoder/register values.
5. Reproduce a small official TSPulse search result before changing
   preprocessing.
6. Run the DuckPD benchmark with native representations, TSPulse, and MOMENT.
7. Resolve outer scaling through evidence rather than convenience.
8. Publish the full runtime and retrieval record, including negative slices.
9. Promote a built-in optional adapter only if the predeclared gate passes.
10. Update API, design, compatibility, roadmap, changelog, package extras, and
    examples together if promotion occurs.

A first implementation should not add multichannel pooling, interpolation,
padding, imputation, fine-tuning, forecasting, text alignment, automatic model
download, or ANN-specific behavior. Those are separate contracts.

## Independent verification checklist

The reviewing agent should explicitly verify or correct each item below:

### TSPulse artifact and license

- [ ] The named search branch still resolves to
  `b12164578f7b893ada0028c00d292ba10383d25a`.
- [ ] `model.safetensors` SHA-256 and size match this document.
- [ ] The source tag `v0.3.9` resolves to the stated commit.
- [ ] The PyPI wheel SHA-256 and Python requirement match this document.
- [ ] Apache-2.0 applies separately to source and weights with no hidden model
  terms or acceptable-use restriction.
- [ ] Any `NOTICE` obligations relevant to redistribution are captured.

### TSPulse semantics

- [ ] `get_embeddings()` at the pinned source revision returns `[B, C, D]`.
- [ ] Decoder/register output is exactly 240 values per channel for the pinned
  checkpoint.
- [ ] The register slice is the intended published search representation.
- [ ] Model evaluation disables stochastic masking and dropout.
- [ ] `mask_type="user"` plus no mask is equivalent to an all-observed mask, or
  the adapter pins the differing behavior.
- [ ] The exact official retrieval preprocessing is reconstructed, including
  the corpus-fitted outer standard scaler.
- [ ] The model's internal RevIN behavior, affine parameters, epsilon, and
  minimum-scale handling are captured completely.
- [ ] Univariate pretraining and the lack of qualified zero-shot channel-role
  interaction are correctly characterized.

### Published evidence

- [ ] Table 28 and Table 29 values were transcribed correctly.
- [ ] The evaluated MOMENT and Chronos variants are identified exactly.
- [ ] Euclidean distance and any output normalization are identified exactly.
- [ ] Query generation, index construction, and possible query/index ancestry
  are understood before treating scores as independent generalization evidence.
- [ ] Runtime table units, hardware, batch size, shape, and measured boundary are
  transcribed correctly.
- [ ] Official reproducibility code matches the paper's reported benchmark,
  rather than only demonstrating a similar workflow.

### MOMENT

- [ ] The model revision, safetensors digest, parameter count, and MIT terms are
  correct.
- [ ] The exact source used with the immutable checkpoint is selected.
- [ ] Default embedding reduction averages channels and then valid patches as
  described.
- [ ] RevIN, finite-value replacement, mask semantics, and output dtype are
  captured.
- [ ] Published `momentfm==0.1.4` dependency pins and the status of a compatible
  0.1.5 release are rechecked.

### VQShape and exclusions

- [ ] VQShape's exact token and histogram shapes and normalization are traced
  from the pinned source.
- [ ] The proposed release archive is downloaded and independently hashed.
- [ ] VQShape's source and checkpoint licenses are independently confirmed.
- [ ] TRACE and TOTEM truly lack applicable license grants rather than merely
  missing GitHub license detection.
- [ ] No newer immutable generic checkpoints have appeared for TS2Vec,
  TimeSiam, T-Rep, TOTEM, UniTS, or another retrieval-specific model.

### DuckPD integration

- [ ] The proposed one-channel adapter can be expressed without changing the
  current provider protocol.
- [ ] Python 3.14 policy is resolved before any built-in claim.
- [ ] Fingerprints cover scaler state, extraction point, distance semantics,
  output normalization, source revision, and complete artifact manifest.
- [ ] The benchmark labels and splits answer a product question and cannot be
  solved through overlap or event duplication.

## Open questions

1. Does external standard scaling change TSPulse neighbors after internal
   affine RevIN, and is any gain stable across chronology-held-out data?
2. Does raw L2 outperform unit-normalized cosine for DuckPD's target tasks?
3. Does TSPulse's desired shift invariance erase timing distinctions needed for
   event-reaction retrieval?
4. Does pretraining on Bitcoin create a meaningful advantage or a
   contamination concern for any proposed market benchmark?
5. Can TSPulse be made available on Python 3.14 without an unsupported fork?
6. Is a single return channel sufficient for the valuable retrieval tasks, or
   does the product need learned target/volume/volatility interaction?
7. If multichannel behavior is required, is fine-tuning TSPulse preferable to
   defining an untrained concatenation or mean-pooling convention?
8. Does VQShape's histogram representation provide useful interpretability at
   acceptable retrieval quality?
9. Which native representation is the actual strongest baseline for each named
   task?
10. What improvement is valuable enough to justify PyTorch, model preparation,
    cold start, and long-term artifact support?

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

### MOMENT

- [MOMENT-1-small model card][moment-card]
- [MOMENT-1-small immutable artifact metadata][moment-artifact]
- [Pinned MOMENT config][moment-config]
- [MOMENT embedding implementation][moment-source]
- [`momentfm` PyPI metadata][moment-pypi]
- [MOMENT representation-learning tutorial][moment-tutorial]

### VQShape and screened candidates

- [VQShape repository and usage][vqshape]
- [VQShape checkpoint release][vqshape-release]
- [TS2Vec repository and usage][ts2vec]
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
[ts2vec]: https://github.com/zhihanyue/ts2vec
[timesiam]: https://github.com/thuml/TimeSiam
[trep]: https://github.com/Let-it-Care/T-Rep
[totem]: https://github.com/SaberaTalukder/TOTEM
[units]: https://github.com/mims-harvard/UniTS
[units-release]: https://github.com/mims-harvard/UniTS/releases/tag/ckpt
[trace]: https://github.com/Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder
