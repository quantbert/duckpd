from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pyarrow as pa
import pytest

import duckpd
from duckpd.errors import UnsupportedOperationError


class _FakeTensor:
    def __init__(self, values: np.ndarray[Any, Any]) -> None:
        self.values = values

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def to(self, *_args: object, **_kwargs: object) -> _FakeTensor:
        return self

    def detach(self) -> _FakeTensor:
        return self

    def contiguous(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.values


class _AdapterTensor:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def to(self, *, device: str | None = None, dtype: str | None = None) -> _AdapterTensor:
        del device
        if dtype == "float32":
            return _AdapterTensor(self.values.astype(np.float32))
        if dtype == "long":
            return _AdapterTensor(self.values.astype(np.int64))
        return self

    def detach(self) -> _AdapterTensor:
        return self

    def contiguous(self) -> _AdapterTensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.values

    def transpose(self, left: int, right: int) -> _AdapterTensor:
        return _AdapterTensor(self.values.swapaxes(left, right))

    def mean(self, dim: int | tuple[int, ...]) -> _AdapterTensor:
        return _AdapterTensor(self.values.mean(axis=dim, dtype=np.float32))

    def reshape(self, *shape: int) -> _AdapterTensor:
        return _AdapterTensor(self.values.reshape(*shape))

    def squeeze(self, dimension: int) -> _AdapterTensor:
        return _AdapterTensor(cast("Any", np).squeeze(self.values, axis=dimension))

    def __getitem__(self, key: object) -> _AdapterTensor:
        return _AdapterTensor(cast("Any", self.values)[key])


class _AdapterTorch:
    float32 = "float32"
    long = "long"

    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()

    @staticmethod
    def from_numpy(values: object) -> _AdapterTensor:
        return _AdapterTensor(values)

    @staticmethod
    def stack(values: list[_AdapterTensor], *, dim: int) -> _AdapterTensor:
        return _AdapterTensor(cast("Any", np).stack([value.values for value in values], axis=dim))

    @staticmethod
    def zeros(shape: tuple[int, ...], *, dtype: str, device: str) -> _AdapterTensor:
        del device
        return _AdapterTensor(np.zeros(shape, dtype=np.int64 if dtype == "long" else np.float32))

    @staticmethod
    def ones_like(value: _AdapterTensor) -> _AdapterTensor:
        return _AdapterTensor(np.ones_like(value.values))

    @staticmethod
    def empty(shape: tuple[int, ...], *, dtype: str, device: str) -> _AdapterTensor:
        del device, dtype
        return _AdapterTensor(np.empty(shape, dtype=np.float32))

    @staticmethod
    def cat(values: tuple[_AdapterTensor, ...], *, dim: int) -> _AdapterTensor:
        return _AdapterTensor(
            cast("Any", np).concatenate([value.values for value in values], axis=dim)
        )

    @staticmethod
    def full(
        shape: tuple[int, ...],
        value: int,
        *,
        dtype: str,
        device: str,
    ) -> _AdapterTensor:
        del device
        return _AdapterTensor(np.full(shape, value, dtype=np.int64 if dtype == "long" else None))


class _FakeMomentPipeline:
    loads: ClassVar[list[tuple[str, bool, dict[str, str]]]] = []
    inputs: ClassVar[list[tuple[np.ndarray[Any, Any], str]]] = []

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        local_files_only: bool,
        model_kwargs: dict[str, str],
    ) -> _FakeMomentPipeline:
        cls.loads.append((model, local_files_only, model_kwargs))
        return cls()

    def init(self) -> None:
        return None

    def eval(self) -> _FakeMomentPipeline:
        return self

    def to(self, _device: str) -> _FakeMomentPipeline:
        return self

    def __call__(self, *, x_enc: _FakeTensor, reduction: str) -> object:
        self.inputs.append((x_enc.values.copy(), reduction))
        first = x_enc.values[:, 0, :].sum(axis=1)
        second = x_enc.values[:, 1, :].sum(axis=1)
        stack = cast("Any", np).stack
        values = stack((first, second, first + second), axis=1).astype(np.float32)
        output = ModuleType("moment_output")
        output.embeddings = _FakeTensor(values)  # type: ignore[attr-defined]
        return output


def _install_fake_moment_runtime(
    monkeypatch: pytest.MonkeyPatch,
    downloads: list[tuple[str, str, tuple[str, ...]]],
) -> None:
    torch = ModuleType("torch")
    torch.cuda = ModuleType("cuda")  # type: ignore[attr-defined]
    torch.cuda.is_available = lambda: True  # type: ignore[attr-defined]
    torch.version = ModuleType("version")  # type: ignore[attr-defined]
    torch.version.hip = "7.2"  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.from_numpy = lambda values: _FakeTensor(values)  # type: ignore[attr-defined]
    torch.inference_mode = nullcontext  # type: ignore[attr-defined]

    huggingface_hub = ModuleType("huggingface_hub")

    def snapshot_download(
        *,
        repo_id: str,
        revision: str,
        local_dir: str | Path,
        allow_patterns: tuple[str, ...],
    ) -> str:
        downloads.append((repo_id, revision, allow_patterns))
        target = Path(local_dir)
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}")
        (target / "model.safetensors").write_bytes(b"verified-moment")
        return str(target)

    huggingface_hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    momentfm = ModuleType("momentfm")
    momentfm.MOMENTPipeline = _FakeMomentPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    monkeypatch.setitem(sys.modules, "momentfm", momentfm)


def _moment_model() -> duckpd.EmbeddingModelSpec:
    return duckpd.embedding_model(
        "test/moment",
        revision="0123456789abcdef0123456789abcdef01234567",
        dimension=3,
        backend="moment",
        normalize=False,
        pooling="mean",
        input=duckpd.series_embedding_input(
            length=2,
            channels=("target", "covariate"),
            roles=("target", "past_covariate"),
            normalization="moment-revin-v1",
        ),
    )


def test_moment_provider_prepares_verified_cache_and_serves_series_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[tuple[str, str, tuple[str, ...]]] = []
    _install_fake_moment_runtime(monkeypatch, downloads)
    _FakeMomentPipeline.loads.clear()
    _FakeMomentPipeline.inputs.clear()
    model = _moment_model()
    provider = duckpd.MomentEmbeddingProvider(
        model,
        device="cuda",
        cache_dir=tmp_path / "explicit",
    )

    with pytest.raises(UnsupportedOperationError, match="not prepared"):
        provider.embed_windows(
            pa.record_batch(
                [pa.array([[1.0, 2.0]], type=pa.list_(pa.float32(), 2))],
                names=["target"],
            )
        )

    prepared = provider.prepare()
    batch = pa.record_batch(
        [
            pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32(), 2)),
            pa.array([[10.0, 20.0], [30.0, 40.0]], type=pa.list_(pa.float32(), 2)),
        ],
        names=["target", "covariate"],
    )
    assert provider.embed_windows(batch).to_pylist() == [
        pytest.approx([3.0, 30.0, 33.0]),
        pytest.approx([7.0, 70.0, 77.0]),
    ]
    assert prepared.backend == "moment"
    assert prepared.execution_providers == ("PyTorchROCm",)
    assert prepared.artifact_digest is not None
    assert provider.thread_safe is False
    assert _FakeMomentPipeline.inputs[0][0].shape == (2, 2, 2)
    assert _FakeMomentPipeline.inputs[0][1] == "mean"

    restored = duckpd.MomentEmbeddingProvider(model, cache_dir=tmp_path / "explicit")
    assert restored.prepare().artifact_digest == prepared.artifact_digest
    assert len(downloads) == 1
    assert all(local_only for _, local_only, _ in _FakeMomentPipeline.loads)

    representation = duckpd.series_representation(
        window=2,
        channels=("target", "covariate"),
        sampling="observations",
        data_contract="test/moment-input/v1",
        encoder=model,
    )
    session = duckpd.connect()
    automatic = session.prepare_embedding_model(model, cache_dir=tmp_path / "automatic")
    query = session.embed_series_query(
        {"target": [1.0, 2.0], "covariate": [10.0, 20.0]},
        representation=representation,
    )
    assert automatic.backend == "moment"
    assert automatic.execution_providers == ("PyTorchCPU",)
    assert query.values == pytest.approx((3.0, 30.0, 33.0))


def _transformers_model(
    *,
    length: int,
    channels: tuple[str, ...],
    roles: tuple[duckpd.SeriesChannelRole, ...],
    normalization: str,
    pooling: str,
    frequency: duckpd.SeriesFrequencyInputSpec | None = None,
    temporal: duckpd.SeriesTemporalInputSpec | None = None,
    static: tuple[duckpd.SeriesStaticInputSpec, ...] = (),
) -> duckpd.EmbeddingModelSpec:
    return duckpd.embedding_model(
        "research/transformers-series",
        revision="0123456789abcdef0123456789abcdef01234567",
        backend="transformers",
        dimension=4,
        normalize=False,
        pooling=pooling,
        input=duckpd.series_embedding_input(
            length=length,
            channels=channels,
            roles=roles,
            normalization=normalization,
            provider_abi="transformers-series-v1",
            frequency=frequency,
            temporal=temporal,
            static=static,
        ),
    )


def test_transformers_series_adapter_registry_and_static_validation() -> None:
    from duckpd.series_providers import _ADAPTERS

    assert {model_type: adapter.class_name for model_type, adapter in _ADAPTERS.items()} == {
        "patchtst": "PatchTSTModel",
        "patchtsmixer": "PatchTSMixerModel",
        "timesfm": "TimesFmModel",
        "timesfm2_5": "TimesFm2_5Model",
        "time_series_transformer": "TimeSeriesTransformerModel",
        "informer": "InformerModel",
        "autoformer": "AutoformerModel",
    }

    patch_model = _transformers_model(
        length=4,
        channels=("target", "known"),
        roles=("target", "known_future_covariate"),
        normalization="patchtst-config-scaling-v1",
        pooling="mean-channels-patches-v1",
    )
    patch_config = SimpleNamespace(
        context_length=4,
        num_input_channels=2,
        d_model=4,
        do_mask_input=False,
        mask_input=False,
    )
    _ADAPTERS["patchtst"].validate(patch_config, patch_model)
    patch_config.do_mask_input = True
    with pytest.raises(ValueError, match="masked-input"):
        _ADAPTERS["patchtst"].validate(patch_config, patch_model)

    timesfm_model = _transformers_model(
        length=4,
        channels=("target",),
        roles=("target",),
        normalization="timesfm-masked-mean-std-v1",
        pooling="mean-valid-patches-v1",
        frequency=duckpd.series_frequency_input(
            cadence=duckpd.series_cadence("hour"),
        ),
    )
    _ADAPTERS["timesfm"].validate(
        SimpleNamespace(context_length=8, patch_length=2, hidden_size=4),
        timesfm_model,
    )

    temporal = duckpd.series_temporal_input(
        cadence=duckpd.series_cadence("hour"),
        timezone="UTC",
        anchor="last",
        recipe="gluonts-calendar-v1",
        features=("hour_of_day",),
    )
    encoder_model = _transformers_model(
        length=4,
        channels=("target", "dynamic"),
        roles=("target", "past_covariate"),
        normalization="hf-time-series-scaler-v1",
        pooling="mean-encoder-time-v1",
        temporal=temporal,
        static=(
            duckpd.series_static_input("scale", kind="real"),
            duckpd.series_static_input("series_id", kind="categorical", cardinality=3),
        ),
    )
    encoder_config = SimpleNamespace(
        d_model=4,
        lags_sequence=[1, 2],
        context_length=2,
        input_size=1,
        num_dynamic_real_features=1,
        num_time_features=1,
        num_static_real_features=1,
        num_static_categorical_features=1,
        cardinality=[3],
    )
    for model_type in ("time_series_transformer", "informer", "autoformer"):
        _ADAPTERS[model_type].validate(encoder_config, encoder_model)


def test_transformers_series_provider_rejects_noncanonical_arrow_metadata() -> None:
    from duckpd.series_embeddings import (
        SeriesQuerySnapshot,
        make_series_provider_batch,
    )

    model = _transformers_model(
        length=2,
        channels=("target",),
        roles=("target",),
        normalization="patchtsmixer-config-scaling-v1",
        pooling="mean-channels-patches-v1",
    )
    representation = duckpd.series_representation(
        window=2,
        channels=("target",),
        sampling="observations",
        data_contract="test/transformers-series/v1",
        encoder=model,
    )
    batch = make_series_provider_batch(
        (SeriesQuerySnapshot((("target", (1.0, 2.0)),), None, None, ()),),
        representation,
    )
    provider = duckpd.TransformersSeriesEmbeddingProvider(model)
    provider._validate_batch(batch)

    metadata = dict(batch.schema.metadata or {})
    metadata[b"duckpd.unexpected"] = b"value"
    invalid = cast("Any", pa.RecordBatch).from_arrays(
        cast("Any", batch).columns,
        schema=cast("Any", batch.schema).with_metadata(metadata),
    )
    with pytest.raises(ValueError, match="canonical provider boundary"):
        provider._validate_batch(invalid)


def test_transformers_series_provider_prepares_one_attested_bare_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfig:
        model_type = "patchtst"
        context_length = 4
        num_input_channels = 2
        d_model = 4
        do_mask_input = False
        mask_input = False

        def __init__(self, name_or_path: str = "model") -> None:
            self._name_or_path = name_or_path

        def to_dict(self) -> dict[str, object]:
            return {
                "_name_or_path": self._name_or_path,
                "model_type": self.model_type,
                "context_length": self.context_length,
                "num_input_channels": self.num_input_channels,
                "d_model": self.d_model,
                "do_mask_input": self.do_mask_input,
            }

    class FakePatchTSTModel:
        def __init__(self) -> None:
            self.config = FakeConfig()
            self.eval_calls = 0
            self.devices: list[str] = []

        def eval(self) -> FakePatchTSTModel:
            self.eval_calls += 1
            return self

        def to(self, device: str) -> FakePatchTSTModel:
            self.devices.append(device)
            return self

        def __call__(self, *, past_values: _AdapterTensor) -> object:
            rows = past_values.shape[0]
            hidden = np.arange(rows * 16, dtype=np.float32).reshape(rows, 2, 2, 4)
            return SimpleNamespace(last_hidden_state=_AdapterTensor(hidden))

    downloads: list[tuple[str, str, tuple[str, ...]]] = []
    load_options: list[dict[str, object]] = []
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch.version = SimpleNamespace(hip=None)  # type: ignore[attr-defined]
    torch.float32 = _AdapterTorch.float32  # type: ignore[attr-defined]
    torch.from_numpy = _AdapterTorch.from_numpy  # type: ignore[attr-defined]
    torch.stack = _AdapterTorch.stack  # type: ignore[attr-defined]
    torch.inference_mode = nullcontext  # type: ignore[attr-defined]
    huggingface_hub = ModuleType("huggingface_hub")

    def snapshot_download(
        *,
        repo_id: str,
        revision: str,
        local_dir: str | Path,
        allow_patterns: tuple[str, ...],
    ) -> str:
        downloads.append((repo_id, revision, allow_patterns))
        target = Path(local_dir)
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}")
        (target / "model.safetensors").write_bytes(b"verified-transformer")
        return str(target)

    huggingface_hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.__version__ = "4.57.6"  # type: ignore[attr-defined]
    transformers.PatchTSTModel = FakePatchTSTModel  # type: ignore[attr-defined]

    class AutoConfig:
        @staticmethod
        def from_pretrained(root: str, **options: object) -> FakeConfig:
            load_options.append(options)
            return FakeConfig(root)

    class AutoModel:
        @staticmethod
        def from_pretrained(_root: str, **options: object) -> tuple[object, object]:
            load_options.append(options)
            return (
                FakePatchTSTModel(),
                {
                    "missing_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                    "unexpected_keys": [],
                },
            )

    transformers.AutoConfig = AutoConfig  # type: ignore[attr-defined]
    transformers.AutoModel = AutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    model = _transformers_model(
        length=4,
        channels=("target", "known"),
        roles=("target", "known_future_covariate"),
        normalization="patchtst-config-scaling-v1",
        pooling="mean-channels-patches-v1",
    )
    provider = duckpd.TransformersSeriesEmbeddingProvider(
        model,
        cache_dir=tmp_path,
    )
    prepared = provider.prepare()
    assert prepared.artifact_digest is not None
    assert prepared.execution_providers == ("PyTorchCPU",)
    assert downloads == [
        (
            model.model,
            model.revision,
            ("config.json", "*.safetensors", "pytorch_model*.bin"),
        )
    ]
    assert all(options["trust_remote_code"] is False for options in load_options)
    assert all(options["local_files_only"] is True for options in load_options)
    assert load_options[-1]["ignore_mismatched_sizes"] is False
    assert load_options[-1]["output_loading_info"] is True
    from duckpd.series_embeddings import SeriesQuerySnapshot, make_series_provider_batch

    representation = duckpd.series_representation(
        window=4,
        channels=("target", "known"),
        sampling="observations",
        data_contract="test/provider-lifecycle/v1",
        encoder=model,
    )
    batch = make_series_provider_batch(
        (
            SeriesQuerySnapshot(
                (
                    ("target", (1.0, 2.0, 3.0, 4.0)),
                    ("known", (10.0, 20.0, 30.0, 40.0)),
                ),
                None,
                None,
                (),
            ),
        ),
        representation,
    )
    assert provider.embed_windows(batch).type == pa.list_(pa.float32(), 4)

    restored = duckpd.TransformersSeriesEmbeddingProvider(
        model,
        cache_dir=tmp_path,
    )
    assert restored.prepare().artifact_digest == prepared.artifact_digest
    assert len(downloads) == 1


def test_transformers_series_adapters_pack_and_pool_exact_tensor_layouts() -> None:
    from duckpd.series_embeddings import (
        SeriesQuerySnapshot,
        make_series_provider_batch,
        snapshot_series_query,
    )
    from duckpd.series_providers import _ADAPTERS

    patch_model_spec = _transformers_model(
        length=4,
        channels=("target", "known"),
        roles=("target", "known_future_covariate"),
        normalization="patchtst-config-scaling-v1",
        pooling="mean-channels-patches-v1",
    )
    patch_representation = duckpd.series_representation(
        window=4,
        channels=("target", "known"),
        sampling="observations",
        data_contract="test/patch-layout/v1",
        encoder=patch_model_spec,
    )
    patch_batch = make_series_provider_batch(
        (
            SeriesQuerySnapshot(
                (
                    ("target", (1.0, 2.0, 3.0, 4.0)),
                    ("known", (10.0, 20.0, 30.0, 40.0)),
                ),
                None,
                None,
                (),
            ),
        ),
        patch_representation,
    )

    class PatchModel:
        config = SimpleNamespace(d_model=4)
        _duckpd_specification = patch_model_spec

        def __init__(self) -> None:
            self.past_values: _AdapterTensor | None = None
            self.hidden = np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4)

        def __call__(self, *, past_values: _AdapterTensor) -> object:
            self.past_values = past_values
            return SimpleNamespace(last_hidden_state=_AdapterTensor(self.hidden))

    patch_model = PatchModel()
    patch_output = _ADAPTERS["patchtst"].embed(
        patch_model,
        patch_batch,
        _AdapterTorch,
        "cpu",
    )
    assert patch_model.past_values is not None
    np.testing.assert_array_equal(
        patch_model.past_values.values,
        np.asarray(
            [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        patch_output.values,
        patch_model.hidden.mean(axis=(1, 2), dtype=np.float32),
    )

    timesfm_spec = _transformers_model(
        length=4,
        channels=("target",),
        roles=("target",),
        normalization="timesfm-masked-mean-std-v1",
        pooling="mean-valid-patches-v1",
        frequency=duckpd.series_frequency_input(
            cadence=duckpd.series_cadence("month", mode="civil"),
        ),
    )
    timesfm_representation = duckpd.series_representation(
        window=4,
        channels=("target",),
        sampling="observations",
        data_contract="test/timesfm-layout/v1",
        encoder=timesfm_spec,
    )
    timesfm_batch = make_series_provider_batch(
        (SeriesQuerySnapshot((("target", (1.0, 2.0, 3.0, 4.0)),), None, None, ()),),
        timesfm_representation,
    )

    class TimesFmModel:
        config = SimpleNamespace(patch_length=2)
        _duckpd_specification = timesfm_spec

        def __init__(self) -> None:
            self.arguments: dict[str, _AdapterTensor] = {}
            self.hidden = np.arange(8, dtype=np.float32).reshape(1, 2, 4)

        def __call__(self, **arguments: _AdapterTensor) -> object:
            self.arguments = arguments
            return SimpleNamespace(last_hidden_state=_AdapterTensor(self.hidden))

    timesfm_model = TimesFmModel()
    timesfm_output = _ADAPTERS["timesfm"].embed(
        timesfm_model,
        timesfm_batch,
        _AdapterTorch,
        "cpu",
    )
    np.testing.assert_array_equal(
        timesfm_model.arguments["past_values_padding"].values,
        np.zeros((1, 4), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        timesfm_model.arguments["freq"].values,
        np.asarray([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        timesfm_output.values,
        timesfm_model.hidden.mean(axis=1, dtype=np.float32),
    )

    timesfm25_spec = _transformers_model(
        length=4,
        channels=("target",),
        roles=("target",),
        normalization="timesfm2_5-config-normalization-v1",
        pooling="mean-valid-patches-v1",
    )
    timesfm25_representation = duckpd.series_representation(
        window=4,
        channels=("target",),
        sampling="observations",
        data_contract="test/timesfm25-layout/v1",
        encoder=timesfm25_spec,
    )
    timesfm25_batch = make_series_provider_batch(
        (SeriesQuerySnapshot((("target", (1.0, 2.0, 3.0, 4.0)),), None, None, ()),),
        timesfm25_representation,
    )
    timesfm25_model = TimesFmModel()
    timesfm25_model._duckpd_specification = timesfm25_spec
    timesfm25_output = _ADAPTERS["timesfm2_5"].embed(
        timesfm25_model,
        timesfm25_batch,
        _AdapterTorch,
        "cpu",
    )
    assert set(timesfm25_model.arguments) == {
        "past_values",
        "past_values_padding",
    }
    np.testing.assert_array_equal(
        timesfm25_output.values,
        timesfm25_model.hidden.mean(axis=1, dtype=np.float32),
    )

    temporal = duckpd.series_temporal_input(
        cadence=duckpd.series_cadence("hour"),
        timezone="UTC",
        anchor="last",
        recipe="gluonts-calendar-v1",
        features=("hour_of_day",),
    )
    encoder_spec = _transformers_model(
        length=4,
        channels=("target", "dynamic"),
        roles=("target", "past_covariate"),
        normalization="hf-time-series-scaler-v1",
        pooling="mean-encoder-time-v1",
        temporal=temporal,
        static=(
            duckpd.series_static_input("scale", kind="real"),
            duckpd.series_static_input("series_id", kind="categorical", cardinality=3),
        ),
    )
    encoder_representation = duckpd.series_representation(
        window=4,
        channels=("target", "dynamic"),
        sampling="observations",
        data_contract="test/encoder-layout/v1",
        encoder=encoder_spec,
    )
    encoder_query = duckpd.series_query(
        {
            "target": (1.0, 2.0, 3.0, 4.0),
            "dynamic": (10.0, 20.0, 30.0, 40.0),
        },
        time=datetime(2024, 1, 1, 3, tzinfo=UTC),
        static={"scale": 1.5, "series_id": 2},
    )
    encoder_batch = make_series_provider_batch(
        (snapshot_series_query(encoder_query, encoder_representation),),
        encoder_representation,
    )

    class Encoder:
        def __init__(self) -> None:
            self.inputs: _AdapterTensor | None = None
            self.hidden = np.arange(8, dtype=np.float32).reshape(1, 2, 4)

        def __call__(self, *, inputs_embeds: _AdapterTensor, return_dict: bool) -> object:
            assert return_dict is True
            self.inputs = inputs_embeds
            return SimpleNamespace(last_hidden_state=_AdapterTensor(self.hidden))

    class EncoderModel:
        config = SimpleNamespace(context_length=2)
        _duckpd_specification = encoder_spec

        def __init__(self, *, autoformer: bool) -> None:
            self.autoformer = autoformer
            self.arguments: dict[str, _AdapterTensor | None] = {}
            self.encoder = Encoder()

        def create_network_inputs(
            self,
            **arguments: _AdapterTensor | None,
        ) -> tuple[_AdapterTensor, ...]:
            self.arguments = arguments
            lagged = _AdapterTensor(np.ones((1, 4, 2), dtype=np.float32))
            if self.autoformer:
                temporal_features = _AdapterTensor(np.ones((1, 4, 3), dtype=np.float32))
                return lagged, temporal_features
            return (_AdapterTensor(np.ones((1, 4, 4), dtype=np.float32)),)

        def get_encoder(self) -> Encoder:
            return self.encoder

    for model_type in ("time_series_transformer", "informer", "autoformer"):
        encoder_model = EncoderModel(autoformer=model_type == "autoformer")
        output = _ADAPTERS[model_type].embed(
            encoder_model,
            encoder_batch,
            _AdapterTorch,
            "cpu",
        )
        past_values = encoder_model.arguments["past_values"]
        past_time_features = encoder_model.arguments["past_time_features"]
        static_real = encoder_model.arguments["static_real_features"]
        static_categorical = encoder_model.arguments["static_categorical_features"]
        assert isinstance(past_values, _AdapterTensor)
        assert isinstance(past_time_features, _AdapterTensor)
        assert isinstance(static_real, _AdapterTensor)
        assert isinstance(static_categorical, _AdapterTensor)
        assert past_values.shape == (1, 4)
        assert past_time_features.shape == (1, 4, 2)
        assert static_real.shape == (1, 1)
        assert static_categorical.shape == (1, 1)
        assert encoder_model.encoder.inputs is not None
        assert encoder_model.encoder.inputs.shape == (
            (1, 2, 5) if model_type == "autoformer" else (1, 2, 4)
        )
        np.testing.assert_array_equal(
            output.values,
            encoder_model.encoder.hidden.mean(axis=1, dtype=np.float32),
        )


def test_transformers_series_adapters_reject_incompatible_checkpoint_contracts() -> None:
    from duckpd.series_providers import _ADAPTERS

    def config_with(config: SimpleNamespace, **changes: object) -> SimpleNamespace:
        return SimpleNamespace(**{**vars(config), **changes})

    patch = _transformers_model(
        length=4,
        channels=("target",),
        roles=("target",),
        normalization="patchtst-config-scaling-v1",
        pooling="mean-channels-patches-v1",
    )
    patch_config = SimpleNamespace(
        context_length=4,
        num_input_channels=1,
        d_model=4,
        do_mask_input=False,
        mask_input=False,
    )
    with pytest.raises(ValueError, match="dimension"):
        _ADAPTERS["patchtst"].validate(config_with(patch_config, d_model=3), patch)
    patch_input = cast("duckpd.SeriesEmbeddingInputSpec", patch.input)
    rich_patch = replace(
        patch,
        input=replace(
            patch_input,
            temporal=duckpd.series_temporal_input(
                cadence=duckpd.series_cadence("hour"),
                timezone="UTC",
                anchor="last",
                recipe="gluonts-calendar-v1",
                features=("hour_of_day",),
            ),
        ),
    )
    with pytest.raises(ValueError, match="rejects frequency, temporal, and static"):
        _ADAPTERS["patchtst"].validate(patch_config, rich_patch)
    with pytest.raises(ValueError, match="must be an integer"):
        _ADAPTERS["patchtst"].validate(
            config_with(patch_config, context_length="4"),
            patch,
        )
    with pytest.raises(ValueError, match="masked-input"):
        _ADAPTERS["patchtst"].validate(
            config_with(patch_config, do_mask_input=True),
            patch,
        )

    timesfm_config = SimpleNamespace(context_length=8, patch_length=2, hidden_size=4)
    timesfm = _transformers_model(
        length=4,
        channels=("target",),
        roles=("target",),
        normalization="timesfm-masked-mean-std-v1",
        pooling="mean-valid-patches-v1",
        frequency=duckpd.series_frequency_input(cadence=duckpd.series_cadence("hour")),
    )
    timesfm_input = cast("duckpd.SeriesEmbeddingInputSpec", timesfm.input)
    with pytest.raises(ValueError, match="requires exactly one target"):
        _ADAPTERS["timesfm"].validate(
            timesfm_config,
            replace(
                timesfm,
                input=replace(
                    timesfm_input,
                    channels=("target", "covariate"),
                    roles=("target", "past_covariate"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="rejects static"):
        _ADAPTERS["timesfm"].validate(
            timesfm_config,
            replace(
                timesfm,
                input=replace(
                    timesfm_input,
                    frequency=None,
                    static=(duckpd.series_static_input("scale", kind="real"),),
                ),
            ),
        )
    with pytest.raises(ValueError, match="exceeds"):
        _ADAPTERS["timesfm"].validate(
            timesfm_config,
            replace(timesfm, input=replace(timesfm_input, length=12)),
        )
    with pytest.raises(ValueError, match="multiple"):
        _ADAPTERS["timesfm"].validate(
            timesfm_config,
            replace(timesfm, input=replace(timesfm_input, length=3)),
        )
    with pytest.raises(ValueError, match="requires frequency"):
        _ADAPTERS["timesfm"].validate(
            timesfm_config,
            replace(timesfm, input=replace(timesfm_input, frequency=None)),
        )

    timesfm25 = _transformers_model(
        length=4,
        channels=("target",),
        roles=("target",),
        normalization="timesfm2_5-config-normalization-v1",
        pooling="mean-valid-patches-v1",
    )
    timesfm25_input = cast("duckpd.SeriesEmbeddingInputSpec", timesfm25.input)
    with pytest.raises(ValueError, match="rejects frequency"):
        _ADAPTERS["timesfm2_5"].validate(
            timesfm_config,
            replace(
                timesfm25,
                input=replace(
                    timesfm25_input,
                    frequency=duckpd.series_frequency_input(cadence=duckpd.series_cadence("hour")),
                ),
            ),
        )

    encoder = _transformers_model(
        length=4,
        channels=("target",),
        roles=("target",),
        normalization="hf-time-series-scaler-v1",
        pooling="mean-encoder-time-v1",
    )
    encoder_config = SimpleNamespace(
        d_model=4,
        lags_sequence=[1, 2],
        context_length=2,
        input_size=1,
        num_dynamic_real_features=0,
        num_time_features=0,
        num_static_real_features=0,
        num_static_categorical_features=0,
        cardinality=[],
    )
    _ADAPTERS["informer"].validate(encoder_config, encoder)
    with pytest.raises(ValueError, match="non-empty integer"):
        _ADAPTERS["informer"].validate(
            config_with(encoder_config, lags_sequence=[]),
            encoder,
        )
    with pytest.raises(ValueError, match="non-empty integer"):
        _ADAPTERS["informer"].validate(
            config_with(encoder_config, lags_sequence=["1"]),
            encoder,
        )
    with pytest.raises(ValueError, match="temporal declaration"):
        _ADAPTERS["informer"].validate(
            config_with(encoder_config, num_time_features=1),
            encoder,
        )
    static_encoder = replace(
        encoder,
        input=replace(
            cast("duckpd.SeriesEmbeddingInputSpec", encoder.input),
            static=(
                duckpd.series_static_input(
                    "series_id",
                    kind="categorical",
                    cardinality=3,
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="static categorical cardinalities"):
        _ADAPTERS["informer"].validate(
            config_with(
                encoder_config,
                num_static_categorical_features=1,
                cardinality=[4],
            ),
            static_encoder,
        )
    for cardinality in ("3", [0]):
        with pytest.raises(ValueError, match="positive integers"):
            _ADAPTERS["informer"].validate(
                config_with(
                    encoder_config,
                    num_static_categorical_features=1,
                    cardinality=cardinality,
                ),
                static_encoder,
            )


def test_transformers_series_provider_validates_extended_reserved_fields() -> None:
    from duckpd.series_embeddings import (
        make_series_provider_batch,
        snapshot_series_query,
    )

    temporal = duckpd.series_temporal_input(
        cadence=duckpd.series_cadence("hour"),
        timezone="UTC",
        anchor="last",
        recipe="gluonts-calendar-v1",
        features=("hour_of_day",),
    )
    model = _transformers_model(
        length=2,
        channels=("target", "dynamic"),
        roles=("target", "past_covariate"),
        normalization="hf-time-series-scaler-v1",
        pooling="mean-encoder-time-v1",
        temporal=temporal,
        static=(
            duckpd.series_static_input("scale", kind="real"),
            duckpd.series_static_input("series_id", kind="categorical", cardinality=3),
        ),
    )
    representation = duckpd.series_representation(
        window=2,
        channels=("target", "dynamic"),
        sampling="observations",
        data_contract="test/extended-provider-boundary/v1",
        encoder=model,
    )
    query = duckpd.series_query(
        {"target": (1.0, 2.0), "dynamic": (3.0, 4.0)},
        time=datetime(2024, 1, 1, 1, tzinfo=UTC),
        static={"scale": 1.5, "series_id": 2},
    )
    batch = make_series_provider_batch(
        (snapshot_series_query(query, representation),),
        representation,
    )
    provider = duckpd.TransformersSeriesEmbeddingProvider(model)
    provider._validate_batch(batch)

    categorical_index = batch.schema.get_field_index("__duckpd_static_categorical")
    arrays = list(cast("Any", batch).columns)
    arrays[categorical_index] = pa.array([[3]], type=pa.list_(pa.int64(), 1))
    out_of_range = cast("Any", pa.RecordBatch).from_arrays(
        arrays,
        schema=batch.schema,
    )
    with pytest.raises(ValueError, match="outside its declared cardinality"):
        provider._validate_batch(out_of_range)

    wrong_role_field = pa.field(
        "target",
        pa.list_(pa.float32(), 2),
        nullable=False,
        metadata={b"duckpd.channel_role": b"past_covariate"},
    )
    wrong_role_schema = cast("Any", pa).schema(
        [wrong_role_field, *list(cast("Any", batch.schema))[1:]],
        metadata=batch.schema.metadata,
    )
    wrong_role = cast("Any", pa.RecordBatch).from_arrays(
        cast("Any", batch).columns,
        schema=wrong_role_schema,
    )
    with pytest.raises(ValueError, match="invalid Arrow schema"):
        provider._validate_batch(wrong_role)

    malformed_metadata = dict(batch.schema.metadata or {})
    malformed_metadata[b"duckpd.representation_fingerprint"] = b"not-a-fingerprint"
    malformed = cast("Any", pa.RecordBatch).from_arrays(
        cast("Any", batch).columns,
        schema=cast("Any", batch.schema).with_metadata(malformed_metadata),
    )
    with pytest.raises(ValueError, match="canonical representation fingerprint"):
        provider._validate_batch(malformed)


def test_transformers_series_provider_rejects_invalid_lifecycle_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_model = duckpd.embedding_model(
        "research/text",
        revision="0123456789abcdef0123456789abcdef01234567",
        backend="transformers",
        dimension=4,
    )
    with pytest.raises(ValueError, match="requires a Transformers series model"):
        duckpd.TransformersSeriesEmbeddingProvider(text_model)

    series_model = _transformers_model(
        length=2,
        channels=("target",),
        roles=("target",),
        normalization="patchtst-scaler-v1",
        pooling="mean-last-hidden-state-v1",
    )
    with pytest.raises(ValueError, match="device must"):
        duckpd.TransformersSeriesEmbeddingProvider(
            series_model,
            device=cast("Any", "tpu"),
        )

    provider = duckpd.TransformersSeriesEmbeddingProvider(
        series_model,
        cache_dir=tmp_path,
    )
    assert provider.specification == series_model
    assert provider.thread_safe is False
    with pytest.raises(UnsupportedOperationError, match="not prepared"):
        provider.embed_windows(pa.record_batch([], names=[]))

    import duckpd.series_providers as series_providers

    def missing_import(_name: str) -> ModuleType:
        raise ImportError

    monkeypatch.setattr(series_providers.importlib, "import_module", missing_import)
    with pytest.raises(UnsupportedOperationError, match="require PyTorch"):
        provider.prepare()


def test_transformers_series_provider_enforces_output_and_empty_batch_contracts() -> None:
    from duckpd.series_embeddings import (
        SeriesQuerySnapshot,
        make_series_provider_batch,
    )

    model = _transformers_model(
        length=2,
        channels=("target",),
        roles=("target",),
        normalization="patchtst-scaler-v1",
        pooling="mean-last-hidden-state-v1",
    )
    representation = duckpd.series_representation(
        window=2,
        channels=("target",),
        sampling="observations",
        data_contract="test/provider-output/v1",
        encoder=model,
    )
    snapshot = SeriesQuerySnapshot((("target", (1.0, 2.0)),), None, None, ())
    batch = make_series_provider_batch((snapshot,), representation)
    empty_batch = make_series_provider_batch((), representation)
    provider = duckpd.TransformersSeriesEmbeddingProvider(model)
    provider._torch = _AdapterTorch
    provider._model = object()

    class OutputAdapter:
        model_type = "test-output"

        def __init__(self, values: object) -> None:
            self.values = values

        def embed(
            self,
            _model: object,
            _batch: pa.RecordBatch,
            _torch: object,
            _device: str,
        ) -> _AdapterTensor:
            return _AdapterTensor(self.values)

    provider._adapter = cast("Any", OutputAdapter([[1.0, 2.0, 3.0, 4.0]]))
    assert provider.embed_windows(batch).to_pylist() == [[1.0, 2.0, 3.0, 4.0]]
    assert provider.embed_windows(empty_batch).to_pylist() == []

    provider._adapter = cast("Any", OutputAdapter([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError, match="pooled shape"):
        provider.embed_windows(batch)
    provider._adapter = cast("Any", OutputAdapter([[1.0, 2.0, np.nan, 4.0]]))
    with pytest.raises(ValueError, match="nonfinite embeddings"):
        provider.embed_windows(batch)

    wrong_names = cast("Any", pa.RecordBatch).from_arrays(
        cast("Any", batch).columns,
        names=["other"],
    )
    with pytest.raises(ValueError, match="do not match expected order"):
        provider._validate_batch(wrong_names)
    missing_metadata = cast("Any", pa.RecordBatch).from_arrays(
        cast("Any", batch).columns,
        schema=cast("Any", batch.schema).remove_metadata(),
    )
    with pytest.raises(ValueError, match="canonical representation fingerprint"):
        provider._validate_batch(missing_metadata)


def test_transformers_series_provider_rejects_unattested_local_models() -> None:
    model = _transformers_model(
        length=4,
        channels=("target",),
        roles=("target",),
        normalization="patchtst-config-scaling-v1",
        pooling="mean-channels-patches-v1",
    )
    provider = duckpd.TransformersSeriesEmbeddingProvider(model)

    class ExpectedModel:
        pass

    config = SimpleNamespace(
        model_type="patchtst",
        context_length=4,
        num_input_channels=1,
        d_model=4,
        do_mask_input=False,
        mask_input=False,
    )

    class AutoConfig:
        current: ClassVar[object] = config

        @classmethod
        def from_pretrained(cls, _root: str, **_options: object) -> object:
            return cls.current

    class AutoModel:
        current: ClassVar[object] = (
            ExpectedModel(),
            {
                "missing_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
                "unexpected_keys": ["head.weight"],
            },
        )

        @classmethod
        def from_pretrained(cls, _root: str, **_options: object) -> object:
            return cls.current

    transformers = SimpleNamespace(
        AutoConfig=AutoConfig,
        AutoModel=AutoModel,
        PatchTSTModel=ExpectedModel,
    )
    loaded = provider._load_local(transformers, Path("."))
    assert type(loaded[2]) is ExpectedModel

    AutoConfig.current = SimpleNamespace(model_type="unknown")
    with pytest.raises(UnsupportedOperationError, match="unsupported"):
        provider._load_local(transformers, Path("."))
    AutoConfig.current = config

    without_architecture = SimpleNamespace(
        AutoConfig=AutoConfig,
        AutoModel=AutoModel,
    )
    with pytest.raises(UnsupportedOperationError, match="requires Transformers architecture"):
        provider._load_local(without_architecture, Path("."))

    invalid_loading_results = cast(
        "tuple[tuple[object, str], ...]",
        (
            (ExpectedModel(), "did not return loading information"),
            ((ExpectedModel(),), "did not return loading information"),
            ((object(), {}), "expected bare class"),
            ((ExpectedModel(), {"missing_keys": ["encoder.weight"]}), "non-empty"),
            (
                (ExpectedModel(), {"unexpected_keys": ["encoder.weight"]}),
                "non-task-head",
            ),
        ),
    )
    for result, message in invalid_loading_results:
        AutoModel.current = result
        with pytest.raises(UnsupportedOperationError, match=message):
            provider._load_local(transformers, Path("."))


def test_transformers_series_adapters_reject_malformed_hidden_states() -> None:
    from duckpd.series_embeddings import (
        SeriesQuerySnapshot,
        make_series_provider_batch,
    )
    from duckpd.series_providers import _ADAPTERS

    specification = _transformers_model(
        length=4,
        channels=("target",),
        roles=("target",),
        normalization="patchtst-config-scaling-v1",
        pooling="mean-channels-patches-v1",
    )
    representation = duckpd.series_representation(
        window=4,
        channels=("target",),
        sampling="observations",
        data_contract="test/malformed-hidden-state/v1",
        encoder=specification,
    )
    batch = make_series_provider_batch(
        (SeriesQuerySnapshot((("target", (1.0, 2.0, 3.0, 4.0)),), None, None, ()),),
        representation,
    )

    class MalformedModel:
        config = SimpleNamespace(d_model=4)
        _duckpd_specification = specification

        def __init__(self, hidden: object | None) -> None:
            self.hidden = hidden

        def __call__(self, **_arguments: object) -> SimpleNamespace:
            return SimpleNamespace(last_hidden_state=self.hidden)

    adapter = _ADAPTERS["patchtst"]
    with pytest.raises(ValueError, match="no last_hidden_state"):
        adapter.embed(MalformedModel(None), batch, _AdapterTorch, "cpu")
    with pytest.raises(ValueError, match="hidden rank"):
        adapter.embed(
            MalformedModel(_AdapterTensor(np.zeros((1, 1, 4), dtype=np.float32))),
            batch,
            _AdapterTorch,
            "cpu",
        )
    with pytest.raises(ValueError, match="hidden shape"):
        adapter.embed(
            MalformedModel(_AdapterTensor(np.zeros((2, 1, 1, 4), dtype=np.float32))),
            batch,
            _AdapterTorch,
            "cpu",
        )

    valid_model = MalformedModel(_AdapterTensor(np.zeros((1, 1, 1, 4), dtype=np.float32)))
    missing = pa.record_batch(
        [pa.array([[1.0, 2.0, 3.0, 4.0]], type=pa.list_(pa.float32(), 4))],
        names=["other"],
    )
    with pytest.raises(ValueError, match="missing 'target'"):
        adapter.embed(valid_model, missing, _AdapterTorch, "cpu")

    malformed_columns = cast(
        "tuple[tuple[pa.Array[Any], str], ...]",
        (
            (
                pa.array([[1.0, 2.0, 3.0, 4.0]], type=pa.list_(pa.float32())),
                "FixedSizeListArray",
            ),
            (
                pa.array([[1.0, 2.0]], type=pa.list_(pa.float32(), 2)),
                "list width 4",
            ),
            (
                pa.array([None], type=pa.list_(pa.float32(), 4)),
                "null rows",
            ),
            (
                pa.array([[1.0, None, 3.0, 4.0]], type=pa.list_(pa.float32(), 4)),
                "null values",
            ),
        ),
    )
    for column, message in malformed_columns:
        malformed = pa.record_batch([column], names=["target"])
        with pytest.raises((TypeError, ValueError), match=message):
            adapter.embed(valid_model, malformed, _AdapterTorch, "cpu")
