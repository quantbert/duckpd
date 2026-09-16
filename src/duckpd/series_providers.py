"""Optional first-party providers for learned time-series embeddings."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

import pyarrow as pa

from duckpd._locking import exclusive_file_lock
from duckpd.embeddings import (
    EmbeddingModelSpec,
    PreparedModelInfo,
    SeriesEmbeddingInputSpec,
    _check_prepare_limits,
    _directory_digest,
    _validate_prepare_limits,
)
from duckpd.errors import UnsupportedOperationError


class MomentEmbeddingProvider:
    """Optional PyTorch provider backed by the MOMENT foundation model runtime."""

    def __init__(
        self,
        specification: EmbeddingModelSpec,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        cache_dir: str | Path | None = None,
        prepare_timeout_seconds: float | None = None,
        max_download_bytes: int | None = None,
    ) -> None:
        if specification.backend != "moment":
            raise ValueError("MomentEmbeddingProvider requires backend='moment'")
        if not isinstance(specification.input, SeriesEmbeddingInputSpec):
            raise ValueError("MomentEmbeddingProvider requires a series embedding input")
        if specification.pooling != "mean":
            raise ValueError("MomentEmbeddingProvider pooling must be 'mean'")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        _validate_prepare_limits(prepare_timeout_seconds, max_download_bytes)
        self._specification = specification
        self._device = device
        cache_root = Path(cache_dir or Path.home() / ".cache" / "duckpd" / "embeddings")
        self._cache_dir = cache_root / specification.fingerprint
        self._prepare_timeout_seconds = prepare_timeout_seconds
        self._max_download_bytes = max_download_bytes
        self._numpy: Any | None = None
        self._torch: Any | None = None
        self._pipeline: Any | None = None
        self._prepared: PreparedModelInfo | None = None

    @property
    def specification(self) -> EmbeddingModelSpec:
        return self._specification

    @property
    def thread_safe(self) -> bool:
        return False

    def prepare(self) -> PreparedModelInfo:
        if self._prepared is not None:
            return self._prepared
        try:
            numpy = cast("Any", importlib.import_module("numpy"))
            torch = cast("Any", importlib.import_module("torch"))
            huggingface_hub = cast("Any", importlib.import_module("huggingface_hub"))
            momentfm = cast("Any", importlib.import_module("momentfm"))
        except ImportError:
            raise UnsupportedOperationError(
                "MOMENT embeddings require NumPy, PyTorch, huggingface-hub, and momentfm"
            ) from None
        if self._device == "cuda" and not torch.cuda.is_available():
            raise UnsupportedOperationError(
                "MOMENT CUDA device requested, but PyTorch cannot access a CUDA or ROCm GPU"
            )

        manifest_name = f"duckpd-{self._specification.fingerprint}.json"
        manifest_path = self._cache_dir / manifest_name
        lock_path = self._cache_dir.with_name(f".{self._cache_dir.name}.lock")
        started = perf_counter()
        with exclusive_file_lock(lock_path):
            if self._cache_dir.exists():
                artifact_digest = self._verified_cache_digest(
                    manifest_path,
                    started=started,
                )
                pipeline = self._load_pipeline(momentfm.MOMENTPipeline, self._cache_dir)
                if _directory_digest(self._cache_dir) != artifact_digest:
                    raise UnsupportedOperationError(
                        "MOMENT backend mutated verified artifacts while loading"
                    )
            else:
                self._cache_dir.parent.mkdir(parents=True, exist_ok=True)
                staging = self._cache_dir.with_name(f".{self._cache_dir.name}.tmp-{uuid4().hex}")
                shutil.rmtree(staging, ignore_errors=True)
                try:
                    huggingface_hub.snapshot_download(
                        repo_id=self._specification.model,
                        revision=self._specification.revision,
                        local_dir=staging,
                        allow_patterns=(
                            "config.json",
                            "*.safetensors",
                            "pytorch_model*.bin",
                        ),
                    )
                    _check_prepare_limits(
                        staging,
                        started=started,
                        timeout_seconds=self._prepare_timeout_seconds,
                        max_bytes=self._max_download_bytes,
                    )
                    pipeline = self._load_pipeline(momentfm.MOMENTPipeline, staging)
                    artifact_digest = _directory_digest(staging)
                    (staging / manifest_name).write_text(
                        json.dumps(
                            {
                                "model_fingerprint": self._specification.fingerprint,
                                "model_specification": self._specification.to_dict(),
                                "artifact_digest": artifact_digest,
                            },
                            sort_keys=True,
                        )
                    )
                    os.rename(staging, self._cache_dir)
                except BaseException:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise

        runtime = (
            "PyTorchROCm"
            if self._device == "cuda" and torch.version.hip is not None
            else "PyTorchCUDA"
            if self._device == "cuda"
            else "PyTorchCPU"
        )
        self._numpy = numpy
        self._torch = torch
        self._pipeline = pipeline
        self._prepared = PreparedModelInfo(
            self._specification.fingerprint,
            "moment",
            str(self._cache_dir),
            artifact_digest,
            (runtime,),
        )
        return self._prepared

    def embed_windows(self, batch: pa.RecordBatch) -> pa.Array[Any]:
        if self._numpy is None or self._torch is None or self._pipeline is None:
            raise UnsupportedOperationError(
                "Embedding model is not prepared; call session.prepare_embedding_model(model)"
            )
        input_spec = cast("SeriesEmbeddingInputSpec", self._specification.input)
        if batch.schema.names != list(input_spec.channels):
            raise ValueError("Arrow channel order does not match the MOMENT input contract")
        channel_values = [
            self._channel_values(
                cast("pa.Array[Any]", batch.column(index)),
                input_spec.length,
                batch.num_rows,
            )
            for index in range(batch.num_columns)
        ]
        values = self._numpy.stack(channel_values, axis=1)
        inputs = self._torch.from_numpy(values).to(
            device=self._device,
            dtype=self._torch.float32,
        )
        with self._torch.inference_mode():
            embeddings = self._pipeline(
                x_enc=inputs,
                reduction=self._specification.pooling,
            ).embeddings
        if embeddings is None:
            raise ValueError("MOMENT returned no embeddings")
        output = (
            embeddings.detach().to(device="cpu", dtype=self._torch.float32).contiguous().numpy()
        )
        expected_shape = (batch.num_rows, self._specification.dimension)
        if output.shape != expected_shape:
            raise ValueError(f"MOMENT returned shape {output.shape}; expected {expected_shape}")
        child = pa.array(output.reshape(-1), type=pa.float32())
        maker = cast("Any", pa.FixedSizeListArray)
        return cast(
            "pa.Array[Any]",
            maker.from_arrays(child, self._specification.dimension),
        )

    def close(self) -> None:
        self._numpy = None
        self._torch = None
        self._pipeline = None
        self._prepared = None

    def _verified_cache_digest(self, manifest_path: Path, *, started: float) -> str:
        if not manifest_path.is_file():
            raise UnsupportedOperationError(
                "Embedding cache exists without a verified DuckPD manifest"
            )
        _check_prepare_limits(
            self._cache_dir,
            started=started,
            timeout_seconds=self._prepare_timeout_seconds,
            max_bytes=self._max_download_bytes,
        )
        artifact_digest = _directory_digest(self._cache_dir)
        try:
            raw_manifest = json.loads(manifest_path.read_text())
        except (OSError, TypeError, json.JSONDecodeError):
            raise UnsupportedOperationError(
                "Prepared embedding cache has an invalid verification manifest"
            ) from None
        if not isinstance(raw_manifest, dict):
            raise UnsupportedOperationError(
                "Prepared embedding cache has an invalid verification manifest"
            )
        recorded = cast("dict[str, object]", raw_manifest)
        if (
            recorded.get("model_fingerprint") != self._specification.fingerprint
            or recorded.get("artifact_digest") != artifact_digest
            or recorded.get("model_specification") != self._specification.to_dict()
        ):
            raise UnsupportedOperationError(
                "Prepared embedding artifacts changed since their verified cache promotion"
            )
        return artifact_digest

    def _load_pipeline(self, pipeline_factory: Any, cache_dir: Path) -> Any:
        pipeline = pipeline_factory.from_pretrained(
            str(cache_dir),
            local_files_only=True,
            model_kwargs={"task_name": "embedding"},
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Only reconstruction head is pre-trained.*",
            )
            pipeline.init()
        pipeline.eval()
        pipeline.to(self._device)
        return pipeline

    @staticmethod
    def _channel_values(
        column: pa.Array[Any],
        length: int,
        rows: int,
    ) -> Any:
        if not isinstance(column, pa.FixedSizeListArray):
            raise TypeError("MOMENT input channels must be FixedSizeListArray values")
        values = cast("Any", column)
        if values.type.list_size != length or values.type.value_type != pa.float32():
            raise ValueError(f"MOMENT input channels must have type float32[{length}]")
        if values.null_count:
            raise ValueError("MOMENT input channels must not contain nulls")
        start = values.offset * length
        child = values.values.slice(start, rows * length)
        if child.null_count:
            raise ValueError("MOMENT input channels must not contain nulls")
        flattened = child.to_numpy(zero_copy_only=False)
        return flattened.reshape(rows, length)


class _TransformersSeriesAdapter(Protocol):
    model_type: str
    class_name: str

    def validate(self, config: Any, specification: EmbeddingModelSpec) -> None: ...

    def embed(
        self,
        model: Any,
        batch: pa.RecordBatch,
        torch: Any,
        device: str,
    ) -> Any: ...


def _config_int(config: Any, name: str, *, default: int | None = None) -> int:
    value = getattr(config, name, default)
    if type(value) is not int:
        raise ValueError(f"Transformers series config.{name} must be an integer; found {value!r}")
    return value


def _require_equal(
    specification: EmbeddingModelSpec,
    adapter: str,
    field: str,
    actual: object,
    expected: object,
) -> None:
    if actual != expected:
        raise ValueError(
            f"{specification.model} adapter {adapter} requires {field}={expected!r}; "
            f"found {actual!r}"
        )


def _list_values(column: pa.Array[Any], width: int, rows: int, *, field: str) -> Any:
    if not isinstance(column, pa.FixedSizeListArray):
        raise TypeError(f"{field} must be a FixedSizeListArray")
    values = cast("Any", column)
    if values.type.list_size != width:
        raise ValueError(f"{field} must have list width {width}; found {values.type.list_size}")
    if values.null_count:
        raise ValueError(f"{field} must not contain null rows")
    start = values.offset * width
    child = values.values.slice(start, rows * width)
    if child.null_count:
        raise ValueError(f"{field} must not contain null values")
    return child.to_numpy(zero_copy_only=False, writable=True).reshape(rows, width)


def _field_tensor(
    batch: pa.RecordBatch,
    name: str,
    *,
    width: int,
    torch: Any,
    device: str,
    dtype: Any,
) -> Any:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"Transformers series batch is missing {name!r}")
    values = _list_values(
        cast("pa.Array[Any]", batch.column(index)),
        width,
        batch.num_rows,
        field=name,
    )
    return torch.from_numpy(values).to(device=device, dtype=dtype)


def _channel_tensors(
    batch: pa.RecordBatch,
    input_spec: SeriesEmbeddingInputSpec,
    *,
    torch: Any,
    device: str,
) -> list[Any]:
    return [
        _field_tensor(
            batch,
            channel,
            width=input_spec.length,
            torch=torch,
            device=device,
            dtype=torch.float32,
        )
        for channel in input_spec.channels
    ]


def _hidden_state(output: Any, *, model: str, adapter: str) -> Any:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        raise ValueError(f"{model} adapter {adapter} returned no last_hidden_state")
    return hidden


def _validate_hidden(
    hidden: Any,
    *,
    expected: tuple[int, ...],
    model: str,
    adapter: str,
) -> None:
    actual = tuple(hidden.shape)
    if actual != expected:
        raise ValueError(
            f"{model} adapter {adapter} returned hidden shape {actual}; expected {expected}"
        )


class _PatchBackboneAdapter:
    def __init__(self, model_type: str, class_name: str, normalization: str) -> None:
        self.model_type = model_type
        self.class_name = class_name
        self._normalization = normalization

    def validate(self, config: Any, specification: EmbeddingModelSpec) -> None:
        input_spec = cast("SeriesEmbeddingInputSpec", specification.input)
        _require_equal(
            specification,
            self.model_type,
            "input.normalization",
            input_spec.normalization,
            self._normalization,
        )
        _require_equal(
            specification,
            self.model_type,
            "input.length",
            input_spec.length,
            _config_int(config, "context_length"),
        )
        _require_equal(
            specification,
            self.model_type,
            "channel count",
            len(input_spec.channels),
            _config_int(config, "num_input_channels"),
        )
        _require_equal(
            specification,
            self.model_type,
            "dimension",
            specification.dimension,
            _config_int(config, "d_model"),
        )
        _require_equal(
            specification,
            self.model_type,
            "pooling",
            specification.pooling,
            "mean-channels-patches-v1",
        )
        if input_spec.temporal is not None or input_spec.static:
            raise ValueError(
                f"{specification.model} adapter {self.model_type} rejects temporal and static "
                "inputs"
            )
        if self.model_type == "patchtst" and (
            getattr(config, "do_mask_input", False) is True
            or getattr(config, "mask_input", False) is True
        ):
            raise ValueError(
                f"{specification.model} adapter patchtst rejects masked-input configurations"
            )

    def embed(self, model: Any, batch: pa.RecordBatch, torch: Any, device: str) -> Any:
        specification = cast("EmbeddingModelSpec", model._duckpd_specification)
        input_spec = cast("SeriesEmbeddingInputSpec", specification.input)
        channels = _channel_tensors(batch, input_spec, torch=torch, device=device)
        past_values = torch.stack(channels, dim=1).transpose(1, 2)
        output = model(past_values=past_values)
        hidden = _hidden_state(
            output,
            model=specification.model,
            adapter=self.model_type,
        )
        config = model.config
        if len(hidden.shape) != 4:
            raise ValueError(
                f"{specification.model} adapter {self.model_type} returned hidden rank "
                f"{len(hidden.shape)}; expected 4"
            )
        expected = (
            batch.num_rows,
            len(input_spec.channels),
            int(hidden.shape[2]),
            _config_int(config, "d_model"),
        )
        _validate_hidden(
            hidden,
            expected=expected,
            model=specification.model,
            adapter=self.model_type,
        )
        return hidden.to(dtype=torch.float32).mean(dim=(1, 2))


class _TimesFm25Adapter:
    model_type = "timesfm2_5"
    class_name = "TimesFm2_5Model"

    def validate(self, config: Any, specification: EmbeddingModelSpec) -> None:
        input_spec = cast("SeriesEmbeddingInputSpec", specification.input)
        if input_spec.roles != ("target",) or len(input_spec.channels) != 1:
            raise ValueError(
                f"{specification.model} adapter {self.model_type} requires exactly one target "
                "channel and rejects covariates"
            )
        if input_spec.temporal is not None or input_spec.static:
            raise ValueError(
                f"{specification.model} adapter {self.model_type} rejects temporal and static "
                "inputs"
            )
        if input_spec.length > _config_int(config, "context_length"):
            raise ValueError(
                f"{specification.model} adapter {self.model_type} input.length exceeds "
                "config.context_length"
            )
        patch_length = _config_int(config, "patch_length")
        if input_spec.length % patch_length:
            raise ValueError(
                f"{specification.model} adapter {self.model_type} input.length must be a "
                f"multiple of config.patch_length={patch_length}"
            )
        _require_equal(
            specification,
            self.model_type,
            "dimension",
            specification.dimension,
            _config_int(config, "hidden_size"),
        )
        _require_equal(
            specification,
            self.model_type,
            "pooling",
            specification.pooling,
            "mean-valid-patches-v1",
        )
        _require_equal(
            specification,
            self.model_type,
            "input.normalization",
            input_spec.normalization,
            "timesfm2_5-config-normalization-v1",
        )

    def embed(self, model: Any, batch: pa.RecordBatch, torch: Any, device: str) -> Any:
        specification = cast("EmbeddingModelSpec", model._duckpd_specification)
        input_spec = cast("SeriesEmbeddingInputSpec", specification.input)
        values = _channel_tensors(batch, input_spec, torch=torch, device=device)[0]
        padding = torch.zeros(
            (batch.num_rows, input_spec.length),
            dtype=torch.long,
            device=device,
        )
        hidden = _hidden_state(
            model(past_values=values, past_values_padding=padding),
            model=specification.model,
            adapter=self.model_type,
        )
        expected = (
            batch.num_rows,
            input_spec.length // _config_int(model.config, "patch_length"),
            specification.dimension,
        )
        _validate_hidden(
            hidden,
            expected=expected,
            model=specification.model,
            adapter=self.model_type,
        )
        return hidden.to(dtype=torch.float32).mean(dim=1)


class _EncoderDecoderAdapter:
    def __init__(self, model_type: str, class_name: str) -> None:
        self.model_type = model_type
        self.class_name = class_name

    def validate(self, config: Any, specification: EmbeddingModelSpec) -> None:
        input_spec = cast("SeriesEmbeddingInputSpec", specification.input)
        _require_equal(
            specification,
            self.model_type,
            "input.normalization",
            input_spec.normalization,
            "hf-time-series-scaler-v1",
        )
        _require_equal(
            specification,
            self.model_type,
            "pooling",
            specification.pooling,
            "mean-encoder-time-v1",
        )
        _require_equal(
            specification,
            self.model_type,
            "dimension",
            specification.dimension,
            _config_int(config, "d_model"),
        )
        raw_lags = cast("object", getattr(config, "lags_sequence", None))
        if not isinstance(raw_lags, (list, tuple)) or not raw_lags:
            raise ValueError(
                f"{specification.model} adapter {self.model_type} requires a non-empty "
                "integer config.lags_sequence"
            )
        lag_values = cast("list[object] | tuple[object, ...]", raw_lags)
        if any(type(value) is not int or value < 0 for value in lag_values):
            raise ValueError(
                f"{specification.model} adapter {self.model_type} requires a non-empty "
                "integer config.lags_sequence"
            )
        lags = cast("tuple[int, ...]", lag_values)
        _require_equal(
            specification,
            self.model_type,
            "input.length",
            input_spec.length,
            _config_int(config, "context_length") + max(lags),
        )
        target_count = sum(role == "target" for role in input_spec.roles)
        dynamic_count = len(input_spec.roles) - target_count
        _require_equal(
            specification,
            self.model_type,
            "target channel count",
            target_count,
            _config_int(config, "input_size"),
        )
        _require_equal(
            specification,
            self.model_type,
            "dynamic channel count",
            dynamic_count,
            _config_int(config, "num_dynamic_real_features", default=0),
        )
        time_width = _config_int(config, "num_time_features", default=0)
        if (input_spec.temporal is not None) != (time_width > 0):
            raise ValueError(
                f"{specification.model} adapter {self.model_type} temporal declaration must "
                f"match config.num_time_features={time_width}"
            )
        generated_width = input_spec.temporal.width if input_spec.temporal is not None else 0
        _require_equal(
            specification,
            self.model_type,
            "generated temporal feature width",
            generated_width,
            time_width,
        )
        real_specs = tuple(item for item in input_spec.static if item.kind == "real")
        categorical_specs = tuple(item for item in input_spec.static if item.kind == "categorical")
        _require_equal(
            specification,
            self.model_type,
            "static real feature count",
            len(real_specs),
            _config_int(config, "num_static_real_features", default=0),
        )
        _require_equal(
            specification,
            self.model_type,
            "static categorical feature count",
            len(categorical_specs),
            _config_int(config, "num_static_categorical_features", default=0),
        )
        expected_cardinality_values: list[int] = []
        for specification_item in categorical_specs:
            assert specification_item.cardinality is not None
            expected_cardinality_values.append(specification_item.cardinality)
        expected_cardinality = tuple(expected_cardinality_values)
        raw_cardinality = cast("object", getattr(config, "cardinality", []))
        if not isinstance(raw_cardinality, (list, tuple)):
            raise ValueError(
                f"{specification.model} adapter {self.model_type} config.cardinality must "
                "contain positive integers"
            )
        cardinality_values = cast(
            "list[object] | tuple[object, ...]",
            raw_cardinality,
        )
        actual_cardinality: list[int] = []
        for value in cardinality_values:
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"{specification.model} adapter {self.model_type} "
                    "config.cardinality must contain positive integers"
                )
            actual_cardinality.append(value)
        _require_equal(
            specification,
            self.model_type,
            "static categorical cardinalities",
            tuple(actual_cardinality),
            expected_cardinality,
        )

    def embed(self, model: Any, batch: pa.RecordBatch, torch: Any, device: str) -> Any:
        specification = cast("EmbeddingModelSpec", model._duckpd_specification)
        input_spec = cast("SeriesEmbeddingInputSpec", specification.input)
        channel_tensors = _channel_tensors(batch, input_spec, torch=torch, device=device)
        targets = [
            tensor
            for tensor, role in zip(channel_tensors, input_spec.roles, strict=True)
            if role == "target"
        ]
        dynamic = [
            tensor
            for tensor, role in zip(channel_tensors, input_spec.roles, strict=True)
            if role != "target"
        ]
        past_values = torch.stack(targets, dim=-1)
        if len(targets) == 1:
            past_values = past_values.squeeze(-1)
        past_observed_mask = torch.ones_like(past_values)

        generated_width = input_spec.temporal.width if input_spec.temporal is not None else 0
        if generated_width:
            temporal = _field_tensor(
                batch,
                "__duckpd_past_time_features",
                width=input_spec.length * generated_width,
                torch=torch,
                device=device,
                dtype=torch.float32,
            ).reshape(batch.num_rows, input_spec.length, generated_width)
        else:
            temporal = torch.empty(
                (batch.num_rows, input_spec.length, 0),
                dtype=torch.float32,
                device=device,
            )
        if dynamic:
            dynamic_tensor = torch.stack(dynamic, dim=-1)
            past_time_features = torch.cat((temporal, dynamic_tensor), dim=-1)
        else:
            past_time_features = temporal

        real_specs = tuple(item for item in input_spec.static if item.kind == "real")
        categorical_specs = tuple(item for item in input_spec.static if item.kind == "categorical")
        static_real = (
            _field_tensor(
                batch,
                "__duckpd_static_real",
                width=len(real_specs),
                torch=torch,
                device=device,
                dtype=torch.float32,
            )
            if real_specs
            else None
        )
        static_categorical = (
            _field_tensor(
                batch,
                "__duckpd_static_categorical",
                width=len(categorical_specs),
                torch=torch,
                device=device,
                dtype=torch.long,
            )
            if categorical_specs
            else None
        )
        network = model.create_network_inputs(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical,
            static_real_features=static_real,
        )
        context_length = _config_int(model.config, "context_length")
        encoder_input = network[0][:, :context_length, ...]
        output = model.get_encoder()(inputs_embeds=encoder_input, return_dict=True)
        hidden = _hidden_state(
            output,
            model=specification.model,
            adapter=self.model_type,
        )
        expected = (batch.num_rows, context_length, specification.dimension)
        _validate_hidden(
            hidden,
            expected=expected,
            model=specification.model,
            adapter=self.model_type,
        )
        return hidden.to(dtype=torch.float32).mean(dim=1)


_ADAPTERS: dict[str, _TransformersSeriesAdapter] = {
    "patchtst": _PatchBackboneAdapter(
        "patchtst",
        "PatchTSTModel",
        "patchtst-config-scaling-v1",
    ),
    "patchtsmixer": _PatchBackboneAdapter(
        "patchtsmixer",
        "PatchTSMixerModel",
        "patchtsmixer-config-scaling-v1",
    ),
    "timesfm2_5": _TimesFm25Adapter(),
    "time_series_transformer": _EncoderDecoderAdapter(
        "time_series_transformer",
        "TimeSeriesTransformerModel",
    ),
}


class TransformersSeriesEmbeddingProvider:
    """First-party bare-backbone provider for allowlisted Transformers series models."""

    def __init__(
        self,
        specification: EmbeddingModelSpec,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        cache_dir: str | Path | None = None,
        prepare_timeout_seconds: float | None = None,
        max_download_bytes: int | None = None,
    ) -> None:
        if specification.backend != "transformers" or not isinstance(
            specification.input,
            SeriesEmbeddingInputSpec,
        ):
            raise ValueError(
                "TransformersSeriesEmbeddingProvider requires a Transformers series model"
            )
        if specification.input.provider_abi != "transformers-series-v1":
            raise ValueError(
                "TransformersSeriesEmbeddingProvider requires provider_abi='transformers-series-v1'"
            )
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        _validate_prepare_limits(prepare_timeout_seconds, max_download_bytes)
        self._specification = specification
        self._device = device
        cache_root = Path(cache_dir or Path.home() / ".cache" / "duckpd" / "embeddings")
        self._cache_dir = cache_root / specification.fingerprint
        self._prepare_timeout_seconds = prepare_timeout_seconds
        self._max_download_bytes = max_download_bytes
        self._torch: Any | None = None
        self._model: Any | None = None
        self._adapter: _TransformersSeriesAdapter | None = None
        self._prepared: PreparedModelInfo | None = None

    @property
    def specification(self) -> EmbeddingModelSpec:
        return self._specification

    @property
    def thread_safe(self) -> bool:
        return False

    def prepare(self) -> PreparedModelInfo:
        if self._prepared is not None:
            return self._prepared
        try:
            torch = cast("Any", importlib.import_module("torch"))
            transformers = cast("Any", importlib.import_module("transformers"))
            huggingface_hub = cast("Any", importlib.import_module("huggingface_hub"))
        except ImportError:
            raise UnsupportedOperationError(
                "Transformers series embeddings require PyTorch, Transformers, and huggingface-hub"
            ) from None
        if self._device == "cuda" and not torch.cuda.is_available():
            raise UnsupportedOperationError(
                "Transformers series CUDA device requested, but PyTorch cannot access "
                "a CUDA or ROCm GPU"
            )

        manifest_name = f"duckpd-{self._specification.fingerprint}.json"
        manifest_path = self._cache_dir / manifest_name
        lock_path = self._cache_dir.with_name(f".{self._cache_dir.name}.lock")
        started = perf_counter()
        with exclusive_file_lock(lock_path):
            if self._cache_dir.exists():
                artifact_digest, recorded = self._verified_manifest(
                    manifest_path,
                    started=started,
                )
                config, adapter, model, loading = self._load_local(
                    transformers,
                    self._cache_dir,
                )
                self._verify_manifest_runtime(
                    recorded,
                    transformers,
                    config,
                    adapter,
                    model,
                    loading,
                )
                if _directory_digest(self._cache_dir) != artifact_digest:
                    raise UnsupportedOperationError(
                        "Transformers series backend mutated verified artifacts while loading"
                    )
            else:
                self._cache_dir.parent.mkdir(parents=True, exist_ok=True)
                staging = self._cache_dir.with_name(f".{self._cache_dir.name}.tmp-{uuid4().hex}")
                shutil.rmtree(staging, ignore_errors=True)
                try:
                    huggingface_hub.snapshot_download(
                        repo_id=self._specification.model,
                        revision=self._specification.revision,
                        local_dir=staging,
                        allow_patterns=(
                            "config.json",
                            "*.safetensors",
                            "pytorch_model*.bin",
                        ),
                    )
                    _check_prepare_limits(
                        staging,
                        started=started,
                        timeout_seconds=self._prepare_timeout_seconds,
                        max_bytes=self._max_download_bytes,
                    )
                    config, adapter, model, loading = self._load_local(
                        transformers,
                        staging,
                    )
                    artifact_digest = _directory_digest(staging)
                    manifest = self._manifest(
                        transformers,
                        config,
                        adapter,
                        model,
                        loading,
                        artifact_digest,
                    )
                    (staging / manifest_name).write_text(
                        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                        encoding="utf-8",
                    )
                    os.rename(staging, self._cache_dir)
                except BaseException:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise

        model.eval()
        model.to(self._device)
        model._duckpd_specification = self._specification
        runtime = (
            "PyTorchROCm"
            if self._device == "cuda" and torch.version.hip is not None
            else "PyTorchCUDA"
            if self._device == "cuda"
            else "PyTorchCPU"
        )
        self._torch = torch
        self._model = model
        self._adapter = adapter
        self._prepared = PreparedModelInfo(
            self._specification.fingerprint,
            "transformers",
            str(self._cache_dir),
            artifact_digest,
            (runtime,),
        )
        return self._prepared

    def embed_windows(self, batch: pa.RecordBatch) -> pa.Array[Any]:
        if self._torch is None or self._model is None or self._adapter is None:
            raise UnsupportedOperationError(
                "Embedding model is not prepared; call session.prepare_embedding_model(model)"
            )
        self._validate_batch(batch)
        if batch.num_rows == 0:
            child = pa.array([], type=pa.float32())
            maker = cast("Any", pa.FixedSizeListArray)
            return cast("pa.Array[Any]", maker.from_arrays(child, self._specification.dimension))
        with self._torch.inference_mode():
            pooled = self._adapter.embed(
                self._model,
                batch,
                self._torch,
                self._device,
            )
        output = pooled.detach().to(device="cpu", dtype=self._torch.float32).contiguous().numpy()
        expected = (batch.num_rows, self._specification.dimension)
        if output.shape != expected:
            raise ValueError(
                f"{self._specification.model} adapter {self._adapter.model_type} returned "
                f"pooled shape {output.shape}; expected {expected}"
            )
        child = pa.array(output.reshape(-1), type=pa.float32())
        if child.null_count:
            raise ValueError(
                f"{self._specification.model} adapter {self._adapter.model_type} returned "
                "nonfinite embeddings"
            )
        values = child.to_numpy(zero_copy_only=False)
        numpy = importlib.import_module("numpy")
        if not bool(numpy.isfinite(values).all()):
            raise ValueError(
                f"{self._specification.model} adapter {self._adapter.model_type} returned "
                "nonfinite embeddings"
            )
        maker = cast("Any", pa.FixedSizeListArray)
        return cast(
            "pa.Array[Any]",
            maker.from_arrays(child, self._specification.dimension),
        )

    def close(self) -> None:
        self._torch = None
        self._model = None
        self._adapter = None
        self._prepared = None

    def _load_local(
        self,
        transformers: Any,
        root: Path,
    ) -> tuple[Any, _TransformersSeriesAdapter, Any, dict[str, object]]:
        options = {
            "trust_remote_code": False,
            "local_files_only": True,
        }
        config = transformers.AutoConfig.from_pretrained(str(root), **options)
        raw_model_type = getattr(config, "model_type", None)
        model_type = raw_model_type if isinstance(raw_model_type, str) else None
        adapter = _ADAPTERS.get(model_type) if model_type is not None else None
        if adapter is None:
            raise UnsupportedOperationError(
                f"{self._specification.model} has unsupported Transformers series "
                f"config.model_type={model_type!r}"
            )
        expected_class = getattr(transformers, adapter.class_name, None)
        if expected_class is None:
            raise UnsupportedOperationError(
                f"{self._specification.model} adapter {adapter.model_type} requires "
                f"Transformers architecture {adapter.class_name}; install a release exposing it "
                "through AutoModel"
            )
        adapter.validate(config, self._specification)
        raw_loaded = cast(
            "object",
            transformers.AutoModel.from_pretrained(
                str(root),
                trust_remote_code=False,
                local_files_only=True,
                ignore_mismatched_sizes=False,
                output_loading_info=True,
            ),
        )
        if not isinstance(raw_loaded, tuple):
            raise UnsupportedOperationError(
                "Transformers AutoModel did not return loading information"
            )
        loaded_values = cast("tuple[object, ...]", raw_loaded)
        if len(loaded_values) != 2 or not isinstance(loaded_values[1], dict):
            raise UnsupportedOperationError(
                "Transformers AutoModel did not return loading information"
            )
        model = cast("Any", loaded_values[0])
        loading = cast("dict[str, object]", loaded_values[1])
        if type(model) is not expected_class:
            raise UnsupportedOperationError(
                f"{self._specification.model} adapter {adapter.model_type} resolved "
                f"{type(model).__name__}; expected bare class {adapter.class_name}"
            )
        for field in ("missing_keys", "mismatched_keys", "error_msgs"):
            value = loading.get(field, [])
            if value:
                raise UnsupportedOperationError(
                    f"{self._specification.model} adapter {adapter.model_type} has non-empty "
                    f"loading information {field}: {value!r}"
                )
        unexpected = cast("list[object]", loading.get("unexpected_keys", []))
        if any(
            not isinstance(key, str)
            or re.search(
                r"(?:^|[._])(classifier|decoder|forecast|head|projection|regression)"
                r"(?:[._]|$)",
                key.lower(),
            )
            is None
            for key in unexpected
        ):
            raise UnsupportedOperationError(
                f"{self._specification.model} adapter {adapter.model_type} has "
                f"non-task-head unexpected keys: {unexpected!r}"
            )
        return config, adapter, model, loading

    def _manifest(
        self,
        transformers: Any,
        config: Any,
        adapter: _TransformersSeriesAdapter,
        model: Any,
        loading: dict[str, object],
        artifact_digest: str,
    ) -> dict[str, object]:
        raw_config = cast("dict[str, object]", config.to_dict())
        stable_config = {key: value for key, value in raw_config.items() if key != "_name_or_path"}
        config_payload = json.dumps(
            stable_config,
            sort_keys=True,
            separators=(",", ":"),
        )
        unexpected = sorted(cast("list[str]", loading.get("unexpected_keys", [])))
        loading_payload = json.dumps(unexpected, separators=(",", ":"))
        return {
            "model_fingerprint": self._specification.fingerprint,
            "model_specification": self._specification.to_dict(),
            "artifact_digest": artifact_digest,
            "model_type": adapter.model_type,
            "resolved_class": type(model).__name__,
            "adapter_abi": "transformers-series-v1",
            "pooling": self._specification.pooling,
            "configuration_digest": hashlib.sha256(config_payload.encode()).hexdigest(),
            "transformers_version": str(getattr(transformers, "__version__", "unknown")),
            "execution_provider": self._device,
            "unexpected_keys": unexpected,
            "unexpected_keys_digest": hashlib.sha256(loading_payload.encode()).hexdigest(),
        }

    def _verified_manifest(
        self,
        manifest_path: Path,
        *,
        started: float,
    ) -> tuple[str, dict[str, object]]:
        if not manifest_path.is_file():
            raise UnsupportedOperationError(
                "Embedding cache exists without a verified DuckPD manifest"
            )
        _check_prepare_limits(
            self._cache_dir,
            started=started,
            timeout_seconds=self._prepare_timeout_seconds,
            max_bytes=self._max_download_bytes,
        )
        artifact_digest = _directory_digest(self._cache_dir)
        try:
            raw_recorded = cast(
                "object",
                json.loads(manifest_path.read_text(encoding="utf-8")),
            )
        except (OSError, TypeError, json.JSONDecodeError):
            raise UnsupportedOperationError(
                "Prepared embedding cache has an invalid verification manifest"
            ) from None
        if not isinstance(raw_recorded, dict):
            raise UnsupportedOperationError(
                "Prepared embedding cache has an invalid verification manifest"
            )
        recorded = cast("dict[str, object]", raw_recorded)
        if (
            recorded.get("model_fingerprint") != self._specification.fingerprint
            or recorded.get("artifact_digest") != artifact_digest
            or recorded.get("model_specification") != self._specification.to_dict()
            or recorded.get("adapter_abi") != "transformers-series-v1"
        ):
            raise UnsupportedOperationError(
                "Prepared embedding artifacts changed since their verified cache promotion"
            )
        return artifact_digest, recorded

    def _verify_manifest_runtime(
        self,
        recorded: dict[str, object],
        transformers: Any,
        config: Any,
        adapter: _TransformersSeriesAdapter,
        model: Any,
        loading: dict[str, object],
    ) -> None:
        current = self._manifest(
            transformers,
            config,
            adapter,
            model,
            loading,
            cast("str", recorded["artifact_digest"]),
        )
        checked = {
            "model_type",
            "resolved_class",
            "adapter_abi",
            "pooling",
            "configuration_digest",
            "unexpected_keys",
            "unexpected_keys_digest",
            "transformers_version",
            "execution_provider",
        }
        if any(recorded.get(field) != current.get(field) for field in checked):
            raise UnsupportedOperationError(
                "Prepared Transformers series runtime does not match its verified manifest"
            )

    def _validate_batch(self, batch: pa.RecordBatch) -> None:
        input_spec = cast("SeriesEmbeddingInputSpec", self._specification.input)
        expected_names = list(input_spec.channels)
        temporal_width = 0
        if input_spec.temporal is not None:
            expected_names.append("__duckpd_past_time_features")
            temporal_width = input_spec.length * input_spec.temporal.width
        real_specs = tuple(item for item in input_spec.static if item.kind == "real")
        categorical_specs = tuple(item for item in input_spec.static if item.kind == "categorical")
        if real_specs:
            expected_names.append("__duckpd_static_real")
        if categorical_specs:
            expected_names.append("__duckpd_static_categorical")
        if batch.schema.names != expected_names:
            raise ValueError(
                f"{self._specification.model} adapter batch fields {batch.schema.names} "
                f"do not match expected order {expected_names}"
            )

        expected_metadata = {
            b"duckpd.model_fingerprint": self._specification.fingerprint.encode(),
            b"duckpd.provider_abi": b"transformers-series-v1",
            b"duckpd.mask_semantics": b"complete_rows_only-v1",
        }
        representation_fingerprint = (batch.schema.metadata or {}).get(
            b"duckpd.representation_fingerprint"
        )
        if (
            representation_fingerprint is None
            or re.fullmatch(
                rb"[0-9a-f]{64}",
                representation_fingerprint,
            )
            is None
        ):
            raise ValueError(
                f"{self._specification.model} adapter requires a canonical "
                "representation fingerprint"
            )
        expected_metadata[b"duckpd.representation_fingerprint"] = representation_fingerprint
        if input_spec.temporal is not None:
            expected_metadata[b"duckpd.time_feature_shape"] = (
                f"{input_spec.length},{input_spec.temporal.width}".encode()
            )
        metadata = batch.schema.metadata or {}
        if metadata != expected_metadata:
            raise ValueError(
                f"{self._specification.model} adapter batch metadata does not match "
                "the canonical provider boundary"
            )

        def validate_field(
            index: int,
            name: str,
            width: int,
            value_type: pa.DataType,
            *,
            field_metadata: dict[bytes, bytes] | None = None,
        ) -> Any:
            expected = pa.field(
                name,
                pa.list_(value_type, width),
                nullable=False,
                metadata=field_metadata,
            )
            field = cast("pa.Field[Any]", cast("Any", batch.schema).field(index))
            if (
                field.name != name
                or field.type != expected.type
                or field.nullable
                or field.metadata != field_metadata
            ):
                raise ValueError(
                    f"{self._specification.model} adapter field {name!r} has "
                    f"invalid Arrow schema {field}"
                )
            return _list_values(
                cast("pa.Array[Any]", batch.column(index)),
                width,
                batch.num_rows,
                field=name,
            )

        for index, (channel, role) in enumerate(
            zip(input_spec.channels, input_spec.roles, strict=True)
        ):
            validate_field(
                index,
                channel,
                input_spec.length,
                pa.float32(),
                field_metadata={b"duckpd.channel_role": role.encode()},
            )
        next_index = len(input_spec.channels)
        if input_spec.temporal is not None:
            validate_field(
                next_index,
                "__duckpd_past_time_features",
                temporal_width,
                pa.float32(),
            )
            next_index += 1
        if real_specs:
            validate_field(
                next_index,
                "__duckpd_static_real",
                len(real_specs),
                pa.float32(),
            )
            next_index += 1
        if categorical_specs:
            values = validate_field(
                next_index,
                "__duckpd_static_categorical",
                len(categorical_specs),
                pa.int64(),
            )
            if batch.num_rows:
                for index, specification in enumerate(categorical_specs):
                    assert specification.cardinality is not None
                    column = values[:, index]
                    if bool((column < 0).any() or (column >= specification.cardinality).any()):
                        raise ValueError(
                            f"{self._specification.model} adapter categorical static "
                            f"{specification.name!r} is outside its declared cardinality"
                        )
