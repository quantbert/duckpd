"""Attested local-bundle provider for externally trained TS2Vec encoders."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pyarrow as pa

from duckpd.errors import UnsupportedOperationError
from duckpd.series_embeddings import (
    PreparedSeriesModelInfo,
    SeriesChannelRole,
    SeriesEmbeddingModelSpec,
)

_TS2VEC_MANIFEST_NAME = "duckpd-ts2vec-manifest.json"
_TS2VEC_WEIGHTS_NAME = "model.safetensors"
_TS2VEC_SOURCE_REPOSITORY = "https://github.com/zhihanyue/ts2vec"
_TS2VEC_SOURCE_REVISION = "b0088e14a99706c05451316dc6db8d3da9351163"
_TS2VEC_ADAPTER_REVISION = "duckpd-ts2vec-full-series-max-cpu-v1"
_TS2VEC_INPUT_NORMALIZATION = "artifact-train-standard-score-v1"
_TS2VEC_POOLING = "full-series-max-v1"
_TS2VEC_RUNTIME_MINIMUM = (2, 1)


@dataclass(frozen=True)
class _TS2VecBundle:
    root: Path
    manifest: dict[str, object]
    specification: SeriesEmbeddingModelSpec
    architecture: dict[str, object]
    means: tuple[float, ...]
    scales: tuple[float, ...]


def ts2vec_series_embedding_model(bundle_dir: str | Path) -> SeriesEmbeddingModelSpec:
    """Verify a local TS2Vec bundle and return its portable representation identity."""
    return _read_bundle(Path(bundle_dir)).specification


class TS2VecProvider:
    """CPU inference provider for one immutable, externally trained TS2Vec bundle."""

    def __init__(
        self,
        specification: SeriesEmbeddingModelSpec,
        *,
        bundle_dir: str | Path,
        prepare_timeout_seconds: float | None = None,
        max_artifact_bytes: int | None = None,
    ) -> None:
        _validate_prepare_limits(prepare_timeout_seconds, max_artifact_bytes)
        self._specification = specification
        self._bundle_dir = Path(bundle_dir)
        self._prepare_timeout_seconds = prepare_timeout_seconds
        self._max_artifact_bytes = max_artifact_bytes
        self._torch: Any | None = None
        self._runtime: Any | None = None
        self._model: Any | None = None
        self._means: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._scales: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._prepared: PreparedSeriesModelInfo | None = None
        self._last_call_metrics: dict[str, float] = {}

    @property
    def specification(self) -> SeriesEmbeddingModelSpec:
        return self._specification

    @property
    def thread_safe(self) -> bool:
        return False

    @property
    def last_call_metrics(self) -> dict[str, float]:
        """Return phase timings for the most recent successful provider call."""
        return dict(self._last_call_metrics)

    def prepare(self) -> PreparedSeriesModelInfo:
        if self._prepared is not None:
            return self._prepared
        started = perf_counter()
        bundle = _read_bundle(self._bundle_dir)
        _validate_specification(self._specification, bundle.specification)
        weights_path = bundle.root / _TS2VEC_WEIGHTS_NAME
        if (
            self._max_artifact_bytes is not None
            and weights_path.stat().st_size > self._max_artifact_bytes
        ):
            raise UnsupportedOperationError("TS2Vec artifact exceeds the configured size limit")
        _check_timeout(started, self._prepare_timeout_seconds)
        torch, runtime, load_file, runtime_versions = _load_runtime()
        state = load_file(str(weights_path), device="cpu")
        model = runtime.build_encoder(bundle.architecture)
        try:
            model.load_state_dict(state, strict=True)
        except (KeyError, RuntimeError, ValueError):
            raise UnsupportedOperationError(
                "TS2Vec weights do not match the attested architecture"
            ) from None
        model.eval()
        model.to("cpu")
        _check_timeout(started, self._prepare_timeout_seconds)
        verified = _read_bundle(self._bundle_dir)
        if verified.specification != bundle.specification:
            raise UnsupportedOperationError("TS2Vec bundle changed while it was loading")

        self._torch = torch
        self._runtime = runtime
        self._model = model
        self._means = np.asarray(bundle.means, dtype=np.float32)
        self._scales = np.asarray(bundle.scales, dtype=np.float32)
        self._prepared = PreparedSeriesModelInfo(
            model_fingerprint=self._specification.fingerprint,
            resolved_revision=self._specification.revision,
            artifact_sha256=self._specification.artifact_sha256,
            backend="ts2vec",
            adapter_revision=self._specification.adapter_revision,
            input_length=self._specification.input_length,
            input_channels=self._specification.input_channels,
            input_roles=self._specification.input_roles,
            input_normalization=self._specification.input_normalization,
            pooling=self._specification.pooling,
            dimension=self._specification.dimension,
            cache_path=str(bundle.root),
            execution_providers=("PyTorchCPU",),
            runtime_versions=runtime_versions,
        )
        return self._prepared

    def embed_windows(self, batch: pa.RecordBatch) -> pa.Array[Any]:
        if (
            self._model is None
            or self._torch is None
            or self._runtime is None
            or self._means is None
            or self._scales is None
        ):
            raise UnsupportedOperationError(
                "TS2Vec model is not prepared; call "
                "session.prepare_series_embedding_model(model, artifact_dir=...)"
            )
        _validate_batch(batch, self._specification)
        self._last_call_metrics = {}
        conversion_started = perf_counter()
        raw_batch = cast("Any", batch)
        values = np.empty(
            (
                raw_batch.num_rows,
                self._specification.input_length,
                len(self._specification.input_channels),
            ),
            dtype=np.float32,
        )
        for index in range(raw_batch.num_columns):
            column = raw_batch.column(index)
            values[:, :, index] = column.values.to_numpy(zero_copy_only=False).reshape(
                raw_batch.num_rows, self._specification.input_length
            )
        values -= self._means
        values /= self._scales
        tensor = self._torch.from_numpy(values).to(dtype=self._torch.float32, device="cpu")
        input_conversion_seconds = perf_counter() - conversion_started
        model_started = perf_counter()
        with self._torch.no_grad():
            output = self._runtime.full_series_encode(self._model, tensor)
        model_execution_seconds = perf_counter() - model_started
        output_started = perf_counter()
        expected_shape = (raw_batch.num_rows, self._specification.dimension)
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"TS2Vec returned shape {tuple(output.shape)}; expected {expected_shape}"
            )
        if output.dtype != self._torch.float32:
            raise ValueError("TS2Vec must return float32 full-series embeddings")
        if not bool(self._torch.isfinite(output).all().item()):
            raise ValueError("TS2Vec returned non-finite full-series embeddings")
        flattened = output.detach().cpu().contiguous().numpy().reshape(-1)
        child = pa.array(flattened, type=pa.float32())
        maker = cast("Any", pa.FixedSizeListArray)
        result = cast(
            "pa.Array[Any]",
            maker.from_arrays(child, self._specification.dimension),
        )
        self._last_call_metrics = {
            "input_conversion_seconds": input_conversion_seconds,
            "model_execution_seconds": model_execution_seconds,
            "output_conversion_seconds": perf_counter() - output_started,
        }
        return result

    def close(self) -> None:
        self._last_call_metrics = {}
        self._prepared = None
        self._means = None
        self._scales = None
        self._model = None
        self._runtime = None
        self._torch = None


def _read_bundle(root: Path) -> _TS2VecBundle:
    if root.is_symlink() or not root.is_dir():
        raise UnsupportedOperationError("TS2Vec artifact_dir must be a regular local directory")
    expected_files = {_TS2VEC_MANIFEST_NAME, _TS2VEC_WEIGHTS_NAME}
    actual_files: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise UnsupportedOperationError(
                "TS2Vec bundles must contain only regular attested files"
            )
        actual_files.add(path.name)
    if actual_files != expected_files:
        raise UnsupportedOperationError(
            "TS2Vec bundle file set mismatch; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    manifest_path = root / _TS2VEC_MANIFEST_NAME
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise UnsupportedOperationError("TS2Vec bundle manifest is invalid JSON") from None
    if not isinstance(raw, dict):
        raise UnsupportedOperationError("TS2Vec bundle manifest must be an object")
    manifest = cast("dict[str, object]", raw)
    _validate_manifest_fields(manifest)

    artifacts = _mapping(manifest["artifacts"], "artifacts")
    if set(artifacts) != {_TS2VEC_WEIGHTS_NAME}:
        raise UnsupportedOperationError("TS2Vec manifest must attest exactly model.safetensors")
    record = _mapping(artifacts[_TS2VEC_WEIGHTS_NAME], "model artifact")
    if set(record) != {"size", "sha256"}:
        raise UnsupportedOperationError("TS2Vec model artifact record is invalid")
    expected_size = _integer(record["size"], "model artifact size", minimum=1)
    expected_digest = _sha256(record["sha256"], "model artifact sha256")
    weights_path = root / _TS2VEC_WEIGHTS_NAME
    if (
        weights_path.stat().st_size != expected_size
        or _file_sha256(weights_path) != expected_digest
    ):
        raise UnsupportedOperationError("TS2Vec model.safetensors failed verification")

    architecture = _mapping(manifest["architecture"], "architecture")
    channels, roles, input_length = _validate_input(_mapping(manifest["input"], "input"))
    dimension = _validate_architecture(architecture, len(channels))
    means, scales = _validate_preprocessing(
        _mapping(manifest["preprocessing"], "preprocessing"),
        len(channels),
    )
    _validate_output(_mapping(manifest["output"], "output"), dimension)
    model = _nonempty_string(manifest["model"], "model")
    revision = _nonempty_string(manifest["revision"], "revision")
    specification = SeriesEmbeddingModelSpec(
        model=model,
        revision=revision,
        artifact_sha256=_canonical_digest(manifest),
        backend="ts2vec",
        dimension=dimension,
        input_length=input_length,
        input_channels=channels,
        input_roles=cast("tuple[SeriesChannelRole, ...]", roles),
        input_normalization=_TS2VEC_INPUT_NORMALIZATION,
        pooling=_TS2VEC_POOLING,
        adapter_revision=_TS2VEC_ADAPTER_REVISION,
    )
    return _TS2VecBundle(
        root=root,
        manifest=manifest,
        specification=specification,
        architecture=architecture,
        means=means,
        scales=scales,
    )


def _validate_manifest_fields(manifest: dict[str, object]) -> None:
    fields = {
        "schema_version",
        "backend",
        "model",
        "revision",
        "source",
        "architecture",
        "input",
        "preprocessing",
        "pooling",
        "output",
        "runtime",
        "training",
        "artifacts",
    }
    if set(manifest) != fields:
        raise UnsupportedOperationError("TS2Vec manifest field set is invalid")
    if manifest["schema_version"] != 1 or manifest["backend"] != "ts2vec":
        raise UnsupportedOperationError("TS2Vec manifest schema or backend is unsupported")
    source = _mapping(manifest["source"], "source")
    if source != {
        "repository": _TS2VEC_SOURCE_REPOSITORY,
        "revision": _TS2VEC_SOURCE_REVISION,
        "license": "MIT",
        "implementation": "duckpd-pinned-port-v1",
    }:
        raise UnsupportedOperationError("TS2Vec source attestation is unsupported")
    if manifest["pooling"] != _TS2VEC_POOLING:
        raise UnsupportedOperationError("TS2Vec bundle must use full-series max pooling")
    runtime = _mapping(manifest["runtime"], "runtime")
    if set(runtime) != {"python", "numpy", "torch", "safetensors"} or not all(
        isinstance(value, str) and value for value in runtime.values()
    ):
        raise UnsupportedOperationError("TS2Vec training runtime attestation is invalid")
    training = _mapping(manifest["training"], "training")
    required_training = {
        "dataset",
        "dataset_sha256",
        "data_contract",
        "derived_channels",
        "tickers",
        "points_per_ticker",
        "split",
        "window_stride",
        "seed",
        "iterations",
        "batch_size",
        "learning_rate",
        "temporal_unit",
        "objective",
        "losses",
    }
    if set(training) != required_training:
        raise UnsupportedOperationError("TS2Vec training provenance field set is invalid")
    _sha256(training["dataset_sha256"], "training dataset sha256")
    if training["objective"] != "hierarchical-contrastive-v1":
        raise UnsupportedOperationError("TS2Vec training objective is unsupported")


def _validate_input(value: dict[str, object]) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    if set(value) != {"length", "channels", "roles", "dtype", "missingness"}:
        raise UnsupportedOperationError("TS2Vec input contract field set is invalid")
    channels = _string_tuple(value["channels"], "input channels")
    roles = _string_tuple(value["roles"], "input roles")
    if len(channels) != len(roles) or not channels:
        raise UnsupportedOperationError("TS2Vec input channels and roles must align")
    if any(role not in {"target", "past_covariate"} for role in roles) or "target" not in roles:
        raise UnsupportedOperationError("TS2Vec input roles are unsupported")
    if value["dtype"] != "float32" or value["missingness"] != "complete-only":
        raise UnsupportedOperationError("TS2Vec requires complete float32 input")
    length = _integer(value["length"], "input length", minimum=2, maximum=16_384)
    return channels, roles, length


def _validate_architecture(value: dict[str, object], channel_count: int) -> int:
    fields = {
        "input_dims",
        "output_dims",
        "hidden_dims",
        "depth",
        "kernel_size",
        "training_mask",
        "weights",
    }
    if set(value) != fields:
        raise UnsupportedOperationError("TS2Vec architecture field set is invalid")
    input_dims = _integer(value["input_dims"], "input_dims", minimum=1, maximum=128)
    output_dims = _integer(value["output_dims"], "output_dims", minimum=1, maximum=1_024)
    _integer(value["hidden_dims"], "hidden_dims", minimum=1, maximum=512)
    _integer(value["depth"], "depth", minimum=1, maximum=20)
    if input_dims != channel_count or value["kernel_size"] != 3:
        raise UnsupportedOperationError("TS2Vec architecture does not match its channels")
    if value["training_mask"] != "binomial-v1" or value["weights"] != "swa-averaged":
        raise UnsupportedOperationError("TS2Vec mask or weight selection is unsupported")
    return output_dims


def _validate_preprocessing(
    value: dict[str, object],
    channel_count: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if set(value) != {"kind", "mean", "scale", "fitted_on"}:
        raise UnsupportedOperationError("TS2Vec preprocessing field set is invalid")
    if value["kind"] != _TS2VEC_INPUT_NORMALIZATION or value["fitted_on"] != "training-points-only":
        raise UnsupportedOperationError("TS2Vec preprocessing recipe is unsupported")
    means = _float_tuple(value["mean"], "preprocessing mean")
    scales = _float_tuple(value["scale"], "preprocessing scale")
    if len(means) != channel_count or len(scales) != channel_count:
        raise UnsupportedOperationError("TS2Vec preprocessing state does not match channels")
    if any(scale <= 0 for scale in scales):
        raise UnsupportedOperationError("TS2Vec preprocessing scales must be positive")
    return means, scales


def _validate_output(value: dict[str, object], dimension: int) -> None:
    if value != {
        "dimension": dimension,
        "dtype": "float32",
        "normalization": "none",
    }:
        raise UnsupportedOperationError("TS2Vec output contract is unsupported")


def _validate_specification(
    requested: SeriesEmbeddingModelSpec,
    expected: SeriesEmbeddingModelSpec,
) -> None:
    mismatches = [
        name
        for name in SeriesEmbeddingModelSpec.__dataclass_fields__
        if getattr(requested, name) != getattr(expected, name)
    ]
    if mismatches:
        raise ValueError(
            "TS2Vec specification does not match its attested bundle; mismatched fields: "
            + ", ".join(mismatches)
        )


def _load_runtime() -> tuple[Any, Any, Any, tuple[tuple[str, str], ...]]:
    try:
        torch = cast("Any", importlib.import_module("torch"))
        safetensors = cast("Any", importlib.import_module("safetensors"))
        safetensors_torch = cast("Any", importlib.import_module("safetensors.torch"))
        runtime = cast("Any", importlib.import_module("duckpd._ts2vec_torch"))
    except ImportError:
        raise UnsupportedOperationError(
            "TS2Vec requires the optional 'duckpd[ts2vec]' extra"
        ) from None
    torch_version = str(torch.__version__).split("+", maxsplit=1)[0]
    try:
        torch_parts = tuple(int(part) for part in torch_version.split(".")[:2])
    except ValueError:
        raise UnsupportedOperationError("TS2Vec found an invalid PyTorch version") from None
    if torch_parts < _TS2VEC_RUNTIME_MINIMUM:
        raise UnsupportedOperationError("TS2Vec requires PyTorch 2.1 or newer")
    try:
        safetensors_version = distribution_version("safetensors")
    except PackageNotFoundError:
        safetensors_version = str(safetensors.__version__)
    return (
        torch,
        runtime,
        safetensors_torch.load_file,
        (
            ("python", platform.python_version()),
            ("torch", str(torch.__version__)),
            ("safetensors", safetensors_version),
        ),
    )


def _validate_batch(batch: pa.RecordBatch, specification: SeriesEmbeddingModelSpec) -> None:
    raw_batch = cast("Any", batch)
    if raw_batch.schema.names != list(specification.input_channels):
        raise ValueError("TS2Vec batch channel order does not match its specification")
    if raw_batch.num_columns != len(specification.input_channels):
        raise ValueError("TS2Vec batch channel count does not match its specification")
    expected_type = pa.list_(pa.float32(), specification.input_length)
    for column in raw_batch.columns:
        if column.type != expected_type or column.null_count or column.values.null_count:
            raise ValueError("TS2Vec requires complete FixedSizeList<float32> input windows")


def _validate_prepare_limits(
    timeout_seconds: float | None,
    max_artifact_bytes: int | None,
) -> None:
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("prepare_timeout_seconds must be finite and positive")
    if max_artifact_bytes is not None and (
        type(max_artifact_bytes) is not int or max_artifact_bytes <= 0
    ):
        raise ValueError("max_artifact_bytes must be a positive integer")


def _check_timeout(started: float, timeout_seconds: float | None) -> None:
    if timeout_seconds is not None and perf_counter() - started > timeout_seconds:
        raise UnsupportedOperationError("TS2Vec preparation exceeded its timeout")


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise UnsupportedOperationError(f"TS2Vec {field} must be an object")
    raw = cast("dict[object, object]", value)
    if not all(isinstance(key, str) for key in raw):
        raise UnsupportedOperationError(f"TS2Vec {field} must be an object")
    return {cast("str", key): item for key, item in raw.items()}


def _integer(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise UnsupportedOperationError(f"TS2Vec {field} is outside the supported range")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise UnsupportedOperationError(f"TS2Vec {field} must be a non-empty string")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise UnsupportedOperationError(f"TS2Vec {field} must be an array")
    items = cast("list[object]", value)
    if not all(type(item) is str and item for item in items):
        raise UnsupportedOperationError(f"TS2Vec {field} must be non-empty strings")
    return tuple(cast("list[str]", items))


def _float_tuple(value: object, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise UnsupportedOperationError(f"TS2Vec {field} must be an array")
    items = cast("list[object]", value)
    converted: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise UnsupportedOperationError(f"TS2Vec {field} must contain finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise UnsupportedOperationError(f"TS2Vec {field} must contain finite numbers")
        converted.append(number)
    return tuple(converted)


def _sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise UnsupportedOperationError(f"TS2Vec {field} must be a lowercase SHA-256")
    return value


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize_ts2vec_bundle(root: Path, manifest: dict[str, object]) -> SeriesEmbeddingModelSpec:
    """Finalize and verify a producer-written weights file and manifest payload."""
    weights_path = root / _TS2VEC_WEIGHTS_NAME
    if not weights_path.is_file() or weights_path.is_symlink():
        raise ValueError("TS2Vec producer must write a regular model.safetensors file")
    weights_digest = _file_sha256(weights_path)
    complete = dict(manifest)
    complete["revision"] = f"weights-{weights_digest[:40]}"
    complete["artifacts"] = {
        _TS2VEC_WEIGHTS_NAME: {
            "size": weights_path.stat().st_size,
            "sha256": weights_digest,
        }
    }
    (root / _TS2VEC_MANIFEST_NAME).write_text(
        json.dumps(complete, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return _read_bundle(root).specification
