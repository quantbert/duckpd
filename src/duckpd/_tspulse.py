"""Pinned CPU provider for TSPulse's published univariate search representation."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

import pyarrow as pa

from duckpd._locking import exclusive_file_lock
from duckpd.errors import UnsupportedOperationError
from duckpd.series_embeddings import PreparedSeriesModelInfo, SeriesEmbeddingModelSpec

_TSPULSE_MODEL = "ibm-granite/granite-timeseries-tspulse-r1"
_TSPULSE_REVISION = "b12164578f7b893ada0028c00d292ba10383d25a"
_TSPULSE_PACKAGE_VERSION = "0.3.9"
_TSPULSE_SOURCE_REVISION = "fe7a35697723e2a2f5246ae979474bfc554e26c0"
_TSPULSE_PACKAGE_WHEEL_SHA256 = "07ca9c503cfa9f7e34d6ee5616d1a37ae301cabc6daed0546b249ddbc2c8f272"
_TSPULSE_ADAPTER_REVISION = "duckpd-tspulse-decoder-register-cpu-v1"
_TSPULSE_INPUT_NORMALIZATION = "internal-revin-affine-v1"
_TSPULSE_POOLING = "decoder-register-v1"
_TSPULSE_DIMENSION = 240
_TSPULSE_INPUT_LENGTH = 512
_TSPULSE_MANIFEST_NAME = "duckpd-manifest.json"
_TSPULSE_FILE_RECORDS: dict[str, tuple[int, str]] = {
    "config.json": (
        3_522,
        "4a09cdf00f3146a932954cf5a47ff0e767224de1dce8ae456e73d56bbec81621",
    ),
    "model.safetensors": (
        4_305_624,
        "b9332ae796ec7c313f991ed32dbec62c29a8e673281decb7308f955bdda7aae0",
    ),
}


def _artifact_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend": "tspulse",
        "model": _TSPULSE_MODEL,
        "revision": _TSPULSE_REVISION,
        "artifacts": {
            name: {"size": size, "sha256": digest}
            for name, (size, digest) in sorted(_TSPULSE_FILE_RECORDS.items())
        },
        "runtime": {
            "distribution": "granite-tsfm",
            "version": _TSPULSE_PACKAGE_VERSION,
            "source_revision": _TSPULSE_SOURCE_REVISION,
            "published_wheel_sha256": _TSPULSE_PACKAGE_WHEEL_SHA256,
            "python": ">=3.11,<3.14",
        },
        "adapter": {
            "revision": _TSPULSE_ADAPTER_REVISION,
            "input_length": _TSPULSE_INPUT_LENGTH,
            "input_channels": 1,
            "input_roles": ["target"],
            "input_dtype": "float32",
            "missingness": "complete-only",
            "mask": "all-observed-v1",
            "outer_scaling": "none",
            "internal_scaling": _TSPULSE_INPUT_NORMALIZATION,
            "component": "decoder",
            "mode": "register",
            "dimension": _TSPULSE_DIMENSION,
            "output_normalization": "none",
            "device": "cpu",
        },
        "licenses": {"model": "Apache-2.0", "source": "Apache-2.0"},
    }


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def tspulse_series_embedding_model(channel: str) -> SeriesEmbeddingModelSpec:
    """Return the only built-in TSPulse representation supported by DuckPD."""
    if type(channel) is not str or not channel:
        raise ValueError("channel must be a non-empty string")
    return SeriesEmbeddingModelSpec(
        model=_TSPULSE_MODEL,
        revision=_TSPULSE_REVISION,
        artifact_sha256=_canonical_digest(_artifact_manifest()),
        backend="tspulse",
        dimension=_TSPULSE_DIMENSION,
        input_length=_TSPULSE_INPUT_LENGTH,
        input_channels=(channel,),
        input_roles=("target",),
        input_normalization=_TSPULSE_INPUT_NORMALIZATION,
        pooling=_TSPULSE_POOLING,
        adapter_revision=_TSPULSE_ADAPTER_REVISION,
    )


class TSPulseProvider:
    """CPU-only provider for the pinned TSPulse decoder/register checkpoint."""

    def __init__(
        self,
        specification: SeriesEmbeddingModelSpec,
        *,
        cache_dir: str | Path | None = None,
        prepare_timeout_seconds: float | None = None,
        max_download_bytes: int | None = None,
    ) -> None:
        _validate_specification(specification)
        _validate_prepare_limits(prepare_timeout_seconds, max_download_bytes)
        self._specification = specification
        cache_root = Path(cache_dir or Path.home() / ".cache" / "duckpd" / "series-embeddings")
        self._cache_dir = cache_root / specification.fingerprint
        self._prepare_timeout_seconds = prepare_timeout_seconds
        self._max_download_bytes = max_download_bytes
        self._torch: Any | None = None
        self._model: Any | None = None
        self._get_embeddings: Any | None = None
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
        torch, model_factory, get_embeddings, runtime_versions = _load_runtime()
        started = perf_counter()
        manifest_path = self._cache_dir / _TSPULSE_MANIFEST_NAME
        lock_path = self._cache_dir.with_name(f".{self._cache_dir.name}.lock")
        with exclusive_file_lock(lock_path):
            if self._cache_dir.exists():
                _check_prepare_limits(
                    self._cache_dir,
                    started=started,
                    timeout_seconds=self._prepare_timeout_seconds,
                    max_bytes=self._max_download_bytes,
                )
                _verify_cache(self._cache_dir, self._specification)
            else:
                expected_size = sum(size for size, _ in _TSPULSE_FILE_RECORDS.values())
                if (
                    self._max_download_bytes is not None
                    and expected_size > self._max_download_bytes
                ):
                    raise UnsupportedOperationError(
                        "TSPulse artifacts exceed the configured download-size limit"
                    )
                self._cache_dir.parent.mkdir(parents=True, exist_ok=True)
                staging = self._cache_dir.with_name(f".{self._cache_dir.name}.tmp-{uuid4().hex}")
                download_cache = staging.with_name(f".{staging.name}.download")
                shutil.rmtree(staging, ignore_errors=True)
                shutil.rmtree(download_cache, ignore_errors=True)
                try:
                    staging.mkdir()
                    _download_artifacts(staging, download_cache)
                    _check_prepare_limits(
                        staging,
                        started=started,
                        timeout_seconds=self._prepare_timeout_seconds,
                        max_bytes=self._max_download_bytes,
                    )
                    _verify_artifacts(staging)
                    manifest_path = staging / _TSPULSE_MANIFEST_NAME
                    manifest_path.write_text(
                        json.dumps(
                            {
                                "artifact_manifest": _artifact_manifest(),
                                "artifact_manifest_sha256": self._specification.artifact_sha256,
                                "model_fingerprint": self._specification.fingerprint,
                                "model_specification": self._specification.to_dict(),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        encoding="utf-8",
                    )
                    os.rename(staging, self._cache_dir)
                except BaseException:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
                finally:
                    shutil.rmtree(download_cache, ignore_errors=True)
                _verify_cache(self._cache_dir, self._specification)

            model = model_factory.from_pretrained(
                str(self._cache_dir),
                local_files_only=True,
                num_input_channels=1,
                mask_type="user",
            )
            model.eval()
            model.to("cpu")
            _validate_loaded_model(model)
            _verify_cache(self._cache_dir, self._specification)

        self._torch = torch
        self._model = model
        self._get_embeddings = get_embeddings
        self._prepared = PreparedSeriesModelInfo(
            model_fingerprint=self._specification.fingerprint,
            resolved_revision=self._specification.revision,
            artifact_sha256=self._specification.artifact_sha256,
            backend=self._specification.backend,
            adapter_revision=self._specification.adapter_revision,
            input_length=self._specification.input_length,
            input_channels=self._specification.input_channels,
            input_roles=self._specification.input_roles,
            input_normalization=self._specification.input_normalization,
            pooling=self._specification.pooling,
            dimension=self._specification.dimension,
            cache_path=str(self._cache_dir),
            execution_providers=("PyTorchCPU",),
            runtime_versions=runtime_versions,
        )
        return self._prepared

    def embed_windows(self, batch: pa.RecordBatch) -> pa.Array[Any]:
        if self._model is None or self._torch is None or self._get_embeddings is None:
            raise UnsupportedOperationError(
                "TSPulse model is not prepared; call session.prepare_series_embedding_model(model)"
            )
        _validate_batch(batch, self._specification)
        self._last_call_metrics = {}
        conversion_started = perf_counter()
        raw_batch = cast("Any", batch)
        column = raw_batch.column(0)
        values = column.values.to_numpy(zero_copy_only=False)
        owned_values = values.reshape(raw_batch.num_rows, _TSPULSE_INPUT_LENGTH, 1).copy()
        tensor = self._torch.from_numpy(owned_values)
        tensor = tensor.to(dtype=self._torch.float32, device="cpu")
        observed = self._torch.ones_like(tensor, dtype=self._torch.bool)
        input_conversion_seconds = perf_counter() - conversion_started
        model_started = perf_counter()
        with self._torch.no_grad():
            output = self._get_embeddings(
                self._model,
                tensor,
                past_observed_mask=observed,
                component="decoder",
                mode="register",
            )
        model_execution_seconds = perf_counter() - model_started
        output_started = perf_counter()
        expected_shape = (raw_batch.num_rows, 1, _TSPULSE_DIMENSION)
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"TSPulse returned shape {tuple(output.shape)}; expected {expected_shape}"
            )
        if output.dtype != self._torch.float32:
            raise ValueError("TSPulse must return float32 decoder/register embeddings")
        if not bool(self._torch.isfinite(output).all().item()):
            raise ValueError("TSPulse returned non-finite decoder/register embeddings")
        flattened = output[:, 0, :].detach().cpu().contiguous().numpy().reshape(-1)
        child = pa.array(flattened, type=pa.float32())
        maker = cast("Any", pa.FixedSizeListArray)
        result = cast("pa.Array[Any]", maker.from_arrays(child, _TSPULSE_DIMENSION))
        self._last_call_metrics = {
            "input_conversion_seconds": input_conversion_seconds,
            "model_execution_seconds": model_execution_seconds,
            "output_conversion_seconds": perf_counter() - output_started,
        }
        return result

    def close(self) -> None:
        self._last_call_metrics = {}
        self._prepared = None
        self._get_embeddings = None
        self._model = None
        self._torch = None


def _validate_specification(specification: SeriesEmbeddingModelSpec) -> None:
    expected = tspulse_series_embedding_model(specification.input_channels[0])
    mismatches = [
        name
        for name in SeriesEmbeddingModelSpec.__dataclass_fields__
        if getattr(specification, name) != getattr(expected, name)
    ]
    if mismatches:
        raise ValueError(
            "TSPulseProvider supports only the pinned univariate decoder/register contract; "
            "mismatched fields: " + ", ".join(mismatches)
        )


def _load_runtime() -> tuple[Any, Any, Any, tuple[tuple[str, str], ...]]:
    if sys.version_info >= (3, 14):
        raise UnsupportedOperationError(
            "TSPulse requires Python 3.11-3.13 because granite-tsfm==0.3.9 excludes Python 3.14"
        )
    try:
        granite_version = distribution_version("granite-tsfm")
    except PackageNotFoundError:
        raise UnsupportedOperationError(
            "TSPulse requires the optional 'duckpd[tspulse]' extra"
        ) from None
    if granite_version != _TSPULSE_PACKAGE_VERSION:
        raise UnsupportedOperationError(
            f"TSPulse requires granite-tsfm=={_TSPULSE_PACKAGE_VERSION}; found {granite_version}"
        )
    try:
        torch = cast("Any", importlib.import_module("torch"))
        transformers = cast("Any", importlib.import_module("transformers"))
        module = cast("Any", importlib.import_module("tsfm_public.models.tspulse"))
        helpers = cast("Any", importlib.import_module("tsfm_public.models.tspulse.utils.helpers"))
    except ImportError:
        raise UnsupportedOperationError(
            "TSPulse runtime is incomplete; reinstall the optional 'duckpd[tspulse]' extra"
        ) from None
    return (
        torch,
        module.TSPulseForReconstruction,
        helpers.get_embeddings,
        (
            ("python", platform.python_version()),
            ("granite-tsfm", granite_version),
            ("torch", str(torch.__version__)),
            ("transformers", str(transformers.__version__)),
        ),
    )


def _download_artifacts(target: Path, download_cache: Path) -> None:
    try:
        hub = cast("Any", importlib.import_module("huggingface_hub"))
    except ImportError:
        raise UnsupportedOperationError(
            "TSPulse artifact preparation requires huggingface-hub"
        ) from None
    for filename in sorted(_TSPULSE_FILE_RECORDS):
        source = hub.hf_hub_download(
            repo_id=_TSPULSE_MODEL,
            filename=filename,
            revision=_TSPULSE_REVISION,
            cache_dir=str(download_cache),
        )
        shutil.copyfile(source, target / filename)


def _verify_cache(root: Path, specification: SeriesEmbeddingModelSpec) -> None:
    _verify_artifacts(root)
    manifest_path = root / _TSPULSE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise UnsupportedOperationError("TSPulse cache exists without a DuckPD manifest")
    try:
        recorded = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, TypeError, json.JSONDecodeError):
        raise UnsupportedOperationError("TSPulse cache has an invalid DuckPD manifest") from None
    expected = {
        "artifact_manifest": _artifact_manifest(),
        "artifact_manifest_sha256": specification.artifact_sha256,
        "model_fingerprint": specification.fingerprint,
        "model_specification": json.loads(json.dumps(specification.to_dict())),
    }
    if recorded != expected or specification.artifact_sha256 != _canonical_digest(
        _artifact_manifest()
    ):
        raise UnsupportedOperationError("TSPulse cache manifest does not match its specification")


def _verify_artifacts(root: Path) -> None:
    if root.is_symlink():
        raise UnsupportedOperationError("TSPulse cache root must not be a symlink")
    allowed = {*_TSPULSE_FILE_RECORDS, _TSPULSE_MANIFEST_NAME}
    actual: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise UnsupportedOperationError(
                "TSPulse cache must contain only regular attested files"
            )
        actual.add(path.name)
    unexpected = actual - allowed
    missing = set(_TSPULSE_FILE_RECORDS) - actual
    if unexpected or missing:
        raise UnsupportedOperationError(
            "TSPulse cache file set mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for filename, (expected_size, expected_digest) in _TSPULSE_FILE_RECORDS.items():
        path = root / filename
        if path.stat().st_size != expected_size or _file_sha256(path) != expected_digest:
            raise UnsupportedOperationError(f"TSPulse artifact {filename!r} failed verification")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_loaded_model(model: Any) -> None:
    config = model.config
    expected: dict[str, object] = {
        "context_length": 512,
        "num_input_channels": 1,
        "patch_register_tokens": 10,
        "scaling": "revin",
        "revin_affine": True,
        "minimum_scale": 0.001,
        "mask_type": "user",
    }
    mismatches = [name for name, value in expected.items() if getattr(config, name, None) != value]
    if list(getattr(config, "decoder_d_model_layerwise", ())) != [24, 24]:
        mismatches.append("decoder_d_model_layerwise")
    if list(getattr(config, "decoder_num_patches_layerwise", ())) != [128, 128]:
        mismatches.append("decoder_num_patches_layerwise")
    if mismatches:
        raise UnsupportedOperationError(
            "Loaded TSPulse architecture does not match the pinned search contract: "
            + ", ".join(mismatches)
        )


def _validate_batch(batch: pa.RecordBatch, specification: SeriesEmbeddingModelSpec) -> None:
    raw_batch = cast("Any", batch)
    if raw_batch.schema.names != list(specification.input_channels):
        raise ValueError("TSPulse batch channel order does not match its specification")
    if raw_batch.num_columns != 1:
        raise ValueError("TSPulse supports exactly one target channel")
    column = raw_batch.column(0)
    expected_type = pa.list_(pa.float32(), _TSPULSE_INPUT_LENGTH)
    if column.type != expected_type or column.null_count or column.values.null_count:
        raise ValueError("TSPulse requires complete FixedSizeList<float32>[512] target windows")


def _validate_prepare_limits(
    timeout_seconds: float | None,
    max_download_bytes: int | None,
) -> None:
    if timeout_seconds is not None and (isinstance(timeout_seconds, bool) or timeout_seconds <= 0):
        raise ValueError("prepare_timeout_seconds must be positive")
    if max_download_bytes is not None and (
        type(max_download_bytes) is not int or max_download_bytes <= 0
    ):
        raise ValueError("max_download_bytes must be a positive integer")


def _check_prepare_limits(
    root: Path,
    *,
    started: float,
    timeout_seconds: float | None,
    max_bytes: int | None,
) -> None:
    if timeout_seconds is not None and perf_counter() - started > timeout_seconds:
        raise UnsupportedOperationError("TSPulse preparation exceeded its timeout")
    if max_bytes is not None:
        size = sum((root / filename).stat().st_size for filename in _TSPULSE_FILE_RECORDS)
        if size > max_bytes:
            raise UnsupportedOperationError(
                "TSPulse artifacts exceed the configured download-size limit"
            )
