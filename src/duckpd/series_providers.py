"""Optional first-party providers for learned time-series embeddings."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast
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
                staging = self._cache_dir.with_name(
                    f".{self._cache_dir.name}.tmp-{uuid4().hex}"
                )
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
            self._channel_values(batch.column(index), input_spec.length, batch.num_rows)
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
            embeddings.detach()
            .to(device="cpu", dtype=self._torch.float32)
            .contiguous()
            .numpy()
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
