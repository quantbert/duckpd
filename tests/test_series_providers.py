from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

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
        values = np.stack((first, second, first + second), axis=1).astype(np.float32)
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
