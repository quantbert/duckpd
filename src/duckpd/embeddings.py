"""Typed text-embedding providers and lazy semantic DataFrame operations."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable
from uuid import uuid4

import pyarrow as pa

from duckpd.errors import UnsupportedOperationError

if TYPE_CHECKING:
    from duckpd.frame import DataFrame


NullTextPolicy = Literal["error", "empty"]
EmbeddingBackend = Literal["fastembed", "transformers", "custom"]


class _FastEmbedModel(Protocol):
    def embed(self, texts: Sequence[str]) -> Iterable[Iterable[float]]: ...

    def query_embed(self, text: str) -> Iterable[Iterable[float]]: ...


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Immutable text-embedding semantics carried by logical plans and columns."""

    model: str
    revision: str
    dimension: int
    backend: EmbeddingBackend = "fastembed"
    normalize: bool = True
    pooling: str = "model-default"
    document_prefix: str = ""
    query_prefix: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if not self.revision or self.revision.casefold() in {"main", "master", "latest", "head"}:
            raise ValueError("revision must identify an immutable model revision")
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not self.pooling:
            raise ValueError("pooling must be non-empty")

    @property
    def fingerprint(self) -> str:
        """Stable, credential-free identity of the complete embedding space."""
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class EmbeddingColumnSpec:
    """Persistable model identity attached to one fixed-size vector column."""

    model: EmbeddingModelSpec

    @property
    def fingerprint(self) -> str:
        return self.model.fingerprint


@dataclass(frozen=True)
class EmbeddedQuery:
    """One finite query vector retaining its embedding-space identity."""

    values: tuple[float, ...]
    model_fingerprint: str


@dataclass(frozen=True)
class PreparedModelInfo:
    """Verified local model artifacts owned by one session."""

    model_fingerprint: str
    backend: str
    cache_path: str | None
    artifact_digest: str | None
    execution_providers: tuple[str, ...]
    preparation_seconds: float = 0.0


@runtime_checkable
class TextEmbeddingProvider(Protocol):
    """Batch-oriented provider contract used by DuckPD's Arrow execution boundary."""

    @property
    def specification(self) -> EmbeddingModelSpec: ...

    def prepare(self) -> PreparedModelInfo: ...

    def embed_documents(self, texts: Sequence[str]) -> pa.Array[Any]: ...

    def embed_query(self, text: str) -> EmbeddedQuery: ...


class FastEmbedProvider:
    """Optional CPU ONNX provider backed by Qdrant FastEmbed."""

    def __init__(self, specification: EmbeddingModelSpec, *, cache_dir: str | Path | None = None):
        if specification.backend != "fastembed":
            raise ValueError("FastEmbedProvider requires backend='fastembed'")
        self._specification = specification
        cache_root = Path(cache_dir or Path.home() / ".cache" / "duckpd" / "embeddings")
        self._cache_dir = cache_root / specification.fingerprint
        self._model: _FastEmbedModel | None = None
        self._prepared: PreparedModelInfo | None = None

    @property
    def specification(self) -> EmbeddingModelSpec:
        return self._specification

    def prepare(self) -> PreparedModelInfo:
        if self._prepared is not None:
            return self._prepared
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError:
            raise UnsupportedOperationError(
                "Local text embeddings require the optional 'duckpd[embeddings]' extra"
            ) from None

        manifest_name = f"duckpd-{self._specification.fingerprint}.json"
        manifest_path = self._cache_dir / manifest_name
        if self._cache_dir.exists():
            if not manifest_path.is_file():
                raise UnsupportedOperationError(
                    "Embedding cache exists without a verified DuckPD manifest"
                )
            artifact_digest = _directory_digest(self._cache_dir)
            try:
                recorded = cast("dict[str, object]", json.loads(manifest_path.read_text()))
            except (OSError, TypeError, json.JSONDecodeError):
                raise UnsupportedOperationError(
                    "Prepared embedding cache has an invalid verification manifest"
                ) from None
            if (
                recorded.get("model_fingerprint") != self._specification.fingerprint
                or recorded.get("artifact_digest") != artifact_digest
            ):
                raise UnsupportedOperationError(
                    "Prepared embedding artifacts changed since their verified cache promotion"
                )
            model = cast(
                "_FastEmbedModel",
                TextEmbedding(  # pyright: ignore[reportUnknownVariableType]
                    model_name=self._specification.model,
                    cache_dir=str(self._cache_dir),
                    providers=["CPUExecutionProvider"],
                ),
            )
            if _directory_digest(self._cache_dir) != artifact_digest:
                raise UnsupportedOperationError(
                    "Embedding backend mutated verified artifacts while loading"
                )
        else:
            self._cache_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = self._cache_dir.with_name(f".{self._cache_dir.name}.tmp-{uuid4().hex}")
            shutil.rmtree(staging, ignore_errors=True)
            try:
                model = cast(
                    "_FastEmbedModel",
                    TextEmbedding(  # pyright: ignore[reportUnknownVariableType]
                        model_name=self._specification.model,
                        cache_dir=str(staging),
                        providers=["CPUExecutionProvider"],
                    ),
                )
                artifact_digest = _directory_digest(staging)
                (staging / manifest_name).write_text(
                    json.dumps(
                        {
                            "model_fingerprint": self._specification.fingerprint,
                            "artifact_digest": artifact_digest,
                        },
                        sort_keys=True,
                    )
                )
                try:
                    os.rename(staging, self._cache_dir)
                except FileExistsError:
                    shutil.rmtree(staging, ignore_errors=True)
                    return self.prepare()
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        self._model = model
        self._prepared = PreparedModelInfo(
            self._specification.fingerprint,
            "fastembed",
            str(self._cache_dir),
            artifact_digest,
            ("CPUExecutionProvider",),
        )
        return self._prepared

    def embed_documents(self, texts: Sequence[str]) -> pa.Array[Any]:
        model = self._require_model()
        values = list(model.embed([self._specification.document_prefix + text for text in texts]))
        return _vectors_to_arrow(values, self._specification)

    def embed_query(self, text: str) -> EmbeddedQuery:
        model = self._require_model()
        values = list(model.query_embed(self._specification.query_prefix + text))
        if len(values) != 1:
            raise ValueError("Embedding provider returned an invalid query row count")
        array = _vectors_to_arrow(values, self._specification)
        rows = cast("list[list[float]]", array.to_pylist())
        return EmbeddedQuery(tuple(rows[0]), self._specification.fingerprint)

    def _require_model(self) -> _FastEmbedModel:
        if self._model is None:
            raise UnsupportedOperationError(
                "Embedding model is not prepared; call session.prepare_embedding_model(model)"
            )
        return self._model


class TransformersEmbeddingProvider:
    """Optional PyTorch provider backed by Hugging Face Transformers."""

    def __init__(
        self,
        specification: EmbeddingModelSpec,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        batch_size: int = 64,
        cache_dir: str | Path | None = None,
    ):
        if specification.backend != "transformers":
            raise ValueError("TransformersEmbeddingProvider requires backend='transformers'")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if specification.pooling not in {"cls", "mean"}:
            raise ValueError("TransformersEmbeddingProvider pooling must be 'cls' or 'mean'")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._specification = specification
        self._device = device
        self._batch_size = batch_size
        cache_root = Path(cache_dir or Path.home() / ".cache" / "duckpd" / "embeddings")
        self._cache_dir = cache_root / specification.fingerprint
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._prepared: PreparedModelInfo | None = None

    @property
    def specification(self) -> EmbeddingModelSpec:
        return self._specification

    def prepare(self) -> PreparedModelInfo:
        if self._prepared is not None:
            return self._prepared
        try:
            torch = cast("Any", importlib.import_module("torch"))
            transformers = cast("Any", importlib.import_module("transformers"))
        except ImportError:
            raise UnsupportedOperationError(
                "Transformers embeddings require PyTorch and Transformers; install a "
                "PyTorch build for the requested accelerator and the 'transformers' package"
            ) from None

        if self._device == "cuda" and not torch.cuda.is_available():
            raise UnsupportedOperationError(
                "Transformers CUDA device requested, but PyTorch cannot access a CUDA or ROCm GPU"
            )

        tokenizer_factory = transformers.AutoTokenizer
        model_factory = transformers.AutoModel
        manifest_name = f"duckpd-{self._specification.fingerprint}.json"
        manifest_path = self._cache_dir / manifest_name
        if self._cache_dir.exists():
            if not manifest_path.is_file():
                raise UnsupportedOperationError(
                    "Embedding cache exists without a verified DuckPD manifest"
                )
            artifact_digest = _directory_digest(self._cache_dir)
            try:
                recorded = cast("dict[str, object]", json.loads(manifest_path.read_text()))
            except (OSError, TypeError, json.JSONDecodeError):
                raise UnsupportedOperationError(
                    "Prepared embedding cache has an invalid verification manifest"
                ) from None
            if (
                recorded.get("model_fingerprint") != self._specification.fingerprint
                or recorded.get("artifact_digest") != artifact_digest
            ):
                raise UnsupportedOperationError(
                    "Prepared embedding artifacts changed since their verified cache promotion"
                )
            tokenizer, model = self._load_model(
                tokenizer_factory,
                model_factory,
                self._cache_dir,
                local_files_only=True,
            )
            if _directory_digest(self._cache_dir) != artifact_digest:
                raise UnsupportedOperationError(
                    "Embedding backend mutated verified artifacts while loading"
                )
        else:
            self._cache_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = self._cache_dir.with_name(f".{self._cache_dir.name}.tmp-{uuid4().hex}")
            shutil.rmtree(staging, ignore_errors=True)
            try:
                tokenizer, model = self._load_model(
                    tokenizer_factory,
                    model_factory,
                    staging,
                    local_files_only=False,
                )
                artifact_digest = _directory_digest(staging)
                (staging / manifest_name).write_text(
                    json.dumps(
                        {
                            "model_fingerprint": self._specification.fingerprint,
                            "artifact_digest": artifact_digest,
                        },
                        sort_keys=True,
                    )
                )
                try:
                    os.rename(staging, self._cache_dir)
                except FileExistsError:
                    shutil.rmtree(staging, ignore_errors=True)
                    return self.prepare()
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise

        model.eval()
        model.to(self._device)
        if self._device == "cuda":
            runtime = "PyTorchROCm" if torch.version.hip is not None else "PyTorchCUDA"
        else:
            runtime = "PyTorchCPU"
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._prepared = PreparedModelInfo(
            self._specification.fingerprint,
            "transformers",
            str(self._cache_dir),
            artifact_digest,
            (runtime,),
        )
        return self._prepared

    def embed_documents(self, texts: Sequence[str]) -> pa.Array[Any]:
        values = self._embed([self._specification.document_prefix + text for text in texts])
        return _vectors_to_arrow(values, self._specification)

    def embed_query(self, text: str) -> EmbeddedQuery:
        values = self._embed([self._specification.query_prefix + text])
        if len(values) != 1:
            raise ValueError("Embedding provider returned an invalid query row count")
        array = _vectors_to_arrow(values, self._specification)
        rows = cast("list[list[float]]", array.to_pylist())
        return EmbeddedQuery(tuple(rows[0]), self._specification.fingerprint)

    def _load_model(
        self,
        tokenizer_factory: Any,
        model_factory: Any,
        cache_dir: Path,
        *,
        local_files_only: bool,
    ) -> tuple[Any, Any]:
        options = {
            "revision": self._specification.revision,
            "cache_dir": str(cache_dir),
            "local_files_only": local_files_only,
        }
        tokenizer = tokenizer_factory.from_pretrained(self._specification.model, **options)
        model = model_factory.from_pretrained(self._specification.model, **options)
        return tokenizer, model

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._torch is None or self._tokenizer is None or self._model is None:
            raise UnsupportedOperationError(
                "Embedding model is not prepared; call session.prepare_embedding_model(model)"
            )
        values: list[list[float]] = []
        for offset in range(0, len(texts), self._batch_size):
            batch = texts[offset : offset + self._batch_size]
            inputs = self._tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}
            with self._torch.inference_mode():
                hidden = self._model(**inputs).last_hidden_state
                if self._specification.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                if self._specification.normalize:
                    pooled = self._torch.nn.functional.normalize(pooled, dim=1)
            values.extend(pooled.detach().to("cpu", dtype=self._torch.float32).tolist())
        return values


class SemanticMethods:
    """Lazy text-to-text retrieval operations for a DataFrame."""

    def __init__(self, frame: DataFrame) -> None:
        self._frame = frame

    def search(
        self,
        query: str,
        *,
        columns: str | Sequence[str],
        model: EmbeddingModelSpec,
        metric: Literal["cosine", "l2", "inner_product"] = "cosine",
        k: int = 10,
        batch_size: int = 256,
        separator: str = "\n\n",
        null_policy: NullTextPolicy = "error",
        distance_column: str = "_distance",
        tie_breaker: str | None = None,
    ) -> DataFrame:
        """Lazily embed eligible documents and retain an exact bounded top-k."""
        from duckpd._logical import SemanticSearchPlan
        from duckpd.vector import _search_metadata

        text_columns = _text_columns(self._frame, columns)
        settings = _embedding_settings(
            self._frame,
            model=model,
            batch_size=batch_size,
            separator=separator,
            null_policy=null_policy,
            output_label=None,
        )
        query_key = self._frame._session._register_embedding_query(model, query)
        vector_metric, distance, tie_column, metadata = _search_metadata(
            self._frame,
            metric=metric,
            k=k,
            distance_column=distance_column,
            tie_breaker=tie_breaker,
        )
        plan = SemanticSearchPlan(
            self._frame._plan,
            text_columns,
            None,
            query_key,
            model,
            settings.batch_size,
            settings.separator,
            settings.null_policy,
            vector_metric,
            k,
            distance,
            tie_column.id if tie_column is not None else None,
            metadata,
        )
        from duckpd.frame import DataFrame

        return DataFrame(self._frame._session, plan)


def embed_text(
    frame: DataFrame,
    *,
    columns: str | Sequence[str],
    into: str,
    model: EmbeddingModelSpec,
    batch_size: int = 256,
    separator: str = "\n\n",
    null_policy: NullTextPolicy = "error",
) -> DataFrame:
    """Build a lazy, row-preserving text embedding plan."""
    from duckpd._logical import Column, ColumnId, EmbeddingPlan, Nullability
    from duckpd._metadata import after_embedding
    from duckpd.frame import DataFrame

    text_columns = _text_columns(frame, columns)
    settings = _embedding_settings(
        frame,
        model=model,
        batch_size=batch_size,
        separator=separator,
        null_policy=null_policy,
        output_label=into,
    )
    output = Column(
        ColumnId.create(),
        into,
        f"FLOAT[{model.dimension}]",
        nullable=Nullability.NON_NULL,
        embedding=EmbeddingColumnSpec(model),
    )
    metadata = after_embedding(frame._plan.metadata, output)
    return DataFrame(
        frame._session,
        EmbeddingPlan(
            frame._plan,
            text_columns,
            output,
            model,
            settings.batch_size,
            settings.separator,
            settings.null_policy,
            metadata,
        ),
    )


def embedding_model(
    model: str,
    *,
    revision: str,
    dimension: int,
    backend: EmbeddingBackend = "fastembed",
    normalize: bool = True,
    pooling: str = "model-default",
    document_prefix: str = "",
    query_prefix: str = "",
) -> EmbeddingModelSpec:
    """Create a side-effect-free immutable embedding model specification."""
    return EmbeddingModelSpec(
        model,
        revision,
        dimension,
        backend,
        normalize,
        pooling,
        document_prefix,
        query_prefix,
    )


@dataclass(frozen=True)
class _EmbeddingSettings:
    batch_size: int
    separator: str
    null_policy: NullTextPolicy


def _embedding_settings(
    frame: DataFrame,
    *,
    model: EmbeddingModelSpec,
    batch_size: int,
    separator: str,
    null_policy: NullTextPolicy,
    output_label: str | None,
) -> _EmbeddingSettings:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if null_policy not in {"error", "empty"}:
        raise ValueError("null_policy must be 'error' or 'empty'")
    if output_label is not None:
        if not output_label:
            raise ValueError("into must be a non-empty string label")
        if output_label in frame.columns:
            raise ValueError(f"embedding output column {output_label!r} already exists")
    if model.dimension <= 0:
        raise ValueError("model dimension must be positive")
    return _EmbeddingSettings(batch_size, separator, null_policy)


def _validate_embedding_array(  # pyright: ignore[reportUnusedFunction]
    value: pa.Array[Any],
    specification: EmbeddingModelSpec,
    *,
    expected_rows: int,
) -> pa.Array[Any]:
    if not isinstance(value, pa.FixedSizeListArray):
        raise TypeError("Embedding provider must return a pyarrow.FixedSizeListArray")
    fixed = cast("Any", value)
    if len(fixed) != expected_rows:
        raise ValueError(f"Embedding provider returned {len(fixed)} rows; expected {expected_rows}")
    if fixed.type.list_size != specification.dimension:
        raise ValueError(
            f"Embedding provider returned dimension {fixed.type.list_size}; "
            f"expected {specification.dimension}"
        )
    if not pa.types.is_float32(fixed.type.value_type):
        raise TypeError("Embedding provider must return float32 values")
    if value.null_count:
        raise ValueError("Embedding provider returned null vectors")
    rows = cast("list[list[float] | None]", fixed.to_pylist())
    for row in rows:
        if row is None or any(item != item or abs(item) == float("inf") for item in row):
            raise ValueError("Embedding provider returned null or non-finite values")
        if specification.normalize:
            norm = sum(item * item for item in row) ** 0.5
            if abs(norm - 1.0) > 1e-4:
                raise ValueError("Embedding provider returned an unnormalized vector")
    return cast("pa.Array[Any]", fixed)


def _directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    resolved_root = root.resolve()
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and not path.resolve().is_relative_to(resolved_root):
            raise UnsupportedOperationError(
                "Embedding cache must not contain symbolic links outside its root"
            )
        if not path.is_file() or path.name.startswith("duckpd-"):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _make_fixed_array(rows: Sequence[Sequence[float]], dimension: int) -> pa.Array[Any]:
    flattened = pa.array((item for row in rows for item in row), type=pa.float32())
    maker = cast("Any", pa.FixedSizeListArray)
    return cast("pa.Array[Any]", maker.from_arrays(flattened, dimension))


def _vectors_to_arrow(
    vectors: Sequence[object],
    specification: EmbeddingModelSpec,
) -> pa.Array[Any]:
    rows: list[list[float]] = []
    for vector in vectors:
        values = [float(item) for item in vector]  # type: ignore[union-attr]
        if len(values) != specification.dimension:
            raise ValueError(
                f"Embedding provider returned dimension {len(values)}; "
                f"expected {specification.dimension}"
            )
        if not all(value == value and abs(value) != float("inf") for value in values):
            raise ValueError("Embedding provider returned non-finite values")
        if specification.normalize:
            norm = sum(value * value for value in values) ** 0.5
            if norm == 0:
                raise ValueError("Embedding provider returned a zero vector")
            values = [value / norm for value in values]
        rows.append(values)
    return _make_fixed_array(rows, specification.dimension)


def _text_columns(frame: DataFrame, columns: str | Sequence[str]):
    labels = (columns,) if isinstance(columns, str) else tuple(columns)
    if not labels:
        raise ValueError("columns must contain at least one text column")
    selected = tuple(frame._column(label) for label in labels)
    unsupported = next(
        (column for column in selected if not column.duckdb_type.upper().startswith("VARCHAR")),
        None,
    )
    if unsupported is not None:
        raise UnsupportedOperationError(
            f"Text embedding requires VARCHAR columns; {unsupported.label!r} "
            f"has type {unsupported.duckdb_type}"
        )
    return tuple(column.id for column in selected)
