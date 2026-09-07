"""Streaming text embeddings and semantic retrieval contracts."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pandas as pd
import pyarrow as pa
import pytest

import duckpd
from duckpd._logical import EmbeddingPlan, SemanticSearchPlan
from duckpd.embeddings import (
    EmbeddedQuery,
    EmbeddingModelSpec,
    FastEmbedProvider,
    PreparedModelInfo,
    _directory_digest,
    _make_fixed_array,
    _validate_embedding_array,
    _vectors_to_arrow,
)
from duckpd.errors import MaterializationError, UnsupportedOperationError


class KeywordProvider:
    def __init__(self, specification: EmbeddingModelSpec) -> None:
        self._specification = specification
        self.document_batches: list[tuple[str, ...]] = []
        self.queries: list[str] = []

    @property
    def specification(self) -> EmbeddingModelSpec:
        return self._specification

    def prepare(self) -> PreparedModelInfo:
        return PreparedModelInfo(
            self.specification.fingerprint,
            "custom",
            None,
            "test-artifact",
            ("CPUExecutionProvider",),
        )

    def embed_documents(self, texts: Sequence[str]) -> pa.Array:  # type: ignore[type-arg]
        self.document_batches.append(tuple(texts))
        return _make_fixed_array([self._vector(text) for text in texts], 3)

    def embed_query(self, text: str) -> EmbeddedQuery:
        self.queries.append(text)
        return EmbeddedQuery(tuple(self._vector(text)), self.specification.fingerprint)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        values = [
            1.0 if any(word in lowered for word in ("ai", "chip", "gpu")) else 0.0,
            1.0 if any(word in lowered for word in ("earnings", "revenue", "profit")) else 0.0,
            1.0,
        ]
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]


def _model(name: str = "test/keyword") -> EmbeddingModelSpec:
    return duckpd.embedding_model(
        name,
        revision="0123456789abcdef0123456789abcdef01234567",
        dimension=3,
        backend="custom",
        pooling="keyword-test",
    )


def _frame() -> tuple[duckpd.Session, duckpd.DataFrame, KeywordProvider]:
    session = duckpd.connect()
    model = _model()
    provider = KeywordProvider(model)
    session.register_embedding_provider(model, provider)
    session.prepare_embedding_model(model)
    frame = session.from_pandas(
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "title": [
                    "AI chip demand rises",
                    "Quarterly earnings beat",
                    "AI revenue accelerates",
                    "Board names a director",
                ],
                "description": ["GPU sales", "profit growth", "earnings and chips", "governance"],
                "kind": ["tech", "finance", "tech", "other"],
            }
        )
    )
    return session, frame, provider


def test_embed_text_is_lazy_batched_typed_and_row_preserving() -> None:
    session, frame, provider = _frame()
    model = provider.specification
    execution_count = session.execution_count

    embedded = frame.embed_text(
        columns=["title", "description"],
        into="embedding",
        model=model,
        batch_size=2,
    )

    assert isinstance(embedded._plan, EmbeddingPlan)
    assert session.execution_count == execution_count
    assert provider.document_batches == []
    output = embedded.to_arrow()
    assert output.num_rows == 4
    assert output.column_names == ["id", "title", "description", "kind", "embedding"]
    assert output.schema.field("embedding").type == pa.list_(  # pyright: ignore[reportUnknownMemberType]
        pa.float32(), 3
    )
    assert [len(batch) for batch in provider.document_batches] == [2, 2]
    assert embedded._plan.metadata.row_identity == frame._plan.metadata.row_identity
    assert embedded._plan.metadata.ordering == frame._plan.metadata.ordering


def test_embed_text_reports_materialization_progress(capsys: pytest.CaptureFixture[str]) -> None:
    _, frame, provider = _frame()

    frame.embed_text(
        columns=["title", "description"],
        into="embedding",
        model=provider.specification,
        batch_size=2,
    ).collect()

    progress_output = capsys.readouterr().err
    assert "Embedding text" in progress_output
    assert "4/4" in progress_output


def test_transient_search_matches_persisted_search_text_and_prefilter() -> None:
    _, frame, provider = _frame()
    model = provider.specification
    embedded = frame.embed_text(columns=["title", "description"], into="embedding", model=model)
    persisted = embedded.persist("embedded_news")

    query = "AI revenue and chip earnings"
    stored = persisted.vector.search_text(
        query,
        column="embedding",
        model=model,
        k=4,
        tie_breaker="id",
    )
    transient = frame.semantic.search(
        query,
        columns=["title", "description"],
        model=model,
        k=4,
        tie_breaker="id",
    )

    assert isinstance(transient._plan, SemanticSearchPlan)
    assert stored.collect()["id"].tolist() == transient.collect()["id"].tolist()
    assert provider.queries == [query, query]

    eligible = frame[frame["kind"] == "finance"]
    before = eligible.semantic.search(
        query,
        columns=["title", "description"],
        model=model,
        k=2,
        tie_breaker="id",
    )
    after = transient[transient["kind"] == "finance"]
    assert before.collect()["id"].tolist() == [2]
    assert after.collect()["id"].tolist() == [2]


@pytest.mark.parametrize("metric", ["cosine", "l2", "inner_product"])
def test_transient_search_matches_exact_vector_oracle_for_every_metric(
    metric: str,
) -> None:
    session, frame, provider = _frame()
    model = provider.specification
    query = session.embed_query("AI revenue and chip earnings", model=model)
    embedded = frame.embed_text(
        columns=["title", "description"],
        into="embedding",
        model=model,
    ).persist("metric_embeddings")

    oracle = embedded.vector.search(
        query.values,
        column="embedding",
        metric=metric,  # pyright: ignore[reportArgumentType]
        k=4,
        tie_breaker="id",
    )
    transient = frame.semantic.search(
        "AI revenue and chip earnings",
        columns=["title", "description"],
        model=model,
        metric=metric,  # pyright: ignore[reportArgumentType]
        k=4,
        tie_breaker="id",
    )
    assert transient.collect()["id"].tolist() == oracle.collect()["id"].tolist()

    empty = frame[frame["id"] < 0].semantic.search(
        "AI revenue",
        columns="title",
        model=model,
        metric=metric,  # pyright: ignore[reportArgumentType]
        k=4,
        tie_breaker="id",
    )
    assert empty.collect().empty
    assert session.inspect_prepared_embedding_models()[0].model_fingerprint == model.fingerprint


def test_embedding_metadata_survives_parquet_and_table_round_trips(tmp_path: Path) -> None:
    session, frame, provider = _frame()
    model = provider.specification
    embedded = frame.embed_text(columns="title", into="embedding", model=model)
    output = tmp_path / "embedded.parquet"

    embedded.write_parquet(output)
    restored = session.read_parquet(output)
    vector_column = restored._column("embedding")
    assert vector_column.embedding is not None
    assert vector_column.embedding.fingerprint == model.fingerprint
    assert restored.vector.search_text("AI chips", column="embedding", model=model, k=1).collect()[
        "id"
    ].tolist() == [1]

    embedded.save_as_table("embedded_table")
    table = session.table("embedded_table")
    assert table._column("embedding").embedding is not None
    reloaded_session = duckpd.connect()
    reloaded_provider = KeywordProvider(model)
    reloaded_session.register_embedding_provider(model, reloaded_provider)
    reloaded_session.prepare_embedding_model(model)
    reloaded = reloaded_session.read_parquet(output)
    assert reloaded._column("embedding").embedding is not None
    assert reloaded.vector.search_text("AI chips", column="embedding", model=model, k=1).collect()[
        "id"
    ].tolist() == [1]

    first = tmp_path / "part-1.parquet"
    second = tmp_path / "part-2.parquet"
    embedded[embedded["id"] <= 2].write_parquet(first)
    embedded[embedded["id"] > 2].write_parquet(second)
    multi_file = session.read_parquet(str(tmp_path / "part-*.parquet"))
    assert multi_file._column("embedding").embedding is not None
    assert multi_file.collect()["id"].tolist() == [1, 2, 3, 4]


def test_embedding_validation_and_redaction_precede_source_execution() -> None:
    session, frame, provider = _frame()
    model = provider.specification
    execution_count = session.execution_count

    with pytest.raises(ValueError, match="already exists"):
        frame.embed_text(columns="title", into="title", model=model)
    with pytest.raises(UnsupportedOperationError, match="VARCHAR"):
        frame.embed_text(columns="id", into="embedding", model=model)
    with pytest.raises(ValueError, match="null_policy"):
        frame.embed_text(  # pyright: ignore[reportArgumentType]
            columns="title",
            into="embedding",
            model=model,
            null_policy="drop",  # pyright: ignore[reportArgumentType]
        )
    with pytest.raises(UnsupportedOperationError, match="matching"):
        frame.embed_text(columns="title", into="embedding", model=model).vector.search_text(
            "private query",
            column="embedding",
            model=_model("different/model"),
        )
    assert session.execution_count == execution_count

    semantic = frame.semantic.search(
        "private query",
        columns="title",
        model=model,
        k=1,
    )
    explained = semantic.explain("logical")
    assert "private query" not in explained
    assert '"query": "<redacted>"' in explained
    assert "transient_streaming_exact" in explained
    assert '"model_prepared": true' in explained


def test_embedding_profile_and_relational_composition_are_observable() -> None:
    _, frame, provider = _frame()
    model = provider.specification
    embedded = frame.embed_text(
        columns="title",
        into="embedding",
        model=model,
        batch_size=2,
    )

    profile = embedded.profile()
    assert profile.embedding_metrics is not None
    assert profile.embedding_metrics["document_rows"] == 4
    assert profile.embedding_metrics["document_batches"] == 2
    assert profile.embedding_metrics["text_bytes"] > 0
    assert profile.embedding_metrics["document_rows_per_second"] > 0
    assert profile.embedding_metrics["source_rows"] == 4
    assert profile.embedding_metrics["prepared_models"] == 1
    assert profile.embedding_metrics["preparation_seconds"] >= 0
    assert profile.embedding_metrics["provider_retries"] == 0

    projected = embedded[["id", "embedding"]]
    batches = list(projected.to_arrow_batches(batch_size=2))
    assert sum(batch.num_rows for batch in batches) == 4
    assert projected._column("embedding").embedding is not None

    combined = duckpd.concat([projected, projected], ignore_index=True)
    assert combined._column("embedding").embedding is not None
    renamed = embedded.rename(columns={"embedding": "features"})
    assert renamed._column("features").embedding is not None
    replaced = embedded.assign(embedding=lambda current: current["id"] + 1)
    assert replaced._column("embedding").embedding is None

    grouped = embedded.groupby("kind").size().collect()
    assert grouped["size"].sum() == 4

    assert combined.collect().shape == (8, 2)

    lookup = frame._session.from_pandas(
        pd.DataFrame({"id": [1, 2, 3, 4], "weight": [10, 20, 30, 40]})
    )
    joined = embedded.merge(lookup, on="id")[["id", "weight", "embedding"]]
    assert joined.collect()["weight"].tolist() == [10, 20, 30, 40]


def test_null_policy_and_invalid_provider_output_fail_explicitly() -> None:
    session = duckpd.connect()
    model = _model()
    provider = KeywordProvider(model)
    session.register_embedding_provider(model, provider)
    session.prepare_embedding_model(model)
    frame = session.from_pandas(pd.DataFrame({"text": ["AI", None]}))

    with pytest.raises(MaterializationError):
        frame.embed_text(columns="text", into="embedding", model=model).collect()
    result = frame.embed_text(
        columns="text",
        into="embedding",
        model=model,
        null_policy="empty",
    ).collect()
    assert result.shape == (2, 2)


class WrongDimensionProvider(KeywordProvider):
    def embed_documents(self, texts: Sequence[str]) -> pa.Array:  # type: ignore[type-arg]
        return _make_fixed_array([[1.0, 0.0] for _ in texts], 2)


def test_invalid_provider_shape_aborts_without_partial_result(tmp_path: Path) -> None:
    session = duckpd.connect()
    model = _model()
    provider = WrongDimensionProvider(model)
    session.register_embedding_provider(model, provider)
    session.prepare_embedding_model(model)
    frame = session.from_pandas(pd.DataFrame({"text": ["AI", "earnings"]}))

    with pytest.raises(MaterializationError):
        frame.embed_text(columns="text", into="embedding", model=model).collect()

    output = tmp_path / "invalid.parquet"
    with pytest.raises(MaterializationError):
        frame.embed_text(columns="text", into="embedding", model=model).write_parquet(output)
    assert not output.exists()
    output.write_bytes(b"previous-version")
    with pytest.raises(MaterializationError):
        frame.embed_text(columns="text", into="embedding", model=model).write_parquet(
            output,
            overwrite=True,
        )
    assert output.read_bytes() == b"previous-version"
    assert not list(tmp_path.glob(".duckpd_staging_*"))


def test_model_preparation_is_explicit_and_core_backend_is_optional() -> None:
    session = duckpd.connect()
    model = duckpd.embedding_model(
        "BAAI/bge-small-en-v1.5",
        revision="0123456789abcdef0123456789abcdef01234567",
        dimension=384,
    )
    frame = session.from_pandas(pd.DataFrame({"text": ["hello"]}))
    lazy = frame.embed_text(columns="text", into="embedding", model=model)
    assert session.execution_count == 0
    with pytest.raises(UnsupportedOperationError, match="prepare_embedding_model"):
        lazy.collect()


def test_embedding_model_spec_rejects_mutable_or_incomplete_identity() -> None:
    with pytest.raises(ValueError, match="model"):
        duckpd.embedding_model("", revision="abc123", dimension=3)
    with pytest.raises(ValueError, match="immutable"):
        duckpd.embedding_model("test/model", revision="latest", dimension=3)
    with pytest.raises(ValueError, match="backend"):
        FastEmbedProvider(_model())
    with pytest.raises(ValueError, match="dimension"):
        duckpd.embedding_model("test/model", revision="abc123", dimension=0)
    with pytest.raises(ValueError, match="pooling"):
        duckpd.embedding_model("test/model", revision="abc123", dimension=3, pooling="")


def test_provider_output_validation_rejects_every_unsafe_vector_shape() -> None:
    model = _model()
    valid = _make_fixed_array([[1.0, 0.0, 0.0]], 3)
    assert _validate_embedding_array(valid, model, expected_rows=1).equals(valid)

    with pytest.raises(ValueError, match="returned 1 rows"):
        _validate_embedding_array(valid, model, expected_rows=2)
    with pytest.raises(TypeError, match="FixedSizeListArray"):
        _validate_embedding_array(pa.array([[1.0, 0.0, 0.0]]), model, expected_rows=1)
    with pytest.raises(ValueError, match="dimension"):
        _validate_embedding_array(_make_fixed_array([[1.0, 0.0]], 2), model, expected_rows=1)
    with pytest.raises(ValueError, match="non-finite"):
        _validate_embedding_array(
            _make_fixed_array([[float("nan"), 0.0, 1.0]], 3),
            model,
            expected_rows=1,
        )
    with pytest.raises(ValueError, match="unnormalized"):
        _validate_embedding_array(_make_fixed_array([[1.0, 1.0, 1.0]], 3), model, expected_rows=1)
    with pytest.raises(ValueError, match="dimension"):
        _vectors_to_arrow([[1.0, 0.0]], model)
    with pytest.raises(ValueError, match="non-finite"):
        _vectors_to_arrow([[float("inf"), 0.0, 1.0]], model)
    with pytest.raises(ValueError, match="zero"):
        _vectors_to_arrow([[0.0, 0.0, 0.0]], model)


class FakeFastEmbedding:
    document_inputs: ClassVar[list[list[str]]] = []
    query_inputs: ClassVar[list[str]] = []

    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: str,
        providers: list[str],
    ) -> None:
        assert model_name == "test/fake-fastembed"
        assert providers == ["CPUExecutionProvider"]
        artifact = Path(cache_dir) / "model.onnx"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if not artifact.exists():
            artifact.write_bytes(b"verified-model")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_inputs.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]

    def query_embed(self, text: str) -> list[list[float]]:
        self.query_inputs.append(text)
        return [[0.0, 1.0, 0.0]]


def test_fastembed_provider_prepares_verifies_and_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("fastembed")
    module.TextEmbedding = FakeFastEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    model = duckpd.embedding_model(
        "test/fake-fastembed",
        revision="0123456789abcdef0123456789abcdef01234567",
        dimension=3,
        document_prefix="passage: ",
        query_prefix="query: ",
    )
    provider = FastEmbedProvider(model, cache_dir=tmp_path)
    assert provider.specification is model
    with pytest.raises(UnsupportedOperationError, match="not prepared"):
        provider.embed_documents(["hello"])

    prepared = provider.prepare()
    assert provider.prepare() is prepared
    assert prepared.artifact_digest == _directory_digest(Path(prepared.cache_path or ""))
    assert prepared.execution_providers == ("CPUExecutionProvider",)
    assert provider.embed_documents(["hello"]).to_pylist() == [[1.0, 0.0, 0.0]]
    assert provider.embed_query("hello") == EmbeddedQuery((0.0, 1.0, 0.0), model.fingerprint)
    assert FakeFastEmbedding.document_inputs[-1] == ["passage: hello"]
    assert FakeFastEmbedding.query_inputs[-1] == "query: hello"

    assert FastEmbedProvider(model, cache_dir=tmp_path).prepare().artifact_digest == (
        prepared.artifact_digest
    )
    (Path(prepared.cache_path or "") / "model.onnx").write_bytes(b"changed")
    with pytest.raises(UnsupportedOperationError, match="changed"):
        FastEmbedProvider(model, cache_dir=tmp_path).prepare()


def test_fastembed_cache_rejects_external_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (root / "unsafe").symlink_to(outside)
    with pytest.raises(UnsupportedOperationError, match="symbolic links"):
        _directory_digest(root)


def test_missing_optional_embedding_backend_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "fastembed", None)  # type: ignore[arg-type]
    session = duckpd.connect()
    model = duckpd.embedding_model(
        "BAAI/bge-small-en-v1.5",
        revision="0123456789abcdef0123456789abcdef01234567",
        dimension=384,
    )
    with pytest.raises(UnsupportedOperationError, match="duckpd\\[embeddings\\]"):
        session.prepare_embedding_model(model)
