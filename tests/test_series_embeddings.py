from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from threading import Lock
from time import sleep
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import duckpd
from duckpd._logical import SeriesRepresentationPlan
from duckpd.errors import MaterializationError, UnsupportedOperationError
from duckpd.frame import DataFrame
from duckpd.series_embeddings import (
    SeriesColumnSpec,
    SeriesNormalization,
    SeriesWindowSpec,
    ZeroScalePolicy,
)


def _native_representation(
    *, data_contract: str = "market/simple-return/v1"
) -> duckpd.SeriesRepresentationSpec:
    return duckpd.series_representation(
        window=2,
        channels=("simple_return",),
        sampling="observations",
        step="PT1M",
        data_contract=data_contract,
        normalization="center",
        unit_norm=True,
        zero_scale="null",
    )


def _learned_model() -> duckpd.EmbeddingModelSpec:
    return duckpd.embedding_model(
        "research/return-encoder",
        revision="immutable-revision-v1",
        backend="custom",
        dimension=3,
        normalize=False,
        pooling="mean-valid-v1",
        input=duckpd.series_embedding_input(
            length=2,
            channels=("simple_return",),
            roles=("target",),
            normalization="none",
        ),
    )


def _learned_representation(
    *,
    channels: tuple[str, ...] = ("simple_return",),
    roles: tuple[duckpd.SeriesChannelRole, ...] = ("target",),
    normalization: SeriesNormalization = "none",
    unit_norm: bool = False,
    zero_scale: ZeroScalePolicy = "error",
) -> duckpd.SeriesRepresentationSpec:
    model = duckpd.embedding_model(
        "research/return-encoder",
        revision="immutable-revision-v1",
        backend="custom",
        dimension=3,
        normalize=unit_norm,
        pooling="mean-valid-v1",
        input=duckpd.series_embedding_input(
            length=2,
            channels=channels,
            roles=roles,
            normalization="provider-none-v1",
        ),
    )
    return duckpd.series_representation(
        window=2,
        channels=channels,
        sampling="observations",
        data_contract="market/learned-input/v1",
        normalization=normalization,
        unit_norm=unit_norm,
        zero_scale=zero_scale,
        encoder=model,
    )


class _RecordingSeriesProvider:
    def __init__(
        self,
        model: duckpd.EmbeddingModelSpec,
        *,
        thread_safe: bool = True,
        delay: float = 0.0,
        output_mode: str = "valid",
        prepared_override: dict[str, object] | None = None,
    ) -> None:
        self._model = model
        self._thread_safe = thread_safe
        self._delay = delay
        self.output_mode = output_mode
        self.prepared_override = prepared_override or {}
        self.prepare_calls = 0
        self.batches: list[pa.RecordBatch] = []
        self.close_calls = 0
        self.max_active = 0
        self._active = 0
        self._lock = Lock()

    @property
    def specification(self) -> duckpd.EmbeddingModelSpec:
        return self._model

    @property
    def thread_safe(self) -> bool:
        return self._thread_safe

    def prepare(self) -> duckpd.PreparedModelInfo:
        self.prepare_calls += 1
        values: dict[str, object] = {
            "model_fingerprint": self._model.fingerprint,
            "backend": self._model.backend,
            "cache_path": None,
            "artifact_digest": "a" * 64,
            "execution_providers": ("CPUExecutionProvider",),
        }
        values.update(self.prepared_override)
        return duckpd.PreparedModelInfo(**values)  # type: ignore[arg-type]

    def embed_windows(self, batch: pa.RecordBatch) -> pa.Array[Any]:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self._delay:
                sleep(self._delay)
            self.batches.append(batch)
            batch_columns = cast("list[pa.Array[Any]]", cast("Any", batch).columns)
            rows_by_channel = [
                cast("list[list[float]]", column.to_pylist()) for column in batch_columns
            ]
            vectors = [
                [
                    float(rows_by_channel[0][row][0]),
                    float(rows_by_channel[0][row][-1]),
                    float(
                        sum(
                            value for channel_rows in rows_by_channel for value in channel_rows[row]
                        )
                    ),
                ]
                for row in range(batch.num_rows)
            ]
            if self.output_mode == "rows":
                vectors = vectors[:-1]
            elif self.output_mode == "dimension":
                vectors = [row[:2] for row in vectors]
            elif self.output_mode == "nan":
                vectors[0][0] = float("nan")
            elif self.output_mode == "zero":
                vectors = [[0.0] * self._model.dimension for _ in vectors]
            if self.output_mode == "wrong_type":
                return pa.array(vectors, type=pa.list_(pa.float32()))
            if self.output_mode == "null":
                return pa.array(
                    [None for _ in vectors],
                    type=pa.list_(pa.float32(), self._model.dimension),
                )
            if self.output_mode == "child_null":
                vectors[0][0] = None  # type: ignore[assignment]
            return pa.array(
                vectors,
                type=pa.list_(pa.float32(), len(vectors[0]) if vectors else self._model.dimension),
            )
        finally:
            with self._lock:
                self._active -= 1

    def close(self) -> None:
        self.close_calls += 1


def _attach_series_metadata(
    frame: DataFrame,
    label: str,
    representation: duckpd.SeriesRepresentationSpec,
) -> DataFrame:
    columns = tuple(
        replace(column, series=SeriesColumnSpec(representation))
        if column.label == label
        else column
        for column in frame._plan.metadata.columns
    )
    metadata = replace(frame._plan.metadata, columns=columns)
    return DataFrame(frame._session, replace(frame._plan, metadata=metadata))


def test_series_representation_is_canonical_and_round_trips() -> None:
    specification = _native_representation()

    assert specification.dimension == 2
    assert specification.step == "PT1M"
    assert specification == duckpd.SeriesRepresentationSpec.from_dict(specification.to_dict())
    assert (
        specification.fingerprint
        == duckpd.SeriesRepresentationSpec.from_dict(specification.to_dict()).fingerprint
    )
    assert (
        specification.fingerprint
        == "950b04612f9f38fd6a12731a941229b34bab55cc8e23d7b5707394f87a8b3652"
    )


def test_learned_series_model_is_part_of_representation_identity() -> None:
    model = _learned_model()
    specification = duckpd.series_representation(
        window=2,
        channels=("simple_return",),
        sampling="observations",
        data_contract="market/simple-return/v1",
        encoder=model,
    )

    assert model == duckpd.EmbeddingModelSpec.from_dict(model.to_dict())
    assert len(model.fingerprint) == 64
    assert specification.dimension == 3
    assert specification == duckpd.SeriesRepresentationSpec.from_dict(specification.to_dict())
    assert specification.fingerprint != _native_representation().fingerprint


@pytest.mark.parametrize(
    "factory",
    [
        lambda: duckpd.series_representation(
            window=True,
            channels=("value",),
            sampling="observations",
            data_contract="value/v1",
        ),
        lambda: duckpd.series_representation(
            window=2,
            channels=("value", "value"),
            sampling="observations",
            data_contract="value/v1",
        ),
        lambda: duckpd.series_representation(
            window=2,
            channels=("value",),
            sampling="fixed_grid",
            data_contract="value/v1",
        ),
        lambda: duckpd.series_representation(
            window=2,
            channels=(),
            sampling="observations",
            data_contract="value/v1",
        ),
        lambda: duckpd.series_representation(
            window=2,
            channels=("value",),
            sampling="unknown",  # pyright: ignore[reportArgumentType]
            data_contract="value/v1",
        ),
        lambda: duckpd.series_representation(
            window=2,
            channels=("value",),
            sampling="observations",
            data_contract="",
        ),
        lambda: duckpd.series_representation(
            window=2,
            channels=("value",),
            sampling="observations",
            data_contract="value/v1",
            normalization="unknown",  # pyright: ignore[reportArgumentType]
        ),
        lambda: duckpd.series_representation(
            window=2,
            channels=("value",),
            sampling="observations",
            data_contract="value/v1",
            unit_norm="yes",  # pyright: ignore[reportArgumentType]
        ),
        lambda: duckpd.series_representation(
            window=2,
            channels=("value",),
            sampling="observations",
            data_contract="value/v1",
            zero_scale="zero",  # pyright: ignore[reportArgumentType]
        ),
        lambda: duckpd.series_representation(
            window=2,
            channels=("value",),
            sampling="observations",
            step="PT0S",
            data_contract="value/v1",
        ),
        lambda: duckpd.series_representation(
            window=3,
            channels=("simple_return",),
            sampling="observations",
            data_contract="value/v1",
            encoder=_learned_model(),
        ),
        lambda: duckpd.series_representation(
            window=2,
            channels=("other",),
            sampling="observations",
            data_contract="value/v1",
            encoder=_learned_model(),
        ),
    ],
)
def test_series_representation_rejects_ambiguous_contracts(
    factory: Callable[[], object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_series_model_rejects_mutable_or_incomplete_identity() -> None:
    model = _learned_model()
    input_spec = model.input
    assert isinstance(input_spec, duckpd.SeriesEmbeddingInputSpec)
    invalid = (
        lambda: replace(model, model=""),
        lambda: replace(model, revision="main"),
        lambda: replace(model, dimension=True),
        lambda: replace(model, backend="fastembed"),
        lambda: replace(model, input=replace(input_spec, length=0)),
        lambda: replace(model, input=replace(input_spec, channels=("",))),
        lambda: replace(model, input=replace(input_spec, channels=("x", "x"))),
    )
    for factory in invalid:
        with pytest.raises((TypeError, ValueError)):
            factory()


def test_serialized_series_contract_rejects_schema_drift() -> None:
    specification = _native_representation()
    payload = specification.to_dict()
    corruptions = (
        {**payload, "unknown": True},
        {**payload, "schema_version": 2},
        {**payload, "output_type": "DOUBLE"},
        {**payload, "channels": "simple_return"},
        {**payload, "encoder": "model"},
        {**payload, "layout": "unknown"},
        {**payload, "dimension": 99},
    )
    for corrupted in corruptions:
        with pytest.raises((TypeError, ValueError)):
            duckpd.SeriesRepresentationSpec.from_dict(corrupted)

    learned = duckpd.series_representation(
        window=2,
        channels=("simple_return",),
        sampling="observations",
        data_contract="value/v1",
        encoder=_learned_model(),
    ).to_dict()
    with pytest.raises(ValueError, match="native layout"):
        duckpd.SeriesRepresentationSpec.from_dict({**learned, "layout": "native"})
    with pytest.raises(ValueError, match="unknown fields"):
        duckpd.EmbeddingModelSpec.from_dict({**_learned_model().to_dict(), "extra": 1})


def test_window_column_and_query_metadata_validate_boundaries() -> None:
    fixed = duckpd.SeriesWindowSpec(
        2,
        sampling="fixed_grid",
        step=timedelta(days=1, hours=2, minutes=3, seconds=4, microseconds=500_000),
        origin="event",
    )
    assert fixed.step == "P1DT2H3M4.5S"
    assert duckpd.SeriesColumnSpec(_native_representation()).fingerprint == (
        _native_representation().fingerprint
    )
    for factory in (
        lambda: duckpd.SeriesWindowSpec(0),
        lambda: duckpd.SeriesWindowSpec(2, sampling="bad"),  # pyright: ignore[reportArgumentType]
        lambda: duckpd.SeriesWindowSpec(2, sampling="fixed_grid"),
        lambda: duckpd.SeriesWindowSpec(2, origin="bad"),  # pyright: ignore[reportArgumentType]
        lambda: duckpd.SeriesColumnSpec(_native_representation(), origin="bad"),  # pyright: ignore[reportArgumentType]
        lambda: duckpd.EmbeddedSeriesQuery((), "a" * 64),
        lambda: duckpd.EmbeddedSeriesQuery((float("nan"),), "a" * 64),
        lambda: duckpd.EmbeddedSeriesQuery((1.0,), "invalid"),
    ):
        with pytest.raises(ValueError):
            factory()


def test_direct_aliases_preserve_series_metadata_and_transforms_drop_it() -> None:
    session = duckpd.connect()
    frame = _attach_series_metadata(
        session.from_pandas(pd.DataFrame({"value": [1.0, 2.0]})),
        "value",
        _native_representation(),
    )

    aliased = frame.assign(alias=frame["value"])
    transformed = frame.assign(changed=frame["value"] + 1)
    renamed = frame.rename(columns={"value": "renamed"})
    as_frame = frame["value"].to_frame("alias")

    assert aliased._column("alias").series == frame._column("value").series
    assert transformed._column("changed").series is None
    assert renamed._column("renamed").series == frame._column("value").series
    assert as_frame._column("alias").series == frame._column("value").series
    assert session.execution_count == 0


def test_join_and_asof_payloads_preserve_series_metadata() -> None:
    session = duckpd.connect()
    vectors = _attach_series_metadata(
        session.sql(
            "SELECT * FROM (VALUES "
            "(1, TIMESTAMP '2024-01-01', [1.0, 2.0]::FLOAT[2]), "
            "(2, TIMESTAMP '2024-01-02', [2.0, 3.0]::FLOAT[2])"
            ") AS t(id, ts, vector)",
            order_by="ts",
        ),
        "vector",
        _native_representation(),
    )
    keys = session.from_pandas(pd.DataFrame({"id": [1, 2]}))
    events = session.sql(
        "SELECT * FROM (VALUES "
        "(1, TIMESTAMP '2024-01-01 12:00:00'), "
        "(2, TIMESTAMP '2024-01-02 12:00:00')"
        ") AS t(id, ts)",
        order_by="ts",
    )

    merged = keys.merge(vectors, on="id")
    aligned = duckpd.merge_asof(events, vectors, on="ts", by="id")

    assert merged._column("vector").series == vectors._column("vector").series
    assert aligned._column("vector").series == vectors._column("vector").series
    assert session.execution_count == 0


def test_series_metadata_round_trips_through_managed_sinks(tmp_path: Path) -> None:
    session = duckpd.connect()
    frame = _attach_series_metadata(
        session.sql("SELECT 1 AS id, [1.0, 2.0]::FLOAT[2] AS vector"),
        "vector",
        _native_representation(),
    )
    path = tmp_path / "vectors.parquet"

    frame.write_parquet(path)
    parquet = session.read_parquet(path)
    frame.save_as_table("series_vectors")
    table = session.table("series_vectors")

    assert parquet._column("vector").series == frame._column("vector").series
    assert table._column("vector").series == frame._column("vector").series


def test_concat_requires_compatible_series_representations() -> None:
    session = duckpd.connect()
    left = _attach_series_metadata(
        session.from_pandas(pd.DataFrame({"value": [1.0]})),
        "value",
        _native_representation(),
    )
    compatible = _attach_series_metadata(
        session.from_pandas(pd.DataFrame({"value": [2.0]})),
        "value",
        _native_representation(),
    )
    conflicting = _attach_series_metadata(
        session.from_pandas(pd.DataFrame({"value": [3.0]})),
        "value",
        _native_representation(data_contract="other/v1"),
    )

    result = duckpd.concat([left, compatible])

    assert result._column("value").series == left._column("value").series
    with pytest.raises(ValueError, match="incompatible series representations"):
        duckpd.concat([left, conflicting])
    assert session.execution_count == 0


def test_typed_series_query_checks_representation_before_execution() -> None:
    session = duckpd.connect()
    frame = _attach_series_metadata(
        session.sql("SELECT 1 AS id, [1.0, 2.0]::FLOAT[2] AS vector"),
        "vector",
        _native_representation(),
    )
    series = frame._column("vector").series
    assert series is not None
    query = duckpd.EmbeddedSeriesQuery((1.0, 2.0), series.fingerprint)

    result = frame.vector.search(query, column="vector", metric="l2")
    wrong_query = duckpd.EmbeddedSeriesQuery(
        (1.0, 2.0),
        _native_representation(data_contract="other/v1").fingerprint,
    )
    with pytest.raises(UnsupportedOperationError, match="fingerprint"):
        frame.vector.search(wrong_query, column="vector")
    with pytest.raises(UnsupportedOperationError, match="fingerprint"):
        frame["vector"].vector.distance(wrong_query)
    assert session.execution_count == 0

    assert result.collect()["id"].tolist() == [1]  # pyright: ignore[reportUnknownMemberType]
    assert session.execution_count == 1


def test_search_series_matches_independent_window_and_distance_oracles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = duckpd.connect()
    source_data = pd.DataFrame(
        {
            "row": list(range(6)),
            "x": [1.0, 2.0, 4.0, 7.0, 11.0, 16.0],
            "y": [10.0, 7.0, 5.0, 4.0, 2.0, -1.0],
        }
    )
    source = session.from_pandas(source_data, order_by="row")
    windows = source.assign(
        x_window=lambda frame: frame["x"].rolling(3).to_array(),
        y_window=lambda frame: frame["y"].rolling(3).to_array(),
    )
    representation = duckpd.series_representation(
        window=3,
        channels=("x", "y"),
        sampling="observations",
        data_contract="xy/v1",
        normalization="center",
        unit_norm=True,
    )
    candidates = windows.embed_series(
        columns={"x": "x_window", "y": "y_window"},
        into="vector",
        representation=representation,
    )
    raw_query: dict[str, list[float]] = {
        "x": [2.0, 4.0, 7.0],
        "y": [7.0, 5.0, 4.0],
    }

    searched = candidates[candidates["row"] >= 2].vector.search_series(
        raw_query,
        column="vector",
        representation=representation,
        metric="l2",
        k=4,
        tie_breaker="row",
    )
    raw_query["x"][0] = 999.0

    operations = json.loads(searched.explain(mode="json"))["execution_boundaries"][
        "embedding_operations"
    ]
    search_operation = next(item for item in operations if item["operation"] == "search_series")
    assert search_operation["representation_fingerprint"] == representation.fingerprint
    assert search_operation["query"] == "<redacted>"
    assert session.execution_count == 0

    output = tmp_path / "series-search.parquet"

    def fail_if_materialized(_frame: DataFrame) -> None:
        raise AssertionError("direct search sink must not materialize through pandas")

    with monkeypatch.context() as patch:
        patch.setattr(DataFrame, "collect", fail_if_materialized)
        patch.setattr(DataFrame, "to_pandas", fail_if_materialized)
        searched.write_parquet(output)
    result = session.read_parquet(output).collect()

    def represent(channels: list[list[float]]) -> tuple[float, ...]:
        centered: list[float] = []
        for channel in channels:
            mean = sum(channel) / len(channel)
            centered.extend(value - mean for value in channel)
        norm = sum(value * value for value in centered) ** 0.5
        return tuple(float(np.float32(value / norm)) for value in centered)

    query_vector = represent([[2.0, 4.0, 7.0], [7.0, 5.0, 4.0]])
    oracle: list[tuple[float, int]] = []
    for endpoint in range(2, len(source_data)):
        start = endpoint - 2
        vector = represent(
            [
                source_data["x"].iloc[start : endpoint + 1].tolist(),
                source_data["y"].iloc[start : endpoint + 1].tolist(),
            ]
        )
        distance = (
            sum(
                (value - query_value) ** 2
                for value, query_value in zip(vector, query_vector, strict=True)
            )
            ** 0.5
        )
        oracle.append((distance, endpoint))
    expected = sorted(oracle)

    assert result["row"].tolist() == [row for _, row in expected]
    np.testing.assert_allclose(
        result["_distance"].to_numpy(),
        [distance for distance, _ in expected],
        rtol=1e-6,
        atol=1e-7,
    )


def test_series_query_eager_form_and_planning_failures() -> None:
    representation = duckpd.series_representation(
        window=2,
        channels=("value",),
        sampling="observations",
        data_contract="value/v1",
        normalization="center",
        unit_norm=True,
    )
    session = duckpd.connect()
    frame = _attach_series_metadata(
        session.sql(
            "SELECT * FROM (VALUES "
            "(1, [-0.70710677, 0.70710677]::FLOAT[2]), "
            "(2, [0.70710677, -0.70710677]::FLOAT[2])"
            ") AS t(id, vector)"
        ),
        "vector",
        representation,
    )

    eager = session.embed_series_query(
        {"value": [2.0, 4.0]},
        representation=representation,
    )
    np.testing.assert_allclose(
        eager.values,
        [-0.70710677, 0.70710677],
        rtol=1e-6,
        atol=1e-7,
    )
    assert session.execution_count == 1
    assert frame.vector.search(eager, column="vector", metric="l2").collect()["id"].tolist() == [
        1,
        2,
    ]

    incompatible = replace(representation, data_contract="other/v1")
    invalid_calls = (
        lambda: frame.vector.search_series({}, column="vector"),
        lambda: frame.vector.search_series({"value": [1.0]}, column="vector"),
        lambda: frame.vector.search_series({"value": [1.0, float("nan")]}, column="vector"),
        lambda: frame.vector.search_series(
            {"value": [1.0, 2.0]},
            column="vector",
            representation=incompatible,
        ),
        lambda: session.sql("SELECT [1.0, 2.0]::FLOAT[2] AS vector").vector.search_series(
            {"value": [1.0, 2.0]}, column="vector"
        ),
    )
    for call in invalid_calls:
        with pytest.raises((TypeError, ValueError, UnsupportedOperationError)):
            call()

    zero_scale = _attach_series_metadata(
        session.sql("SELECT 1 AS id, [1.0, 2.0]::FLOAT[2] AS vector"),
        "vector",
        replace(representation, normalization="zscore", unit_norm=False),
    ).vector.search_series({"value": [3.0, 3.0]}, column="vector")
    executions_before_failure = session.execution_count
    with pytest.raises(MaterializationError):
        zero_scale.collect()
    assert session.execution_count == executions_before_failure + 1


@pytest.mark.parametrize(
    ("normalization", "expected"),
    [
        ("none", [1.0, 2.0, 10.0, 14.0]),
        ("center", [-0.5, 0.5, -2.0, 2.0]),
        ("zscore", [-1.0, 1.0, -1.0, 1.0]),
    ],
)
def test_embed_series_native_multichannel_normalization_is_lazy_and_ordered(
    normalization: str,
    expected: list[float],
) -> None:
    session = duckpd.connect()
    source = session.from_pandas(
        pd.DataFrame(
            {
                "row": [1, 2, 3],
                "x": [1.0, 2.0, 4.0],
                "y": [10.0, 14.0, 20.0],
            }
        ),
        order_by="row",
    )
    windows = source.assign(
        x_window=lambda frame: frame["x"].rolling(2).to_array(),
        y_window=lambda frame: frame["y"].rolling(2).to_array(),
    )
    representation = duckpd.series_representation(
        window=2,
        channels=("x", "y"),
        sampling="observations",
        data_contract="xy/v1",
        normalization=normalization,  # pyright: ignore[reportArgumentType]
    )

    embedded = windows.embed_series(
        columns={"y": "y_window", "x": "x_window"},
        into="vector",
        representation=representation,
        batch_size=17,
    )

    assert isinstance(embedded._plan, SeriesRepresentationPlan)
    assert embedded._plan.channels[0][0] == "x"
    assert embedded._plan.batch_size == 17
    assert embedded._plan.metadata.ordering == windows._plan.metadata.ordering
    assert embedded._plan.metadata.row_identity == windows._plan.metadata.row_identity
    assert embedded._column("vector").duckdb_type == "FLOAT[4]"
    assert embedded._column("vector").series == SeriesColumnSpec(representation)
    assert session.execution_count == 0
    operation = json.loads(embedded.explain(mode="json"))["execution_boundaries"][
        "embedding_operations"
    ][0]
    assert operation["operation"] == "embed_series"
    assert operation["representation_fingerprint"] == representation.fingerprint
    assert session.execution_count == 0

    result = embedded.collect()["vector"]  # pyright: ignore[reportUnknownMemberType]
    assert pd.isna(result.iloc[0])
    np.testing.assert_allclose(result.iloc[1], expected, rtol=1e-6, atol=1e-7)
    assert session.execution_count == 1


def test_embed_series_applies_unit_norm_after_channel_major_flattening() -> None:
    frame = duckpd.from_pandas(
        pd.DataFrame({"row": [1, 2], "x": [1.0, 2.0], "y": [10.0, 14.0]}),
        order_by="row",
    )
    windows = frame.assign(
        x_window=lambda current: current["x"].rolling(2).to_array(),
        y_window=lambda current: current["y"].rolling(2).to_array(),
    )
    representation = duckpd.series_representation(
        window=2,
        channels=("x", "y"),
        sampling="observations",
        data_contract="xy/v1",
        normalization="center",
        unit_norm=True,
    )

    result = windows.embed_series(
        columns={"x": "x_window", "y": "y_window"},
        into="vector",
        representation=representation,
    ).collect()["vector"]  # pyright: ignore[reportUnknownMemberType]

    expected = np.array([-0.5, 0.5, -2.0, 2.0], dtype=np.float64)
    expected /= sum(float(value * value) for value in expected) ** 0.5
    np.testing.assert_allclose(result.iloc[1], expected, rtol=1e-6, atol=1e-7)


def test_embed_series_null_and_zero_scale_policies() -> None:
    session = duckpd.connect()
    source = session.from_pandas(
        pd.DataFrame({"row": [1, 2, 3], "value": [5.0, 5.0, 5.0]}),
        order_by="row",
    )
    windows = source.assign(window=lambda frame: frame["value"].rolling(2).to_array())
    null_representation = duckpd.series_representation(
        window=2,
        channels=("value",),
        sampling="observations",
        data_contract="constant/v1",
        normalization="zscore",
        zero_scale="null",
    )
    error_representation = replace(null_representation, zero_scale="error")

    propagated = windows.embed_series(
        columns={"value": "window"},
        into="vector",
        representation=null_representation,
    ).collect()
    assert propagated["vector"].isna().all()  # pyright: ignore[reportUnknownMemberType]

    with pytest.raises(MaterializationError):
        windows.embed_series(
            columns={"value": "window"},
            into="vector",
            representation=error_representation,
        ).collect()
    with pytest.raises(MaterializationError):
        windows.embed_series(
            columns={"value": "window"},
            into="vector",
            representation=null_representation,
            null_policy="error",
        ).collect()


def test_embed_series_rejects_invalid_contracts_before_execution() -> None:
    session = duckpd.connect()
    source = session.from_pandas(
        pd.DataFrame({"row": [1, 2], "value": [1.0, 2.0]}),
        order_by="row",
    )
    windows = source.assign(window=lambda frame: frame["value"].rolling(2).to_array())
    representation = duckpd.series_representation(
        window=2,
        channels=("value",),
        sampling="observations",
        data_contract="value/v1",
    )
    fixed_grid = replace(representation, sampling="fixed_grid", step="PT1M")

    invalid_calls = (
        lambda: windows.embed_series(
            columns={},
            into="vector",
            representation=representation,
        ),
        lambda: windows.embed_series(
            columns={"other": "window"},
            into="vector",
            representation=representation,
        ),
        lambda: windows.embed_series(
            columns={"value": "value"},
            into="vector",
            representation=representation,
        ),
        lambda: windows.embed_series(
            columns={"value": "window"},
            into="value",
            representation=representation,
        ),
        lambda: windows.embed_series(
            columns={"value": "window"},
            into="vector",
            representation=fixed_grid,
        ),
        lambda: windows.embed_series(
            columns={"value": "window"},
            into="vector",
            representation=representation,
            batch_size=0,
        ),
        lambda: windows.embed_series(
            columns={"value": "window"},
            into="vector",
            representation=representation,
            null_policy="invalid",  # pyright: ignore[reportArgumentType]
        ),
    )
    for call in invalid_calls:
        with pytest.raises((TypeError, ValueError, UnsupportedOperationError)):
            call()
    assert session.execution_count == 0


@pytest.mark.parametrize(
    "array_sql",
    (
        "[1.0, NULL]::FLOAT[2]",
        "[1.0, 'Infinity'::FLOAT]::FLOAT[2]",
    ),
)
def test_embed_series_rejects_invalid_array_children_at_execution(
    array_sql: str,
) -> None:
    session = duckpd.connect()
    frame = session.sql(f"SELECT {array_sql} AS window")
    columns = tuple(
        replace(
            column,
            series_window=SeriesWindowSpec(
                window=2,
                order_by=("asserted-row-order",),
                origin="application_asserted",
            ),
        )
        if column.label == "window"
        else column
        for column in frame._plan.metadata.columns
    )
    frame = DataFrame(
        session,
        replace(frame._plan, metadata=replace(frame._plan.metadata, columns=columns)),
    )
    representation = duckpd.series_representation(
        window=2,
        channels=("value",),
        sampling="observations",
        data_contract="value/v1",
    )

    embedded = frame.embed_series(
        columns={"value": "window"},
        into="vector",
        representation=representation,
    )

    assert session.execution_count == 0
    with pytest.raises(MaterializationError):
        embedded.collect()


def test_native_series_representation_round_trips_through_direct_parquet_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = duckpd.connect()
    source = session.from_pandas(
        pd.DataFrame({"row": [1, 2, 3], "value": [1.0, 2.0, 4.0]}),
        order_by="row",
    )
    windows = source.assign(window=lambda frame: frame["value"].rolling(2).to_array())
    representation = duckpd.series_representation(
        window=2,
        channels=("value",),
        sampling="observations",
        data_contract="value/v1",
        normalization="center",
    )
    embedded = windows.embed_series(
        columns={"value": "window"},
        into="vector",
        representation=representation,
    )
    output = tmp_path / "series-vectors.parquet"

    def fail_if_materialized(_frame: DataFrame) -> None:
        raise AssertionError("direct sink must not materialize through pandas")

    with monkeypatch.context() as patch:
        patch.setattr(DataFrame, "collect", fail_if_materialized)
        patch.setattr(DataFrame, "to_pandas", fail_if_materialized)
        embedded.write_parquet(output)
    restored = session.read_parquet(output)

    assert restored._column("vector").series == SeriesColumnSpec(
        representation,
        origin="sidecar",
    )
    values = restored.collect()["vector"]  # pyright: ignore[reportUnknownMemberType]
    np.testing.assert_array_equal(values.iloc[1], np.array([-0.5, 0.5], dtype=np.float32))


def _learned_windows(
    session: duckpd.Session,
    representation: duckpd.SeriesRepresentationSpec,
    *,
    batch_size: int = 2,
) -> DataFrame:
    source = session.from_pandas(
        pd.DataFrame(
            {
                "row": range(5),
                "target": [1.0, 2.0, 3.0, 4.0, 5.0],
                "past": [10.0, 20.0, 30.0, 40.0, 50.0],
                "known": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        ),
        order_by="row",
    )
    windows = source.assign(
        target_window=lambda frame: frame["target"].rolling(2).to_array(),
        past_window=lambda frame: frame["past"].rolling(2).to_array(),
        known_window=lambda frame: frame["known"].rolling(2).to_array(),
    )
    return windows.embed_series(
        columns={
            "target": "target_window",
            "past": "past_window",
            "known": "known_window",
        },
        into="vector",
        representation=representation,
        batch_size=batch_size,
    )


def test_learned_series_model_roles_are_canonical_and_strict() -> None:
    representation = _learned_representation(
        channels=("target", "past", "known"),
        roles=("target", "past_covariate", "known_future_covariate"),
    )
    model = representation.encoder
    assert model is not None

    restored = duckpd.EmbeddingModelSpec.from_dict(json.loads(json.dumps(model.to_dict())))

    assert restored == model
    assert restored.fingerprint == model.fingerprint
    input_spec = model.input
    assert isinstance(input_spec, duckpd.SeriesEmbeddingInputSpec)
    with pytest.raises(ValueError, match="align"):
        replace(model, input=replace(input_spec, roles=("target",)))
    with pytest.raises(ValueError, match="valid series roles"):
        replace(
            model,
            input=replace(input_spec, roles=("target", "invalid", "known_future_covariate")),
        )
    with pytest.raises(ValueError, match="target"):
        replace(
            model,
            input=replace(
                input_spec,
                roles=("past_covariate", "past_covariate", "known_future_covariate"),
            ),
        )


def test_learned_provider_lifecycle_is_explicit_bounded_and_masked() -> None:
    representation = _learned_representation(
        channels=("target", "past", "known"),
        roles=("target", "past_covariate", "known_future_covariate"),
    )
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model)
    session = duckpd.connect()
    session.register_embedding_provider(model, provider)
    embedded = _learned_windows(session, representation, batch_size=2)

    operation = json.loads(embedded.explain(mode="json"))["execution_boundaries"][
        "embedding_operations"
    ][0]
    assert "__duckpd_embed_series_" in embedded.explain(mode="sql")
    assert operation["backend"] == "custom"
    assert operation["model_prepared"] is False
    assert operation["input_roles"] == [
        "target",
        "past_covariate",
        "known_future_covariate",
    ]
    assert provider.prepare_calls == 0
    assert provider.batches == []
    executions_before_failure = session.execution_count
    with pytest.raises(UnsupportedOperationError, match="not prepared"):
        embedded.collect()
    assert session.execution_count == executions_before_failure
    assert provider.batches == []

    prepared = session.prepare_embedding_model(model)
    assert prepared.model_fingerprint == model.fingerprint
    assert prepared.backend == "custom"
    assert prepared.execution_providers == ("CPUExecutionProvider",)
    assert session.prepare_embedding_model(model) is prepared
    assert provider.prepare_calls == 1

    result = embedded.collect()

    assert pd.isna(result["vector"].iloc[0])
    np.testing.assert_array_equal(
        result["vector"].iloc[1],
        np.array([1.0, 2.0, 234.0], dtype=np.float32),
    )
    assert [batch.num_rows for batch in provider.batches] == [2, 2]
    for batch in provider.batches:
        assert batch.schema.names == ["target", "past", "known"]
        assert batch.schema.metadata is not None
        assert batch.schema.metadata[b"duckpd.mask_semantics"] == b"complete_rows_only-v1"
        fields = cast("list[Any]", list(batch.schema))
        assert [field.metadata[b"duckpd.channel_role"] for field in fields] == [
            b"target",
            b"past_covariate",
            b"known_future_covariate",
        ]
        assert all(field.type == pa.list_(pa.float32(), 2) for field in fields)
        columns = cast("list[pa.Array[Any]]", cast("Any", batch).columns)
        assert all(column.null_count == 0 for column in columns)
    assert session._embedding_metrics["series_complete_rows"] == 4
    assert session._embedding_metrics["series_propagated_null_rows"] == 1
    assert session._embedding_metrics["series_max_provider_batch_rows"] == 2

    eager = session.embed_series_query(
        {
            "target": [1.0, 2.0],
            "past": [10.0, 20.0],
            "known": [100.0, 101.0],
        },
        representation=representation,
    )
    assert eager.values == (1.0, 2.0, 234.0)


def test_learned_series_query_matches_corpus_once_per_execution() -> None:
    representation = _learned_representation(
        channels=("target", "past", "known"),
        roles=("target", "past_covariate", "known_future_covariate"),
        normalization="center",
        unit_norm=True,
    )
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model)
    session = duckpd.connect()
    session.register_embedding_provider(model, provider)
    session.prepare_embedding_model(model)
    candidates = _learned_windows(session, representation, batch_size=2)
    raw_query = {
        "target": [2.0, 3.0],
        "past": [20.0, 30.0],
        "known": [101.0, 102.0],
    }
    searched = candidates[candidates["row"] >= 1].vector.search_series(
        raw_query,
        column="vector",
        metric="l2",
        k=4,
        tie_breaker="row",
    )
    raw_query["target"][0] = 999.0

    first = searched.collect()
    first_query_calls = session._embedding_metrics["series_query_provider_calls"]
    first_corpus_calls = session._embedding_metrics["series_corpus_provider_calls"]
    second = searched.collect()

    assert first["row"].iloc[0] == 1
    assert second["row"].tolist() == first["row"].tolist()
    assert first_query_calls == 1
    assert session._embedding_metrics["series_query_provider_calls"] == 2
    assert session._embedding_metrics["series_corpus_provider_calls"] == 2 * first_corpus_calls
    assert session._embedding_metrics["series_query_cache_entries"] == 1


@pytest.mark.parametrize(
    "output_mode",
    ("rows", "dimension", "nan", "null", "child_null", "wrong_type", "zero"),
)
def test_learned_provider_rejects_invalid_output_without_cache(output_mode: str) -> None:
    representation = _learned_representation(unit_norm=output_mode == "zero")
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model, output_mode=output_mode)
    session = duckpd.connect()
    session.register_embedding_provider(model, provider)
    session.prepare_embedding_model(model)

    with pytest.raises((TypeError, ValueError)):
        session.embed_series_query(
            {"simple_return": [1.0, 2.0]},
            representation=representation,
        )

    assert session._embedded_series_queries == {}
    assert session.inspect_prepared_embedding_models()[0].model_fingerprint == (model.fingerprint)


def test_failed_series_preparation_is_never_promoted() -> None:
    representation = _learned_representation()
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(
        model,
        prepared_override={"model_fingerprint": "f" * 64},
    )
    session = duckpd.connect()
    session.register_embedding_provider(model, provider)

    for _ in range(2):
        with pytest.raises(UnsupportedOperationError, match="mismatched fingerprint"):
            session.prepare_embedding_model(model)

    assert provider.prepare_calls == 2
    assert session.inspect_prepared_embedding_models() == ()


def test_non_thread_safe_series_provider_calls_never_overlap_and_close_once() -> None:
    representation = _learned_representation()
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model, thread_safe=False, delay=0.01)
    session = duckpd.connect()
    session.register_embedding_provider(model, provider)
    session.prepare_embedding_model(model)

    def encode(value: float) -> duckpd.EmbeddedSeriesQuery:
        return session.embed_series_query(
            {"simple_return": [value, value + 1.0]},
            representation=representation,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(encode, (1.0, 2.0, 3.0, 4.0)))

    assert len(results) == 4
    assert provider.max_active == 1
    session.close()
    session.close()
    assert provider.close_calls == 1


def test_learned_series_profile_reports_lifecycle_and_arrow_resources() -> None:
    representation = _learned_representation(
        channels=("target", "past", "known"),
        roles=("target", "past_covariate", "known_future_covariate"),
    )
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model)
    session = duckpd.connect()
    session.register_embedding_provider(model, provider)
    session.prepare_embedding_model(model)

    profile = _learned_windows(session, representation, batch_size=2).profile()
    metrics = profile.embedding_metrics

    assert metrics is not None
    assert metrics["series_preparation_count"] == 1
    assert metrics["series_model_cache_bytes"] == 0
    assert metrics["series_corpus_provider_calls"] == 2
    assert metrics["series_corpus_rows"] == 4
    assert metrics["series_max_provider_batch_rows"] == 2
    assert metrics["series_arrow_bytes"] > 0
    assert metrics["series_peak_arrow_batch_bytes"] > 0


def test_learned_zero_scale_and_null_masks_skip_provider_calls() -> None:
    representation = _learned_representation(
        normalization="zscore",
        zero_scale="null",
    )
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model)
    session = duckpd.connect()
    session.register_embedding_provider(model, provider)
    session.prepare_embedding_model(model)
    source = session.from_pandas(
        pd.DataFrame({"row": range(4), "value": [3.0] * 4}),
        order_by="row",
    )
    windows = source.assign(window=lambda frame: frame["value"].rolling(2).to_array())
    embedded = windows.embed_series(
        columns={"simple_return": "window"},
        into="vector",
        representation=representation,
    )

    result = embedded.collect()

    assert result["vector"].isna().all()
    assert provider.batches == []
    assert session._embedding_metrics["series_complete_rows"] == 0
    assert session._embedding_metrics["series_propagated_null_rows"] == 1
    assert session._embedding_metrics["series_zero_scale_rows"] == 3


def test_failed_learned_sink_leaves_no_output_or_query_cache(tmp_path: Path) -> None:
    representation = _learned_representation()
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model, output_mode="nan")
    session = duckpd.connect()
    session.register_embedding_provider(model, provider)
    session.prepare_embedding_model(model)
    source = session.from_pandas(
        pd.DataFrame({"row": range(3), "value": [1.0, 2.0, 3.0]}),
        order_by="row",
    )
    windows = source.assign(window=lambda frame: frame["value"].rolling(2).to_array())
    embedded = windows.embed_series(
        columns={"simple_return": "window"},
        into="vector",
        representation=representation,
    )
    output = tmp_path / "invalid-series.parquet"

    with pytest.raises(MaterializationError):
        embedded.write_parquet(output)

    assert not output.exists()
    assert session._embedded_series_queries == {}


def test_series_provider_registration_and_preparation_are_strict() -> None:
    representation = _learned_representation()
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model)
    session = duckpd.connect()

    with pytest.raises(ValueError, match="backend='custom'"):
        replace(model, backend="fastembed")
    with pytest.raises(ValueError, match="specification"):
        session.register_embedding_provider(_learned_model(), provider)

    invalid_thread_safety = _RecordingSeriesProvider(model)
    invalid_thread_safety._thread_safe = cast("Any", "yes")
    with pytest.raises(TypeError, match="thread_safe"):
        session.register_embedding_provider(model, invalid_thread_safety)

    session.register_embedding_provider(model, provider)
    with pytest.raises(ValueError, match="different embedding provider"):
        session.register_embedding_provider(model, _RecordingSeriesProvider(model))

    with pytest.raises(UnsupportedOperationError, match="register_embedding_provider"):
        duckpd.connect().prepare_embedding_model(model)

    class InvalidPreparationProvider(_RecordingSeriesProvider):
        def prepare(self) -> duckpd.PreparedModelInfo:
            return cast("duckpd.PreparedModelInfo", {})

    invalid_model = replace(model, revision="immutable-revision-v2")
    invalid_session = duckpd.connect()
    invalid_session.register_embedding_provider(
        invalid_model,
        InvalidPreparationProvider(invalid_model),
    )
    with pytest.raises(TypeError, match="PreparedModelInfo"):
        invalid_session.prepare_embedding_model(invalid_model)
    assert invalid_session.inspect_prepared_embedding_models() == ()


def test_series_preparation_measures_local_cache_bytes(tmp_path: Path) -> None:
    cache = tmp_path / "series-model"
    cache.mkdir()
    (cache / "weights.bin").write_bytes(b"1234")
    (cache / "config.json").write_bytes(b"12")
    representation = _learned_representation()
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(
        model,
        prepared_override={"cache_path": str(cache)},
    )
    session = duckpd.connect()
    session.register_embedding_provider(model, provider)

    session.prepare_embedding_model(model)

    assert session._embedding_metrics["series_model_cache_bytes"] == 6


def test_learned_null_error_fails_before_provider_inference() -> None:
    representation = _learned_representation()
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model)
    session = duckpd.connect()
    session.register_embedding_provider(model, provider)
    session.prepare_embedding_model(model)
    source = session.from_pandas(
        pd.DataFrame({"row": [0, 1], "value": [1.0, 2.0]}),
        order_by="row",
    )
    windows = source.assign(window=lambda frame: frame["value"].rolling(2).to_array())
    embedded = windows.embed_series(
        columns={"simple_return": "window"},
        into="vector",
        representation=representation,
        null_policy="error",
    )

    with pytest.raises(MaterializationError):
        embedded.collect()

    assert provider.batches == []
