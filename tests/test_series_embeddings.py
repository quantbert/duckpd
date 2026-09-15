from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from threading import Lock
from time import sleep
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import duckpd
import duckpd._ts2vec as ts2vec
import duckpd._tspulse as tspulse
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


def _learned_model() -> duckpd.SeriesEmbeddingModelSpec:
    return duckpd.series_embedding_model(
        "research/return-encoder",
        revision="immutable-revision-v1",
        artifact_sha256="a" * 64,
        backend="custom",
        dimension=3,
        input_length=2,
        input_channels=("simple_return",),
        input_roles=("target",),
        input_normalization="none",
        pooling="mean-valid-v1",
        adapter_revision="adapter-v1",
    )


def _learned_representation(
    *,
    channels: tuple[str, ...] = ("simple_return",),
    roles: tuple[duckpd.SeriesChannelRole, ...] = ("target",),
    normalization: SeriesNormalization = "none",
    unit_norm: bool = False,
    zero_scale: ZeroScalePolicy = "error",
) -> duckpd.SeriesRepresentationSpec:
    model = duckpd.series_embedding_model(
        "research/return-encoder",
        revision="immutable-revision-v1",
        artifact_sha256="a" * 64,
        backend="custom",
        dimension=3,
        input_length=2,
        input_channels=channels,
        input_roles=roles,
        input_normalization="provider-none-v1",
        pooling="mean-valid-v1",
        adapter_revision="adapter-v1",
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
        model: duckpd.SeriesEmbeddingModelSpec,
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
    def specification(self) -> duckpd.SeriesEmbeddingModelSpec:
        return self._model

    @property
    def thread_safe(self) -> bool:
        return self._thread_safe

    def prepare(self) -> duckpd.PreparedSeriesModelInfo:
        self.prepare_calls += 1
        values: dict[str, object] = {
            "model_fingerprint": self._model.fingerprint,
            "resolved_revision": self._model.revision,
            "artifact_sha256": self._model.artifact_sha256,
            "backend": self._model.backend,
            "adapter_revision": self._model.adapter_revision,
            "input_length": self._model.input_length,
            "input_channels": self._model.input_channels,
            "input_roles": self._model.input_roles,
            "input_normalization": self._model.input_normalization,
            "pooling": self._model.pooling,
            "dimension": self._model.dimension,
            "cache_path": None,
            "execution_providers": ("CPUExecutionProvider",),
            "runtime_versions": (("fake-series-runtime", "1.0"),),
        }
        values.update(self.prepared_override)
        return duckpd.PreparedSeriesModelInfo(**values)  # type: ignore[arg-type]

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

    assert model == duckpd.SeriesEmbeddingModelSpec.from_dict(model.to_dict())
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
    invalid = (
        lambda: replace(model, model=""),
        lambda: replace(model, revision="main"),
        lambda: replace(model, artifact_sha256="A" * 64),
        lambda: replace(model, dimension=True),
        lambda: replace(model, input_length=0),
        lambda: replace(model, input_channels=("",)),
        lambda: replace(model, input_channels=("x", "x")),
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
        duckpd.SeriesEmbeddingModelSpec.from_dict({**_learned_model().to_dict(), "extra": 1})


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

    restored = duckpd.SeriesEmbeddingModelSpec.from_dict(json.loads(json.dumps(model.to_dict())))

    assert restored == model
    assert restored.fingerprint == model.fingerprint
    with pytest.raises(ValueError, match="one-for-one"):
        replace(model, input_roles=("target",))
    with pytest.raises(ValueError, match="input_roles"):
        replace(model, input_roles=("target", "invalid", "known_future_covariate"))
    with pytest.raises(ValueError, match="target"):
        replace(
            model,
            input_roles=(
                "past_covariate",
                "past_covariate",
                "known_future_covariate",
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
    session.register_series_embedding_provider(model, provider)
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

    prepared = session.prepare_series_embedding_model(model)
    assert prepared.model_fingerprint == model.fingerprint
    assert prepared.resolved_revision == model.revision
    assert prepared.runtime_versions == (("fake-series-runtime", "1.0"),)
    assert session.prepare_series_embedding_model(model) is prepared
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
    session.register_series_embedding_provider(model, provider)
    session.prepare_series_embedding_model(model)
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
    representation = _learned_representation()
    if output_mode == "zero":
        representation = replace(representation, unit_norm=True)
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model, output_mode=output_mode)
    session = duckpd.connect()
    session.register_series_embedding_provider(model, provider)
    session.prepare_series_embedding_model(model)

    with pytest.raises((TypeError, ValueError)):
        session.embed_series_query(
            {"simple_return": [1.0, 2.0]},
            representation=representation,
        )

    assert session._embedded_series_queries == {}
    assert session.inspect_prepared_series_embedding_models()[0].model_fingerprint == (
        model.fingerprint
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("model_fingerprint", "f" * 64),
        ("resolved_revision", "different-revision"),
        ("artifact_sha256", "b" * 64),
        ("adapter_revision", "different-adapter"),
        ("dimension", 4),
        ("input_roles", ("past_covariate",)),
    ),
)
def test_failed_series_preparation_is_never_promoted(field: str, invalid: object) -> None:
    representation = _learned_representation()
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model, prepared_override={field: invalid})
    session = duckpd.connect()
    session.register_series_embedding_provider(model, provider)

    for _ in range(2):
        with pytest.raises(UnsupportedOperationError, match="attestation mismatch"):
            session.prepare_series_embedding_model(model)

    assert provider.prepare_calls == 2
    assert session.inspect_prepared_series_embedding_models() == ()


def test_non_thread_safe_series_provider_calls_never_overlap_and_close_once() -> None:
    representation = _learned_representation()
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model, thread_safe=False, delay=0.01)
    session = duckpd.connect()
    session.register_series_embedding_provider(model, provider)
    session.prepare_series_embedding_model(model)

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
    session.register_series_embedding_provider(model, provider)
    session.prepare_series_embedding_model(model)

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
    session.register_series_embedding_provider(model, provider)
    session.prepare_series_embedding_model(model)
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
    session.register_series_embedding_provider(model, provider)
    session.prepare_series_embedding_model(model)
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


def test_prepared_series_model_info_rejects_incomplete_runtime_contracts() -> None:
    model = _learned_model()
    info = _RecordingSeriesProvider(model).prepare()
    invalid_changes: tuple[dict[str, object], ...] = (
        {"model_fingerprint": ""},
        {"resolved_revision": ""},
        {"artifact_sha256": "invalid"},
        {"backend": ""},
        {"adapter_revision": ""},
        {"input_length": 0},
        {"input_channels": ("",)},
        {"input_roles": ()},
        {"input_roles": ("invalid",)},
        {"input_normalization": ""},
        {"pooling": ""},
        {"dimension": 0},
        {"cache_path": ""},
        {"execution_providers": ()},
        {"execution_providers": ("",)},
        {"runtime_versions": ()},
        {"runtime_versions": (("runtime", ""),)},
        {"preparation_seconds": -1.0},
        {"preparation_seconds": float("nan")},
    )

    for changes in invalid_changes:
        with pytest.raises((TypeError, ValueError)):
            replace(info, **changes)


def test_series_provider_registration_and_preparation_are_strict() -> None:
    representation = _learned_representation()
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model)
    session = duckpd.connect()

    with pytest.raises(UnsupportedOperationError, match="backend='custom'"):
        noncustom = replace(model, backend="onnx-runtime")
        session.register_series_embedding_provider(
            noncustom,
            _RecordingSeriesProvider(noncustom),
        )
    with pytest.raises(ValueError, match="specification"):
        session.register_series_embedding_provider(_learned_model(), provider)

    invalid_thread_safety = _RecordingSeriesProvider(model)
    invalid_thread_safety._thread_safe = cast("Any", "yes")
    with pytest.raises(TypeError, match="thread_safe"):
        session.register_series_embedding_provider(model, invalid_thread_safety)

    session.register_series_embedding_provider(model, provider)
    with pytest.raises(ValueError, match="different series provider"):
        session.register_series_embedding_provider(model, _RecordingSeriesProvider(model))

    with pytest.raises(UnsupportedOperationError, match="not registered"):
        duckpd.connect().prepare_series_embedding_model(model)

    class InvalidPreparationProvider(_RecordingSeriesProvider):
        def prepare(self) -> duckpd.PreparedSeriesModelInfo:
            return cast("duckpd.PreparedSeriesModelInfo", {})

    invalid_model = replace(model, artifact_sha256="c" * 64)
    invalid_session = duckpd.connect()
    invalid_session.register_series_embedding_provider(
        invalid_model,
        InvalidPreparationProvider(invalid_model),
    )
    with pytest.raises(TypeError, match="PreparedSeriesModelInfo"):
        invalid_session.prepare_series_embedding_model(invalid_model)
    assert invalid_session.inspect_prepared_series_embedding_models() == ()


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
    session.register_series_embedding_provider(model, provider)

    session.prepare_series_embedding_model(model)

    assert session._embedding_metrics["series_model_cache_bytes"] == 6


def test_learned_null_error_fails_before_provider_inference() -> None:
    representation = _learned_representation()
    model = representation.encoder
    assert model is not None
    provider = _RecordingSeriesProvider(model)
    session = duckpd.connect()
    session.register_series_embedding_provider(model, provider)
    session.prepare_series_embedding_model(model)
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


class _FakeTSPulseTensor:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.values.dtype

    def to(self, *, dtype: Any, device: str) -> _FakeTSPulseTensor:
        assert device == "cpu"
        return _FakeTSPulseTensor(self.values.astype(dtype, copy=False))

    def clone(self) -> _FakeTSPulseTensor:
        return _FakeTSPulseTensor(self.values.copy())

    def detach(self) -> _FakeTSPulseTensor:
        return self

    def cpu(self) -> _FakeTSPulseTensor:
        return self

    def contiguous(self) -> _FakeTSPulseTensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.values

    def all(self) -> _FakeTSPulseTensor:
        return _FakeTSPulseTensor(self.values.all())

    def item(self) -> object:
        return self.values.item()

    def __getitem__(self, key: Any) -> _FakeTSPulseTensor:
        return _FakeTSPulseTensor(self.values[key])


class _FakeTSPulseTorch:
    __version__ = "2.10.0"
    float32 = np.float32
    bool = np.bool_

    @staticmethod
    def from_numpy(values: np.ndarray[Any, Any]) -> _FakeTSPulseTensor:
        return _FakeTSPulseTensor(values)

    @staticmethod
    def ones_like(
        tensor: _FakeTSPulseTensor,
        *,
        dtype: Any,
    ) -> _FakeTSPulseTensor:
        return _FakeTSPulseTensor(np.ones_like(tensor.values, dtype=dtype))

    @staticmethod
    def no_grad() -> nullcontext[None]:
        return nullcontext()

    @staticmethod
    def isfinite(tensor: _FakeTSPulseTensor) -> _FakeTSPulseTensor:
        return _FakeTSPulseTensor(np.isfinite(tensor.values))


class _FakeTSPulseModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            context_length=512,
            num_input_channels=1,
            patch_register_tokens=10,
            scaling="revin",
            revin_affine=True,
            minimum_scale=0.001,
            mask_type="user",
            decoder_d_model_layerwise=[24, 24],
            decoder_num_patches_layerwise=[128, 128],
        )
        self.eval_calls = 0
        self.devices: list[str] = []

    def eval(self) -> None:
        self.eval_calls += 1

    def to(self, device: str) -> None:
        self.devices.append(device)


class _FakeTSPulseFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.models: list[_FakeTSPulseModel] = []

    def from_pretrained(self, path: str, **kwargs: object) -> _FakeTSPulseModel:
        self.calls.append((path, kwargs))
        model = _FakeTSPulseModel()
        self.models.append(model)
        return model


def _install_fake_tspulse_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeTSPulseFactory, list[np.ndarray[Any, Any]]]:
    files = {name: name.encode() for name in ("config.json", "model.safetensors")}
    monkeypatch.setattr(
        tspulse,
        "_TSPULSE_FILE_RECORDS",
        {
            name: (len(content), hashlib.sha256(content).hexdigest())
            for name, content in files.items()
        },
    )

    def download(target: Path, _download_cache: Path) -> None:
        for name, content in files.items():
            (target / name).write_bytes(content)

    factory = _FakeTSPulseFactory()
    provider_inputs: list[np.ndarray[Any, Any]] = []

    def get_embeddings(
        _model: _FakeTSPulseModel,
        values: _FakeTSPulseTensor,
        *,
        past_observed_mask: _FakeTSPulseTensor,
        component: str,
        mode: str,
    ) -> _FakeTSPulseTensor:
        assert component == "decoder"
        assert mode == "register"
        assert past_observed_mask.values.all()
        provider_inputs.append(values.values.copy())
        starts = values.values[:, :1, 0]
        vectors = starts + np.arange(240, dtype=np.float32)[None, :]
        return _FakeTSPulseTensor(vectors[:, None, :])

    monkeypatch.setattr(tspulse, "_download_artifacts", download)
    monkeypatch.setattr(
        tspulse,
        "_load_runtime",
        lambda: (
            _FakeTSPulseTorch,
            factory,
            get_embeddings,
            (
                ("python", "3.12.0"),
                ("granite-tsfm", "0.3.9"),
                ("torch", "2.10.0"),
                ("transformers", "4.57.6"),
            ),
        ),
    )
    return factory, provider_inputs


def test_tspulse_provider_contract_is_deliberately_narrow() -> None:
    model = duckpd.tspulse_series_embedding_model("simple_return")

    assert model.backend == "tspulse"
    assert model.dimension == 240
    assert model.input_length == 512
    assert model.input_channels == ("simple_return",)
    assert model.input_roles == ("target",)
    assert model.input_normalization == "internal-revin-affine-v1"
    assert model.pooling == "decoder-register-v1"
    assert len(model.artifact_sha256) == 64

    with pytest.raises(ValueError, match="pinned univariate"):
        duckpd.TSPulseProvider(replace(model, dimension=241))
    with pytest.raises(ValueError, match="pinned univariate"):
        duckpd.TSPulseProvider(
            replace(
                model,
                input_channels=("simple_return", "volume"),
                input_roles=("target", "past_covariate"),
            )
        )
    representation = duckpd.series_representation(
        window=512,
        channels=("simple_return",),
        sampling="observations",
        data_contract="market/simple-return/v1",
        encoder=model,
    )
    with pytest.raises(ValueError, match="normalization='none'"):
        replace(representation, normalization="center")
    with pytest.raises(ValueError, match="unit_norm=False"):
        replace(representation, unit_norm=True)
    with pytest.raises(UnsupportedOperationError, match="backend='custom'"):
        duckpd.connect().register_series_embedding_provider(
            model,
            _RecordingSeriesProvider(model),
        )


def test_tspulse_provider_rejects_invalid_setup_before_inference(tmp_path: Path) -> None:
    model = duckpd.tspulse_series_embedding_model("simple_return")
    provider = duckpd.TSPulseProvider(model, cache_dir=tmp_path)

    assert provider.specification is model
    assert provider.thread_safe is False
    with pytest.raises(UnsupportedOperationError, match="not prepared"):
        provider.embed_windows(pa.record_batch([], names=[]))
    with pytest.raises(ValueError, match="non-empty"):
        duckpd.tspulse_series_embedding_model("")
    with pytest.raises(ValueError, match="prepare_timeout_seconds"):
        duckpd.TSPulseProvider(model, prepare_timeout_seconds=True)
    with pytest.raises(ValueError, match="max_download_bytes"):
        duckpd.TSPulseProvider(model, max_download_bytes=True)


def test_tspulse_failed_download_never_promotes_partial_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_tspulse_runtime(monkeypatch)
    model = duckpd.tspulse_series_embedding_model("simple_return")
    limited_root = tmp_path / "limited"
    with duckpd.connect() as session:
        with pytest.raises(UnsupportedOperationError, match="download-size limit"):
            session.prepare_series_embedding_model(
                model,
                cache_dir=limited_root,
                max_download_bytes=1,
            )
        assert session.inspect_prepared_series_embedding_models() == ()
    assert not (limited_root / model.fingerprint).exists()

    def fail_download(target: Path, _download_cache: Path) -> None:
        (target / "config.json").write_bytes(b"partial")
        raise RuntimeError("download failed")

    monkeypatch.setattr(tspulse, "_download_artifacts", fail_download)
    failed_root = tmp_path / "failed"
    with duckpd.connect() as session:
        with pytest.raises(RuntimeError, match="download failed"):
            session.prepare_series_embedding_model(model, cache_dir=failed_root)
        assert session.inspect_prepared_series_embedding_models() == ()
    assert not (failed_root / model.fingerprint).exists()
    assert not any(path.is_dir() for path in failed_root.iterdir())


def test_tspulse_runtime_gate_fails_before_cache_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = duckpd.tspulse_series_embedding_model("simple_return")
    original_version_info = tspulse.sys.version_info

    monkeypatch.setattr(tspulse.sys, "version_info", (3, 14, 0))
    with pytest.raises(UnsupportedOperationError, match=r"Python 3\.11-3\.13"):
        duckpd.TSPulseProvider(model, cache_dir=tmp_path).prepare()

    monkeypatch.setattr(tspulse.sys, "version_info", original_version_info)

    def missing_distribution(_name: str) -> str:
        raise tspulse.PackageNotFoundError

    monkeypatch.setattr(tspulse, "distribution_version", missing_distribution)
    with pytest.raises(UnsupportedOperationError, match=r"duckpd\[tspulse\]"):
        duckpd.TSPulseProvider(model, cache_dir=tmp_path).prepare()

    def wrong_distribution(_name: str) -> str:
        return "0.3.8"

    monkeypatch.setattr(tspulse, "distribution_version", wrong_distribution)
    with pytest.raises(UnsupportedOperationError, match=r"found 0\.3\.8"):
        duckpd.TSPulseProvider(model, cache_dir=tmp_path).prepare()

    def expected_distribution(_name: str) -> str:
        return "0.3.9"

    monkeypatch.setattr(tspulse, "distribution_version", expected_distribution)

    def missing_runtime(_name: str) -> Any:
        raise ImportError

    monkeypatch.setattr(tspulse.importlib, "import_module", missing_runtime)
    with pytest.raises(UnsupportedOperationError, match="runtime is incomplete"):
        duckpd.TSPulseProvider(model, cache_dir=tmp_path).prepare()

    assert list(tmp_path.iterdir()) == []


def test_tspulse_provider_prepares_attests_and_encodes_query_and_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, provider_inputs = _install_fake_tspulse_runtime(monkeypatch)
    model = duckpd.tspulse_series_embedding_model("simple_return")
    representation = duckpd.series_representation(
        window=512,
        channels=("simple_return",),
        sampling="observations",
        step="PT1M",
        data_contract="market/simple-return/v1",
        encoder=model,
    )
    session = duckpd.connect()

    prepared = session.prepare_series_embedding_model(model, cache_dir=tmp_path)
    query = session.embed_series_query(
        {"simple_return": [float(value) for value in range(512)]},
        representation=representation,
    )
    source = session.from_pandas(
        pd.DataFrame(
            {
                "row": range(513),
                "value": np.arange(513, dtype=np.float32),
            }
        ),
        order_by="row",
    )
    windows = source.assign(window=lambda frame: frame["value"].rolling(512).to_array())
    embedded = windows.embed_series(
        columns={"simple_return": "window"},
        into="vector",
        representation=representation,
        batch_size=1,
    )
    result = embedded[embedded["row"] >= 511].collect()

    assert prepared.backend == "tspulse"
    assert prepared.execution_providers == ("PyTorchCPU",)
    assert prepared.runtime_versions[1:] == (
        ("granite-tsfm", "0.3.9"),
        ("torch", "2.10.0"),
        ("transformers", "4.57.6"),
    )
    assert factory.calls[0][1] == {
        "local_files_only": True,
        "num_input_channels": 1,
        "mask_type": "user",
    }
    assert factory.models[0].eval_calls == 1
    assert factory.models[0].devices == ["cpu"]
    np.testing.assert_array_equal(query.values, np.arange(240, dtype=np.float32))
    np.testing.assert_array_equal(result["vector"].iloc[0], query.values)
    np.testing.assert_array_equal(
        result["vector"].iloc[1],
        np.arange(240, dtype=np.float32) + 1,
    )
    assert [values.shape for values in provider_inputs] == [
        (1, 512, 1),
        (1, 512, 1),
        (1, 512, 1),
    ]

    cache = Path(cast("str", prepared.cache_path))
    assert {path.name for path in cache.iterdir()} == {
        "config.json",
        "model.safetensors",
        "duckpd-manifest.json",
    }
    manifest = json.loads((cache / "duckpd-manifest.json").read_text())
    assert manifest["artifact_manifest_sha256"] == model.artifact_sha256
    assert manifest["model_fingerprint"] == model.fingerprint
    for metric in (
        "series_corpus_python_arrow_boundary_seconds",
        "series_input_conversion_seconds",
        "series_model_execution_seconds",
        "series_output_conversion_seconds",
    ):
        assert session._embedding_metrics[metric] > 0

    session.close()
    exact_artifact_bytes = sum(
        path.stat().st_size for path in cache.iterdir() if path.name != "duckpd-manifest.json"
    )
    with duckpd.connect() as reuse:
        reused = reuse.prepare_series_embedding_model(
            model,
            cache_dir=tmp_path,
            max_download_bytes=exact_artifact_bytes,
        )
    assert reused.cache_path == str(cache)
    direct = duckpd.TSPulseProvider(model, cache_dir=tmp_path)
    directly_prepared = direct.prepare()
    assert direct.prepare() is directly_prepared
    direct.close()
    (cache / "model.safetensors").write_bytes(b"changed")
    second = duckpd.connect()
    with pytest.raises(UnsupportedOperationError, match="failed verification"):
        second.prepare_series_embedding_model(model, cache_dir=tmp_path)
    assert second.inspect_prepared_series_embedding_models() == ()


@pytest.mark.skipif(
    os.environ.get("DUCKPD_TSPULSE_REAL_TESTS") != "1",
    reason="set DUCKPD_TSPULSE_REAL_TESTS=1 to run the pinned optional runtime",
)
def test_tspulse_real_runtime_matches_pinned_golden_output(tmp_path: Path) -> None:
    model = duckpd.tspulse_series_embedding_model("simple_return")
    representation = duckpd.series_representation(
        window=512,
        channels=("simple_return",),
        sampling="observations",
        step="PT1M",
        data_contract="market/simple-return/v1",
        encoder=model,
    )
    with duckpd.connect() as session:
        session.prepare_series_embedding_model(
            model,
            cache_dir=tmp_path,
            timeout_seconds=300,
            max_download_bytes=5_000_000,
        )
        query = session.embed_series_query(
            {
                "simple_return": np.linspace(
                    -1.0,
                    1.0,
                    512,
                    dtype=np.float32,
                ).tolist()
            },
            representation=representation,
        )

    digest = hashlib.sha256(np.asarray(query.values, dtype="<f4").tobytes()).hexdigest()
    assert digest == "5a99f88075d2e25619176d13d1bb5caf61d6ad5a07956e28c268316f8a820dfb"


def _write_fake_ts2vec_bundle(root: Path) -> duckpd.SeriesEmbeddingModelSpec:
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"fake-safe-tensors")
    return ts2vec.finalize_ts2vec_bundle(
        root,
        {
            "schema_version": 1,
            "backend": "ts2vec",
            "model": "duckpd/test-ts2vec",
            "source": {
                "repository": "https://github.com/zhihanyue/ts2vec",
                "revision": "b0088e14a99706c05451316dc6db8d3da9351163",
                "license": "MIT",
                "implementation": "duckpd-pinned-port-v1",
            },
            "architecture": {
                "input_dims": 3,
                "output_dims": 4,
                "hidden_dims": 8,
                "depth": 2,
                "kernel_size": 3,
                "training_mask": "binomial-v1",
                "weights": "swa-averaged",
            },
            "input": {
                "length": 3,
                "channels": ["close_return", "bar_return", "intrabar_range"],
                "roles": ["target", "past_covariate", "past_covariate"],
                "dtype": "float32",
                "missingness": "complete-only",
            },
            "preprocessing": {
                "kind": "artifact-train-standard-score-v1",
                "mean": [1.0, 10.0, 100.0],
                "scale": [2.0, 5.0, 10.0],
                "fitted_on": "training-points-only",
            },
            "pooling": "full-series-max-v1",
            "output": {
                "dimension": 4,
                "dtype": "float32",
                "normalization": "none",
            },
            "runtime": {
                "python": "3.12.0",
                "numpy": "2.0.0",
                "torch": "2.10.0",
                "safetensors": "0.5.0",
            },
            "training": {
                "dataset": "synthetic.parquet",
                "dataset_sha256": "a" * 64,
                "data_contract": "test/ts2vec/v1",
                "derived_channels": {},
                "tickers": ["TEST"],
                "points_per_ticker": {"TEST": 10},
                "split": {},
                "window_stride": 1,
                "seed": 7,
                "iterations": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "temporal_unit": 0,
                "objective": "hierarchical-contrastive-v1",
                "losses": [1.0],
            },
        },
    )


_DELETE_MANIFEST_VALUE = object()


def _mutate_ts2vec_manifest(
    root: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    manifest_path = root / "duckpd-ts2vec-manifest.json"
    manifest = cast(
        "dict[str, object]",
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    updated = deepcopy(manifest)
    target = updated
    for key in path[:-1]:
        target = cast("dict[str, object]", target[key])
    if value is _DELETE_MANIFEST_VALUE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    manifest_path.write_text(json.dumps(updated), encoding="utf-8")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("extra",), True, "field set"),
        (("schema_version",), 2, "schema or backend"),
        (("source",), "not-an-object", "source must be an object"),
        (("source", "revision"), "moving-main", "source attestation"),
        (("pooling",), "mean", "full-series max pooling"),
        (("runtime", "torch"), _DELETE_MANIFEST_VALUE, "runtime attestation"),
        (("training", "losses"), _DELETE_MANIFEST_VALUE, "provenance field set"),
        (("training", "dataset_sha256"), "not-a-digest", "lowercase SHA-256"),
        (("training", "objective"), "other", "objective"),
        (("input", "extra"), True, "input contract field set"),
        (("input", "channels"), "not-an-array", "must be an array"),
        (("input", "channels"), [1, 2, 3], "must be non-empty strings"),
        (("input", "roles"), ["target"], "channels and roles"),
        (
            ("input", "roles"),
            ["known_future_covariate"] * 3,
            "input roles",
        ),
        (("input", "dtype"), "float64", "complete float32"),
        (("input", "length"), 1, "outside the supported range"),
        (("architecture", "extra"), True, "architecture field set"),
        (("architecture", "input_dims"), 2, "does not match its channels"),
        (("architecture", "weights"), "raw", "weight selection"),
        (("preprocessing", "extra"), True, "preprocessing field set"),
        (("preprocessing", "fitted_on"), "all-points", "preprocessing recipe"),
        (("preprocessing", "mean"), "not-an-array", "must be an array"),
        (
            ("preprocessing", "mean"),
            [1.0, None, 100.0],
            "must contain finite numbers",
        ),
        (
            ("preprocessing", "mean"),
            [1.0, float("nan"), 100.0],
            "must contain finite numbers",
        ),
        (("preprocessing", "mean"), [1.0], "state does not match channels"),
        (("preprocessing", "scale"), [2.0, 0.0, 10.0], "scales must be positive"),
        (("output", "normalization"), "unit", "output contract"),
        (("model",), "", "non-empty string"),
        (("artifacts",), {"other": {}}, "attest exactly model.safetensors"),
        (
            ("artifacts", "model.safetensors", "extra"),
            True,
            "artifact record",
        ),
        (("artifacts", "model.safetensors", "size"), True, "supported range"),
    ],
)
def test_ts2vec_rejects_manifest_contract_mutations(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    bundle = tmp_path / "bundle"
    _write_fake_ts2vec_bundle(bundle)
    _mutate_ts2vec_manifest(bundle, path, value)

    with pytest.raises(UnsupportedOperationError, match=message):
        duckpd.ts2vec_series_embedding_model(bundle)


def test_ts2vec_rejects_nonregular_or_incomplete_bundles(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(UnsupportedOperationError, match="regular local directory"):
        duckpd.ts2vec_series_embedding_model(missing)

    bundle = tmp_path / "bundle"
    _write_fake_ts2vec_bundle(bundle)
    (bundle / "unexpected").write_text("not attested", encoding="utf-8")
    with pytest.raises(UnsupportedOperationError, match="file set mismatch"):
        duckpd.ts2vec_series_embedding_model(bundle)

    (bundle / "unexpected").unlink()
    manifest = bundle / "duckpd-ts2vec-manifest.json"
    manifest.write_text("{", encoding="utf-8")
    with pytest.raises(UnsupportedOperationError, match="invalid JSON"):
        duckpd.ts2vec_series_embedding_model(bundle)


def test_ts2vec_rejects_nonfile_entries_and_nonobject_manifest(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_fake_ts2vec_bundle(bundle)
    nested = bundle / "nested"
    nested.mkdir()
    with pytest.raises(UnsupportedOperationError, match="only regular attested files"):
        duckpd.ts2vec_series_embedding_model(bundle)

    nested.rmdir()
    manifest = bundle / "duckpd-ts2vec-manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(UnsupportedOperationError, match="manifest must be an object"):
        duckpd.ts2vec_series_embedding_model(bundle)


def test_ts2vec_provider_requires_preparation_and_exact_specification(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    model = _write_fake_ts2vec_bundle(bundle)
    provider = duckpd.TS2VecProvider(model, bundle_dir=bundle)
    batch = pa.record_batch(
        [
            pa.array(
                [[1.0, 2.0, 3.0]],
                type=pa.list_(pa.float32(), 3),
            )
        ],
        names=["close_return"],
    )
    with pytest.raises(UnsupportedOperationError, match="not prepared"):
        provider.embed_windows(batch)

    mismatched = replace(model, dimension=model.dimension + 1)
    bad_provider = duckpd.TS2VecProvider(mismatched, bundle_dir=bundle)
    with pytest.raises(ValueError, match="mismatched fields: dimension"):
        bad_provider.prepare()


def test_ts2vec_validates_provider_limits_and_producer_output(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    model = _write_fake_ts2vec_bundle(bundle)
    with pytest.raises(ValueError, match="prepare_timeout_seconds"):
        duckpd.TS2VecProvider(
            model,
            bundle_dir=bundle,
            prepare_timeout_seconds=True,
        )
    with pytest.raises(ValueError, match="max_artifact_bytes"):
        duckpd.TS2VecProvider(
            model,
            bundle_dir=bundle,
            max_artifact_bytes=True,
        )

    with duckpd.connect() as session, pytest.raises(ValueError, match="cache_dir"):
        session.prepare_series_embedding_model(
            model,
            artifact_dir=bundle,
            cache_dir=tmp_path / "cache",
        )
    with (
        duckpd.connect() as session,
        pytest.raises(
            UnsupportedOperationError,
            match="size limit",
        ),
    ):
        session.prepare_series_embedding_model(
            model,
            artifact_dir=bundle,
            max_artifact_bytes=1,
        )

    tspulse_model = duckpd.tspulse_series_embedding_model("simple_return")
    with duckpd.connect() as session, pytest.raises(ValueError, match="artifact_dir"):
        session.prepare_series_embedding_model(
            tspulse_model,
            artifact_dir=bundle,
        )
    with (
        duckpd.connect() as session,
        pytest.raises(
            ValueError,
            match="max_artifact_bytes",
        ),
    ):
        session.prepare_series_embedding_model(
            tspulse_model,
            max_artifact_bytes=1,
        )

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match=r"regular model\.safetensors"):
        ts2vec.finalize_ts2vec_bundle(incomplete, {})


class _FakeTS2VecModel:
    def __init__(self, architecture: dict[str, object]) -> None:
        self.architecture = architecture
        self.loaded: tuple[dict[str, object], bool] | None = None
        self.eval_calls = 0
        self.devices: list[str] = []

    def load_state_dict(self, state: dict[str, object], *, strict: bool) -> None:
        self.loaded = (state, strict)

    def eval(self) -> None:
        self.eval_calls += 1

    def to(self, device: str) -> None:
        self.devices.append(device)


class _FakeTS2VecRuntime:
    models: ClassVar[list[_FakeTS2VecModel]] = []
    inputs: ClassVar[list[np.ndarray[Any, Any]]] = []

    @classmethod
    def build_encoder(cls, architecture: dict[str, object]) -> _FakeTS2VecModel:
        model = _FakeTS2VecModel(architecture)
        cls.models.append(model)
        return model

    @classmethod
    def full_series_encode(
        cls,
        _model: _FakeTS2VecModel,
        values: _FakeTSPulseTensor,
    ) -> _FakeTSPulseTensor:
        cls.inputs.append(values.values.copy())
        maxima = values.values.max(axis=1)
        mean_target = values.values[:, :, 0].mean(axis=1, keepdims=True)
        return _FakeTSPulseTensor(
            np.concatenate(  # pyright: ignore[reportUnknownMemberType]
                (maxima, mean_target),
                axis=1,
            )
        )


def test_ts2vec_attested_bundle_drives_query_and_corpus_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    model = _write_fake_ts2vec_bundle(bundle)
    _FakeTS2VecRuntime.models = []
    _FakeTS2VecRuntime.inputs = []

    def load_weights(_path: str, device: str) -> dict[str, object]:
        return {"weight": (device, 1)}

    def fake_runtime() -> tuple[
        object,
        object,
        object,
        tuple[tuple[str, str], ...],
    ]:
        return (
            _FakeTSPulseTorch,
            _FakeTS2VecRuntime,
            load_weights,
            (
                ("python", "3.12.0"),
                ("torch", "2.10.0"),
                ("safetensors", "0.5.0"),
            ),
        )

    monkeypatch.setattr(ts2vec, "_load_runtime", fake_runtime)

    assert model == duckpd.ts2vec_series_embedding_model(bundle)
    assert model.backend == "ts2vec"
    assert model.input_channels == (
        "close_return",
        "bar_return",
        "intrabar_range",
    )
    assert model.input_roles == ("target", "past_covariate", "past_covariate")
    assert model.input_length == 3
    assert model.dimension == 4
    representation = duckpd.series_representation(
        window=3,
        channels=model.input_channels,
        sampling="observations",
        data_contract="test/ts2vec/v1",
        encoder=model,
    )
    with pytest.raises(ValueError, match="TS2Vec representations"):
        replace(representation, unit_norm=True)

    with (
        duckpd.connect() as missing,
        pytest.raises(UnsupportedOperationError, match="requires artifact_dir"),
    ):
        missing.prepare_series_embedding_model(model)

    with (
        duckpd.connect() as wrong_limit,
        pytest.raises(ValueError, match="max_download_bytes"),
    ):
        wrong_limit.prepare_series_embedding_model(
            model,
            artifact_dir=bundle,
            max_download_bytes=len(b"fake-safe-tensors"),
        )

    with duckpd.connect() as session:
        prepared = session.prepare_series_embedding_model(
            model,
            artifact_dir=bundle,
            max_artifact_bytes=len(b"fake-safe-tensors"),
        )
        query = session.embed_series_query(
            {
                "close_return": [1.0, 3.0, 5.0],
                "bar_return": [10.0, 15.0, 20.0],
                "intrabar_range": [100.0, 110.0, 120.0],
            },
            representation=representation,
        )
        source = session.from_pandas(
            pd.DataFrame(
                {
                    "row": range(4),
                    "close_return": [1.0, 3.0, 5.0, 7.0],
                    "bar_return": [10.0, 15.0, 20.0, 25.0],
                    "intrabar_range": [100.0, 110.0, 120.0, 130.0],
                }
            ),
            order_by="row",
        )
        windows = source.assign(
            close_window=lambda frame: frame["close_return"].rolling(3).to_array(),
            bar_window=lambda frame: frame["bar_return"].rolling(3).to_array(),
            range_window=lambda frame: frame["intrabar_range"].rolling(3).to_array(),
        )
        embedded = windows.embed_series(
            columns={
                "close_return": "close_window",
                "bar_return": "bar_window",
                "intrabar_range": "range_window",
            },
            into="vector",
            representation=representation,
            batch_size=1,
        )
        result = embedded[embedded["row"] >= 2].collect()

    assert prepared.backend == "ts2vec"
    assert prepared.cache_path == str(bundle)
    assert prepared.execution_providers == ("PyTorchCPU",)
    np.testing.assert_array_equal(query.values, [2.0, 2.0, 2.0, 1.0])
    np.testing.assert_array_equal(result["vector"].iloc[0], query.values)
    np.testing.assert_array_equal(result["vector"].iloc[1], [3.0, 3.0, 3.0, 2.0])
    assert [values.shape for values in _FakeTS2VecRuntime.inputs] == [
        (1, 3, 3),
        (1, 3, 3),
        (1, 3, 3),
    ]
    for metric in (
        "series_corpus_python_arrow_boundary_seconds",
        "series_input_conversion_seconds",
        "series_model_execution_seconds",
        "series_output_conversion_seconds",
    ):
        assert session._embedding_metrics[metric] > 0
    assert _FakeTS2VecRuntime.models[0].loaded == (
        {"weight": ("cpu", 1)},
        True,
    )
    assert _FakeTS2VecRuntime.models[0].eval_calls == 1
    assert _FakeTS2VecRuntime.models[0].devices == ["cpu"]

    (bundle / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(UnsupportedOperationError, match="failed verification"):
        duckpd.ts2vec_series_embedding_model(bundle)
