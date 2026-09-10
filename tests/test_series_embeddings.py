from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import duckpd
from duckpd._logical import SeriesRepresentationPlan
from duckpd.errors import MaterializationError, UnsupportedOperationError
from duckpd.frame import DataFrame
from duckpd.series_embeddings import SeriesColumnSpec, SeriesWindowSpec


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
        input_normalization="none",
        pooling="mean-valid-v1",
        adapter_revision="adapter-v1",
    )


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
    assert session.execution_count == 0

    assert result.collect()["id"].tolist() == [1]  # pyright: ignore[reportUnknownMemberType]
    assert session.execution_count == 1


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
    learned = duckpd.series_representation(
        window=2,
        channels=("simple_return",),
        sampling="observations",
        data_contract="value/v1",
        encoder=_learned_model(),
    )

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
            columns={"simple_return": "window"},
            into="vector",
            representation=learned,
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
