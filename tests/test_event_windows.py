from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import duckpd
from duckpd._logical import EventWindowPlan
from duckpd.errors import MaterializationError, UnsupportedOperationError


def _source_frames() -> tuple[duckpd.Session, duckpd.DataFrame, duckpd.DataFrame]:
    session = duckpd.connect()
    observations = session.sql(
        """
        SELECT * FROM (VALUES
          ('A', TIMESTAMP '2024-01-01 12:00:00', TIMESTAMP '2024-01-01 12:01:00', 1.0),
          ('A', TIMESTAMP '2024-01-01 12:01:00', TIMESTAMP '2024-01-01 12:02:00', 2.0),
          ('A', TIMESTAMP '2024-01-01 12:02:00', TIMESTAMP '2024-01-01 12:03:00', 3.0),
          ('A', TIMESTAMP '2024-01-01 12:03:00', TIMESTAMP '2024-01-01 12:04:00', 4.0),
          ('A', TIMESTAMP '2024-01-01 12:04:00', TIMESTAMP '2024-01-01 12:05:00', 5.0),
          ('B', TIMESTAMP '2024-01-01 12:00:00', TIMESTAMP '2024-01-01 12:01:00', 10.0),
          ('B', TIMESTAMP '2024-01-01 12:01:00', TIMESTAMP '2024-01-01 12:02:00', 20.0),
          ('B', TIMESTAMP '2024-01-01 12:02:00', TIMESTAMP '2024-01-01 12:10:00', 30.0)
        ) AS t(symbol, bar_time, available_at, ret)
        """
    )
    events = session.sql(
        """
        SELECT * FROM (VALUES
          (10, 'A', TIMESTAMP '2024-01-01 12:02:30', TIMESTAMP '2024-01-01 12:02:31'),
          (20, 'B', TIMESTAMP '2024-01-01 12:02:30', TIMESTAMP '2024-01-01 12:02:45'),
          (30, 'A', TIMESTAMP '2024-01-01 12:05:30', TIMESTAMP '2024-01-01 12:05:31')
        ) AS t(event_id, symbol, event_time, event_available_at)
        """
    )
    return session, observations, events


def _windows(
    observations: duckpd.DataFrame,
    events: duckpd.DataFrame,
    *,
    anchor: str = "floor",
    window: tuple[int, int] = (-2, 1),
    incomplete: str = "null",
) -> duckpd.DataFrame:
    return observations.event_windows(
        events,
        on="bar_time",
        bar_label="start",
        event_on="event_time",
        by="symbol",
        event_id="event_id",
        columns={"ret_window": "ret"},
        window=window,
        step="PT1M",
        anchor=anchor,  # pyright: ignore[reportArgumentType]
        available_at="available_at",
        event_available_at="event_available_at",
        incomplete=incomplete,  # pyright: ignore[reportArgumentType]
    )


def test_event_windows_are_lazy_exact_and_point_in_time_safe() -> None:
    session, observations, events = _source_frames()

    windows = _windows(observations, events)

    assert isinstance(windows._plan, EventWindowPlan)
    assert windows.columns == (
        "event_id",
        "symbol",
        "event_time",
        "event_available_at",
        "ret_window",
        "window_window_start",
        "window_window_end",
        "window_window_count",
        "window_window_complete",
        "window_window_available_at",
    )
    assert session.execution_count == 0
    operation = json.loads(windows.explain(mode="json"))["execution_boundaries"][
        "embedding_operations"
    ][0]
    assert operation == {
        "operation": "event_windows",
        "backend": "native",
        "window": [-2, 1],
        "length": 3,
        "pre_aggregation_cardinality": {
            "factor": "event_rows",
            "multiplier": 3,
        },
        "step_ns": 60_000_000_000,
        "anchor": "floor",
        "bar_label": "start",
        "incomplete": "null",
        "keys": ["symbol"],
        "boundary": "duckdb_native_exact_grid",
        "channels": ["ret_window"],
        "persistence": "lazy",
    }
    assert session.execution_count == 0

    result = windows.collect()

    np.testing.assert_array_equal(result.loc[0, "ret_window"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.loc[1, "ret_window"], [10.0, 20.0, 30.0])
    assert result.loc[0, "window_window_start"] == pd.Timestamp("2024-01-01 12:00:00")
    assert result.loc[0, "window_window_end"] == pd.Timestamp("2024-01-01 12:03:00")
    assert result.loc[0, "window_window_available_at"] == pd.Timestamp("2024-01-01 12:03:00")
    assert result["window_window_count"].tolist() == [3, 3, 2]
    assert result["window_window_complete"].tolist() == [True, True, False]
    assert result.loc[2, "ret_window"] is pd.NA
    assert pd.isna(result.loc[2, "window_window_available_at"])
    assert session.execution_count == 1


def test_event_anchor_floor_and_ceil_have_exact_grid_boundaries() -> None:
    _, observations, events = _source_frames()
    event_subset = events[events["event_id"] == 10]

    floor = _windows(observations, event_subset, anchor="floor", window=(-1, 2)).collect()
    ceil = _windows(observations, event_subset, anchor="ceil", window=(-1, 2)).collect()

    np.testing.assert_array_equal(floor.loc[0, "ret_window"], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(ceil.loc[0, "ret_window"], [3.0, 4.0, 5.0])
    assert floor.loc[0, "window_window_start"] == pd.Timestamp("2024-01-01 12:01:00")
    assert ceil.loc[0, "window_window_start"] == pd.Timestamp("2024-01-01 12:02:00")


def test_event_windows_enforce_bounded_source_contracts() -> None:
    _, observations, events = _source_frames()
    incomplete_event = events[events["event_id"] == 30]

    with pytest.raises(MaterializationError):
        _windows(observations, incomplete_event, incomplete="error").collect()

    session = duckpd.connect()
    off_grid = session.sql(
        """
        SELECT 'A' AS symbol,
               TIMESTAMP '2024-01-01 12:00:30' AS bar_time,
               TIMESTAMP '2024-01-01 12:01:30' AS available_at,
               1.0 AS ret
        """
    )
    one_event = session.sql(
        """
        SELECT 1 AS event_id, 'A' AS symbol,
               TIMESTAMP '2024-01-01 12:01:00' AS event_time,
               TIMESTAMP '2024-01-01 12:01:01' AS event_available_at
        """
    )
    with pytest.raises(MaterializationError):
        _windows(off_grid, one_event, window=(-1, 1))[["event_id"]].collect()

    early_availability = session.sql(
        """
        SELECT 'A' AS symbol,
               TIMESTAMP '2024-01-01 12:00:00' AS bar_time,
               TIMESTAMP '2024-01-01 12:00:59' AS available_at,
               1.0 AS ret
        """
    )
    with pytest.raises(MaterializationError):
        _windows(early_availability, one_event, window=(-1, 1)).collect()

    duplicate_slot = session.sql(
        """
        SELECT * FROM (VALUES
          ('A', TIMESTAMP '2024-01-01 12:00:00', TIMESTAMP '2024-01-01 12:01:00', 1.0),
          ('A', TIMESTAMP '2024-01-01 12:00:00', TIMESTAMP '2024-01-01 12:01:00', 1.5),
          ('A', TIMESTAMP '2024-01-01 12:01:00', TIMESTAMP '2024-01-01 12:02:00', 2.0)
        ) AS t(symbol, bar_time, available_at, ret)
        """
    )
    with pytest.raises(MaterializationError):
        _windows(duplicate_slot, one_event, window=(-1, 1)).collect()

    overflow_value = session.sql(
        """
        SELECT * FROM (VALUES
          ('A', TIMESTAMP '2024-01-01 12:00:00',
           TIMESTAMP '2024-01-01 12:01:00', 1e100::DOUBLE),
          ('A', TIMESTAMP '2024-01-01 12:01:00',
           TIMESTAMP '2024-01-01 12:02:00', 2.0::DOUBLE)
        ) AS t(symbol, bar_time, available_at, ret)
        """
    )
    with pytest.raises(MaterializationError):
        _windows(overflow_value, one_event, window=(-1, 1)).collect()

    bounded = session.sql(
        """
        SELECT * FROM (VALUES
          ('A', TIMESTAMP '2024-01-01 12:00:00', TIMESTAMP '2024-01-01 12:01:00', 1.0),
          ('A', TIMESTAMP '2024-01-01 12:01:00', TIMESTAMP '2024-01-01 12:02:00', 2.0),
          ('A', TIMESTAMP '2024-01-01 13:00:30', TIMESTAMP '2024-01-01 13:01:30', 3.0)
        ) AS t(symbol, bar_time, available_at, ret)
        """
    )
    result = _windows(bounded, one_event, window=(-1, 1)).collect()
    np.testing.assert_array_equal(result.loc[0, "ret_window"], [1.0, 2.0])


def test_event_windows_validate_identity_and_timestamp_contracts() -> None:
    session, observations, events = _source_frames()
    duplicate_events = duckpd.concat([events, events])

    with pytest.raises(MaterializationError):
        _windows(observations, duplicate_events).collect()
    with pytest.raises(ValueError, match="window"):
        _windows(observations, events, window=(1, 1))
    with pytest.raises(UnsupportedOperationError, match="bar_label"):
        observations.event_windows(
            events,
            on="bar_time",
            bar_label="end",  # pyright: ignore[reportArgumentType]
            event_on="event_time",
            by="symbol",
            event_id="event_id",
            columns={"ret_window": "ret"},
            window=(-2, 1),
            step="PT1M",
            anchor="floor",
            available_at="available_at",
            event_available_at="event_available_at",
        )
    assert session.execution_count == 1


def test_event_windows_compose_with_native_embedding_and_exact_search() -> None:
    session, observations, events = _source_frames()
    candidates = _windows(observations, events[events["event_id"] != 30])
    representation = duckpd.series_representation(
        window=3,
        channels=("ret",),
        sampling="fixed_grid",
        step="PT1M",
        data_contract="market/simple-return/v1",
        normalization="none",
    )

    matches = candidates.embed_series(
        columns={"ret": "ret_window"},
        into="shape",
        representation=representation,
        null_policy="error",
    ).vector.search_series(
        {"ret": [1.0, 2.0, 3.0]},
        column="shape",
        representation=representation,
        metric="l2",
        k=1,
        tie_breaker="event_id",
    )

    assert session.execution_count == 0
    operations = json.loads(matches.explain(mode="json"))["execution_boundaries"][
        "embedding_operations"
    ]
    assert [operation["operation"] for operation in operations] == [
        "search_series",
        "embed_series",
        "event_windows",
    ]
    result = matches.collect()
    assert result["event_id"].tolist() == [10]
    assert result["_distance"].tolist() == [0.0]
    assert session.execution_count == 1


def test_event_windows_match_independent_utc_grid_oracle() -> None:
    session = duckpd.connect()
    start = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
    observations_data = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "bar_time": timestamp,
                "available_at": timestamp + pd.Timedelta(minutes=1, seconds=(index % 3) * 10),
                "ret": float(index + scale),
            }
            for symbol, scale in (("A", 0), ("B", 100))
            for index, timestamp in enumerate(pd.date_range(start, periods=8, freq="1min"))
            if not (symbol == "B" and index == 3)
        ]
    )
    events_data = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "symbol": ["A", "A", "B", "B"],
            "event_time": [
                start + pd.Timedelta(minutes=3, seconds=20),
                start + pd.Timedelta(minutes=5),
                start + pd.Timedelta(minutes=2, seconds=40),
                start + pd.Timedelta(minutes=6, seconds=10),
            ],
        }
    )
    events_data["event_available_at"] = events_data["event_time"] + pd.Timedelta(seconds=5)
    observations = session.from_pandas(observations_data)
    events = session.from_pandas(events_data)

    indexed = observations_data.set_index(["symbol", "bar_time"])
    for anchor in ("floor", "ceil"):
        result = _windows(
            observations,
            events,
            anchor=anchor,
            window=(-2, 2),
        ).collect()
        for row in result.itertuples(index=False):
            event = events_data.loc[events_data["event_id"] == row.event_id].iloc[0]
            anchor_time = (
                event["event_time"].floor("1min")
                if anchor == "floor"
                else event["event_time"].ceil("1min")
            )
            expected_times = [anchor_time + pd.Timedelta(minutes=offset) for offset in range(-2, 2)]
            available_rows = [
                indexed.loc[(event["symbol"], timestamp)]
                for timestamp in expected_times
                if (event["symbol"], timestamp) in indexed.index
            ]
            complete = len(available_rows) == 4

            assert row.window_window_start == expected_times[0]
            assert row.window_window_end == expected_times[-1] + pd.Timedelta(minutes=1)
            assert row.window_window_count == len(available_rows)
            assert row.window_window_complete is complete
            if complete:
                np.testing.assert_array_equal(
                    row.ret_window,
                    [float(value["ret"]) for value in available_rows],
                )
                assert row.window_window_available_at == max(
                    event["event_available_at"],
                    *(value["available_at"] for value in available_rows),
                )
            else:
                assert row.ret_window is pd.NA
                assert pd.isna(row.window_window_available_at)


def test_late_bar_availability_survives_output_cutoff_filter() -> None:
    _, observations, events = _source_frames()
    windows = _windows(observations, events[events["event_id"] != 30])

    eligible = windows[
        windows["window_window_available_at"] <= pd.Timestamp("2024-01-01 12:05:00")
    ].collect()

    assert eligible["event_id"].tolist() == [10]
    assert eligible["window_window_available_at"].tolist() == [pd.Timestamp("2024-01-01 12:03:00")]


def test_composite_event_identity_preserves_revisions_and_incomplete_rows() -> None:
    session, observations, _ = _source_frames()
    revisions = session.sql(
        """
        SELECT * FROM (VALUES
          (10, 1, 'A', TIMESTAMP '2024-01-01 12:02:30',
           TIMESTAMP '2024-01-01 12:02:31', 'initial'),
          (10, 2, 'A', TIMESTAMP '2024-01-01 12:02:30',
           TIMESTAMP '2024-01-01 12:04:00', 'corrected'),
          (30, 1, 'A', TIMESTAMP '2024-01-01 12:05:30',
           TIMESTAMP '2024-01-01 12:05:31', 'incomplete')
        ) AS t(event_id, revision, symbol, event_time, event_available_at, headline)
        """
    )

    windows = observations.event_windows(
        revisions,
        on="bar_time",
        bar_label="start",
        event_on="event_time",
        by="symbol",
        event_id=("event_id", "revision"),
        columns={"ret_window": "ret"},
        window=(-2, 1),
        step="PT1M",
        anchor="floor",
        available_at="available_at",
        event_available_at="event_available_at",
    ).collect()

    assert windows[["event_id", "revision"]].values.tolist() == [
        [10, 1],
        [10, 2],
        [30, 1],
    ]
    assert windows["window_window_available_at"].tolist()[:2] == [
        pd.Timestamp("2024-01-01 12:03:00"),
        pd.Timestamp("2024-01-01 12:04:00"),
    ]
    assert pd.isna(windows.loc[2, "window_window_available_at"])
    assert windows.loc[2, "ret_window"] is pd.NA


def test_exact_late_fusion_scores_full_eligible_set() -> None:
    session, observations, _ = _source_frames()
    events = session.sql(
        """
        SELECT * FROM (VALUES
          (10, 'A', TIMESTAMP '2024-01-01 12:02:30',
           TIMESTAMP '2024-01-01 12:02:31', [0.0, 1.0]::FLOAT[2]),
          (20, 'B', TIMESTAMP '2024-01-01 12:02:30',
           TIMESTAMP '2024-01-01 12:02:45', [1.0, 0.0]::FLOAT[2]),
          (99, 'A', TIMESTAMP '2024-01-01 12:02:30',
           TIMESTAMP '2024-01-01 12:02:31', [1.0, 0.0]::FLOAT[2])
        ) AS t(event_id, symbol, event_time, event_available_at, news_vector)
        """
    )
    representation = duckpd.series_representation(
        window=3,
        channels=("ret",),
        sampling="fixed_grid",
        step="PT1M",
        data_contract="market/simple-return/v1",
    )
    bank = _windows(observations, events).embed_series(
        columns={"ret": "ret_window"},
        into="reaction_vector",
        representation=representation,
        null_policy="error",
    )
    reaction_query = session.embed_series_query(
        {"ret": [1.0, 2.0, 3.0]},
        representation=representation,
    )
    eligible = bank[
        (bank["window_window_available_at"] <= pd.Timestamp("2024-01-01 12:15:00"))
        & bank["news_vector"].notna()
        & bank["reaction_vector"].notna()
    ]
    scored_with_query_event = eligible.assign(
        text_distance=lambda frame: frame["news_vector"].vector.distance([1.0, 0.0], metric="l2"),
        reaction_distance=lambda frame: frame["reaction_vector"].vector.distance(
            reaction_query, metric="l2"
        ),
    )
    scored_with_query_event = scored_with_query_event.assign(
        combined_distance=lambda frame: (
            0.4 * frame["text_distance"] + 0.6 * frame["reaction_distance"]
        )
    )
    exact = (
        scored_with_query_event[scored_with_query_event["event_id"] != 99]
        .sort_values("event_id")
        .nsmallest(1, ["combined_distance", "event_id"])
        .collect()
    )
    including_overlap = (
        scored_with_query_event.sort_values("event_id")
        .nsmallest(1, ["combined_distance", "event_id"])
        .collect()
    )
    text_limited = (
        eligible[eligible["event_id"] != 99]
        .vector.search(
            [1.0, 0.0],
            column="news_vector",
            metric="l2",
            k=1,
            tie_breaker="event_id",
        )
        .collect()
    )

    assert exact["event_id"].tolist() == [10]
    assert including_overlap["event_id"].tolist() == [99]
    assert text_limited["event_id"].tolist() == [20]
