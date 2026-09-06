from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import duckpd
from duckpd.errors import UnsupportedOperationError


def test_string_methods_match_pandas() -> None:
    source = pd.DataFrame(
        {
            "name": ["  apple ", "BANANA", "Cherry", None],
        }
    )
    frame = duckpd.from_pandas(source)

    transformed = frame.assign(
        upper=frame["name"].str.upper(),
        lower=frame["name"].str.lower(),
        stripped=frame["name"].str.strip(),
        length=frame["name"].str.len(),
        starts=frame["name"].str.startswith("B"),
        ends=frame["name"].str.endswith("y"),
        contains=frame["name"].str.contains("err"),
        replaced=frame["name"].str.replace("BANANA", "ORANGE"),
        starts_na=frame["name"].str.startswith("B", na=True),
        ends_na=frame["name"].str.endswith("y", na=True),
    )

    expected = source.assign(
        upper=source["name"].str.upper(),
        lower=source["name"].str.lower(),
        stripped=source["name"].str.strip(),
        length=source["name"].str.len(),
        starts=source["name"].str.startswith("B"),
        ends=source["name"].str.endswith("y"),
        contains=source["name"].str.contains("err"),
        replaced=source["name"].str.replace("BANANA", "ORANGE"),
        starts_na=source["name"].str.startswith("B", na=True),
        ends_na=source["name"].str.endswith("y", na=True),
    )

    assert_frame_equal(transformed.collect(), expected)


def test_string_contains_regex_and_literal_options_match_pandas() -> None:
    source = pd.DataFrame({"value": ["abc", "x.y", "ABC", None]})
    frame = duckpd.from_pandas(source)

    result = frame.assign(
        regex=frame["value"].str.contains("."),
        literal=frame["value"].str.contains(".", regex=False),
        insensitive=frame["value"].str.contains("abc", case=False),
        literal_insensitive=frame["value"].str.contains(".", case=False, regex=False),
        contains_na=frame["value"].str.contains("abc", na=True),
    )
    expected = source.assign(
        regex=source["value"].str.contains("."),
        literal=source["value"].str.contains(".", regex=False),
        insensitive=source["value"].str.contains("abc", case=False),
        literal_insensitive=source["value"].str.contains(".", case=False, regex=False),
        contains_na=source["value"].str.contains("abc", na=True),
    )

    assert_frame_equal(result.collect(), expected)

    with pytest.raises(UnsupportedOperationError, match="regex flags"):
        frame["value"].str.contains("abc", flags=2)


def test_datetime_properties_match_pandas() -> None:
    source = pd.DataFrame(
        {
            "ts": pd.Series(
                ["2023-01-15 14:30:45", "2023-06-20 09:15:00", None],
                dtype="datetime64[ns]",
            ),
        }
    )
    frame = duckpd.from_pandas(source)

    transformed = frame.assign(
        year=frame["ts"].dt.year,
        month=frame["ts"].dt.month,
        day=frame["ts"].dt.day,
        hour=frame["ts"].dt.hour,
        minute=frame["ts"].dt.minute,
        second=frame["ts"].dt.second,
        strftime=frame["ts"].dt.strftime("%Y/%m/%d"),
        to_period=frame["ts"].dt.to_period("M"),
    )

    expected = source.assign(
        year=source["ts"].dt.year,
        month=source["ts"].dt.month,
        day=source["ts"].dt.day,
        hour=source["ts"].dt.hour,
        minute=source["ts"].dt.minute,
        second=source["ts"].dt.second,
        strftime=source["ts"].dt.strftime("%Y/%m/%d"),
        to_period=source["ts"].dt.strftime("%Y-%m"),
    )

    assert_frame_equal(transformed.collect(), expected)


def test_fixed_temporal_rounding_matches_pandas_at_negative_and_tied_instants() -> None:
    source = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "1969-12-31 23:29:59.999999999",
                    "1970-01-01 00:30:00",
                    "1970-01-01 01:30:00",
                    None,
                ],
                format="mixed",
            )
        }
    )
    frame = duckpd.from_pandas(source)

    for operation in ("floor", "ceil", "round"):
        result = getattr(frame["ts"].dt, operation)("1h").collect()
        expected = getattr(source["ts"].dt, operation)("1h")
        assert_series_equal(result, expected)


def test_timezone_conversion_localization_and_utc_rounding_match_pandas() -> None:
    source = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2024-03-10 06:30Z", "2024-03-10 07:30Z", None],
            ).tz_convert("America/New_York")
        }
    )
    frame = duckpd.from_pandas(source)

    converted = frame["ts"].dt.tz_convert("UTC")
    assert_series_equal(converted.collect(), source["ts"].dt.tz_convert("UTC"))
    assert_series_equal(
        converted.dt.floor("1h").collect(),
        source["ts"].dt.tz_convert("UTC").dt.floor("1h"),
    )
    assert_series_equal(
        (frame["ts"] + timedelta(hours=2)).collect(),
        source["ts"] + timedelta(hours=2),
    )
    assert_series_equal(
        frame["ts"].dt.tz_localize(None).collect(),
        source["ts"].dt.tz_localize(None),
    )

    naive = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", None])})
    naive_frame = duckpd.from_pandas(naive)
    assert_series_equal(
        naive_frame["ts"].dt.tz_localize("UTC").collect(),
        naive["ts"].dt.tz_localize("UTC"),
    )


def test_temporal_arithmetic_and_duration_comparisons_match_pandas() -> None:
    source = pd.DataFrame(
        {
            "start": pd.Series(
                pd.to_datetime(["2024-01-01 00:00", "2024-01-02 12:00", None])
            ).dt.as_unit("us"),
            "end": pd.Series(
                pd.to_datetime(["2024-01-01 01:30", "2024-01-03 12:00", None])
            ).dt.as_unit("us"),
            "duration": pd.Series(pd.to_timedelta(["90min", "1D", None])).dt.as_unit("us"),
        }
    )
    frame = duckpd.from_pandas(source)

    assert_series_equal(
        (frame["start"] + timedelta(hours=2)).collect(),
        source["start"] + timedelta(hours=2),
    )
    assert_series_equal(
        (frame["end"] - frame["start"]).collect(),
        source["end"] - source["start"],
    )
    assert_series_equal(
        (frame["duration"] >= timedelta(hours=2)).collect(),
        source["duration"] >= timedelta(hours=2),
    )


def test_temporal_ambiguities_reject_before_execution() -> None:
    session = duckpd.connect()
    aware = pd.DataFrame(
        {"ts": pd.to_datetime(["2024-03-10 06:30Z"]).tz_convert("America/New_York")}
    )
    aware_frame = session.from_pandas(aware)
    naive_frame = session.from_pandas(pd.DataFrame({"ts": pd.to_datetime(["2024-01-01"])}))

    with pytest.raises(UnsupportedOperationError, match="tz_convert\\('UTC'\\)"):
        aware_frame["ts"].dt.floor("1h")
    with pytest.raises(UnsupportedOperationError, match="supports UTC only"):
        naive_frame["ts"].dt.tz_localize("America/New_York")
    with pytest.raises(TypeError, match="tz-naive"):
        naive_frame["ts"].dt.tz_convert("UTC")
    with pytest.raises(UnsupportedOperationError, match="Unsupported temporal"):
        naive_frame["ts"] + 1

    assert session.execution_count == 0


def test_temporal_validation_boundaries_reject_during_planning() -> None:
    session = duckpd.connect()
    naive = session.from_pandas(pd.DataFrame({"ts": pd.to_datetime(["2024-01-01"])}))
    aware = session.from_pandas(pd.DataFrame({"ts": pd.to_datetime(["2024-01-01"], utc=True)}))
    plain = session.from_pandas(pd.DataFrame({"value": ["x"]}))

    with pytest.raises(AttributeError, match="datetimelike"):
        _ = plain["value"].dt
    with pytest.raises(TypeError, match="fixed-duration"):
        naive["ts"].dt.floor(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive fixed duration"):
        naive["ts"].dt.floor("M")
    with pytest.raises(ValueError, match="positive fixed duration"):
        naive["ts"].dt.floor("0s")
    with pytest.raises(UnsupportedOperationError, match="ambiguous='raise'"):
        naive["ts"].dt.round("1h", ambiguous="infer")
    with pytest.raises(TypeError, match="tz=None"):
        naive["ts"].dt.tz_localize(None)
    with pytest.raises(TypeError, match="Already tz-aware"):
        aware["ts"].dt.tz_localize("UTC")
    with pytest.raises(TypeError, match="non-empty string"):
        aware["ts"].dt.tz_convert("")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown timezone"):
        aware["ts"].dt.tz_convert("Mars/Olympus")

    assert session.execution_count == 0


def test_categorical_validation_boundaries_reject_during_planning() -> None:
    session = duckpd.connect()
    frame = session.from_pandas(
        pd.DataFrame(
            {
                "left": pd.Categorical(["a"], categories=["a", "b"]),
                "right": pd.Categorical(["a"], categories=["a", "c"]),
                "plain": ["a"],
            }
        )
    )

    with pytest.raises(AttributeError, match="categorical Series"):
        _ = frame["plain"].cat
    with pytest.raises(TypeError, match="do not support arithmetic"):
        _ = frame["left"] + "a"
    with pytest.raises(TypeError, match="identical categories"):
        _ = frame["left"] == frame["right"]
    with pytest.raises(TypeError, match="outside the categories"):
        _ = frame["left"].cat.as_ordered() < "missing"

    assert session.execution_count == 0


def test_categorical_accessor_and_identity_transforms_preserve_metadata() -> None:
    source = pd.DataFrame(
        {
            "kind": pd.Categorical(
                ["medium", "low", None, "high"],
                categories=["medium", "low", "high", "unused"],
                ordered=False,
            ),
            "value": [2, 1, 0, 3],
        }
    )
    frame = duckpd.from_pandas(source)

    assert frame["kind"].cat.categories.equals(source["kind"].cat.categories)
    assert frame["kind"].cat.ordered is False
    assert_series_equal(frame["kind"].cat.codes.collect(), source["kind"].cat.codes)

    transformed = (
        frame[frame["value"] > 0]
        .assign(ordered_kind=lambda current: current["kind"].cat.as_ordered())
        .rename(columns={"kind": "renamed_kind"})
    )
    expected = (
        source[source["value"] > 0]
        .assign(ordered_kind=source["kind"].cat.as_ordered())
        .rename(columns={"kind": "renamed_kind"})
    )
    collected = transformed.collect()
    assert_frame_equal(collected, expected.reset_index(drop=True))
    assert collected["ordered_kind"].cat.ordered is True

    assert_frame_equal(
        frame.sort_values("kind").collect(),
        source.sort_values("kind").reset_index(drop=True),
    )

    other_source = pd.DataFrame(
        {
            "kind": pd.Categorical(
                ["unused"],
                categories=source["kind"].cat.categories,
                ordered=False,
            ),
            "value": [4],
        }
    )
    concatenated = duckpd.concat(
        [frame, duckpd.from_pandas(other_source)],
        ignore_index=True,
    )
    assert_frame_equal(
        concatenated.collect(),
        pd.concat([source, other_source], ignore_index=True),
    )

    ordered = frame["kind"].cat.as_ordered()
    assert_series_equal(
        (ordered < "high").collect(),
        source["kind"].cat.as_ordered() < "high",
    )
    with pytest.raises(TypeError, match="Unordered Categoricals"):
        _ = frame["kind"] < "high"

    numeric_source = pd.DataFrame(
        {
            "kind": pd.Categorical(
                [2, 1],
                categories=[2, 1, 3],
            )
        }
    )
    numeric = duckpd.from_pandas(numeric_source)
    assert_series_equal(
        (numeric["kind"].cat.as_ordered() < 3).collect(),
        numeric_source["kind"].cat.as_ordered() < 3,
    )
    unordered = ordered.cat.as_unordered().collect()
    assert unordered.dtype == source["kind"].dtype
    with pytest.raises(UnsupportedOperationError, match="string categories"):
        numeric.sort_values("kind")


def test_v01_acceptance_workflow_runs_end_to_end(tmp_path: object) -> None:
    from pathlib import Path

    if not isinstance(tmp_path, Path):
        raise TypeError("Expected tmp_path to be a Path")

    orders_df = pd.DataFrame(
        {
            "order_id": [f"ord_{i}" for i in range(1, 11)],
            "created_at": pd.to_datetime(
                [
                    "2023-01-10",
                    "2023-01-15",
                    "2023-01-20",
                    "2023-02-05",
                    "2023-02-12",
                    "2023-02-28",
                    "2023-03-01",
                    "2023-03-15",
                    "2023-03-20",
                    "2023-03-25",
                ]
            ),
            "status": [
                "paid",
                "pending",
                "paid",
                "paid",
                "refunded",
                "paid",
                "paid",
                "paid",
                "pending",
                "paid",
            ],
            "customer_id": [101, 102, 101, 103, 102, 101, 104, 103, 102, 104],
            "amount": [
                150.0,
                80.0,
                200.0,
                50.0,
                120.0,
                300.0,
                450.0,
                90.0,
                60.0,
                110.0,
            ],
            "refund_amount": [0.0, 0.0, 20.0, 0.0, 120.0, 50.0, 0.0, 10.0, 0.0, 0.0],
        }
    )

    parquet_file = tmp_path / "orders.parquet"
    out_file = tmp_path / "monthly.parquet"
    orders_df.to_parquet(parquet_file, index=False)

    with duckpd.connect() as session:
        orders = session.read_parquet(
            parquet_file,
            index="order_id",
            order_by=["created_at", "order_id"],
        )

        monthly = (
            orders[orders["status"] == "paid"]
            .assign(
                month=lambda frame: frame["created_at"].dt.to_period("M"),
                net_amount=lambda frame: frame["amount"] - frame["refund_amount"],
            )
            .groupby(["month", "customer_id"], as_index=False)
            .agg(
                revenue=("net_amount", "sum"),
                order_count=("order_id", "size"),
            )
            .sort_values(["month", "revenue"], ascending=[True, False])
        )

        # 1. explain()
        explanation = monthly.explain()
        assert "DuckPD logical plan:" in explanation
        assert "DuckDB SQL:" in explanation
        assert "DuckDB physical plan:" in explanation

        # 2. head(2)
        preview = monthly.head(2)
        assert len(preview) == 2

        # 3. collect()
        result = monthly.collect()

        # 4. write_parquet()
        monthly.write_parquet(out_file)
        assert out_file.exists()

    # Compare with pandas equivalent
    pandas_orders = pd.read_parquet(parquet_file)
    expected_monthly = (
        pandas_orders[pandas_orders["status"] == "paid"]
        .assign(
            month=lambda frame: frame["created_at"].dt.strftime("%Y-%m"),
            net_amount=lambda frame: frame["amount"] - frame["refund_amount"],
        )
        .groupby(["month", "customer_id"], as_index=False)
        .agg(
            revenue=("net_amount", "sum"),
            order_count=("order_id", "size"),
        )
        .sort_values(["month", "revenue"], ascending=[True, False])
        .reset_index(drop=True)
    )

    assert_frame_equal(result, expected_monthly)
    saved_result = pd.read_parquet(out_file)
    assert_frame_equal(saved_result, expected_monthly)


def test_accessor_invalid_arguments_raise_early() -> None:
    df = pd.DataFrame({"s": ["abc"], "ts": pd.Series(["2023-01-01"], dtype="datetime64[ns]")})
    frame = duckpd.from_pandas(df)

    with pytest.raises(UnsupportedOperationError, match="currently supports 'Y', 'M', 'D'"):
        frame["ts"].dt.to_period("Q")  # type: ignore[arg-type]
