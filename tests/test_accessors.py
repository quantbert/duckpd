from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

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
    )

    assert_frame_equal(transformed.collect(), expected)


def test_string_contains_regex_and_literal_options_match_pandas() -> None:
    source = pd.DataFrame({"value": ["abc", "x.y", "ABC", None]})
    frame = duckpd.from_pandas(source)

    result = frame.assign(
        regex=frame["value"].str.contains("."),
        literal=frame["value"].str.contains(".", regex=False),
        insensitive=frame["value"].str.contains("abc", case=False),
    )
    expected = source.assign(
        regex=source["value"].str.contains("."),
        literal=source["value"].str.contains(".", regex=False),
        insensitive=source["value"].str.contains("abc", case=False),
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
