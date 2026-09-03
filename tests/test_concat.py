"""Differential tests for duckpd.concat (row-wise union)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import duckpd
from duckpd.errors import (
    AlignmentError,
    UnorderedOperationError,
    UnsupportedOperationError,
)


@pytest.fixture
def df1() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 2],
            "b": ["x", "y"],
            "c": [10.0, 20.0],
        }
    )


@pytest.fixture
def df2() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [3, 4],
            "b": ["z", "w"],
            "c": [30.0, 40.0],
        }
    )


# --- Basic row-wise concatenation -------------------------------------------


def test_concat_identical_schemas_matches_pandas(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> None:
    session = duckpd.connect()
    f1 = session.from_pandas(df1)
    f2 = session.from_pandas(df2)

    result = duckpd.concat([f1, f2])
    assert session.execution_count == 0
    assert result.columns == ("a", "b", "c")

    expected = pd.concat([df1, df2]).reset_index(drop=True)
    assert_frame_equal(result.collect().reset_index(drop=True), expected)
    assert session.execution_count == 1


def test_concat_disjoint_columns_pads_nulls() -> None:
    d1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    d2 = pd.DataFrame({"b": ["z", "w"], "c": [10.0, 20.0]})

    session = duckpd.connect()
    f1 = session.from_pandas(d1)
    f2 = session.from_pandas(d2)

    result = duckpd.concat([f1, f2])
    assert session.execution_count == 0
    assert result.columns == ("a", "b", "c")

    expected = pd.concat([d1, d2]).reset_index(drop=True)
    assert_frame_equal(result.collect().reset_index(drop=True), expected)
    assert session.execution_count == 1


def test_concat_three_or_more_frames(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    df3 = pd.DataFrame({"a": [5], "b": ["v"], "c": [50.0]})

    session = duckpd.connect()
    f1 = session.from_pandas(df1)
    f2 = session.from_pandas(df2)
    f3 = session.from_pandas(df3)

    result = duckpd.concat([f1, f2, f3])
    expected = pd.concat([df1, df2, df3]).reset_index(drop=True)
    assert_frame_equal(result.collect().reset_index(drop=True), expected)


def test_concat_with_series() -> None:
    session = duckpd.connect()
    f = session.from_pandas(pd.DataFrame({"x": [1, 2]}))
    s = f["x"]

    # concat series and frame
    result = duckpd.concat([f, s])
    expected = pd.concat(
        [pd.DataFrame({"x": [1, 2]}), pd.Series([1, 2], name="x")]
    ).reset_index(drop=True)
    assert_frame_equal(result.collect().reset_index(drop=True), expected)


def test_concat_preserves_explicit_indexes() -> None:
    d1 = pd.DataFrame({"idx": ["r1", "r2"], "val": [10, 20]})
    d2 = pd.DataFrame({"idx": ["r3", "r4"], "val": [30, 40]})

    session = duckpd.connect()
    f1 = session.from_pandas(d1, index="idx")
    f2 = session.from_pandas(d2, index="idx")

    result = duckpd.concat([f1, f2], ignore_index=False)
    assert result.index_names == ("idx",)

    expected = pd.concat([d1.set_index("idx"), d2.set_index("idx")])
    assert_frame_equal(result.collect(), expected)


def test_concat_ignore_index_drops_index() -> None:
    d1 = pd.DataFrame({"idx": ["r1", "r2"], "val": [10, 20]})
    d2 = pd.DataFrame({"idx": ["r3", "r4"], "val": [30, 40]})

    session = duckpd.connect()
    f1 = session.from_pandas(d1, index="idx")
    f2 = session.from_pandas(d2, index="idx")

    result = duckpd.concat([f1, f2], ignore_index=True)
    assert result.index_names == ()


def test_concat_mixed_types_promotes_to_double() -> None:
    d1 = pd.DataFrame({"val": [1, 2]})  # integer
    d2 = pd.DataFrame({"val": [3.5, 4.5]})  # float

    session = duckpd.connect()
    f1 = session.from_pandas(d1)
    f2 = session.from_pandas(d2)

    result = duckpd.concat([f1, f2])
    expected = pd.concat([d1, d2]).reset_index(drop=True)
    assert_frame_equal(result.collect().reset_index(drop=True), expected)


def test_concat_preserves_nullable_uint64_exactly() -> None:
    first = pd.DataFrame({"value": pd.Series([2**63 + 1, None], dtype="UInt64")})
    second = pd.DataFrame({"value": pd.Series([2**63 + 2], dtype="UInt64")})

    result = duckpd.concat(
        [duckpd.from_pandas(first), duckpd.from_pandas(second)]
    ).collect()

    expected = pd.concat([first, second]).reset_index(drop=True)
    assert_frame_equal(result, expected)
    assert result["value"].tolist() == [2**63 + 1, pd.NA, 2**63 + 2]


def test_concat_promotes_integer_width_without_using_float() -> None:
    first = pd.DataFrame({"value": pd.Series([-1], dtype="int8")})
    second = pd.DataFrame({"value": pd.Series([255], dtype="uint8")})

    result = duckpd.concat(
        [duckpd.from_pandas(first), duckpd.from_pandas(second)]
    ).collect()

    expected = pd.concat([first, second]).reset_index(drop=True)
    assert_frame_equal(result, expected)


def test_concat_reconciles_decimal_precision_and_scale() -> None:
    first = pd.DataFrame({"value": [Decimal("1.25"), None]})
    second = pd.DataFrame({"value": [Decimal("123.4")]})

    result = duckpd.concat(
        [duckpd.from_pandas(first), duckpd.from_pandas(second)]
    ).collect()

    expected = pd.concat([first, second]).reset_index(drop=True)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([True, False], [False, True]),
        (["alpha", None], ["beta"]),
        ([b"a", None], [b"b"]),
        ([date(2024, 1, 1), None], [date(2024, 1, 2)]),
        (
            [pd.Timestamp("2024-01-01", tz="UTC"), pd.NaT],
            [pd.Timestamp("2024-01-02", tz="UTC")],
        ),
    ],
)
def test_concat_preserves_matching_non_numeric_types(
    first: list[object], second: list[object]
) -> None:
    left = pd.DataFrame({"value": first})
    right = pd.DataFrame({"value": second})

    result = duckpd.concat(
        [duckpd.from_pandas(left), duckpd.from_pandas(right)]
    ).collect()

    expected = pd.concat([left, right]).reset_index(drop=True)
    timezone_alias_differs = (
        isinstance(first[0], pd.Timestamp) and first[0].tzinfo is not None
    )
    assert_frame_equal(result, expected, check_dtype=not timezone_alias_differs)


@pytest.mark.parametrize(
    ("left_value", "right_value"),
    [
        (True, 1),
        ("1", 1),
        (b"1", "1"),
        (date(2024, 1, 1), pd.Timestamp("2024-01-01")),
        (-(2**63), 2**63),
    ],
)
def test_concat_rejects_lossy_dtype_reconciliation_before_execution(
    left_value: object, right_value: object
) -> None:
    session = duckpd.connect()
    left = session.from_pandas(pd.DataFrame({"value": [left_value]}))
    right = session.from_pandas(pd.DataFrame({"value": [right_value]}))

    with pytest.raises(UnsupportedOperationError, match="cannot losslessly reconcile"):
        duckpd.concat([left, right])

    assert session.execution_count == 0


def test_concat_preserves_typed_null_only_column() -> None:
    first = pd.DataFrame({"value": pd.Series([None], dtype="UInt64")})
    second = pd.DataFrame({"value": pd.Series([2**63 + 1], dtype="UInt64")})

    result = duckpd.concat(
        [duckpd.from_pandas(first), duckpd.from_pandas(second)]
    ).collect()

    expected = pd.concat([first, second]).reset_index(drop=True)
    assert_frame_equal(result, expected)


def test_concat_preserves_input_sequence_and_snapshot_order() -> None:
    first = pd.DataFrame({"key": [2, 1, 1], "value": [20, 10, 11]})
    second = pd.DataFrame({"key": [None, 1], "value": [99, 12]})
    session = duckpd.connect()
    combined = duckpd.concat([session.from_pandas(first), session.from_pandas(second)])

    expected = pd.concat([first, second], ignore_index=True)
    assert_frame_equal(combined.collect().reset_index(drop=True), expected)
    sliced = cast("duckpd.DataFrame", combined.iloc[1:4]).collect()
    assert_frame_equal(
        sliced.reset_index(drop=True), expected.iloc[1:4].reset_index(drop=True)
    )
    assert_frame_equal(
        combined[["value"]].cumsum().collect().reset_index(drop=True),
        expected[["value"]].cumsum(),
    )


def test_concat_declared_orders_are_stable_with_duplicate_and_null_keys() -> None:
    first = pd.DataFrame({"key": [2.0, 1.0, 1.0], "value": [20, 10, 11]})
    second = pd.DataFrame({"key": [None, 1.0], "value": [99, 12]})
    session = duckpd.connect()
    combined = duckpd.concat(
        [
            session.from_pandas(first, order_by="key"),
            session.from_pandas(second, order_by="key"),
        ]
    )

    expected = pd.concat(
        [
            first.sort_values("key", kind="stable"),
            second.sort_values("key", kind="stable"),
        ],
        ignore_index=True,
    )
    assert_frame_equal(combined.collect().reset_index(drop=True), expected)


def test_concat_with_unordered_input_rejects_positional_operations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unordered.csv"
    pd.DataFrame({"value": [3, 1, 2]}).to_csv(path, index=False)
    session = duckpd.connect()
    combined = duckpd.concat(
        [session.from_pandas(pd.DataFrame({"value": [0]})), session.read_csv(path)]
    )

    with pytest.raises(UnorderedOperationError):
        combined.iloc[1:]
    with pytest.raises(UnorderedOperationError):
        combined["value"].cumsum()


def test_concat_is_lazy(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    session = duckpd.connect()
    f1 = session.from_pandas(df1)
    f2 = session.from_pandas(df2)

    _ = duckpd.concat([f1, f2])
    assert session.execution_count == 0


def test_concat_pipeline_transformation(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    session = duckpd.connect()
    f1 = session.from_pandas(df1)
    f2 = session.from_pandas(df2)

    combined = (
        duckpd.concat([f1, f2])
        .assign(total=lambda f: f["a"] * f["c"])
        .sort_values("total", ascending=False)
        .limit(2)
    )

    expected = (
        pd.concat([df1, df2])
        .assign(total=lambda f: f["a"] * f["c"])
        .sort_values("total", ascending=False)
        .head(2)
        .reset_index(drop=True)
    )
    assert_frame_equal(combined.collect().reset_index(drop=True), expected)


# --- Error cases -------------------------------------------------------------


def test_concat_empty_objects_raises() -> None:
    with pytest.raises(ValueError, match="No objects to concatenate"):
        duckpd.concat([])


def test_concat_invalid_types_raises() -> None:
    with pytest.raises(TypeError, match="must be DataFrame or Series"):
        duckpd.concat(["not_a_df"])  # type: ignore[list-item]


def test_concat_different_sessions_raises_alignment_error(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> None:
    s1 = duckpd.connect()
    s2 = duckpd.connect()
    f1 = s1.from_pandas(df1)
    f2 = s2.from_pandas(df2)

    with pytest.raises(AlignmentError, match="different sessions"):
        duckpd.concat([f1, f2])


def test_concat_invalid_axis_raises(df1: pd.DataFrame) -> None:
    f1 = duckpd.from_pandas(df1)
    with pytest.raises(UnsupportedOperationError, match="axis=0"):
        duckpd.concat([f1, f1], axis=1)


def test_concat_invalid_join_raises(df1: pd.DataFrame) -> None:
    f1 = duckpd.from_pandas(df1)
    with pytest.raises(UnsupportedOperationError, match="join='outer'"):
        duckpd.concat([f1, f1], join="inner")
