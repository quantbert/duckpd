"""Differential tests for duckpd.concat (row-wise union)."""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import duckpd
from duckpd.errors import AlignmentError, UnsupportedOperationError


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
