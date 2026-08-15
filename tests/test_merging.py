from __future__ import annotations

from typing import Literal

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import duckpd
from duckpd.errors import AlignmentError

MergeHow = Literal["left", "right", "outer", "inner", "cross"]


@pytest.fixture
def left_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": [1, 2, 3],
            "val_l": ["a", "b", "c"],
            "shared": [10, 20, 30],
        }
    )


@pytest.fixture
def right_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": [2, 3, 4],
            "val_r": ["x", "y", "z"],
            "shared": [200, 300, 400],
        }
    )


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer", "cross"])
def test_merge_how_matches_pandas(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    how: MergeHow,
) -> None:
    session = duckpd.connect()
    l_frame = session.from_pandas(left_df)
    r_frame = session.from_pandas(right_df)

    if how == "cross":
        result = l_frame.merge(r_frame, how="cross", sort=True)
        expected = (
            left_df.merge(right_df, how="cross")
            .sort_values(["key_x", "key_y"])
            .reset_index(drop=True)
        )
        # duckpd sort in merge sorts by key columns
        assert_frame_equal(
            result.collect().sort_values(["key_x", "key_y"]).reset_index(drop=True),
            expected,
        )
    else:
        result = l_frame.merge(r_frame, on="key", how=how, sort=True)
        assert session.execution_count == 0

        expected = left_df.merge(right_df, on="key", how=how, sort=True)
        assert_frame_equal(result.collect(), expected)
        assert session.execution_count == 1


def test_merge_with_left_on_right_on_different_names() -> None:
    l_pd = pd.DataFrame({"lkey": [1, 2, 3], "val": ["a", "b", "c"]})
    r_pd = pd.DataFrame({"rkey": [2, 3, 4], "val": ["x", "y", "z"]})

    session = duckpd.connect()
    l_frame = session.from_pandas(l_pd)
    r_frame = session.from_pandas(r_pd)

    result = l_frame.merge(
        r_frame, left_on="lkey", right_on="rkey", how="inner", sort=True
    )
    expected = l_pd.merge(r_pd, left_on="lkey", right_on="rkey", how="inner", sort=True)

    assert_frame_equal(result.collect(), expected)


def test_merge_null_keys_match_pandas() -> None:
    l_pd = pd.DataFrame({"key": [1.0, None, 2.0], "val_l": ["a", "b", "c"]})
    r_pd = pd.DataFrame({"key": [1.0, None, 3.0], "val_r": ["x", "y", "z"]})

    session = duckpd.connect()
    l_frame = session.from_pandas(l_pd)
    r_frame = session.from_pandas(r_pd)

    # In pandas, null == null matches in merges
    result_inner = l_frame.merge(r_frame, on="key", how="inner", sort=True)
    expected_inner = l_pd.merge(r_pd, on="key", how="inner", sort=True)
    assert_frame_equal(result_inner.collect(), expected_inner)

    result_outer = l_frame.merge(r_frame, on="key", how="outer", sort=True)
    expected_outer = l_pd.merge(r_pd, on="key", how="outer", sort=True)
    assert_frame_equal(result_outer.collect(), expected_outer)


def test_merge_with_index_keys() -> None:
    l_pd = pd.DataFrame(
        {"val_l": ["a", "b", "c"]}, index=pd.Index([1, 2, 3], name="index")
    )
    r_pd = pd.DataFrame(
        {"val_r": ["x", "y", "z"]}, index=pd.Index([2, 3, 4], name="index")
    )

    session = duckpd.connect()
    l_frame_idx = session.from_pandas(l_pd.reset_index()).set_index("index")
    r_frame_idx = session.from_pandas(r_pd.reset_index()).set_index("index")

    result = l_frame_idx.merge(
        r_frame_idx, left_index=True, right_index=True, how="inner", sort=True
    )
    expected = l_pd.merge(
        r_pd, left_index=True, right_index=True, how="inner", sort=True
    )
    assert_frame_equal(result.collect(), expected)


def test_join_shorthand_matches_pandas() -> None:
    l_pd = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    r_pd = pd.DataFrame({"c": [1, 2, 3], "d": ["p", "q", "r"]})

    session = duckpd.connect()
    l_frame = session.from_pandas(l_pd)
    r_frame = session.from_pandas(r_pd)

    # join with left_index & right_index
    l_idx = l_frame.set_index("a")
    r_idx = r_frame.set_index("c")
    result = l_idx.join(r_idx, how="inner", sort=True)
    expected = l_pd.set_index("a").join(r_pd.set_index("c"), how="inner", sort=True)
    expected.index.name = "a"
    assert_frame_equal(result.collect(), expected)


def test_merge_cross_session_raises_alignment_error(
    left_df: pd.DataFrame, right_df: pd.DataFrame
) -> None:
    s1 = duckpd.connect()
    s2 = duckpd.connect()

    f1 = s1.from_pandas(left_df)
    f2 = s2.from_pandas(right_df)

    with pytest.raises(AlignmentError, match="different sessions"):
        f1.merge(f2, on="key")


def test_merge_invalid_arguments_raise_early(
    left_df: pd.DataFrame, right_df: pd.DataFrame
) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(left_df)
    other = session.from_pandas(right_df)

    with pytest.raises(ValueError, match="Invalid how"):
        frame.merge(other, how="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="two strings or None"):
        frame.merge(other, on="key", suffixes=("_x",))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Can not pass on"):
        frame.merge(other, how="cross", on="key")

    with pytest.raises(ValueError, match="Cannot pass on with left_on"):
        frame.merge(other, on="key", left_on="key")

    with pytest.raises(ValueError, match="left_index=True requires left frame"):
        frame.merge(other, left_index=True, right_on="key")


def test_merge_suffixes_reject_ambiguous_output() -> None:
    session = duckpd.connect()
    left = session.from_pandas(pd.DataFrame({"key": [1], "value": [2], "value_x": [3]}))
    right = session.from_pandas(pd.DataFrame({"key": [1], "value": [4]}))

    with pytest.raises(ValueError, match="no suffix"):
        left.merge(right, on="key", suffixes=(None, None))
    with pytest.raises(ValueError, match="duplicate columns"):
        left.merge(right, on="key", suffixes=("_x", "_x"))

    expected = pd.DataFrame({"key": [1], "value": [2], "value_x": [3], "value_r": [4]})
    result = left.merge(right, on="key", suffixes=(None, "_r"))
    assert_frame_equal(result.collect(), expected)
