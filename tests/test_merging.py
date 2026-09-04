from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import duckpd
from duckpd.errors import AlignmentError, MergeError, UnorderedOperationError

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


def test_outer_merge_promotes_plain_integer_and_boolean_nulls_like_pandas() -> None:
    left = pd.DataFrame({"key": [1, 2], "count": [10, 20], "flag": [True, False]})
    right = pd.DataFrame({"key": [2, 3], "value": [1, 2]})
    session = duckpd.connect()

    result = session.from_pandas(left).merge(
        session.from_pandas(right),
        on="key",
        how="outer",
        sort=True,
    )

    assert session.execution_count == 0
    assert_frame_equal(
        result.collect(), left.merge(right, on="key", how="outer", sort=True)
    )


@pytest.mark.parametrize("sort", [False, True])
def test_merge_does_not_claim_total_order_with_duplicate_keys(sort: bool) -> None:
    left = pd.DataFrame({"key": [2, 1, 1], "left_value": [20, 10, 11]})
    right = pd.DataFrame({"key": [1, 1, 2], "right_value": [100, 101, 200]})
    session = duckpd.connect()
    merged = session.from_pandas(left).merge(
        session.from_pandas(right), on="key", sort=sort
    )

    assert merged.ordering == ()
    with pytest.raises(UnorderedOperationError):
        merged.iloc[1:]
    with pytest.raises(UnorderedOperationError):
        merged["left_value"].cumsum()

    expected = left.merge(right, on="key", sort=sort)
    assert_frame_equal(
        merged.collect().sort_values(list(merged.columns)).reset_index(drop=True),
        expected.sort_values(list(expected.columns)).reset_index(drop=True),
    )


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


def test_merge_invalid_validate_raises_early() -> None:
    session = duckpd.connect()
    left = session.from_pandas(pd.DataFrame({"key": [1, 2]}))
    right = session.from_pandas(pd.DataFrame({"key": [1, 2]}))

    with pytest.raises(ValueError, match="is not a valid argument"):
        left.merge(right, on="key", validate="invalid_mode")

    assert session.execution_count == 0


@pytest.mark.parametrize(
    "val_param",
    ["1:1", "one_to_one"],
)
def test_merge_validate_one_to_one(val_param: str) -> None:
    session = duckpd.connect()
    left_unique = session.from_pandas(
        pd.DataFrame({"key": [1, 2], "val_l": ["a", "b"]})
    )
    right_unique = session.from_pandas(
        pd.DataFrame({"key": [1, 2], "val_r": ["x", "y"]})
    )
    left_dup = session.from_pandas(pd.DataFrame({"key": [1, 1], "val_l": ["a", "b"]}))
    right_dup = session.from_pandas(pd.DataFrame({"key": [1, 1], "val_r": ["x", "y"]}))

    # Passing 1:1
    merged = left_unique.merge(right_unique, on="key", validate=val_param)
    assert session.execution_count == 0  # Lazy until collect
    expected = pd.DataFrame({"key": [1, 2], "val_l": ["a", "b"], "val_r": ["x", "y"]})
    assert_frame_equal(merged.collect(), expected)

    # Left duplicates fail 1:1 (lazy creation does not execute)
    bad_left = left_dup.merge(right_unique, on="key", validate=val_param)
    assert session.execution_count == 3  # 2 validation checks + 1 collect
    with pytest.raises(
        MergeError,
        match="Merge keys are not unique in left dataset; not a one-to-one merge",
    ) as exc_info:
        bad_left.collect()
    assert isinstance(exc_info.value, ValueError)

    # Right duplicates fail 1:1
    bad_right = left_unique.merge(right_dup, on="key", validate=val_param)
    with pytest.raises(
        MergeError,
        match="Merge keys are not unique in right dataset; not a one-to-one merge",
    ):
        bad_right.collect()


@pytest.mark.parametrize(
    "val_param",
    ["1:m", "one_to_many"],
)
def test_merge_validate_one_to_many(val_param: str) -> None:
    session = duckpd.connect()
    left_unique = session.from_pandas(
        pd.DataFrame({"key": [1, 2], "val_l": ["a", "b"]})
    )
    right_dup = session.from_pandas(pd.DataFrame({"key": [1, 1], "val_r": ["x", "y"]}))
    left_dup = session.from_pandas(pd.DataFrame({"key": [1, 1], "val_l": ["a", "b"]}))

    # Passing 1:m
    merged = left_unique.merge(right_dup, on="key", validate=val_param)
    expected = pd.DataFrame({"key": [1, 1], "val_l": ["a", "a"], "val_r": ["x", "y"]})
    assert_frame_equal(
        merged.collect().sort_values(["val_r"]).reset_index(drop=True),
        expected.sort_values(["val_r"]).reset_index(drop=True),
    )
    bad = left_dup.merge(right_dup, on="key", validate=val_param)
    with pytest.raises(
        MergeError,
        match="Merge keys are not unique in left dataset; not a one-to-many merge",
    ):
        bad.collect()


@pytest.mark.parametrize(
    "val_param",
    ["m:1", "many_to_one"],
)
def test_merge_validate_many_to_one(val_param: str) -> None:
    session = duckpd.connect()
    left_dup = session.from_pandas(pd.DataFrame({"key": [1, 1], "val_l": ["a", "b"]}))
    right_unique = session.from_pandas(
        pd.DataFrame({"key": [1, 2], "val_r": ["x", "y"]})
    )
    right_dup = session.from_pandas(pd.DataFrame({"key": [1, 1], "val_r": ["x", "y"]}))

    # Passing m:1
    merged = left_dup.merge(right_unique, on="key", validate=val_param)
    assert_frame_equal(
        merged.collect(),
        pd.DataFrame({"key": [1, 1], "val_l": ["a", "b"], "val_r": ["x", "x"]}),
    )

    # Failing m:1 (right has duplicates)
    bad = left_dup.merge(right_dup, on="key", validate=val_param)
    with pytest.raises(
        MergeError,
        match="Merge keys are not unique in right dataset; not a many-to-one merge",
    ):
        bad.collect()


@pytest.mark.parametrize(
    "val_param",
    ["m:m", "many_to_many"],
)
def test_merge_validate_many_to_many(val_param: str) -> None:
    session = duckpd.connect()
    left_dup = session.from_pandas(pd.DataFrame({"key": [1, 1], "val_l": ["a", "b"]}))
    right_dup = session.from_pandas(pd.DataFrame({"key": [1, 1], "val_r": ["x", "y"]}))

    merged = left_dup.merge(right_dup, on="key", validate=val_param)
    assert len(merged.collect()) == 4


def test_merge_validate_with_null_keys() -> None:
    session = duckpd.connect()
    # Duplicate nulls in left dataset
    left_dup_null = session.from_pandas(
        pd.DataFrame({"key": [1.0, None, None], "val_l": ["a", "b", "c"]})
    )
    right = session.from_pandas(pd.DataFrame({"key": [1.0, 2.0], "val_r": ["x", "y"]}))

    with pytest.raises(
        MergeError,
        match="Merge keys are not unique in left dataset; not a one-to-one merge",
    ):
        left_dup_null.merge(right, on="key", validate="1:1").collect()

    # Single null in left and right dataset (unique!)
    left_single_null = session.from_pandas(
        pd.DataFrame({"key": [1.0, None], "val_l": ["a", "b"]})
    )
    right_single_null = session.from_pandas(
        pd.DataFrame({"key": [1.0, 2.0], "val_r": ["x", "y"]})
    )
    merged = left_single_null.merge(right_single_null, on="key", validate="1:1")
    assert len(merged.collect()) == 1


def test_merge_validate_composite_keys() -> None:
    session = duckpd.connect()
    left_unique_comp = session.from_pandas(
        pd.DataFrame({"k1": [1, 1], "k2": ["a", "b"], "v": [10, 20]})
    )
    right_unique_comp = session.from_pandas(
        pd.DataFrame({"k1": [1, 2], "k2": ["a", "b"], "w": [100, 200]})
    )
    left_dup_comp = session.from_pandas(
        pd.DataFrame({"k1": [1, 1], "k2": ["a", "a"], "v": [10, 20]})
    )

    # Composite unique succeeds
    merged = left_unique_comp.merge(right_unique_comp, on=["k1", "k2"], validate="1:1")
    assert len(merged.collect()) == 1

    # Composite duplicate fails
    bad = left_dup_comp.merge(right_unique_comp, on=["k1", "k2"], validate="1:1")
    with pytest.raises(
        MergeError,
        match="Merge keys are not unique in left dataset; not a one-to-one merge",
    ):
        bad.collect()


def test_merge_validate_cross_join() -> None:
    session = duckpd.connect()
    one_row_1 = session.from_pandas(pd.DataFrame({"a": [1]}))
    one_row_2 = session.from_pandas(pd.DataFrame({"b": [2]}))
    multi_row = session.from_pandas(pd.DataFrame({"a": [1, 2]}))

    # 1x1 cross join with 1:1 succeeds
    assert len(one_row_1.merge(one_row_2, how="cross", validate="1:1").collect()) == 1

    # multi-row cross join with 1:1 fails
    with pytest.raises(MergeError, match="Merge keys are not unique in left dataset"):
        multi_row.merge(one_row_2, how="cross", validate="1:1").collect()

    # multi-row with m:m succeeds
    assert len(multi_row.merge(multi_row, how="cross", validate="m:m").collect()) == 4


def test_join_shorthand_with_validate() -> None:
    session = duckpd.connect()
    left = session.from_pandas(
        pd.DataFrame({"idx": [1, 2], "val": [10, 20]}), index="idx"
    )
    right = session.from_pandas(
        pd.DataFrame({"idx": [1, 2], "val": [100, 200]}), index="idx"
    )
    bad_right = session.from_pandas(
        pd.DataFrame({"idx": [1, 1], "val": [100, 200]}), index="idx"
    )

    # Valid join
    res = left.join(right, lsuffix="_l", rsuffix="_r", validate="1:1").collect()
    assert len(res) == 2

    # Invalid join
    with pytest.raises(MergeError, match="Merge keys are not unique in right dataset"):
        left.join(bad_right, lsuffix="_l", rsuffix="_r", validate="1:1").collect()


def test_merge_validate_on_all_execution_boundaries(tmp_path: Path) -> None:
    session = duckpd.connect()
    left = session.from_pandas(pd.DataFrame({"key": [1, 1], "val": [1, 2]}))
    right = session.from_pandas(pd.DataFrame({"key": [1, 2], "val": [3, 4]}))
    merged = left.merge(right, on="key", suffixes=("_l", "_r"), validate="1:1")

    # explain() does not execute validation (remains a zero-scan plan inspect)
    plan_text = merged.explain()
    assert "Join" in plan_text or "HASH_JOIN" in plan_text

    # to_arrow fails
    with pytest.raises(MergeError):
        merged.to_arrow()

    # to_arrow_batches fails
    with pytest.raises(MergeError):
        merged.to_arrow_batches(batch_size=100)

    # write_parquet fails and does not produce a file
    parquet_out = tmp_path / "invalid.parquet"
    with pytest.raises(MergeError):
        merged.write_parquet(parquet_out)
    assert not parquet_out.exists()

    # write_csv fails
    csv_out = tmp_path / "invalid.csv"
    with pytest.raises(MergeError):
        merged.write_csv(csv_out)
    assert not csv_out.exists()

    # persist fails
    with pytest.raises(MergeError):
        merged.persist("invalid_stage")

    # reductions fail
    with pytest.raises(MergeError):
        merged["key"].count()


def test_nested_validation_reports_upstream_loc_error_first() -> None:
    session = duckpd.connect()
    left = session.from_pandas(
        pd.DataFrame({"id": [1, 2], "value": [10, 20]}), index="id"
    )
    right = session.from_pandas(
        pd.DataFrame({"id": [1, 2], "other": [100, 200]}), index="id"
    )
    missing = cast("duckpd.DataFrame", left.loc[[99, 99]])
    merged = missing.merge(
        right,
        left_index=True,
        right_index=True,
        validate="1:1",
    )

    with pytest.raises(KeyError, match="not in index"):
        merged.collect()
