"""Differential tests for single-frame pandas API methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import duckpd
from duckpd.errors import AlignmentError, UnsupportedOperationError


@pytest.fixture
def mixed_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "integer": [1, 2, 3],
            "floating": [1.0, np.nan, 3.0],
            "string": ["a", None, "c"],
            "boolean": [True, False, True],
        }
    )


# --- isna / notna -----------------------------------------------------------


def test_series_isna_matches_pandas(mixed_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(mixed_source)

    for label in mixed_source.columns:
        result = frame.assign(_check=frame[label].isna()).collect()["_check"]
        expected = mixed_source[label].isna()
        assert_series_equal(pd.Series(result, name=label), expected, check_dtype=False)
    assert session.execution_count == len(mixed_source.columns)


def test_series_notna_matches_pandas(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    for label in mixed_source.columns:
        result = frame.assign(_check=frame[label].notna()).collect()["_check"]
        expected = mixed_source[label].notna()
        assert_series_equal(pd.Series(result, name=label), expected, check_dtype=False)


def test_series_isna_notna_are_lazy(mixed_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(mixed_source)

    _ = frame["floating"].isna()
    _ = frame["floating"].notna()
    _ = frame["floating"].isnull()
    _ = frame["floating"].notnull()

    assert session.execution_count == 0


def test_dataframe_isna_matches_pandas(mixed_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(mixed_source)

    result = frame.isna()
    assert session.execution_count == 0

    expected = mixed_source.isna()
    assert_frame_equal(result.collect(), expected, check_dtype=False)
    assert session.execution_count == 1


def test_dataframe_notna_matches_pandas(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    result = frame.notna()
    expected = mixed_source.notna()
    assert_frame_equal(result.collect(), expected, check_dtype=False)


def test_dataframe_isna_preserves_index() -> None:
    source = pd.DataFrame({"a": [1, None], "b": [None, 2]})
    frame = duckpd.from_pandas(source).set_index("a")

    result = frame.isna().collect()
    expected = source.set_index("a").isna()
    assert_frame_equal(result, expected, check_dtype=False)


# --- rename -----------------------------------------------------------------


def test_rename_columns_dict_matches_pandas(mixed_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(mixed_source)

    result = frame.rename(columns={"integer": "int", "string": "text"})
    assert session.execution_count == 0
    assert result.columns == ("int", "floating", "text", "boolean")

    expected = mixed_source.rename(columns={"integer": "int", "string": "text"})
    assert_frame_equal(result.collect(), expected, check_dtype=False)
    assert session.execution_count == 1


def test_rename_with_mapper_kwarg_matches_pandas(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    result = frame.rename(mapper={"integer": "int"}, axis="columns")
    expected = mixed_source.rename(columns={"integer": "int"})
    assert_frame_equal(result.collect(), expected, check_dtype=False)


def test_rename_preserves_index_metadata() -> None:
    source = pd.DataFrame({"row_id": [1, 2], "value": [10, 20]})
    frame = duckpd.from_pandas(source, index="row_id", order_by="row_id")

    result = frame.rename(columns={"value": "amount"})

    assert result.columns == ("amount",)
    assert result.index_names == ("row_id",)
    assert result.ordering == ("row_id",)


def test_rename_no_op_returns_self(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    result = frame.rename(columns=None)

    assert result is frame


def test_rename_empty_mapping_returns_self(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    result = frame.rename(columns={})

    assert result is frame


def test_rename_missing_label_raises(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    with pytest.raises(KeyError):
        frame.rename(columns={"missing": "new"})


def test_rename_missing_label_ignored(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    result = frame.rename(columns={"missing": "new"}, errors="ignore")
    assert result.columns == tuple(mixed_source.columns)


def test_rename_rejects_duplicate_output_labels() -> None:
    frame = duckpd.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))

    with pytest.raises(ValueError, match="duplicate column labels"):
        frame.rename(columns={"a": "value", "b": "value"})


def test_rename_invalid_arguments_raise_early(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    with pytest.raises(UnsupportedOperationError, match="copy=False"):
        frame.rename(columns={"integer": "int"}, copy=False)
    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        frame.rename(columns={"integer": "int"}, inplace=True)
    with pytest.raises(UnsupportedOperationError, match="MultiIndex"):
        frame.rename(columns={"integer": "int"}, level=0)
    with pytest.raises(UnsupportedOperationError, match="renaming index"):
        frame.rename(index={"a": "b"})
    with pytest.raises(TypeError, match="both mapper and columns"):
        frame.rename(mapper={"a": "b"}, columns={"c": "d"})
    with pytest.raises(TypeError, match="only a dict"):
        frame.rename(mapper=["a", "b"], axis="columns")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="errors must be"):
        frame.rename(columns={"a": "b"}, errors="invalid")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedOperationError, match="renaming index"):
        frame.rename(mapper={"a": "b"})


# --- drop -------------------------------------------------------------------


def test_drop_single_column_matches_pandas(mixed_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(mixed_source)

    result = frame.drop(columns="integer")
    assert session.execution_count == 0
    assert result.columns == ("floating", "string", "boolean")

    expected = mixed_source.drop(columns="integer")
    assert_frame_equal(result.collect(), expected)
    assert session.execution_count == 1


def test_drop_multiple_columns_matches_pandas(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    result = frame.drop(["integer", "string"], axis=1)
    expected = mixed_source.drop(columns=["integer", "string"])
    assert_frame_equal(result.collect(), expected)


def test_drop_with_columns_kwarg_matches_pandas(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    result = frame.drop(columns=["floating", "boolean"])
    expected = mixed_source.drop(columns=["floating", "boolean"])
    assert_frame_equal(result.collect(), expected)


def test_drop_preserves_index_metadata() -> None:
    source = pd.DataFrame({"row_id": [1, 2, 3], "seq": [30, 10, 20], "value": [8, 6, 7]})
    frame = duckpd.from_pandas(source, index="row_id", order_by="seq")

    result = frame.drop(columns="value")

    assert result.columns == ("seq",)
    assert result.index_names == ("row_id",)
    assert result.ordering == ("seq",)


def test_drop_missing_label_raises(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    with pytest.raises(KeyError):
        frame.drop(columns="missing")


def test_drop_missing_label_ignored(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    result = frame.drop(columns="missing", errors="ignore")
    assert result.columns == tuple(mixed_source.columns)


def test_drop_all_columns_raises(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    with pytest.raises(ValueError, match="empty projections"):
        frame.drop(columns=list(mixed_source.columns))


def test_drop_invalid_arguments_raise_early(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        frame.drop("integer", inplace=True)
    with pytest.raises(UnsupportedOperationError, match="MultiIndex"):
        frame.drop("integer", level=0)
    with pytest.raises(UnsupportedOperationError, match="dropping rows by index"):
        frame.drop(index=["a"])
    with pytest.raises(ValueError, match="errors must be"):
        frame.drop("integer", errors="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="axis must be"):
        frame.drop("integer", axis="invalid")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedOperationError, match="dropping rows by index"):
        frame.drop("integer")


def test_drop_empty_labels_returns_self(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    result = frame.drop(columns=[])

    assert result is frame


# --- astype -----------------------------------------------------------------


def test_series_astype_matches_pandas() -> None:
    source = pd.DataFrame({"val": [1, 2, 3]})
    session = duckpd.connect()
    frame = session.from_pandas(source)

    float_series = frame.assign(converted=frame["val"].astype("float64"))
    assert session.execution_count == 0
    expected = source.assign(converted=source["val"].astype("float64"))
    assert_frame_equal(float_series.collect(), expected)
    assert session.execution_count == 1

    str_series = frame.assign(converted=frame["val"].astype("str"))
    expected_str = source.assign(converted=source["val"].astype("str"))
    assert_frame_equal(str_series.collect(), expected_str)


def test_series_astype_errors_ignore() -> None:
    source = pd.DataFrame({"val": [1, 2, 3]})
    frame = duckpd.from_pandas(source)

    same_series = frame.assign(converted=frame["val"].astype("invalid_type_name", errors="ignore"))
    assert_frame_equal(same_series.collect(), source.assign(converted=source["val"]))


def test_series_astype_invalid_raises() -> None:
    source = pd.DataFrame({"val": [1, 2, 3]})
    frame = duckpd.from_pandas(source)

    with pytest.raises(TypeError, match="Unsupported dtype"):
        frame["val"].astype("invalid_type_name")
    with pytest.raises(UnsupportedOperationError, match="copy=False"):
        frame["val"].astype("float64", copy=False)
    with pytest.raises(ValueError, match="errors must be"):
        frame["val"].astype("float64", errors="invalid")  # type: ignore[arg-type]


def test_dataframe_astype_scalar_type_matches_pandas() -> None:
    source = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    session = duckpd.connect()
    frame = session.from_pandas(source)

    result = frame.astype("float64")
    assert session.execution_count == 0
    expected = source.astype("float64")
    assert_frame_equal(result.collect(), expected)
    assert session.execution_count == 1


def test_dataframe_astype_dict_matches_pandas() -> None:
    source = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    frame = duckpd.from_pandas(source)

    result = frame.astype({"a": "float64", "b": "str"})
    expected = source.astype({"a": "float64", "b": "str"})
    assert_frame_equal(result.collect(), expected)


def test_dataframe_astype_preserves_index() -> None:
    source = pd.DataFrame({"idx": [10, 20], "val": [1, 2]})
    frame = duckpd.from_pandas(source, index="idx").astype("float64")

    expected = source.set_index("idx").astype("float64")
    assert_frame_equal(frame.collect(), expected)


def test_dataframe_astype_invalid_raises() -> None:
    source = pd.DataFrame({"a": [1, 2]})
    frame = duckpd.from_pandas(source)

    with pytest.raises(TypeError, match="Unsupported dtype"):
        frame.astype("invalid_dtype")
    with pytest.raises(UnsupportedOperationError, match="copy=False"):
        frame.astype("float64", copy=False)


# --- fillna -----------------------------------------------------------------


def test_series_fillna_matches_pandas() -> None:
    source = pd.DataFrame({"val": [1.0, np.nan, 3.0]})
    session = duckpd.connect()
    frame = session.from_pandas(source)

    filled = frame.assign(res=frame["val"].fillna(0.0))
    assert session.execution_count == 0
    expected = source.assign(res=source["val"].fillna(0.0))
    assert_frame_equal(filled.collect(), expected)
    assert session.execution_count == 1


def test_series_fillna_invalid_raises() -> None:
    source = pd.DataFrame({"val": [1.0, np.nan, 3.0]})
    frame = duckpd.from_pandas(source)

    with pytest.raises(ValueError, match="Must specify a value"):
        frame["val"].fillna()
    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        frame["val"].fillna(0.0, inplace=True)
    with pytest.raises(UnsupportedOperationError, match="limit"):
        frame["val"].fillna(0.0, limit=1)


def test_dataframe_fillna_scalar_matches_pandas() -> None:
    source = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, np.nan, 30.0]})
    session = duckpd.connect()
    frame = session.from_pandas(source)

    result = frame.fillna(0.0)
    assert session.execution_count == 0
    expected = source.fillna(0.0)
    assert_frame_equal(result.collect(), expected)
    assert session.execution_count == 1


def test_dataframe_fillna_dict_matches_pandas() -> None:
    source = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, 20.0, np.nan]})
    frame = duckpd.from_pandas(source)

    result = frame.fillna({"a": 0.0, "b": -1.0})
    expected = source.fillna({"a": 0.0, "b": -1.0})
    assert_frame_equal(result.collect(), expected)


def test_dataframe_fillna_preserves_index() -> None:
    source = pd.DataFrame({"idx": [1, 2], "val": [1.0, np.nan]})
    frame = duckpd.from_pandas(source, index="idx").fillna(0.0)

    expected = source.set_index("idx").fillna(0.0)
    assert_frame_equal(frame.collect(), expected)


def test_dataframe_fillna_invalid_raises() -> None:
    source = pd.DataFrame({"a": [1.0, np.nan]})
    frame = duckpd.from_pandas(source)

    with pytest.raises(ValueError, match="Must specify a value"):
        frame.fillna()
    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        frame.fillna(0.0, inplace=True)
    with pytest.raises(UnsupportedOperationError, match="limit"):
        frame.fillna(0.0, limit=1)
    with pytest.raises(UnsupportedOperationError, match="axis=0"):
        frame.fillna(0.0, axis=1)


# --- dropna -----------------------------------------------------------------


def test_series_dropna_matches_pandas() -> None:
    source = pd.DataFrame({"val": [1.0, np.nan, 3.0, np.nan, 5.0]})
    session = duckpd.connect()
    frame = session.from_pandas(source)

    dropped = frame["val"].dropna()
    assert session.execution_count == 0

    dropped_frame = duckpd.DataFrame(dropped._session, dropped._plan)
    res_df = dropped_frame.collect()
    exp_df = source.dropna().reset_index(drop=True)
    assert_frame_equal(res_df.reset_index(drop=True), exp_df)


def test_series_dropna_invalid_raises() -> None:
    source = pd.DataFrame({"val": [1.0, np.nan]})
    frame = duckpd.from_pandas(source)

    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        frame["val"].dropna(inplace=True)
    with pytest.raises(UnsupportedOperationError, match="ignore_index=True"):
        frame["val"].dropna(ignore_index=True)
    with pytest.raises(ValueError, match="axis"):
        frame["val"].dropna(axis=1)  # type: ignore[arg-type]


def test_dataframe_dropna_how_any_matches_pandas() -> None:
    source = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [10.0, 20.0, 30.0, np.nan],
        }
    )
    session = duckpd.connect()
    frame = session.from_pandas(source)

    result = frame.dropna(how="any")
    assert session.execution_count == 0

    expected = source.dropna(how="any").reset_index(drop=True)
    assert_frame_equal(result.collect().reset_index(drop=True), expected)
    assert session.execution_count == 1


def test_dataframe_dropna_how_all_matches_pandas() -> None:
    source = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [10.0, 20.0, np.nan, np.nan],
        }
    )
    frame = duckpd.from_pandas(source)

    result = frame.dropna(how="all")
    expected = source.dropna(how="all").reset_index(drop=True)
    assert_frame_equal(result.collect().reset_index(drop=True), expected)


def test_dataframe_dropna_subset_matches_pandas() -> None:
    source = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [np.nan, 20.0, 30.0, np.nan],
        }
    )
    frame = duckpd.from_pandas(source)

    result = frame.dropna(subset=["a"])
    expected = source.dropna(subset=["a"]).reset_index(drop=True)
    assert_frame_equal(result.collect().reset_index(drop=True), expected)


def test_dataframe_dropna_thresh_matches_pandas() -> None:
    source = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [10.0, np.nan, 30.0, 40.0],
            "c": [np.nan, np.nan, 300.0, 400.0],
        }
    )
    frame = duckpd.from_pandas(source)

    result = frame.dropna(thresh=2)
    expected = source.dropna(thresh=2).reset_index(drop=True)
    assert_frame_equal(result.collect().reset_index(drop=True), expected)


def test_dataframe_dropna_invalid_raises() -> None:
    source = pd.DataFrame({"a": [1.0, np.nan]})
    frame = duckpd.from_pandas(source)

    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        frame.dropna(inplace=True)
    with pytest.raises(UnsupportedOperationError, match="ignore_index=True"):
        frame.dropna(ignore_index=True)
    with pytest.raises(ValueError, match="how must be"):
        frame.dropna(how="invalid")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedOperationError, match="axis=1"):
        frame.dropna(axis=1)
    with pytest.raises(TypeError, match="both how and thresh"):
        frame.dropna(how="any", thresh=1)
    with pytest.raises(TypeError, match="thresh must be an integer"):
        frame.dropna(thresh=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        frame.dropna(thresh=-1)


# --- where / mask -----------------------------------------------------------


def test_series_where_matches_pandas() -> None:
    source = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
    session = duckpd.connect()
    frame = session.from_pandas(source)

    s = frame["val"]
    res = frame.assign(out=s.where(s > 2, 0))
    assert session.execution_count == 0

    exp = source.assign(out=source["val"].where(source["val"] > 2, 0))
    assert_frame_equal(res.collect(), exp)
    assert session.execution_count == 1


def test_series_mask_matches_pandas() -> None:
    source = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
    frame = duckpd.from_pandas(source)

    s = frame["val"]
    res = frame.assign(out=s.mask(s > 2, 0))
    exp = source.assign(out=source["val"].mask(source["val"] > 2, 0))
    assert_frame_equal(res.collect(), exp)


def test_dataframe_where_matches_pandas() -> None:
    source = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    session = duckpd.connect()
    frame = session.from_pandas(source)

    res = frame.where(frame["a"] > 1, 0)
    assert session.execution_count == 0

    exp = source.where(source["a"] > 1, 0)
    assert_frame_equal(res.collect(), exp)
    assert session.execution_count == 1


def test_dataframe_mask_matches_pandas() -> None:
    source = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    frame = duckpd.from_pandas(source)

    res = frame.mask(frame["a"] > 1, -1)
    exp = source.mask(source["a"] > 1, -1)
    assert_frame_equal(res.collect(), exp)


def test_dataframe_where_mask_various_types() -> None:
    source = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    frame = duckpd.from_pandas(source)

    # other as dict (pandas doesn't support dict `other` in where,
    # so verify result directly against expected DataFrame)
    res_dict = frame.where(frame["a"] > 1, {"a": -1, "b": -2})
    exp_dict = pd.DataFrame({"a": [-1, 2, 3], "b": [-2, 5, 6]})
    assert_frame_equal(res_dict.collect(), exp_dict, check_dtype=False)

    # other as Series
    res_ser = frame.where(frame["a"] > 1, frame["b"])
    assert res_ser.columns == ("a", "b")

    # cond as bool
    res_bool_t = frame.where(True, 0)
    assert_frame_equal(res_bool_t.collect(), source, check_dtype=False)
    res_bool_f = frame.where(False, 0)
    assert_frame_equal(
        res_bool_f.collect(),
        pd.DataFrame({"a": [0, 0, 0], "b": [0, 0, 0]}),
        check_dtype=False,
    )

    # cond as Series from same frame
    res_df = frame.where(frame["a"] > 1, 0)
    assert res_df.columns == ("a", "b")


def test_dataframe_where_mask_invalid_raises() -> None:
    source = pd.DataFrame({"a": [1, 2]})
    frame = duckpd.from_pandas(source)

    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        frame.where(True, inplace=True)
    with pytest.raises(UnsupportedOperationError, match="axis=0"):
        frame.where(True, axis=1)
    with pytest.raises(TypeError, match="cond must be"):
        frame.where("invalid_cond")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="other must be"):
        frame.where(True, other=object())

    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        frame.mask(True, inplace=True)
    with pytest.raises(UnsupportedOperationError, match="axis=0"):
        frame.mask(True, axis=1)


def test_series_where_mask_various_types() -> None:
    source = pd.DataFrame({"a": [1, 2, 3]})
    frame = duckpd.from_pandas(source)

    # cond as bool
    res_t = frame.assign(out=frame["a"].where(True, 0))
    assert_frame_equal(res_t.collect(), source.assign(out=[1, 2, 3]))

    res_f = frame.assign(out=frame["a"].where(False, 0))
    assert_frame_equal(res_f.collect(), source.assign(out=[0, 0, 0]))

    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        frame["a"].where(True, inplace=True)
    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        frame["a"].mask(True, inplace=True)
    with pytest.raises(TypeError, match="cond must be"):
        frame["a"].where("invalid")  # type: ignore[arg-type]


def test_where_mask_cross_frame_alignment_error() -> None:
    s1 = duckpd.from_pandas(pd.DataFrame({"a": [1, 2]}))
    s2 = duckpd.from_pandas(pd.DataFrame({"a": [1, 2]}))

    with pytest.raises(AlignmentError, match="explicit index alignment"):
        s1["a"].where(s2["a"] > 0)
    with pytest.raises(AlignmentError, match="explicit index alignment"):
        s1.where(s2["a"] > 0)


# --- combined pipeline ------------------------------------------------------


def test_isna_rename_drop_pipeline_matches_pandas(mixed_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(mixed_source)

    result = frame.rename(columns={"integer": "int"}).drop(columns="boolean").isna().collect()

    expected = mixed_source.rename(columns={"integer": "int"}).drop(columns="boolean").isna()
    assert_frame_equal(result, expected, check_dtype=False)


# --- clip and replace tests -------------------------------------------------


def test_series_clip_matches_pandas() -> None:
    source = pd.Series([1.0, 5.0, 10.0, None, -3.0, 20.0], name="x")
    frame = duckpd.from_pandas(pd.DataFrame({"x": source}))
    s = frame["x"]

    # lower and upper
    res1 = s.clip(lower=0, upper=10).collect()
    exp1 = source.clip(lower=0, upper=10)
    assert_series_equal(res1, exp1)

    # lower only
    res2 = s.clip(lower=2).collect()
    exp2 = source.clip(lower=2)
    assert_series_equal(res2, exp2)

    # upper only
    res3 = s.clip(upper=8).collect()
    exp3 = source.clip(upper=8)
    assert_series_equal(res3, exp3)

    # invalid lower > upper
    with pytest.raises(ValueError, match="Cannot set lower > upper"):
        s.clip(lower=10, upper=5)

    # inplace=True raises
    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        s.clip(lower=0, upper=10, inplace=True)


def test_series_clip_with_series_bounds() -> None:
    pdf = pd.DataFrame({"val": [10, 20, 30], "low": [15, 15, 15], "high": [25, 25, 25]})
    df = duckpd.from_pandas(pdf)
    res = df["val"].clip(lower=df["low"], upper=df["high"]).collect()
    exp = pdf["val"].clip(lower=pdf["low"], upper=pdf["high"])
    assert_series_equal(res, exp)

    # different frame raises AlignmentError
    df2 = duckpd.from_pandas(pdf)
    with pytest.raises(AlignmentError, match="different frames"):
        df["val"].clip(lower=df2["low"])


def test_dataframe_clip_matches_pandas() -> None:
    pdf = pd.DataFrame(
        {
            "a": [1, 5, 20, None, -3],
            "b": [10.5, 20.0, 30.5, 40.0, 50.0],
        }
    )
    df = duckpd.from_pandas(pdf)

    # scalar bounds
    res = df.clip(lower=2, upper=25).collect()
    exp = pdf.clip(lower=2, upper=25)
    assert_frame_equal(res, exp)

    # dict bounds per column
    res_dict = df.clip(lower={"a": 0, "b": 15}, upper={"a": 10, "b": 35}).collect()
    exp_dict = pdf.clip(
        lower=pd.Series({"a": 0, "b": 15}),
        upper=pd.Series({"a": 10, "b": 35}),
        axis=1,
    )
    assert_frame_equal(res_dict, exp_dict)

    # invalid lower > upper
    with pytest.raises(ValueError, match="Cannot set lower > upper"):
        df.clip(lower=10, upper=5)

    # inplace raises
    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        df.clip(lower=0, upper=10, inplace=True)


def test_series_replace_matches_pandas() -> None:
    source = pd.Series(["apple", "banana", "cherry", "banana", None], name="fruit")
    df = duckpd.from_pandas(pd.DataFrame({"fruit": source}))
    s = df["fruit"]

    # dict mapping
    res1 = s.replace({"banana": "BANANA", "cherry": "CHERRY"}).collect()
    exp1 = source.replace({"banana": "BANANA", "cherry": "CHERRY"})
    assert_series_equal(res1, exp1)

    # scalar replacement
    res2 = s.replace("apple", "APPLE").collect()
    exp2 = source.replace("apple", "APPLE")
    assert_series_equal(res2, exp2)

    # list replacement to scalar
    num_s = pd.Series([1, 2, 3, 2, 1], name="n")
    num_df = duckpd.from_pandas(pd.DataFrame({"n": num_s}))
    res3 = num_df["n"].replace([1, 2], 0).collect()
    exp3 = num_s.replace([1, 2], 0)
    assert_series_equal(res3, exp3)

    # list replacement to list
    res4 = num_df["n"].replace([1, 2], [10, 20]).collect()
    exp4 = num_s.replace([1, 2], [10, 20])
    assert_series_equal(res4, exp4)

    # inplace raises
    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        s.replace("apple", "APPLE", inplace=True)


def test_dataframe_replace_matches_pandas() -> None:
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [10, 20, 30],
        }
    )
    df = duckpd.from_pandas(pdf)

    # column-specific nested dict
    res1 = df.replace({"a": {1: 100, 2: 200}, "b": {"x": "X"}}).collect()
    exp1 = pdf.replace({"a": {1: 100, 2: 200}, "b": {"x": "X"}})
    assert_frame_equal(res1, exp1)

    # global dict across columns
    res2 = df.replace({1: 100, 2: 200, "x": "XX"}).collect()
    exp2 = pdf.replace({1: 100, 2: 200, "x": "XX"})
    assert_frame_equal(res2, exp2)

    # scalar replacement
    res3 = df.replace(1, 999).collect()
    exp3 = pdf.replace(1, 999)
    assert_frame_equal(res3, exp3)

    # list replacement to list mismatch error
    with pytest.raises(ValueError, match="Replacement list lengths must match"):
        df.replace([1, 2], [10])


def test_clip_replace_and_rename_edge_cases() -> None:
    pdf = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df = duckpd.from_pandas(pdf)
    s = df["a"]

    # clip with no bounds
    assert s.clip() is s
    assert df.clip() is df

    # clip invalid axis
    with pytest.raises(UnsupportedOperationError, match="axis=0"):
        s.clip(lower=0, axis=1)
    with pytest.raises(UnsupportedOperationError, match="axis=0"):
        df.clip(lower=0, axis=1)

    # replace with no args
    assert s.replace() is s
    assert df.replace() is df

    # replace unsupported parameters
    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        df.replace(1, 10, inplace=True)
    with pytest.raises(UnsupportedOperationError, match="regex=True"):
        s.replace(1, 10, regex=True)
    with pytest.raises(UnsupportedOperationError, match="regex=True"):
        df.replace(1, 10, regex=True)
    with pytest.raises(UnsupportedOperationError, match="limit"):
        s.replace(1, 10, limit=1)
    with pytest.raises(UnsupportedOperationError, match="limit"):
        df.replace(1, 10, limit=1)
    with pytest.raises(UnsupportedOperationError, match="method"):
        s.replace(1, 10, method="pad")
    with pytest.raises(UnsupportedOperationError, match="method"):
        df.replace(1, 10, method="pad")

    # Series replace length mismatch
    with pytest.raises(ValueError, match="lengths must match"):
        s.replace([1, 2], [10])

    # DataFrame replace column dict with scalar value
    res_col_scalar = df.replace({"a": 1}, 99).collect()
    assert res_col_scalar["a"].tolist() == [99, 2]
    assert res_col_scalar["b"].tolist() == ["x", "y"]

    # Series rename edge cases
    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        s.rename("new", inplace=True)
    with pytest.raises(UnsupportedOperationError, match="copy=False"):
        s.rename("new", copy=False)
    with pytest.raises(UnsupportedOperationError, match="MultiIndex"):
        s.rename("new", level=1)
    with pytest.raises(UnsupportedOperationError, match="index labels"):
        s.rename({0: 1})


def test_dataframe_and_series_sample() -> None:
    pdf = pd.DataFrame(
        {
            "a": range(50),
            "b": [f"val_{i}" for i in range(50)],
        }
    )
    df = duckpd.from_pandas(pdf)
    s = df["a"]

    # Lazy plan building
    from duckpd._logical import SamplePlan

    sampled_df = df.sample(n=5, random_state=42)
    assert isinstance(sampled_df._plan, SamplePlan)
    assert sampled_df._plan.n == 5
    assert sampled_df._plan.seed == 42

    # Execution with n
    res_df = sampled_df.collect()
    assert len(res_df) == 5
    assert list(res_df.columns) == ["a", "b"]

    # Series sample with n
    res_s = s.sample(n=7, random_state=42).collect()
    assert len(res_s) == 7
    assert isinstance(res_s, pd.Series)

    # Determinism with same random_state
    res_df2 = df.sample(n=5, random_state=42).collect()
    assert res_df.equals(res_df2)
    res_s2 = s.sample(n=7, random_state=42).collect()
    assert res_s.equals(res_s2)

    # frac exact count matching pandas on small/empty inputs
    empty_pdf = pd.DataFrame({"x": [], "y": []})
    empty_df = duckpd.from_pandas(empty_pdf)
    assert len(empty_df.sample(0).collect()) == len(empty_pdf.sample(0))
    assert len(empty_df.sample(frac=0.5).collect()) == len(empty_pdf.sample(frac=0.5))

    small_pdf = pd.DataFrame({"x": range(7)})
    small_df = duckpd.from_pandas(small_pdf)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert len(small_df.sample(frac=frac).collect()) == len(small_pdf.sample(frac=frac))
        assert len(small_df["x"].sample(frac=frac).collect()) == len(
            small_pdf["x"].sample(frac=frac)
        )

    # ignore_index True vs False
    indexed_pdf = pd.DataFrame({"k": ["row1", "row2", "row3"], "v": [10, 20, 30]})
    indexed_df = duckpd.from_pandas(indexed_pdf).set_index("k")
    # With explicit index preserved
    sample_idx_preserved = indexed_df.sample(2, random_state=1, ignore_index=False).collect()
    assert all(idx in ["row1", "row2", "row3"] for idx in sample_idx_preserved.index)
    assert sample_idx_preserved.index.name == "k"
    # With ignore_index
    sample_idx_ignored = indexed_df.sample(2, random_state=1, ignore_index=True).collect()
    assert list(sample_idx_ignored.index) == [0, 1]
    assert list(sample_idx_ignored.columns) == ["v"]

    # Series ignore_index
    s_indexed = indexed_df["v"]
    s_sample_preserved = s_indexed.sample(2, random_state=1, ignore_index=False).collect()
    assert all(idx in ["row1", "row2", "row3"] for idx in s_sample_preserved.index)
    s_sample_ignored = s_indexed.sample(2, random_state=1, ignore_index=True).collect()
    assert list(s_sample_ignored.index) == [0, 1]
    assert s_sample_ignored.name == "v"

    # Validation and error handling
    with pytest.raises(UnsupportedOperationError, match="axis"):
        df.sample(axis=1)
    with pytest.raises(UnsupportedOperationError, match="axis"):
        s.sample(axis="columns")
    with pytest.raises(UnsupportedOperationError, match="replace=True"):
        df.sample(replace=True)
    with pytest.raises(UnsupportedOperationError, match="replace=True"):
        s.sample(replace=True)
    with pytest.raises(UnsupportedOperationError, match="weights"):
        df.sample(weights=[1] * 50)
    with pytest.raises(UnsupportedOperationError, match="weights"):
        s.sample(weights=[1] * 50)
    with pytest.raises(ValueError, match="Only one of 'n' or 'frac'"):
        df.sample(n=5, frac=0.5)
    with pytest.raises(ValueError, match="negative number"):
        df.sample(n=-1)
    with pytest.raises(ValueError, match="negative fraction"):
        df.sample(frac=-0.1)
    with pytest.raises(ValueError, match="Replace has to be set to True"):
        df.sample(frac=1.1)
    with pytest.raises(ValueError, match="finite"):
        s.sample(frac=float("nan"))
    with pytest.raises(ValueError, match="larger sample than population"):
        small_df.sample(n=8).collect()
    with pytest.raises(ValueError, match="larger sample than population"):
        small_df["x"].sample(n=8).collect()
    with pytest.raises(ValueError, match="larger sample than population"):
        empty_df.sample().collect()
    with pytest.raises(ValueError, match="integer"):
        df.sample(n="5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="float"):
        df.sample(frac="0.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="random_state"):
        df.sample(random_state="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between 0 and 2\\*\\*31 - 1"):
        df.sample(random_state=-1)
    with pytest.raises(ValueError, match="between 0 and 2\\*\\*31 - 1"):
        s.sample(random_state=2**31)


def test_seeded_row_count_sample_is_repeatable_with_parallel_session() -> None:
    with duckpd.connect(threads=4) as session:
        frame = session.sql("SELECT i FROM range(200000) AS values(i)")
        first = frame.sample(n=500, random_state=123).collect()
        second = frame.sample(n=500, random_state=123).collect()

    pd.testing.assert_frame_equal(first, second)
