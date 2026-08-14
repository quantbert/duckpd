"""Differential tests for distinct, count, and top-n operations."""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import duckpd
from duckpd.errors import UnsupportedOperationError


@pytest.fixture
def dup_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 2, 1, 3, 2, 1],
            "b": ["x", "y", "x", "z", "y", "x"],
            "c": [10.0, 20.0, 10.0, 30.0, 20.0, 10.0],
        }
    )


@pytest.fixture
def numeric_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "val": [3, 1, 4, 1, 5, 9, 2, 6],
            "grp": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )


# --- nunique ----------------------------------------------------------------


def test_series_nunique_matches_pandas(dup_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(dup_source)

    result = frame["a"].nunique()
    expected = dup_source["a"].nunique()
    assert result == expected
    assert session.execution_count == 1


def test_series_nunique_with_nulls() -> None:
    source = pd.DataFrame({"val": [1, None, 2, None, 1]})
    frame = duckpd.from_pandas(source)

    result = frame["val"].nunique()
    expected = source["val"].nunique()
    assert result == expected


def test_dataframe_nunique_matches_pandas(dup_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(dup_source)

    result = frame.nunique()
    expected = dup_source.nunique()
    assert_series_equal(result, expected)
    assert session.execution_count == 1


def test_dataframe_nunique_is_lazy(dup_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(dup_source)

    _ = frame.nunique()
    assert session.execution_count == 1  # nunique is an eager reduction


def test_dataframe_nunique_invalid_axis(dup_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(dup_source)

    with pytest.raises(UnsupportedOperationError, match="axis=0"):
        frame.nunique(axis=1)  # type: ignore[arg-type]


# --- unique -----------------------------------------------------------------


def test_series_unique_matches_pandas(dup_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(dup_source)

    result = frame["a"].unique()
    expected = dup_source["a"].unique()
    # Order is not guaranteed by SQL; compare as sets
    assert sorted(result.dropna().tolist()) == sorted(expected.tolist())


def test_series_unique_with_nulls() -> None:
    source = pd.DataFrame({"val": [1, None, 2, None, 1]})
    frame = duckpd.from_pandas(source)

    result = frame["val"].unique()
    expected = source["val"].unique()
    # DuckDB DISTINCT drops nulls; pandas keeps one NaN
    assert sorted(v for v in result if pd.notna(v)) == sorted(
        v for v in expected if pd.notna(v)
    )


# --- value_counts -----------------------------------------------------------


def test_series_value_counts_matches_pandas(dup_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(dup_source)

    result = frame["a"].value_counts()
    expected = dup_source["a"].value_counts()

    # value_counts returns a Series indexed by the values, sorted by count desc
    assert session.execution_count == 1
    # Compare sorted by count (order may differ for ties)
    assert sorted(result.items()) == sorted(expected.items())


def test_series_value_counts_sort_false(dup_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(dup_source)

    result = frame["a"].value_counts(sort=False)
    expected = dup_source["a"].value_counts(sort=False)

    assert sorted(result.items()) == sorted(expected.items())


def test_series_value_counts_ascending(dup_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(dup_source)

    result = frame["a"].value_counts(ascending=True)
    expected = dup_source["a"].value_counts(ascending=True)

    assert result.tolist() == expected.tolist()


def test_series_value_counts_with_nulls() -> None:
    source = pd.DataFrame({"val": [1, None, 2, None, 1, 2]})
    frame = duckpd.from_pandas(source)

    result = frame["val"].value_counts(dropna=True)
    expected = source["val"].value_counts(dropna=True)

    assert sorted(result.items()) == sorted(expected.items())


# --- drop_duplicates --------------------------------------------------------


def test_dataframe_drop_duplicates_all_columns(dup_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(dup_source)

    result = frame.drop_duplicates()
    assert session.execution_count == 0

    expected = dup_source.drop_duplicates().reset_index(drop=True)
    assert_frame_equal(
        result.collect().sort_values(["a", "b"]).reset_index(drop=True),
        expected.sort_values(["a", "b"]).reset_index(drop=True),
    )
    assert session.execution_count == 1


def test_dataframe_drop_duplicates_subset(dup_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(dup_source)

    result = frame.drop_duplicates(subset=["a"])
    expected = dup_source.drop_duplicates(subset=["a"]).reset_index(drop=True)

    assert_frame_equal(
        result.collect().sort_values(["a"]).reset_index(drop=True),
        expected.sort_values(["a"]).reset_index(drop=True),
    )


def test_dataframe_drop_duplicates_subset_single_string(
    dup_source: pd.DataFrame,
) -> None:
    frame = duckpd.from_pandas(dup_source)

    result = frame.drop_duplicates(subset="a")
    expected = dup_source.drop_duplicates(subset="a").reset_index(drop=True)

    assert_frame_equal(
        result.collect().sort_values(["a"]).reset_index(drop=True),
        expected.sort_values(["a"]).reset_index(drop=True),
    )


def test_dataframe_drop_duplicates_is_lazy(dup_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(dup_source)

    _ = frame.drop_duplicates()
    assert session.execution_count == 0


def test_dataframe_drop_duplicates_invalid_args(dup_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(dup_source)

    with pytest.raises(UnsupportedOperationError, match="inplace=True"):
        frame.drop_duplicates(inplace=True)
    with pytest.raises(UnsupportedOperationError, match="ignore_index=True"):
        frame.drop_duplicates(ignore_index=True)
    with pytest.raises(ValueError, match="keep must be"):
        frame.drop_duplicates(keep="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="subset must not be empty"):
        frame.drop_duplicates(subset=[])


def test_series_drop_duplicates_matches_pandas(dup_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(dup_source)

    result_series = frame["a"].drop_duplicates()
    # Materialize by wrapping in a DataFrame
    result_df = duckpd.DataFrame(result_series._session, result_series._plan)
    result = result_df.collect().iloc[:, 0]
    expected = dup_source["a"].drop_duplicates().reset_index(drop=True)

    assert sorted(result.tolist()) == sorted(expected.tolist())


# --- nlargest / nsmallest ---------------------------------------------------


def test_dataframe_nlargest_matches_pandas(numeric_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(numeric_source)

    result = frame.nlargest(3, "val")
    assert session.execution_count == 0

    expected = numeric_source.nlargest(3, "val").reset_index(drop=True)
    assert_frame_equal(result.collect().reset_index(drop=True), expected)
    assert session.execution_count == 1


def test_dataframe_nsmallest_matches_pandas(numeric_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(numeric_source)

    result = frame.nsmallest(3, "val")
    expected = numeric_source.nsmallest(3, "val").reset_index(drop=True)
    assert_frame_equal(result.collect().reset_index(drop=True), expected)


def test_dataframe_nlargest_multiple_columns() -> None:
    source = pd.DataFrame(
        {"a": [1, 2, 2, 1], "b": [10, 20, 30, 40], "c": ["w", "x", "y", "z"]}
    )
    frame = duckpd.from_pandas(source)

    result = frame.nlargest(2, ["a", "b"])
    expected = source.nlargest(2, ["a", "b"]).reset_index(drop=True)
    assert_frame_equal(result.collect().reset_index(drop=True), expected)


def test_dataframe_nlargest_is_lazy(numeric_source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(numeric_source)

    _ = frame.nlargest(3, "val")
    assert session.execution_count == 0


def test_dataframe_nlargest_invalid_args(numeric_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(numeric_source)

    with pytest.raises(ValueError, match="n must be non-negative"):
        frame.nlargest(-1, "val")
    with pytest.raises(UnsupportedOperationError, match="keep='all'"):
        frame.nlargest(3, "val", keep="all")
    with pytest.raises(ValueError, match="keep must be"):
        frame.nlargest(3, "val", keep="invalid")  # type: ignore[arg-type]


def test_series_nlargest_matches_pandas(numeric_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(numeric_source)

    result_series = frame["val"].nlargest(3)
    result_df = duckpd.DataFrame(result_series._session, result_series._plan)
    result = result_df.collect().iloc[:, 0]
    expected = numeric_source["val"].nlargest(3).reset_index(drop=True)
    assert sorted(result.tolist()) == sorted(expected.tolist())


def test_series_nsmallest_matches_pandas(numeric_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(numeric_source)

    result_series = frame["val"].nsmallest(3)
    result_df = duckpd.DataFrame(result_series._session, result_series._plan)
    result = result_df.collect().iloc[:, 0]
    expected = numeric_source["val"].nsmallest(3).reset_index(drop=True)
    assert sorted(result.tolist()) == sorted(expected.tolist())


def test_series_nlargest_invalid_args() -> None:
    frame = duckpd.from_pandas(pd.DataFrame({"val": [1, 2, 3]}))

    with pytest.raises(ValueError, match="n must be non-negative"):
        frame["val"].nlargest(-1)
    with pytest.raises(ValueError, match="keep must be"):
        frame["val"].nlargest(1, keep="invalid")  # type: ignore[arg-type]


# --- combined pipeline ------------------------------------------------------


def test_drop_duplicates_then_nlargest_pipeline(numeric_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(numeric_source)

    result = (
        frame.drop_duplicates(subset=["grp"])
        .nlargest(1, "val")
        .collect()
        .reset_index(drop=True)
    )
    expected = (
        numeric_source.drop_duplicates(subset=["grp"])
        .nlargest(1, "val")
        .reset_index(drop=True)
    )
    assert_frame_equal(result, expected)


def test_value_counts_nunique_pipeline(dup_source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(dup_source)

    vc = frame["a"].value_counts()
    n = frame["a"].nunique()

    assert len(vc) == n
