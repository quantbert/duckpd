from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import duckpd
from duckpd.errors import UnsupportedOperationError


def is_missing_number(value: object) -> bool:
    """Return whether a supported numeric reduction scalar is missing."""
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, np.float64):
        return bool(np.isnan(value))
    return False


@pytest.fixture
def numeric_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "integer": pd.Series([1, 2, 3], dtype="int64"),
            "floating": [1.0, np.nan, 3.0],
            "boolean": [True, False, True],
        }
    )


@pytest.mark.parametrize(
    "operation", ["count", "sum", "mean", "min", "max", "std", "var", "median"]
)
@pytest.mark.parametrize("skipna", [True, False])
def test_dataframe_reductions_match_pandas(
    numeric_source: pd.DataFrame,
    operation: str,
    skipna: bool,
) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(numeric_source)

    if operation == "count":
        result = frame.count()
        expected = numeric_source.count()
    else:
        result = getattr(frame, operation)(skipna=skipna)
        expected = getattr(numeric_source, operation)(skipna=skipna)

    assert_series_equal(result, expected)
    assert session.execution_count == 1


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool"])
@pytest.mark.parametrize(
    "operation", ["count", "sum", "mean", "min", "max", "std", "var", "median"]
)
def test_empty_dataframe_reductions_match_pandas(
    dtype: str,
    operation: str,
) -> None:
    source = pd.DataFrame({"value": pd.Series([], dtype=dtype)})
    frame = duckpd.from_pandas(source)

    result = getattr(frame, operation)()
    expected = getattr(source, operation)()

    assert_series_equal(result, expected)


def test_dataframe_sum_min_count_matches_pandas() -> None:
    source = pd.DataFrame(
        {
            "left": [1.0, np.nan, 3.0],
            "right": [np.nan, np.nan, np.nan],
        }
    )
    frame = duckpd.from_pandas(source)

    for min_count in (0, 1, 2, 3, 4):
        assert_series_equal(
            frame.sum(min_count=min_count),
            source.sum(min_count=min_count),
        )


@pytest.mark.parametrize(
    "operation", ["sum", "mean", "min", "max", "std", "var", "median"]
)
@pytest.mark.parametrize("skipna", [True, False])
def test_series_reductions_match_pandas(
    numeric_source: pd.DataFrame,
    operation: str,
    skipna: bool,
) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(numeric_source)

    result = getattr(frame["floating"], operation)(skipna=skipna)
    expected = getattr(numeric_source["floating"], operation)(skipna=skipna)

    if is_missing_number(expected):
        assert is_missing_number(result)
    else:
        assert result == expected
        assert type(result) is type(expected)
    assert session.execution_count == 1


def test_series_count_size_and_sum_min_count_match_pandas(
    numeric_source: pd.DataFrame,
) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(numeric_source)
    series = frame["floating"]
    expected = numeric_source["floating"]

    assert series.count() == expected.count()
    assert series.size == expected.size
    for min_count in (0, 2, 3, 4):
        result = series.sum(min_count=min_count)
        pandas_result = expected.sum(min_count=min_count)
        if is_missing_number(pandas_result):
            assert is_missing_number(result)
        else:
            assert result == pandas_result

    assert session.execution_count == 6


def test_size_counts_visible_elements_and_excludes_hidden_index() -> None:
    source = pd.DataFrame({"row_id": [1, 2, 3], "value": [10, 20, 30]})
    session = duckpd.connect()
    frame = session.from_pandas(source, index="row_id")

    assert frame.size == source.set_index("row_id").size
    assert_series_equal(frame.count(), source.set_index("row_id").count())
    assert session.execution_count == 2


def test_numeric_only_excludes_unsupported_columns() -> None:
    source = pd.DataFrame({"value": [1, 2, 3], "label": ["a", None, "c"]})
    frame = duckpd.from_pandas(source)

    assert_series_equal(frame.count(numeric_only=True), source.count(numeric_only=True))
    assert_series_equal(frame.sum(numeric_only=True), source.sum(numeric_only=True))


def test_assigned_numeric_expression_can_be_reduced() -> None:
    source = pd.DataFrame({"quantity": [2, 3], "price": [10.0, 4.0]})
    session = duckpd.connect()
    frame = session.from_pandas(source).assign(
        notional=lambda current: current["quantity"] * current["price"]
    )

    result = frame["notional"].sum()

    assert result == 32.0
    assert session.execution_count == 1


def test_boolean_expression_can_be_reduced() -> None:
    source = pd.DataFrame({"value": [1, 3, 4]})
    frame = duckpd.from_pandas(source)

    result = (frame["value"] > 2).sum()

    assert result == (source["value"] > 2).sum()


InvalidReduction = Callable[[duckpd.DataFrame], object]


def sum_rows(frame: duckpd.DataFrame) -> object:
    return frame.sum(axis=1)


def mean_all_axes(frame: duckpd.DataFrame) -> object:
    return frame.mean(axis=None)


def sum_negative_min_count(frame: duckpd.DataFrame) -> object:
    return frame.sum(min_count=-1)


def sum_float_min_count(frame: duckpd.DataFrame) -> object:
    return frame.sum(min_count=1.5)  # type: ignore[arg-type]


def sum_mixed_frame(frame: duckpd.DataFrame) -> object:
    return frame.sum()


def mean_string_series(frame: duckpd.DataFrame) -> object:
    return frame["label"].mean()


def sum_series_numeric_only(frame: duckpd.DataFrame) -> object:
    return frame["value"].sum(numeric_only=True)


INVALID_REDUCTIONS: list[tuple[InvalidReduction, type[Exception], str]] = [
    (sum_rows, UnsupportedOperationError, "axis=0"),
    (mean_all_axes, UnsupportedOperationError, "axis=0"),
    (sum_negative_min_count, ValueError, "non-negative"),
    (sum_float_min_count, TypeError, "integer"),
    (sum_mixed_frame, UnsupportedOperationError, "numeric_only=True"),
    (mean_string_series, UnsupportedOperationError, "numeric and boolean"),
    (sum_series_numeric_only, UnsupportedOperationError, "numeric_only=True"),
]


def test_reductions_std_var_ddof() -> None:
    source = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    session = duckpd.connect()
    frame = session.from_pandas(source)

    assert frame["a"].std(ddof=0) == source["a"].std(ddof=0)
    assert frame["a"].var(ddof=0) == source["a"].var(ddof=0)
    assert_series_equal(frame.std(ddof=0), source.std(ddof=0))
    assert_series_equal(frame.var(ddof=0), source.var(ddof=0))


def test_reductions_quantile() -> None:
    source = pd.DataFrame(
        {"a": [10.0, 20.0, 30.0, 40.0, 50.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    session = duckpd.connect()
    frame = session.from_pandas(source)

    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        res_s = frame["a"].quantile(q)
        exp_s = source["a"].quantile(q)
        assert res_s == exp_s

        res_df = frame.quantile(q)
        exp_df = source.quantile(q)
        assert_series_equal(res_df, exp_df)


def test_reductions_any_all() -> None:
    source = pd.DataFrame(
        {
            "all_true": [True, True, True],
            "mixed": [True, False, True],
            "all_false": [False, False, False],
            "with_null": [True, None, False],
        }
    )
    session = duckpd.connect()
    frame = session.from_pandas(source)

    for col in source.columns:
        assert frame[col].any() == source[col].any()
        assert frame[col].all() == source[col].all()

    assert_series_equal(frame.any(), source.any())
    assert_series_equal(frame.all(), source.all())


def test_reductions_any_all_string_truthiness_and_nulls() -> None:
    source = pd.DataFrame({"value": pd.Series(["", "present", None], dtype=object)})
    frame = duckpd.from_pandas(source)

    for skipna in (True, False):
        assert frame["value"].any(skipna=skipna) == source["value"].any(skipna=skipna)
        assert frame["value"].all(skipna=skipna) == source["value"].all(skipna=skipna)

    nullable = pd.DataFrame({"value": pd.Series([False, None], dtype=object)})
    nullable_frame = duckpd.from_pandas(nullable)
    assert nullable_frame["value"].any(skipna=False) == nullable["value"].any(
        skipna=False
    )

    floating = pd.DataFrame({"value": [0.0, float("nan")]})
    floating_frame = duckpd.from_pandas(floating)
    assert floating_frame["value"].any(skipna=False) == floating["value"].any(
        skipna=False
    )
    assert floating_frame["value"].all(skipna=False) == floating["value"].all(
        skipna=False
    )


def test_reductions_bool_only_and_invalid_args() -> None:
    source = pd.DataFrame(
        {
            "b": [True, False, True],
            "n": [1, 2, 3],
            "s": ["a", "b", "c"],
        }
    )
    frame = duckpd.from_pandas(source)

    assert_series_equal(frame.any(bool_only=True), source.any(bool_only=True))
    assert_series_equal(frame.all(bool_only=True), source.all(bool_only=True))
    assert frame["b"].any(bool_only=True) == source["b"].any(bool_only=True)
    assert frame["b"].all(bool_only=True) == source["b"].all(bool_only=True)
    assert not frame["s"].any(bool_only=True)
    assert frame["s"].all(bool_only=True)

    with pytest.raises(UnsupportedOperationError, match="ddof"):
        frame["n"].std(ddof=2)
    with pytest.raises(TypeError, match="ddof"):
        frame["n"].std(ddof="bad")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedOperationError, match="interpolation"):
        frame["n"].quantile(0.5, interpolation="nearest")
    with pytest.raises(ValueError, match="between 0 and 1"):
        frame["n"].quantile(1.5)
    with pytest.raises(UnsupportedOperationError, match="scalar"):
        frame["n"].quantile([0.25, 0.75])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("operation", "exception", "message"),
    INVALID_REDUCTIONS,
)
def test_unsupported_reductions_fail_before_execution(
    operation: InvalidReduction,
    exception: type[Exception],
    message: str,
) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(pd.DataFrame({"value": [1, 2], "label": ["a", "b"]}))

    with pytest.raises(exception, match=message):
        operation(frame)

    assert session.execution_count == 0
