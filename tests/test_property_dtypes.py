# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUntypedFunctionDecorator=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false
"""Hypothesis property-based differential tests for dtypes and null handling."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

import duckpd

# Common Hypothesis settings: fast execution in CI, no per-example deadline


def is_missing_scalar(value: object) -> bool:
    """Check whether a scalar result is null/missing."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


HYPOTHESIS_SETTINGS = settings(max_examples=25, deadline=None)


@given(
    st.lists(
        st.one_of(st.none(), st.integers(min_value=-100_000, max_value=100_000)),
        min_size=1,
        max_size=40,
    )
)
@HYPOTHESIS_SETTINGS
def test_property_nullable_integer_roundtrip_and_reductions(
    data: list[int | None],
) -> None:
    pdf = pd.DataFrame({"val": pd.Series(data, dtype="Int64")})
    frame = duckpd.from_pandas(pdf)

    # 1. Roundtrip collect
    collected = frame.collect()
    assert_frame_equal(collected, pdf)

    # 2. Count reduction
    assert frame["val"].count() == pdf["val"].count()

    # 3. Sum reduction
    sum_dp = frame["val"].sum(skipna=True)
    sum_pd = pdf["val"].sum(skipna=True)
    if is_missing_scalar(sum_pd):
        assert is_missing_scalar(sum_dp)
    else:
        assert sum_dp == sum_pd

    # 4. Min/Max reduction
    min_dp = frame["val"].min()
    min_pd = pdf["val"].min()
    if is_missing_scalar(min_pd):
        assert is_missing_scalar(min_dp)
    else:
        assert min_dp == min_pd


@given(
    st.lists(
        st.one_of(
            st.none(),
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
        ),
        min_size=1,
        max_size=40,
    )
)
@HYPOTHESIS_SETTINGS
def test_property_float_roundtrip_and_reductions(
    data: list[float | None],
) -> None:
    pdf = pd.DataFrame({"val": pd.Series(data, dtype="float64")})
    frame = duckpd.from_pandas(pdf)

    # Roundtrip collect
    collected = frame.collect()
    assert_frame_equal(collected, pdf)

    # Count
    assert frame["val"].count() == pdf["val"].count()

    # Sum (allowing small floating tolerances)
    sum_dp = frame["val"].sum()
    sum_pd = pdf["val"].sum()
    if is_missing_scalar(sum_pd):
        assert is_missing_scalar(sum_dp)
    else:
        assert np.isclose(
            float(cast("float", sum_dp)),
            float(cast("float", sum_pd)),
            rtol=1e-5,
            atol=1e-8,
        )


@given(
    st.lists(
        st.one_of(st.none(), st.booleans()),
        min_size=1,
        max_size=40,
    )
)
@HYPOTHESIS_SETTINGS
def test_property_boolean_reductions(data: list[bool | None]) -> None:
    pdf = pd.DataFrame({"flag": pd.Series(data, dtype="boolean")})
    frame = duckpd.from_pandas(pdf)

    # any(skipna=True)
    any_dp = frame["flag"].any(skipna=True)
    any_pd = pdf["flag"].any(skipna=True)
    if is_missing_scalar(any_pd):
        assert is_missing_scalar(any_dp)
    else:
        assert bool(any_dp) == bool(any_pd)

    # all(skipna=True)
    all_dp = frame["flag"].all(skipna=True)
    all_pd = pdf["flag"].all(skipna=True)
    if is_missing_scalar(all_pd):
        assert is_missing_scalar(all_dp)
    else:
        assert bool(all_dp) == bool(all_pd)


@given(
    st.lists(
        st.one_of(
            st.none(),
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", max_size=15),
        ),
        min_size=1,
        max_size=30,
    )
)
@HYPOTHESIS_SETTINGS
def test_property_string_accessors(data: list[str | None]) -> None:
    pdf = pd.DataFrame({"text": data})
    frame = duckpd.from_pandas(pdf)

    # str.upper
    res_upper = frame["text"].str.upper().collect().tolist()
    exp_upper = pdf["text"].str.upper().tolist()
    assert len(res_upper) == len(exp_upper)
    for act, exp in zip(res_upper, exp_upper, strict=True):
        if is_missing_scalar(exp):
            assert is_missing_scalar(act)
        else:
            assert act == exp

    # str.len
    res_len = frame["text"].str.len().collect().tolist()
    exp_len = pdf["text"].str.len().tolist()
    assert len(res_len) == len(exp_len)
    for act, exp in zip(res_len, exp_len, strict=True):
        if is_missing_scalar(exp):
            assert is_missing_scalar(act)
        else:
            assert int(act) == int(exp)


@given(
    st.lists(st.integers(min_value=-500, max_value=500), min_size=1, max_size=25),
    st.lists(st.integers(min_value=-500, max_value=500), min_size=1, max_size=25),
)
@HYPOTHESIS_SETTINGS
def test_property_concat_differentials(first: list[int], second: list[int]) -> None:
    d1 = pd.DataFrame({"a": first})
    d2 = pd.DataFrame({"a": second})

    session = duckpd.connect()
    f1 = session.from_pandas(d1)
    f2 = session.from_pandas(d2)

    combined = duckpd.concat([f1, f2]).collect()
    expected = pd.concat([d1, d2], ignore_index=True)
    assert_frame_equal(combined.reset_index(drop=True), expected)


@given(
    st.lists(
        st.one_of(st.none(), st.integers(min_value=-1000, max_value=1000)),
        min_size=2,
        max_size=30,
    )
)
@HYPOTHESIS_SETTINGS
def test_property_missing_value_transformations(
    data: list[int | None],
) -> None:
    pdf = pd.DataFrame({"x": pd.Series(data, dtype="float64")})
    frame = duckpd.from_pandas(pdf)

    # isna()
    assert_frame_equal(frame.isna().collect(), pdf.isna())

    # notna()
    assert_frame_equal(frame.notna().collect(), pdf.notna())

    # fillna(0.0)
    assert_frame_equal(frame.fillna(0.0).collect(), pdf.fillna(0.0))

    # dropna()
    assert_frame_equal(
        frame.dropna().collect().reset_index(drop=True),
        pdf.dropna().reset_index(drop=True),
    )
