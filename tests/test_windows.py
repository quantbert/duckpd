"""Differential tests for window operations, cumulative functions, rank, dedup."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import duckpd as dp
from duckpd.errors import UnorderedOperationError


def test_series_cumsum_skipna_differential() -> None:
    data = [1.0, 2.0, np.nan, 4.0, 5.0]
    s_pd = pd.Series(data)
    df_dp = dp.from_pandas(pd.DataFrame({"a": data, "id": range(len(data))}), order_by="id")

    res_dp_skipna = df_dp["a"].cumsum(skipna=True).collect()
    pd.testing.assert_series_equal(res_dp_skipna, s_pd.cumsum(skipna=True), check_names=False)

    res_dp_noskip = df_dp["a"].cumsum(skipna=False).collect()
    pd.testing.assert_series_equal(res_dp_noskip, s_pd.cumsum(skipna=False), check_names=False)


def test_series_cummin_cummax_cumprod_differential() -> None:
    data = [3.0, 1.0, np.nan, 4.0, 2.0]
    s_pd = pd.Series(data)
    df_dp = dp.from_pandas(pd.DataFrame({"a": data, "id": range(len(data))}), order_by="id")

    # cummin
    res_min_skip = df_dp["a"].cummin(skipna=True).collect()
    pd.testing.assert_series_equal(res_min_skip, s_pd.cummin(skipna=True), check_names=False)
    res_min_noskip = df_dp["a"].cummin(skipna=False).collect()
    pd.testing.assert_series_equal(res_min_noskip, s_pd.cummin(skipna=False), check_names=False)

    # cummax
    res_max_skip = df_dp["a"].cummax(skipna=True).collect()
    pd.testing.assert_series_equal(res_max_skip, s_pd.cummax(skipna=True), check_names=False)
    res_max_noskip = df_dp["a"].cummax(skipna=False).collect()
    pd.testing.assert_series_equal(res_max_noskip, s_pd.cummax(skipna=False), check_names=False)

    # cumprod
    res_prod_skip = df_dp["a"].cumprod(skipna=True).collect()
    pd.testing.assert_series_equal(res_prod_skip, s_pd.cumprod(skipna=True), check_names=False)
    res_prod_noskip = df_dp["a"].cumprod(skipna=False).collect()
    pd.testing.assert_series_equal(res_prod_noskip, s_pd.cumprod(skipna=False), check_names=False)


def test_dataframe_cum_transforms_differential() -> None:
    pdf = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "a": [10.0, 20.0, np.nan, 30.0, 40.0],
            "b": [1, 2, 3, 4, 5],
        }
    )
    df = dp.from_pandas(pdf, order_by="id")

    # DataFrame cumsum
    res_df_cumsum = df[["a", "b"]].cumsum().collect()
    exp_df_cumsum = pdf[["a", "b"]].cumsum()
    pd.testing.assert_frame_equal(res_df_cumsum, exp_df_cumsum)

    # DataFrame cumprod
    res_df_cumprod = df[["a", "b"]].cumprod().collect()
    exp_df_cumprod = pdf[["a", "b"]].cumprod()
    pd.testing.assert_frame_equal(res_df_cumprod, exp_df_cumprod)


def test_cumulative_uses_stable_pandas_snapshot_order() -> None:
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    df = dp.from_pandas(pdf)

    pd.testing.assert_series_equal(df["a"].cumsum().collect(), pdf["a"].cumsum())
    pd.testing.assert_frame_equal(df.cumsum().collect(), pdf.cumsum())


def test_cumulative_uses_csv_source_order(tmp_path: Path) -> None:
    path = tmp_path / "ordered.csv"
    source = pd.DataFrame({"a": [1, 2, 3]})
    source.to_csv(path, index=False)
    df = dp.read_csv(path)

    pd.testing.assert_series_equal(df["a"].cumsum().collect(), source["a"].cumsum())


def test_shift_diff_pct_change_differential() -> None:
    data = [10.0, 20.0, 25.0, 50.0, 100.0]
    s_pd = pd.Series(data)
    df_dp = dp.from_pandas(pd.DataFrame({"a": data, "id": range(len(data))}), order_by="id")

    # Shift positive & negative
    res_shift_1 = df_dp["a"].shift(1).collect()
    pd.testing.assert_series_equal(res_shift_1, s_pd.shift(1), check_names=False)

    res_shift_neg1 = df_dp["a"].shift(-1).collect()
    pd.testing.assert_series_equal(res_shift_neg1, s_pd.shift(-1), check_names=False)

    # Shift with fill_value
    res_shift_fill = df_dp["a"].shift(1, fill_value=0.0).collect()
    pd.testing.assert_series_equal(res_shift_fill, s_pd.shift(1, fill_value=0.0), check_names=False)

    # Diff
    res_diff = df_dp["a"].diff(1).collect()
    pd.testing.assert_series_equal(res_diff, s_pd.diff(1), check_names=False)

    # Pct_change
    res_pct = df_dp["a"].pct_change(1).collect()
    pd.testing.assert_series_equal(res_pct, s_pd.pct_change(1), check_names=False)

    # DataFrame shift / diff / pct_change
    df_data = pd.DataFrame({"id": range(5), "a": data, "b": [1, 2, 3, 4, 5]})
    df_inst = dp.from_pandas(df_data, order_by="id")

    res_df_shift = df_inst[["a", "b"]].shift(1).collect()
    pd.testing.assert_frame_equal(res_df_shift, df_data[["a", "b"]].shift(1))

    res_df_diff = df_inst[["a", "b"]].diff(1).collect()
    pd.testing.assert_frame_equal(res_df_diff, df_data[["a", "b"]].diff(1))

    res_df_pct = df_inst[["a", "b"]].pct_change(1).collect()
    pd.testing.assert_frame_equal(res_df_pct, df_data[["a", "b"]].pct_change(1))


def test_rank_methods_differential() -> None:
    data = [10.0, 20.0, 20.0, 30.0, np.nan]
    s_pd = pd.Series(data)
    df_dp = dp.from_pandas(pd.DataFrame({"a": data, "id": range(len(data))}), order_by="id")

    for method in ("average", "min", "max", "first", "dense"):
        res_dp = df_dp["a"].rank(method=method).collect()  # type: ignore[arg-type]
        exp_pd = s_pd.rank(method=method)  # type: ignore[arg-type]
        pd.testing.assert_series_equal(res_dp, exp_pd, check_names=False)

    # Percent rank
    res_pct = df_dp["a"].rank(pct=True).collect()
    exp_pct = s_pd.rank(pct=True)
    pd.testing.assert_series_equal(res_pct, exp_pct, check_names=False)

    # na_option='bottom'
    res_bottom = df_dp["a"].rank(na_option="bottom").collect()
    exp_bottom = s_pd.rank(na_option="bottom")
    pd.testing.assert_series_equal(res_bottom, exp_bottom, check_names=False)

    # DataFrame rank
    df_input = pd.DataFrame({"id": range(len(data)), "a": data, "b": [5, 4, 3, 2, 1]})
    df_obj = dp.from_pandas(df_input, order_by="id")
    res_df_rank = df_obj[["a", "b"]].rank().collect()
    exp_df_rank = df_input[["a", "b"]].rank()
    pd.testing.assert_frame_equal(res_df_rank, exp_df_rank)


def test_drop_duplicates_keep_modes_differential() -> None:
    pdf = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "k": ["A", "B", "A", "C", "B"],
            "val": [10, 20, 30, 40, 50],
        }
    )
    df = dp.from_pandas(pdf, order_by="id")

    # keep='first' with order
    res_first = df.drop_duplicates(subset=["k"], keep="first").sort_values("id").collect()
    exp_first = pdf.drop_duplicates(subset=["k"], keep="first")
    pd.testing.assert_frame_equal(
        res_first.reset_index(drop=True), exp_first.reset_index(drop=True)
    )

    # keep='last' with order
    res_last = df.drop_duplicates(subset=["k"], keep="last").sort_values("id").collect()
    exp_last = pdf.drop_duplicates(subset=["k"], keep="last")
    pd.testing.assert_frame_equal(res_last.reset_index(drop=True), exp_last.reset_index(drop=True))

    # keep=False with order
    res_false = df.drop_duplicates(subset=["k"], keep=False).sort_values("id").collect()
    exp_false = pdf.drop_duplicates(subset=["k"], keep=False)
    pd.testing.assert_frame_equal(
        res_false.reset_index(drop=True), exp_false.reset_index(drop=True)
    )


def test_series_rolling_and_expanding_differential() -> None:
    data = [10.0, 20.0, np.nan, 30.0, 50.0]
    s_pd = pd.Series(data)
    df_dp = dp.from_pandas(pd.DataFrame({"a": data, "id": range(len(data))}), order_by="id")

    # rolling sum, mean, min, max, std, var, count
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].rolling(2).sum()).collect(),
        s_pd.rolling(2).sum(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].rolling(2, min_periods=1).mean()).collect(),
        s_pd.rolling(2, min_periods=1).mean(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].rolling(2, min_periods=1).min()).collect(),
        s_pd.rolling(2, min_periods=1).min(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].rolling(2, min_periods=1).max()).collect(),
        s_pd.rolling(2, min_periods=1).max(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].rolling(2).count()).collect(),
        s_pd.rolling(2).count(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].rolling(3, min_periods=2).std()).collect(),
        s_pd.rolling(3, min_periods=2).std(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].rolling(3, min_periods=2).var()).collect(),
        s_pd.rolling(3, min_periods=2).var(),
        check_names=False,
    )

    # expanding sum, mean, min, max, std, var, count
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].expanding().sum()).collect(),
        s_pd.expanding().sum(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].expanding(2).mean()).collect(),
        s_pd.expanding(2).mean(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].expanding().min()).collect(),
        s_pd.expanding().min(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].expanding().max()).collect(),
        s_pd.expanding().max(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].expanding().count()).collect(),
        s_pd.expanding().count(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].expanding(2).std()).collect(),
        s_pd.expanding(2).std(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        cast("dp.Series", df_dp["a"].expanding(2).var()).collect(),
        s_pd.expanding(2).var(),
        check_names=False,
    )


def test_window_min_periods_zero_and_validation() -> None:
    source = pd.DataFrame({"id": [1, 2], "value": [np.nan, np.nan]})
    frame = dp.from_pandas(source, order_by="id")

    rolling = cast("dp.Series", frame["value"].rolling(2, min_periods=0).sum())
    expanding = cast("dp.Series", frame["value"].expanding(min_periods=0).sum())
    pd.testing.assert_series_equal(
        rolling.collect(),
        source["value"].rolling(2, min_periods=0).sum(),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        expanding.collect(),
        source["value"].expanding(min_periods=0).sum(),
        check_names=False,
    )

    with pytest.raises(ValueError, match="non-negative"):
        frame["value"].rolling(2, min_periods=-1)
    with pytest.raises(ValueError, match="exceed"):
        frame["value"].rolling(2, min_periods=3)
    with pytest.raises(ValueError, match="integer"):
        frame["value"].expanding(min_periods=1.5)  # type: ignore[arg-type]


def test_dataframe_rolling_and_expanding_differential() -> None:
    pdf = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "a": [10.0, 20.0, np.nan, 30.0, 50.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    df = dp.from_pandas(pdf, order_by="id")

    # DataFrame rolling
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].rolling(2).sum()).collect(),
        pdf[["a", "b"]].rolling(2).sum(),
    )
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].rolling(2, min_periods=1).mean()).collect(),
        pdf[["a", "b"]].rolling(2, min_periods=1).mean(),
    )
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].rolling(2).min()).collect(),
        pdf[["a", "b"]].rolling(2).min(),
    )
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].rolling(2).max()).collect(),
        pdf[["a", "b"]].rolling(2).max(),
    )
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].rolling(2).count()).collect(),
        pdf[["a", "b"]].rolling(2).count(),
    )
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].rolling(2).std()).collect(),
        pdf[["a", "b"]].rolling(2).std(),
    )
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].rolling(2).var()).collect(),
        pdf[["a", "b"]].rolling(2).var(),
    )

    # DataFrame expanding
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].expanding().sum()).collect(),
        pdf[["a", "b"]].expanding().sum(),
    )
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].expanding().mean()).collect(),
        pdf[["a", "b"]].expanding().mean(),
    )
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].expanding().min()).collect(),
        pdf[["a", "b"]].expanding().min(),
    )
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].expanding().max()).collect(),
        pdf[["a", "b"]].expanding().max(),
    )
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].expanding().count()).collect(),
        pdf[["a", "b"]].expanding().count(),
    )
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].expanding(2).std()).collect(),
        pdf[["a", "b"]].expanding(2).std(),
    )
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df[["a", "b"]].expanding(2).var()).collect(),
        pdf[["a", "b"]].expanding(2).var(),
    )


def test_series_groupby_rolling_matches_pandas_with_multiple_keys() -> None:
    source = pd.DataFrame(
        {
            "row": [10, 11, 12, 13, 14, 15, 16],
            "ticker": ["B", "A", "B", "A", None, "B", "A"],
            "venue": ["X", "X", "X", "Y", "X", "X", "X"],
            "sequence": [1, 1, 2, 2, 2, 3, 3],
            "close": [10.0, 20.0, 12.0, 22.0, 30.0, 14.0, 24.0],
        }
    )
    pandas_frame = source.set_index("row")
    session = dp.connect()
    frame = session.from_pandas(source, index="row", order_by="sequence")

    result = (
        frame.groupby(
            ["ticker", "venue"],
            sort=False,
            dropna=False,
        )["close"]
        .rolling(3, min_periods=2)
        .mean()
    )

    assert session.execution_count == 0
    expected = (
        pandas_frame.groupby(
            ["ticker", "venue"],
            sort=False,
            dropna=False,
        )["close"]
        .rolling(3, min_periods=2)
        .mean()
    )
    pd.testing.assert_series_equal(cast("dp.Series", result).collect(), expected)
    assert session.execution_count == 1


def test_dataframe_groupby_rolling_as_index_false_matches_pandas() -> None:
    source = pd.DataFrame(
        {
            "row": [4, 2, 5, 1, 3],
            "ticker": ["B", "A", "B", "A", "B"],
            "sequence": [1, 1, 2, 2, 3],
            "close": [10.0, 20.0, 12.0, 22.0, 14.0],
            "volume": [100.0, 200.0, 120.0, 220.0, 140.0],
        }
    )
    pandas_frame = source.set_index("row")
    frame = dp.from_pandas(source, index="row", order_by="sequence")

    result = (
        frame.groupby("ticker", as_index=False, sort=True)[["close", "volume"]]
        .rolling(2, min_periods=1)
        .sum()
    )
    expected = (
        pandas_frame.groupby("ticker", as_index=False, sort=True)[["close", "volume"]]
        .rolling(2, min_periods=1)
        .sum()
    )

    pd.testing.assert_frame_equal(cast("dp.DataFrame", result).collect(), expected)


def test_grouped_rolling_assigns_to_origin_without_materialization() -> None:
    source = pd.DataFrame(
        {
            "ticker": ["B", "A", "B", None, "A", "B"],
            "sequence": [1, 1, 2, 2, 3, 3],
            "close": [10.0, 20.0, 12.0, 30.0, 24.0, 14.0],
        }
    )
    session = dp.connect()
    frame = session.from_pandas(source, order_by="sequence")
    moving_average = (
        frame.groupby("ticker", sort=False, dropna=True)["close"].rolling(2, min_periods=1).mean()
    )

    result = frame.assign(moving_average=moving_average)

    assert session.execution_count == 0
    expected = source.assign(
        moving_average=source.groupby(
            "ticker",
            sort=False,
            dropna=True,
        )["close"].transform(lambda values: values.rolling(2, min_periods=1).mean())
    )
    pd.testing.assert_frame_equal(result.collect(), expected)
    assert session.execution_count == 1


def test_projected_dataframe_grouped_rolling_assigns_to_origin() -> None:
    source = pd.DataFrame(
        {
            "group": ["A", "B", "A", "B", "A"],
            "sequence": [1, 1, 2, 2, 3],
            "x": [10.0, 20.0, 12.0, 18.0, 14.0],
            "y": [100.0, 200.0, 120.0, 180.0, 140.0],
        }
    )
    session = dp.connect()
    frame = session.from_pandas(source, order_by="sequence")
    rolling = frame.groupby("group", sort=False)[["x", "y"]].rolling(2, min_periods=1).mean()

    frame[["mean_x", "mean_y"]] = cast("dp.DataFrame", rolling)

    assert session.execution_count == 0
    expected = source.assign(
        mean_x=source.groupby("group", sort=False)["x"].transform(
            lambda values: values.rolling(2, min_periods=1).mean()
        ),
        mean_y=source.groupby("group", sort=False)["y"].transform(
            lambda values: values.rolling(2, min_periods=1).mean()
        ),
    )
    pd.testing.assert_frame_equal(frame.collect(), expected)
    assert session.execution_count == 1


def test_multi_ticker_moving_average_crossover_pipeline() -> None:
    source = pd.DataFrame(
        {
            "ticker": ["A", "B", "A", "B", "A", "B"],
            "date": [1, 1, 2, 2, 3, 3],
            "close": [10.0, 20.0, 12.0, 18.0, 14.0, 22.0],
        }
    )
    session = dp.connect()
    prices = session.from_pandas(source, order_by=["date", "ticker"])

    features = prices.assign(
        fast_ma=lambda frame: frame.groupby("ticker")["close"].rolling(2).mean(),
        slow_ma=lambda frame: frame.groupby("ticker")["close"].rolling(3).mean(),
    ).assign(ma_cross=lambda frame: frame["fast_ma"] > frame["slow_ma"])

    assert session.execution_count == 0
    expected = source.assign(
        fast_ma=[np.nan, np.nan, 11.0, 19.0, 13.0, 20.0],
        slow_ma=[np.nan, np.nan, np.nan, np.nan, 12.0, 20.0],
        ma_cross=pd.Series(
            [pd.NA, pd.NA, pd.NA, pd.NA, True, False],
            dtype="boolean",
        ),
    )
    pd.testing.assert_frame_equal(features.collect(), expected)
    assert features.ordering == ("date", "ticker")
    assert session.execution_count == 1


def test_grouped_rolling_rejects_unordered_input_before_execution() -> None:
    session = dp.connect()
    frame = session.sql("select * from (values ('a', 1.0), ('a', 2.0)) t(\"group\", value)")

    with pytest.raises(UnorderedOperationError):
        frame.groupby("group")["value"].rolling(2).sum()
    with pytest.raises(UnorderedOperationError):
        frame.groupby("group")[["value"]].rolling(2).sum()
    assert session.execution_count == 0


SERIES_ORDER_OPERATIONS: tuple[Callable[[dp.Series], object], ...] = (
    lambda series: series.cumsum(),
    lambda series: series.cummin(),
    lambda series: series.cummax(),
    lambda series: series.cumprod(),
    lambda series: series.shift(),
    lambda series: series.diff(),
    lambda series: series.pct_change(),
    lambda series: series.rank(method="first"),
    lambda series: series.rolling(2).sum(),
    lambda series: series.expanding(2).sum(),
    lambda series: series.drop_duplicates(keep="first"),
    lambda series: series.drop_duplicates(keep="last"),
    lambda series: series.nlargest(2),
    lambda series: series.nsmallest(2),
)


@pytest.mark.parametrize("operation", SERIES_ORDER_OPERATIONS)
def test_every_series_order_dependent_operation_rejects_unordered_input(
    operation: Callable[[dp.Series], object],
) -> None:
    series = dp.connect().sql("select * from (values (3), (1), (2)) t(value)")["value"]

    with pytest.raises(UnorderedOperationError):
        operation(series)


DATAFRAME_ORDER_OPERATIONS: tuple[Callable[[dp.DataFrame], object], ...] = (
    lambda frame: frame.cumsum(),
    lambda frame: frame.cummin(),
    lambda frame: frame.cummax(),
    lambda frame: frame.cumprod(),
    lambda frame: frame.rolling(2).sum(),
    lambda frame: frame.expanding(2).sum(),
    lambda frame: frame.head(2),
    lambda frame: frame.drop_duplicates(keep="first"),
    lambda frame: frame.drop_duplicates(keep="last"),
    lambda frame: frame.nlargest(2, "left"),
    lambda frame: frame.nsmallest(2, "left"),
)


@pytest.mark.parametrize("operation", DATAFRAME_ORDER_OPERATIONS)
def test_every_dataframe_order_dependent_operation_rejects_unordered_input(
    operation: Callable[[dp.DataFrame], object],
) -> None:
    frame = dp.connect().sql('select * from (values (3, 6), (1, 4), (2, 5)) t("left", "right")')

    with pytest.raises(UnorderedOperationError):
        operation(frame)


def test_order_independent_duplicate_removal_accepts_unordered_input(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unordered-duplicates.csv"
    source = pd.DataFrame({"value": [2, 1, 2]})
    source.to_csv(path, index=False)

    result = dp.read_csv(path).drop_duplicates(keep=False).collect()

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        source.drop_duplicates(keep=False).reset_index(drop=True),
    )
