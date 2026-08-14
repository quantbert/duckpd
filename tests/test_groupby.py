from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import duckpd
from duckpd.errors import UnsupportedOperationError


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dept": ["sales", "engineering", "sales", "engineering", "sales"],
            "region": ["east", "east", "west", "west", "east"],
            "revenue": [100.0, 200.0, 150.0, 250.0, 50.0],
            "units": [10, 20, 15, 25, 5],
            "bonus_eligible": [True, True, False, True, False],
        }
    )


def test_groupby_named_agg_as_index_false_matches_pandas(
    sample_sales: pd.DataFrame,
) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(sample_sales)

    result = frame.groupby(["dept", "region"], as_index=False).agg(
        total_rev=("revenue", "sum"),
        avg_units=("units", "mean"),
        emp_count=("bonus_eligible", "count"),
        total_size=("units", "size"),
    )

    assert session.execution_count == 0

    expected = sample_sales.groupby(["dept", "region"], as_index=False).agg(
        total_rev=("revenue", "sum"),
        avg_units=("units", "mean"),
        emp_count=("bonus_eligible", "count"),
        total_size=("units", "size"),
    )

    assert_frame_equal(result.collect(), expected)
    assert session.execution_count == 1


def test_groupby_named_agg_as_index_true_matches_pandas(
    sample_sales: pd.DataFrame,
) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(sample_sales)

    result = frame.groupby("dept", as_index=True).agg(
        total_rev=("revenue", "sum"),
        max_units=("units", "max"),
        min_units=("units", "min"),
    )

    assert session.execution_count == 0

    expected = sample_sales.groupby("dept", as_index=True).agg(
        total_rev=("revenue", "sum"),
        max_units=("units", "max"),
        min_units=("units", "min"),
    )

    assert_frame_equal(result.collect(), expected)
    assert session.execution_count == 1


def test_groupby_dropna_behavior(sample_sales: pd.DataFrame) -> None:
    df_with_nulls = pd.DataFrame(
        {
            "group": ["A", None, "A", "B", None],
            "val": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    frame = duckpd.from_pandas(df_with_nulls)

    # dropna=True (default)
    result_dropna = frame.groupby("group", as_index=False, dropna=True).agg(
        total=("val", "sum")
    )
    expected_dropna = df_with_nulls.groupby("group", as_index=False, dropna=True).agg(
        total=("val", "sum")
    )
    assert_frame_equal(result_dropna.collect(), expected_dropna)

    # dropna=False
    result_keepna = frame.groupby("group", as_index=False, dropna=False).agg(
        total=("val", "sum")
    )
    expected_keepna = df_with_nulls.groupby("group", as_index=False, dropna=False).agg(
        total=("val", "sum")
    )
    assert_frame_equal(result_keepna.collect(), expected_keepna)


def test_groupby_sort_false_preserves_order() -> None:
    df = pd.DataFrame(
        {
            "group": ["Z", "A", "Z", "A"],
            "val": [1.0, 2.0, 3.0, 4.0],
        }
    )
    frame = duckpd.from_pandas(df)

    res_sort = frame.groupby("group", sort=True, as_index=False).agg(s=("val", "sum"))
    exp_sort = df.groupby("group", sort=True, as_index=False).agg(s=("val", "sum"))
    assert_frame_equal(res_sort.collect(), exp_sort)


def test_groupby_pipeline_with_filter_assign_sort_limit(
    sample_sales: pd.DataFrame,
) -> None:
    frame = duckpd.from_pandas(sample_sales)

    result = (
        frame[frame["units"] > 5]
        .assign(rev_per_unit=lambda f: f["revenue"] / f["units"])
        .groupby("dept", as_index=False)
        .agg(
            total_rev=("revenue", "sum"),
            avg_rate=("rev_per_unit", "mean"),
        )
        .sort_values("total_rev", ascending=False)
        .limit(1)
    )

    expected = (
        sample_sales[sample_sales["units"] > 5]
        .assign(rev_per_unit=lambda f: f["revenue"] / f["units"])
        .groupby("dept", as_index=False)
        .agg(
            total_rev=("revenue", "sum"),
            avg_rate=("rev_per_unit", "mean"),
        )
        .sort_values("total_rev", ascending=False)
        .head(1)
        .reset_index(drop=True)
    )

    assert_frame_equal(result.collect(), expected)


def test_groupby_invalid_arguments_raise_early(sample_sales: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(sample_sales)

    with pytest.raises(ValueError, match="No group keys"):
        frame.groupby([])

    with pytest.raises(ValueError, match="Duplicate group keys"):
        frame.groupby(["dept", "dept"])

    with pytest.raises(KeyError):
        frame.groupby("non_existent_column")

    g = frame.groupby("dept")

    with pytest.raises(ValueError, match="Must provide at least one aggregation"):
        g.agg()

    with pytest.raises(UnsupportedOperationError, match="Positional aggregation"):
        g.agg("sum")

    with pytest.raises(TypeError, match=r"must be a \(column_name, agg_func\) tuple"):
        g.agg(out="not_a_tuple")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"Target column name .* must be a string"):
        g.agg(out=(123, "sum"))  # type: ignore[arg-type]

    def dummy_func(x: object) -> object:
        return x

    with pytest.raises(UnsupportedOperationError, match="Callable aggregators"):
        g.agg(out=("revenue", dummy_func))

    with pytest.raises(
        UnsupportedOperationError, match="Unsupported aggregate function"
    ):
        g.agg(out=("revenue", "median"))

    with pytest.raises(UnsupportedOperationError, match="requires numeric or boolean"):
        g.agg(out=("region", "sum"))
