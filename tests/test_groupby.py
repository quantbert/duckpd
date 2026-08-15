from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import duckpd
from duckpd.errors import AlignmentError, UnsupportedOperationError


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


def test_groupby_rejects_unobserved_categorical_groups() -> None:
    source = pd.DataFrame({"group": ["a"], "value": [1]})
    frame = duckpd.from_pandas(source)

    with pytest.raises(UnsupportedOperationError, match="unobserved categorical"):
        frame.groupby("group", observed=False)


def test_groupby_sort_false_preserves_order() -> None:
    df = pd.DataFrame(
        {
            "group": ["Z", "A", "Z", "A"],
            "val": [1.0, 2.0, 3.0, 4.0],
        }
    )
    frame = duckpd.from_pandas(df)

    result = frame.groupby("group", sort=False, as_index=False).agg(s=("val", "sum"))
    expected = df.groupby("group", sort=False, as_index=False).agg(s=("val", "sum"))
    assert_frame_equal(result.collect(), expected)

    size_result = frame.groupby("group", sort=False).size().collect()
    size_expected = df.groupby("group", sort=False).size().to_frame("size")
    assert_frame_equal(size_result, size_expected)


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


def test_groupby_dict_agg_matches_pandas(sample_sales: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(sample_sales)

    result = frame.groupby("dept", as_index=True).agg(
        {"revenue": "sum", "units": "mean"}
    )
    assert session.execution_count == 0

    expected = sample_sales.groupby("dept", as_index=True).agg(
        {"revenue": "sum", "units": "mean"}
    )
    assert_frame_equal(result.collect(), expected)
    assert session.execution_count == 1


def test_groupby_func_str_agg_matches_pandas(sample_sales: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(sample_sales)

    result = frame.groupby("dept", as_index=True).agg("sum")
    expected = sample_sales.groupby("dept", as_index=True)[
        ["revenue", "units", "bonus_eligible"]
    ].sum()
    assert_frame_equal(result.collect(), expected)


def test_groupby_convenience_methods_match_pandas(sample_sales: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(sample_sales)

    g = frame.groupby("dept", as_index=True)
    pg = sample_sales.groupby("dept", as_index=True)

    assert_frame_equal(g.sum(numeric_only=True).collect(), pg.sum(numeric_only=True))
    assert_frame_equal(g.mean(numeric_only=True).collect(), pg.mean(numeric_only=True))
    assert_frame_equal(g.min(numeric_only=True).collect(), pg.min(numeric_only=True))
    assert_frame_equal(g.max(numeric_only=True).collect(), pg.max(numeric_only=True))
    assert_frame_equal(
        g.count().collect(),
        pg[["region", "revenue", "units", "bonus_eligible"]].count(),
    )
    assert_frame_equal(g.size().collect(), pg.size().to_frame("size"))


def test_groupby_column_indexing_matches_pandas(sample_sales: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(sample_sales)

    g = frame.groupby("dept", as_index=True)
    pg = sample_sales.groupby("dept", as_index=True)

    assert_frame_equal(
        g["revenue"].sum().collect(), pg["revenue"].sum().to_frame("revenue")
    )
    assert_frame_equal(
        g["revenue"].mean().collect(), pg["revenue"].mean().to_frame("revenue")
    )
    assert_frame_equal(
        g["revenue"].min().collect(), pg["revenue"].min().to_frame("revenue")
    )
    assert_frame_equal(
        g["revenue"].max().collect(), pg["revenue"].max().to_frame("revenue")
    )
    assert_frame_equal(
        g[["revenue", "units"]].sum().collect(), pg[["revenue", "units"]].sum()
    )


def test_series_groupby_matches_pandas(sample_sales: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(sample_sales)

    s = frame["revenue"]
    ps = sample_sales["revenue"]

    sg = s.groupby(frame["dept"], as_index=True)
    psg = ps.groupby(sample_sales["dept"], as_index=True)

    assert_frame_equal(sg.sum().collect(), psg.sum().to_frame("revenue"))
    assert_frame_equal(sg.mean().collect(), psg.mean().to_frame("revenue"))
    assert_frame_equal(sg.min().collect(), psg.min().to_frame("revenue"))
    assert_frame_equal(sg.max().collect(), psg.max().to_frame("revenue"))
    assert_frame_equal(sg.count().collect(), psg.count().to_frame("revenue"))
    assert_frame_equal(sg.std().collect(), psg.std().to_frame("revenue"))
    assert_frame_equal(sg.var().collect(), psg.var().to_frame("revenue"))
    assert_frame_equal(sg.median().collect(), psg.median().to_frame("revenue"))
    assert_frame_equal(sg.size().collect(), psg.size().to_frame("size"))


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

    with pytest.raises(ValueError, match="Cannot pass both func and named"):
        g.agg("sum", total=("revenue", "sum"))

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
        g.agg(out=("revenue", "invalid_agg_func"))

    with pytest.raises(UnsupportedOperationError, match="requires numeric or boolean"):
        g.agg(out=("region", "sum"))


def test_series_groupby_various_key_types(sample_sales: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(sample_sales)

    # by as string label
    sg_str = frame["revenue"].groupby("dept")
    assert_frame_equal(
        sg_str.sum().collect(),
        sample_sales.groupby("dept")["revenue"].sum().to_frame("revenue"),
    )

    # by as unnamed Series
    unnamed_grp = frame["dept"] + "_suffix"
    sg_unnamed = frame["revenue"].groupby(unnamed_grp)
    res_unnamed = sg_unnamed.sum().collect()
    assert len(res_unnamed) == 2

    # by as sequence of Series
    sg_seq_ser = frame["revenue"].groupby([frame["dept"], frame["region"]])
    assert len(sg_seq_ser.sum().collect()) == 4

    # by as sequence of strings
    sg_seq_str = frame["revenue"].groupby(["dept", "region"])
    assert len(sg_seq_str.sum().collect()) == 4

    # cross-frame alignment error
    s2 = duckpd.connect().from_pandas(sample_sales)
    with pytest.raises(AlignmentError, match="different frame"):
        frame["revenue"].groupby(s2["dept"])
    with pytest.raises(AlignmentError, match="different frame"):
        frame["revenue"].groupby([frame["dept"], s2["region"]])


def test_groupby_error_cases(sample_sales: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(sample_sales)
    g = frame.groupby("dept")

    with pytest.raises(
        UnsupportedOperationError, match="Unsupported agg argument type"
    ):
        g.agg(123)  # type: ignore[arg-type]

    with pytest.raises(
        TypeError, match="Dictionary aggregation keys must be column name strings"
    ):
        g.agg({123: "sum"})  # type: ignore[dict-item]

    with pytest.raises(UnsupportedOperationError, match="only single string function"):
        g.agg({"revenue": ["sum", "mean"]})  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="by must be a string, Series, or sequence"):
        frame["revenue"].groupby(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="No group keys were provided"):
        frame["revenue"].groupby([])

    with pytest.raises(TypeError, match="Group keys must be all strings or all Series"):
        frame["revenue"].groupby(["dept", frame["region"]])  # type: ignore[list-item]

    with pytest.raises(UnsupportedOperationError, match="function name string"):
        frame["revenue"].groupby("dept").agg(123)  # type: ignore[arg-type]
