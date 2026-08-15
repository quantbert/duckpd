"""Differential tests for CSV I/O, persistence, __setitem__, and .loc/.iloc indexing."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import duckpd as dp
from duckpd.errors import UnorderedOperationError


def test_csv_read_write_differential(tmp_path: Path) -> None:
    csv_file = tmp_path / "test.csv"
    pdf = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 80.0, 88.5],
        }
    )
    pdf.to_csv(csv_file, index=False)

    df_dp = dp.read_csv(csv_file)
    pd.testing.assert_frame_equal(df_dp.collect(), pdf)

    # Test write_csv and to_csv
    out_csv = tmp_path / "out.csv"
    df_dp.write_csv(out_csv)
    pdf_out = pd.read_csv(out_csv)
    pd.testing.assert_frame_equal(pdf_out, pdf)


def test_persist_and_to_pandas() -> None:
    session = dp.Session()
    pdf = pd.DataFrame({"x": [10, 20, 30], "y": [1, 2, 3]})
    df = session.from_pandas(pdf)
    enriched = df.assign(z=df["x"] * df["y"])

    persisted = enriched.persist("custom_temp_table")
    # persist itself executes 1 query to create table.
    # calling persisted.to_pandas() executes 1 query on the table.
    # enriched.collect() executes 1 query on enriched.
    pd.testing.assert_frame_equal(persisted.to_pandas(), enriched.collect())
    assert session.execution_count == 3


def test_dataframe_setitem_lazy() -> None:
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    df = dp.from_pandas(pdf)

    # Scalar assignment
    df["c"] = 100
    pdf["c"] = 100
    pd.testing.assert_frame_equal(df.collect(), pdf, check_dtype=False)

    # Multi-column scalar assignment
    df[["m1", "m2"]] = 50
    pdf[["m1", "m2"]] = 50
    pd.testing.assert_frame_equal(df.collect(), pdf, check_dtype=False)

    # Multi-column sequence assignment
    df[["s1", "s2"]] = (df["a"] * 10, df["b"] * 10)
    pdf["s1"] = pdf["a"] * 10
    pdf["s2"] = pdf["b"] * 10
    pd.testing.assert_frame_equal(df.collect(), pdf, check_dtype=False)

    # Multi-column DataFrame assignment
    sub_df = df[["s1", "s2"]]
    df[["d1", "d2"]] = sub_df
    pdf["d1"] = pdf["s1"]
    pdf["d2"] = pdf["s2"]
    pd.testing.assert_frame_equal(df.collect(), pdf, check_dtype=False)

    # Series expression assignment
    df["d"] = df["a"] + df["b"]
    pdf["d"] = pdf["a"] + pdf["b"]
    pd.testing.assert_frame_equal(df.collect(), pdf, check_dtype=False)

    # Column replacement
    df["a"] = df["a"] * 2
    pdf["a"] = pdf["a"] * 2
    pd.testing.assert_frame_equal(df.collect(), pdf, check_dtype=False)

    # Invalid empty keys / mismatched lengths
    with pytest.raises(ValueError):
        df[[]] = 1
    with pytest.raises(ValueError):
        df[["x", "y"]] = [1]


def test_loc_reads_and_masked_assignment() -> None:
    pdf = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "val": [10, 20, 30, 40],
            "val2": [100, 200, 300, 400],
            "flag": [True, False, True, False],
        }
    )
    df = dp.from_pandas(pdf, index="id")

    # .loc read with explicit index (single scalar and list)
    res_loc = cast("dp.DataFrame", df.loc[2]).collect()
    assert res_loc.iloc[0]["val"] == 20

    res_loc_list = cast("dp.DataFrame", df.loc[[1, 3]]).collect()
    assert len(res_loc_list) == 2

    # .loc with column selection (single column and list of columns)
    res_loc_col = cast("dp.Series", df.loc[:, "val"]).collect()
    pd.testing.assert_series_equal(res_loc_col, pdf.set_index("id")["val"])

    res_loc_cols = cast("dp.DataFrame", df.loc[:, ["val", "val2"]]).collect()
    pd.testing.assert_frame_equal(res_loc_cols, pdf.set_index("id")[["val", "val2"]])

    # .loc masked read
    mask_dp = df["flag"]
    res_mask = cast("dp.DataFrame", df.loc[mask_dp]).collect()
    pdf_indexed = pdf.set_index("id")
    exp_mask = pdf_indexed[pdf_indexed["flag"]]
    pd.testing.assert_frame_equal(res_mask, exp_mask)

    # .loc masked assignment
    df.loc[df["val"] > 25, "val"] = 999
    pdf_expected = pdf.copy().set_index("id")
    pdf_expected.loc[pdf_expected["val"] > 25, "val"] = 999
    pd.testing.assert_frame_equal(df.collect(), pdf_expected)

    # .loc masked assignment across multiple columns
    df.loc[df["flag"], ["val", "val2"]] = 0
    pdf_expected.loc[pdf_expected["flag"], ["val", "val2"]] = 0
    pd.testing.assert_frame_equal(df.collect(), pdf_expected)


def test_iloc_slicing_differential() -> None:
    pdf = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "val": [10, 20, 30, 40, 50],
        }
    )
    df = dp.from_pandas(pdf, order_by="id")

    # .iloc slices
    res_slice = df.iloc[1:4].collect()
    exp_slice = pdf.iloc[1:4].reset_index(drop=True)
    pd.testing.assert_frame_equal(res_slice.reset_index(drop=True), exp_slice)

    # .iloc with start only
    res_slice_start = df.iloc[2:].collect()
    pd.testing.assert_frame_equal(
        res_slice_start.reset_index(drop=True), pdf.iloc[2:].reset_index(drop=True)
    )

    # .iloc without ordering raises UnorderedOperationError
    df_unordered = dp.from_pandas(pdf)
    with pytest.raises(UnorderedOperationError):
        df_unordered.iloc[1:3]
