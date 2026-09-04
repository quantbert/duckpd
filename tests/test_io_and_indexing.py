"""Differential tests for CSV I/O, persistence, __setitem__, and .loc/.iloc indexing."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import duckdb
import pandas as pd
import pytest

import duckpd as dp
from duckpd import CommitReport, ConcurrentModificationError
from duckpd._executor import CommitFailurePoint
from duckpd.errors import UnorderedOperationError, UnsupportedOperationError


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


def test_persist_preserves_index_order_and_window_semantics() -> None:
    source = pd.DataFrame(
        {
            "row_id": [3, 1, 2, 4],
            "key": [2.0, 1.0, 1.0, None],
            "value": [30, 10, 20, 40],
        }
    )
    session = dp.connect()
    frame = session.from_pandas(source, index="row_id", order_by="key")

    persisted = frame.persist("ordered_indexed_stage")

    assert persisted.index_names == ("row_id",)
    assert persisted.ordering == ("key",)
    expected = source.sort_values("key", kind="stable").set_index("row_id")
    pd.testing.assert_frame_equal(persisted.collect(), expected)
    pd.testing.assert_series_equal(
        persisted["value"].cumsum().collect(),
        expected["value"].cumsum(),
    )
    sliced = cast("dp.DataFrame", persisted.iloc[1:3]).collect()
    pd.testing.assert_frame_equal(sliced, expected.iloc[1:3])


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


def test_assignment_builds_plans_without_compiling_or_reading_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "lazy_assignment.parquet"
    pdf = pd.DataFrame({"value": [1, 2, 3], "flag": [True, False, True]})
    pdf.to_parquet(source)

    with dp.connect() as session:
        frame = session.read_parquet(source)

        def fail_compile(_plan: object) -> None:
            raise AssertionError("assignment attempted to compile the source")

        with monkeypatch.context() as patch:
            patch.setattr(session._compiler, "compile", fail_compile)
            frame["added"] = frame["value"] * 10
            frame.loc[frame["flag"], "value"] = 99
            assigned = frame.assign(derived=frame["added"] + frame["value"])
            filtered = assigned[assigned["derived"] > 0]
            chained = filtered.assign(final=filtered["derived"] * 2)
            assert session.execution_count == 0

        expected = pdf.copy()
        expected["added"] = expected["value"] * 10
        expected.loc[expected["flag"], "value"] = 99
        expected["derived"] = expected["added"] + expected["value"]
        expected["final"] = expected["derived"] * 2
        pd.testing.assert_frame_equal(chained.collect(), expected)


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
    pd.testing.assert_frame_equal(res_loc_list, pdf.set_index("id").loc[[1, 3]])

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
    res_slice = cast("dp.DataFrame", df.iloc[1:4]).collect()
    exp_slice = pdf.iloc[1:4].reset_index(drop=True)
    pd.testing.assert_frame_equal(res_slice.reset_index(drop=True), exp_slice)

    # .iloc with start only
    res_slice_start = cast("dp.DataFrame", df.iloc[2:]).collect()
    pd.testing.assert_frame_equal(
        res_slice_start.reset_index(drop=True), pdf.iloc[2:].reset_index(drop=True)
    )

    # Snapshot-backed pandas sources have a stable implicit source order.
    df_unordered = dp.from_pandas(pdf)
    pd.testing.assert_frame_equal(
        cast("dp.DataFrame", df_unordered.iloc[1:3]).collect().reset_index(drop=True),
        pdf.iloc[1:3].reset_index(drop=True),
    )


def test_iloc_rejects_external_scan_without_declared_order(tmp_path: Path) -> None:
    path = tmp_path / "unordered.csv"
    pd.DataFrame({"value": [3, 1, 2]}).to_csv(path, index=False)
    frame = dp.read_csv(path)

    with pytest.raises(UnorderedOperationError):
        frame.iloc[1:3]


def test_loc_multiindex_exact_partial_and_null_keys_are_lazy() -> None:
    pdf = pd.DataFrame(
        {
            "group": ["a", "a", "b", None],
            "item": [1, 2, 1, 1],
            "value": [10, 20, 30, 40],
        }
    )
    session = dp.connect()
    frame = session.from_pandas(pdf).set_index(["group", "item"])

    exact = cast("dp.DataFrame", frame.loc[("a", 2)])
    partial = cast("dp.DataFrame", frame.loc[("a",)])
    null_key = cast("dp.DataFrame", frame.loc[(None, 1)])

    assert session.execution_count == 0
    indexed = pdf.set_index(["group", "item"])
    pd.testing.assert_frame_equal(exact.collect(), indexed.loc[[("a", 2)]])
    pd.testing.assert_frame_equal(partial.collect(), indexed.loc[["a"]])
    assert null_key.collect()["value"].tolist() == [40]


def test_loc_multiindex_column_projection_and_protected_assignment() -> None:
    pdf = pd.DataFrame({"group": ["a", "b"], "item": [1, 2], "value": [10, 20]})
    frame = dp.from_pandas(pdf).set_index(["group", "item"], drop=False)

    result = cast("dp.Series", frame.loc[("a", 1), "value"])
    indexed = pdf.set_index(["group", "item"], drop=False)
    expected = indexed.loc[indexed.index.isin([("a", 1)]), "value"]
    pd.testing.assert_series_equal(result.collect(), expected)

    with pytest.raises(ValueError, match="index or ordering"):
        frame.loc[frame["value"] > 0, "group"] = "changed"


def test_iloc_two_dimensional_column_selection_is_lazy() -> None:
    pdf = pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30], "b": [40, 50, 60]})
    session = dp.connect()
    frame = session.from_pandas(pdf, order_by="id")

    sliced = cast("dp.DataFrame", frame.iloc[1:, [2, 0]])
    column = cast("dp.Series", frame.iloc[:, 1])

    assert session.execution_count == 0
    pd.testing.assert_frame_equal(
        sliced.collect().reset_index(drop=True),
        pdf.iloc[1:, [2, 0]].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(column.collect(), pdf.iloc[:, 1])


def test_loc_list_order_and_duplicates_match_pandas() -> None:
    pdf = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["a", "b", "c", "d"],
            "val": [10, 20, 30, 40],
        }
    )
    session = dp.connect()
    df = session.from_pandas(pdf, index="id")

    # 1. Reordered list of keys [3, 1, 4]
    res_reordered = cast("dp.DataFrame", df.loc[[3, 1, 4]])
    assert session.execution_count == 0  # Lazy
    expected_reordered = pdf.set_index("id").loc[[3, 1, 4]]
    pd.testing.assert_frame_equal(res_reordered.collect(), expected_reordered)

    # 2. Duplicate keys in requested list [2, 1, 2]
    res_dups = cast("dp.DataFrame", df.loc[[2, 1, 2]])
    expected_dups = pdf.set_index("id").loc[[2, 1, 2]]
    pd.testing.assert_frame_equal(res_dups.collect(), expected_dups)


def test_loc_list_source_with_duplicate_index_matches_pandas() -> None:
    pdf = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "val": [10, 20, 30],
        }
    )
    df = dp.from_pandas(pdf, index="id")

    # Requesting [1] should return both rows where id == 1
    res = cast("dp.DataFrame", df.loc[[1]]).collect()
    expected = pdf.set_index("id").loc[[1]]
    pd.testing.assert_frame_equal(res, expected)

    # Requesting [2, 1] should return row for 2, then both rows for 1
    res_multi = cast("dp.DataFrame", df.loc[[2, 1]]).collect()
    expected_multi = pdf.set_index("id").loc[[2, 1]]
    pd.testing.assert_frame_equal(res_multi, expected_multi)


def test_loc_list_empty_matches_pandas() -> None:
    pdf = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    df = dp.from_pandas(pdf, index="id")

    res = cast("dp.DataFrame", df.loc[[]]).collect()
    expected = pdf.set_index("id").loc[[]]
    pd.testing.assert_frame_equal(res, expected)


def test_loc_set_raises_type_error() -> None:
    pdf = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    df = dp.from_pandas(pdf, index="id")

    with pytest.raises(TypeError, match="Passing a set as an indexer is not supported"):
        df.loc[{1, 2}]


def test_loc_list_missing_key_raises_key_error() -> None:
    pdf = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    session = dp.connect()
    df = session.from_pandas(pdf, index="id")

    missing_lazy = cast("dp.DataFrame", df.loc[[1, 99]])
    assert session.execution_count == 0  # Lazy until execution

    with pytest.raises(KeyError, match="not in index"):
        missing_lazy.collect()


@pytest.mark.parametrize(
    "index_label",
    ["_loc_order_", "_loc_k_0", "__duckpd_matched__", 'odd " index'],
)
def test_loc_list_internal_names_cannot_collide(index_label: str) -> None:
    source = pd.DataFrame({index_label: [10, 20], "value": ["a", "b"]})
    frame = dp.from_pandas(source, index=index_label)

    result = cast("dp.DataFrame", frame.loc[[20, 10]]).collect()
    expected = source.set_index(index_label).loc[[20, 10]]

    pd.testing.assert_frame_equal(result, expected)


def test_loc_list_unordered_duplicate_source_does_not_claim_total_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate-index.csv"
    pd.DataFrame({"id": [1, 1, 2], "value": ["a", "b", "c"]}).to_csv(path, index=False)
    selected = cast("dp.DataFrame", dp.read_csv(path, index="id").loc[[1, 2]])

    assert selected._plan.metadata.ordering.keys == ()
    with pytest.raises(UnorderedOperationError):
        selected.iloc[0:2]
    with pytest.raises(UnorderedOperationError):
        selected["value"].shift()


def test_loc_list_multiindex_matches_pandas() -> None:
    pdf = pd.DataFrame(
        {
            "g": ["a", "a", "b", "c"],
            "i": [1, 2, 1, 1],
            "val": [10, 20, 30, 40],
        }
    )
    df = dp.from_pandas(pdf, index=["g", "i"])

    # MultiIndex list of tuples in custom order with duplicates
    keys = [("b", 1), ("a", 2), ("b", 1)]
    res = cast("dp.DataFrame", df.loc[keys]).collect()
    expected = pdf.set_index(["g", "i"]).loc[keys]
    pd.testing.assert_frame_equal(res, expected)

    # Missing tuple key raises KeyError
    with pytest.raises(KeyError, match="not in index"):
        df.loc[[("a", 1), ("z", 99)]].collect()


def test_loc_list_enables_deterministic_positional_and_window_operations() -> None:
    pdf = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "val": [10, 20, 30, 40],
        }
    )
    df = dp.from_pandas(pdf, index="id")

    # Because loc with a list establishes an explicit request ordering,
    # positional and window operations work deterministically
    reordered = cast("dp.DataFrame", df.loc[[4, 2, 1]])
    assert (
        reordered.ordering == ()
    )  # hidden order column is not exposed in public ordering

    # .iloc slicing
    sliced = cast("dp.DataFrame", reordered.iloc[0:2]).collect()
    expected_sliced = pdf.set_index("id").loc[[4, 2, 1]].iloc[0:2]
    pd.testing.assert_frame_equal(sliced, expected_sliced)

    # cumulative sum
    cumsum = reordered["val"].cumsum().collect()
    expected_cumsum = pdf.set_index("id").loc[[4, 2, 1]]["val"].cumsum()
    pd.testing.assert_series_equal(cumsum, expected_cumsum)


def test_save_as_table() -> None:
    with dp.connect() as session:
        pdf = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df = session.from_pandas(pdf)

        # 1. Default mode is "error", creates new table
        df.save_as_table("tbl_test")
        read1 = session.table("tbl_test").collect()
        pd.testing.assert_frame_equal(read1, pdf)

        # 2. Saving again with mode="error" raises ValueError
        with pytest.raises(ValueError, match="already exists"):
            df.save_as_table("tbl_test", mode="error")

        # 3. Overwrite mode replaces content
        pdf_over = pd.DataFrame({"a": [10], "b": ["z"]})
        session.from_pandas(pdf_over).save_as_table("tbl_test", mode="overwrite")
        read2 = session.table("tbl_test").collect()
        pd.testing.assert_frame_equal(read2, pdf_over)

        # 4. Append mode appends rows (even if columns are reordered)
        pdf_app = pd.DataFrame({"b": ["w"], "a": [20]})
        session.from_pandas(pdf_app).save_as_table("tbl_test", mode="append")
        read3 = session.table("tbl_test").collect()
        expected_app = pd.DataFrame({"a": [10, 20], "b": ["z", "w"]})
        pd.testing.assert_frame_equal(read3, expected_app)

        # 5. Append mode to non-existent table creates it
        session.from_pandas(pdf).save_as_table("tbl_new", mode="append")
        pd.testing.assert_frame_equal(session.table("tbl_new").collect(), pdf)

        # 6. Append mode with missing/extra columns raises ValueError
        pdf_extra = pd.DataFrame({"a": [1], "b": ["x"], "c": [99]})
        with pytest.raises(ValueError, match="Schema mismatch when appending"):
            session.from_pandas(pdf_extra).save_as_table("tbl_test", mode="append")

        # 7. Append mode with same column names but incompatible type raises ValueError
        pdf_bad_type = pd.DataFrame({"a": ["not_an_int"], "b": ["valid_str"]})
        with pytest.raises(ValueError, match="column 'a': expected"):
            session.from_pandas(pdf_bad_type).save_as_table("tbl_test", mode="append")

        # 8. Failed append leaves original table intact
        read_after_fail = session.table("tbl_test").collect()
        pd.testing.assert_frame_equal(read_after_fail, expected_app)
        # 9. Invalid mode raises ValueError
        with pytest.raises(ValueError, match="Unknown mode"):
            df.save_as_table("tbl_test", mode="invalid")  # type: ignore[arg-type]


def test_save_as_table_failed_overwrite_rolls_back() -> None:
    with dp.connect() as session:
        original = pd.DataFrame({"a": [1]})
        session.from_pandas(original).save_as_table("rollback_target")
        failing = session.sql("SELECT CAST(error('boom') AS BIGINT) AS a")

        with pytest.raises(duckdb.InvalidInputException, match="boom"):
            failing.save_as_table("rollback_target", mode="overwrite")

        pd.testing.assert_frame_equal(
            session.table("rollback_target").collect(), original
        )


def test_parquet_atomic_commit_with_index_preservation(tmp_path: Path) -> None:
    src_file = tmp_path / "data.parquet"
    pdf = pd.DataFrame(
        {"val": [10, 20, 30]},
        index=pd.Index([1, 2, 3], name="id"),
    )
    pdf.to_parquet(src_file)

    df = dp.read_parquet(src_file, index="id")
    df["val"] = df["val"] * 2

    report = df.commit(retain_previous=True)
    assert isinstance(report, CommitReport)
    assert report.rows_written == 3
    assert report.bytes_written > 0
    assert report.backup_path is not None
    assert Path(report.backup_path).exists()
    pd.testing.assert_frame_equal(pd.read_parquet(report.backup_path), pdf)

    # pandas/Arrow metadata reconstructs the original named index after commit.
    disk_pdf = pd.read_parquet(src_file)
    expected = pd.DataFrame(
        {"val": [20, 40, 60]},
        index=pd.Index([1, 2, 3], name="id"),
    )
    pd.testing.assert_frame_equal(disk_pdf, expected)

    # DataFrame handle reflects committed state and continues working
    res = df.collect()
    assert res["val"].tolist() == [20, 40, 60]
    assert res.index.name == "id"


def test_parquet_atomic_commit_rejections(tmp_path: Path) -> None:
    src_file = tmp_path / "source.parquet"
    pdf = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    pdf.to_parquet(src_file)

    df = dp.read_parquet(src_file, index="id")

    # 1. Filtered plans are rejected (must be row-preserving)
    filtered = df[df["val"] > 10]
    with pytest.raises(UnsupportedOperationError, match="row-preserving plan"):
        filtered.commit()

    # 2. Schema-altering plans (extra columns) are rejected
    extra = df.assign(extra=100)
    with pytest.raises(UnsupportedOperationError, match="schema preservation"):
        extra.commit()

    # 3. Non-Parquet source is rejected
    mem_df = dp.from_pandas(pdf)
    with pytest.raises(UnsupportedOperationError, match="only supports ParquetSource"):
        mem_df.commit()


def test_parquet_atomic_commit_concurrency_and_failure_injection(
    tmp_path: Path,
) -> None:
    import os
    import time

    src_file = tmp_path / "concurrency.parquet"
    pdf = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    pdf.to_parquet(src_file)

    df = dp.read_parquet(src_file, index="id")
    df["val"] = df["val"] + 100

    # 1. Concurrent modification raises ConcurrentModificationError
    def touch_concurrent(point: CommitFailurePoint) -> None:
        if point == "after_staging_write":
            time.sleep(0.01)
            os.utime(src_file, None)

    with pytest.raises(ConcurrentModificationError, match="modified concurrently"):
        df._session._executor.commit(df._plan, _failure_injector=touch_concurrent)

    # 2. Failure injection leaves original file intact and staging cleaned up
    def blow_up(point: CommitFailurePoint) -> None:
        if point == "before_replace":
            raise RuntimeError("Simulated failure")

    with pytest.raises(RuntimeError, match="Simulated failure"):
        df._session._executor.commit(df._plan, _failure_injector=blow_up)

    # Original file is intact and readable
    disk_pdf = pd.read_parquet(src_file)
    assert disk_pdf["val"].tolist() == [10, 20, 30]
    assert disk_pdf["id"].tolist() == [1, 2, 3]

    # Staging files are cleaned up
    staging_files = list(tmp_path.glob(".duckpd_staging_*"))
    assert len(staging_files) == 0


@pytest.mark.parametrize(
    "failure_point",
    [
        "before_staging",
        "during_write",
        "after_staging_write",
        "during_validation",
        "before_backup",
        "after_backup",
        "before_replace",
    ],
)
def test_parquet_commit_failure_matrix_preserves_original(
    tmp_path: Path, failure_point: CommitFailurePoint
) -> None:
    source = tmp_path / "matrix.parquet"
    original = pd.DataFrame({"value": [1, 2, 3]})
    original.to_parquet(source)
    original_bytes = source.read_bytes()
    frame = dp.read_parquet(source)
    frame["value"] = frame["value"] + 10
    original_plan = frame._plan

    def inject(point: CommitFailurePoint) -> None:
        if point == failure_point:
            raise RuntimeError(f"injected at {point}")

    with pytest.raises(RuntimeError, match=f"injected at {failure_point}"):
        frame._session._executor.commit(
            frame._plan,
            retain_previous=True,
            _failure_injector=inject,
        )

    assert frame._plan is original_plan
    assert source.read_bytes() == original_bytes
    pd.testing.assert_frame_equal(pd.read_parquet(source), original)
    assert not list(tmp_path.glob(".duckpd_staging_*"))
    assert not list(tmp_path.glob("matrix_backup_*.parquet"))


def test_parquet_commit_replace_failure_preserves_original(tmp_path: Path) -> None:
    source = tmp_path / "replace.parquet"
    original = pd.DataFrame({"value": [1, 2, 3]})
    original.to_parquet(source)
    original_bytes = source.read_bytes()
    frame = dp.read_parquet(source)
    frame["value"] = frame["value"] + 10

    def fail_replace(_source: Path, _staging: Path) -> None:
        raise OSError("injected replace failure")

    with pytest.raises(OSError, match="injected replace failure"):
        frame._session._executor.commit(
            frame._plan,
            retain_previous=True,
            _replace_file=fail_replace,
        )

    assert source.read_bytes() == original_bytes
    pd.testing.assert_frame_equal(pd.read_parquet(source), original)
    assert not list(tmp_path.glob(".duckpd_staging_*"))
    assert not list(tmp_path.glob("replace_backup_*.parquet"))


def test_parquet_commit_resolves_relative_source_at_read_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "source"
    other_dir = tmp_path / "other"
    source_dir.mkdir()
    other_dir.mkdir()
    original_path = source_dir / "data.parquet"
    decoy_path = other_dir / "data.parquet"
    pd.DataFrame({"value": [1, 2]}).to_parquet(original_path)
    pd.DataFrame({"value": [90, 99]}).to_parquet(decoy_path)

    with dp.connect() as session:
        monkeypatch.chdir(source_dir)
        frame = session.read_parquet("data.parquet")
        frame["value"] = frame["value"] + 10
        monkeypatch.chdir(other_dir)
        frame.commit()

    assert pd.read_parquet(original_path)["value"].tolist() == [11, 12]
    assert pd.read_parquet(decoy_path)["value"].tolist() == [90, 99]


@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="Windows ACL preservation uses ReplaceFileW rather than POSIX modes",
)
def test_parquet_commit_preserves_posix_permissions(tmp_path: Path) -> None:
    import stat

    source = tmp_path / "restricted.parquet"
    pd.DataFrame({"value": [1, 2]}).to_parquet(source)
    source.chmod(0o600)

    frame = dp.read_parquet(source)
    frame["value"] = frame["value"] + 1
    frame.commit()

    assert stat.S_IMODE(source.stat().st_mode) == 0o600


def test_parquet_commit_supports_declared_compression_codecs(
    tmp_path: Path,
) -> None:
    for compression in (
        "uncompressed",
        "brotli",
        "snappy",
        "lz4",
        "lz4_raw",
        "gzip",
        "zstd",
    ):
        source = tmp_path / f"{compression}.parquet"
        pd.DataFrame({"value": [1, 2]}).to_parquet(source)
        frame = dp.read_parquet(source)
        frame["value"] = frame["value"] + 1

        frame.commit(compression=compression)

        assert pd.read_parquet(source)["value"].tolist() == [2, 3]
