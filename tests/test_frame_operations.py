from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal

import duckpd
from duckpd.errors import AlignmentError

InvalidOperation = Callable[[duckpd.DataFrame], object]

INVALID_BOUNDED_OPERATIONS: list[tuple[InvalidOperation, str]] = [
    (lambda frame: frame.limit(-1), "count must be non-negative"),
    (lambda frame: frame.limit(1, offset=-1), "offset must be non-negative"),
    (lambda frame: frame.head(-1), "count must be non-negative"),
    (lambda frame: frame.sort_values([]), "at least one column"),
    (
        lambda frame: frame.sort_values(["value", "other"], ascending=[True]),
        "Length of ascending",
    ),
]


@pytest.fixture
def source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [1, 3, 2, 4],
            "other": [10, 20, 30, 40],
            "label": ["a", "y'quoted", "b", None],
        }
    )


def test_expression_pipeline_is_lazy(source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(source)

    result = (
        frame[(frame["value"] > 1) & (frame["other"] >= 20)]
        .assign(
            total=lambda current: current["value"] + current["other"],
            doubled=lambda current: current["total"] * 2,
        )
        .sort_values("doubled", ascending=False)[["label", "doubled"]]
        .limit(2)
    )

    assert session.execution_count == 0
    expected = pd.DataFrame(
        {"label": [None, "b"], "doubled": [88, 64]},
        index=[3, 2],
    ).reset_index(drop=True)
    assert_frame_equal(result.collect(), expected)
    assert session.execution_count == 1


def test_hostile_string_literal_is_not_interpolated(source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(source)

    result = frame[frame["label"] == "y'quoted"][["value", "label"]].collect()

    expected = source.loc[source["label"] == "y'quoted", ["value", "label"]]
    assert_frame_equal(result, expected.reset_index(drop=True))


def test_quoted_column_identifier_is_structured() -> None:
    source = pd.DataFrame({'odd "name': [1, 2]})
    frame = duckpd.from_pandas(source)

    result = frame.assign(calculated=frame['odd "name'] + 1)[
        ['odd "name', "calculated"]
    ].collect()

    expected = source.assign(calculated=[2, 3])
    assert_frame_equal(result, expected)


def test_reverse_and_unary_arithmetic() -> None:
    frame = duckpd.from_pandas(pd.DataFrame({"value": [2]}))

    result = frame.assign(
        add=1 + frame["value"],
        subtract=10 - frame["value"],
        multiply=3 * frame["value"],
        divide=8 / frame["value"],
        modulo=7 % frame["value"],
        negative=-frame["value"],
        positive=+frame["value"],
    ).collect()

    expected = pd.DataFrame(
        {
            "value": [2],
            "add": [3],
            "subtract": [8],
            "multiply": [6],
            "divide": [4.0],
            "modulo": [1],
            "negative": [-2],
            "positive": [2],
        }
    )
    assert_frame_equal(result, expected)


def test_boolean_invert() -> None:
    frame = duckpd.from_pandas(pd.DataFrame({"flag": [True, False]}))

    result = frame[~frame["flag"]].collect()

    assert_frame_equal(result, pd.DataFrame({"flag": [False]}))


def test_head_is_bounded_and_eager(source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(source)

    result = frame.sort_values("value").head(2)

    assert result["value"].tolist() == [1, 2]
    assert session.execution_count == 1


def test_arrow_outputs(source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(source).limit(3)

    table = frame.to_arrow()
    batches = list(frame.to_arrow_batches(batch_size=2))

    assert isinstance(table, pa.Table)
    assert table.num_rows == 3
    assert [batch.num_rows for batch in batches] == [2, 1]


def test_from_arrow_is_lazy() -> None:
    session = duckpd.connect()
    table = pa.table({"value": [1, 2]})

    frame = session.from_arrow(table)

    assert session.execution_count == 0
    assert_frame_equal(frame.collect(), pd.DataFrame({"value": [1, 2]}))


def test_write_parquet_does_not_use_pandas_conversion(
    source: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.parquet"
    frame = duckpd.from_pandas(source).assign(total=lambda item: item["value"] + 1)

    def fail_pandas_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("pandas write path was used")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_pandas_write)
    frame.write_parquet(path)

    result = pd.read_parquet(path)
    assert result["total"].tolist() == [2, 4, 3, 5]


def test_explain_contains_all_plan_views(source: pd.DataFrame) -> None:
    session = duckpd.connect()
    frame = session.from_pandas(source).assign(total=lambda item: item["value"] + 1)

    explanation = frame.explain()

    assert "DuckPD logical plan:" in explanation
    assert "DuckDB SQL:" in explanation
    assert "DuckDB physical plan:" in explanation
    assert "total" in explanation
    assert session.execution_count == 1


def test_cross_frame_series_requires_alignment() -> None:
    left = duckpd.from_pandas(pd.DataFrame({"value": [1]}))
    right = duckpd.from_pandas(pd.DataFrame({"value": [1]}))

    with pytest.raises(AlignmentError, match="explicit index alignment"):
        _ = left["value"] + right["value"]
    with pytest.raises(AlignmentError, match="explicit index alignment"):
        _ = left[right["value"] > 0]


@pytest.mark.parametrize(
    ("operation", "message"),
    INVALID_BOUNDED_OPERATIONS,
)
def test_invalid_bounded_operations_fail_before_execution(
    source: pd.DataFrame,
    operation: InvalidOperation,
    message: str,
) -> None:
    frame = duckpd.from_pandas(source)

    with pytest.raises(ValueError, match=message):
        operation(frame)


def test_invalid_projection_and_assignment_fail_early(source: pd.DataFrame) -> None:
    frame = duckpd.from_pandas(source)

    with pytest.raises(KeyError):
        _ = frame["missing"]
    with pytest.raises(ValueError, match="duplicate column labels"):
        _ = frame[["value", "value"]]
    with pytest.raises(TypeError, match="scalar or Series"):
        frame.assign(invalid=frame)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        frame.to_arrow_batches(0)
    with pytest.raises(ValueError, match=r"truth value.*ambiguous"):
        bool(frame["value"])
