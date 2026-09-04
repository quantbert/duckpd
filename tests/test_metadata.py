from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import duckpd
from duckpd._logical import (
    AggregateOperator,
    BinaryExpression,
    BinaryOperator,
    ColumnRef,
    FilterPlan,
    LimitPlan,
    LiteralValue,
    LogicalPlan,
    ProjectPlan,
    ScanPlan,
    SortPlan,
    expression_metadata,
)
from duckpd._metadata import validate_metadata
from duckpd._reductions import aggregate_plan
from duckpd._typing import binary_numeric_type


def test_source_index_and_order_are_lazy() -> None:
    source = pd.DataFrame(
        {
            "order_id": [2, 1, 3],
            "created": [20, 10, 30],
            "amount": [50.0, 25.0, 75.0],
        }
    )
    session = duckpd.connect()

    frame = session.from_pandas(
        source,
        index="order_id",
        order_by=["created", "order_id"],
    )

    assert frame.columns == ("created", "amount")
    assert frame.index_names == ("order_id",)
    assert frame.ordering == ("created", "order_id")
    assert session.execution_count == 0
    expected = source.sort_values(["created", "order_id"]).set_index("order_id")
    assert_frame_equal(frame.collect(), expected)


def test_index_and_order_survive_projection_filter_and_limit() -> None:
    source = pd.DataFrame(
        {"row_id": [3, 1, 2], "sequence": [30, 10, 20], "value": [8, 6, 7]}
    )
    frame = duckpd.from_pandas(source, order_by="sequence").set_index("row_id")

    result = frame[frame["value"] >= 7][["value"]].limit(2)

    assert result.columns == ("value",)
    assert result.index_names == ("row_id",)
    assert result.ordering == ("sequence",)
    expected = source.sort_values("sequence").set_index("row_id")
    expected = expected.loc[expected["value"] >= 7, ["value"]].head(2)
    assert_frame_equal(result.collect(), expected)
    assert list(result.to_arrow().column_names) == ["value"]


def test_set_index_allows_duplicates_and_reset_restores_column() -> None:
    source = pd.DataFrame({"group": ["a", "a", "b"], "value": [1, 2, 3]})
    frame = duckpd.from_pandas(source).set_index("group")

    indexed = frame.collect()
    restored = frame.reset_index().collect()

    assert indexed.index.tolist() == ["a", "a", "b"]
    assert_frame_equal(restored, source)


def test_reset_index_drop_discards_hidden_index() -> None:
    source = pd.DataFrame({"row_id": [10, 20], "value": [1, 2]})

    result = duckpd.from_pandas(source).set_index("row_id").reset_index(drop=True)

    assert_frame_equal(result.collect(), source[["value"]])


def test_reset_index_drop_clears_ordering_on_removed_index() -> None:
    source = pd.DataFrame({"row_id": [2, 1], "value": [20, 10]})

    result = duckpd.from_pandas(source, index="row_id", order_by="row_id").reset_index(
        drop=True
    )

    assert result.ordering == ()
    expected = source.sort_values("row_id")[["value"]].reset_index(drop=True)
    assert_frame_equal(result.collect(), expected)


def test_set_index_drop_false_matches_pandas_collection() -> None:
    source = pd.DataFrame({"row_id": [10, 20], "value": [1, 2]})

    result = duckpd.from_pandas(source).set_index("row_id", drop=False)

    assert_frame_equal(result.collect(), source.set_index("row_id", drop=False))
    with pytest.raises(ValueError, match="already exists"):
        result.reset_index()


def test_metadata_columns_cannot_be_replaced() -> None:
    source = pd.DataFrame({"row_id": [1], "sequence": [1], "value": [2]})
    frame = duckpd.from_pandas(source, order_by="sequence").set_index("row_id")

    with pytest.raises(ValueError, match="index or ordering column"):
        frame.assign(row_id=2)
    with pytest.raises(ValueError, match="index or ordering column"):
        frame.assign(sequence=2)


def test_metadata_rejects_invalid_inputs_without_execution() -> None:
    source = pd.DataFrame({"row_id": [1], "value": [2]})
    session = duckpd.connect()
    frame = session.from_pandas(source)

    with pytest.raises(KeyError):
        session.from_pandas(source, index="missing")
    with pytest.raises(KeyError):
        session.from_pandas(source, order_by="missing")
    with pytest.raises(ValueError, match="must be unique"):
        session.from_pandas(source, order_by=["value", "value"])
    with pytest.raises(ValueError, match="at least one column"):
        frame.set_index([])
    with pytest.raises(ValueError, match="no explicit index"):
        frame.reset_index()
    assert session.execution_count == 0


def test_every_plan_has_valid_metadata() -> None:
    source = pd.DataFrame({"row_id": [1, 2], "sequence": [2, 1], "value": [3, 4]})
    frame = (
        duckpd.from_pandas(source, order_by="sequence")
        .set_index("row_id")
        .assign(calculated=lambda item: item["value"] + 1)
        .sort_values("calculated")[["calculated"]]
        .limit(1)
    )

    plan: LogicalPlan = frame._plan
    while not isinstance(plan, ScanPlan):
        validate_metadata(plan.metadata)
        assert isinstance(plan, (FilterPlan, ProjectPlan, SortPlan, LimitPlan))
        plan = plan.input
    validate_metadata(plan.metadata)


def test_expression_metadata_tracks_scalar_and_length_semantics() -> None:
    frame = duckpd.from_pandas(pd.DataFrame({"value": [1]}))
    column = ColumnRef(frame._plan.metadata.visible_columns[0].id)
    literal = LiteralValue(1)
    expression = BinaryExpression(column, BinaryOperator.ADD, literal)

    column_metadata = expression_metadata(column)
    literal_metadata = expression_metadata(literal)
    result_metadata = expression_metadata(expression)

    assert column_metadata.preserves_length
    assert not column_metadata.is_scalar_like
    assert literal_metadata.is_literal
    assert literal_metadata.is_scalar_like
    assert result_metadata.is_elementwise
    assert result_metadata.preserves_length
    assert not result_metadata.is_scalar_like


def test_arithmetic_metadata_uses_duckdb_numeric_promotion() -> None:
    source = pd.DataFrame(
        {
            "floating": pd.Series([1.0, 2.0], dtype="float32"),
            "decimal": [Decimal("1.20"), Decimal("2.30")],
        }
    )
    frame = duckpd.from_pandas(source)

    result = frame.assign(
        float_sum=frame["floating"] + frame["floating"],
        decimal_sum=frame["decimal"] + frame["decimal"],
        decimal_product=frame["decimal"] * frame["decimal"],
    )

    dtypes = {
        column.label: column.duckdb_type
        for column in result._plan.metadata.visible_columns
    }
    assert dtypes["float_sum"] == "FLOAT"
    assert dtypes["decimal_sum"] == "DECIMAL(4,2)"
    assert dtypes["decimal_product"] == "DECIMAL(6,4)"


@pytest.mark.parametrize(
    ("left", "right", "operator", "expected"),
    [
        ("FLOAT", "FLOAT", "add", "FLOAT"),
        ("DOUBLE", "FLOAT", "multiply", "DOUBLE"),
        ("DECIMAL(10,2)", "DECIMAL(8,1)", "add", "DECIMAL(11,2)"),
        ("DECIMAL(10,2)", "DECIMAL(8,1)", "multiply", "DECIMAL(18,3)"),
        ("DECIMAL(10,2)", "DECIMAL(8,1)", "modulo", "DECIMAL(10,2)"),
        ("DECIMAL(10,2)", "INTEGER", "add", "DECIMAL(13,2)"),
        ("DECIMAL(10,2)", "INTEGER", "multiply", "DECIMAL(18,2)"),
        ("DECIMAL(10,2)", "INTEGER", "true_divide", "DOUBLE"),
        ("DECIMAL(18,18)", "DECIMAL(18,18)", "multiply", "DECIMAL(36,36)"),
        ("DECIMAL(30,20)", "DECIMAL(30,20)", "multiply", "UNKNOWN"),
        ("INTEGER", "INTEGER", "add", "INTEGER"),
        ("TINYINT", "SMALLINT", "add", "SMALLINT"),
        ("BIGINT", "UBIGINT", "add", "UNKNOWN"),
        ("BOOLEAN", "BOOLEAN", "add", "UNKNOWN"),
        ("DECIMAL(10,2)", "VARCHAR", "add", "UNKNOWN"),
    ],
)
def test_binary_numeric_type_matches_duckdb_promotion(
    left: str,
    right: str,
    operator: str,
    expected: str,
) -> None:
    assert binary_numeric_type(left, right, operator) == expected


def test_decimal_promotion_never_emits_invalid_precision_or_scale() -> None:
    decimal_types = [
        f"DECIMAL({precision},{scale})"
        for precision in (1, 9, 18, 19, 30, 38)
        for scale in {0, precision // 2, precision}
    ]

    for left in decimal_types:
        for right in decimal_types:
            for operator in ("add", "subtract", "multiply", "modulo"):
                result = binary_numeric_type(left, right, operator)
                if not result.startswith("DECIMAL("):
                    assert result == "UNKNOWN"
                    continue
                precision, scale = (
                    int(part)
                    for part in result.removeprefix("DECIMAL(")
                    .removesuffix(")")
                    .split(",")
                )
                assert 0 <= scale <= precision <= 38


def test_global_aggregate_clears_index_and_order_metadata() -> None:
    source = pd.DataFrame({"row_id": [1, 2], "value": [10, 20]})
    frame = duckpd.from_pandas(source, index="row_id", order_by="row_id")
    value = next(
        column
        for column in frame._plan.metadata.visible_columns
        if column.label == "value"
    )

    plan = aggregate_plan(
        frame._plan,
        ((value.label, ColumnRef(value.id), value.duckdb_type),),
        AggregateOperator.SUM,
    )

    validate_metadata(plan.metadata)
    assert plan.metadata.index.columns == ()
    assert plan.metadata.ordering.keys == ()
    assert [column.label for column in plan.metadata.columns] == ["value"]


def test_parquet_source_accepts_index_and_order(tmp_path: Path) -> None:
    source = pd.DataFrame({"row_id": [2, 1], "sequence": [20, 10], "value": [4, 3]})
    path = tmp_path / "source.parquet"
    source.to_parquet(path, index=False)

    result = duckpd.read_parquet(path, index="row_id", order_by="sequence").collect()

    expected = source.sort_values("sequence").set_index("row_id")
    assert_frame_equal(result, expected)
