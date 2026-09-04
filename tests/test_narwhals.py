"""Narwhals plugin and generated compatibility contract tests."""

from __future__ import annotations

import math
from datetime import date, timedelta
from decimal import Decimal
from importlib.metadata import entry_points, version
from pathlib import Path
from typing import Any, cast

import narwhals as nw
import pandas as pd
import pyarrow as pa
import pytest
from narwhals.exceptions import (
    ColumnNotFoundError,
    DuplicateError,
    InvalidOperationError,
    MultiOutputExpressionError,
)

import duckpd
from duckpd.errors import UnsupportedOperationError
from duckpd.frame import DataFrame
from scripts.generate_compatibility import load_matrix, render_matrix


def _collect_once(
    lazy: nw.LazyFrame[DataFrame],
    session: duckpd.Session,
) -> pa.Table:
    assert session.execution_count == 0
    assert isinstance(lazy.to_native(), DataFrame)
    assert session.execution_count == 0
    result = lazy.collect().to_native()
    assert isinstance(result, pa.Table)
    assert session.execution_count == 1
    return result


def test_narwhals_entrypoint_wraps_duckpd_without_execution() -> None:
    with duckpd.connect() as session:
        native = session.from_pandas(
            pd.DataFrame({"key": [3, 1, 2], "value": [30, 10, 20]})
        )

        lazy = nw.from_native(native)
        transformed = lazy.select("key", "value").sort("key").head(2)

        assert session.execution_count == 0
        round_tripped = transformed.to_native()
        assert isinstance(round_tripped, DataFrame)
        assert session.execution_count == 0

        collected = transformed.collect().to_native()
        assert isinstance(collected, pa.Table)
        assert collected.to_pydict() == {"key": [1, 2], "value": [10, 20]}
        assert session.execution_count == 1


def test_narwhals_metadata_operations_remain_lazy() -> None:
    with duckpd.connect() as session:
        native = session.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
        lazy = nw.from_native(native)

        transformed = lazy.rename({"a": "renamed"}).drop("b")

        assert transformed.columns == ["renamed"]
        assert list(transformed.collect_schema()) == ["renamed"]
        assert isinstance(transformed.to_native(), DataFrame)
        assert session.execution_count == 0


def test_narwhals_schema_uses_promoted_arithmetic_metadata() -> None:
    source = pd.DataFrame({"value": pd.Series([1.0, 2.0], dtype="float32")})
    with duckpd.connect() as session:
        result = nw.from_native(session.from_pandas(source)).with_columns(
            (nw.col("value") + nw.col("value")).alias("float_sum")
        )

        assert result.collect_schema()["float_sum"] == nw.Float32
        assert session.execution_count == 0


def test_narwhals_drop_and_rename_missing_column_contracts() -> None:
    with duckpd.connect() as session:
        lazy = nw.from_native(session.from_pandas(pd.DataFrame({"a": [1], "b": [2]})))

        with pytest.raises(ColumnNotFoundError, match="missing"):
            lazy.drop("missing")

        assert lazy.drop("missing", strict=False).columns == ["a", "b"]
        assert lazy.rename({"missing": "renamed"}).columns == ["a", "b"]
        assert session.execution_count == 0


def test_narwhals_expression_pipeline_is_lazy_and_executes_once() -> None:
    source = pd.DataFrame(
        {
            "amount": [1.0, None, 6.0, 4.0],
            "active": [False, True, False, True],
        }
    )
    with duckpd.connect() as session:
        native = session.from_pandas(source)

        transformed = (
            nw.from_native(native)
            .with_columns(
                (nw.col("amount") * nw.lit(2)).alias("gross"),
                nw.col("amount").is_null().alias("missing"),
            )
            .filter((nw.col("gross") > nw.lit(2)) & nw.col("active"))
            .select(
                nw.col("gross").cast(nw.Int64),
                nw.col("missing"),
            )
        )

        assert session.execution_count == 0
        assert isinstance(transformed.to_native(), DataFrame)
        assert session.execution_count == 0

        collected = transformed.collect().to_native()

        assert collected.to_pydict() == {"gross": [8], "missing": [False]}
        assert session.execution_count == 1


def test_narwhals_expressions_support_empty_inputs_and_scalar_broadcast() -> None:
    source = pd.DataFrame({"amount": pd.Series([], dtype="float64")})
    with duckpd.connect() as session:
        transformed = nw.from_native(session.from_pandas(source)).select(
            (nw.col("amount") + nw.lit(3)).alias("adjusted")
        )

        assert session.execution_count == 0
        assert transformed.collect().to_native().to_pydict() == {"adjusted": []}
        assert session.execution_count == 1


def test_narwhals_expression_operator_subset() -> None:
    with duckpd.connect() as session:
        lazy = nw.from_native(session.from_pandas(pd.DataFrame({"a": [5], "b": [2]})))

        collected = lazy.select(
            (nw.col("a") + 2).alias("add"),
            (nw.col("a") - 2).alias("subtract"),
            (10 - nw.col("a")).alias("reverse_subtract"),
            (nw.col("a") * 2).alias("multiply"),
            (nw.col("a") / 2).alias("divide"),
            (10 / nw.col("a")).alias("reverse_divide"),
            (nw.col("a") % 2).alias("modulo"),
            (10 % nw.col("a")).alias("reverse_modulo"),
            (nw.col("a") != nw.col("b")).alias("not_equal"),
            (nw.col("a") <= nw.col("b")).alias("less_equal"),
            (
                ((nw.col("a") >= 5) | (nw.col("b") < 0)) & ~(nw.col("a") == nw.col("b"))
            ).alias("predicate"),
        ).collect()

        assert collected.to_native().to_pydict() == {
            "add": [7],
            "subtract": [3],
            "reverse_subtract": [5],
            "multiply": [10],
            "divide": [2.5],
            "reverse_divide": [2.0],
            "modulo": [1],
            "reverse_modulo": [0],
            "not_equal": [True],
            "less_equal": [False],
            "predicate": [True],
        }


@pytest.mark.skipif(
    tuple(int(part) for part in version("narwhals").split(".")[:2]) < (2, 18),
    reason="Narwhals 2.17 does not expose unary negation on public expressions",
)
def test_narwhals_expression_unary_negation() -> None:
    with duckpd.connect() as session:
        result = (
            nw.from_native(session.from_pandas(pd.DataFrame({"a": [5]})))
            .select((-nw.col("a")).alias("negate"))
            .collect()
            .to_native()
        )

    assert result.to_pydict() == {"negate": [-5]}


def test_narwhals_expression_validation_fails_before_execution() -> None:
    with duckpd.connect() as session:
        lazy = nw.from_native(
            session.from_pandas(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
        )

        with pytest.raises(ValueError, match="one output"):
            lazy.select(nw.col("a", "b").alias("renamed"))
        with pytest.raises(DuplicateError, match="unique output names"):
            lazy.select(nw.col("a"), nw.col("a"))
        with pytest.raises(MultiOutputExpressionError, match="Multi-output"):
            lazy.select(nw.col("a", "b") + nw.col("a", "b", "c"))
        casted = lazy.select(nw.col("a").cast(nw.Datetime("us")))
        assert casted.collect_schema()["a"] == nw.Datetime("us")
        with pytest.raises(UnsupportedOperationError, match="ignore_nulls=True"):
            lazy.select(nw.all_horizontal(nw.col("a"), ignore_nulls=True))

        assert session.execution_count == 0


def test_narwhals_filter_combines_constraints() -> None:
    with duckpd.connect() as session:
        lazy = nw.from_native(
            session.from_pandas(pd.DataFrame({"a": [1, 1, 2], "b": [2, 3, 2]}))
        )

        result = lazy.filter(a=1, b=2).collect().to_native()

        assert result.to_pydict() == {"a": [1], "b": [2]}


def test_narwhals_string_expression_namespace_is_lazy() -> None:
    source = pd.DataFrame({"text": ["  alpha  ", "ßeta", "", None]})
    with duckpd.connect() as session:
        transformed = nw.from_native(session.from_pandas(source)).select(
            nw.col("text").str.to_uppercase().alias("upper"),
            nw.col("text").str.to_lowercase().alias("lower"),
            nw.col("text").str.strip_chars().alias("stripped"),
            nw.col("text").str.len_chars().alias("length"),
            nw.col("text").str.starts_with("  a").alias("starts"),
            nw.col("text").str.ends_with("  ").alias("ends"),
            nw.col("text").str.contains(r"^  a").alias("regex_match"),
            nw.col("text").str.contains("alpha", literal=True).alias("literal_match"),
            nw.col("text").str.replace("a", "X", literal=True).alias("replace_once"),
            nw.col("text").str.replace_all("a", "X", literal=True).alias("replace_all"),
            nw.col("text").str.replace(r"a.", "X").alias("regex_replace"),
        )

        assert session.execution_count == 0
        assert transformed.to_native().columns == (
            "upper",
            "lower",
            "stripped",
            "length",
            "starts",
            "ends",
            "regex_match",
            "literal_match",
            "replace_once",
            "replace_all",
            "regex_replace",
        )
        assert session.execution_count == 0

        result = transformed.collect().to_native().to_pydict()

        assert result == {
            "upper": ["  ALPHA  ", "ẞETA", "", None],
            "lower": ["  alpha  ", "ßeta", "", None],
            "stripped": ["alpha", "ßeta", "", None],
            "length": [9, 4, 0, None],
            "starts": [True, False, False, None],
            "ends": [True, False, False, None],
            "regex_match": [True, False, False, None],
            "literal_match": [True, False, False, None],
            "replace_once": ["  Xlpha  ", "ßetX", "", None],
            "replace_all": ["  XlphX  ", "ßetX", "", None],
            "regex_replace": ["  Xpha  ", "ßeta", "", None],
        }
        assert session.execution_count == 1


def test_narwhals_datetime_expression_namespace_is_lazy() -> None:
    source = pd.DataFrame(
        {
            "ts": pd.Series(
                [pd.Timestamp("2024-02-03 04:05:06"), pd.NaT],
                dtype="datetime64[ns]",
            )
        }
    )
    with duckpd.connect() as session:
        transformed = nw.from_native(session.from_pandas(source)).select(
            nw.col("ts").dt.year().alias("year"),
            nw.col("ts").dt.month().alias("month"),
            nw.col("ts").dt.day().alias("day"),
            nw.col("ts").dt.hour().alias("hour"),
            nw.col("ts").dt.minute().alias("minute"),
            nw.col("ts").dt.second().alias("second"),
            nw.col("ts").dt.date().alias("date"),
            nw.col("ts").dt.to_string("%Y-%m-%d %H:%M:%S").alias("formatted"),
        )

        assert session.execution_count == 0
        result = transformed.collect().to_native().to_pydict()

        assert result == {
            "year": [2024, None],
            "month": [2, None],
            "day": [3, None],
            "hour": [4, None],
            "minute": [5, None],
            "second": [6, None],
            "date": [date(2024, 2, 3), None],
            "formatted": ["2024-02-03 04:05:06", None],
        }
        assert session.execution_count == 1


def test_narwhals_accessor_validation_fails_before_execution() -> None:
    with duckpd.connect() as session:
        lazy = nw.from_native(session.from_pandas(pd.DataFrame({"number": [1]})))

        with pytest.raises(InvalidOperationError, match="VARCHAR"):
            lazy.select(nw.col("number").str.to_uppercase())
        with pytest.raises(InvalidOperationError, match="temporal"):
            lazy.select(nw.col("number").dt.year())
        with pytest.raises(UnsupportedOperationError, match="only n=1"):
            lazy.select(nw.col("number").cast(nw.String).str.replace("1", "x", n=2))

        assert session.execution_count == 0


def test_narwhals_accessors_support_empty_inputs() -> None:
    source = pd.DataFrame(
        {
            "text": pd.Series([], dtype="string"),
            "ts": pd.Series([], dtype="datetime64[ns]"),
        }
    )
    with duckpd.connect() as session:
        transformed = nw.from_native(session.from_pandas(source)).select(
            nw.col("text").str.to_uppercase().alias("upper"),
            nw.col("ts").dt.year().alias("year"),
        )

        assert session.execution_count == 0
        assert transformed.collect().to_native().to_pydict() == {
            "upper": [],
            "year": [],
        }


def test_narwhals_expression_missing_columns_fail_before_execution() -> None:
    with duckpd.connect() as session:
        lazy = nw.from_native(session.from_pandas(pd.DataFrame({"a": [1]})))

        with pytest.raises(ColumnNotFoundError, match="missing"):
            lazy.select(nw.col("missing"))

        assert session.execution_count == 0


def test_narwhals_global_aggregations_are_lazy_and_execute_once() -> None:
    source = pd.DataFrame({"value": [1.0, None, 3.0]})
    with duckpd.connect() as session:
        lazy = nw.from_native(session.from_pandas(source))

        aggregated = lazy.select(
            nw.col("value").sum().alias("total"),
            nw.col("value").mean().alias("average"),
            nw.col("value").count().alias("non_null"),
            nw.col("value").len().alias("rows"),
            nw.col("value").min().alias("minimum"),
            nw.col("value").max().alias("maximum"),
            nw.col("value").median().alias("median"),
            nw.col("value").std(ddof=1).alias("sample_std"),
            nw.col("value").var(ddof=1).alias("sample_var"),
        )

        assert session.execution_count == 0
        assert aggregated.collect().to_native().to_pydict() == {
            "total": [4.0],
            "average": [2.0],
            "non_null": [2],
            "rows": [3],
            "minimum": [1.0],
            "maximum": [3.0],
            "median": [2.0],
            "sample_std": [2**0.5],
            "sample_var": [2.0],
        }
        assert session.execution_count == 1


def test_narwhals_lazy_group_by_aggregates_without_collection() -> None:
    source = pd.DataFrame(
        {
            "group": ["a", "a", "b", None],
            "value": [1.0, 2.0, 4.0, 8.0],
        }
    )
    with duckpd.connect() as session:
        lazy = nw.from_native(session.from_pandas(source))

        aggregated = (
            lazy.group_by("group", drop_null_keys=True)
            .agg(
                nw.col("value").sum().alias("total"),
                nw.col("value").count().alias("count"),
            )
            .sort("group")
        )

        assert session.execution_count == 0
        assert aggregated.collect().to_native().to_pydict() == {
            "group": ["a", "b"],
            "total": [3.0, 4.0],
            "count": [2, 1],
        }
        assert session.execution_count == 1


def test_narwhals_group_by_accepts_expression_keys() -> None:
    source = pd.DataFrame({"group": ["a", "A", "b"], "value": [1.0, 2.0, 4.0]})
    with duckpd.connect() as session:
        result = (
            nw.from_native(session.from_pandas(source))
            .group_by(nw.col("group").str.to_uppercase().alias("normalized"))
            .agg(nw.col("value").sum().alias("total"))
            .sort("normalized")
        )

        assert session.execution_count == 0
        assert result.collect().to_native().to_pydict() == {
            "normalized": ["A", "B"],
            "total": [3.0, 4.0],
        }
        assert session.execution_count == 1


def test_narwhals_group_by_honors_population_ddof() -> None:
    source = pd.DataFrame({"group": ["a", "a", "b"], "value": [1.0, 2.0, 4.0]})
    with duckpd.connect() as session:
        result = (
            nw.from_native(session.from_pandas(source))
            .group_by("group")
            .agg(nw.col("value").std(ddof=0).alias("population_std"))
            .sort("group")
        )

        assert session.execution_count == 0
        assert result.collect().to_native().to_pydict() == {
            "group": ["a", "b"],
            "population_std": [0.5, 0.0],
        }
        assert session.execution_count == 1


def test_narwhals_aggregate_validation_fails_before_execution() -> None:
    with duckpd.connect() as session:
        lazy = nw.from_native(
            session.from_pandas(pd.DataFrame({"text": ["a", "b"], "value": [1, 2]}))
        )

        with pytest.raises(
            UnsupportedOperationError,
            match="supports only numeric data",
        ):
            lazy.select(nw.col("text").sum())
        assert session.execution_count == 0
        with pytest.raises(
            UnsupportedOperationError,
            match="does not yet broadcast aggregate expressions",
        ):
            lazy.with_columns(nw.col("value").sum().alias("total"))
        with pytest.raises(
            UnsupportedOperationError,
            match="does not yet broadcast aggregate expressions",
        ):
            lazy.select("value", nw.col("value").sum().alias("total"))
        assert session.execution_count == 0


def test_narwhals_numeric_expression_surface_is_lazy() -> None:
    source = pd.DataFrame({"x": [-2.5, 4.0]})
    with duckpd.connect() as session:
        lazy = nw.from_native(session.from_pandas(source)).select(
            nw.col("x").abs().alias("absolute"),
            nw.col("x").round().alias("rounded"),
            nw.col("x").floor().alias("floor"),
            nw.col("x").ceil().alias("ceil"),
            nw.col("x").clip(-1, 3).alias("clipped"),
            (nw.col("x") // 2).alias("floor_divide"),
            (nw.col("x") ** 2).alias("power"),
            nw.col("x").exp().alias("exp"),
            nw.col("x").abs().sqrt().alias("sqrt"),
            nw.col("x").abs().log(base=2).alias("log2"),
            nw.col("x").cos().alias("cos"),
            nw.col("x").sin().alias("sin"),
        )

        result = _collect_once(lazy, session).to_pydict()

    assert result["absolute"] == [2.5, 4.0]
    assert result["rounded"] == [-3.0, 4.0]
    assert result["floor"] == [-3.0, 4.0]
    assert result["ceil"] == [-2.0, 4.0]
    assert result["clipped"] == [-1.0, 3.0]
    assert result["floor_divide"] == [-2.0, 2.0]
    assert result["power"] == [6.25, 16.0]
    assert result["exp"] == pytest.approx([math.exp(-2.5), math.exp(4.0)])
    assert result["sqrt"] == pytest.approx([math.sqrt(2.5), 2.0])
    assert result["log2"] == pytest.approx([math.log2(2.5), 2.0])
    assert result["cos"] == pytest.approx([math.cos(-2.5), math.cos(4.0)])
    assert result["sin"] == pytest.approx([math.sin(-2.5), math.sin(4.0)])


def test_narwhals_ordered_expression_surface() -> None:
    source = pd.DataFrame({"pos": [0, 1, 2, 3], "x": [3.0, 1.0, 2.0, 4.0]})
    with duckpd.connect() as session:
        lazy = nw.from_native(session.from_pandas(source, order_by="pos")).select(
            nw.col("x").cum_sum().over(order_by="pos").alias("cumulative"),
            nw.col("x").cum_min().over(order_by="pos").alias("minimum"),
            nw.col("x").cum_max().over(order_by="pos").alias("maximum"),
            nw.col("x").cum_prod().over(order_by="pos").alias("product"),
            nw.col("x").cum_count().over(order_by="pos").alias("count"),
            nw.col("x").diff().over(order_by="pos").alias("difference"),
            nw.col("x").shift(1).over(order_by="pos").alias("shifted"),
            nw.col("x").rank(method="ordinal").over(order_by="pos").alias("rank"),
            nw.col("x")
            .rolling_sum(2, min_samples=1)
            .over(order_by="pos")
            .alias("rolling"),
        )

        result = _collect_once(lazy, session).to_pydict()

    assert result == {
        "cumulative": [3.0, 4.0, 6.0, 10.0],
        "minimum": [3.0, 1.0, 1.0, 1.0],
        "maximum": [3.0, 3.0, 3.0, 4.0],
        "product": [3.0, 3.0, 6.0, 24.0],
        "count": [1, 2, 3, 4],
        "difference": [None, -2.0, 1.0, 2.0],
        "shifted": [None, 3.0, 1.0, 2.0],
        "rank": [3.0, 1.0, 2.0, 4.0],
        "rolling": [3.0, 4.0, 3.0, 6.0],
    }


def test_narwhals_ordered_expression_rejections_are_pre_execution() -> None:
    with duckpd.connect() as session:
        lazy = nw.from_native(
            session.from_pandas(pd.DataFrame({"x": [1, 2]}), order_by="x")
        )

        with pytest.raises(UnsupportedOperationError, match="reverse=True"):
            lazy.select(nw.col("x").cum_sum(reverse=True))
        with pytest.raises(ValueError, match="log base"):
            lazy.select(nw.col("x").log(base=1))

        assert session.execution_count == 0


def test_narwhals_relational_methods_are_lazy() -> None:
    source = pd.DataFrame(
        {"pos": [0, 1, 2, 3], "key": ["a", "a", "b", None], "value": [2, 1, 4, 3]}
    )
    with duckpd.connect() as session:
        lazy = nw.from_native(session.from_pandas(source, order_by="pos"))

        transformed = (
            lazy.drop_nulls(subset=["key"])
            .unique(subset=["key"], keep="first", order_by=["pos"])
            .top_k(2, by="value")
            .with_row_index("row", order_by=["value"])
        )

        result = _collect_once(transformed, session).to_pydict()

    assert result == {
        "row": [0, 1],
        "pos": [0, 2],
        "key": ["a", "b"],
        "value": [2, 4],
    }


def test_narwhals_join_mappings_are_lazy() -> None:
    with duckpd.connect() as session:
        left = nw.from_native(
            session.from_pandas(pd.DataFrame({"key": [1, 2], "left": ["a", "b"]}))
        )
        right = nw.from_native(
            session.from_pandas(pd.DataFrame({"key": [2, 3], "right": ["c", "d"]}))
        )

        joined = left.join(right, on="key", how="full").sort("key")
        result = _collect_once(joined, session).to_pydict()

    assert result == {
        "key": [1, 2, 3],
        "left": ["a", "b", None],
        "right": [None, "c", "d"],
    }


def test_narwhals_intentional_relational_exclusions_fail_early() -> None:
    with duckpd.connect() as session:
        lazy = nw.from_native(session.from_pandas(pd.DataFrame({"x": [1, 2]})))

        with pytest.raises(UnsupportedOperationError, match="nested dtype"):
            lazy.explode("x")
        with pytest.raises(UnsupportedOperationError, match="typed plan"):
            lazy.unpivot(on=["x"])
        with pytest.raises(UnsupportedOperationError, match="as-of"):
            lazy.join_asof(lazy, on="x")

        assert session.execution_count == 0


def test_narwhals_scalar_schema_conversion() -> None:
    table = pa.table(
        {
            "decimal": pa.array([Decimal("1.25")], type=pa.decimal128(8, 2)),
            "timestamp": pa.array(
                [pd.Timestamp("2024-01-01", tz="UTC")],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "duration": pa.array([timedelta(seconds=1)], type=pa.duration("us")),
        }
    )
    with duckpd.connect() as session:
        lazy = nw.from_native(session.from_arrow(table))

        assert lazy.collect_schema() == nw.Schema(
            {
                "decimal": nw.Decimal(8, 2),
                "timestamp": nw.Datetime("us", "UTC"),
                "duration": nw.Duration("us"),
            }
        )
        assert session.execution_count == 0


def test_narwhals_public_scan_is_explicitly_unsupported(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.parquet"
    pd.DataFrame({"x": [2, 1]}).to_parquet(source_path)

    with pytest.raises(AssertionError, match="UNKNOWN Implementation"):
        nw.scan_parquet(source_path, backend=cast("Any", "duckpd"))


def test_narwhals_sink_and_collection_backends(tmp_path: Path) -> None:
    source_path = tmp_path / "source.parquet"
    sink_path = tmp_path / "sink.parquet"
    pd.DataFrame({"x": [2, 1]}).to_parquet(source_path)

    with duckpd.connect() as session:
        lazy = nw.from_native(session.read_parquet(source_path)).sort("x")
        pandas_result = lazy.collect(backend="pandas").to_native()
        assert pandas_result["x"].tolist() == [1, 2]
        assert session.execution_count == 1

        lazy.sink_parquet(sink_path)
        assert session.execution_count == 2

    assert pd.read_parquet(sink_path)["x"].tolist() == [1, 2]


def test_narwhals_plugin_entrypoint_is_packaged() -> None:
    matches = [
        entrypoint
        for entrypoint in entry_points(group="narwhals.plugins")
        if entrypoint.name == "duckpd"
    ]
    assert len(matches) == 1
    assert matches[0].value == "duckpd._narwhals_plugin"


def test_generated_narwhals_compatibility_is_current() -> None:
    root = Path(__file__).resolve().parents[1]
    matrix = load_matrix(root / "docs" / "narwhals-compatibility.json")
    generated = (root / "docs" / "NARWHALS_COMPATIBILITY.md").read_text(
        encoding="utf-8"
    )

    assert generated == render_matrix(matrix)
