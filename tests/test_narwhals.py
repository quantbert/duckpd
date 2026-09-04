"""Narwhals plugin and generated compatibility contract tests."""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

import narwhals as nw
import pandas as pd
import pyarrow as pa
import pytest
from narwhals.exceptions import ColumnNotFoundError

import duckpd
from duckpd.frame import DataFrame
from scripts.generate_compatibility import load_matrix, render_matrix


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


def test_narwhals_drop_and_rename_missing_column_contracts() -> None:
    with duckpd.connect() as session:
        lazy = nw.from_native(session.from_pandas(pd.DataFrame({"a": [1], "b": [2]})))

        with pytest.raises(ColumnNotFoundError, match="missing"):
            lazy.drop("missing")

        assert lazy.drop("missing", strict=False).columns == ["a", "b"]
        assert lazy.rename({"missing": "renamed"}).columns == ["a", "b"]
        assert session.execution_count == 0


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
