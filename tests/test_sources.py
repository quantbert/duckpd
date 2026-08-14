from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal

import duckpd
from duckpd.errors import SessionClosedError, UnsupportedOperationError


def test_from_pandas_is_lazy_until_collect() -> None:
    source = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})
    session = duckpd.connect()

    frame = session.from_pandas(source)

    assert frame.columns == ("value", "label")
    assert session.execution_count == 0
    assert_frame_equal(frame.collect(), source)
    assert session.execution_count == 1


def test_repr_does_not_execute() -> None:
    session = duckpd.connect()
    frame = session.from_pandas(pd.DataFrame({"value": [1]}))

    representation = repr(frame)

    assert "Columns: ['value']" in representation
    assert session.execution_count == 0


def test_read_parquet_is_lazy_until_collect(tmp_path: Path) -> None:
    source = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})
    path = tmp_path / "source.parquet"
    source.to_parquet(path, index=False)
    session = duckpd.connect()

    frame = session.read_parquet(path)

    assert session.execution_count == 0
    assert_frame_equal(frame.collect(), source)


def test_module_helper_owns_an_implicit_session() -> None:
    source = pd.DataFrame({"value": [1]})

    frame = duckpd.from_pandas(source)

    assert_frame_equal(frame.to_pandas(), source)


def test_closed_session_fails_clearly() -> None:
    session = duckpd.connect()
    frame = session.from_pandas(pd.DataFrame({"value": [1]}))
    session.close()

    with pytest.raises(SessionClosedError, match="session is closed"):
        frame.collect()


def test_session_context_manager_closes_connection() -> None:
    with duckpd.connect() as session:
        assert not session.closed

    assert session.closed
    session.close()


def test_invalid_source_labels_are_rejected() -> None:
    session = duckpd.connect()
    duplicate = pd.DataFrame([[1, 2]], columns=["value", "value"])
    non_string = pd.DataFrame([[1]], columns=[1])

    with pytest.raises(ValueError, match="duplicate column labels"):
        session.from_pandas(duplicate)
    with pytest.raises(TypeError, match="string column labels"):
        session.from_pandas(non_string)


def test_empty_parquet_path_list_is_rejected() -> None:
    session = duckpd.connect()

    with pytest.raises(ValueError, match="At least one Parquet path"):
        session.read_parquet([])


def test_sql_source_is_lazy_and_read_only() -> None:
    session = duckpd.connect()

    frame = session.sql("select 1 as value union all select 2")

    assert session.execution_count == 0
    assert frame.collect()["value"].tolist() == [1, 2]
    with pytest.raises(UnsupportedOperationError, match="read-only SELECT"):
        session.sql("create table unsafe as select 1")
    with pytest.raises(UnsupportedOperationError, match="exactly one SELECT"):
        session.sql("select 1; select 2")


def test_table_source_is_lazy(tmp_path: Path) -> None:
    database = tmp_path / "source.duckdb"
    with duckdb.connect(database) as setup:
        setup.execute("create table items as select 1 as value")

    session = duckpd.connect(database, read_only=True)
    frame = session.table("items")

    assert session.execution_count == 0
    expected = pd.DataFrame({"value": pd.Series([1], dtype="int32")})
    assert_frame_equal(frame.collect(), expected)


def test_module_arrow_helper_is_lazy() -> None:
    frame = duckpd.from_arrow(pa.table({"value": [1, 2]}))

    assert_frame_equal(frame.collect(), pd.DataFrame({"value": [1, 2]}))


def test_session_resource_configuration(tmp_path: Path) -> None:
    session = duckpd.connect(
        memory_limit="64MB",
        temp_directory=tmp_path / "spill",
        max_temp_directory_size="128MB",
        threads=1,
    )

    settings = session.sql(
        """
        select
            current_setting('threads') as threads,
            current_setting('temp_directory') as temp_directory
        """
    ).collect()

    assert settings.loc[0, "threads"] == 1
    assert settings.loc[0, "temp_directory"] == str(tmp_path / "spill")
