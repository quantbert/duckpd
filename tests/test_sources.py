from __future__ import annotations

from datetime import date
from decimal import Decimal
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


def test_from_pandas_preserves_nullable_unsigned_integers() -> None:
    source = pd.DataFrame({"value": pd.Series([2**63 + 1, None], dtype="UInt64")})

    result = duckpd.from_pandas(source).collect()

    assert_frame_equal(result, source)


def test_from_pandas_preserves_supported_nullable_and_temporal_dtypes() -> None:
    source = pd.DataFrame(
        {
            "integer": pd.Series([1, None], dtype="Int16"),
            "unsigned": pd.Series([1, None], dtype="UInt32"),
            "boolean": pd.Series([True, None], dtype="boolean"),
            "string": pd.Series(["value", None], dtype="string"),
            "timestamp": pd.Series(
                [pd.Timestamp("2025-01-01"), pd.NaT], dtype="datetime64[ns]"
            ),
            "zoned": pd.Series([pd.Timestamp("2025-01-01", tz="UTC"), pd.NaT]),
            "duration": pd.Series([pd.Timedelta(days=1), pd.NaT]),
            "date": pd.Series([date(2025, 1, 1), None], dtype=object),
            "decimal": pd.Series([Decimal("1.20"), None], dtype=object),
            "binary": pd.Series([b"value", None], dtype=object),
        }
    )

    result = duckpd.from_pandas(source).collect()

    assert_frame_equal(result, source)


def test_nested_source_types_fail_before_execution() -> None:
    session = duckpd.connect()

    with pytest.raises(
        UnsupportedOperationError,
        match=r"nested DuckDB type.*INTEGER\[\]",
    ):
        session.from_pandas(pd.DataFrame({"nested": [[1, 2], None]}))

    assert session.execution_count == 0


def test_sql_null_collection_policy_is_explicit_by_dtype_family() -> None:
    result = (
        duckpd.connect()
        .sql(
            """
        SELECT
            NULL::BOOLEAN AS boolean_value,
            NULL::BIGINT AS integer_value,
            NULL::DOUBLE AS float_value,
            NULL::VARCHAR AS string_value,
            NULL::TIMESTAMP AS timestamp_value,
            NULL::TIMESTAMPTZ AS zoned_value,
            NULL::INTERVAL AS duration_value,
            NULL::BLOB AS binary_value,
            NULL::DATE AS date_value,
            NULL::DECIMAL(10, 2) AS decimal_value
        """
        )
        .collect()
    )

    assert result.dtypes.astype(str).to_dict() == {
        "boolean_value": "boolean",
        "integer_value": "float64",
        "float_value": "float64",
        "string_value": "object",
        "timestamp_value": "datetime64[us]",
        "zoned_value": "datetime64[us, Etc/UTC]",
        "duration_value": "timedelta64[us]",
        "binary_value": "object",
        "date_value": "object",
        "decimal_value": "object",
    }
    assert result.isna().all().all()


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


def test_module_helpers_share_a_context_local_session() -> None:
    left = duckpd.from_pandas(pd.DataFrame({"value": [1]}))
    right = duckpd.from_pandas(pd.DataFrame({"value": [2]}))

    result = duckpd.concat([left, right])

    assert left._session is right._session
    assert_frame_equal(result.collect(), pd.DataFrame({"value": [1, 2]}))


def test_explicit_session_override_remains_isolated() -> None:
    implicit = duckpd.from_pandas(pd.DataFrame({"value": [1]}))
    explicit_session = duckpd.connect()
    explicit = duckpd.from_pandas(
        pd.DataFrame({"value": [2]}), session=explicit_session
    )

    assert implicit._session is not explicit._session


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
