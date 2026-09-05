from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict, cast
from uuid import uuid4

import duckdb
import pandas as pd
import pytest

import duckpd
from duckpd._logical import (
    ParquetSource,
    RemoteTableSource,
    ScanPlan,
    SortPlan,
    TopKPlan,
)
from duckpd._quoting import quote_identifier, quote_literal
from duckpd.errors import (
    MaterializationError,
    RemoteAttachmentError,
    RemoteScanWarning,
)


class _FakeRelation:
    def __init__(
        self,
        columns: Sequence[str] = ("id", "amount"),
        types: Sequence[str] = ("INTEGER", "DOUBLE"),
    ) -> None:
        self.columns = list(columns)
        self.types = list(types)

    def sql_query(self) -> str:
        return "SELECT * FROM remote_source"

    def project(self, *_expressions: object) -> _FakeRelation:
        return self


class _FakeConnection:
    def __init__(
        self,
        *,
        fail_attach: bool = False,
        fail_inspect: bool = False,
        fail_prefix: str | None = None,
    ) -> None:
        self.queries: list[tuple[str, tuple[object, ...]]] = []
        self.installed: list[str] = []
        self.loaded: list[str] = []
        self.closed = False
        self._fail_attach = fail_attach
        self._fail_inspect = fail_inspect
        self._fail_prefix = fail_prefix

    def install_extension(self, name: str) -> None:
        self.installed.append(name)

    def load_extension(self, name: str) -> None:
        self.loaded.append(name)

    def execute(
        self,
        query: str,
        parameters: Sequence[object] | None = None,
    ) -> _FakeConnection:
        values = tuple(parameters or ())
        self.queries.append((query, values))
        if self._fail_attach and query.startswith("ATTACH"):
            raise duckdb.IOException("connection exposed user and super-secret")
        if self._fail_prefix is not None and query.startswith(self._fail_prefix):
            raise duckdb.IOException("operation exposed reader and super-secret")
        return self

    def fetchone(self) -> tuple[str, str]:
        return ("analyzed_plan", "TABLE_SCAN")

    def sql(self, _query: str) -> _FakeRelation:
        if self._fail_inspect:
            raise duckdb.IOException("connection exposed reader and super-secret")
        return _FakeRelation()

    def read_parquet(self, _paths: str | list[str], **_kwargs: object) -> _FakeRelation:
        if self._fail_inspect:
            raise duckdb.IOException("source exposed reader and super-secret")
        return _FakeRelation()

    def from_df(self, value: pd.DataFrame) -> _FakeRelation:
        return _FakeRelation(
            [str(column) for column in value.columns],
            ["BIGINT"] * len(value.columns),
        )

    def create_function(self, *_args: object, **_kwargs: object) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _session_with_fake(
    *,
    fail_attach: bool = False,
    fail_inspect: bool = False,
    fail_prefix: str | None = None,
) -> tuple[duckpd.Session, _FakeConnection]:
    session = duckpd.connect()
    session._connection.close()
    connection = _FakeConnection(
        fail_attach=fail_attach,
        fail_inspect=fail_inspect,
        fail_prefix=fail_prefix,
    )
    session._connection = cast("Any", connection)
    return session, connection


def test_structured_postgres_attachment_uses_secret_and_read_only() -> None:
    session, connection = _session_with_fake()

    attachment = session.attach_postgres(
        "analytics",
        host="db.internal",
        port=5433,
        database="warehouse",
        user="reader",
        password="super-secret",
        schema="reporting",
        unbounded_scan="error",
    )

    assert attachment.engine == "postgres"
    assert "super-secret" not in repr(attachment)
    assert connection.installed == ["postgres"]
    assert connection.loaded == ["postgres"]
    secret_query, secret_parameters = connection.queries[0]
    assert secret_query.startswith("CREATE TEMPORARY SECRET")
    assert "super-secret" not in secret_query
    assert secret_parameters == (
        "db.internal",
        5433,
        "warehouse",
        "reader",
        "super-secret",
    )
    attach_query, attach_parameters = connection.queries[1]
    assert "TYPE postgres" in attach_query
    assert "READ_ONLY" in attach_query
    assert "SCHEMA 'reporting'" in attach_query
    assert attach_parameters == ()

    frame = attachment.table("orders", order_by="id")
    assert session.execution_count == 0
    assert isinstance(frame._plan, SortPlan)
    assert isinstance(frame._plan.input, ScanPlan)
    assert isinstance(frame._plan.input.source, RemoteTableSource)
    explanation = frame.explain("logical")
    assert "analytics.reporting.orders" in explanation
    assert '"projection": true' in explanation
    assert '"filter": true' in explanation
    assert "reader" not in explanation
    assert "super-secret" not in explanation
    data = json.loads(frame.explain("json"))
    remote = data["execution_boundaries"]["remote"]
    assert remote == [
        {
            "kind": "remote_table",
            "engine": "postgres",
            "source": "analytics.reporting.orders",
            "location": "postgres://db.internal:5433/warehouse",
            "estimated_transfer_bytes": None,
            "unbounded_scan": "error",
            "pushdown": {
                "projection": True,
                "filter": True,
                "aggregation": False,
                "join": False,
                "window": False,
                "limit": False,
                "sort": False,
            },
        }
    ]
    with pytest.raises(ValueError, match="already in use"):
        session.attach_postgres("analytics", secret="managed")

    with pytest.raises(MaterializationError, match="no proven transfer bound"):
        frame.collect()
    with pytest.raises(MaterializationError, match="no proven transfer bound"):
        frame.explain("analyze")
    assert session.execution_count == 0
    warn_frame = attachment.table("orders", unbounded_scan="warn")
    with pytest.warns(RemoteScanWarning, match="no proven transfer bound"):
        analyzed = warn_frame.explain("analyze")
    assert "TABLE_SCAN" in analyzed
    assert session.execution_count == 1
    topk = frame.nlargest(1, "amount")
    with pytest.raises(MaterializationError, match="no proven transfer bound"):
        topk.collect()
    assert session.execution_count == 1
    optimized_topk = session._compiler.optimize(topk._plan).plan
    assert isinstance(optimized_topk, TopKPlan)
    with pytest.raises(MaterializationError, match="no proven transfer bound"):
        session._executor._validate_execution(optimized_topk)

    attachment.refresh_schema()
    assert connection.queries[-1][0] == "CALL pg_clear_cache()"
    attachment.detach()
    assert attachment.closed
    assert any(query.startswith("DETACH") for query, _ in connection.queries)
    assert any(query.startswith("DROP SECRET") for query, _ in connection.queries)
    with pytest.raises(RemoteAttachmentError, match="not available"):
        attachment.refresh_schema()


def test_mysql_attachment_accepts_existing_secret_and_session_table() -> None:
    session, connection = _session_with_fake()

    attachment = session.attach_mysql(
        "sales",
        secret="managed_mysql",
        unbounded_scan="allow",
    )
    frame = session.table("sales.orders")

    assert attachment.engine == "mysql"
    assert isinstance(frame._plan, ScanPlan)
    assert isinstance(frame._plan.source, RemoteTableSource)
    source = frame._plan.source
    assert source.capabilities.projection
    assert source.capabilities.filter
    assert not any(query.startswith("CREATE") for query, _ in connection.queries)
    assert 'SECRET "managed_mysql"' in connection.queries[0][0]

    attachment.refresh_schema()
    assert connection.queries[-1][0] == "CALL mysql_clear_cache()"
    warn_frame = attachment.table("orders", unbounded_scan="warn")
    with pytest.warns(RemoteScanWarning, match="no proven transfer bound"):
        session._executor._validate_execution(warn_frame._plan)
    session._executor._validate_execution(frame._plan)
    session.close()
    assert connection.closed
    assert attachment.closed
    assert not any(query.startswith("DROP SECRET") for query, _ in connection.queries)


def test_attachment_validation_and_failure_redact_credentials() -> None:
    session, _ = _session_with_fake()
    with pytest.raises(ValueError, match="attachment alias"):
        session.attach_postgres(
            "bad.alias",
            host="host",
            database="db",
            user="user",
            password="secret",
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        session.attach_mysql("db", secret="managed", host="host")
    with pytest.raises(ValueError, match="unbounded_scan"):
        session.attach_mysql(
            "policy", secret="managed", unbounded_scan=cast("Any", "no")
        )
    with pytest.raises(ValueError, match="schema must be non-empty"):
        session.attach_postgres("schema", secret="managed", schema="")
    with pytest.raises(ValueError, match="parameters are missing"):
        session.attach_postgres("missing", host="host")
    with pytest.raises(ValueError, match="must be non-empty"):
        session.attach_mysql(
            "empty",
            host="host",
            database="db",
            user="user",
            password="",
        )
    with pytest.raises(ValueError, match="port must be"):
        session.attach_mysql(
            "port",
            host="host",
            database="db",
            user="user",
            password="secret",
            port=cast("Any", True),
        )

    failed_session, failed_connection = _session_with_fake(fail_attach=True)
    with pytest.raises(RemoteAttachmentError) as captured:
        failed_session.attach_postgres(
            "analytics",
            host="db.internal",
            database="warehouse",
            user="reader",
            password="super-secret",
        )
    message = str(captured.value)
    assert "reader" not in message
    assert "super-secret" not in message
    assert any(
        query.startswith("DROP SECRET") for query, _ in failed_connection.queries
    )


def test_remote_fragment_and_cross_source_movement_are_explicit() -> None:
    session, _ = _session_with_fake()
    attachment = session.attach_postgres(
        "analytics", secret="managed", unbounded_scan="allow"
    )
    remote = attachment.table("orders")
    session.register_arrow_udf(
        "adjust",
        cast("Any", abs),
        ["DOUBLE"],
        "DOUBLE",
    )
    computed = remote.assign(adjusted=remote["amount"].map_arrow("adjust"))
    computed_fragment = json.loads(computed.explain("json"))["execution_boundaries"][
        "source_fragments"
    ][0]
    assert "projection" not in computed_fragment["pushdown_candidates"]
    assert "projection" in computed_fragment["local_required"]
    planned = remote[remote["amount"] > 0][["id"]].limit(5)

    fragment = json.loads(planned.explain("json"))["execution_boundaries"][
        "source_fragments"
    ][0]
    assert fragment["requested"] == ["projection", "filter", "limit"]
    assert fragment["pushdown_candidates"] == ["projection", "filter"]
    assert fragment["local_required"] == ["limit"]

    local = session.from_pandas(pd.DataFrame({"id": [1], "amount": [2.0]}))
    joined = remote.merge(local, on="id")
    movement = json.loads(joined.explain("json"))["execution_boundaries"]["movement"]
    assert movement[0]["kind"] == "cross_source_join"
    assert movement[0]["strategy"] == "stream_inputs_to_duckdb"
    assert movement[0]["materializes_in_python"] is False


def test_remote_schema_inspection_failure_redacts_credentials() -> None:
    session, _ = _session_with_fake(fail_inspect=True)
    attachment = session.attach_postgres("analytics", secret="managed")

    with pytest.raises(RemoteAttachmentError) as captured:
        attachment.table("orders")

    message = str(captured.value)
    assert "reader" not in message
    assert "super-secret" not in message


def test_object_store_secrets_and_remote_parquet_are_credential_safe() -> None:
    session, connection = _session_with_fake()

    secret = session.create_s3_secret(
        "warehouse",
        key_id="reader",
        secret="super-secret",
        region="us-east-1",
        scope="s3://analytics/reports",
    )
    query, parameters = connection.queries[-1]
    assert "reader" not in query
    assert "super-secret" not in query
    assert parameters == (
        "reader",
        "super-secret",
        "us-east-1",
        "s3://analytics/reports",
    )
    assert "super-secret" not in repr(secret)

    frame = session.read_parquet("s3://analytics/reports/orders.parquet")
    assert isinstance(frame._plan, ScanPlan)
    assert isinstance(frame._plan.source, ParquetSource)
    assert connection.installed[-1] == "httpfs"
    assert "super-secret" not in frame.explain("json")
    gcs = session.create_gcs_secret(
        "lake", key_id="gcs-reader", secret="gcs-secret", scope="gcs://lake/"
    )
    assert gcs.provider == "gcs"
    gcs.drop()
    assert gcs.closed
    with pytest.raises(RemoteAttachmentError, match="not available"):
        gcs.drop()

    secret.drop()
    assert secret.closed


def test_object_store_secret_validation_and_failures_are_redacted() -> None:
    session, connection = _session_with_fake()

    chain = session.create_s3_secret(
        "chain",
        credential_chain=True,
        region="us-west-2",
        endpoint="s3.us-west-2.amazonaws.com",
    )
    assert chain.provider == "s3"
    assert connection.installed[-1] == "aws"
    assert connection.queries[-1][1] == (
        "credential_chain",
        "us-west-2",
        "s3.us-west-2.amazonaws.com",
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        session.create_s3_secret(
            "mixed", key_id="key", secret="secret", credential_chain=True
        )
    with pytest.raises(ValueError, match="both be non-empty"):
        session.create_s3_secret("missing", key_id="key")
    with pytest.raises(ValueError, match="both be non-empty"):
        session.create_gcs_secret("gcs_missing", key_id="", secret="secret")
    with pytest.raises(ValueError, match="secret name"):
        session.create_s3_secret("bad.name", key_id="key", secret="secret")
    with pytest.raises(ValueError, match="already in use"):
        session.create_s3_secret("chain", credential_chain=True)
    with pytest.raises(ValueError, match="credential-free"):
        session.create_gcs_secret(
            "bad_scope",
            key_id="key",
            secret="secret",
            scope="s3://wrong-provider/path",
        )

    failed, _ = _session_with_fake(fail_prefix="CREATE TEMPORARY SECRET")
    with pytest.raises(RemoteAttachmentError) as captured:
        failed.create_s3_secret("failure", key_id="reader", secret="super-secret")
    assert "reader" not in str(captured.value)
    assert "super-secret" not in str(captured.value)

    drop_failed, _ = _session_with_fake(fail_prefix="DROP SECRET")
    owned = drop_failed.create_gcs_secret("owned", key_id="key", secret="secret")
    with pytest.raises(RemoteAttachmentError, match="Failed to drop"):
        owned.drop()

    session.close()
    assert chain.closed


def test_remote_parquet_inspection_failure_is_redacted() -> None:
    session, _ = _session_with_fake(fail_inspect=True)

    with pytest.raises(RemoteAttachmentError) as captured:
        session.read_parquet("https://example.test/data.parquet")

    assert "reader" not in str(captured.value)
    assert "super-secret" not in str(captured.value)


def test_remote_parquet_rejects_credentials_and_unsupported_schemes() -> None:
    session, _ = _session_with_fake()

    with pytest.raises(ValueError, match="must not contain credentials"):
        session.read_parquet("https://reader:secret@example.test/data.parquet")
    with pytest.raises(ValueError, match="must not contain credentials"):
        session.read_parquet("https://example.test/data.parquet?token=secret")
    with pytest.raises(ValueError, match="must use http"):
        session.read_parquet("ftp://example.test/data.parquet")
    with pytest.raises(ValueError, match="credential-free s3"):
        session.create_s3_secret(
            "unsafe",
            key_id="key",
            secret="secret",
            scope="s3://key:secret@bucket/path",
        )


def test_sqlite_attachment_is_read_only_and_refreshes(tmp_path: Path) -> None:
    path = tmp_path / "source.sqlite"
    path.touch()
    session, connection = _session_with_fake()

    attachment = session.attach_sqlite("catalog", path)
    query, parameters = connection.queries[-1]
    assert query == (
        f"ATTACH {quote_literal(str(path.resolve()))} "
        'AS "catalog" (TYPE sqlite, READ_ONLY)'
    )
    assert parameters == ()
    assert attachment.engine == "sqlite"

    frame = attachment.table("orders")
    assert isinstance(frame._plan, ScanPlan)
    source = cast("RemoteTableSource", frame._plan.source)
    assert source.engine == "sqlite"
    assert source.capabilities.filter

    attachment.refresh_schema()
    attach_queries = [sql for sql, _ in connection.queries if sql.startswith("ATTACH")]
    assert len(attach_queries) == 2
    attachment.detach()
    assert attachment.closed


def test_sqlite_attachment_reads_fresh_commits_without_write_access(
    tmp_path: Path,
) -> None:
    path = tmp_path / "live.sqlite"
    writer = sqlite3.connect(path)
    writer.execute("CREATE TABLE orders(id INTEGER, amount REAL)")
    writer.execute("INSERT INTO orders VALUES (1, 10.5)")
    writer.commit()

    with duckpd.connect() as session:
        attachment = session.attach_sqlite("catalog", path)
        frame = attachment.table("orders", order_by="id")
        assert frame.collect()["id"].tolist() == [1]

        writer.execute("INSERT INTO orders VALUES (2, 20.0)")
        writer.commit()
        assert frame.collect()["id"].tolist() == [1, 2]
        with pytest.raises(duckdb.Error):
            session._connection.execute(
                'INSERT INTO "catalog"."orders" VALUES (3, 30.0)'
            )

        writer.execute("ALTER TABLE orders ADD COLUMN note TEXT")
        writer.commit()
        attachment.refresh_schema()
        assert attachment.table("orders").columns == ("id", "amount", "note")

    writer.close()


class _RemoteSettings(TypedDict):
    host: str
    port: int
    database: str
    user: str
    password: str


def _remote_settings(engine: str) -> _RemoteSettings | None:
    prefix = f"DUCKPD_TEST_{engine.upper()}_"
    host = os.environ.get(prefix + "HOST")
    if host is None:
        return None
    return {
        "host": host,
        "port": int(os.environ[prefix + "PORT"]),
        "database": os.environ[prefix + "DATABASE"],
        "user": os.environ[prefix + "USER"],
        "password": os.environ[prefix + "PASSWORD"],
    }


def _attach_writer(engine: str, settings: _RemoteSettings) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.install_extension(engine)
    connection.load_extension(engine)
    secret_name = f"writer_{uuid4().hex}"
    connection.execute(
        f"CREATE TEMPORARY SECRET {quote_identifier(secret_name)} "
        f"(TYPE {engine}, HOST ?, PORT ?, DATABASE ?, USER ?, PASSWORD ?)",
        [
            settings["host"],
            settings["port"],
            settings["database"],
            settings["user"],
            settings["password"],
        ],
    )
    connection.execute(
        f"ATTACH '' AS writer (TYPE {engine}, SECRET {quote_identifier(secret_name)})"
    )
    return connection


def _attach_reader(
    engine: str,
    session: duckpd.Session,
    settings: _RemoteSettings,
) -> duckpd.AttachedDatabase:
    if engine == "postgres":
        return session.attach_postgres(
            "remote",
            host=settings["host"],
            port=settings["port"],
            database=settings["database"],
            user=settings["user"],
            password=settings["password"],
            unbounded_scan="allow",
        )
    return session.attach_mysql(
        "remote",
        host=settings["host"],
        port=settings["port"],
        database=settings["database"],
        user=settings["user"],
        password=settings["password"],
        unbounded_scan="allow",
    )


@pytest.mark.parametrize("engine", ["postgres", "mysql"])
def test_remote_attachment_refresh_visibility_and_read_only(engine: str) -> None:
    settings = _remote_settings(engine)
    if settings is None:
        pytest.skip(f"{engine} integration environment is not configured")

    table_name = f"duckpd_refresh_{uuid4().hex}"
    qualified_writer = (
        f"writer.public.{quote_identifier(table_name)}"
        if engine == "postgres"
        else f"writer.{quote_identifier(table_name)}"
    )
    writer = _attach_writer(engine, settings)
    writer.execute(
        f"CREATE TABLE {qualified_writer}"
        "(id INTEGER, amount DOUBLE, ignored VARCHAR(16))"
    )
    writer.execute(f"INSERT INTO {qualified_writer} VALUES (1, 10.5, 'unused')")

    session = duckpd.connect()
    attachment = _attach_reader(engine, session, settings)
    frame = attachment.table(table_name, order_by="id")
    pd.testing.assert_frame_equal(
        frame.collect(),
        pd.DataFrame({"id": [1], "amount": [10.5], "ignored": ["unused"]}),
        check_dtype=False,
    )
    filtered = attachment.table(table_name)
    filtered = filtered[filtered["amount"] > 5][["id"]]
    fragment = json.loads(filtered.explain("json"))["execution_boundaries"][
        "source_fragments"
    ][0]
    assert fragment["pushdown_candidates"] == ["projection", "filter"]
    assert fragment["local_required"] == []
    analyzed = filtered.explain("analyze")
    assert table_name in analyzed
    physical_plan = analyzed.rsplit("TABLE_SCAN", 1)
    assert len(physical_plan) == 2
    scan_details = "\n".join(
        line.replace("│", "").strip() for line in physical_plan[1].splitlines()
    )
    assert "ignored" not in scan_details
    assert re.search(r"Projections:\s+(?:amount\s+id|id\s+amount)\b", scan_details)
    assert re.search(r"Filters:\s+amount\s*>\s*5(?:\.0)?\b", scan_details)

    writer.execute("BEGIN")
    writer.execute(f"INSERT INTO {qualified_writer} VALUES (2, 20.0, 'unused')")
    assert frame.collect()["id"].tolist() == [1]
    writer.execute("COMMIT")
    assert frame.collect()["id"].tolist() == [1, 2]

    with pytest.raises(duckdb.Error):
        target = (
            f"remote.public.{quote_identifier(table_name)}"
            if engine == "postgres"
            else f"remote.{quote_identifier(table_name)}"
        )
        session._connection.execute(f"INSERT INTO {target} VALUES (3, 30.0, 'blocked')")

    if engine == "mysql":
        writer.execute(
            "CALL mysql_execute('writer', ?)",
            [f"ALTER TABLE {table_name} ADD COLUMN note VARCHAR(64)"],
        )
    else:
        writer.execute(f"ALTER TABLE {qualified_writer} ADD COLUMN note TEXT")
    attachment.refresh_schema()
    assert attachment.table(table_name).columns == (
        "id",
        "amount",
        "ignored",
        "note",
    )

    attachment.detach()
    with pytest.raises(RemoteAttachmentError, match="not available"):
        attachment.table(table_name)
    writer.execute(f"DROP TABLE {qualified_writer}")
    writer.close()
    session.close()
