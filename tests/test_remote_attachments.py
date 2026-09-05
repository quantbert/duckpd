from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, ClassVar, TypedDict, cast
from uuid import uuid4

import duckdb
import pandas as pd
import pytest

import duckpd
from duckpd._logical import RemoteTableSource, ScanPlan, SortPlan, TopKPlan
from duckpd._quoting import quote_identifier
from duckpd.errors import (
    MaterializationError,
    RemoteAttachmentError,
    RemoteScanWarning,
)


class _FakeRelation:
    columns: ClassVar[list[str]] = ["id", "amount"]
    types: ClassVar[list[str]] = ["INTEGER", "DOUBLE"]


class _FakeConnection:
    def __init__(
        self, *, fail_attach: bool = False, fail_inspect: bool = False
    ) -> None:
        self.queries: list[tuple[str, tuple[object, ...]]] = []
        self.installed: list[str] = []
        self.loaded: list[str] = []
        self.closed = False
        self._fail_attach = fail_attach
        self._fail_inspect = fail_inspect

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
        return self

    def sql(self, _query: str) -> _FakeRelation:
        if self._fail_inspect:
            raise duckdb.IOException("connection exposed reader and super-secret")
        return _FakeRelation()

    def close(self) -> None:
        self.closed = True


def _session_with_fake(
    *,
    fail_attach: bool = False,
    fail_inspect: bool = False,
) -> tuple[duckpd.Session, _FakeConnection]:
    session = duckpd.connect()
    session._connection.close()
    connection = _FakeConnection(
        fail_attach=fail_attach,
        fail_inspect=fail_inspect,
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
    assert session.execution_count == 0
    topk = frame.nlargest(1, "amount")
    with pytest.raises(MaterializationError, match="no proven transfer bound"):
        topk.collect()
    assert session.execution_count == 0
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
    assert not source.capabilities.filter
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


def test_remote_schema_inspection_failure_redacts_credentials() -> None:
    session, _ = _session_with_fake(fail_inspect=True)
    attachment = session.attach_postgres("analytics", secret="managed")

    with pytest.raises(RemoteAttachmentError) as captured:
        attachment.table("orders")

    message = str(captured.value)
    assert "reader" not in message
    assert "super-secret" not in message


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
    writer.execute(f"CREATE TABLE {qualified_writer}(id INTEGER, amount DOUBLE)")
    writer.execute(f"INSERT INTO {qualified_writer} VALUES (1, 10.5)")

    session = duckpd.connect()
    attachment = _attach_reader(engine, session, settings)
    frame = attachment.table(table_name, order_by="id")
    pd.testing.assert_frame_equal(
        frame.collect(),
        pd.DataFrame({"id": [1], "amount": [10.5]}),
        check_dtype=False,
    )

    writer.execute("BEGIN")
    writer.execute(f"INSERT INTO {qualified_writer} VALUES (2, 20.0)")
    assert frame.collect()["id"].tolist() == [1]
    writer.execute("COMMIT")
    assert frame.collect()["id"].tolist() == [1, 2]

    with pytest.raises(duckdb.Error):
        target = (
            f"remote.public.{quote_identifier(table_name)}"
            if engine == "postgres"
            else f"remote.{quote_identifier(table_name)}"
        )
        session._connection.execute(f"INSERT INTO {target} VALUES (3, 30.0)")

    if engine == "mysql":
        writer.execute(
            "CALL mysql_execute('writer', ?)",
            [f"ALTER TABLE {table_name} ADD COLUMN note VARCHAR(64)"],
        )
    else:
        writer.execute(f"ALTER TABLE {qualified_writer} ADD COLUMN note TEXT")
    attachment.refresh_schema()
    assert attachment.table(table_name).columns == ("id", "amount", "note")

    attachment.detach()
    with pytest.raises(RemoteAttachmentError, match="not available"):
        attachment.table(table_name)
    writer.execute(f"DROP TABLE {qualified_writer}")
    writer.close()
    session.close()
