"""DuckDB connection and resource ownership."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlsplit
from uuid import uuid4

import duckdb
import pandas as pd
import pyarrow as pa
from _duckdb._func import FunctionNullHandling, PythonUDFType

from duckpd._compiler import DuckDBCompiler
from duckpd._executor import Executor, MaterializationReport
from duckpd._logical import (
    ArrowSource,
    ColumnRef,
    CsvSource,
    NullPlacement,
    OrderColumn,
    OrderSpec,
    PandasSource,
    ParquetSource,
    RemoteTableSource,
    RowIdentity,
    ScanPlan,
    SortDirection,
    SortKey,
    SortPlan,
    SourceCapabilities,
    SourceKind,
    SourceProvenance,
    SqlSource,
    TableSource,
    sanitize_source_location,
)
from duckpd._metadata import after_sort, sort_keys_for_labels, source_metadata
from duckpd._quoting import quote_identifier, quote_literal
from duckpd.errors import (
    RemoteAttachmentError,
    SessionClosedError,
    UnsupportedOperationError,
)

if TYPE_CHECKING:
    from duckpd.frame import DataFrame

    ArrowUDF = Callable[
        ...,
        pa.Table | pa.Array[Any] | pa.ChunkedArray[Any],
    ]
else:
    ArrowUDF = Callable[..., object]


@dataclass(frozen=True)
class ArrowUDFSpec:
    """Validated Arrow UDF contract registered with one Session."""

    name: str
    input_types: tuple[str, ...]
    return_type: str
    null_handling: Literal["default", "special"]
    exception_handling: Literal["default", "return_null"]
    deterministic: bool
    side_effects: bool
    batch_independent: bool


@dataclass(frozen=True)
class _ObjectStoreSecretState:
    name: str
    provider: Literal["s3", "gcs"]


@dataclass(frozen=True)
class _RemoteAttachmentState:
    alias: str
    engine: Literal["postgres", "mysql", "sqlite"]
    location: str
    secret_name: str | None
    owns_secret: bool
    default_schema: str | None
    capabilities: SourceCapabilities
    unbounded_scan: Literal["error", "warn", "allow"]


@dataclass(frozen=True)
class ObjectStoreSecret:
    """A session-owned temporary credential for S3-compatible storage."""

    _session: Session
    name: str
    provider: Literal["s3", "gcs"]

    @property
    def closed(self) -> bool:
        """Whether the temporary secret has been removed."""
        return self.name not in self._session._object_store_secrets

    def drop(self) -> None:
        """Remove the temporary secret from the owning session."""
        self._session._drop_object_store_secret(self.name)

    def __repr__(self) -> str:
        return f"ObjectStoreSecret(name={self.name!r}, provider={self.provider!r})"


@dataclass(frozen=True)
class AttachedDatabase:
    """A credential-free handle to one read-only DuckDB attachment."""

    _session: Session
    alias: str
    engine: Literal["postgres", "mysql", "sqlite"]

    @property
    def closed(self) -> bool:
        """Whether this attachment is no longer available."""
        return self.alias not in self._session._attachments

    def table(
        self,
        name: str,
        *,
        schema: str | None = None,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
        unbounded_scan: Literal["error", "warn", "allow"] | None = None,
    ) -> DataFrame:
        """Create a lazy frame for one attached remote table."""
        return self._session._remote_table(
            self.alias,
            name,
            schema=schema,
            index=index,
            order_by=order_by,
            unbounded_scan=unbounded_scan,
        )

    def refresh_schema(self) -> None:
        """Refresh schema metadata for this attachment."""
        self._session._refresh_remote_schema(self.alias)

    def detach(self) -> None:
        """Detach the remote database and remove its temporary secret."""
        self._session._detach_remote(self.alias)

    def __repr__(self) -> str:
        return (
            f"AttachedDatabase(alias={self.alias!r}, engine={self.engine!r}, "
            f"read_only=True)"
        )


_POSTGRES_CAPABILITIES = SourceCapabilities(projection=True, filter=True)
_MYSQL_CAPABILITIES = SourceCapabilities(projection=True, filter=True)
_SQLITE_CAPABILITIES = SourceCapabilities(projection=True, filter=True)


class Session:
    """Own a DuckDB connection and every source referenced by its plans."""

    def __init__(
        self,
        database: str | Path = ":memory:",
        *,
        read_only: bool = False,
        memory_limit: str | None = None,
        temp_directory: str | Path | None = None,
        max_temp_directory_size: str | None = None,
        threads: int | None = None,
        fallback: Literal["error"] = "error",
    ) -> None:
        if fallback != "error":
            raise ValueError("fallback must be 'error'; implicit fallback is forbidden")
        config: dict[str, str | bool | int | float | list[str]] = {}
        if memory_limit is not None:
            config["memory_limit"] = memory_limit
        if temp_directory is not None:
            config["temp_directory"] = str(temp_directory)
        if max_temp_directory_size is not None:
            config["max_temp_directory_size"] = max_temp_directory_size
        if threads is not None:
            config["threads"] = threads
        config["preserve_insertion_order"] = True

        self._connection = duckdb.connect(
            database=str(database),
            read_only=read_only,
            config=config,
        )
        self._registered_sources: dict[str, object] = {}
        self._fallback: Literal["error"] = fallback
        self._arrow_udfs: dict[str, ArrowUDFSpec] = {}
        self._attachments: dict[str, _RemoteAttachmentState] = {}
        self._object_store_secrets: dict[str, _ObjectStoreSecretState] = {}
        self._last_materialization_report: MaterializationReport | None = None
        self._closed = False
        self._execution_count = 0
        self._compiler = DuckDBCompiler(self)
        self._executor = Executor(self, self._compiler)

    @property
    def execution_count(self) -> int:
        """Number of explicit execution boundaries entered by this session."""
        return self._execution_count

    @property
    def closed(self) -> bool:
        """Whether this session has released its connection."""
        return self._closed

    @property
    def fallback(self) -> Literal["error"]:
        """Unsupported operations fail instead of materializing a fallback."""
        return self._fallback

    @property
    def last_materialization_report(self) -> MaterializationReport | None:
        """Metrics for the latest explicit bounded materialization."""
        return self._last_materialization_report

    def register_arrow_udf(
        self,
        name: str,
        function: ArrowUDF,
        input_types: Sequence[str],
        return_type: str,
        *,
        null_handling: Literal["default", "special"] = "default",
        exception_handling: Literal["default", "return_null"] = "default",
        deterministic: bool = True,
        side_effects: bool = False,
        batch_independent: bool = True,
    ) -> ArrowUDFSpec:
        """Register an explicitly typed, batch-independent Arrow scalar UDF."""
        self._ensure_open()
        if not name or not name.isidentifier():
            raise ValueError("Arrow UDF name must be a non-empty identifier")
        declared_inputs = tuple(item.strip().upper() for item in input_types)
        declared_return = return_type.strip().upper()
        if not declared_inputs or any(not item for item in declared_inputs):
            raise ValueError("Arrow UDF input_types must not be empty")
        if not declared_return:
            raise ValueError("Arrow UDF return_type must not be empty")
        if not batch_independent:
            raise UnsupportedOperationError(
                "Arrow UDF fallback must be independent for every input batch"
            )
        if deterministic and side_effects:
            raise ValueError("A deterministic Arrow UDF cannot declare side effects")
        registry_name = name.casefold()
        if registry_name in self._arrow_udfs:
            raise ValueError(f"Arrow UDF {name!r} is already registered")
        spec = ArrowUDFSpec(
            name=name,
            input_types=declared_inputs,
            return_type=declared_return,
            null_handling=null_handling,
            exception_handling=exception_handling,
            deterministic=deterministic,
            side_effects=side_effects,
            batch_independent=batch_independent,
        )
        null_option = (
            FunctionNullHandling.SPECIAL
            if null_handling == "special"
            else FunctionNullHandling.DEFAULT
        )
        exception_option = (
            duckdb.PythonExceptionHandling.RETURN_NULL
            if exception_handling == "return_null"
            else duckdb.PythonExceptionHandling.DEFAULT
        )
        self._connection.create_function(  # pyright: ignore[reportUnknownMemberType]
            name,
            function,
            list(declared_inputs),
            declared_return,
            type=PythonUDFType.ARROW,
            null_handling=null_option,
            exception_handling=exception_option,
            side_effects=side_effects or not deterministic,
        )
        self._arrow_udfs[registry_name] = spec
        return spec

    def _arrow_udf(self, name: str) -> ArrowUDFSpec:
        try:
            return self._arrow_udfs[name.casefold()]
        except KeyError:
            raise KeyError(
                f"Arrow UDF {name!r} is not registered in this session"
            ) from None

    def from_pandas(
        self,
        value: pd.DataFrame,
        *,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Create a lazy frame while retaining the pandas source."""
        from duckpd.frame import DataFrame

        self._ensure_open()
        if not value.columns.is_unique:
            msg = "DuckPD does not yet support duplicate column labels"
            raise ValueError(msg)
        labels = cast("list[object]", value.columns.to_list())
        if not all(isinstance(label, str) for label in labels):
            msg = "DuckPD currently requires string column labels"
            raise TypeError(msg)

        key = uuid4().hex
        ordinal_label = f"__duckpd_row_ordinal_{key}__"
        snapshot = value.copy()
        snapshot[ordinal_label] = range(len(snapshot))
        self._registered_sources[key] = snapshot
        source = PandasSource(key)
        plan = self._source_plan(
            source,
            index=index,
            order_by=order_by,
            stable_order_label=ordinal_label,
        )
        return DataFrame(self, plan)

    def from_arrow(
        self,
        value: pa.Table | pa.RecordBatch,
        *,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Create a lazy frame while retaining the Arrow source."""
        from duckpd.frame import DataFrame

        self._ensure_open()
        key = uuid4().hex
        ordinal_label = f"__duckpd_row_ordinal_{key}__"
        ordered_value = value.append_column(
            ordinal_label, pa.array(range(value.num_rows), type=pa.int64())
        )
        self._registered_sources[key] = ordered_value
        source = ArrowSource(key)
        plan = self._source_plan(
            source,
            index=index,
            order_by=order_by,
            stable_order_label=ordinal_label,
        )
        return DataFrame(self, plan)

    def create_s3_secret(
        self,
        name: str,
        *,
        key_id: str | None = None,
        secret: str | None = None,
        region: str | None = None,
        endpoint: str | None = None,
        scope: str | None = None,
        credential_chain: bool = False,
    ) -> ObjectStoreSecret:
        """Create a temporary scoped S3 secret without storing credentials in plans."""
        if credential_chain:
            if key_id is not None or secret is not None:
                raise ValueError(
                    "credential_chain cannot be combined with key_id or secret"
                )
            return self._create_object_store_secret(
                "s3",
                name,
                provider="credential_chain",
                region=region,
                endpoint=endpoint,
                scope=scope,
            )
        if not key_id or not secret:
            raise ValueError("S3 key_id and secret must both be non-empty")
        return self._create_object_store_secret(
            "s3",
            name,
            key_id=key_id,
            secret=secret,
            region=region,
            endpoint=endpoint,
            scope=scope,
        )

    def create_gcs_secret(
        self,
        name: str,
        *,
        key_id: str,
        secret: str,
        scope: str | None = None,
    ) -> ObjectStoreSecret:
        """Create a temporary scoped GCS HMAC secret."""
        if not key_id or not secret:
            raise ValueError("GCS key_id and secret must both be non-empty")
        return self._create_object_store_secret(
            "gcs", name, key_id=key_id, secret=secret, scope=scope
        )

    def _create_object_store_secret(
        self,
        secret_type: Literal["s3", "gcs"],
        name: str,
        *,
        key_id: str | None = None,
        secret: str | None = None,
        provider: Literal["credential_chain"] | None = None,
        region: str | None = None,
        endpoint: str | None = None,
        scope: str | None = None,
    ) -> ObjectStoreSecret:
        self._ensure_open()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(
                "secret name must start with a letter or underscore and contain "
                "only letters, digits, and underscores"
            )
        if name in self._object_store_secrets:
            raise ValueError(f"object-store secret {name!r} is already in use")
        if scope is not None:
            parsed_scope = urlsplit(scope)
            expected_schemes = {"s3"} if secret_type == "s3" else {"gcs", "gs"}
            if (
                parsed_scope.scheme not in expected_schemes
                or not parsed_scope.netloc
                or parsed_scope.username is not None
                or parsed_scope.query
                or parsed_scope.fragment
            ):
                schemes = " or ".join(f"{item}://" for item in sorted(expected_schemes))
                raise ValueError(f"scope must be a credential-free {schemes} URI")

        extension = "aws" if provider == "credential_chain" else "httpfs"
        quoted_name = quote_identifier(name)
        fields = [f"TYPE {secret_type}"]
        parameters: list[object] = []
        for field, value in (
            ("PROVIDER", provider),
            ("KEY_ID", key_id),
            ("SECRET", secret),
            ("REGION", region),
            ("ENDPOINT", endpoint),
            ("SCOPE", scope),
        ):
            if value is not None:
                fields.append(f"{field} ?")
                parameters.append(value)
        try:
            self._connection.install_extension(extension)
            self._connection.load_extension(extension)
            self._connection.execute(
                f"CREATE TEMPORARY SECRET {quoted_name} ({', '.join(fields)})",
                parameters,
            )
        except duckdb.Error as error:
            raise RemoteAttachmentError(
                f"Failed to create {secret_type} secret {name!r} "
                f"({type(error).__name__})"
            ) from None
        self._object_store_secrets[name] = _ObjectStoreSecretState(name, secret_type)
        return ObjectStoreSecret(self, name, secret_type)

    def _drop_object_store_secret(self, name: str) -> None:
        self._ensure_open()
        if name not in self._object_store_secrets:
            raise RemoteAttachmentError(
                f"Object-store secret {name!r} is not available"
            )
        try:
            self._connection.execute(f"DROP SECRET {quote_identifier(name)}")
        except duckdb.Error as error:
            raise RemoteAttachmentError(
                f"Failed to drop object-store secret {name!r} ({type(error).__name__})"
            ) from None
        del self._object_store_secrets[name]

    def read_parquet(
        self,
        path: str | Path | Sequence[str | Path],
        *,
        hive_partitioning: bool = False,
        union_by_name: bool = False,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Create a lazy scan over local, HTTP, S3, or GCS Parquet files."""
        from duckpd.frame import DataFrame

        self._ensure_open()
        if isinstance(path, (str, Path)):
            raw_paths = (str(path),)
        else:
            raw_paths = tuple(str(item) for item in path)
        paths = tuple(
            item if "://" in item else str(Path(item).expanduser().resolve())
            for item in raw_paths
        )
        if not paths:
            msg = "At least one Parquet path is required"
            raise ValueError(msg)
        remote = False
        for item in paths:
            if "://" not in item:
                continue
            parsed = urlsplit(item)
            if parsed.scheme not in {"http", "https", "s3", "gcs", "gs"}:
                raise ValueError(
                    "Remote Parquet paths must use http, https, s3, gcs, or gs"
                )
            if (
                parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "Remote Parquet paths must not contain credentials, query "
                    "parameters, or fragments; configure a scoped secret instead"
                )
            remote = True
        if remote:
            try:
                self._connection.install_extension("httpfs")
                self._connection.load_extension("httpfs")
            except duckdb.Error as error:
                raise RemoteAttachmentError(
                    f"Failed to enable remote Parquet access ({type(error).__name__})"
                ) from None

        ordinal_label = f"__duckpd_scan_ordinal_{uuid4().hex}__"
        source = ParquetSource(
            paths,
            hive_partitioning,
            union_by_name,
            stable_order_label=ordinal_label,
            native_order=(
                len(paths) == 1 and not any(char in paths[0] for char in "*?[]")
            ),
        )
        try:
            plan = self._source_plan(
                source,
                index=index,
                order_by=order_by,
                stable_order_label=ordinal_label,
            )
        except duckdb.Error as error:
            if not remote:
                raise
            raise RemoteAttachmentError(
                f"Failed to inspect remote Parquet source ({type(error).__name__})"
            ) from None
        return DataFrame(self, plan)

    def read_csv(
        self,
        path: str | Path | Sequence[str | Path],
        *,
        header: bool = True,
        delimiter: str = ",",
        auto_detect: bool = True,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Create a lazy scan over one or more CSV files."""
        from duckpd.frame import DataFrame

        self._ensure_open()
        paths = (
            (str(path),)
            if isinstance(path, (str, Path))
            else tuple(str(item) for item in path)
        )
        if not paths:
            msg = "At least one CSV path is required"
            raise ValueError(msg)

        ordinal_label = f"__duckpd_scan_ordinal_{uuid4().hex}__"
        source = CsvSource(
            paths,
            header=header,
            delimiter=delimiter,
            auto_detect=auto_detect,
            stable_order_label=ordinal_label,
        )
        plan = self._source_plan(
            source,
            index=index,
            order_by=order_by,
            stable_order_label=ordinal_label,
        )
        return DataFrame(self, plan)

    def attach_postgres(
        self,
        alias: str,
        *,
        host: str | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int | None = None,
        schema: str | None = None,
        sslmode: str | None = None,
        secret: str | None = None,
        unbounded_scan: Literal["error", "warn", "allow"] = "warn",
    ) -> AttachedDatabase:
        """Attach PostgreSQL through DuckDB's read-only postgres extension."""
        return self._attach_remote(
            "postgres",
            alias,
            host=host,
            database=database,
            user=user,
            password=password,
            port=port,
            schema=schema,
            sslmode=sslmode,
            secret=secret,
            unbounded_scan=unbounded_scan,
        )

    def attach_mysql(
        self,
        alias: str,
        *,
        host: str | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int | None = None,
        secret: str | None = None,
        unbounded_scan: Literal["error", "warn", "allow"] = "warn",
    ) -> AttachedDatabase:
        """Attach MySQL through DuckDB's read-only mysql extension."""
        return self._attach_remote(
            "mysql",
            alias,
            host=host,
            database=database,
            user=user,
            password=password,
            port=port,
            schema=None,
            sslmode=None,
            secret=secret,
            unbounded_scan=unbounded_scan,
        )

    def attach_sqlite(
        self,
        alias: str,
        path: str | Path,
        *,
        unbounded_scan: Literal["error", "warn", "allow"] = "allow",
    ) -> AttachedDatabase:
        """Attach an existing SQLite database through DuckDB in read-only mode."""
        self._ensure_open()
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise ValueError("SQLite attachment path must be an existing file")
        return self._attach_sqlite(alias, resolved, unbounded_scan=unbounded_scan)

    def _attach_sqlite(
        self,
        alias: str,
        path: Path,
        *,
        unbounded_scan: Literal["error", "warn", "allow"],
    ) -> AttachedDatabase:
        self._validate_attachment(alias, unbounded_scan)
        quoted_alias = quote_identifier(alias)
        try:
            self._connection.install_extension("sqlite")
            self._connection.load_extension("sqlite")
            self._connection.execute(
                f"ATTACH {quote_literal(str(path))} AS {quoted_alias} "
                "(TYPE sqlite, READ_ONLY)"
            )
        except duckdb.Error as error:
            raise RemoteAttachmentError(
                f"Failed to attach SQLite database as {alias!r} "
                f"({type(error).__name__})"
            ) from None
        self._attachments[alias] = _RemoteAttachmentState(
            alias=alias,
            engine="sqlite",
            location=str(path),
            secret_name=None,
            owns_secret=False,
            default_schema=None,
            capabilities=_SQLITE_CAPABILITIES,
            unbounded_scan=unbounded_scan,
        )
        return AttachedDatabase(self, alias, "sqlite")

    def _validate_attachment(
        self,
        alias: str,
        unbounded_scan: Literal["error", "warn", "allow"],
    ) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", alias):
            raise ValueError(
                "attachment alias must start with a letter or underscore and "
                "contain only letters, digits, and underscores"
            )
        if alias in self._attachments:
            raise ValueError(f"attachment alias {alias!r} is already in use")
        if unbounded_scan not in {"error", "warn", "allow"}:
            raise ValueError("unbounded_scan must be 'error', 'warn', or 'allow'")

    def _attach_remote(
        self,
        engine: Literal["postgres", "mysql"],
        alias: str,
        *,
        host: str | None,
        database: str | None,
        user: str | None,
        password: str | None,
        port: int | None,
        schema: str | None,
        sslmode: str | None,
        secret: str | None,
        unbounded_scan: Literal["error", "warn", "allow"],
    ) -> AttachedDatabase:
        self._ensure_open()
        self._validate_attachment(alias, unbounded_scan)
        if schema is not None and not schema:
            raise ValueError("schema must be non-empty when provided")

        owns_secret = secret is None
        if secret is None:
            required = {
                "host": host,
                "database": database,
                "user": user,
                "password": password,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "structured connection parameters are missing: "
                    + ", ".join(missing)
                )
            if any(value == "" for value in required.values()):
                raise ValueError("structured connection parameters must be non-empty")
            secret_name = f"__duckpd_{engine}_{uuid4().hex}"
        else:
            if not secret:
                raise ValueError("secret must be non-empty")
            if any(
                value is not None
                for value in (host, database, user, password, port, sslmode)
            ):
                raise ValueError(
                    "secret cannot be combined with structured connection parameters"
                )
            secret_name = secret

        default_port = 5432 if engine == "postgres" else 3306
        resolved_port = default_port if port is None else port
        if type(resolved_port) is not int or not 1 <= resolved_port <= 65535:
            raise ValueError("port must be an integer between 1 and 65535")

        quoted_alias = quote_identifier(alias)
        quoted_secret = quote_identifier(secret_name)
        attached = False
        try:
            self._connection.install_extension(engine)
            self._connection.load_extension(engine)
            if owns_secret:
                secret_sql = (
                    f"CREATE TEMPORARY SECRET {quoted_secret} "
                    f"(TYPE {engine}, HOST ?, PORT ?, DATABASE ?, USER ?, PASSWORD ?"
                )
                parameters: list[object] = [
                    cast("str", host),
                    resolved_port,
                    cast("str", database),
                    cast("str", user),
                    cast("str", password),
                ]
                if sslmode is not None:
                    secret_sql += ", SSLMODE ?"
                    parameters.append(sslmode)
                secret_sql += ")"
                self._connection.execute(secret_sql, parameters)

            attach_sql = (
                f"ATTACH '' AS {quoted_alias} "
                f"(TYPE {engine}, SECRET {quoted_secret}, READ_ONLY"
            )
            if engine == "postgres" and schema is not None:
                attach_sql += f", SCHEMA {quote_literal(schema)}"
            attach_sql += ")"
            self._connection.execute(attach_sql)
            attached = True
        except duckdb.Error as error:
            if attached:
                with suppress(duckdb.Error):
                    self._connection.execute(f"DETACH {quoted_alias}")
            if owns_secret:
                with suppress(duckdb.Error):
                    self._connection.execute(f"DROP SECRET {quoted_secret}")
            raise RemoteAttachmentError(
                f"Failed to attach {engine} database as {alias!r} "
                f"({type(error).__name__})"
            ) from None

        location = (
            f"{engine}://{host}:{resolved_port}/{database}"
            if owns_secret
            else f"{engine}://attached/{alias}"
        )
        state = _RemoteAttachmentState(
            alias=alias,
            engine=engine,
            location=location,
            secret_name=secret_name,
            owns_secret=owns_secret,
            default_schema=schema,
            capabilities=(
                _POSTGRES_CAPABILITIES if engine == "postgres" else _MYSQL_CAPABILITIES
            ),
            unbounded_scan=unbounded_scan,
        )
        self._attachments[alias] = state
        return AttachedDatabase(self, alias, engine)

    def _remote_table(
        self,
        alias: str,
        name: str,
        *,
        schema: str | None,
        index: str | Sequence[str] | None,
        order_by: str | Sequence[str] | None,
        unbounded_scan: Literal["error", "warn", "allow"] | None,
    ) -> DataFrame:
        from duckpd.frame import DataFrame

        self._ensure_open()
        try:
            attachment = self._attachments[alias]
        except KeyError:
            raise RemoteAttachmentError(
                f"Remote attachment {alias!r} is not available"
            ) from None
        if not name:
            raise ValueError("remote table name must be non-empty")
        effective_schema = attachment.default_schema if schema is None else schema
        if effective_schema == "":
            raise ValueError("remote schema must be non-empty when provided")
        policy = attachment.unbounded_scan if unbounded_scan is None else unbounded_scan
        if policy not in {"error", "warn", "allow"}:
            raise ValueError("unbounded_scan must be 'error', 'warn', or 'allow'")
        source = RemoteTableSource(
            engine=attachment.engine,
            attachment=alias,
            table=name,
            schema=effective_schema,
            location=attachment.location,
            capabilities=attachment.capabilities,
            unbounded_scan=policy,
        )
        try:
            plan = self._source_plan(source, index=index, order_by=order_by)
        except duckdb.Error as error:
            raise RemoteAttachmentError(
                f"Failed to inspect {attachment.engine} table from attachment "
                f"{alias!r} ({type(error).__name__})"
            ) from None
        return DataFrame(self, plan)

    def _refresh_remote_schema(self, alias: str) -> None:
        self._ensure_open()
        try:
            attachment = self._attachments[alias]
        except KeyError:
            raise RemoteAttachmentError(
                f"Remote attachment {alias!r} is not available"
            ) from None
        if attachment.engine == "sqlite":
            location = Path(attachment.location)
            self._connection.execute(f"DETACH {quote_identifier(alias)}")
            del self._attachments[alias]
            self._attach_sqlite(
                alias,
                location,
                unbounded_scan=attachment.unbounded_scan,
            )
            return
        function = (
            "pg_clear_cache" if attachment.engine == "postgres" else "mysql_clear_cache"
        )
        try:
            self._connection.execute(f"CALL {function}()")
        except duckdb.Error as error:
            raise RemoteAttachmentError(
                f"Failed to refresh {attachment.engine} schema cache "
                f"({type(error).__name__})"
            ) from None

    def _detach_remote(self, alias: str) -> None:
        self._ensure_open()
        try:
            attachment = self._attachments[alias]
        except KeyError:
            raise RemoteAttachmentError(
                f"Remote attachment {alias!r} is not available"
            ) from None
        try:
            self._connection.execute(f"DETACH {quote_identifier(alias)}")
            if attachment.owns_secret and attachment.secret_name is not None:
                self._connection.execute(
                    f"DROP SECRET {quote_identifier(attachment.secret_name)}"
                )
        except duckdb.Error as error:
            raise RemoteAttachmentError(
                f"Failed to detach {attachment.engine} database {alias!r} "
                f"({type(error).__name__})"
            ) from None
        del self._attachments[alias]

    def table(
        self,
        name: str,
        *,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
        unbounded_scan: Literal["error", "warn", "allow"] | None = None,
    ) -> DataFrame:
        """Create a lazy frame for a local or attached remote table."""
        from duckpd.frame import DataFrame

        self._ensure_open()
        parts = name.split(".")
        if parts[0] in self._attachments:
            if len(parts) == 2:
                schema = None
                table_name = parts[1]
            elif len(parts) == 3:
                schema = parts[1]
                table_name = parts[2]
            else:
                raise ValueError(
                    "attached table names must be 'alias.table' or 'alias.schema.table'"
                )
            return self._remote_table(
                parts[0],
                table_name,
                schema=schema,
                index=index,
                order_by=order_by,
                unbounded_scan=unbounded_scan,
            )
        if unbounded_scan is not None:
            raise ValueError("unbounded_scan applies only to attached remote tables")
        source = TableSource(name)
        plan = self._source_plan(source, index=index, order_by=order_by)
        return DataFrame(self, plan)

    def sql(
        self,
        query: str,
        *,
        index: str | Sequence[str] | None = None,
        order_by: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Create a lazy frame from exactly one read-only SQL query."""
        from duckpd.frame import DataFrame

        self._ensure_open()
        statements = self._connection.extract_statements(query)
        if len(statements) != 1:
            msg = "Session.sql() requires exactly one SELECT statement"
            raise UnsupportedOperationError(msg)
        if statements[0].type != duckdb.StatementType.SELECT:
            msg = "Session.sql() only accepts read-only SELECT statements"
            raise UnsupportedOperationError(msg)

        source = SqlSource(query)
        plan = self._source_plan(source, index=index, order_by=order_by)
        return DataFrame(self, plan)

    def close(self) -> None:
        """Release attachments, temporary secrets, and retained sources."""
        if self._closed:
            return
        for attachment in self._attachments.values():
            with suppress(duckdb.Error):
                self._connection.execute(f"DETACH {quote_identifier(attachment.alias)}")
            if attachment.owns_secret and attachment.secret_name is not None:
                with suppress(duckdb.Error):
                    self._connection.execute(
                        f"DROP SECRET {quote_identifier(attachment.secret_name)}"
                    )
        for name in self._object_store_secrets:
            with suppress(duckdb.Error):
                self._connection.execute(f"DROP SECRET {quote_identifier(name)}")
        self._object_store_secrets.clear()
        self._attachments.clear()
        self._registered_sources.clear()
        self._connection.close()
        self._closed = True

    def __enter__(self) -> Session:
        self._ensure_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            msg = "DuckPD session is closed"
            raise SessionClosedError(msg)

    def _get_registered_source(self, key: str) -> object:
        self._ensure_open()
        return self._registered_sources[key]

    def _begin_execution(self) -> None:
        self._ensure_open()
        self._execution_count += 1

    def _source_plan(
        self,
        source: (
            ArrowSource
            | CsvSource
            | PandasSource
            | ParquetSource
            | RemoteTableSource
            | SqlSource
            | TableSource
        ),
        *,
        index: str | Sequence[str] | None,
        order_by: str | Sequence[str] | None,
        stable_order_label: str | None = None,
    ) -> ScanPlan | SortPlan:
        columns = self._compiler.inspect_source(source)
        if (
            isinstance(source, ParquetSource)
            and source.native_order
            and any(column.label == "file_row_number" for column in columns)
        ):
            source = replace(source, native_order=False)
        index_labels = self._normalize_labels(index)
        provenance = self._source_provenance(source)
        metadata = source_metadata(
            columns,
            index_labels=index_labels,
            provenance=provenance,
        )
        stable_order_key: OrderColumn | None = None
        if stable_order_label is not None:
            stable_column = next(
                column
                for column in metadata.columns
                if column.label == stable_order_label
            )
            stable_order_key = OrderColumn(
                stable_column.id,
                SortDirection.ASCENDING,
                NullPlacement.LAST,
            )
            updated_columns = tuple(
                replace(column, hidden=True)
                if column.id == stable_column.id
                else column
                for column in metadata.columns
            )
            metadata = replace(
                metadata,
                columns=updated_columns,
                ordering=OrderSpec((stable_order_key,)),
                row_identity=RowIdentity(
                    (stable_column.id,),
                    stable=True,
                    unique=True,
                    source_key=(
                        provenance.locations[0] if provenance.locations else None
                    ),
                ),
            )
        scan = ScanPlan(source, metadata)
        order_labels = self._normalize_labels(order_by)
        if not order_labels:
            return scan
        keys = sort_keys_for_labels(metadata, order_labels)
        if stable_order_key is not None:
            keys = (
                *keys,
                SortKey(
                    ColumnRef(stable_order_key.column_id),
                    stable_order_key.direction,
                    stable_order_key.null_placement,
                ),
            )
        return SortPlan(scan, keys, after_sort(metadata, keys))

    @staticmethod
    def _source_provenance(
        source: (
            ArrowSource
            | CsvSource
            | PandasSource
            | ParquetSource
            | RemoteTableSource
            | SqlSource
            | TableSource
        ),
    ) -> SourceProvenance:
        if isinstance(source, PandasSource):
            return SourceProvenance(
                SourceKind.PANDAS,
                (source.key,),
                fingerprint=hashlib.sha256(source.key.encode()).hexdigest(),
            )
        if isinstance(source, ArrowSource):
            return SourceProvenance(
                SourceKind.ARROW,
                (source.key,),
                fingerprint=hashlib.sha256(source.key.encode()).hexdigest(),
            )
        if isinstance(source, (ParquetSource, CsvSource)):
            canonical = tuple(
                (
                    sanitize_source_location(path)
                    if "://" in path
                    else str(Path(path).resolve())
                )
                for path in source.paths
            )
            fingerprints: list[str] = []
            for original, location in zip(source.paths, canonical, strict=True):
                if "://" in original or any(char in original for char in "*?[]"):
                    fingerprints.append(hashlib.sha256(original.encode()).hexdigest())
                    continue
                stat = Path(location).stat()
                fingerprints.append(f"{location}:{stat.st_size}:{stat.st_mtime_ns}")
            fingerprint = hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
            if isinstance(source, ParquetSource):
                writable = (
                    len(canonical) == 1
                    and "://" not in canonical[0]
                    and not source.hive_partitioning
                )
                return SourceProvenance(
                    SourceKind.PARQUET,
                    canonical,
                    fingerprint=fingerprint,
                    writable=writable,
                )
            return SourceProvenance(
                SourceKind.CSV,
                canonical,
                fingerprint=fingerprint,
            )
        if isinstance(source, RemoteTableSource):
            if source.engine == "postgres":
                kind = SourceKind.POSTGRES
            elif source.engine == "mysql":
                kind = SourceKind.MYSQL
            else:
                kind = SourceKind.SQLITE
            location = (
                f"{sanitize_source_location(source.location)}/{source.qualified_name}"
            )
            return SourceProvenance(
                kind,
                (location,),
                writable=False,
                capabilities=source.capabilities,
            )
        if isinstance(source, TableSource):
            return SourceProvenance(
                SourceKind.TABLE,
                (source.name,),
                writable=True,
            )
        return SourceProvenance(
            SourceKind.SQL,
            fingerprint=hashlib.sha256(source.query.encode()).hexdigest(),
        )

    @staticmethod
    def _normalize_labels(value: str | Sequence[str] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        labels = (value,) if isinstance(value, str) else tuple(value)
        if len(labels) != len(set(labels)):
            raise ValueError("Metadata column labels must be unique")
        return labels


def connect(
    database: str | Path = ":memory:",
    *,
    read_only: bool = False,
    memory_limit: str | None = None,
    temp_directory: str | Path | None = None,
    max_temp_directory_size: str | None = None,
    threads: int | None = None,
    fallback: Literal["error"] = "error",
) -> Session:
    """Create a DuckPD session."""
    return Session(
        database,
        read_only=read_only,
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        max_temp_directory_size=max_temp_directory_size,
        threads=threads,
        fallback=fallback,
    )
