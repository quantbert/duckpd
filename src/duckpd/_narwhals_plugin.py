"""Experimental Narwhals plugin for DuckPD lazy frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from types import ModuleType

    from narwhals._utils import Version
    from narwhals.dtypes import DType

    from duckpd.frame import DataFrame

NATIVE_PACKAGE = "duckpd"


def is_native(native_object: object) -> bool:
    """Return whether an object is a DuckPD lazy DataFrame."""
    from duckpd.frame import DataFrame

    return isinstance(native_object, DataFrame)


def __narwhals_namespace__(version: Version) -> DuckPDNamespace:
    """Return the Narwhals namespace for this plugin."""
    return DuckPDNamespace(version)


class DuckPDNamespace:
    """Narwhals plugin namespace for constructing compliant lazy frames."""

    def __init__(self, version: Version) -> None:
        self._version = version

    def from_native(self, data: object, /) -> DuckPDLazyFrame:
        if not is_native(data):
            raise TypeError(f"Expected duckpd.DataFrame, got {type(data).__name__}")
        return DuckPDLazyFrame(cast("DataFrame", data), version=self._version)


class DuckPDLazyFrame:
    """Narrow Narwhals-compliant wrapper which preserves DuckPD laziness."""

    def __init__(self, native_frame: DataFrame, *, version: Version) -> None:
        from narwhals._utils import Implementation

        self._native_frame = native_frame
        self._version = version
        self._implementation = Implementation.UNKNOWN

    @property
    def native(self) -> DataFrame:
        return self._native_frame

    @property
    def columns(self) -> list[str]:
        return list(self._native_frame.columns)

    @property
    def schema(self) -> dict[str, DType]:
        return self.collect_schema()

    def collect_schema(self) -> dict[str, DType]:
        dtypes = self._version.dtypes
        scalar_types: dict[str, DType] = {
            "BOOLEAN": dtypes.Boolean(),
            "TINYINT": dtypes.Int8(),
            "SMALLINT": dtypes.Int16(),
            "INTEGER": dtypes.Int32(),
            "BIGINT": dtypes.Int64(),
            "UTINYINT": dtypes.UInt8(),
            "USMALLINT": dtypes.UInt16(),
            "UINTEGER": dtypes.UInt32(),
            "UBIGINT": dtypes.UInt64(),
            "FLOAT": dtypes.Float32(),
            "DOUBLE": dtypes.Float64(),
            "VARCHAR": dtypes.String(),
            "BLOB": dtypes.Binary(),
            "DATE": dtypes.Date(),
            "TIME": dtypes.Time(),
        }
        return {
            column.label: scalar_types.get(column.duckdb_type, dtypes.Unknown())
            for column in self._native_frame._plan.metadata.visible_columns
        }

    def __narwhals_lazyframe__(self) -> DuckPDLazyFrame:
        return self

    def __narwhals_namespace__(self) -> DuckPDNamespace:
        return DuckPDNamespace(self._version)

    def __native_namespace__(self) -> ModuleType:
        import duckpd

        return duckpd

    def _with_version(self, version: Version) -> DuckPDLazyFrame:
        return DuckPDLazyFrame(self._native_frame, version=version)

    def _with_native(self, frame: DataFrame) -> DuckPDLazyFrame:
        return DuckPDLazyFrame(frame, version=self._version)

    def simple_select(self, *column_names: str) -> DuckPDLazyFrame:
        return self._with_native(self._native_frame[list(column_names)])

    def head(self, n: int) -> DuckPDLazyFrame:
        return self._with_native(self._native_frame.limit(n))

    def drop(self, columns: Sequence[str], *, strict: bool) -> DuckPDLazyFrame:
        if strict:
            missing = sorted(set(columns) - set(self.columns))
            if missing:
                raise KeyError(f"Columns not found: {missing}")
        retained = [name for name in self.columns if name not in columns]
        return self._with_native(self._native_frame[retained])

    def rename(self, mapping: Mapping[str, str]) -> DuckPDLazyFrame:
        return self._with_native(self._native_frame.rename(columns=dict(mapping)))

    def sort(
        self,
        *by: str,
        descending: bool | Sequence[bool],
        nulls_last: bool,
    ) -> DuckPDLazyFrame:
        ascending = (
            not descending
            if isinstance(descending, bool)
            else [not value for value in descending]
        )
        return self._with_native(
            self._native_frame.sort_values(
                list(by),
                ascending=ascending,
                na_position="last" if nulls_last else "first",
            )
        )

    def collect(self, backend: object = None, **kwargs: Any) -> object:
        from narwhals._arrow.dataframe import ArrowDataFrame
        from narwhals._utils import Implementation

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported collect arguments: {unexpected}")
        if backend not in (None, Implementation.PYARROW, "pyarrow"):
            raise ValueError(f"Unsupported collect backend: {backend!r}")
        return ArrowDataFrame(
            native_dataframe=self._native_frame.to_arrow(),
            version=self._version,
            validate_column_names=True,
            validate_backend_version=True,
        )
