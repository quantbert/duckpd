"""Experimental Narwhals plugin for DuckPD lazy frames."""

from __future__ import annotations

import re
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from types import ModuleType

    from narwhals._expression_parsing import ExprMetadata
    from narwhals._utils import Version
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType

    from duckpd.frame import DataFrame
    from duckpd.series import Series

NATIVE_PACKAGE = "duckpd"


def is_native(native_object: object) -> bool:
    """Return whether an object is a DuckPD lazy DataFrame."""
    from duckpd.frame import DataFrame

    return isinstance(native_object, DataFrame)


def __narwhals_namespace__(version: Version) -> DuckPDNamespace:
    """Return the Narwhals namespace for this plugin."""
    return DuckPDNamespace(version)


class DuckPDExpr:
    """Narwhals expression translated into DuckPD Series expressions."""

    _opt_metadata: ExprMetadata | None = None

    def __init__(
        self,
        call: Callable[[DuckPDLazyFrame], Sequence[Series]],
        *,
        evaluate_output_names: Callable[[DuckPDLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        version: Version,
        aggregation: tuple[str, int] | None = None,
    ) -> None:
        from narwhals._utils import Implementation

        self._call = call
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._version = version
        self._aggregation = aggregation
        self._implementation = Implementation.UNKNOWN

    @property
    def _metadata(self) -> ExprMetadata:
        if self._opt_metadata is None:
            raise AssertionError(
                "Narwhals expression metadata has not been initialized"
            )
        return self._opt_metadata

    def __call__(self, frame: DuckPDLazyFrame) -> Sequence[Series]:
        return self._call(frame)

    def __narwhals_expr__(self) -> DuckPDExpr:
        return self

    def __narwhals_namespace__(self) -> DuckPDNamespace:
        return DuckPDNamespace(self._version)

    def broadcast(self) -> DuckPDExpr:
        return self._clone()

    def alias(self, name: str) -> DuckPDExpr:
        def alias_output_names(names: Sequence[str]) -> Sequence[str]:
            if len(names) != 1:
                raise ValueError(
                    f"Expected expression with one output, found output names: {names}"
                )
            return [name]

        return self._clone(alias_output_names=alias_output_names)

    def cast(self, dtype: IntoDType) -> DuckPDExpr:
        target = _narwhals_dtype_to_duckdb(dtype)
        return self._elementwise(lambda series: series.astype(target))

    def is_null(self) -> DuckPDExpr:
        return self._elementwise(lambda series: series.isna())

    @property
    def str(self) -> DuckPDExprStringNamespace:
        return DuckPDExprStringNamespace(self)

    @property
    def dt(self) -> DuckPDExprDatetimeNamespace:
        return DuckPDExprDatetimeNamespace(self)

    def sum(self) -> DuckPDExpr:
        return self._aggregate("sum")

    def min(self) -> DuckPDExpr:
        return self._aggregate("min")

    def max(self) -> DuckPDExpr:
        return self._aggregate("max")

    def mean(self) -> DuckPDExpr:
        return self._aggregate("mean")

    def count(self) -> DuckPDExpr:
        return self._aggregate("count")

    def len(self) -> DuckPDExpr:
        return self._aggregate("size")

    def median(self) -> DuckPDExpr:
        return self._aggregate("median")

    def std(self, *, ddof: int) -> DuckPDExpr:
        if ddof not in {0, 1}:
            raise NotImplementedError("DuckPD supports Narwhals std with ddof=0 or 1")
        return self._aggregate("std", ddof=ddof)

    def var(self, *, ddof: int) -> DuckPDExpr:
        if ddof not in {0, 1}:
            raise NotImplementedError("DuckPD supports Narwhals var with ddof=0 or 1")
        return self._aggregate("var", ddof=ddof)

    def _aggregate(self, name: str, *, ddof: int = 1) -> DuckPDExpr:
        if self._aggregation is not None:
            raise NotImplementedError("Nested Narwhals aggregations are not supported")
        return self._clone(aggregation=(name, ddof))

    def __invert__(self) -> DuckPDExpr:
        return self._elementwise(lambda series: ~series)

    def __neg__(self) -> DuckPDExpr:
        return self._elementwise(lambda series: -series)

    def __eq__(self, other: object) -> DuckPDExpr:  # type: ignore[override]
        return self._binary(other, lambda left, right: left == right)

    def __ne__(self, other: object) -> DuckPDExpr:  # type: ignore[override]
        return self._binary(other, lambda left, right: left != right)

    def __lt__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left < right)

    def __le__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left <= right)

    def __gt__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left > right)

    def __ge__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left >= right)

    def __and__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left & right)

    def __or__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left | right)

    def __add__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left + right)

    def __sub__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left - right)

    def __rsub__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: right - left).alias("literal")

    def __mul__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left * right)

    def __truediv__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left / right)

    def __rtruediv__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: right / left).alias("literal")

    def __mod__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left % right)

    def __rmod__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: right % left).alias("literal")

    def _clone(
        self,
        *,
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None = None,
        aggregation: tuple[str, int] | None = None,
    ) -> DuckPDExpr:
        return DuckPDExpr(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=(
                self._alias_output_names
                if alias_output_names is None
                else alias_output_names
            ),
            version=self._version,
            aggregation=self._aggregation if aggregation is None else aggregation,
        )

    def _elementwise(self, operation: Callable[[Series], Series]) -> DuckPDExpr:
        def call(frame: DuckPDLazyFrame) -> Sequence[Series]:
            return [operation(series) for series in self(frame)]

        return DuckPDExpr(
            call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def _binary(
        self,
        other: object,
        operation: Callable[[Series, Series], Series],
    ) -> DuckPDExpr:
        if not isinstance(other, DuckPDExpr):
            raise TypeError(f"Expected a DuckPD expression, got {type(other).__name__}")

        def call(frame: DuckPDLazyFrame) -> Sequence[Series]:
            left = list(self(frame))
            right = list(other(frame))
            if len(left) == 1:
                left *= len(right)
            elif len(right) == 1:
                right *= len(left)
            if len(left) != len(right):
                from narwhals.exceptions import MultiOutputExpressionError

                raise MultiOutputExpressionError(
                    "Cannot combine expressions with different output counts"
                )
            return [
                operation(left_value, right_value)
                for left_value, right_value in zip(left, right, strict=True)
            ]

        return DuckPDExpr(
            call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )


class DuckPDExprStringNamespace:
    """Narwhals string operations backed by DuckPD scalar expressions."""

    def __init__(self, expression: DuckPDExpr) -> None:
        self._expression = expression

    def to_uppercase(self) -> DuckPDExpr:
        return self._expression._elementwise(
            lambda series: _require_string(series).str.upper()
        )

    def to_lowercase(self) -> DuckPDExpr:
        return self._expression._elementwise(
            lambda series: _require_string(series).str.lower()
        )

    def strip_chars(self, characters: str | None) -> DuckPDExpr:
        from duckpd._logical import LiteralValue

        chars = string.whitespace if characters is None else characters
        return self._expression._elementwise(
            lambda series: _require_string(series)._call_function(
                "trim", LiteralValue(chars)
            )
        )

    def len_chars(self) -> DuckPDExpr:
        return self._expression._elementwise(
            lambda series: _require_string(series).str.len()
        )

    def starts_with(self, prefix: DuckPDExpr) -> DuckPDExpr:
        return self._expression._binary(
            prefix,
            lambda series, value: _require_string(series)._call_function(
                "starts_with", _require_string(value)._expression
            ),
        )

    def ends_with(self, suffix: DuckPDExpr) -> DuckPDExpr:
        return self._expression._binary(
            suffix,
            lambda series, value: _require_string(series)._call_function(
                "ends_with", _require_string(value)._expression
            ),
        )

    def contains(self, pattern: DuckPDExpr, *, literal: bool) -> DuckPDExpr:
        function = "contains" if literal else "regexp_matches"
        return self._expression._binary(
            pattern,
            lambda series, value: _require_string(series)._call_function(
                function, _require_string(value)._expression
            ),
        )

    def replace(
        self,
        value: DuckPDExpr,
        *,
        pattern: str,
        literal: bool,
        n: int,
    ) -> DuckPDExpr:
        from duckpd._logical import LiteralValue
        from duckpd.errors import UnsupportedOperationError

        if n != 1:
            raise UnsupportedOperationError(
                "DuckPD Narwhals str.replace currently supports only n=1"
            )
        regex_pattern = re.escape(pattern) if literal else pattern
        return self._expression._binary(
            value,
            lambda series, replacement: _require_string(series)._call_function(
                "regexp_replace",
                LiteralValue(regex_pattern),
                _require_string(replacement)._expression,
            ),
        )

    def replace_all(
        self,
        value: DuckPDExpr,
        *,
        pattern: str,
        literal: bool,
    ) -> DuckPDExpr:
        from duckpd._logical import LiteralValue

        regex_pattern = re.escape(pattern) if literal else pattern
        return self._expression._binary(
            value,
            lambda series, replacement: _require_string(series)._call_function(
                "regexp_replace",
                LiteralValue(regex_pattern),
                _require_string(replacement)._expression,
                LiteralValue("g"),
            ),
        )


class DuckPDExprDatetimeNamespace:
    """Narwhals datetime fields backed by DuckPD scalar expressions."""

    def __init__(self, expression: DuckPDExpr) -> None:
        self._expression = expression

    def year(self) -> DuckPDExpr:
        return self._date_field("year")

    def month(self) -> DuckPDExpr:
        return self._date_field("month")

    def day(self) -> DuckPDExpr:
        return self._date_field("day")

    def hour(self) -> DuckPDExpr:
        return self._time_field("hour")

    def minute(self) -> DuckPDExpr:
        return self._time_field("minute")

    def second(self) -> DuckPDExpr:
        return self._time_field("second")

    def date(self) -> DuckPDExpr:
        return self._expression._elementwise(
            lambda series: _require_temporal(
                series, allow_date=True, allow_time=False
            ).astype("DATE")
        )

    def to_string(self, format: str) -> DuckPDExpr:
        return self._expression._elementwise(
            lambda series: _require_temporal(
                series, allow_date=True, allow_time=True
            ).dt.strftime(format)
        )

    def _date_field(self, field: str) -> DuckPDExpr:
        return self._expression._elementwise(
            lambda series: getattr(
                _require_temporal(series, allow_date=True, allow_time=False).dt,
                field,
            )
        )

    def _time_field(self, field: str) -> DuckPDExpr:
        return self._expression._elementwise(
            lambda series: getattr(
                _require_temporal(series, allow_date=False, allow_time=True).dt,
                field,
            )
        )


def _require_string(series: Series) -> Series:
    from narwhals.exceptions import InvalidOperationError

    from duckpd._reductions import expression_type

    dtype = expression_type(series._plan, series._expression)
    if dtype not in {"VARCHAR", "UNKNOWN"}:
        raise InvalidOperationError(
            f"String operation requires VARCHAR input, found {dtype}"
        )
    return series


def _require_temporal(
    series: Series,
    *,
    allow_date: bool,
    allow_time: bool,
) -> Series:
    from narwhals.exceptions import InvalidOperationError

    from duckpd._reductions import expression_type

    dtype = expression_type(series._plan, series._expression)
    supported = dtype == "UNKNOWN" or dtype.startswith("TIMESTAMP")
    supported = supported or (allow_date and dtype == "DATE")
    supported = supported or (allow_time and dtype.startswith("TIME"))
    if not supported:
        raise InvalidOperationError(
            f"Datetime operation requires temporal input, found {dtype}"
        )
    return series


def _narwhals_dtype_to_duckdb(dtype: IntoDType) -> str:
    base_name = dtype.base_type().__name__
    scalar_types = {
        "Boolean": "BOOLEAN",
        "Int8": "TINYINT",
        "Int16": "SMALLINT",
        "Int32": "INTEGER",
        "Int64": "BIGINT",
        "UInt8": "UTINYINT",
        "UInt16": "USMALLINT",
        "UInt32": "UINTEGER",
        "UInt64": "UBIGINT",
        "Float32": "FLOAT",
        "Float64": "DOUBLE",
        "String": "VARCHAR",
        "Binary": "BLOB",
        "Date": "DATE",
        "Time": "TIME",
    }
    if target := scalar_types.get(base_name):
        return target
    raise NotImplementedError(f"Casting to Narwhals dtype {dtype!r} is not supported")


def _fixed_names(names: Sequence[str]) -> Callable[[DuckPDLazyFrame], Sequence[str]]:
    return lambda _frame: names


def _literal_name(_frame: DuckPDLazyFrame) -> Sequence[str]:
    return ["literal"]


def _evaluate_exprs(
    frame: DuckPDLazyFrame, expressions: Sequence[DuckPDExpr]
) -> dict[str, Series]:
    evaluated: list[tuple[str, Series]] = []
    for expression in expressions:
        values = expression(frame)
        names = expression._evaluate_output_names(frame)
        if expression._alias_output_names is not None:
            names = expression._alias_output_names(names)
        if len(names) != len(values):
            raise AssertionError("Narwhals expression output names do not match values")
        evaluated.extend(zip(names, values, strict=True))

    output_names = [name for name, _value in evaluated]
    if len(output_names) != len(set(output_names)):
        from narwhals.exceptions import DuplicateError

        raise DuplicateError(f"Expected unique output names, got: {output_names}")
    return dict(evaluated)


def _evaluate_aggregations(
    frame: DuckPDLazyFrame,
    expressions: Sequence[DuckPDExpr],
) -> list[tuple[str, Series, str, int]]:
    evaluated: list[tuple[str, Series, str, int]] = []
    for expression in expressions:
        if expression._aggregation is None:
            from narwhals.exceptions import InvalidOperationError

            raise InvalidOperationError(
                "DuckPD aggregate requires aggregation expressions"
            )
        function, ddof = expression._aggregation
        evaluated.extend(
            (name, series, function, ddof)
            for name, series in _evaluate_exprs(frame, (expression,)).items()
        )
    output_names = [name for name, _series, _function, _ddof in evaluated]
    if len(output_names) != len(set(output_names)):
        from narwhals.exceptions import DuplicateError

        raise DuplicateError(f"Expected unique output names, got: {output_names}")
    return evaluated


class DuckPDNamespace:
    """Narwhals plugin namespace for constructing compliant lazy frames."""

    def __init__(self, version: Version) -> None:
        from narwhals._utils import Implementation

        self._version = version
        self._implementation = Implementation.UNKNOWN

    def col(self, *names: str) -> DuckPDExpr:
        def call(frame: DuckPDLazyFrame) -> Sequence[Series]:
            try:
                return [frame.native[name] for name in names]
            except KeyError as error:
                from narwhals.exceptions import ColumnNotFoundError

                raise ColumnNotFoundError(
                    f"Column not found: {error.args[0]!r}"
                ) from None

        return DuckPDExpr(
            call,
            evaluate_output_names=_fixed_names(names),
            alias_output_names=None,
            version=self._version,
        )

    def lit(self, value: object, dtype: IntoDType | None) -> DuckPDExpr:
        def call(frame: DuckPDLazyFrame) -> Sequence[Series]:
            from duckpd.series import Series

            expression = frame.native._coerce_expression(value)
            result = Series(
                frame.native._session,
                frame.native._plan,
                expression,
                "literal",
            )
            return (
                [result.astype(_narwhals_dtype_to_duckdb(dtype))] if dtype else [result]
            )

        return DuckPDExpr(
            call,
            evaluate_output_names=_literal_name,
            alias_output_names=None,
            version=self._version,
        )

    def all_horizontal(
        self,
        *expressions: DuckPDExpr,
        ignore_nulls: bool,
    ) -> DuckPDExpr:
        if ignore_nulls:
            from duckpd.errors import UnsupportedOperationError

            raise UnsupportedOperationError(
                "DuckPD Narwhals expressions do not support "
                "all_horizontal(ignore_nulls=True)"
            )
        if not expressions:
            return self.lit(True, None)
        result = expressions[0]
        for expression in expressions[1:]:
            result = result & expression
        return result

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

    def _evaluate_single_output_expr(self, expression: DuckPDExpr) -> Series:
        values = expression(self)
        if len(values) != 1:
            from narwhals.exceptions import MultiOutputExpressionError

            raise MultiOutputExpressionError(
                f"Expected one expression output, found {len(values)}"
            )
        return values[0]

    @staticmethod
    def _reject_aggregation_broadcast(
        expressions: Sequence[DuckPDExpr],
        *,
        operation: str,
    ) -> None:
        if any(expression._aggregation is not None for expression in expressions):
            from duckpd.errors import UnsupportedOperationError

            raise UnsupportedOperationError(
                f"DuckPD Narwhals {operation} does not yet broadcast aggregate "
                "expressions; use select() with only aggregations"
            )

    def select(self, *expressions: DuckPDExpr) -> DuckPDLazyFrame:
        self._reject_aggregation_broadcast(expressions, operation="select")
        outputs = _evaluate_exprs(self, expressions)
        projected = self._native_frame.assign(**outputs)
        return self._with_native(projected[list(outputs)])

    def aggregate(self, *expressions: DuckPDExpr) -> DuckPDLazyFrame:
        from duckpd._logical import (
            AggregateExpression,
            AggregateOperator,
            AggregatePlan,
            Column,
            ColumnId,
        )
        from duckpd._metadata import after_aggregate
        from duckpd._reductions import expression_type, is_numeric_type
        from duckpd.errors import UnsupportedOperationError
        from duckpd.frame import DataFrame

        operator_by_name = {
            "sum": AggregateOperator.SUM,
            "min": AggregateOperator.MIN,
            "max": AggregateOperator.MAX,
            "mean": AggregateOperator.MEAN,
            "count": AggregateOperator.COUNT,
            "size": AggregateOperator.SIZE,
            "median": AggregateOperator.MEDIAN,
            "std": AggregateOperator.STD,
            "var": AggregateOperator.VAR,
        }
        aggregates: list[AggregateExpression] = []
        for name, series, function, ddof in _evaluate_aggregations(self, expressions):
            operator = operator_by_name[function]
            input_type = expression_type(self._native_frame._plan, series._expression)
            if operator not in {AggregateOperator.COUNT, AggregateOperator.SIZE} and (
                not is_numeric_type(input_type)
            ):
                raise UnsupportedOperationError(
                    f"Narwhals {function} currently supports only numeric data; "
                    f"column {name!r} has DuckDB type {input_type}"
                )
            output_type = (
                "BIGINT"
                if operator in {AggregateOperator.COUNT, AggregateOperator.SIZE}
                else (
                    "DOUBLE"
                    if operator
                    in {
                        AggregateOperator.MEAN,
                        AggregateOperator.MEDIAN,
                        AggregateOperator.STD,
                        AggregateOperator.VAR,
                    }
                    else input_type
                )
            )
            output = Column(ColumnId.create(), name, output_type)
            aggregates.append(
                AggregateExpression(
                    output,
                    operator,
                    None if operator is AggregateOperator.SIZE else series._expression,
                    None if operator is AggregateOperator.SIZE else input_type,
                    ddof=ddof,
                )
            )
        columns = tuple(aggregate.column for aggregate in aggregates)
        plan = AggregatePlan(
            self._native_frame._plan,
            tuple(aggregates),
            after_aggregate(columns),
        )
        return self._with_native(DataFrame(self._native_frame._session, plan))

    def with_columns(self, *expressions: DuckPDExpr) -> DuckPDLazyFrame:
        self._reject_aggregation_broadcast(expressions, operation="with_columns")
        outputs = _evaluate_exprs(self, expressions)
        return self._with_native(self._native_frame.assign(**outputs))

    def filter(self, predicate: DuckPDExpr) -> DuckPDLazyFrame:
        return self._with_native(
            self._native_frame[self._evaluate_single_output_expr(predicate)]
        )

    def group_by(
        self,
        keys: Sequence[str] | Sequence[DuckPDExpr],
        *,
        drop_null_keys: bool,
    ) -> DuckPDLazyGroupBy:
        return DuckPDLazyGroupBy(self, keys, drop_null_keys=drop_null_keys)

    def head(self, n: int) -> DuckPDLazyFrame:
        return self._with_native(self._native_frame.limit(n))

    def drop(self, columns: Sequence[str], *, strict: bool) -> DuckPDLazyFrame:
        if strict:
            missing = sorted(set(columns) - set(self.columns))
            if missing:
                from narwhals.exceptions import ColumnNotFoundError

                raise ColumnNotFoundError(f"Columns not found: {missing}")
        retained = [name for name in self.columns if name not in columns]
        return self._with_native(self._native_frame[retained])

    def rename(self, mapping: Mapping[str, str]) -> DuckPDLazyFrame:
        existing = set(self.columns)
        applicable = {old: new for old, new in mapping.items() if old in existing}
        if not applicable:
            return self._with_native(self._native_frame)
        return self._with_native(self._native_frame.rename(columns=applicable))

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


class DuckPDLazyGroupBy:
    """Narwhals lazy group-by translated to DuckPD aggregate plans."""

    def __init__(
        self,
        frame: DuckPDLazyFrame,
        keys: Sequence[str] | Sequence[DuckPDExpr],
        *,
        drop_null_keys: bool,
    ) -> None:
        self._frame = frame
        self._drop_null_keys = drop_null_keys
        if all(isinstance(key, str) for key in keys):
            self._key_names = [cast("str", key) for key in keys]
            return
        if not all(isinstance(key, DuckPDExpr) for key in keys):
            raise TypeError("Narwhals group-by keys must all be strings or expressions")
        key_expressions = tuple(cast("DuckPDExpr", key) for key in keys)
        outputs = _evaluate_exprs(frame, key_expressions)
        self._frame = frame._with_native(frame.native.assign(**outputs))
        self._key_names = list(outputs)

    def agg(self, *expressions: DuckPDExpr) -> DuckPDLazyFrame:
        evaluated = _evaluate_aggregations(self._frame, expressions)
        temporary: dict[str, Series] = {}
        specifications: dict[str, tuple[str, str]] = {}
        occupied = set(self._frame.columns)
        ddof_by_name: dict[str, int] = {}
        for position, (name, series, function, ddof) in enumerate(evaluated):
            temporary_name = f"__duckpd_narwhals_agg_{position}__"
            while temporary_name in occupied:
                temporary_name = f"_{temporary_name}"
            occupied.add(temporary_name)
            temporary[temporary_name] = series
            specifications[name] = (temporary_name, function)
            ddof_by_name[name] = ddof

        prepared = self._frame.native.assign(**temporary)
        result = prepared.groupby(
            self._key_names,
            as_index=False,
            sort=False,
            dropna=self._drop_null_keys,
        ).agg(**specifications)
        if any(ddof != 1 for ddof in ddof_by_name.values()):
            from duckpd._logical import AggregatePlan
            from duckpd.frame import DataFrame

            if not isinstance(result._plan, AggregatePlan):
                raise AssertionError("Grouped aggregation did not build AggregatePlan")
            aggregates = tuple(
                replace(
                    aggregate,
                    ddof=ddof_by_name.get(aggregate.column.label, aggregate.ddof),
                )
                for aggregate in result._plan.aggregates
            )
            result = DataFrame(
                result._session,
                replace(result._plan, aggregates=aggregates),
            )
        return self._frame._with_native(result)
