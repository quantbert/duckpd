"""Experimental Narwhals plugin for DuckPD lazy frames."""

from __future__ import annotations

import math
import re
import string
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from io import BytesIO
from pathlib import Path
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
            raise AssertionError("Narwhals expression metadata has not been initialized")
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

    def over(
        self,
        partition_by: Sequence[str],
        order_by: Sequence[str],
    ) -> DuckPDExpr:
        if partition_by:
            from duckpd.errors import UnsupportedOperationError

            raise UnsupportedOperationError("DuckPD Narwhals partitioned windows are not supported")

        def call(frame: DuckPDLazyFrame) -> Sequence[Series]:
            from duckpd.errors import UnorderedOperationError

            metadata = frame.native._plan.metadata
            labels_by_id = {column.id: column.label for column in metadata.columns}
            actual = tuple(labels_by_id[key.column_id] for key in metadata.ordering.keys)
            expected = tuple(order_by)
            if not expected or actual[: len(expected)] != expected:
                raise UnorderedOperationError(
                    "Narwhals window order_by must match DuckPD's guaranteed "
                    f"ordering; expected {expected!r}, found {actual!r}"
                )
            return self(frame)

        return DuckPDExpr(
            call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
            aggregation=self._aggregation,
        )

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

    def abs(self) -> DuckPDExpr:
        return self._elementwise(lambda series: series._call_function("abs"))

    def round(self, decimals: int) -> DuckPDExpr:
        from duckpd._logical import CastExpression, LiteralValue

        return self._elementwise(
            lambda series: series._call_function(
                "round",
                CastExpression(LiteralValue(decimals), "INTEGER"),
            )
        )

    def floor(self) -> DuckPDExpr:
        return self._elementwise(lambda series: series._call_function("floor"))

    def ceil(self) -> DuckPDExpr:
        return self._elementwise(lambda series: series._call_function("ceil"))

    def exp(self) -> DuckPDExpr:
        return self._elementwise(lambda series: series._call_function("exp"))

    def sqrt(self) -> DuckPDExpr:
        return self._elementwise(lambda series: series._call_function("sqrt"))

    def cos(self) -> DuckPDExpr:
        return self._elementwise(lambda series: series._call_function("cos"))

    def sin(self) -> DuckPDExpr:
        return self._elementwise(lambda series: series._call_function("sin"))

    def log(self, base: float) -> DuckPDExpr:
        if not math.isfinite(base) or base <= 0 or base == 1:
            raise ValueError("log base must be finite, positive, and not equal to 1")
        denominator = math.log(base)
        return self._elementwise(lambda series: series._call_function("ln") / denominator)

    def clip(self, lower_bound: DuckPDExpr, upper_bound: DuckPDExpr) -> DuckPDExpr:
        return self.clip_lower(lower_bound).clip_upper(upper_bound)

    def clip_lower(self, lower_bound: DuckPDExpr) -> DuckPDExpr:
        return self._binary(lower_bound, lambda series, lower: series.clip(lower=lower))

    def clip_upper(self, upper_bound: DuckPDExpr) -> DuckPDExpr:
        return self._binary(upper_bound, lambda series, upper: series.clip(upper=upper))

    def is_finite(self) -> DuckPDExpr:
        return self._elementwise(lambda series: series._call_function("isfinite"))

    def is_nan(self) -> DuckPDExpr:
        return self._elementwise(lambda series: series._call_function("isnan"))

    def fill_nan(self, value: DuckPDExpr) -> DuckPDExpr:
        return self._binary(
            value,
            lambda series, replacement: series.where(~series._call_function("isnan"), replacement),
        )

    def fill_null(
        self,
        value: DuckPDExpr | None,
        strategy: str | None,
        limit: int | None,
    ) -> DuckPDExpr:
        if strategy is not None or limit is not None:
            from duckpd.errors import UnsupportedOperationError

            raise UnsupportedOperationError(
                "DuckPD Narwhals fill_null supports only an explicit value"
            )
        if value is None:
            raise TypeError("fill_null requires a replacement value")
        return self._binary(value, lambda series, replacement: series.fillna(replacement))

    def cum_sum(self, *, reverse: bool) -> DuckPDExpr:
        return self._ordered(lambda series: series.cumsum(), reverse=reverse, name="cum_sum")

    def cum_min(self, *, reverse: bool) -> DuckPDExpr:
        return self._ordered(lambda series: series.cummin(), reverse=reverse, name="cum_min")

    def cum_max(self, *, reverse: bool) -> DuckPDExpr:
        return self._ordered(lambda series: series.cummax(), reverse=reverse, name="cum_max")

    def cum_prod(self, *, reverse: bool) -> DuckPDExpr:
        return self._ordered(lambda series: series.cumprod(), reverse=reverse, name="cum_prod")

    def cum_count(self, *, reverse: bool) -> DuckPDExpr:
        return self._ordered(
            lambda series: series.notna().astype("BIGINT").cumsum(),
            reverse=reverse,
            name="cum_count",
        )

    def diff(self) -> DuckPDExpr:
        return self._elementwise(lambda series: series.diff())

    def shift(self, n: int) -> DuckPDExpr:
        return self._elementwise(lambda series: series.shift(n))

    def rank(self, method: str, *, descending: bool) -> DuckPDExpr:
        pandas_method = "first" if method == "ordinal" else method
        if pandas_method not in {"average", "min", "max", "dense", "first"}:
            raise ValueError(f"Unsupported rank method: {method!r}")
        return self._elementwise(
            lambda series: series.rank(
                method=cast("Any", pandas_method),
                ascending=not descending,
                na_option="keep",
            )
        )

    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> DuckPDExpr:
        return self._elementwise(
            lambda series: cast(
                "Series",
                series.rolling(window_size, min_periods=min_samples, center=center).sum(),
            )
        )

    def rolling_mean(self, window_size: int, *, min_samples: int, center: bool) -> DuckPDExpr:
        return self._elementwise(
            lambda series: cast(
                "Series",
                series.rolling(window_size, min_periods=min_samples, center=center).mean(),
            )
        )

    def rolling_var(
        self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
        ddof: int,
    ) -> DuckPDExpr:
        return self._elementwise(
            lambda series: cast(
                "Series",
                series.rolling(window_size, min_periods=min_samples, center=center).var(ddof=ddof),
            )
        )

    def rolling_std(
        self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
        ddof: int,
    ) -> DuckPDExpr:
        return self._elementwise(
            lambda series: cast(
                "Series",
                series.rolling(window_size, min_periods=min_samples, center=center).std(ddof=ddof),
            )
        )

    def _ordered(
        self,
        function: Callable[[Series], Series],
        *,
        reverse: bool,
        name: str,
    ) -> DuckPDExpr:
        if reverse:
            from duckpd.errors import UnsupportedOperationError

            raise UnsupportedOperationError(f"DuckPD Narwhals {name} does not support reverse=True")
        return self._elementwise(function)

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

    def __radd__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: right + left).alias("literal")

    def __sub__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left - right)

    def __rsub__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: right - left).alias("literal")

    def __mul__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left * right)

    def __rmul__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: right * left).alias("literal")

    def __truediv__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left / right)

    def __rtruediv__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: right / left).alias("literal")

    def __mod__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: left % right)

    def __rmod__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: right % left).alias("literal")

    def __floordiv__(self, other: object) -> DuckPDExpr:
        return self._binary(other, lambda left, right: (left / right)._call_function("floor"))

    def __rfloordiv__(self, other: object) -> DuckPDExpr:
        return self._binary(
            other, lambda left, right: (right / left)._call_function("floor")
        ).alias("literal")

    def __pow__(self, other: object) -> DuckPDExpr:
        return self._binary(
            other,
            lambda left, right: left._call_function("pow", right._expression),
        )

    def __rpow__(self, other: object) -> DuckPDExpr:
        return self._binary(
            other,
            lambda left, right: right._call_function("pow", left._expression),
        ).alias("literal")

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
                self._alias_output_names if alias_output_names is None else alias_output_names
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
        def call(frame: DuckPDLazyFrame) -> Sequence[Series]:
            from duckpd.series import Series

            left = list(self(frame))
            if isinstance(other, DuckPDExpr):
                right = list(other(frame))
            else:
                right = [
                    Series(
                        frame.native._session,
                        frame.native._plan,
                        frame.native._coerce_expression(other),
                        "literal",
                    )
                ]
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
        return self._expression._elementwise(lambda series: _require_string(series).str.upper())

    def to_lowercase(self) -> DuckPDExpr:
        return self._expression._elementwise(lambda series: _require_string(series).str.lower())

    def strip_chars(self, characters: str | None) -> DuckPDExpr:
        from duckpd._logical import LiteralValue

        chars = string.whitespace if characters is None else characters
        return self._expression._elementwise(
            lambda series: _require_string(series)._call_function("trim", LiteralValue(chars))
        )

    def len_chars(self) -> DuckPDExpr:
        return self._expression._elementwise(lambda series: _require_string(series).str.len())

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
            lambda series: _require_temporal(series, allow_date=True, allow_time=False).astype(
                "DATE"
            )
        )

    def to_string(self, format: str) -> DuckPDExpr:
        return self._expression._elementwise(
            lambda series: _require_temporal(series, allow_date=True, allow_time=True).dt.strftime(
                format
            )
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
        raise InvalidOperationError(f"String operation requires VARCHAR input, found {dtype}")
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
        raise InvalidOperationError(f"Datetime operation requires temporal input, found {dtype}")
    return series


def _narwhals_dtype_to_duckdb(dtype: IntoDType) -> str:
    typed_dtype = cast("Any", dtype)
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
    if base_name == "Decimal":
        return f"DECIMAL({typed_dtype.precision}, {typed_dtype.scale})"
    if base_name == "Datetime":
        if typed_dtype.time_zone is not None:
            return "TIMESTAMP WITH TIME ZONE"
        return {
            "s": "TIMESTAMP_S",
            "ms": "TIMESTAMP_MS",
            "us": "TIMESTAMP",
            "ns": "TIMESTAMP_NS",
        }[typed_dtype.time_unit]
    if base_name == "Duration":
        return "INTERVAL"
    raise NotImplementedError(f"Casting to Narwhals dtype {dtype!r} is not supported")


def _duckdb_type_to_narwhals(dtype: str, dtypes: Any) -> DType:
    scalar_types = {
        "BOOLEAN": dtypes.Boolean(),
        "TINYINT": dtypes.Int8(),
        "SMALLINT": dtypes.Int16(),
        "INTEGER": dtypes.Int32(),
        "BIGINT": dtypes.Int64(),
        "HUGEINT": dtypes.Int128(),
        "UTINYINT": dtypes.UInt8(),
        "USMALLINT": dtypes.UInt16(),
        "UINTEGER": dtypes.UInt32(),
        "UBIGINT": dtypes.UInt64(),
        "UHUGEINT": dtypes.UInt128(),
        "FLOAT": dtypes.Float32(),
        "DOUBLE": dtypes.Float64(),
        "VARCHAR": dtypes.String(),
        "BLOB": dtypes.Binary(),
        "DATE": dtypes.Date(),
        "TIME": dtypes.Time(),
        "INTERVAL": dtypes.Duration("us"),
        "TIMESTAMP": dtypes.Datetime("us"),
        "TIMESTAMP_S": dtypes.Datetime("s"),
        "TIMESTAMP_MS": dtypes.Datetime("ms"),
        "TIMESTAMP_NS": dtypes.Datetime("ns"),
        "TIMESTAMP WITH TIME ZONE": dtypes.Datetime("us", "UTC"),
        "TIMESTAMPTZ": dtypes.Datetime("us", "UTC"),
    }
    normalized = dtype.strip().upper()
    if normalized in scalar_types:
        return scalar_types[normalized]
    decimal_match = re.fullmatch(r"DECIMAL\((\d+),\s*(\d+)\)", normalized)
    if decimal_match is not None:
        return dtypes.Decimal(
            precision=int(decimal_match.group(1)),
            scale=int(decimal_match.group(2)),
        )
    return dtypes.Unknown()


def _fixed_names(names: Sequence[str]) -> Callable[[DuckPDLazyFrame], Sequence[str]]:
    return lambda _frame: names


def _literal_name(_frame: DuckPDLazyFrame) -> Sequence[str]:
    return ["literal"]


def _evaluate_exprs(frame: DuckPDLazyFrame, expressions: Sequence[DuckPDExpr]) -> dict[str, Series]:
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

            raise InvalidOperationError("DuckPD aggregate requires aggregation expressions")
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

                raise ColumnNotFoundError(f"Column not found: {error.args[0]!r}") from None

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
            return [result.astype(_narwhals_dtype_to_duckdb(dtype))] if dtype else [result]

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
                "DuckPD Narwhals expressions do not support all_horizontal(ignore_nulls=True)"
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
        return {
            column.label: _duckdb_type_to_narwhals(column.duckdb_type, dtypes)
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

    def _iter_columns(self) -> Iterator[Series]:
        for name in self.columns:
            yield self._native_frame[name]

    def _evaluate_single_output_expr(self, expression: DuckPDExpr) -> Series:
        values = expression(self)
        if len(values) != 1:
            from narwhals.exceptions import MultiOutputExpressionError

            raise MultiOutputExpressionError(f"Expected one expression output, found {len(values)}")
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
        return self._with_native(self._native_frame[self._evaluate_single_output_expr(predicate)])

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

    def drop_nulls(self, subset: Sequence[str] | None) -> DuckPDLazyFrame:
        return self._with_native(self._native_frame.dropna(subset=subset))

    def unique(
        self,
        subset: Sequence[str] | None,
        *,
        keep: str,
        order_by: Sequence[str] | None,
    ) -> DuckPDLazyFrame:
        keep_value: str | bool = False if keep == "none" else ("first" if keep == "any" else keep)
        frame = self._native_frame.sort_values(list(order_by)) if order_by else self._native_frame
        return self._with_native(
            frame.drop_duplicates(
                subset=subset,
                keep=cast("Any", keep_value),
            )
        )

    def top_k(
        self,
        k: int,
        *,
        by: Sequence[str],
        reverse: bool | Sequence[bool],
    ) -> DuckPDLazyFrame:
        ascending = reverse if isinstance(reverse, bool) else list(reverse)
        return self._with_native(
            self._native_frame.sort_values(list(by), ascending=ascending).limit(k)
        )

    def join(
        self,
        other: DuckPDLazyFrame,
        *,
        how: str,
        left_on: Sequence[str] | None,
        right_on: Sequence[str] | None,
        suffix: str,
    ) -> DuckPDLazyFrame:
        from duckpd.errors import UnsupportedOperationError

        if how in {"semi", "anti"}:
            raise UnsupportedOperationError(f"DuckPD Narwhals join does not support how={how!r}")
        merge_how = "outer" if how == "full" else how
        matching_keys = (
            left_on is not None and right_on is not None and tuple(left_on) == tuple(right_on)
        )
        return self._with_native(
            self._native_frame.merge(
                other.native,
                how=cast("Any", merge_how),
                on=list(left_on or ()) if matching_keys else None,
                left_on=None if merge_how == "cross" or matching_keys else left_on,
                right_on=None if merge_how == "cross" or matching_keys else right_on,
                suffixes=("", suffix),
                sort=False,
            )
        )

    def join_asof(
        self,
        other: DuckPDLazyFrame,
        *,
        left_on: str,
        right_on: str,
        by_left: Sequence[str] | None,
        by_right: Sequence[str] | None,
        strategy: str,
        suffix: str,
    ) -> DuckPDLazyFrame:
        from duckpd.errors import UnsupportedOperationError

        raise UnsupportedOperationError("DuckPD Narwhals as-of joins are not supported")

    def explode(self, columns: Sequence[str]) -> DuckPDLazyFrame:
        from duckpd.errors import UnsupportedOperationError

        raise UnsupportedOperationError("DuckPD Narwhals explode requires nested dtype semantics")

    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> DuckPDLazyFrame:
        from duckpd.errors import UnsupportedOperationError

        raise UnsupportedOperationError(
            "DuckPD Narwhals unpivot is not supported by the typed plan"
        )

    def with_row_index(self, name: str, order_by: Sequence[str]) -> DuckPDLazyFrame:
        from duckpd._logical import (
            BinaryExpression,
            BinaryOperator,
            Column,
            ColumnId,
            ColumnRef,
            LiteralValue,
            NamedExpression,
            ProjectPlan,
            SortKey,
            WindowExpression,
        )
        from duckpd._metadata import after_projection
        from duckpd.frame import DataFrame

        if not order_by:
            raise TypeError("DuckPD requires order_by for with_row_index")
        if name in self.columns:
            from narwhals.exceptions import DuplicateError

            raise DuplicateError(f"Column {name!r} already exists")
        ordered = self._native_frame.sort_values(list(order_by))
        metadata = ordered._plan.metadata
        sort_keys = tuple(
            SortKey(
                ColumnRef(key.column_id),
                key.direction,
                key.null_placement,
            )
            for key in metadata.ordering.keys
        )
        output = Column(ColumnId.create(), name, "BIGINT")
        row_number = BinaryExpression(
            WindowExpression("row_number", order_by=sort_keys),
            BinaryOperator.SUBTRACT,
            LiteralValue(1),
        )
        columns = (output, *metadata.columns)
        projections = (
            NamedExpression(output, row_number),
            *(NamedExpression(column, ColumnRef(column.id)) for column in metadata.columns),
        )
        plan = ProjectPlan(
            ordered._plan,
            projections,
            after_projection(metadata, columns),
        )
        return self._with_native(DataFrame(ordered._session, plan))

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
            not descending if isinstance(descending, bool) else [not value for value in descending]
        )
        return self._with_native(
            self._native_frame.sort_values(
                list(by),
                ascending=ascending,
                na_position="last" if nulls_last else "first",
            )
        )

    def collect(self, backend: object = None, **kwargs: Any) -> object:
        from narwhals._utils import Implementation

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported collect arguments: {unexpected}")
        if backend in (None, Implementation.PYARROW, "pyarrow"):
            from narwhals._arrow.dataframe import ArrowDataFrame

            return ArrowDataFrame(
                native_dataframe=self._native_frame.to_arrow(),
                version=self._version,
                validate_column_names=True,
                validate_backend_version=True,
            )
        if backend in (Implementation.PANDAS, "pandas"):
            from narwhals._pandas_like.dataframe import PandasLikeDataFrame

            return PandasLikeDataFrame(
                self._native_frame.collect(),
                implementation=Implementation.PANDAS,
                version=self._version,
                validate_column_names=True,
                validate_backend_version=True,
            )
        if backend in (Implementation.POLARS, "polars"):
            raise ValueError(
                "DuckPD Narwhals collection does not support the optional Polars backend"
            )
        raise ValueError(f"Unsupported collect backend: {backend!r}")

    def sink_parquet(self, file: str | Path | BytesIO) -> None:
        if isinstance(file, BytesIO):
            raise TypeError("DuckPD sink_parquet requires a filesystem path")
        self._native_frame.write_parquet(file)


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
