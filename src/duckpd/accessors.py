"""String and Datetime accessor implementations for DuckPD Series."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Literal

import pandas as pd

from duckpd._logical import (
    BinaryExpression,
    BinaryOperator,
    CaseWhen,
    CastExpression,
    LiteralValue,
)
from duckpd._reductions import (
    expression_categorical,
    expression_timezone,
    expression_type,
    is_timestamp_type,
    is_timezone_aware_type,
)
from duckpd._temporal import fixed_duration_ns, timezone_name
from duckpd.errors import UnsupportedOperationError

if TYPE_CHECKING:
    from duckpd.series import Series


class StringMethods:
    """Vectorized string functions for DuckPD Series."""

    def __init__(self, series: Series) -> None:
        self._series = series

    def upper(self) -> Series:
        """Convert strings to uppercase."""
        return self._series._call_function("upper")

    def lower(self) -> Series:
        """Convert strings to lowercase."""
        return self._series._call_function("lower")

    def strip(self) -> Series:
        """Strip whitespace from each string."""
        return self._series._call_function("trim")

    def len(self) -> Series:
        """Return length of each string."""
        return self._series._call_function("length")

    def startswith(self, pat: str, na: bool = False) -> Series:
        """Test if the start of each string element matches a pattern."""
        if na:
            res = self._series._call_function("starts_with", LiteralValue(pat))
            return res._call_function("coalesce", LiteralValue(True))
        res = self._series._call_function("starts_with", LiteralValue(pat))
        return res._call_function("coalesce", LiteralValue(False))

    def endswith(self, pat: str, na: bool = False) -> Series:
        """Test if the end of each string element matches a pattern."""
        if na:
            res = self._series._call_function("ends_with", LiteralValue(pat))
            return res._call_function("coalesce", LiteralValue(True))
        res = self._series._call_function("ends_with", LiteralValue(pat))
        return res._call_function("coalesce", LiteralValue(False))

    def contains(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: bool = False,
        regex: bool = True,
    ) -> Series:
        """Test if pattern or regex is contained within a string of a Series."""
        if flags != 0:
            raise UnsupportedOperationError(
                "DuckPD str.contains does not currently support regex flags"
            )
        if regex:
            arguments = [LiteralValue(pat)]
            if not case:
                arguments.append(LiteralValue("i"))
            res = self._series._call_function("regexp_matches", *arguments)
        elif case:
            res = self._series._call_function("contains", LiteralValue(pat))
        else:
            res = self._series._call_function("lower")._call_function(
                "contains", LiteralValue(pat.lower())
            )
        if na:
            return res._call_function("coalesce", LiteralValue(True))
        return res._call_function("coalesce", LiteralValue(False))

    def replace(self, pat: str, repl: str) -> Series:
        """Replace occurrences of pattern/regex in the Series with some other string."""
        return self._series._call_function("replace", LiteralValue(pat), LiteralValue(repl))


class CategoricalMethods:
    """Metadata-backed operations for a categorically typed Series."""

    def __init__(self, series: Series) -> None:
        self._series = series
        spec = expression_categorical(series._plan, series._expression)
        if spec is None:
            raise AttributeError("Can only use .cat accessor with a categorical Series")
        self._spec = spec

    @property
    def categories(self) -> pd.Index:
        """Return the complete category universe without executing rows."""
        return pd.Index(self._spec.categories)

    @property
    def ordered(self) -> bool:
        """Whether comparisons use the declared category order."""
        return self._spec.ordered

    @property
    def codes(self) -> Series:
        """Return zero-based category codes and -1 for missing values."""
        from duckpd.series import Series

        expression = LiteralValue(-1)
        for code, category in reversed(tuple(enumerate(self._spec.categories))):
            expression = CaseWhen(
                BinaryExpression(
                    self._series._expression,
                    BinaryOperator.EQUAL,
                    LiteralValue(category),
                ),
                LiteralValue(code),
                expression,
            )
        category_count = len(self._spec.categories)
        codes_type = (
            "TINYINT"
            if category_count <= 127
            else "SMALLINT"
            if category_count <= 32_767
            else "INTEGER"
        )
        return Series(
            self._series._session,
            self._series._plan,
            CastExpression(expression, codes_type),
            None,
        )

    def as_ordered(self) -> Series:
        """Return the same values with ordered categorical metadata."""
        return self._series._call_function(
            "__duckpd_category_ordered",
            LiteralValue(True),
            return_type=expression_type(self._series._plan, self._series._expression),
        )

    def as_unordered(self) -> Series:
        """Return the same values with unordered categorical metadata."""
        return self._series._call_function(
            "__duckpd_category_ordered",
            LiteralValue(False),
            return_type=expression_type(self._series._plan, self._series._expression),
        )


class DatetimeProperties:
    """Accessor object for datetimelike properties of the Series values."""

    def __init__(self, series: Series) -> None:
        self._series = series
        self._type = expression_type(series._plan, series._expression)
        if not is_timestamp_type(self._type):
            raise AttributeError("Can only use .dt accessor with datetimelike values")
        self._timezone = expression_timezone(series._plan, series._expression)
        if is_timezone_aware_type(self._type):
            self._timezone = self._timezone or "UTC"

    def floor(
        self,
        freq: str | timedelta,
        *,
        ambiguous: str = "raise",
        nonexistent: str = "raise",
    ) -> Series:
        """Floor timestamps to a positive fixed-duration boundary."""
        return self._round_fixed(freq, "floor", ambiguous, nonexistent)

    def ceil(
        self,
        freq: str | timedelta,
        *,
        ambiguous: str = "raise",
        nonexistent: str = "raise",
    ) -> Series:
        """Ceil timestamps to a positive fixed-duration boundary."""
        return self._round_fixed(freq, "ceil", ambiguous, nonexistent)

    def round(
        self,
        freq: str | timedelta,
        *,
        ambiguous: str = "raise",
        nonexistent: str = "raise",
    ) -> Series:
        """Round timestamps to the nearest fixed boundary using half-even ties."""
        return self._round_fixed(freq, "round", ambiguous, nonexistent)

    def _round_fixed(
        self,
        freq: str | timedelta,
        mode: Literal["floor", "ceil", "round"],
        ambiguous: str,
        nonexistent: str,
    ) -> Series:
        if ambiguous != "raise" or nonexistent != "raise":
            raise UnsupportedOperationError(
                "DuckPD temporal rounding currently supports only "
                "ambiguous='raise' and nonexistent='raise'"
            )
        if self._timezone not in {None, "UTC", "Etc/UTC"}:
            raise UnsupportedOperationError(
                "Round non-UTC timestamps after .dt.tz_convert('UTC'); "
                "local-wall-time DST rounding is not yet supported"
            )
        duration_ns = fixed_duration_ns(freq, parameter="freq")
        return self._series._call_function(
            "__duckpd_temporal_round",
            LiteralValue(duration_ns),
            LiteralValue(mode),
            return_type=self._type,
        )

    def tz_convert(self, tz: str) -> Series:
        """Convert aware timestamps to another validated IANA timezone."""
        if not is_timezone_aware_type(self._type):
            raise TypeError("Cannot convert tz-naive timestamps; use tz_localize first")
        target = timezone_name(tz)
        return self._series._call_function(
            "__duckpd_tz_convert",
            LiteralValue(target),
            return_type="TIMESTAMP WITH TIME ZONE",
        )

    def tz_localize(
        self,
        tz: str | None,
        *,
        ambiguous: str = "raise",
        nonexistent: str = "raise",
    ) -> Series:
        """Localize naive timestamps to UTC or remove an existing timezone."""
        if ambiguous != "raise" or nonexistent != "raise":
            raise UnsupportedOperationError(
                "DuckPD tz_localize currently supports only "
                "ambiguous='raise' and nonexistent='raise'"
            )
        aware = is_timezone_aware_type(self._type)
        if tz is None:
            if not aware:
                raise TypeError("Cannot localize tz-naive timestamps with tz=None")
            return self._series._call_function(
                "__duckpd_tz_delocalize",
                LiteralValue(self._timezone or "UTC"),
                return_type="TIMESTAMP",
            )
        target = timezone_name(tz)
        if aware:
            raise TypeError("Already tz-aware; use tz_convert to convert timezones")
        if target not in {"UTC", "Etc/UTC"}:
            raise UnsupportedOperationError(
                "DuckPD tz_localize currently supports UTC only; local timezone "
                "DST ambiguity is not represented by DuckDB"
            )
        return self._series._call_function(
            "__duckpd_tz_localize_utc",
            LiteralValue("UTC"),
            return_type="TIMESTAMP WITH TIME ZONE",
        )

    @property
    def year(self) -> Series:
        """The year of the datetime."""
        return self._series._call_function("year")

    @property
    def month(self) -> Series:
        """The month as January=1, December=12."""
        return self._series._call_function("month")

    @property
    def day(self) -> Series:
        """The days of the datetime."""
        return self._series._call_function("day")

    @property
    def hour(self) -> Series:
        """The hours of the datetime."""
        return self._series._call_function("hour")

    @property
    def minute(self) -> Series:
        """The minutes of the datetime."""
        return self._series._call_function("minute")

    @property
    def second(self) -> Series:
        """The seconds of the datetime."""
        return self._series._call_function("second")

    @property
    def date(self) -> Series:
        """Returns a Series of python datetime.date objects or SQL DATE cast."""
        return self._series._call_function("cast", LiteralValue("DATE"))

    def strftime(self, date_format: str) -> Series:
        """Format datetime using specified format string."""
        return self._series._call_function("strftime", LiteralValue(date_format))

    def to_period(self, freq: Literal["Y", "M", "D"] = "M") -> Series:
        """Cast datetime to period representation (as period string like 'YYYY-MM')."""
        if freq == "M":
            return self.strftime("%Y-%m")
        if freq == "Y":
            return self.strftime("%Y")
        if freq == "D":
            return self.strftime("%Y-%m-%d")
        raise UnsupportedOperationError(
            f"to_period currently supports 'Y', 'M', 'D'; got freq={freq!r}"
        )
