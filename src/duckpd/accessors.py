"""String and Datetime accessor implementations for DuckPD Series."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from duckpd._logical import LiteralValue
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


class DatetimeProperties:
    """Accessor object for datetimelike properties of the Series values."""

    def __init__(self, series: Series) -> None:
        self._series = series

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
