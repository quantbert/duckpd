"""DuckPD exception hierarchy."""


class DuckPDError(Exception):
    """Base class for DuckPD errors."""


class SessionClosedError(DuckPDError):
    """Raised when an operation uses a closed session."""


class UnsupportedOperationError(DuckPDError, NotImplementedError):
    """Raised when DuckPD cannot translate an operation safely."""


class UnorderedOperationError(DuckPDError):
    """Raised when an operation requires a stable row order."""


class AlignmentError(DuckPDError):
    """Raised when objects cannot be aligned without ambiguity."""


class MaterializationError(DuckPDError):
    """Raised when a result cannot be materialized safely."""


class ConcurrentModificationError(DuckPDError):
    """Raised when a source changes during a commit."""
