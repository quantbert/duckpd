"""Contained SQL quoting for DuckDB API gaps."""


def quote_identifier(value: str) -> str:
    """Quote one DuckDB identifier without treating dots as qualification."""
    return f'"{value.replace(chr(34), chr(34) * 2)}"'
