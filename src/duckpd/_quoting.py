"""Contained SQL quoting for DuckDB API gaps."""


def quote_identifier(value: str) -> str:
    """Quote one DuckDB identifier without treating dots as qualification."""
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def quote_literal(value: str) -> str:
    """Quote one DuckDB string literal for syntax that rejects parameters."""
    return f"'{value.replace(chr(39), chr(39) * 2)}'"
