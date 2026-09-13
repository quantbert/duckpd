"""Shared catalog identity for deterministic generated series vectors."""

from __future__ import annotations

from typing import Any

RETURN_SHAPE_KEY = "simple-return-shape-8-1m"
RETURN_SHAPE_WINDOW = 8


def series_representations() -> dict[str, dict[str, Any]]:
    """Return the complete compact catalog registry for generated series vectors."""
    return {
        RETURN_SHAPE_KEY: {
            "version": 1,
            "window": RETURN_SHAPE_WINDOW,
            "channels": ["simple_return"],
            "sampling": "fixed_grid",
            "step": "PT1M",
            "data_contract": "simple-return-left-zero-padded/v1",
            "normalization": "none",
            "unit_norm": False,
            "zero_scale": "null",
            "encoder": None,
        }
    }
