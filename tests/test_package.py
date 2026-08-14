from __future__ import annotations

from importlib.metadata import version

import duckpd


def test_package_has_version() -> None:
    assert duckpd.__version__ == version("duckpd")
