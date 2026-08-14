from __future__ import annotations

import duckpd


def test_package_has_version() -> None:
    assert duckpd.__version__ == "0.0.1.dev0"
