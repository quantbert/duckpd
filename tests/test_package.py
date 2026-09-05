from __future__ import annotations

from importlib.metadata import version

import pytest

import duckpd
from scripts.verify_release import project_version, verify, verify_version


def test_package_has_version() -> None:
    assert duckpd.__version__ == version("duckpd")


def test_release_tag_matches_project_and_changelog_version() -> None:
    verify(f"v{project_version()}")


def test_release_metadata_matches_changelog_without_tag() -> None:
    assert verify_version() == project_version()


def test_release_tag_mismatch_fails() -> None:
    with pytest.raises(SystemExit, match="does not match"):
        verify("v999999.0.0-impossible")
