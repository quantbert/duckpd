from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


def changelog_versions() -> set[str]:
    text = (ROOT / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    pattern = r"^## (\d+\.\d+\.\d+(?:[a-z]+\d+)?) - \d{4}-\d{2}-\d{2}$"
    return set(re.findall(pattern, text, re.MULTILINE))


def verify_version() -> str:
    """Verify that project metadata has a corresponding immutable changelog entry."""
    version = project_version()
    if version not in changelog_versions():
        raise SystemExit(f"docs/CHANGELOG.md has no dated release section for {version!r}")
    return version


def verify(tag: str) -> None:
    version = verify_version()
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise SystemExit(
            f"release tag {tag!r} does not match project version {version!r}; "
            f"expected {expected_tag!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify project/changelog versions and an optional release tag."
    )
    parser.add_argument("tag", nargs="?", help="Git release tag, including the v prefix")
    args = parser.parse_args()
    if args.tag is None:
        verify_version()
    else:
        verify(args.tag)


if __name__ == "__main__":
    main()
