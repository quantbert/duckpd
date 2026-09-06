"""Bump project version in pyproject.toml, sync uv.lock, and validate or prepare CHANGELOG.md."""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = ROOT / "pyproject.toml"
CHANGELOG_PATH = ROOT / "docs" / "CHANGELOG.md"

VALID_BUMP_RULES = {"major", "minor", "patch", "alpha", "beta", "rc", "post", "dev"}


def get_current_version() -> str:
    with PYPROJECT_PATH.open("rb") as f:
        return str(tomllib.load(f)["project"]["version"])


def changelog_versions() -> set[str]:
    text = CHANGELOG_PATH.read_text(encoding="utf-8")
    pattern = r"^## (\d+\.\d+\.\d+(?:[a-z]+\d+)?) - \d{4}-\d{2}-\d{2}$"
    return set(re.findall(pattern, text, re.MULTILINE))


def add_changelog_stub(new_version: str) -> None:
    text = CHANGELOG_PATH.read_text(encoding="utf-8")
    today = datetime.date.today().isoformat()
    header = f"## {new_version} - {today}\n\n### Added\n\n### Changed\n\n### Fixed\n\n"
    # Find the first release section (## <version>)
    match = re.search(r"^## \d+\.\d+\.\d+", text, re.MULTILINE)
    if match:
        pos = match.start()
        updated = text[:pos] + header + text[pos:]
    else:
        updated = text + "\n" + header
    CHANGELOG_PATH.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bump duckpd version in pyproject.toml, sync uv.lock, and verify changelog."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="patch",
        help="Bump semantic rule (patch, minor, etc.) or version string (default: patch)",
    )
    args = parser.parse_args()

    old_version = get_current_version()

    if args.target in VALID_BUMP_RULES:
        cmd = ["uv", "version", "--bump", args.target]
    else:
        cmd = ["uv", "version", args.target]

    print(f"Bumping version from {old_version} (rule/target: {args.target})...")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr or res.stdout, file=sys.stderr)
        sys.exit(res.returncode)

    new_version = get_current_version()
    print(f"Updated pyproject.toml: {old_version} -> {new_version}")

    print("Syncing lockfile with 'uv lock'...")
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True)

    # Check docs/CHANGELOG.md
    existing_versions = changelog_versions()
    if new_version not in existing_versions:
        print(f"Creating missing section in docs/CHANGELOG.md for {new_version}...")
        add_changelog_stub(new_version)
        print(f"Created template: '## {new_version} - {datetime.date.today().isoformat()}'")
    else:
        print(f"docs/CHANGELOG.md already contains release section for {new_version}.")

    # Run scripts/verify_release.py
    print("Verifying release metadata with scripts/verify_release.py...")
    verify_script = str(ROOT / "scripts" / "verify_release.py")
    subprocess.run([sys.executable, verify_script], cwd=ROOT, check=True)
    print(f"Version bump complete: {new_version}")


if __name__ == "__main__":
    main()
