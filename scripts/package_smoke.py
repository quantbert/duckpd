"""Validate release artifacts and install the wheel in a clean environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path


def _single_artifact(dist: Path, pattern: str) -> Path:
    matches = list(dist.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} artifact, found {matches}")
    return matches[0]


def _forbidden(path: str) -> bool:
    parts = set(Path(path).parts)
    return (
        "__pycache__" in parts
        or path.endswith((".pyc", ".pyo"))
        or any(part.startswith(".env") for part in parts)
        or any(
            part in {".git", ".venv", ".pytest_cache", ".ruff_cache"} for part in parts
        )
    )


def check_artifacts(root: Path, dist: Path) -> Path:
    """Validate wheel/sdist contents and return the wheel path."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    expected_version = str(project["project"]["version"])
    wheel = _single_artifact(dist, "*.whl")
    sdist = _single_artifact(dist, "*.tar.gz")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        required_suffixes = {
            "duckpd/__init__.py",
            "duckpd/_narwhals_plugin.py",
            "duckpd/py.typed",
        }
        for suffix in required_suffixes:
            if not any(name.endswith(suffix) for name in names):
                raise RuntimeError(f"Wheel is missing {suffix}")
        if any(_forbidden(name) for name in names):
            raise RuntimeError("Wheel contains cache, environment, or secret files")
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode()
        if f"Version: {expected_version}" not in metadata:
            raise RuntimeError("Wheel metadata version does not match pyproject.toml")
        entrypoint_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        entrypoints = archive.read(entrypoint_name).decode()
        if (
            "[narwhals.plugins]" not in entrypoints
            or "duckpd._narwhals_plugin" not in entrypoints
        ):
            raise RuntimeError("Wheel is missing the Narwhals plugin entry point")

    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        if any(_forbidden(name) for name in names):
            raise RuntimeError("Source distribution contains forbidden files")
        if not any(name.endswith("/src/duckpd/py.typed") for name in names):
            raise RuntimeError("Source distribution is missing py.typed")

    return wheel


def smoke_install(wheel: Path, python_spec: str) -> None:
    """Install the wheel into a clean venv and execute a native pipeline."""
    with tempfile.TemporaryDirectory(prefix="duckpd-wheel-smoke-") as directory:
        environment = Path(directory) / "venv"
        subprocess.run(
            ["uv", "venv", "--python", python_spec, str(environment)],
            check=True,
        )
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), str(wheel)],
            check=True,
        )
        code = """
from importlib.metadata import version
import duckpd
assert duckpd.__version__ == version("duckpd")
with duckpd.connect() as session:
    result = session.sql("SELECT 1 AS value UNION ALL SELECT 2").collect()
assert result["value"].tolist() == [1, 2]
"""
        subprocess.run(
            [str(python), "-c", code],
            cwd=directory,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", nargs="?", default="dist")
    parser.add_argument("--artifacts-only", action="store_true")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter used to create the clean smoke-test environment",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dist = (root / args.dist).resolve()
    wheel = check_artifacts(root, dist)
    if not args.artifacts_only:
        smoke_install(wheel, args.python)


if __name__ == "__main__":
    main()
