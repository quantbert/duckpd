"""CLI entry point when invoked as python -m benchmark."""

from __future__ import annotations

import sys

from benchmark.runner import main

if __name__ == "__main__":
    main(sys.argv[1:])
