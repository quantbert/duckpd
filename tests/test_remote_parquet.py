from __future__ import annotations

import re
from collections.abc import Callable, Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import ClassVar, cast
from urllib.request import urlopen

import pandas as pd

import duckpd


class _RangeHandler(BaseHTTPRequestHandler):
    file_path: ClassVar[Path]
    bytes_sent: ClassVar[int] = 0

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(self.file_path.stat().st_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/__bytes__":
            payload = str(type(self).bytes_sent).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        size = self.file_path.stat().st_size
        header = self.headers.get("Range")
        start, end = 0, size - 1
        if header is not None:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
            if match is None or not any(match.groups()):
                self.send_error(416)
                return
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
            else:
                suffix_length = int(match.group(2))
                start = max(0, size - suffix_length)
                end = size - 1
            if start >= size:
                self.send_error(416)
                return
            end = min(end, size - 1)
        payload_length = max(0, end - start + 1)
        self.send_response(206 if header is not None else 200)
        self.send_header("Content-Length", str(payload_length))
        self.send_header("Accept-Ranges", "bytes")
        if header is not None:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with self.file_path.open("rb") as stream:
            stream.seek(start)
            payload = stream.read(payload_length)
        type(self).bytes_sent += len(payload)
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _run_server(path: str, ready: Connection) -> None:
    handler = type(
        "RangeHandler",
        (_RangeHandler,),
        {"file_path": Path(path), "bytes_sent": 0},
    )
    server = HTTPServer(("127.0.0.1", 0), handler)
    ready.send(cast("tuple[str, int]", server.server_address)[1])
    ready.close()
    server.serve_forever()


@contextmanager
def _serve(path: Path) -> Generator[tuple[str, Callable[[], int]], None, None]:
    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_run_server, args=(str(path), sender))
    process.start()
    sender.close()
    port = int(receiver.recv())
    receiver.close()
    base_url = f"http://127.0.0.1:{port}"

    def transferred_bytes() -> int:
        with urlopen(f"{base_url}/__bytes__") as response:
            return int(response.read())

    try:
        yield f"{base_url}/{path.name}", transferred_bytes
    finally:
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def test_http_parquet_executes_pruned_ranges_and_measures_transfer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "remote.parquet"
    rows = 12_000
    pd.DataFrame(
        {
            "category": ["skip"] * 11_000 + ["keep"] * 1_000,
            "value": range(rows),
            "unused": [f"payload-{index:08d}-" + "x" * 256 for index in range(rows)],
        }
    ).to_parquet(path, index=False, row_group_size=1_000, compression=None)

    with _serve(path) as (url, transferred_bytes), duckpd.connect() as session:
        frame = session.read_parquet(url)
        selected = frame[frame["category"] == "keep"][["value"]]
        fragment = selected.explain("json")
        assert '"pushdown_candidates": [' in fragment
        assert '"projection"' in fragment
        assert '"filter"' in fragment
        analyzed = selected.explain("analyze")

        assert "PARQUET_SCAN" in analyzed
        assert "category='keep'" in analyzed
        assert 0 < transferred_bytes() < path.stat().st_size
        assert session.execution_count == 1
