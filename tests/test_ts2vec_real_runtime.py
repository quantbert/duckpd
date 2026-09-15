from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

import duckpd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_GOLDEN_PATH = Path(__file__).with_name("data") / "ts2vec-real-runtime-golden.json"

pytestmark = pytest.mark.skipif(
    os.environ.get("DUCKPD_TS2VEC_REAL_TESTS") != "1",
    reason="set DUCKPD_TS2VEC_REAL_TESTS=1 to run the pinned optional runtime",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_golden() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(_GOLDEN_PATH.read_text(encoding="utf-8")))


def _write_training_dataset(path: Path) -> None:
    timestamps: list[datetime] = []
    tickers: list[str] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    start = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    for ticker_index, ticker in enumerate(("AAA", "BBB")):
        for index in range(73):
            phase = index + ticker_index * 0.375
            open_value = 100.0 + ticker_index * 17.0 + index * 0.08 + np.sin(phase / 5.0)
            close_value = open_value * (1.0 + 0.0025 * np.sin(phase / 3.0 + 0.2))
            high_value = max(open_value, close_value) * (1.0015 + 0.0002 * np.cos(phase))
            low_value = min(open_value, close_value) * (0.9985 - 0.0001 * np.sin(phase))
            timestamps.append(start + timedelta(minutes=index))
            tickers.append(ticker)
            opens.append(open_value)
            highs.append(high_value)
            lows.append(low_value)
            closes.append(close_value)
    table = pa.table(
        {
            "datetime": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
            "ticker": tickers,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )
    parquet.write_table(  # pyright: ignore[reportUnknownMemberType]
        table,
        path,
        compression="zstd",
        use_dictionary=False,
    )


def _train_bundle(dataset: Path, output: Path, training: dict[str, Any]) -> None:
    arguments = [
        sys.executable,
        str(_REPOSITORY_ROOT / "demo" / "train_ts2vec.py"),
        "--dataset",
        str(dataset),
        "--output",
        str(output),
    ]
    for name, value in training.items():
        arguments.extend((f"--{name.replace('_', '-')}", str(value)))
    subprocess.run(
        arguments,
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_ts2vec_training_export_and_inference_match_pinned_golden(
    tmp_path: Path,
) -> None:
    golden = _load_golden()
    runtime = cast("dict[str, str]", golden["runtime"])
    assert platform.system() == runtime["system"]
    assert platform.machine() == runtime["machine"]
    assert platform.python_version() == runtime["python"]
    assert np.__version__ == runtime["numpy"]
    assert pa.__version__ == runtime["pyarrow"]
    torch = cast("Any", importlib.import_module("torch"))
    assert str(torch.__version__) == runtime["torch"]
    assert distribution_version("safetensors") == runtime["safetensors"]

    dataset = tmp_path / "ts2vec-golden.parquet"
    _write_training_dataset(dataset)
    expected = cast("dict[str, Any]", golden["expected"])
    assert _sha256(dataset) == expected["dataset_sha256"]

    training = cast("dict[str, Any]", golden["training"])
    first = tmp_path / "first"
    second = tmp_path / "second"
    _train_bundle(dataset, first, training)
    _train_bundle(dataset, second, training)

    first_weights = first / "model.safetensors"
    second_weights = second / "model.safetensors"
    first_manifest = first / "duckpd-ts2vec-manifest.json"
    second_manifest = second / "duckpd-ts2vec-manifest.json"
    assert first_weights.read_bytes() == second_weights.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert _sha256(first_weights) == expected["weights_sha256"]
    assert _sha256(first_manifest) == expected["manifest_sha256"]

    manifest = cast(
        "dict[str, Any]",
        json.loads(first_manifest.read_text(encoding="utf-8")),
    )
    assert manifest["training"]["losses"] == expected["losses"]
    model = duckpd.ts2vec_series_embedding_model(first)
    assert model.fingerprint == expected["model_fingerprint"]
    representation = duckpd.series_representation(
        window=model.input_length,
        channels=model.input_channels,
        sampling="observations",
        data_contract="demo/synthetic-ohlc/ts2vec-v1",
        encoder=model,
    )
    query_input = cast("dict[str, list[float]]", golden["query"])

    with duckpd.connect() as session:
        session.prepare_series_embedding_model(model, artifact_dir=first)
        query = session.embed_series_query(query_input, representation=representation)
        source = session.from_pandas(
            pd.DataFrame({"row": range(model.input_length), **query_input}),
            order_by="row",
        )
        windows = source.assign(
            close_window=lambda frame: frame["close_return"].rolling(model.input_length).to_array(),
            bar_window=lambda frame: frame["bar_return"].rolling(model.input_length).to_array(),
            range_window=lambda frame: (
                frame["intrabar_range"].rolling(model.input_length).to_array()
            ),
        )
        embedded = windows.embed_series(
            columns={
                "close_return": "close_window",
                "bar_return": "bar_window",
                "intrabar_range": "range_window",
            },
            into="vector",
            representation=representation,
            batch_size=1,
        )
        corpus = embedded[embedded["row"] == model.input_length - 1].collect()

    values = np.asarray(query.values, dtype="<f4")
    expected_values = np.asarray(expected["embedding"], dtype="<f4")
    np.testing.assert_array_equal(values, expected_values)
    assert hashlib.sha256(values.tobytes()).hexdigest() == expected["embedding_sha256"]
    np.testing.assert_array_equal(corpus["vector"].iloc[0], values)
