"""Train and export an attested TS2Vec bundle from generated market data."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pyarrow.parquet as parquet

from duckpd._ts2vec import finalize_ts2vec_bundle

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = DEMO_DIR / "data" / "market-data-smoke.parquet"
DEFAULT_OUTPUT = DEMO_DIR / ".tmp" / "ts2vec-market-smoke"
CHANNELS = ("close_return", "bar_return", "intrabar_range")
ROLES = ("target", "past_covariate", "past_covariate")


@dataclass(frozen=True)
class Arguments:
    """Validated bounded training configuration."""

    dataset: Path
    output: Path
    max_points_per_ticker: int
    input_length: int
    window_stride: int
    output_dims: int
    hidden_dims: int
    depth: int
    iterations: int
    batch_size: int
    learning_rate: float
    temporal_unit: int
    seed: int
    force: bool


@dataclass(frozen=True)
class SeriesShard:
    """One bounded, ordered ticker shard."""

    values: np.ndarray[Any, np.dtype[np.float32]]
    first_time: str
    last_time: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bounded_shards(path: Path, max_points: int) -> dict[str, SeriesShard]:
    """Stream the Parquet file once while retaining only a fixed number per ticker."""
    values: dict[str, list[tuple[float, float, float]]] = {}
    times: dict[str, list[str]] = {}
    previous_close: dict[str, float] = {}
    source = parquet.ParquetFile(path)
    for batch in source.iter_batches(  # pyright: ignore[reportUnknownMemberType]
        batch_size=65_536,
        columns=["datetime", "ticker", "open", "high", "low", "close"],
    ):
        columns = batch.to_pydict()  # pyright: ignore[reportUnknownMemberType]
        rows = zip(
            cast("list[object]", columns["datetime"]),
            cast("list[str]", columns["ticker"]),
            cast("list[float]", columns["open"]),
            cast("list[float]", columns["high"]),
            cast("list[float]", columns["low"]),
            cast("list[float]", columns["close"]),
            strict=True,
        )
        for timestamp, ticker, open_value, high, low, close in rows:
            previous = previous_close.get(ticker)
            previous_close[ticker] = close
            ticker_values = values.setdefault(ticker, [])
            ticker_times = times.setdefault(ticker, [])
            if previous is None or len(ticker_values) >= max_points:
                continue
            features = (
                close / previous - 1.0,
                (close - open_value) / open_value,
                (high - low) / open_value,
            )
            if not all(np.isfinite(features)):
                raise ValueError(f"non-finite derived feature for ticker {ticker!r}")
            ticker_values.append(features)
            ticker_times.append(str(timestamp))

    shards: dict[str, SeriesShard] = {}
    for ticker in sorted(values):
        ticker_values = values[ticker]
        if len(ticker_values) < max_points:
            raise ValueError(
                f"ticker {ticker!r} has {len(ticker_values)} usable points; requested {max_points}"
            )
        shards[ticker] = SeriesShard(
            values=np.asarray(ticker_values, dtype=np.float32),
            first_time=times[ticker][0],
            last_time=times[ticker][-1],
        )
    if not shards:
        raise ValueError("dataset contains no usable ticker series")
    return shards


def _split_bounds(points: int, embargo: int) -> dict[str, tuple[int, int]]:
    usable = points - 2 * embargo
    train_end = int(usable * 0.70)
    validation_length = int(usable * 0.15)
    validation_start = train_end + embargo
    validation_end = validation_start + validation_length
    test_start = validation_end + embargo
    bounds = {
        "train": (0, train_end),
        "validation": (validation_start, validation_end),
        "test": (test_start, points),
    }
    if any(end - start < embargo for start, end in bounds.values()):
        raise ValueError(
            "max_points_per_ticker is too small for three input-length partitions "
            "and two leakage embargoes"
        )
    return bounds


def _training_windows(
    shards: dict[str, SeriesShard],
    split: dict[str, tuple[int, int]],
    *,
    input_length: int,
    stride: int,
) -> tuple[
    dict[str, np.ndarray[Any, np.dtype[np.float32]]],
    list[tuple[str, int]],
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.float32]],
]:
    train_start, train_end = split["train"]
    training_points = cast(
        "np.ndarray[Any, np.dtype[np.float32]]",
        np.concatenate(  # pyright: ignore[reportUnknownMemberType]
            [shard.values[train_start:train_end] for shard in shards.values()],
            axis=0,
        ),
    )
    means = training_points.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = training_points.std(axis=0, dtype=np.float64).astype(np.float32)
    if (  # pyright: ignore[reportUnknownMemberType]
        not np.isfinite(means).all() or not np.isfinite(scales).all() or bool((scales <= 0).any())
    ):
        raise ValueError("training-only standardization state is invalid")
    normalized = {
        ticker: ((shard.values - means) / scales).astype(np.float32, copy=False)
        for ticker, shard in shards.items()
    }
    windows = [
        (ticker, start)
        for ticker in sorted(shards)
        for start in range(train_start, train_end - input_length + 1, stride)
    ]
    if not windows:
        raise ValueError("training split does not contain any complete windows")
    return normalized, windows, means, scales


def train(arguments: Arguments) -> Path:
    """Run bounded CPU training and atomically export one verified bundle."""
    if not arguments.dataset.is_file():
        raise FileNotFoundError(arguments.dataset)
    if arguments.output.exists() and not arguments.force:
        raise FileExistsError(f"{arguments.output} already exists; pass --force to replace it")
    shards = _load_bounded_shards(
        arguments.dataset,
        arguments.max_points_per_ticker,
    )
    split = _split_bounds(arguments.max_points_per_ticker, arguments.input_length)
    normalized, windows, means, scales = _training_windows(
        shards,
        split,
        input_length=arguments.input_length,
        stride=arguments.window_stride,
    )

    torch = cast("Any", importlib.import_module("torch"))
    safetensors_torch = cast("Any", importlib.import_module("safetensors.torch"))
    runtime = cast("Any", importlib.import_module("duckpd._ts2vec_torch"))
    torch.manual_seed(arguments.seed)
    torch.use_deterministic_algorithms(True)
    random = np.random.default_rng(arguments.seed)
    architecture = {
        "input_dims": len(CHANNELS),
        "output_dims": arguments.output_dims,
        "hidden_dims": arguments.hidden_dims,
        "depth": arguments.depth,
        "kernel_size": 3,
        "training_mask": "binomial-v1",
        "weights": "swa-averaged",
    }
    model = runtime.build_encoder(architecture).to("cpu")
    model.train()
    averaged = torch.optim.swa_utils.AveragedModel(model)
    averaged.update_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    losses: list[float] = []
    for iteration in range(arguments.iterations):
        selected = cast(
            "np.ndarray[Any, np.dtype[np.int64]]",
            random.choice(  # pyright: ignore[reportUnknownMemberType]
                len(windows),
                size=arguments.batch_size,
                replace=len(windows) < arguments.batch_size,
            ),
        )
        batch = cast(
            "np.ndarray[Any, np.dtype[np.float32]]",
            np.stack(  # pyright: ignore[reportUnknownMemberType]
                [
                    normalized[windows[int(index)][0]][
                        windows[int(index)][1] : windows[int(index)][1] + arguments.input_length
                    ]
                    for index in selected
                ]
            ),
        )
        tensor = torch.from_numpy(batch).to(dtype=torch.float32, device="cpu")
        loss = runtime.contrastive_train_step(
            model,
            averaged,
            optimizer,
            tensor,
            random,
            temporal_unit=arguments.temporal_unit,
        )
        losses.append(loss)
        print(f"iteration {iteration + 1}/{arguments.iterations}: loss={loss:.6f}")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    staging = arguments.output.with_name(f".{arguments.output.name}.tmp-{uuid4().hex}")
    staging.mkdir()
    try:
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in averaged.module.state_dict().items()
        }
        safetensors_torch.save_file(state, str(staging / "model.safetensors"))
        split_manifest = {
            ticker: {name: [start, end] for name, (start, end) in split.items()}
            for ticker in shards
        }
        manifest: dict[str, object] = {
            "schema_version": 1,
            "backend": "ts2vec",
            "model": "duckpd/demo-market-ts2vec",
            "source": {
                "repository": "https://github.com/zhihanyue/ts2vec",
                "revision": "b0088e14a99706c05451316dc6db8d3da9351163",
                "license": "MIT",
                "implementation": "duckpd-pinned-port-v1",
            },
            "architecture": architecture,
            "input": {
                "length": arguments.input_length,
                "channels": list(CHANNELS),
                "roles": list(ROLES),
                "dtype": "float32",
                "missingness": "complete-only",
            },
            "preprocessing": {
                "kind": "artifact-train-standard-score-v1",
                "mean": means.astype(float).tolist(),
                "scale": scales.astype(float).tolist(),
                "fitted_on": "training-points-only",
            },
            "pooling": "full-series-max-v1",
            "output": {
                "dimension": arguments.output_dims,
                "dtype": "float32",
                "normalization": "none",
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": str(torch.__version__),
                "safetensors": distribution_version("safetensors"),
            },
            "training": {
                "dataset": arguments.dataset.name,
                "dataset_sha256": _file_sha256(arguments.dataset),
                "data_contract": "demo/synthetic-ohlc/ts2vec-v1",
                "derived_channels": {
                    "close_return": "close[t] / close[t-1] - 1",
                    "bar_return": "(close - open) / open",
                    "intrabar_range": "(high - low) / open",
                },
                "tickers": sorted(shards),
                "points_per_ticker": {
                    ticker: len(shard.values) for ticker, shard in shards.items()
                },
                "split": split_manifest,
                "window_stride": arguments.window_stride,
                "seed": arguments.seed,
                "iterations": arguments.iterations,
                "batch_size": arguments.batch_size,
                "learning_rate": arguments.learning_rate,
                "temporal_unit": arguments.temporal_unit,
                "objective": "hierarchical-contrastive-v1",
                "losses": losses,
            },
        }
        specification = finalize_ts2vec_bundle(staging, manifest)
        if arguments.output.exists():
            shutil.rmtree(arguments.output)
        os.rename(staging, arguments.output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"bundle: {arguments.output}")
    print(f"model fingerprint: {specification.fingerprint}")
    print(f"artifact SHA-256: {specification.artifact_sha256}")
    print(f"channels: {json.dumps(CHANNELS)}")
    return arguments.output


def parse_args(argv: Sequence[str] | None = None) -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-points-per-ticker", type=int, default=8_192)
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--window-stride", type=int, default=128)
    parser.add_argument("--output-dims", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, default=64)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--temporal-unit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20_260_911)
    parser.add_argument("--force", action="store_true")
    namespace = parser.parse_args(argv)
    arguments = Arguments(
        dataset=cast("Path", namespace.dataset),
        output=cast("Path", namespace.output),
        max_points_per_ticker=cast("int", namespace.max_points_per_ticker),
        input_length=cast("int", namespace.input_length),
        window_stride=cast("int", namespace.window_stride),
        output_dims=cast("int", namespace.output_dims),
        hidden_dims=cast("int", namespace.hidden_dims),
        depth=cast("int", namespace.depth),
        iterations=cast("int", namespace.iterations),
        batch_size=cast("int", namespace.batch_size),
        learning_rate=cast("float", namespace.learning_rate),
        temporal_unit=cast("int", namespace.temporal_unit),
        seed=cast("int", namespace.seed),
        force=cast("bool", namespace.force),
    )
    for field, value in (
        ("max_points_per_ticker", arguments.max_points_per_ticker),
        ("input_length", arguments.input_length),
        ("window_stride", arguments.window_stride),
        ("output_dims", arguments.output_dims),
        ("hidden_dims", arguments.hidden_dims),
        ("depth", arguments.depth),
        ("iterations", arguments.iterations),
        ("batch_size", arguments.batch_size),
    ):
        if value <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if arguments.learning_rate <= 0 or not np.isfinite(arguments.learning_rate):
        parser.error("--learning-rate must be finite and positive")
    if arguments.temporal_unit < 0:
        parser.error("--temporal-unit must be non-negative")
    if 2 ** (arguments.temporal_unit + 1) > arguments.input_length:
        parser.error("--temporal-unit is too large for --input-length")
    return arguments


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
