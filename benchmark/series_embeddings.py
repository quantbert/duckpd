"""Benchmark learned time-series providers against matched deterministic controls."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import duckpd
from benchmark.metrics import get_peak_rss_bytes
from duckpd.series_embeddings import SeriesEmbeddingModelSpec, SeriesRepresentationSpec

FloatArray = np.ndarray[Any, np.dtype[np.float32]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]
Scope = Literal["univariate", "multivariate"]
_numpy = cast("Any", np)
_pyarrow = cast("Any", pa)


@dataclass(frozen=True)
class SeriesBenchmarkResult:
    """One provider or control measured over one immutable candidate population."""

    name: str
    family: str
    scope: Scope
    channels: tuple[str, ...]
    window: int
    dimension: int | None
    candidate_count: int
    query_endpoint: int
    top_k_endpoints: tuple[int, ...]
    preparation_seconds: float | None
    corpus_execution_seconds: float
    python_arrow_boundary_seconds: float | None
    input_conversion_seconds: float | None
    model_execution_seconds: float | None
    output_conversion_seconds: float | None
    persistence_seconds: float | None
    exact_search_seconds: float
    query_inference_seconds: float | None
    output_bytes: int | None
    peak_rss_bytes: int
    correct: bool
    detail: str | None = None


@dataclass(frozen=True)
class NeighborComparison:
    """Top-k agreement within one matched channel and candidate scope."""

    provider: str
    baseline: str
    comparable_scope: Scope
    compared_neighbors: int
    overlap_count: int
    jaccard: float


@dataclass(frozen=True)
class SeriesEmbeddingBenchmarkReport:
    """Machine-readable runtime evidence without a model-quality claim."""

    track: str
    data_contract: str
    duckpd_version: str
    python_version: str
    platform_system: str
    platform_machine: str
    tspulse_model_fingerprint: str
    ts2vec_model_fingerprint: str
    seed: int
    training_window_count: int
    candidate_count: int
    batch_size: int
    k: int
    dtw_radius: int
    results: tuple[SeriesBenchmarkResult, ...]
    comparisons: tuple[NeighborComparison, ...]
    qualification: str = "runtime-and-neighborhood-evidence-only"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _validate_counts(
    *, training_window_count: int, candidate_count: int, batch_size: int, k: int, dtw_radius: int
) -> None:
    for name, value in (
        ("training_window_count", training_window_count),
        ("candidate_count", candidate_count),
        ("batch_size", batch_size),
        ("k", k),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(dtw_radius) is not int or dtw_radius < 0:
        raise ValueError("dtw_radius must be a non-negative integer")
    if k > candidate_count:
        raise ValueError("k must not exceed candidate_count")


def _disjoint_endpoints(
    *, window: int, training_window_count: int, candidate_count: int
) -> tuple[IntArray, IntArray, int]:
    training = window - 1 + np.arange(training_window_count, dtype=np.int64) * window
    evaluation_start = window - 1 + training_window_count * window
    candidates = evaluation_start + np.arange(candidate_count, dtype=np.int64) * window
    return training, candidates, int(candidates[-1]) + 1


def _synthetic_series(rows: int, seed: int) -> tuple[tuple[str, ...], FloatArray]:
    """Return deterministic joint channels with trend, shocks, and cross-channel structure."""
    random = np.random.default_rng(seed)
    index = np.arange(rows, dtype=np.float32)
    regime = np.asarray(
        _numpy.where((index.astype(np.int64) // 97) % 2 == 0, 1.0, -0.65),
        dtype=np.float32,
    )
    impulse = np.zeros(rows, dtype=np.float32)
    impulse[np.arange(41, rows, 113)] = np.float32(0.018)
    noise = random.normal(0.0, 0.0012, size=rows).astype(np.float32)
    close_return = (
        np.float32(0.0025) * np.sin(index / np.float32(11.0)) * regime + impulse + noise
    ).astype(np.float32)
    bar_return = (
        np.float32(0.72) * close_return
        + np.float32(0.0018) * np.cos(index / np.float32(7.0))
        + random.normal(0.0, 0.0007, size=rows).astype(np.float32)
    ).astype(np.float32)
    intrabar_range = (
        np.float32(0.004)
        + np.float32(1.8) * np.abs(close_return)
        + np.float32(0.001) * (np.sin(index / np.float32(19.0)) + 1.0)
    ).astype(np.float32)
    values = np.asarray(
        _numpy.column_stack((close_return, bar_return, intrabar_range)),
        dtype=np.float32,
    )
    return ("close_return", "bar_return", "intrabar_range"), values


def _windows(values: FloatArray, endpoints: IntArray, window: int) -> FloatArray:
    return np.asarray(
        _numpy.stack(
            [values[int(endpoint) - window + 1 : int(endpoint) + 1] for endpoint in endpoints],
            axis=0,
        ),
        dtype=np.float32,
    )


def _unit_rows(values: FloatArray) -> FloatArray:
    norms = np.asarray(_numpy.linalg.norm(values, axis=1, keepdims=True), dtype=np.float32)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values, dtype=np.float32),
        where=norms > 0,
    ).astype(np.float32, copy=False)


def _standardized_windows(
    training: FloatArray, candidates: FloatArray
) -> tuple[FloatArray, FloatArray]:
    flattened_training = training.reshape(-1, training.shape[-1])
    means = flattened_training.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = flattened_training.std(axis=0, dtype=np.float64).astype(np.float32)
    if not bool(np.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("control training windows have invalid channel scales")
    return (
        ((training - means) / scales).astype(np.float32, copy=False),
        ((candidates - means) / scales).astype(np.float32, copy=False),
    )


def _native_vectors(training: FloatArray, candidates: FloatArray) -> tuple[FloatArray, FloatArray]:
    standardized_training, standardized_candidates = _standardized_windows(training, candidates)
    return (
        _unit_rows(standardized_training.reshape(len(training), -1)),
        _unit_rows(standardized_candidates.reshape(len(candidates), -1)),
    )


def _pca_vectors(
    training: FloatArray, candidates: FloatArray, output_dimension: int
) -> tuple[FloatArray, FloatArray]:
    native_training, native_candidates = _native_vectors(training, candidates)
    center = native_training.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered_training = (native_training - center).astype(np.float32, copy=False)
    dimension = min(output_dimension, len(training) - 1, centered_training.shape[1])
    if dimension <= 0:
        raise ValueError("PCA requires at least two training windows")
    right = np.asarray(
        _numpy.linalg.svd(centered_training, full_matrices=False)[2],
        dtype=np.float32,
    )
    components = right[:dimension]
    return (
        _unit_rows((centered_training @ components.T).astype(np.float32, copy=False)),
        _unit_rows(((native_candidates - center) @ components.T).astype(np.float32, copy=False)),
    )


def _statistical_features(windows: FloatArray) -> FloatArray:
    means = windows.mean(axis=1, dtype=np.float64)
    standard_deviations = windows.std(axis=1, dtype=np.float64)
    minima = windows.min(axis=1)
    maxima = windows.max(axis=1)
    first = windows[:, 0, :]
    last = windows[:, -1, :]
    root_mean_square = np.sqrt(np.mean(np.square(windows), axis=1, dtype=np.float64))
    channel_features = np.asarray(
        _numpy.concatenate(
            (
                means,
                standard_deviations,
                minima,
                maxima,
                first,
                last,
                last - first,
                root_mean_square,
            ),
            axis=1,
        ),
        dtype=np.float32,
    )
    correlations: list[FloatArray] = []
    for left in range(windows.shape[2]):
        for right in range(left + 1, windows.shape[2]):
            left_values = windows[:, :, left]
            right_values = windows[:, :, right]
            left_centered = left_values - left_values.mean(axis=1, keepdims=True)
            right_centered = right_values - right_values.mean(axis=1, keepdims=True)
            denominator = np.sqrt(np.sum(np.square(left_centered), axis=1)) * np.sqrt(
                np.sum(np.square(right_centered), axis=1)
            )
            numerator = np.sum(left_centered * right_centered, axis=1)
            correlations.append(
                np.divide(
                    numerator,
                    denominator,
                    out=np.zeros(len(windows), dtype=np.float32),
                    where=denominator > 0,
                )[:, None]
            )
    if correlations:
        return np.asarray(
            _numpy.concatenate((channel_features, *correlations), axis=1),
            dtype=np.float32,
        )
    return channel_features.astype(np.float32)


def _statistical_vectors(
    training: FloatArray, candidates: FloatArray
) -> tuple[FloatArray, FloatArray]:
    training_features = _statistical_features(training)
    candidate_features = _statistical_features(candidates)
    means = training_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = training_features.std(axis=0, dtype=np.float64).astype(np.float32)
    scales[scales == 0] = 1
    return (
        _unit_rows((training_features - means) / scales),
        _unit_rows((candidate_features - means) / scales),
    )


def _bounded_dtw_distance(left: FloatArray, right: FloatArray, radius: int) -> float:
    length = left.shape[0]
    previous = np.full(length + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for left_index in range(1, length + 1):
        current = np.full(length + 1, np.inf, dtype=np.float64)
        start = max(1, left_index - radius)
        stop = min(length, left_index + radius)
        for right_index in range(start, stop + 1):
            difference = left[left_index - 1] - right[right_index - 1]
            cost = float(np.sum(np.square(difference), dtype=np.float64))
            current[right_index] = cost + min(
                current[right_index - 1],
                previous[right_index],
                previous[right_index - 1],
            )
        previous = current
    return float(np.sqrt(previous[length] / length))


def _persist_and_search_vectors(
    *,
    name: str,
    scope: Scope,
    channels: tuple[str, ...],
    window: int,
    endpoints: IntArray,
    vectors: FloatArray,
    query: FloatArray,
    k: int,
    directory: Path,
    representation_seconds: float,
) -> SeriesBenchmarkResult:
    conversion_started = perf_counter()
    contiguous = np.ascontiguousarray(vectors, dtype=np.float32)
    child = pa.array(contiguous.reshape(-1), type=pa.float32())
    maker = cast("Any", pa.FixedSizeListArray)
    embedding = maker.from_arrays(child, contiguous.shape[1])
    table = cast(
        "pa.Table",
        _pyarrow.table(
            {
                "endpoint": pa.array(endpoints, type=pa.int64()),
                "embedding": embedding,
            }
        ),
    )
    conversion_seconds = perf_counter() - conversion_started
    output = directory / f"{name}.parquet"
    with duckpd.connect() as session:
        frame = session.from_arrow(table)
        persistence_started = perf_counter()
        frame.write_parquet(output)
        persistence_seconds = perf_counter() - persistence_started
        persisted = session.read_parquet(output)
        search = persisted.vector.search(
            query.tolist(),
            column="embedding",
            metric="cosine",
            k=k,
            tie_breaker="endpoint",
        )
        search_started = perf_counter()
        result = search.collect()
        search_seconds = perf_counter() - search_started
    top = tuple(int(value) for value in result["endpoint"])
    query_endpoint = int(endpoints[-1])
    return SeriesBenchmarkResult(
        name=name,
        family=name.split("_", 1)[0],
        scope=scope,
        channels=channels,
        window=window,
        dimension=contiguous.shape[1],
        candidate_count=len(endpoints),
        query_endpoint=query_endpoint,
        top_k_endpoints=top,
        preparation_seconds=None,
        corpus_execution_seconds=representation_seconds,
        python_arrow_boundary_seconds=conversion_seconds,
        input_conversion_seconds=None,
        model_execution_seconds=None,
        output_conversion_seconds=None,
        persistence_seconds=persistence_seconds,
        exact_search_seconds=search_seconds,
        query_inference_seconds=None,
        output_bytes=output.stat().st_size,
        peak_rss_bytes=get_peak_rss_bytes(),
        correct=top[0] == query_endpoint,
    )


def _run_vector_controls(
    *,
    scope: Scope,
    channels: tuple[str, ...],
    window: int,
    training_windows: FloatArray,
    candidate_windows: FloatArray,
    endpoints: IntArray,
    output_dimension: int,
    k: int,
    dtw_radius: int,
    directory: Path,
) -> tuple[SeriesBenchmarkResult, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    values: list[SeriesBenchmarkResult] = []
    for family, transform in (
        ("native", lambda: _native_vectors(training_windows, candidate_windows)),
        (
            "pca",
            lambda: _pca_vectors(training_windows, candidate_windows, output_dimension),
        ),
        ("statistical", lambda: _statistical_vectors(training_windows, candidate_windows)),
    ):
        started = perf_counter()
        _, candidates = transform()
        representation_seconds = perf_counter() - started
        values.append(
            _persist_and_search_vectors(
                name=f"{family}_{scope}",
                scope=scope,
                channels=channels,
                window=window,
                endpoints=endpoints,
                vectors=candidates,
                query=candidates[-1],
                k=k,
                directory=directory,
                representation_seconds=representation_seconds,
            )
        )

    standardized_training, standardized_candidates = _standardized_windows(
        training_windows, candidate_windows
    )
    del standardized_training
    query = standardized_candidates[-1]
    started = perf_counter()
    distances = np.asarray(
        [
            _bounded_dtw_distance(candidate, query, dtw_radius)
            for candidate in standardized_candidates
        ],
        dtype=np.float64,
    )
    dtw_seconds = perf_counter() - started
    ordering = np.asarray(_numpy.lexsort((endpoints, distances)), dtype=np.int64)[:k]
    top = tuple(int(value) for value in endpoints[ordering])
    query_endpoint = int(endpoints[-1])
    values.append(
        SeriesBenchmarkResult(
            name=f"dtw_{scope}",
            family="dtw",
            scope=scope,
            channels=channels,
            window=window,
            dimension=None,
            candidate_count=len(endpoints),
            query_endpoint=query_endpoint,
            top_k_endpoints=top,
            preparation_seconds=None,
            corpus_execution_seconds=dtw_seconds,
            python_arrow_boundary_seconds=None,
            input_conversion_seconds=None,
            model_execution_seconds=None,
            output_conversion_seconds=None,
            persistence_seconds=None,
            exact_search_seconds=dtw_seconds,
            query_inference_seconds=None,
            output_bytes=None,
            peak_rss_bytes=get_peak_rss_bytes(),
            correct=top[0] == query_endpoint,
            detail=f"direct bounded multivariate DTW with radius={dtw_radius}",
        )
    )
    return tuple(values)


def _window_builder(label: str, window: int) -> Callable[[duckpd.DataFrame], duckpd.Series]:
    def build(frame: duckpd.DataFrame) -> duckpd.Series:
        return cast(
            "duckpd.Series",
            frame.groupby("_candidate")[label].rolling(window).to_array(),
        )

    return build


def _frame_with_windows(
    session: duckpd.Session,
    values: FloatArray,
    all_channels: tuple[str, ...],
    channels: tuple[str, ...],
    representation: SeriesRepresentationSpec,
    endpoints: IntArray,
    batch_size: int,
) -> duckpd.DataFrame:
    selected = values[:, [all_channels.index(channel) for channel in channels]]
    candidate_windows = _windows(selected, endpoints, representation.window)
    source_values: dict[str, object] = {
        "endpoint": _numpy.repeat(endpoints, representation.window),
        "_candidate": _numpy.repeat(np.arange(len(endpoints)), representation.window),
        "_position": _numpy.tile(np.arange(representation.window), len(endpoints)),
    }
    source_values.update(
        {
            channel: candidate_windows[:, :, index].reshape(-1)
            for index, channel in enumerate(channels)
        }
    )
    source = session.from_pandas(
        pd.DataFrame(source_values),
        order_by=["_candidate", "_position"],
    )
    assignments = {
        f"{channel}_window": _window_builder(channel, representation.window) for channel in channels
    }
    windows = source.assign(**assignments)
    complete = windows[windows["_position"] == representation.window - 1]
    return complete.embed_series(
        columns={channel: f"{channel}_window" for channel in channels},
        into="embedding",
        representation=representation,
        batch_size=batch_size,
    )


def _run_duckpd_representation(
    *,
    name: str,
    scope: Scope,
    values: FloatArray,
    all_channels: tuple[str, ...],
    representation: SeriesRepresentationSpec,
    endpoints: IntArray,
    batch_size: int,
    k: int,
    directory: Path,
    artifact_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> SeriesBenchmarkResult:
    model = representation.encoder
    channels = representation.channels
    output = directory / f"{name}.parquet"
    with duckpd.connect() as session:
        preparation_seconds: float | None = None
        if model is not None:
            started = perf_counter()
            session.prepare_series_embedding_model(
                model,
                artifact_dir=artifact_dir,
                cache_dir=cache_dir,
                timeout_seconds=300,
                max_artifact_bytes=(512 * 1024 * 1024 if artifact_dir is not None else None),
                max_download_bytes=(10 * 1024 * 1024 if cache_dir is not None else None),
            )
            preparation_seconds = perf_counter() - started
        embedded = _frame_with_windows(
            session,
            values,
            all_channels,
            channels,
            representation,
            endpoints,
            batch_size,
        )
        corpus_started = perf_counter()
        materialized = embedded[["endpoint", "embedding"]].persist(
            f"__duckpd_series_benchmark_{name}"
        )
        corpus_execution_seconds = perf_counter() - corpus_started
        metrics = dict(session._embedding_metrics)
        persistence_started = perf_counter()
        materialized.write_parquet(output)
        persistence_seconds = perf_counter() - persistence_started

        query_values = values[
            int(endpoints[-1]) - representation.window + 1 : int(endpoints[-1]) + 1
        ]
        query = {
            channel: query_values[:, all_channels.index(channel)].astype(float).tolist()
            for channel in channels
        }
        query_before = float(session._embedding_metrics.get("series_query_inference_seconds", 0.0))
        query_started = perf_counter()
        embedded_query = session.embed_series_query(
            query,
            representation=representation,
        )
        query_wall_seconds = perf_counter() - query_started
        query_after = float(session._embedding_metrics.get("series_query_inference_seconds", 0.0))
        persisted = session.read_parquet(output)
        search = persisted.vector.search(
            embedded_query,
            column="embedding",
            metric="cosine",
            k=k,
            tie_breaker="endpoint",
        )
        search_started = perf_counter()
        result = search.collect()
        search_seconds = perf_counter() - search_started

    top = tuple(int(value) for value in result["endpoint"])
    query_endpoint = int(endpoints[-1])
    return SeriesBenchmarkResult(
        name=name,
        family=name.split("_", 1)[0],
        scope=scope,
        channels=channels,
        window=representation.window,
        dimension=representation.dimension,
        candidate_count=len(endpoints),
        query_endpoint=query_endpoint,
        top_k_endpoints=top,
        preparation_seconds=preparation_seconds,
        corpus_execution_seconds=corpus_execution_seconds,
        python_arrow_boundary_seconds=_metric(
            metrics,
            "series_corpus_python_arrow_boundary_seconds",
        ),
        input_conversion_seconds=_metric(metrics, "series_corpus_input_conversion_seconds"),
        model_execution_seconds=_metric(metrics, "series_corpus_model_execution_seconds"),
        output_conversion_seconds=_metric(metrics, "series_corpus_output_conversion_seconds"),
        persistence_seconds=persistence_seconds,
        exact_search_seconds=search_seconds,
        query_inference_seconds=(
            query_after - query_before if model is not None else query_wall_seconds
        ),
        output_bytes=output.stat().st_size,
        peak_rss_bytes=get_peak_rss_bytes(),
        correct=top[0] == query_endpoint,
    )


def _metric(metrics: dict[str, int | float], name: str) -> float | None:
    value = metrics.get(name)
    return float(value) if value is not None else None


def _comparisons(
    results: tuple[SeriesBenchmarkResult, ...], k: int
) -> tuple[NeighborComparison, ...]:
    scopes: tuple[Scope, ...] = ("univariate", "multivariate")
    by_scope: dict[Scope, list[SeriesBenchmarkResult]] = {
        scope: [result for result in results if result.scope == scope] for scope in scopes
    }
    values: list[NeighborComparison] = []
    for scope, scoped in by_scope.items():
        provider = next(result for result in scoped if result.family in {"tspulse", "ts2vec"})
        provider_neighbors = set(provider.top_k_endpoints[1:])
        for baseline in scoped:
            if baseline is provider:
                continue
            baseline_neighbors = set(baseline.top_k_endpoints[1:])
            union = provider_neighbors | baseline_neighbors
            overlap = len(provider_neighbors & baseline_neighbors)
            values.append(
                NeighborComparison(
                    provider=provider.name,
                    baseline=baseline.name,
                    comparable_scope=scope,
                    compared_neighbors=max(0, k - 1),
                    overlap_count=overlap,
                    jaccard=overlap / len(union) if union else 1.0,
                )
            )
    return tuple(values)


def run_series_embedding_benchmark(
    *,
    ts2vec_bundle: str | Path,
    training_window_count: int = 256,
    candidate_count: int = 128,
    batch_size: int = 16,
    k: int = 10,
    dtw_radius: int = 16,
    seed: int = 20_260_914,
    tspulse_cache: str | Path | None = None,
) -> SeriesEmbeddingBenchmarkReport:
    """Run both learned providers and matched native/PCA/statistical/DTW controls."""
    _validate_counts(
        training_window_count=training_window_count,
        candidate_count=candidate_count,
        batch_size=batch_size,
        k=k,
        dtw_radius=dtw_radius,
    )
    bundle = Path(ts2vec_bundle)
    if not bundle.is_dir():
        raise FileNotFoundError(
            f"TS2Vec bundle not found: {bundle}; run demo/train_ts2vec.py first"
        )
    ts2vec_model = duckpd.ts2vec_series_embedding_model(bundle)
    tspulse_model = duckpd.tspulse_series_embedding_model("bar_return")
    all_channels = ("close_return", "bar_return", "intrabar_range")
    if any(channel not in all_channels for channel in ts2vec_model.input_channels):
        raise ValueError(
            "TS2Vec benchmark bundle channels must be drawn from "
            f"{all_channels}; found {ts2vec_model.input_channels}"
        )
    maximum_window = max(ts2vec_model.input_length, tspulse_model.input_length)
    training_endpoints, endpoints, rows = _disjoint_endpoints(
        window=maximum_window,
        training_window_count=training_window_count,
        candidate_count=candidate_count,
    )
    generated_channels, values = _synthetic_series(rows, seed)
    if generated_channels != all_channels:
        raise AssertionError("synthetic benchmark channel contract changed")
    data_contract = "benchmark/synthetic-joint-series/v1"

    specifications: tuple[
        tuple[str, Scope, SeriesEmbeddingModelSpec, SeriesRepresentationSpec, Path | None], ...
    ] = (
        (
            "tspulse_univariate",
            "univariate",
            tspulse_model,
            duckpd.series_representation(
                window=tspulse_model.input_length,
                channels=tspulse_model.input_channels,
                sampling="observations",
                data_contract=data_contract,
                encoder=tspulse_model,
            ),
            None,
        ),
        (
            "ts2vec_multivariate",
            "multivariate",
            ts2vec_model,
            duckpd.series_representation(
                window=ts2vec_model.input_length,
                channels=ts2vec_model.input_channels,
                sampling="observations",
                data_contract=data_contract,
                encoder=ts2vec_model,
            ),
            bundle,
        ),
    )

    with tempfile.TemporaryDirectory(prefix="duckpd-series-benchmark-") as temporary:
        directory = Path(temporary)
        results: list[SeriesBenchmarkResult] = []
        for name, scope, model, representation, artifact in specifications:
            channel_indexes = [all_channels.index(channel) for channel in model.input_channels]
            scoped = values[:, channel_indexes]
            training_windows = _windows(scoped, training_endpoints, model.input_length)
            candidate_windows = _windows(scoped, endpoints, model.input_length)
            results.extend(
                _run_vector_controls(
                    scope=scope,
                    channels=model.input_channels,
                    window=model.input_length,
                    training_windows=training_windows,
                    candidate_windows=candidate_windows,
                    endpoints=endpoints,
                    output_dimension=model.dimension,
                    k=k,
                    dtw_radius=dtw_radius,
                    directory=directory,
                )
            )
            results.append(
                _run_duckpd_representation(
                    name=name,
                    scope=scope,
                    values=values,
                    all_channels=all_channels,
                    representation=representation,
                    endpoints=endpoints,
                    batch_size=batch_size,
                    k=k,
                    directory=directory,
                    artifact_dir=artifact,
                    cache_dir=(
                        Path(tspulse_cache)
                        if tspulse_cache is not None and artifact is None
                        else None
                    ),
                )
            )

    encoded_results = tuple(results)
    if not all(result.correct for result in encoded_results):
        failed = ", ".join(result.name for result in encoded_results if not result.correct)
        raise AssertionError(f"benchmark query self-retrieval failed: {failed}")
    return SeriesEmbeddingBenchmarkReport(
        track="learned_series_retrieval",
        data_contract=data_contract,
        duckpd_version=duckpd.__version__,
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        tspulse_model_fingerprint=tspulse_model.fingerprint,
        ts2vec_model_fingerprint=ts2vec_model.fingerprint,
        seed=seed,
        training_window_count=training_window_count,
        candidate_count=candidate_count,
        batch_size=batch_size,
        k=k,
        dtw_radius=dtw_radius,
        results=encoded_results,
        comparisons=_comparisons(encoded_results, k),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ts2vec-bundle", type=Path, required=True)
    parser.add_argument("--training-windows", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--dtw-radius", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_914)
    parser.add_argument("--tspulse-cache", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = run_series_embedding_benchmark(
            ts2vec_bundle=arguments.ts2vec_bundle,
            training_window_count=arguments.training_windows,
            candidate_count=arguments.candidates,
            batch_size=arguments.batch_size,
            k=arguments.k,
            dtw_radius=arguments.dtw_radius,
            seed=arguments.seed,
            tspulse_cache=arguments.tspulse_cache,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    payload = report.to_json()
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
