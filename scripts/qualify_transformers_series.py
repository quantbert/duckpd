#!/usr/bin/env python3
"""Qualify one pinned Transformers series checkpoint against offline query inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import resource
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from duckpd.embeddings import EmbeddingModelSpec, SeriesEmbeddingInputSpec
from duckpd.series_embeddings import (
    SeriesQuerySnapshot,
    _finalize_series_embeddings,
    _normalize_series_snapshot,
    make_series_provider_batch,
    series_query,
    series_representation,
    snapshot_series_query,
)
from duckpd.series_providers import TransformersSeriesEmbeddingProvider

_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40,64}\Z")
_RFC3339 = re.compile(r"\d{4}-\d{2}-\d{2}T.+(?:Z|[+-]\d{2}:\d{2})\Z")
_ABSOLUTE_TOLERANCE = 1e-5
_RELATIVE_TOLERANCE = 1e-5


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-sizes", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json_object(path: Path, *, owner: str) -> dict[str, object]:
    try:
        value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{owner} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a JSON object")
    mapping = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in mapping):
        raise TypeError(f"{owner} must be a JSON object")
    return cast("dict[str, object]", mapping)


def _parse_timestamp(value: object, *, field: str, line: int) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or _RFC3339.fullmatch(value) is None:
        raise ValueError(f"query line {line} field {field!r} must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as error:
        raise ValueError(
            f"query line {line} field {field!r} must be an RFC 3339 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"query line {line} field {field!r} must include a UTC offset")
    return parsed


def _load_queries(path: Path) -> list[object]:
    queries: list[object] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"queries are not valid UTF-8 text: {path}") from error
    for line_number, text in enumerate(lines, start=1):
        if not text.strip():
            raise ValueError(f"query line {line_number} is empty")
        try:
            raw = cast("object", json.loads(text))
        except json.JSONDecodeError as error:
            raise ValueError(f"query line {line_number} is not valid JSON") from error
        if not isinstance(raw, dict):
            raise TypeError(f"query line {line_number} must be an object")
        mapping = cast("dict[object, object]", raw)
        if any(not isinstance(key, str) for key in mapping):
            raise TypeError(f"query line {line_number} must be an object")
        data = cast("dict[str, object]", mapping)
        unknown = set(data) - {"values", "time", "series_start", "static"}
        if unknown:
            raise ValueError(
                f"query line {line_number} has unknown fields: {', '.join(sorted(unknown))}"
            )
        if "values" not in data:
            raise ValueError(f"query line {line_number} is missing field 'values'")
        values = data["values"]
        static = data.get("static")
        if not isinstance(values, dict):
            raise TypeError(f"query line {line_number} field 'values' must be an object")
        if static is not None and not isinstance(static, dict):
            raise TypeError(f"query line {line_number} field 'static' must be an object")
        queries.append(
            series_query(
                cast("dict[str, object]", values),  # type: ignore[arg-type]
                time=_parse_timestamp(data.get("time"), field="time", line=line_number),
                series_start=_parse_timestamp(
                    data.get("series_start"),
                    field="series_start",
                    line=line_number,
                ),
                static=cast("dict[str, object] | None", static),  # type: ignore[arg-type]
            )
        )
    if not queries:
        raise ValueError("queries file must contain at least one query")
    return queries


def _batch_sizes(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--batch-sizes must be comma-separated positive integers") from error
    if not result or any(item <= 0 for item in result) or len(result) != len(set(result)):
        raise ValueError("--batch-sizes must be unique comma-separated positive integers")
    return result


def _embed(
    provider: TransformersSeriesEmbeddingProvider,
    snapshots: tuple[SeriesQuerySnapshot, ...],
    representation: Any,
    batch_size: int,
) -> tuple[tuple[float, ...], ...]:
    output: list[tuple[float, ...]] = []
    for offset in range(0, len(snapshots), batch_size):
        rows = snapshots[offset : offset + batch_size]
        batch = make_series_provider_batch(rows, representation)
        vectors = _finalize_series_embeddings(
            provider.embed_windows(batch),
            representation,
            expected_rows=len(rows),
            zero_scale_as_null=False,
        )
        if any(vector is None for vector in vectors):
            raise ValueError("qualification produced a null embedding")
        output.extend(cast("tuple[float, ...]", vector) for vector in vectors)
    return tuple(output)


def _differences(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> tuple[float, float]:
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right, strict=True)):
        raise ValueError("qualification runs returned different output shapes")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for a_row, b_row in zip(left, right, strict=True):
        for a, b in zip(a_row, b_row, strict=True):
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("qualification produced a nonfinite embedding")
            absolute = abs(a - b)
            relative = absolute / max(abs(a), abs(b), sys.float_info.min)
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
    return maximum_absolute, maximum_relative


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main() -> int:
    arguments = _arguments()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite qualification output: {arguments.output}")
    specification = EmbeddingModelSpec.from_dict(
        _load_json_object(arguments.spec, owner="model specification")
    )
    if specification.backend != "transformers" or not isinstance(
        specification.input,
        SeriesEmbeddingInputSpec,
    ):
        raise ValueError("qualification requires a Transformers series model specification")
    if _IMMUTABLE_REVISION.fullmatch(specification.revision) is None:
        raise ValueError("qualification requires a lowercase immutable commit revision")
    if specification.input.provider_abi != "transformers-series-v1":
        raise ValueError("qualification requires provider_abi='transformers-series-v1'")

    batch_sizes = _batch_sizes(arguments.batch_sizes)
    raw_queries = _load_queries(arguments.queries)
    representation = series_representation(
        window=specification.input.length,
        channels=specification.input.channels,
        sampling="observations",
        data_contract="qualification/offline-input/v1",
        normalization="none",
        unit_norm=specification.normalize,
        encoder=specification,
    )
    snapshots: list[SeriesQuerySnapshot] = []
    for query in raw_queries:
        snapshot = snapshot_series_query(query, representation)  # type: ignore[arg-type]
        normalized = _normalize_series_snapshot(
            snapshot,
            representation,
            zero_scale_as_null=False,
        )
        assert normalized is not None
        snapshots.append(normalized)
    frozen_snapshots = tuple(snapshots)

    provider = TransformersSeriesEmbeddingProvider(
        specification,
        device=arguments.device,
    )
    preparation_started = perf_counter()
    prepared = provider.prepare()
    preparation_seconds = perf_counter() - preparation_started
    try:
        run_records: list[dict[str, object]] = []
        baseline: tuple[tuple[float, ...], ...] | None = None
        total_rows = 0
        total_seconds = 0.0
        maximum_absolute = 0.0
        maximum_relative = 0.0
        for batch_size in batch_sizes:
            repeated: list[tuple[tuple[float, ...], ...]] = []
            durations: list[float] = []
            for _ in range(2):
                started = perf_counter()
                repeated.append(_embed(provider, frozen_snapshots, representation, batch_size))
                durations.append(perf_counter() - started)
            reference = repeated[0] if baseline is None else baseline
            repeat_absolute, repeat_relative = _differences(repeated[0], repeated[1])
            batch_absolute, batch_relative = _differences(reference, repeated[0])
            maximum_absolute = max(maximum_absolute, repeat_absolute, batch_absolute)
            maximum_relative = max(maximum_relative, repeat_relative, batch_relative)
            baseline = reference
            total_rows += len(frozen_snapshots) * 2
            total_seconds += sum(durations)
            run_records.append(
                {
                    "batch_size": batch_size,
                    "durations_seconds": durations,
                    "repeat_maximum_absolute_difference": repeat_absolute,
                    "repeat_maximum_relative_difference": repeat_relative,
                    "baseline_maximum_absolute_difference": batch_absolute,
                    "baseline_maximum_relative_difference": batch_relative,
                }
            )
        if maximum_absolute > _ABSOLUTE_TOLERANCE and maximum_relative > _RELATIVE_TOLERANCE:
            raise ValueError(
                "qualification output changed beyond deterministic tolerance: "
                f"absolute={maximum_absolute}, relative={maximum_relative}"
            )

        cache_path = Path(cast("str", prepared.cache_path))
        manifest_path = cache_path / f"duckpd-{specification.fingerprint}.json"
        manifest = _load_json_object(manifest_path, owner="prepared-model manifest")
        torch: Any = __import__("torch")
        transformers: Any = __import__("transformers")
        cuda_peak = int(torch.cuda.max_memory_allocated()) if arguments.device == "cuda" else None
        record: dict[str, object] = {
            "qualification_version": 1,
            "status": "adapter_compatible_not_semantically_qualified",
            "artifact": {
                "repository": specification.model,
                "revision": specification.revision,
                "artifact_digest": prepared.artifact_digest,
                "architecture": manifest["model_type"],
                "bare_class": manifest["resolved_class"],
                "configuration_digest": manifest["configuration_digest"],
                "loading_information": {
                    "unexpected_keys": manifest["unexpected_keys"],
                    "unexpected_keys_digest": manifest["unexpected_keys_digest"],
                    "missing_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                },
                "licenses": {"source": "not_provided", "weights": "not_provided"},
            },
            "runtime": {
                "transformers": str(transformers.__version__),
                "pytorch": str(torch.__version__),
                "python": platform.python_version(),
                "operating_system": platform.platform(),
                "accelerator": arguments.device,
                "execution_providers": list(prepared.execution_providers),
            },
            "input": {
                "channels": list(specification.input.channels),
                "roles": list(specification.input.roles),
                "units": "not_provided",
                "frequency": (
                    None
                    if specification.input.frequency is None
                    else specification.input.frequency.to_dict()
                ),
                "temporal": (
                    None
                    if specification.input.temporal is None
                    else specification.input.temporal.to_dict()
                ),
                "static": [item.to_dict() for item in specification.input.static],
                "outer_normalization": "none",
                "internal_normalization": specification.input.normalization,
            },
            "output": {
                "pooling": specification.pooling,
                "dimension": specification.dimension,
                "unit_normalization": specification.normalize,
            },
            "performance": {
                "cold_preparation_seconds": preparation_seconds,
                "cache_bytes": _directory_bytes(cache_path),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "peak_vram_bytes": cuda_peak,
                "throughput_rows_per_second": total_rows / total_seconds,
                "supported_batch_sizes": list(batch_sizes),
            },
            "determinism": {
                "absolute_tolerance": _ABSOLUTE_TOLERANCE,
                "relative_tolerance": _RELATIVE_TOLERANCE,
                "maximum_absolute_difference": maximum_absolute,
                "maximum_relative_difference": maximum_relative,
                "runs": run_records,
            },
            "retrieval_metrics": {
                "status": "not_run",
                "reason": "offline adapter qualification does not establish semantic fitness",
            },
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, arguments.output)
    finally:
        provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
