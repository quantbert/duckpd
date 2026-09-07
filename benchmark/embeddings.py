"""Benchmark DuckPD's optional local text-embedding execution boundary."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import tempfile
from pathlib import Path
from time import perf_counter
from typing import cast

import pandas as pd

import duckpd

MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def run(rows: int, batch_size: int, text_words: int) -> dict[str, object]:
    model = duckpd.embedding_model(
        "BAAI/bge-small-en-v1.5",
        revision=MODEL_REVISION,
        dimension=384,
    )
    vocabulary = [
        "company",
        "reports",
        "artificial",
        "intelligence",
        "chip",
        "revenue",
        "growth",
        "quarterly",
        "demand",
        "supply",
        "customers",
        "data",
        "center",
        "earnings",
        "market",
    ]
    texts = [
        " ".join(vocabulary[(index + offset) % len(vocabulary)] for offset in range(text_words))
        for index in range(rows)
    ]
    with duckpd.connect() as session:
        prepare_started = perf_counter()
        prepared = session.prepare_embedding_model(model)
        preparation_seconds = perf_counter() - prepare_started
        provider = session._embedding_provider(model)

        direct_started = perf_counter()
        direct = provider.embed_documents(texts)
        direct_seconds = perf_counter() - direct_started

        query_started = perf_counter()
        first_query = session.embed_query("AI chip revenue growth", model=model)
        first_query_seconds = perf_counter() - query_started
        query_started = perf_counter()
        second_query = session.embed_query("AI chip revenue growth", model=model)
        repeated_query_seconds = perf_counter() - query_started

        frame = session.from_pandas(pd.DataFrame({"id": range(rows), "text": texts}))
        embedded = frame.embed_text(
            columns="text",
            into="embedding",
            model=model,
            batch_size=batch_size,
        )
        profile = embedded.profile()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "embedded.parquet"
            sink_started = perf_counter()
            embedded.write_parquet(output)
            sink_seconds = perf_counter() - sink_started
            output_bytes = output.stat().st_size
            persisted = session.read_parquet(output).to_arrow()["embedding"].combine_chunks()
            direct_rows = cast("list[list[float]]", direct.to_pylist())
            persisted_rows = cast("list[list[float]]", persisted.to_pylist())
            maximum_absolute_error = max(
                abs(left - right)
                for direct_row, persisted_row in zip(direct_rows, persisted_rows, strict=True)
                for left, right in zip(direct_row, persisted_row, strict=True)
            )
            direct_provider_parity = maximum_absolute_error <= 1e-6
            if not direct_provider_parity:
                raise RuntimeError("DuckPD embedding output differs from direct provider output")

        return {
            "track": "text_embeddings_local_cpu",
            "model": model.model,
            "revision": model.revision,
            "model_fingerprint": model.fingerprint,
            "artifact_digest": prepared.artifact_digest,
            "execution_providers": prepared.execution_providers,
            "rows": rows,
            "text_bytes": sum(len(text.encode()) for text in texts),
            "batch_size": batch_size,
            "approximate_whitespace_tokens": sum(len(text.split()) for text in texts),
            "text_words_per_row": text_words,
            "preparation_seconds": preparation_seconds,
            "direct_provider_seconds": direct_seconds,
            "duckpd_profile": profile.embedding_metrics,
            "sink_seconds": sink_seconds,
            "first_query_seconds": first_query_seconds,
            "repeated_query_seconds": repeated_query_seconds,
            "repeated_query_equal": first_query == second_query,
            "output_bytes": output_bytes,
            "peak_rss_bytes": _peak_rss_bytes(),
            "direct_provider_rows": len(direct),
            "direct_provider_parity": direct_provider_parity,
            "maximum_absolute_error": maximum_absolute_error,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[100, 1_000])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 256])
    parser.add_argument("--text-words", type=int, nargs="+", default=[12, 96])
    args = parser.parse_args()
    values = [*args.rows, *args.batch_sizes, *args.text_words]
    if any(value <= 0 for value in values):
        parser.error("--rows, --batch-sizes, and --text-words must be positive")
    results = [
        run(rows, batch_size, text_words)
        for rows in args.rows
        for batch_size in args.batch_sizes
        for text_words in args.text_words
    ]
    print(json.dumps(results, indent=2, default=list))


if __name__ == "__main__":
    main()
