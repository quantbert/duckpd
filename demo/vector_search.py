"""Search a real remote stock-news archive with streaming text embeddings."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import duckpd as pd

DATA_URL = "https://huggingface.co/datasets/AlphaDojo/dojo_stock_news/resolve/main/data.parquet"
DEMO_DIR = Path(__file__).resolve().parent
EMBEDDED_DATA = DEMO_DIR / "nvidia-news-embedded-transformers.parquet"
QUERY = "AI chip demand and revenue growth"
TRANSFORMER_BATCH_SIZE = 64
MODEL = pd.embedding_model(
    "BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dimension=384,
    backend="transformers",
    pooling="cls",
)


def main() -> None:
    with pd.connect() as session:
        provider = pd.TransformersEmbeddingProvider(
            MODEL,
            device="cuda",
            batch_size=TRANSFORMER_BATCH_SIZE,
        )
        session.register_embedding_provider(MODEL, provider)
        preparation_started = perf_counter()
        prepared = session.prepare_embedding_model(MODEL)
        preparation_seconds = perf_counter() - preparation_started

        if EMBEDDED_DATA.exists():
            embedded = session.read_parquet(EMBEDDED_DATA)
            dataset_status = f"loaded {EMBEDDED_DATA}"
        else:
            build_started = perf_counter()
            news = session.read_parquet(DATA_URL)
            nvidia_news = news[news["symbol"] == "NVDA"]

            embedded = nvidia_news.embed_text(
                columns=["title", "description"],
                into="embedding",
                model=MODEL,
                batch_size=64,
                null_policy="empty",
            )

            embedded.write_parquet(EMBEDDED_DATA)
            build_seconds = perf_counter() - build_started
            dataset_status = (
                f"created {EMBEDDED_DATA} from {DATA_URL} in {build_seconds:.3f} seconds"
            )
            embedded = session.read_parquet(EMBEDDED_DATA)
        query_started = perf_counter()
        matches = embedded.vector.search_text(
            QUERY,
            column="embedding",
            model=MODEL,
            metric="cosine",
            k=5,
            tie_breaker="title",
        )[["symbol", "title", "publisher", "publish_date", "_distance"]]
        result = matches.collect()
        query_seconds = perf_counter() - query_started

        print(f"Embedding dataset: {dataset_status}")
        print(f"Model: {MODEL.model}@{MODEL.revision}")
        print(f"Backend: {prepared.backend} via {prepared.execution_providers}")
        print(f"Model preparation: {preparation_seconds:.3f} seconds")
        print(f"Query: {QUERY!r}")
        print(result.to_string(index=False))
        print(f"Query-to-response: {query_seconds:.3f} seconds")
        print(f"Persisted model fingerprint: {MODEL.fingerprint}")


if __name__ == "__main__":
    main()
