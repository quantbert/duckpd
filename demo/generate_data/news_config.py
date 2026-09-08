"""Pinned source and model configuration for generated news features."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import duckpd as pd

NEWS_SOURCE_REPOSITORY = "AlphaDojo/dojo_stock_news"
NEWS_SOURCE_REVISION = "9d7a641b5cb5eaedc1874c34e74aa186a0a768f1"
NEWS_SOURCE_PATH = "data.parquet"
NEWS_SOURCE_SHA256 = "f8586e5a3515937da130b708006197652fd1560c2058d0dcd0134af5ebebe1fd"
NEWS_SOURCE_ROWS = 3_951_636
NEWS_MODEL_KEY = "bge-small-en-v1.5"
NEWS_MODEL = pd.embedding_model(
    "BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dimension=384,
)
NEWS_TRANSFORMERS_MODEL = pd.embedding_model(
    "BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dimension=384,
    backend="transformers",
    pooling="cls",
)


def news_model(backend: str) -> pd.EmbeddingModelSpec:
    """Return the pinned model specification for a supported generation backend."""
    if backend == "fastembed":
        return NEWS_MODEL
    if backend == "transformers":
        return NEWS_TRANSFORMERS_MODEL
    raise ValueError("embedding backend must be 'fastembed' or 'transformers'")


def embedding_models(
    model: pd.EmbeddingModelSpec = NEWS_MODEL,
) -> dict[str, dict[str, Any]]:
    """Return the serializable embedding-model registry."""
    return {NEWS_MODEL_KEY: asdict(model)}
