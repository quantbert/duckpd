"""Demonstrate cached remote analytics with DuckPD."""

import os
from datetime import timedelta
from pathlib import Path

import duckpd as pd

DEMO_DIRECTORY = Path(__file__).resolve().parent

cache_directory = DEMO_DIRECTORY / "feature_cache"
store = pd.FeatureStore(
    source=os.environ["HF_DESTINATION"],
    cache=cache_directory,
    token=os.environ["HF_TOKEN"],
    features=[
        "ohlcv:close",
        "ohlcv:volume",
        "sma:*",  # * will include all columns from "sma"
    ],
    filters={"ticker": ["001", "017"]},
    start="2024-01-02T08:00:00Z",
    end="2024-01-02T09:00:00Z",
    alignment="exact",
)
print(f"Remote feature store: {store.source}")
print(f"Local feature cache: {cache_directory}")

symbols = store.table("symbology")
print("Symbology sample:")
print(symbols[["ticker", "company_name", "isin", "market_code"]].sort_values("ticker").head())

result = store.features(order_by=["datetime", "ticker"])
print("Exact-aligned feature sample:")
print(result.head())

market_hours = store.table("markets").sort_values(["market_code", "weekday"])
print("Market hours:")
print(market_hours.collect())

# Use the Python API to materialize a bounded pandas training slice.
frame = store.features(
    columns=["datetime", "ticker", "close", "volume", "sma50"],
    order_by=["datetime", "ticker"],
).collect()
print(f"Pandas slice: {frame.shape}")
print(frame.head())

# Stream the same bounded lazy frame in smaller time windows.
for batch in store.feature_batches(
    result,
    start="2024-01-02T08:00:00Z",
    end="2024-01-02T09:00:00Z",
    window=timedelta(minutes=30),
):
    df = batch.collect()
    print(df.head())
