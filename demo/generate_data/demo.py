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
    # filters={"ticker": ["001", "017"]},
    start="2010-01-01T00:00:00Z",
    end="2025-01-01T00:00:00Z",
    alignment="point_in_time",
)
print(f"Remote feature store: {store.source}")
print(f"Local feature cache: {cache_directory}")

symbols = store.table("symbology")
print("Symbology sample:")
print(symbols[["ticker", "company_name", "isin", "market_code"]].sort_values("ticker").head())

result = store.features(order_by=["datetime", "ticker"])
print("Aligned feature sample:")
print(result.head())

market_hours = store.table("markets").sort_values(["market_code", "weekday"])
print("Market hours:")
print(market_hours.collect())

# Dont like SQL queries? Use the Python API to slice features and return a Pandas DataFrame.
# DuckDB supports pandas, polars, arrow etc. as output formats
frame = store.features(
    start="2024-01-02T08:00:00Z",
    end="2024-01-02T09:00:00Z",
    columns=["datetime", "ticker", "close", "volume", "sma50"],
    order_by=["datetime", "ticker"],
).collect()
print(f"Pandas slice: {frame.shape}")
print(frame.head())

# Leaving columns and order params empty will grab all features and columns as is
frame = store.features(
    start="2024-01-02T08:00:00Z",
    end="2024-01-02T09:00:00Z",
).collect()

# You can also leave out time params and just get everything.
# frame = store.features().to_df()
# But be careful with this on large datasets!
# Inside features() is a JOIN operator blocking operator that will load all features into memory.
# This is fine for small datasets, but for large datasets it can be slow and memory intensive.

# A better way it to use the feature_batches() method to stream data in smaller batches.
# Time windows bound the data DuckDB aligns for each training batch.
for batch in store.feature_batches(window=timedelta(days=1000)):
    df = batch.collect()
    print(df.head())

# Or if you like to be more specific in time period and features (just like the features() method)
for batch in store.feature_batches(
    result,
    start="2024-01-01T00:00:00Z",
    end="2025-01-01T00:00:00Z",
    window=timedelta(days=30),
):
    df = batch.collect()
    print(df.head())
