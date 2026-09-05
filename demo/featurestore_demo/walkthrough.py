"""DuckPD Feature Store: End-to-End Walkthrough Demo.

This script demonstrates connecting to the remote Hugging Face Feature Store:
    hf://datasets/hifinab/fdb

It covers:
1. Pure lazy connection and metadata inspection without downloading whole datasets.
2. Static reference table retrieval (symbology).
3. Exact feature alignment (equi-join across datasets).
4. Point-in-time (ASOF) alignment with availability_delay to eliminate lookahead bias.
5. High-speed local cache acceleration (Phase 3 partition mirroring).
6. Rich composition with DuckPD DataFrames: .assign(), filtering, .merge(), and streaming.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path

import duckpd as pd


def load_env_token() -> str | None:
    """Read HF_TOKEN from environment or .env file."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    # Look for .env in project root or current working dir
    for candidate in (Path(".env"), Path(__file__).resolve().parents[2] / ".env"):
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["HF_TOKEN"] = token
                    return token
    return None


def main() -> None:
    print("=" * 80)
    print(" DuckPD Feature Store: Native Partition-Mirrored Demo")
    print("=" * 80)

    token = load_env_token()
    cache_dir = Path(__file__).resolve().parent / ".cache" / "fdb"
    remote_source = "hf://datasets/hifinab/fdb"

    print("\n1. Connecting to Feature Store:")
    print(f"   Source: {remote_source}")
    print(f"   Local Cache: {cache_dir}")
    print(f"   Auth Token Configured: {bool(token)}")

    store = pd.FeatureStore(
        source=remote_source,
        cache=cache_dir,
        token=token,
    )

    # 1. Inspect catalog
    catalog = store.catalog()
    print("\n2. Remote Catalog Inspection (Zero heavy downloads):")
    print(f"   Catalog Name: {catalog.get('name')}")
    print(f"   Catalog Version: {catalog.get('catalog_version')}")
    print(f"   Available Datasets: {[d['name'] for d in catalog.get('datasets', [])]}")
    print(f"   Registered Features ({len(catalog.get('features', {}))} total):")
    for feat_name, meta in list(catalog.get("features", {}).items())[:6]:
        delay = meta.get("availability_delay", "N/A")
        print(f"     - {feat_name:20s} (Delay: {delay}, Safe: {meta.get('lookahead_safe')})")

    # 2. Reference Table Access
    print("\n3. Querying Static Reference Table ('symbology'):")
    t0 = time.perf_counter()
    symbols = store.table("symbology")
    symbols_sample = symbols.sort_values("ticker").head(5)
    print(f"   Loaded in {time.perf_counter() - t0:.4f}s:")
    display_cols = ["ticker", "company_name", "market_code"]
    if "sector" in symbols_sample.columns:
        display_cols.append("sector")
    print(symbols_sample[display_cols])

    # 3. Exact Alignment
    print("\n4. Multi-Family Exact Alignment:")
    print("   Joining 'ohlcv:close' and 'sma:sma200' on exact timestamps...")
    t0 = time.perf_counter()
    exact_features = store.features(
        features={
            "price": "ohlcv:close",
            "sma200": "sma:sma200",
        },
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T09:00:00Z",
        filters={"ticker": ["001", "002"]},
        alignment="exact",
        order_by=["datetime", "ticker"],
    )
    exact_df = exact_features.collect()
    print(f"   Fetched {len(exact_df)} rows in {time.perf_counter() - t0:.4f}s:")
    print(exact_df.head(4))

    # 4. Point-In-Time Alignment (Lookahead Protection)
    print("\n5. Point-in-Time (ASOF) Alignment with Availability Delays:")
    print("   Spine: ohlcv (minute clock).")
    print("   'open' available immediately (PT0S); 'close' available after bar close (PT1M).")
    t0 = time.perf_counter()
    pit_features = store.features(
        features={
            "open": "ohlcv:open",
            "close": "ohlcv:close",
            "sma50": "sma:sma50",
            "sma200": "sma:sma200",
        },
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:06:00Z",
        filters={"ticker": ["001"]},
        alignment="point_in_time",
        spine="ohlcv",
        order_by=["datetime"],
    )
    pit_df = pit_features.collect()
    print(f"   Fetched {len(pit_df)} rows in {time.perf_counter() - t0:.4f}s:")
    print(pit_df)
    print("   Notice: At 08:00, 'close' is NaN because the 08:00 bar close is only known at 08:01!")

    # 5. Composition with DuckPD DataFrames
    print("\n6. Rich DuckPD Composition (Signals & Reference Merging):")
    signals = pit_features.assign(
        trend=lambda df: df["close"] / df["sma200"],
        spread=lambda df: df["close"] - df["open"],
    ).merge(symbols, on="ticker", how="left")
    signals_df = signals.collect()
    print("   Computed trend ratio & merged with company metadata:")
    print(signals_df[["datetime", "ticker", "company_name", "close", "trend", "spread"]])

    # 6. Windowed Batch Streaming for ML / PyTorch
    print("\n7. Windowed Batch Streaming (Zero-Copy Arrow Batches):")
    total_batches = 0
    total_rows = 0
    t0 = time.perf_counter()
    for batch_df in store.feature_batches(
        exact_features,
        window=timedelta(minutes=30),
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T09:00:00Z",
    ):
        with batch_df.to_arrow_batches(batch_size=10_000) as reader:
            for arrow_batch in reader:
                total_batches += 1
                total_rows += arrow_batch.num_rows

    elapsed = time.perf_counter() - t0
    print(f"   Streamed {total_batches} windowed batches ({total_rows} rows) in {elapsed:.4f}s.")

    # 7. Summary of Local Cache Directory
    print("\n8. Local Cache Footprint (~/.cache/fdb):")
    for p in sorted(cache_dir.rglob("*")):
        if p.is_file():
            print(f"   {p.relative_to(cache_dir)}: {p.stat().st_size / 1e6:.2f} MB")

    print("\n" + "=" * 80)
    print(" Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
