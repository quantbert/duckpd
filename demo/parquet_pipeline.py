"""Show lazy Parquet input and a direct DuckDB-backed Parquet write."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas

import duckpd as pd


def main() -> None:
    events = pandas.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "category": ["view", "purchase", "purchase", "refund"],
            "amount": [0.0, 45.0, 80.0, -20.0],
        }
    )

    with TemporaryDirectory(prefix="duckpd-demo-") as temporary_directory:
        directory = Path(temporary_directory)
        source_path = directory / "events.parquet"
        output_path = directory / "large-purchases.parquet"
        events.to_parquet(source_path, index=False)

        with pd.connect(memory_limit="64MB", threads=1) as session:
            lazy_events = session.read_parquet(source_path)
            large_purchases = lazy_events[
                (lazy_events["category"] == "purchase") & (lazy_events["amount"] >= 50)
            ].assign(amount_with_fee=lambda frame: frame["amount"] * 1.03)[
                ["event_id", "amount_with_fee"]
            ]

            print(f"Executions before explain/write: {session.execution_count}")
            print("\nPlan:")
            print(large_purchases.explain())

            large_purchases.write_parquet(output_path)
            print(f"\nExecutions after explain/write: {session.execution_count}")

        print("\nDirect Parquet output:")
        print(pandas.read_parquet(output_path).to_string(index=False))


if __name__ == "__main__":
    main()
