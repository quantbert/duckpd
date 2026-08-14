"""Show eager pandas-compatible reductions over a lazy DuckPD frame."""

from __future__ import annotations

import pandas

import duckpd as pd


def main() -> None:
    trades = pandas.DataFrame(
        {
            "trade_id": [1001, 1002, 1003, 1004],
            "symbol": ["AAPL", "MSFT", "AAPL", "NVDA"],
            "quantity": [10, 5, 8, 2],
            "price": [220.0, 415.0, None, 130.0],
            "settled": [True, True, False, True],
        }
    )

    with pd.connect() as session:
        lazy_trades = session.from_pandas(trades, index="trade_id")
        numeric_trades = lazy_trades.assign(
            notional=lambda frame: frame["quantity"] * frame["price"]
        )

        print("Lazy frame before reductions:")
        print(numeric_trades)
        print(f"Executions: {session.execution_count}")

        print("\nNon-null values by visible column:")
        print(numeric_trades.count().to_string())
        print(f"Executions: {session.execution_count}")

        print("\nColumn sums (strings excluded explicitly):")
        print(numeric_trades.sum(numeric_only=True).to_string())
        print(f"Executions: {session.execution_count}")

        notional = numeric_trades["notional"]
        print("\nNotional Series reductions:")
        print(f"size (including nulls): {notional.size}")
        print(f"count (excluding nulls): {notional.count()}")
        print(f"sum: {notional.sum():.2f}")
        print(f"mean: {notional.mean():.2f}")
        print(f"minimum: {notional.min():.2f}")
        print(f"maximum: {notional.max():.2f}")
        print(f"sum(skipna=False): {notional.sum(skipna=False)}")
        print(f"sum(min_count=4): {notional.sum(min_count=4)}")

        print(f"\nFinal execution count: {session.execution_count}")
        print("Each reduction above is an explicit one-query execution boundary.")


if __name__ == "__main__":
    main()