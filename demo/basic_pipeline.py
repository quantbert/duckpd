"""Show basic lazy transformations over an in-memory pandas DataFrame."""

from __future__ import annotations

import pandas

import duckpd as pd


def main() -> None:
    orders = pandas.DataFrame(
        {
            "order_id": [101, 102, 103, 104],
            "status": ["paid", "pending", "paid", "paid"],
            "gross": [120.0, 80.0, 200.0, 65.0],
            "refund": [0.0, 0.0, 25.0, 5.0],
        }
    )

    lazy_orders = pd.from_pandas(orders).set_index("order_id")
    paid_orders = (
        lazy_orders[lazy_orders["status"] == "paid"]
        .assign(net=lambda frame: frame["gross"] - frame["refund"])
        .sort_values("net", ascending=False)[["net"]]
    )

    print("Lazy frame (no rows have been collected):")
    print(paid_orders)
    print(f"Index: {paid_orders.index_names}")
    print(f"Ordering: {paid_orders.ordering}")
    print("\nTop two paid orders:")
    print(paid_orders.head(2).to_string())
    print("\nComplete pandas result:")
    print(paid_orders.collect().to_string())


if __name__ == "__main__":
    main()
