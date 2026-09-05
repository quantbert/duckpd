

import duckpd as pd

features = pd.FeatureStore(
    source="hf://datasets/hifinab/fdb",
    cache="feature_cache",
    token=getenv("HF_TOKEN"),
    features={"close": "ohlcv:close", "long_average": "sma:sma200"},
    start="2024-01-01T00:00:00Z",
    end="2025-01-01T00:00:00Z",
    alignment="exact",
)

training_data = pd.FeatureStore(
    source="hf://datasets/hifinab/fdb",
    cache="feature_cache",
    token=getenv("HF_TOKEN"),
    features=["ohlcv:close", "sma:sma200"],
    start="2024-01-01T00:00:00Z",
    end="2025-01-01T00:00:00Z",
    alignment="point_in_time",
)

store = FeatureStore(
    source="data",
    features={
        "price": "prices:close",
        "trend": "signals:sma50",
    },
    start="2024-01-01T00:00:00Z",
    end="2025-01-01T00:00:00Z",
    filters={"ticker": ["001", "017"]},
    alignment="point_in_time",
)