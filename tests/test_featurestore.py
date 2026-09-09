"""Unit and integration tests for DuckPD FeatureStore (Phase 1 & Phase 2)."""

from __future__ import annotations

import json
import multiprocessing
import tempfile
from collections.abc import Generator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import duckpd
from duckpd.embeddings import (
    EmbeddedQuery,
    EmbeddingModelSpec,
    PreparedModelInfo,
    _make_fixed_array,
)
from duckpd.errors import UnsupportedOperationError
from duckpd.featurestore import FeatureStore, parse_availability_delay, parse_timestamp


class CatalogEmbeddingProvider:
    def __init__(self, specification: EmbeddingModelSpec) -> None:
        self._specification = specification
        self.prepare_calls = 0
        self.query_calls: list[str] = []

    @property
    def specification(self) -> EmbeddingModelSpec:
        return self._specification

    def prepare(self) -> PreparedModelInfo:
        self.prepare_calls += 1
        return PreparedModelInfo(
            self.specification.fingerprint,
            "test",
            None,
            "catalog-test-artifact",
            ("CPUExecutionProvider",),
        )

    def embed_documents(self, texts: Sequence[str]) -> pa.Array[Any]:
        return _make_fixed_array([[1.0, 0.0, 0.0] for _ in texts], 3)

    def embed_query(self, text: str) -> EmbeddedQuery:
        self.query_calls.append(text)
        return EmbeddedQuery((1.0, 0.0, 0.0), self.specification.fingerprint)


def _add_embedding_catalog(root: Path) -> EmbeddingModelSpec:
    model = duckpd.embedding_model(
        "test/catalog-fastembed",
        revision="0123456789abcdef0123456789abcdef01234567",
        dimension=3,
    )
    partition = root / "ohlcv" / "year=2024" / "data.parquet"
    table = pq.read_table(partition)  # pyright: ignore[reportUnknownMemberType]
    vectors = _make_fixed_array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.5, 0.5, 0.0],
            [0.2, 0.8, 0.0],
            [0.0, 1.0, 0.0],
        ],
        3,
    )
    pq.write_table(  # pyright: ignore[reportUnknownMemberType]
        table.append_column("embedding", vectors), partition
    )

    symbols = root / "symbols" / "data.parquet"
    symbol_table = pq.read_table(symbols)  # pyright: ignore[reportUnknownMemberType]
    symbol_vectors = _make_fixed_array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 3)
    pq.write_table(  # pyright: ignore[reportUnknownMemberType]
        symbol_table.append_column("embedding", symbol_vectors), symbols
    )

    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["embedding_models"] = {"catalog-model": model.to_dict()}
    catalog["features"]["ohlcv:embedding"] = {
        "dataset": "ohlcv",
        "name": "embedding",
        "availability_delay": "PT0S",
        "lookahead_safe": True,
        "embedding_model": "catalog-model",
    }
    next(entry for entry in catalog["datasets"] if entry["name"] == "symbology")["columns"] = {
        "embedding": {"embedding_model": "catalog-model"}
    }
    catalog_path.write_text(json.dumps(catalog))
    return model


def _cache_partition_worker(
    source: str,
    cache: str,
    column: str,
    start_event: Any,
) -> None:
    import duckdb

    from duckpd._feature_sources import ensure_cached_partition

    start_event.wait(timeout=10)
    connection = duckdb.connect()
    try:
        ensure_cached_partition(
            source,
            Path(cache),
            "ohlcv/year=2024/data.parquet",
            ["datetime", "ticker", column],
            connection,
        )
    finally:
        connection.close()


@pytest.fixture
def feature_store_fixture() -> Generator[Path, None, None]:
    """Create a temporary multi-family feature store fixture."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        timestamps = [datetime(2024, 1, 2, 8, minute, tzinfo=UTC) for minute in range(5)]

        # 1. ohlcv family
        ohlcv_df = pd.DataFrame(
            {
                "datetime": timestamps,
                "ticker": ["001"] * 5,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )
        p1 = root / "ohlcv" / "year=2024"
        p1.mkdir(parents=True)
        ohlcv_df.to_parquet(p1 / "data.parquet", index=False)

        # 2. sma family
        sma_df = pd.DataFrame(
            {
                "datetime": timestamps,
                "ticker": ["001"] * 5,
                "sma10": [99.5, 100.0, 100.5, 101.0, 101.5],
            }
        )
        p2 = root / "sma" / "year=2024"
        p2.mkdir(parents=True)
        sma_df.to_parquet(p2 / "data.parquet", index=False)

        # 3. reference table (symbology)
        sym_dir = root / "symbols"
        sym_dir.mkdir(parents=True)
        sym_df = pd.DataFrame(
            {
                "ticker": ["001", "002"],
                "company_name": ["Example Corp 1", "Example Corp 2"],
                "sector": ["Tech", "Finance"],
            }
        )
        sym_df.to_parquet(sym_dir / "data.parquet", index=False)

        # 4. catalog.json
        features = {
            f"ohlcv:{name}": {
                "dataset": "ohlcv",
                "name": name,
                "availability_delay": "PT0S" if name == "open" else "PT1M",
                "lookahead_safe": True,
            }
            for name in ("open", "high", "low", "close", "volume")
        }
        features["sma:sma10"] = {
            "dataset": "sma",
            "name": "sma10",
            "availability_delay": "PT1M",
            "lookahead_safe": True,
        }

        catalog = {
            "catalog_version": 1,
            "name": "test/store",
            "datasets": [
                {
                    "name": "ohlcv",
                    "kind": "timeseries",
                    "time_column": "datetime",
                    "series_keys": ["ticker"],
                    "path_template": "ohlcv/year={year}/data.parquet",
                    "min_time": "2024-01-01T00:00:00Z",
                    "max_time": "2024-12-31T23:59:59Z",
                },
                {
                    "name": "sma",
                    "kind": "timeseries",
                    "time_column": "datetime",
                    "series_keys": ["ticker"],
                    "path_template": "sma/year={year}/data.parquet",
                    "min_time": "2024-01-01T00:00:00Z",
                    "max_time": "2024-12-31T23:59:59Z",
                },
                {
                    "name": "symbology",
                    "kind": "table",
                    "primary_key": ["ticker"],
                    "path_template": "symbols/data.parquet",
                },
            ],
            "features": features,
        }

        (root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        yield root


def test_timestamp_and_delay_parsers() -> None:
    dt = parse_timestamp("2024-01-02T08:00:00Z")
    assert dt == datetime(2024, 1, 2, 8, 0, tzinfo=UTC)

    delay = parse_availability_delay("PT1M", "test:feature")
    assert delay == timedelta(minutes=1)

    delay_zero = parse_availability_delay("PT0S", "test:open")
    assert delay_zero == timedelta(seconds=0)

    delay_days = parse_availability_delay("P1DT2H30M15S", "test:complex")
    assert delay_days == timedelta(days=1, hours=2, minutes=30, seconds=15)

    with pytest.raises(TypeError, match="timestamp must be an ISO 8601 string"):
        parse_timestamp(12345)  # type: ignore

    with pytest.raises(ValueError, match="Invalid ISO timestamp"):
        parse_timestamp("not-a-timestamp")

    with pytest.raises(ValueError, match="Timestamp must include a timezone offset"):
        parse_timestamp("2024-01-01T00:00:00")

    with pytest.raises(ValueError, match="must define availability_delay"):
        parse_availability_delay(None, "foo")

    with pytest.raises(ValueError, match="Invalid availability_delay"):
        parse_availability_delay("P1Y", "bar")


def test_featurestore_initialization_pure_laziness(feature_store_fixture: Path) -> None:
    session = duckpd.connect()
    # Constructor does NOT execute any queries
    store = duckpd.FeatureStore(
        source=feature_store_fixture,
        session=session,
        features={"price": "ohlcv:close"},
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:04:00Z",
        alignment="exact",
    )
    assert session.execution_count == 0
    assert store.source == str(feature_store_fixture.resolve())
    assert store.session is session

    catalog = store.catalog()
    assert catalog["catalog_version"] == 1
    assert "ohlcv" in [d["name"] for d in catalog["datasets"]]


def test_featurestore_reference_table(feature_store_fixture: Path) -> None:
    session = duckpd.connect()
    store = session.feature_store(source=feature_store_fixture)

    symbology_df = store.table("symbology")
    # Table reference should be a lazy DataFrame
    assert isinstance(symbology_df, duckpd.DataFrame)
    assert session.execution_count == 0

    collected = symbology_df.collect()
    assert len(collected) == 2
    assert list(collected["ticker"]) == ["001", "002"]
    assert session.execution_count == 1


def test_exact_alignment_execution(feature_store_fixture: Path) -> None:
    session = duckpd.connect()
    store = session.feature_store(
        source=feature_store_fixture,
        features={
            "price": "ohlcv:close",
            "indicator": "sma:sma10",
        },
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:03:00Z",
        filters={"ticker": ["001"]},
        alignment="exact",
    )

    df = store.features()
    assert isinstance(df, duckpd.DataFrame)
    assert session.execution_count == 0

    # Collect result
    result = df.collect()
    assert len(result) == 3
    assert list(result.columns) == ["datetime", "ticker", "price", "indicator"]
    assert list(result["price"]) == [100.5, 101.5, 102.5]
    assert list(result["indicator"]) == [99.5, 100.0, 100.5]


def test_wildcard_feature_resolution(feature_store_fixture: Path) -> None:
    store = duckpd.FeatureStore(
        source=feature_store_fixture,
        features=["ohlcv:*"],
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:02:00Z",
        alignment="exact",
    )
    df = store.features()
    result = df.collect()
    assert len(result) == 2
    assert "open" in result.columns
    assert "close" in result.columns
    assert "high" in result.columns
    assert "low" in result.columns
    assert "volume" in result.columns


def test_point_in_time_alignment(feature_store_fixture: Path) -> None:
    session = duckpd.connect()
    store = session.feature_store(
        source=feature_store_fixture,
        features={
            "open_val": "ohlcv:open",
            "close_val": "ohlcv:close",
            "sma_val": "sma:sma10",
        },
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:03:00Z",
        alignment="point_in_time",
        spine="ohlcv",
    )

    df = store.features(order_by=["datetime"])
    assert isinstance(df, duckpd.DataFrame)
    assert session.execution_count == 0

    res = df.collect()
    # At 08:00, open (PT0S delay) is 100.0
    # close (PT1M delay) is not available at 08:00 (recorded at 08:00, available at 08:01) -> NULL
    # At 08:01, close_val is 100.5
    assert len(res) == 3
    assert res["open_val"].iloc[0] == 100.0
    assert pd.isna(res["close_val"].iloc[0])
    assert res["close_val"].iloc[1] == 100.5
    assert res["close_val"].iloc[2] == 101.5


def test_point_in_time_alignment_honors_catalog_history_lookback(
    feature_store_fixture: Path,
) -> None:
    from unittest.mock import patch

    catalog_path = feature_store_fixture / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    for dataset in catalog["datasets"]:
        if dataset["kind"] == "timeseries":
            dataset["min_time"] = "2020-01-01T00:00:00Z"
            dataset["history_lookback"] = "P7D"
    catalog_path.write_text(json.dumps(catalog))

    store = duckpd.FeatureStore(feature_store_fixture)
    calls: list[tuple[str, datetime]] = []
    original = store._timeseries_frame

    def capture(
        dataset: str,
        start: datetime,
        end: datetime,
        needed_features: Sequence[str],
    ) -> duckpd.DataFrame:
        calls.append((dataset, start))
        return original(dataset, start, end, needed_features)

    query_start = datetime(2024, 1, 2, 8, tzinfo=UTC)
    with patch.object(store, "_timeseries_frame", side_effect=capture):
        store.features(
            features={"close": "ohlcv:close", "sma": "sma:sma10"},
            start=query_start.isoformat(),
            end="2024-01-02T08:03:00Z",
            alignment="point_in_time",
            spine="ohlcv",
        )

    bounded_start = query_start - timedelta(days=7)
    assert ("sma", bounded_start) in calls
    assert calls.count(("ohlcv", bounded_start)) == 1
    assert all(start >= bounded_start for _, start in calls)


def test_feature_batches_iterator(feature_store_fixture: Path) -> None:
    store = duckpd.FeatureStore(
        source=feature_store_fixture,
        features={"price": "ohlcv:close"},
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:04:00Z",
        alignment="exact",
    )

    batches = list(store.feature_batches(window=timedelta(minutes=2)))
    assert len(batches) == 2
    b1 = batches[0].collect()
    b2 = batches[1].collect()
    assert len(b1) == 2
    assert len(b2) == 2
    assert list(b1["price"]) == [100.5, 101.5]
    assert list(b2["price"]) == [102.5, 103.5]


def test_composition_with_duckpd_dataframe_methods(feature_store_fixture: Path) -> None:
    store = duckpd.FeatureStore(
        source=feature_store_fixture,
        features={"close": "ohlcv:close", "open": "ohlcv:open"},
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:05:00Z",
        alignment="exact",
    )
    df = store.features()

    # Assign new column
    enriched = df.assign(spread=lambda d: d["close"] - d["open"])
    # Filter
    filtered = enriched[enriched["spread"] > 0.4]
    result = filtered.collect()
    assert len(result) == 5
    assert all(result["spread"] == 0.5)

    # Merge with reference table
    symbology = store.table("symbology")
    merged = df.merge(symbology, on="ticker", how="left")
    collected_merged = merged.collect()
    assert "sector" in collected_merged.columns
    assert collected_merged["sector"].iloc[0] == "Tech"


def test_catalog_validation_and_errors() -> None:
    from duckpd._feature_catalog import validate_catalog

    # Bad catalog version
    with pytest.raises(ValueError, match="Unsupported catalog version"):
        validate_catalog({"catalog_version": 2})

    # No datasets
    with pytest.raises(ValueError, match="Catalog must define at least one dataset"):
        validate_catalog({"catalog_version": 1, "datasets": []})

    # Duplicate dataset
    with pytest.raises(ValueError, match="Duplicate catalog dataset"):
        validate_catalog(
            {
                "catalog_version": 1,
                "datasets": [
                    {"name": "d1", "kind": "table"},
                    {"name": "d1", "kind": "table"},
                ],
            }
        )

    # Bad kind
    with pytest.raises(ValueError, match="invalid kind"):
        validate_catalog(
            {
                "catalog_version": 1,
                "datasets": [{"name": "d1", "kind": "unknown"}],
            }
        )

    # Reserved dataset name 'features'
    with pytest.raises(ValueError, match="reserved"):
        validate_catalog(
            {
                "catalog_version": 1,
                "datasets": [{"name": "features", "kind": "table"}],
            }
        )

    # Timeseries missing time_column or series_keys
    with pytest.raises(ValueError, match="requires time_column"):
        validate_catalog(
            {
                "catalog_version": 1,
                "datasets": [{"name": "ts", "kind": "timeseries"}],
            }
        )

    with pytest.raises(ValueError, match="requires series_keys"):
        validate_catalog(
            {
                "catalog_version": 1,
                "datasets": [{"name": "ts", "kind": "timeseries", "time_column": "dt"}],
            }
        )

    # Features must be a mapping
    with pytest.raises(ValueError, match="features must be a mapping"):
        validate_catalog(
            {
                "catalog_version": 1,
                "datasets": [{"name": "t1", "kind": "table"}],
                "features": [],
            }
        )

    # Feature key must equal dataset:name
    with pytest.raises(ValueError, match="must be keyed as"):
        validate_catalog(
            {
                "catalog_version": 1,
                "datasets": [
                    {"name": "ts", "kind": "timeseries", "time_column": "dt", "series_keys": ["k"]}
                ],
                "features": {"bad_key": {"dataset": "ts", "name": "f1"}},
            }
        )

    # Feature must belong to a timeseries dataset
    with pytest.raises(ValueError, match="must belong to a timeseries dataset"):
        validate_catalog(
            {
                "catalog_version": 1,
                "datasets": [{"name": "tbl", "kind": "table"}],
                "features": {"tbl:f1": {"dataset": "tbl", "name": "f1"}},
            }
        )


def test_daily_partition_and_history_lookback_validation() -> None:
    from duckpd._feature_catalog import parse_history_lookback, validate_catalog

    assert parse_history_lookback("P7D", "prices") == timedelta(days=7)
    valid = {
        "catalog_version": 1,
        "datasets": [
            {
                "name": "prices",
                "kind": "timeseries",
                "time_column": "datetime",
                "series_keys": ["ticker"],
                "partitioning": {
                    "column": "datetime",
                    "unit": "day",
                    "timezone": "UTC",
                },
                "history_lookback": "P7D",
            }
        ],
    }
    datasets, _, _ = validate_catalog(valid)
    assert datasets["prices"]["_history_lookback"] == timedelta(days=7)

    for field, value, message in (
        ("column", "event_time", "column must match"),
        ("unit", "hour", "unit must be"),
        ("timezone", "Europe/Stockholm", "timezone must be"),
    ):
        invalid = json.loads(json.dumps(valid))
        invalid["datasets"][0]["partitioning"][field] = value
        with pytest.raises(ValueError, match=message):
            validate_catalog(invalid)

    invalid_history = json.loads(json.dumps(valid))
    invalid_history["datasets"][0]["history_lookback"] = "P1Y"
    with pytest.raises(ValueError, match="Invalid history_lookback"):
        validate_catalog(invalid_history)


def test_feature_resolution_and_filters() -> None:
    from duckpd._feature_catalog import normalize_filters, resolve_features

    datasets = {
        "ts": {"name": "ts", "kind": "timeseries", "time_column": "dt", "series_keys": ["k"]}
    }
    features = {
        "ts:f1": {"dataset": "ts", "name": "f1"},
        "ts:f2": {"dataset": "ts", "name": "f2"},
    }

    # Empty features
    with pytest.raises(ValueError, match="At least one feature must be requested"):
        resolve_features([], features, datasets)

    # Invalid feature type
    with pytest.raises(TypeError, match="features must be a sequence"):
        resolve_features(123, features, datasets)  # type: ignore

    # Unknown feature
    with pytest.raises(ValueError, match="Unknown feature: non_existent"):
        resolve_features(["non_existent"], features, datasets)

    # Wildcard with alias forbidden
    with pytest.raises(ValueError, match="Feature wildcards cannot have output aliases"):
        resolve_features({"my_alias": "ts:*"}, features, datasets)

    # Unknown wildcard group
    with pytest.raises(ValueError, match="Unknown feature group"):
        resolve_features(["unknown:*"], features, datasets)

    # Output alias colliding with reserved key
    with pytest.raises(ValueError, match="Output alias is reserved: dt"):
        resolve_features({"dt": "ts:f1"}, features, datasets)

    # Duplicate output names (when resolved names conflict)
    # E.g. two different features mapping to the same default name or duplicate alias
    features_dup = {
        "ts1:val": {"dataset": "ts1", "name": "val"},
        "ts2:val": {"dataset": "ts2", "name": "val"},
    }
    datasets_dup = {
        "ts1": {"name": "ts1", "kind": "timeseries", "time_column": "dt", "series_keys": ["k"]},
        "ts2": {"name": "ts2", "kind": "timeseries", "time_column": "dt", "series_keys": ["k"]},
    }
    with pytest.raises(ValueError, match="Duplicate output columns"):
        resolve_features(["ts1:val", "ts2:val"], features_dup, datasets_dup)

    # Filters normalization
    assert normalize_filters(None) is None
    with pytest.raises(TypeError, match="filters must be a mapping"):
        normalize_filters(["ticker"])  # type: ignore
    with pytest.raises(ValueError, match="filters cannot be empty"):
        normalize_filters({})
    with pytest.raises(TypeError, match="filter values for 'ticker' must be a sequence"):
        normalize_filters({"ticker": "001"})  # string instead of sequence
    with pytest.raises(TypeError, match="filter values for 'ticker' must be strings"):
        normalize_filters({"ticker": [123]})  # type: ignore
    with pytest.raises(ValueError, match="filter values for 'ticker' cannot be empty"):
        normalize_filters({"ticker": []})

    norm = normalize_filters({"ticker": ["001", "002"]})
    assert norm == {"ticker": ["001", "002"]}


def test_featurestore_error_conditions(feature_store_fixture: Path) -> None:
    # Source not found
    with pytest.raises(FileNotFoundError, match="Feature store source directory not found"):
        FeatureStore(source="/non/existent/path/12345")

    # Catalog not found
    with (
        tempfile.TemporaryDirectory() as empty_dir,
        pytest.raises(FileNotFoundError, match="Catalog not found"),
    ):
        FeatureStore(source=empty_dir)

    # Remote hf without cache raises ValueError
    with pytest.raises(ValueError, match="local cache directory is required"):
        FeatureStore(source="hf://datasets/test/data")

    # Invalid alignment
    with pytest.raises(ValueError, match="alignment must be 'exact' or 'point_in_time'"):
        FeatureStore(
            source=feature_store_fixture,
            features={"price": "ohlcv:close"},
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:02:00Z",
            alignment="invalid",  # type: ignore
        )

    # Spine with exact alignment forbidden
    with pytest.raises(ValueError, match="spine is not supported with exact alignment"):
        FeatureStore(
            source=feature_store_fixture,
            features={"price": "ohlcv:close"},
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:02:00Z",
            alignment="exact",
            spine="ohlcv",
        )

    # Spine required for point_in_time
    with pytest.raises(ValueError, match="spine dataset name is required"):
        FeatureStore(
            source=feature_store_fixture,
            features={"price": "ohlcv:close"},
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:02:00Z",
            alignment="point_in_time",
        )

    # End before start
    with pytest.raises(ValueError, match="end must be later than start"):
        FeatureStore(
            source=feature_store_fixture,
            features={"price": "ohlcv:close"},
            start="2024-01-02T08:02:00Z",
            end="2024-01-02T08:00:00Z",
            alignment="exact",
        )

    # Table errors
    store = FeatureStore(source=feature_store_fixture)
    with pytest.raises(ValueError, match="table name must be a non-empty string"):
        store.table("")
    with pytest.raises(ValueError, match="Unknown catalog dataset"):
        store.table("unknown_table")
    with pytest.raises(ValueError, match="is a timeseries, not a table"):
        store.table("ohlcv")


def test_featurestore_sliced_calls_and_unconfigured_error(feature_store_fixture: Path) -> None:
    store = duckpd.FeatureStore(source=feature_store_fixture)

    # Calling features() without preconfigured or passed features raises
    with pytest.raises(ValueError, match="has no configured feature selection"):
        store.features()

    # Calling features with partial override
    msg = "features, start, and end must be configured or passed together"
    with pytest.raises(ValueError, match=msg):
        store.features(features=["ohlcv:open"])

    # Calling features with end <= start
    with pytest.raises(ValueError, match="end must be later than start"):
        store.features(
            features=["ohlcv:open"],
            start="2024-01-02T08:02:00Z",
            end="2024-01-02T08:00:00Z",
            alignment="exact",
        )

    # Calling features with invalid column / order_by arguments
    store_configured = duckpd.FeatureStore(
        source=feature_store_fixture,
        features={"price": "ohlcv:close"},
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:03:00Z",
        alignment="exact",
    )
    with pytest.raises(TypeError, match="columns must be a sequence"):
        store_configured.features(columns="price")  # type: ignore

    with pytest.raises(TypeError, match="order_by must be a sequence"):
        store_configured.features(order_by="datetime")  # type: ignore

    # Filter with columns projection and order_by
    df = store_configured.features(
        columns=["datetime", "ticker", "price"],
        order_by=["datetime"],
    )
    res = df.collect()
    assert list(res.columns) == ["datetime", "ticker", "price"]
    assert len(res) == 3


def test_feature_batches_validation(feature_store_fixture: Path) -> None:
    store = duckpd.FeatureStore(source=feature_store_fixture)

    with pytest.raises(TypeError, match=r"window must be a datetime\.timedelta"):
        list(store.feature_batches(window="1 day"))  # type: ignore

    with pytest.raises(ValueError, match="window must be positive"):
        list(store.feature_batches(window=timedelta(0)))

    # Without configured start/end
    msg = "FeatureStore start and end must be specified or configured"
    with pytest.raises(ValueError, match=msg):
        list(store.feature_batches(window=timedelta(days=1)))

    # With end <= start
    with pytest.raises(ValueError, match="end must be later than start"):
        list(
            store.feature_batches(
                window=timedelta(days=1),
                start="2024-01-02T08:00:00Z",
                end="2024-01-02T08:00:00Z",
            )
        )

    # Passing existing frame to feature_batches
    store_conf = duckpd.FeatureStore(
        source=feature_store_fixture,
        features={"price": "ohlcv:close"},
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:04:00Z",
        alignment="exact",
    )
    df = store_conf.features()
    batches = list(store_conf.feature_batches(frame=df, window=timedelta(minutes=2)))
    assert len(batches) == 2
    assert len(batches[0].collect()) == 2
    assert len(batches[1].collect()) == 2


def test_pit_alignment_validation(feature_store_fixture: Path) -> None:
    # Feature missing lookahead_safe
    catalog_path = feature_store_fixture / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["features"]["ohlcv:unsafe"] = {
        "dataset": "ohlcv",
        "name": "unsafe",
        "lookahead_safe": False,
        "availability_delay": "PT0S",
    }
    catalog_path.write_text(json.dumps(catalog))

    store = duckpd.FeatureStore(source=feature_store_fixture)
    with pytest.raises(ValueError, match="is not marked lookahead_safe"):
        store.features(
            features=["ohlcv:unsafe"],
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:02:00Z",
            alignment="point_in_time",
            spine="ohlcv",
        )

    # Incompatible series_keys / time_column for PIT
    catalog["datasets"].append(
        {
            "name": "incompatible_ts",
            "kind": "timeseries",
            "time_column": "other_time",
            "series_keys": ["ticker"],
            "path_template": "incompat/data.parquet",
        }
    )
    catalog["features"]["incompatible_ts:val"] = {
        "dataset": "incompatible_ts",
        "name": "val",
        "lookahead_safe": True,
        "availability_delay": "PT0S",
    }
    catalog_path.write_text(json.dumps(catalog))
    store_incompat = duckpd.FeatureStore(source=feature_store_fixture)
    with pytest.raises(ValueError, match=r"compatible time_column and series_keys"):
        store_incompat.features(
            features=["incompatible_ts:val"],
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:02:00Z",
            alignment="point_in_time",
            spine="ohlcv",
        )


def test_metadata_and_path_resolution(feature_store_fixture: Path) -> None:
    from duckpd._feature_sources import (
        available_interval,
        get_dataset_path_template,
        load_dataset_metadata,
        resolve_partition_paths,
    )

    root = feature_store_fixture
    ohlcv_entry = {
        "name": "ohlcv",
        "kind": "timeseries",
        "metadata": "ohlcv/metadata.json",
        "time_column": "datetime",
        "series_keys": ["ticker"],
        "min_time": "2024-01-01T00:00:00Z",
        "max_time": "2024-12-31T23:59:59Z",
    }

    # metadata.json does not exist initially
    meta = load_dataset_metadata(root, ohlcv_entry)
    assert meta == {}

    # write metadata.json with storage path_template
    meta_file = root / "ohlcv" / "metadata.json"
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(
        json.dumps({"storage": {"path_template": "ohlcv/year={year}/data.parquet"}})
    )
    loaded = load_dataset_metadata(root, ohlcv_entry)
    assert loaded["storage"]["path_template"] == "ohlcv/year={year}/data.parquet"

    # get_dataset_path_template fallback
    ts_template = get_dataset_path_template(root, {"name": "sample", "kind": "timeseries"})
    assert ts_template == "sample/year={year}/data.parquet"
    tbl_template = get_dataset_path_template(root, {"name": "sample", "kind": "table"})
    assert tbl_template == "sample/data.parquet"

    # available_interval clipping
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, tzinfo=UTC)
    clipped = available_interval(ohlcv_entry, start, end)
    assert clipped is not None
    assert clipped[0] == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    # Out of range interval
    out_of_range = available_interval(
        ohlcv_entry,
        datetime(2025, 2, 1, tzinfo=UTC),
        datetime(2025, 3, 1, tzinfo=UTC),
    )
    assert out_of_range is None

    # resolve_partition_paths for table
    sym_entry = {
        "name": "symbology",
        "kind": "table",
        "path_template": "symbols/data.parquet",
    }
    paths = resolve_partition_paths(root, sym_entry, start, end)
    assert len(paths) == 1
    assert paths[0].endswith("symbols/data.parquet")


def test_catalog_and_filter_edge_cases() -> None:
    from duckpd._feature_catalog import resolve_features

    # Single feature string sequence
    datasets = {
        "ts": {"name": "ts", "kind": "timeseries", "time_column": "dt", "series_keys": ["k"]}
    }
    features = {"ts:f1": {"dataset": "ts", "name": "f1"}}
    resolved = resolve_features(["f1"], features, datasets)
    assert len(resolved) == 1
    assert resolved[0][0] == "f1"
    assert resolved[0][1] == "ts:f1"

    # Ambiguous short feature name
    features_ambig = {
        "ts1:f1": {"dataset": "ts1", "name": "f1"},
        "ts2:f1": {"dataset": "ts2", "name": "f1"},
    }
    datasets_ambig = {
        "ts1": {"name": "ts1", "kind": "timeseries", "time_column": "dt", "series_keys": ["k"]},
        "ts2": {"name": "ts2", "kind": "timeseries", "time_column": "dt", "series_keys": ["k"]},
    }
    with pytest.raises(ValueError, match="Ambiguous feature 'f1'"):
        resolve_features(["f1"], features_ambig, datasets_ambig)

    # Empty string in feature reference
    with pytest.raises(ValueError, match="Feature references must be non-empty strings"):
        resolve_features([""], features, datasets)

    with pytest.raises(ValueError, match="Feature references must be non-empty strings"):
        resolve_features({"alias": ""}, features, datasets)


def test_session_feature_store_method(feature_store_fixture: Path) -> None:
    session = duckpd.connect()
    fs = session.feature_store(
        source=feature_store_fixture,
        features={"c": "ohlcv:close"},
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:02:00Z",
        alignment="exact",
    )
    assert fs.session is session
    df = fs.features()
    res = df.collect()
    assert len(res) == 2

    # Exact alignment incompatible keys error
    cat_path = feature_store_fixture / "catalog.json"
    cat = json.loads(cat_path.read_text())
    cat["datasets"].append(
        {
            "name": "mismatched",
            "kind": "timeseries",
            "time_column": "diff_dt",
            "series_keys": ["ticker"],
            "path_template": "mismatch/data.parquet",
        }
    )
    cat["features"]["mismatched:val"] = {
        "dataset": "mismatched",
        "name": "val",
    }
    cat_path.write_text(json.dumps(cat))
    fs_mismatch = duckpd.FeatureStore(source=feature_store_fixture)
    with pytest.raises(ValueError, match="compatible time_column and series_keys"):
        fs_mismatch.features(
            features=["ohlcv:close", "mismatched:val"],
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:02:00Z",
            alignment="exact",
        )


def test_remote_feature_store_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid HTTP feature store URI"):
        FeatureStore(source="http:///data", cache="/tmp/cache")

    # Remote hf without cache directory
    with pytest.raises(ValueError, match="local cache directory is required"):
        FeatureStore(source="hf://datasets/test/store")

    with pytest.raises(FileNotFoundError, match="Catalog not found:"):
        FeatureStore(source=tmp_path, catalog_path=tmp_path / "missing.json")


def test_ensure_cached_partition_and_table(feature_store_fixture: Path) -> None:
    from duckpd._feature_sources import (
        ensure_cached_partition,
        ensure_cached_table,
        file_contains_columns,
    )

    with tempfile.TemporaryDirectory() as cache_tmp:
        cache_path = Path(cache_tmp)

        # 1. file_contains_columns on missing file
        assert not file_contains_columns(cache_path / "nonexistent.parquet", ["col1"])

        # 2. Mock filesystem for ensure_cached_table
        class MockFS:
            def open(self, path: str, mode: str = "rb"):  # type: ignore
                return open(feature_store_fixture / "symbols" / "data.parquet", mode)

        cached_table = ensure_cached_table(
            "hf://datasets/test/store",
            cache_path,
            "symbols/data.parquet",
            MockFS(),
        )
        assert cached_table.is_file()
        assert file_contains_columns(cached_table, ["ticker", "company_name"])

        # Second call returns existing file
        cached_table_2 = ensure_cached_table(
            "hf://datasets/test/store",
            cache_path,
            "symbols/data.parquet",
            MockFS(),
        )
        assert cached_table_2 == cached_table

        # 3. Test ensure_cached_partition with local DuckDB connection
        import duckdb

        con = duckdb.connect()
        cached_part = ensure_cached_partition(
            str(feature_store_fixture),
            cache_path,
            "ohlcv/year=2024/data.parquet",
            ["datetime", "ticker", "close"],
            con,
        )
        assert cached_part.is_file()
        assert file_contains_columns(cached_part, ["datetime", "ticker", "close"])

        # Second call returns existing file immediately
        cached_part_2 = ensure_cached_partition(
            str(feature_store_fixture),
            cache_path,
            "ohlcv/year=2024/data.parquet",
            ["datetime", "ticker", "close"],
            con,
        )
        assert cached_part_2 == cached_part


def test_cached_embedding_partition_preserves_fixed_size_schema(tmp_path: Path) -> None:
    import duckdb

    from duckpd._feature_sources import ensure_cached_partition

    source = tmp_path / "source"
    relative_path = "news/year=2024/month=01/day=02/part.parquet"
    partition = source / relative_path
    partition.parent.mkdir(parents=True)
    embedding_type = pa.list_(pa.float32(), 2)
    schema = pa.schema([pa.field("embedding", embedding_type)])
    pq.write_table(  # pyright: ignore[reportUnknownMemberType]
        pa.Table.from_arrays(
            [pa.array([[1.0, 2.0]], type=embedding_type)],
            schema=schema,
        ),
        partition,
    )

    cached = ensure_cached_partition(
        str(source),
        tmp_path / "cache",
        relative_path,
        ["embedding"],
        duckdb.connect(),
        embedding_columns={"embedding": 2},
    )

    assert pq.ParquetFile(cached).schema_arrow.equals(schema)


def test_remote_feature_store_mock_flow(feature_store_fixture: Path) -> None:
    """Exercise remote featurestore workflow with a mock HfFileSystem."""
    with tempfile.TemporaryDirectory() as cache_tmp:
        cache_path = Path(cache_tmp)

        class MockHfFileSystem:
            def __init__(self, token: str | None = None) -> None:
                self.token = token

            def exists(self, path: str) -> bool:
                clean_path = path.replace("datasets/test/fdb/", "")
                return (feature_store_fixture / clean_path).exists()

            def open(self, path: str, mode: str = "rb"):  # type: ignore
                clean_path = path.replace("datasets/test/fdb/", "")
                return open(feature_store_fixture / clean_path, mode)

        session = duckpd.connect()
        # Seed catalog in cache so __init__ doesn't hit remote API
        (cache_path / "catalog.json").write_text(
            (feature_store_fixture / "catalog.json").read_text()
        )

        from unittest.mock import patch

        with patch("huggingface_hub.HfFileSystem", return_value=MockHfFileSystem()):
            store = FeatureStore(
                source="hf://datasets/test/fdb",
                cache=cache_path,
                session=session,
                token="mock-token",
            )

            # 1. Catalog discovery
            catalog = store.catalog()
            assert catalog["name"] == "test/store"

            # 2. Table planning reads schema metadata but defers the full cache write
            sym_df = store.table("symbology")
            assert not (cache_path / "symbols" / "data.parquet").exists()
            assert sym_df.collect()["ticker"].iloc[0] == "001"
            assert (cache_path / "symbols" / "data.parquet").is_file()

            # 3. Features with exact alignment when cache is populated
            ohlcv_cache = cache_path / "ohlcv" / "year=2024"
            ohlcv_cache.mkdir(parents=True)
            import shutil

            shutil.copy(
                feature_store_fixture / "ohlcv" / "year=2024" / "data.parquet",
                ohlcv_cache / "data.parquet",
            )
            feats = store.features(
                features={"price": "ohlcv:close"},
                start="2024-01-02T08:00:00Z",
                end="2024-01-02T08:03:00Z",
                alignment="exact",
            )
            res = feats.collect()
            assert len(res) == 3
            assert list(res["price"]) == [100.5, 101.5, 102.5]

        # Table dataset with {year} error on remote
        store._dataset_entries["invalid_table"] = {
            "name": "invalid_table",
            "kind": "table",
            "path_template": "table/year={year}/data.parquet",
        }
        with pytest.raises(ValueError, match="cannot have year partition template"):
            store.table("invalid_table")

        # Table dataset not a table error on remote
        store._dataset_entries["ts_as_table"] = {
            "name": "ts_as_table",
            "kind": "timeseries",
            "time_column": "datetime",
            "series_keys": ["ticker"],
        }
        with pytest.raises(ValueError, match="is a timeseries, not a table"):
            store.table("ts_as_table")


def test_spine_dataset_errors(feature_store_fixture: Path) -> None:
    # Spine dataset not in catalog error
    with pytest.raises(ValueError, match="Unknown spine dataset"):
        FeatureStore(
            source=feature_store_fixture,
            features={"price": "ohlcv:close"},
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:02:00Z",
            alignment="point_in_time",
            spine="nonexistent_spine",
        )

    # Spine dataset not a timeseries
    with pytest.raises(ValueError, match="must be a timeseries dataset"):
        FeatureStore(
            source=feature_store_fixture,
            features={"price": "ohlcv:close"},
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:02:00Z",
            alignment="point_in_time",
            spine="symbology",
        )


def test_remote_missing_catalog_raises(feature_store_fixture: Path) -> None:
    """Test remote feature store where remote catalog does not exist."""
    with tempfile.TemporaryDirectory() as cache_tmp:
        cache_path = Path(cache_tmp)

        class EmptyHfFileSystem:
            def __init__(self, token: str | None = None) -> None:
                pass

            def open(self, path: str, mode: str = "rb"):
                raise FileNotFoundError(f"File {path} not found")

        from unittest.mock import patch

        with (
            patch("huggingface_hub.HfFileSystem", return_value=EmptyHfFileSystem()),
            pytest.raises(FileNotFoundError, match="Remote catalog not found"),
        ):
            FeatureStore(
                source="hf://datasets/nonexistent/fdb",
                cache=cache_path,
            )

        cached = cache_path / "catalog.json"
        cached.write_text(
            (feature_store_fixture / "catalog.json").read_text(),
            encoding="utf-8",
        )
        with patch("huggingface_hub.HfFileSystem", return_value=EmptyHfFileSystem()):
            store = FeatureStore(
                source="hf://datasets/unavailable/fdb",
                cache=cache_path,
            )
        assert store.catalog()["name"] == "test/store"

        # Failed cache replacement removes its temporary file.

        class CatalogHfFileSystem:
            def open(self, path: str, mode: str = "r"):
                return (feature_store_fixture / "catalog.json").open(mode)

        with (
            patch("huggingface_hub.HfFileSystem", return_value=CatalogHfFileSystem()),
            patch("duckpd.featurestore.os.replace", side_effect=OSError("cache write failed")),
            pytest.raises(OSError, match="cache write failed"),
        ):
            FeatureStore(
                source="hf://datasets/test/fdb",
                cache=cache_path,
            )
        assert not list(cache_path.glob(".catalog.json.*.tmp"))
        # Test ImportError when huggingface_hub is missing
        with (
            patch.dict("sys.modules", {"huggingface_hub": None}),
            pytest.raises(ImportError, match="huggingface-hub is required"),
        ):
            FeatureStore(
                source="hf://datasets/test/fdb",
                cache=cache_path,
            )


def test_featurestore_sync_method(feature_store_fixture: Path) -> None:
    """Test store.sync() pre-warming helper."""
    store = duckpd.FeatureStore(source=feature_store_fixture)

    # 1. Sync tables
    report_tbl = store.sync(tables=["symbology"])
    assert report_tbl.tables_synced == 1
    assert report_tbl.partitions_synced == 0

    # 2. Sync features with explicit start and end
    report_feat = store.sync(
        features={"price": "ohlcv:close"},
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:04:00Z",
    )
    assert report_feat.partitions_synced >= 1
    assert report_feat.bytes_written > 0

    # 3. Validation errors
    with pytest.raises(ValueError, match="Unknown catalog dataset"):
        store.sync(tables=["unknown_table"])

    with pytest.raises(ValueError, match="is not a table dataset"):
        store.sync(tables=["ohlcv"])

    with pytest.raises(ValueError, match="start and end must be specified"):
        store.sync(features=["ohlcv:close"])

    with pytest.raises(ValueError, match="end must be later than start"):
        store.sync(
            features=["ohlcv:close"],
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:00:00Z",
        )


def test_cache_paths_cannot_escape_root(feature_store_fixture: Path, tmp_path: Path) -> None:
    import duckdb

    from duckpd._feature_sources import (
        ensure_cached_partition,
        ensure_cached_table,
        load_dataset_metadata,
    )

    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="remain relative"):
        ensure_cached_partition(
            str(feature_store_fixture),
            cache,
            "../outside.parquet",
            ["ticker"],
            duckdb.connect(),
        )
    with pytest.raises(ValueError, match="remain relative"):
        ensure_cached_table(
            "hf://datasets/test/store",
            cache,
            "../outside.parquet",
            object(),
        )
    with pytest.raises(ValueError, match="remain relative"):
        load_dataset_metadata(
            feature_store_fixture,
            {"metadata": "../outside.json"},
        )
    assert not (tmp_path / "outside.parquet").exists()


def test_concurrent_cache_expansion_preserves_all_columns(
    feature_store_fixture: Path,
    tmp_path: Path,
) -> None:
    from duckpd._feature_sources import file_contains_columns

    cache = tmp_path / "cache"
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    requested = ["open", "high", "low", "close", "volume"]
    processes = [
        context.Process(
            target=_cache_partition_worker,
            args=(str(feature_store_fixture), str(cache), column, start_event),
        )
        for column in requested
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    cached = cache / "ohlcv" / "year=2024" / "data.parquet"
    assert file_contains_columns(cached, ["datetime", "ticker", *requested])


def test_remote_feature_planning_defers_partition_fetch(
    feature_store_fixture: Path,
    tmp_path: Path,
) -> None:
    from unittest.mock import patch

    from duckpd._feature_sources import ensure_cached_partition

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "catalog.json").write_text((feature_store_fixture / "catalog.json").read_text())

    class MockHfFileSystem:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

    calls: list[str] = []

    def project_partition(
        source_uri: str,
        cache_root: Path,
        relative_path: str,
        needed_columns: list[str],
        con: Any,
        filters_sql: str | None = None,
        embedding_columns: dict[str, int] | None = None,
    ) -> Path:
        calls.append(source_uri)
        return ensure_cached_partition(
            str(feature_store_fixture),
            cache_root,
            relative_path,
            needed_columns,
            con,
            filters_sql,
            embedding_columns,
        )

    with (
        patch("huggingface_hub.HfFileSystem", return_value=MockHfFileSystem()),
        patch(
            "duckpd._feature_sources.ensure_cached_partition",
            side_effect=project_partition,
        ),
    ):
        store = FeatureStore(
            source="hf://datasets/test/fdb",
            cache=cache,
        )
        frame = store.features(
            features={"price": "ohlcv:close"},
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:03:00Z",
            alignment="exact",
        )
        assert calls == []
        assert "FeatureParquetSource" in frame.explain(mode="logical")
        result = frame.collect()
        assert calls == ["hf://datasets/test/fdb"]
        assert list(result["price"]) == [100.5, 101.5, 102.5]


def test_remote_constructor_does_not_create_cache(
    feature_store_fixture: Path,
    tmp_path: Path,
) -> None:
    from unittest.mock import patch

    class MockHfFileSystem:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

    cache = tmp_path / "not-created"
    with patch("huggingface_hub.HfFileSystem", return_value=MockHfFileSystem()):
        FeatureStore(
            source="hf://datasets/test/fdb",
            cache=cache,
            catalog_path=feature_store_fixture / "catalog.json",
        )
    assert not cache.exists()


def test_point_in_time_uses_predecessor_older_than_one_year(tmp_path: Path) -> None:
    spine_path = tmp_path / "spine" / "year=2024"
    sparse_path = tmp_path / "sparse" / "year=2022"
    spine_path.mkdir(parents=True)
    sparse_path.mkdir(parents=True)
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02T00:00:00Z"]),
            "ticker": ["A"],
        }
    ).to_parquet(spine_path / "data.parquet", index=False)
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2022-12-31T00:00:00Z"]),
            "ticker": ["A"],
            "signal": [42],
        }
    ).to_parquet(sparse_path / "data.parquet", index=False)
    catalog = {
        "catalog_version": 1,
        "datasets": [
            {
                "name": "spine",
                "kind": "timeseries",
                "time_column": "datetime",
                "series_keys": ["ticker"],
                "min_time": "2024-01-02T00:00:00Z",
                "max_time": "2024-01-02T00:00:00Z",
            },
            {
                "name": "sparse",
                "kind": "timeseries",
                "time_column": "datetime",
                "series_keys": ["ticker"],
                "min_time": "2022-12-31T00:00:00Z",
                "max_time": "2022-12-31T00:00:00Z",
            },
        ],
        "features": {
            "sparse:signal": {
                "dataset": "sparse",
                "name": "signal",
                "availability_delay": "PT0S",
                "lookahead_safe": True,
            }
        },
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))

    frame = FeatureStore(source=tmp_path).features(
        features=["sparse:signal"],
        start="2024-01-02T00:00:00Z",
        end="2024-01-03T00:00:00Z",
        alignment="point_in_time",
        spine="spine",
    )
    logical = frame.explain(mode="logical")
    assert "AsOfJoinPlan" in logical
    assert "SqlSource" not in logical
    result = frame.collect()
    assert list(result["signal"]) == [42]


def test_feature_batches_ignores_catalog_dataset_order(
    feature_store_fixture: Path,
) -> None:
    catalog_path = feature_store_fixture / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["datasets"] = [
        catalog["datasets"][2],
        catalog["datasets"][0],
        catalog["datasets"][1],
    ]
    catalog_path.write_text(json.dumps(catalog))

    store = FeatureStore(
        source=feature_store_fixture,
        features={"price": "ohlcv:close"},
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:04:00Z",
        alignment="exact",
    )
    frame = store.features()
    batches = list(store.feature_batches(frame, window=timedelta(minutes=2)))
    assert [list(batch.collect()["price"]) for batch in batches] == [
        [100.5, 101.5],
        [102.5, 103.5],
    ]


def test_exact_alignment_supports_two_aliases_for_one_feature(
    feature_store_fixture: Path,
) -> None:
    frame = FeatureStore(source=feature_store_fixture).features(
        features={"first": "ohlcv:close", "second": "ohlcv:close"},
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:03:00Z",
        alignment="exact",
    )
    result = frame.collect()
    assert list(result.columns) == ["datetime", "ticker", "first", "second"]
    assert list(result["first"]) == [100.5, 101.5, 102.5]
    assert result["first"].equals(result["second"])


def test_remote_timeseries_uses_metadata_path_template(tmp_path: Path) -> None:
    from unittest.mock import patch

    from duckpd._feature_sources import ensure_cached_partition

    backing = tmp_path / "backing"
    custom_partition = backing / "custom" / "year=2024"
    custom_partition.mkdir(parents=True)
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02T08:00:00Z"]),
            "ticker": ["001"],
            "close": [100.5],
        }
    ).to_parquet(custom_partition / "values.parquet", index=False)
    metadata_path = backing / "ohlcv" / "metadata.json"
    metadata_path.parent.mkdir()
    metadata_path.write_text(
        json.dumps(
            {
                "storage": {
                    "path_template": "custom/year={year}/values.parquet",
                }
            }
        )
    )
    catalog = {
        "catalog_version": 1,
        "datasets": [
            {
                "name": "ohlcv",
                "kind": "timeseries",
                "time_column": "datetime",
                "series_keys": ["ticker"],
                "metadata": "ohlcv/metadata.json",
                "min_time": "2024-01-01T00:00:00Z",
                "max_time": "2024-12-31T23:59:59Z",
            }
        ],
        "features": {
            "ohlcv:close": {
                "dataset": "ohlcv",
                "name": "close",
                "availability_delay": "PT0S",
                "lookahead_safe": True,
            }
        },
    }
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "catalog.json").write_text(json.dumps(catalog))

    class MockHfFileSystem:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def exists(self, path: str) -> bool:
            relative = path.replace("datasets/test/fdb/", "")
            return (backing / relative).exists()

        def open(self, path: str, mode: str = "rb"):  # type: ignore
            relative = path.replace("datasets/test/fdb/", "")
            return open(backing / relative, mode)

    requested_paths: list[str] = []

    def project_partition(
        source_uri: str,
        cache_root: Path,
        relative_path: str,
        needed_columns: list[str],
        con: Any,
        filters_sql: str | None = None,
        embedding_columns: dict[str, int] | None = None,
    ) -> Path:
        requested_paths.append(relative_path)
        return ensure_cached_partition(
            str(backing),
            cache_root,
            relative_path,
            needed_columns,
            con,
            filters_sql,
            embedding_columns,
        )

    with (
        patch("huggingface_hub.HfFileSystem", return_value=MockHfFileSystem()),
        patch(
            "duckpd._feature_sources.ensure_cached_partition",
            side_effect=project_partition,
        ),
    ):
        frame = FeatureStore(
            source="hf://datasets/test/fdb",
            cache=cache,
        ).features(
            features=["ohlcv:close"],
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:01:00Z",
            alignment="exact",
        )
        assert requested_paths == []
        result = frame.collect()

    assert requested_paths == ["custom/year=2024/values.parquet"]
    assert list(result["close"]) == [100.5]


def test_monthly_partition_paths_obey_half_open_interval(tmp_path: Path) -> None:
    from duckpd._feature_sources import (
        get_dataset_path_template,
        partition_paths_for_interval,
        resolve_partition_paths,
    )

    entry = {
        "name": "prices",
        "kind": "timeseries",
        "partitioning": {"unit": "month"},
        "time_column": "datetime",
        "series_keys": ["ticker"],
    }
    path_template = get_dataset_path_template(tmp_path, entry)
    assert path_template == "prices/year={year}/month={month:02d}/data.parquet"

    expected = [
        "prices/year=2024/month=12/data.parquet",
        "prices/year=2025/month=01/data.parquet",
    ]
    assert (
        partition_paths_for_interval(
            entry,
            path_template,
            datetime(2024, 12, 31, 23, 59, tzinfo=UTC),
            datetime(2025, 2, 1, tzinfo=UTC),
        )
        == expected
    )

    for relative_path in expected:
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True)
        pd.DataFrame({"value": [1]}).to_parquet(target, index=False)
    assert [
        Path(path).relative_to(tmp_path).as_posix()
        for path in resolve_partition_paths(
            tmp_path,
            entry,
            datetime(2024, 12, 31, 23, 59, tzinfo=UTC),
            datetime(2025, 2, 1, tzinfo=UTC),
        )
    ] == expected


def test_daily_partition_paths_obey_half_open_interval(tmp_path: Path) -> None:
    from duckpd._feature_sources import get_dataset_path_template, partition_paths_for_interval

    entry = {
        "name": "prices",
        "kind": "timeseries",
        "partitioning": {"unit": "day"},
        "time_column": "datetime",
        "series_keys": ["ticker"],
    }
    path_template = get_dataset_path_template(tmp_path, entry)
    assert path_template == ("prices/year={year}/month={month:02d}/day={day:02d}/part.parquet")
    assert partition_paths_for_interval(
        entry,
        path_template,
        datetime(2024, 12, 31, 23, 59, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
    ) == [
        "prices/year=2024/month=12/day=31/part.parquet",
        "prices/year=2025/month=01/day=01/part.parquet",
    ]
    bounded_entry = {
        **entry,
        "min_time": "2025-01-01T00:00:00Z",
        "max_time": "2025-01-02T00:00:00Z",
    }
    assert (
        partition_paths_for_interval(
            bounded_entry,
            path_template,
            datetime(2024, 12, 1, tzinfo=UTC),
            datetime(2024, 12, 2, tzinfo=UTC),
        )
        == []
    )


def test_http_monthly_store_fetches_only_intersecting_partitions(tmp_path: Path) -> None:
    from unittest.mock import patch

    from duckpd._feature_sources import ensure_cached_partition

    backing = tmp_path / "backing"
    cache = tmp_path / "cache"
    for timestamp, month, value in (
        (datetime(2024, 12, 31, 23, 59, tzinfo=UTC), 12, 10.0),
        (datetime(2025, 1, 1, 0, 0, tzinfo=UTC), 1, 11.0),
    ):
        partition = backing / f"prices/year={timestamp.year}/month={month:02d}"
        partition.mkdir(parents=True)
        pd.DataFrame({"datetime": [timestamp], "ticker": ["001"], "value": [value]}).to_parquet(
            partition / "data.parquet", index=False
        )

    catalog = {
        "catalog_version": 1,
        "name": "test/http-monthly",
        "datasets": [
            {
                "name": "prices",
                "kind": "timeseries",
                "time_column": "datetime",
                "series_keys": ["ticker"],
                "path_template": ("prices/year={year}/month={month:02d}/data.parquet"),
            }
        ],
        "features": {
            "prices:value": {
                "dataset": "prices",
                "name": "value",
                "availability_delay": "PT0S",
                "lookahead_safe": True,
            }
        },
    }
    catalog_path = backing / "catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    calls: list[tuple[str, str]] = []

    def project_partition(
        source_uri: str,
        cache_root: Path,
        relative_path: str,
        needed_columns: list[str],
        con: Any,
        filters_sql: str | None = None,
        embedding_columns: dict[str, int] | None = None,
    ) -> Path:
        calls.append((source_uri, relative_path))
        return ensure_cached_partition(
            str(backing),
            cache_root,
            relative_path,
            needed_columns,
            con,
            filters_sql,
            embedding_columns,
        )

    with patch(
        "duckpd._feature_sources.ensure_cached_partition",
        side_effect=project_partition,
    ):
        frame = FeatureStore(
            source="https://features.example.test/store",
            cache=cache,
            catalog_path=catalog_path,
        ).features(
            features=["prices:value"],
            start="2024-12-31T23:59:00Z",
            end="2025-02-01T00:00:00Z",
            alignment="exact",
        )
        assert calls == []
        result = frame.collect()

    assert calls == [
        (
            "https://features.example.test/store",
            "prices/year=2024/month=12/data.parquet",
        ),
        (
            "https://features.example.test/store",
            "prices/year=2025/month=01/data.parquet",
        ),
    ]
    assert list(result["value"]) == [10.0, 11.0]


def test_http_feature_store_end_to_end(tmp_path: Path) -> None:
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    backing = tmp_path / "backing"
    partition = backing / "prices" / "year=2024"
    partition.mkdir(parents=True)
    pd.DataFrame(
        {
            "datetime": [datetime(2024, 1, 2, tzinfo=UTC)],
            "ticker": ["001"],
            "value": [42.0],
        }
    ).to_parquet(partition / "data.parquet", index=False)
    catalog = {
        "catalog_version": 1,
        "name": "test/http",
        "datasets": [
            {
                "name": "prices",
                "kind": "timeseries",
                "time_column": "datetime",
                "series_keys": ["ticker"],
                "path_template": "prices/year={year}/data.parquet",
            }
        ],
        "features": {
            "prices:value": {
                "dataset": "prices",
                "name": "value",
                "availability_delay": "PT0S",
                "lookahead_safe": True,
            }
        },
    }
    (backing / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(QuietHandler, directory=str(backing)),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        source = f"http://127.0.0.1:{server.server_port}"
        result = (
            FeatureStore(source=source, cache=tmp_path / "cache")
            .features(
                features=["prices:value"],
                start="2024-01-01T00:00:00Z",
                end="2024-02-01T00:00:00Z",
                alignment="exact",
            )
            .collect()
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert list(result["value"]) == [42.0]


def test_catalog_embedding_schema_is_strict_and_dimension_checked(
    feature_store_fixture: Path,
) -> None:
    _add_embedding_catalog(feature_store_fixture)
    catalog_path = feature_store_fixture / "catalog.json"
    valid = json.loads(catalog_path.read_text())

    unknown_field = json.loads(json.dumps(valid))
    unknown_field["embedding_models"]["catalog-model"]["unqualified"] = True
    catalog_path.write_text(json.dumps(unknown_field))
    with pytest.raises(ValueError, match="unknown fields"):
        FeatureStore(feature_store_fixture)

    unknown_reference = json.loads(json.dumps(valid))
    unknown_reference["features"]["ohlcv:embedding"]["embedding_model"] = "missing"
    catalog_path.write_text(json.dumps(unknown_reference))
    with pytest.raises(ValueError, match="unknown embedding model"):
        FeatureStore(feature_store_fixture)

    catalog_path.write_text(json.dumps(valid))
    symbols = feature_store_fixture / "symbols" / "data.parquet"
    table = pq.read_table(symbols).drop(  # pyright: ignore[reportUnknownMemberType]
        ["embedding"]
    )
    pq.write_table(  # pyright: ignore[reportUnknownMemberType]
        table.append_column(
            "embedding",
            _make_fixed_array([[1.0, 0.0], [0.0, 1.0]], 2),
        ),
        symbols,
    )
    with pytest.raises(ValueError, match=r"requires fixed-size float32\[3\]"):
        FeatureStore(feature_store_fixture).table("symbology")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("unsupported_backend", "unsupported backend"),
        ("mutable_revision", "immutable commit"),
        ("fastembed_normalization", "cannot honor"),
        ("transformers_pooling", "pooling"),
        ("unknown_column_field", "unknown fields"),
    ],
)
def test_catalog_embedding_declarations_reject_unsafe_variants(
    feature_store_fixture: Path,
    change: str,
    message: str,
) -> None:
    _add_embedding_catalog(feature_store_fixture)
    catalog_path = feature_store_fixture / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    specification = catalog["embedding_models"]["catalog-model"]
    if change == "unsupported_backend":
        specification["backend"] = "custom"
    elif change == "mutable_revision":
        specification["revision"] = "main"
    elif change == "fastembed_normalization":
        specification["normalize"] = False
    elif change == "transformers_pooling":
        specification["backend"] = "transformers"
    else:
        next(entry for entry in catalog["datasets"] if entry["name"] == "symbology")["columns"][
            "embedding"
        ]["description"] = "not allowed"
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(ValueError, match=message):
        FeatureStore(feature_store_fixture)


def test_feature_store_embedding_policy_rejects_unsafe_limits(
    feature_store_fixture: Path,
) -> None:
    with pytest.raises(TypeError, match="auto_prepare_embeddings"):
        FeatureStore(feature_store_fixture, auto_prepare_embeddings=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout"):
        FeatureStore(feature_store_fixture, embedding_prepare_timeout_seconds=0)
    with pytest.raises(ValueError, match="download_limit"):
        FeatureStore(feature_store_fixture, embedding_download_limit_bytes=True)


def test_catalog_embedding_search_is_lazy_inferred_reused_and_equivalent(
    feature_store_fixture: Path,
) -> None:
    model = _add_embedding_catalog(feature_store_fixture)
    session = duckpd.connect()
    provider = CatalogEmbeddingProvider(model)
    session.register_embedding_provider(model, provider)
    store = FeatureStore(feature_store_fixture, session=session)

    frame = store.features(
        features={"vector": "ohlcv:embedding"},
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:05:00Z",
        alignment="exact",
    )
    vector = next(
        column for column in frame._plan.metadata.visible_columns if column.label == "vector"
    )
    assert vector.embedding is not None
    assert vector.embedding.model == store.embedding_model("catalog-model")
    assert vector.embedding.origin == "catalog"

    pit = store.features(
        features={"vector": "ohlcv:embedding"},
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:05:00Z",
        alignment="point_in_time",
        spine="ohlcv",
    )
    pit_vector = next(
        column for column in pit._plan.metadata.visible_columns if column.label == "vector"
    )
    assert pit_vector.embedding == vector.embedding

    execution_count = session.execution_count
    inferred = frame.vector.search_text(
        "ai chips",
        column="vector",
        k=3,
    )
    explained = inferred.explain("logical")
    assert session.execution_count == execution_count
    assert provider.prepare_calls == 0
    assert provider.query_calls == []
    assert '"model_origin": "catalog"' in explained
    assert '"model_prepared": false' in explained
    assert "ai chips" not in explained

    inferred_result = inferred.collect()
    assert provider.prepare_calls == 1
    explicit_result = frame.vector.search_text(
        "ai chips",
        column="vector",
        model=store.embedding_model("catalog-model"),
        k=3,
    ).collect()
    pd.testing.assert_frame_equal(inferred_result, explicit_result)

    table = store.table("symbology")
    table_vector = next(
        column for column in table._plan.metadata.visible_columns if column.label == "embedding"
    )
    assert table_vector.embedding is not None
    assert table_vector.embedding.origin == "catalog"
    table.vector.search_text(
        "ai chips",
        column="embedding",
        k=1,
        tie_breaker="ticker",
    ).collect()
    assert provider.prepare_calls == 1

    profile = inferred.profile()
    assert profile.embedding_metrics is not None
    assert profile.embedding_metrics["catalog_access_seconds"] >= 0
    assert profile.embedding_metrics["preparation_cache_reused"] == 1
    assert profile.embedding_metrics["query_inference_seconds"] >= 0


def test_disabled_catalog_auto_preparation_fails_before_partition_transfer(
    feature_store_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _add_embedding_catalog(feature_store_fixture)
    session = duckpd.connect()
    store = FeatureStore(
        "https://features.example.test/store",
        cache=tmp_path / "cache",
        catalog_path=feature_store_fixture / "catalog.json",
        session=session,
        auto_prepare_embeddings=True,
    )
    FeatureStore(
        "https://features.example.test/store",
        cache=tmp_path / "cache",
        catalog_path=feature_store_fixture / "catalog.json",
        session=session,
        auto_prepare_embeddings=False,
    )
    frame = store.features(
        features=["ohlcv:embedding"],
        start="2024-01-02T08:00:00Z",
        end="2024-01-02T08:05:00Z",
        alignment="exact",
    )
    transfers = 0

    def transferred(*_args: object, **_kwargs: object) -> list[str]:
        nonlocal transfers
        transfers += 1
        return []

    monkeypatch.setattr(
        "duckpd._feature_sources.materialize_feature_source",
        transferred,
    )
    with pytest.raises(UnsupportedOperationError, match="automatic preparation is disabled"):
        frame.vector.search_text("ai", column="embedding").collect()
    assert transfers == 0


def test_explicit_preparation_supports_disabled_catalog_policy(
    feature_store_fixture: Path,
) -> None:
    model = _add_embedding_catalog(feature_store_fixture)
    session = duckpd.connect()
    provider = CatalogEmbeddingProvider(model)
    session.register_embedding_provider(model, provider)
    store = FeatureStore(
        feature_store_fixture,
        session=session,
        auto_prepare_embeddings=False,
    )
    session.prepare_embedding_model(store.embedding_model("catalog-model"))
    result = (
        store.features(
            features=["ohlcv:embedding"],
            start="2024-01-02T08:00:00Z",
            end="2024-01-02T08:05:00Z",
            alignment="exact",
        )
        .vector.search_text("ai", column="embedding", k=1)
        .collect()
    )
    assert len(result) == 1
    assert provider.prepare_calls == 1


def test_search_text_without_verified_metadata_requires_explicit_model() -> None:
    frame = duckpd.from_pandas(pd.DataFrame({"embedding": [[1.0, 0.0, 0.0]]}))
    with pytest.raises(UnsupportedOperationError, match="cannot infer"):
        frame.vector.search_text("ai", column="embedding")
