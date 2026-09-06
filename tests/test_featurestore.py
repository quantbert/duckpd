"""Unit and integration tests for DuckPD FeatureStore (Phase 1 & Phase 2)."""

from __future__ import annotations

import json
import multiprocessing
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import duckpd
from duckpd.featurestore import FeatureStore, parse_availability_delay, parse_timestamp


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


def test_remote_feature_store_validation() -> None:
    # Non-hf remote uri
    with pytest.raises(ValueError, match="supports hf:// URIs"):
        FeatureStore(source="http://remote.store/data", cache="/tmp/cache")

    # Remote hf without cache directory
    with pytest.raises(ValueError, match="local cache directory is required"):
        FeatureStore(source="hf://datasets/test/store")


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


def test_remote_missing_catalog_raises() -> None:
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
    ) -> Path:
        calls.append(source_uri)
        return ensure_cached_partition(
            str(feature_store_fixture),
            cache_root,
            relative_path,
            needed_columns,
            con,
            filters_sql,
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
