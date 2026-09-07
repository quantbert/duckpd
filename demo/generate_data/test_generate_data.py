from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from build_catalog import DATASETS, build_catalog, inspect_partition, validate_embedding_columns
from gendata import (
    assign_news_rows,
    generate_dataset,
    generate_markets,
    generate_news_dataset,
    generate_sma_dataset,
    generate_symbology,
    session_minutes,
    write_table_dataset,
)
from hfupload import parse_destination
from news_config import NEWS_MODEL

import duckpd as pd


class UploadConfigurationTests(unittest.TestCase):
    def test_parses_supported_destinations(self) -> None:
        self.assertEqual(
            parse_destination("hf://buckets/owner/store"),
            ("buckets", "owner/store"),
        )
        self.assertEqual(
            parse_destination("hf://datasets/owner/store/"),
            ("datasets", "owner/store"),
        )

    def test_rejects_invalid_destination(self) -> None:
        with self.assertRaisesRegex(ValueError, "destination must be"):
            parse_destination("owner/store")


class CatalogTests(unittest.TestCase):
    def test_uses_configured_store_name_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            for dataset in ("ohlcv", "sma"):
                partition = data_root / dataset / "year=2024"
                partition.mkdir(parents=True)
                pq.write_table(
                    pa.table(
                        {
                            "datetime": pa.array(
                                [datetime(2024, 1, 2, 8, 0, tzinfo=UTC)],
                                type=pa.timestamp("us", tz="UTC"),
                            )
                        }
                    ),
                    partition / "data.parquet",
                )
            write_table_dataset(
                data_root / "symbols" / "data.parquet",
                generate_symbology(["007"]),
                overwrite=False,
            )
            write_table_dataset(
                data_root / "markets" / "data.parquet",
                generate_markets(),
                overwrite=False,
            )

            catalog = build_catalog(
                data_root,
                store_name="owner/store",
                source="hf://buckets/owner/store",
            )
            readme = (data_root / "README.md").read_text(encoding="utf-8")

        self.assertEqual(catalog["name"], "owner/store")
        self.assertEqual(catalog["catalog_version"], 1)
        self.assertEqual(
            [dataset["name"] for dataset in catalog["datasets"]],
            ["ohlcv", "sma", "symbology", "markets"],
        )
        self.assertEqual(catalog["datasets"][0]["min_time"], "2024-01-02T08:00:00Z")
        self.assertEqual(catalog["datasets"][2]["primary_key"], ["ticker"])
        self.assertNotIn("primary_key", catalog["datasets"][3])
        self.assertIn('source="hf://buckets/owner/store"', readme)

    def test_adds_news_model_and_monthly_partition_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            for dataset in ("ohlcv", "sma"):
                partition = data_root / dataset / "year=2024"
                partition.mkdir(parents=True)
                pq.write_table(
                    pa.table(
                        {
                            "datetime": pa.array(
                                [datetime(2024, 1, 2, 8, 0, tzinfo=UTC)],
                                type=pa.timestamp("us", tz="UTC"),
                            )
                        }
                    ),
                    partition / "data.parquet",
                )
            news_partition = data_root / "news" / "year=2024" / "month=01"
            news_partition.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "datetime": pa.array(
                            [datetime(2024, 1, 2, 8, 0, tzinfo=UTC)],
                            type=pa.timestamp("us", tz="UTC"),
                        ),
                        "ticker": ["007"],
                        "embedding": pa.array(
                            [[0.0] * NEWS_MODEL.dimension],
                            type=pa.list_(pa.float32(), NEWS_MODEL.dimension),
                        ),
                    }
                ),
                news_partition / "data.parquet",
            )
            (data_root / "news" / "_SUCCESS.json").write_text(
                json.dumps({"source_rows": 1}),
                encoding="utf-8",
            )
            write_table_dataset(
                data_root / "symbols" / "data.parquet",
                generate_symbology(["007"]),
                overwrite=False,
            )
            write_table_dataset(
                data_root / "markets" / "data.parquet",
                generate_markets(),
                overwrite=False,
            )

            catalog = build_catalog(data_root, "owner/store", str(data_root))

        news = next(dataset for dataset in catalog["datasets"] if dataset["name"] == "news")
        self.assertEqual(news["partitioning"]["unit"], "month")
        self.assertIn("bge-small-en-v1.5", catalog["embedding_models"])
        self.assertEqual(
            catalog["features"]["news:embedding"]["embedding_model"],
            "bge-small-en-v1.5",
        )

    def test_rejects_wrong_news_embedding_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            pq.write_table(
                pa.table(
                    {
                        "embedding": pa.array(
                            [[0.0, 1.0]],
                            type=pa.list_(pa.float32(), 2),
                        )
                    }
                ),
                path,
            )

            with self.assertRaisesRegex(ValueError, "dimension 384"):
                validate_embedding_columns([path], DATASETS["news"])

    def test_rejects_non_utc_time_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            pq.write_table(
                pa.table(
                    {
                        "datetime": pa.array(
                            [datetime.fromisoformat("2024-01-02T09:00:00+01:00")]
                        )
                    }
                ),
                path,
            )

            with self.assertRaisesRegex(ValueError, "timezone-aware UTC"):
                inspect_partition(path, Path(directory), "datetime")


class GeneratedDatasetTests(unittest.TestCase):
    def test_generates_partitioned_embedded_news_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "news.parquet"
            output = root / "generated" / "news"
            pq.write_table(
                pa.table(
                    {
                        "publish_date": ["Jan 1, 2026"] * 6,
                        "symbol": ["NVDA", "AMD", "NVDA", "AMD", "NVDA", "AMD"],
                        "title": [f"Title {index}" for index in range(6)],
                        "description": ["Body", None, "Body", "", "Body", "Body"],
                        "publisher": ["Publisher"] * 6,
                        "url": [f"https://example.com/{index}" for index in range(6)],
                        "source": ["fixture"] * 6,
                    }
                ),
                source,
            )
            calls: list[list[str]] = []

            def embed(documents: Sequence[str]) -> pa.Array[Any]:
                calls.append(list(documents))
                values = [[float(index)] * NEWS_MODEL.dimension for index in range(len(documents))]
                return pa.array(values, type=pa.list_(pa.float32(), NEWS_MODEL.dimension))

            generate_news_dataset(
                source,
                output,
                start=date(2024, 1, 2),
                end=date(2024, 3, 2),
                ticker_start=7,
                ticker_count=2,
                overwrite=False,
                batch_size=2,
                embed_documents=embed,
                verify_pinned_source=False,
            )
            generated = pq.read_table(output)
            first_call_count = len(calls)
            generate_news_dataset(
                source,
                output,
                start=date(2024, 1, 2),
                end=date(2024, 3, 2),
                ticker_start=7,
                ticker_count=2,
                overwrite=False,
                batch_size=2,
                embed_documents=embed,
                verify_pinned_source=False,
            )

        self.assertEqual(generated.num_rows, 6)
        self.assertEqual(generated.column("ticker").to_pylist(), ["007", "008"] * 3)
        self.assertEqual(len(set(generated.column("document_id").to_pylist())), 6)
        self.assertEqual(
            generated.schema.field("embedding").type,
            pa.list_(pa.float32(), NEWS_MODEL.dimension),
        )
        self.assertEqual(calls[0], ["Title 0\n\nBody", "Title 1"])
        self.assertEqual(len(calls), first_call_count)

    def test_spreads_news_over_tickers_and_trading_minutes(self) -> None:
        timestamps = [
            datetime(2024, 1, 2, 8, minute, tzinfo=UTC)
            for minute in range(5)
        ]

        tickers, assigned = assign_news_rows(8, ["000", "001"], timestamps)

        self.assertEqual(tickers, ["000", "001"] * 4)
        self.assertEqual(assigned, [timestamps[index] for index in (0, 0, 1, 1, 2, 2, 4, 4)])
        self.assertEqual(len(set(zip(tickers, assigned, strict=True))), 8)

    def test_rejects_more_news_than_available_slots(self) -> None:
        timestamps = [datetime(2024, 1, 2, 8, 0, tzinfo=UTC)]

        with self.assertRaisesRegex(ValueError, "available unique"):
            assign_news_rows(3, ["000", "001"], timestamps)

    def test_builds_a_queryable_duckpd_feature_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            tickers = ["007"]
            generate_dataset(
                data_root / "ohlcv",
                start=date(2024, 1, 2),
                end=date(2024, 1, 3),
                ticker_start=7,
                ticker_count=1,
                seed=42,
                overwrite=False,
            )
            generate_sma_dataset(data_root / "ohlcv", data_root / "sma", overwrite=False)
            write_table_dataset(
                data_root / "symbols" / "data.parquet",
                generate_symbology(tickers),
                overwrite=False,
            )
            write_table_dataset(
                data_root / "markets" / "data.parquet",
                generate_markets(),
                overwrite=False,
            )
            build_catalog(data_root, "owner/store", str(data_root))

            frame = pd.FeatureStore(data_root).features(
                ["ohlcv:close", "sma:sma10"],
                start="2024-01-02T08:00:00Z",
                end="2024-01-02T08:05:00Z",
                alignment="exact",
            )
            result = frame.collect()

        self.assertEqual(result.shape, (5, 4))
        self.assertEqual(result["ticker"].unique().tolist(), ["007"])

    def test_session_timestamps_are_normalized_to_utc(self) -> None:
        winter, summer = session_minutes([date(2024, 1, 2), date(2024, 7, 1)])[::510]

        self.assertEqual(winter, datetime(2024, 1, 2, 8, 0, tzinfo=UTC))
        self.assertEqual(summer, datetime(2024, 7, 1, 7, 0, tzinfo=UTC))

    def test_generates_one_keyed_symbology_row_per_ticker(self) -> None:
        table = generate_symbology(["007", "042"])

        self.assertEqual(table.schema.field("ticker").nullable, False)
        self.assertEqual(table.column("ticker").to_pylist(), ["007", "042"])
        self.assertEqual(table.column("isin").to_pylist(), ["SE0000000007", "SE0000000042"])
        self.assertEqual(table.column("cik").to_pylist(), ["0000000008", "0000000043"])

    def test_generates_non_keyed_market_hours_rows(self) -> None:
        table = generate_markets()

        self.assertEqual(table.num_rows, 10)
        self.assertEqual(set(table.column("market_code").to_pylist()), {"XSTO", "XNYS"})
        self.assertNotIn("id", table.column_names)

    def test_writes_single_file_table_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "symbols" / "data.parquet"
            write_table_dataset(destination, generate_symbology(["007"]), overwrite=False)

            table = pq.read_table(destination)

        self.assertEqual(table.column("ticker").to_pylist(), ["007"])


if __name__ == "__main__":
    unittest.main()