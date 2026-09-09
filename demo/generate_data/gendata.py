"""Generate deterministic time-series and table datasets for a synthetic store.

The default invocation writes data for tickers ``000`` through ``099`` from
2010-01-01 (inclusive) to 2025-01-01 (exclusive). Time-series output uses one
Parquet file per UTC day under ``year=YYYY/month=MM/day=DD/part.parquet`` so
bounded DuckPD queries mirror the production feature-store layout. It also
writes keyed symbology and non-keyed market-hours tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from news_config import (
    NEWS_MODEL,
    NEWS_SOURCE_REVISION,
    NEWS_SOURCE_ROWS,
    NEWS_SOURCE_SHA256,
    news_model,
)

import duckpd as pd

MARKET_TIMEZONE = ZoneInfo("Europe/Stockholm")
MARKET_OPEN = time(9, 0)
MARKET_CLOSE = time(17, 30)

DAILY_PARTITION_LAYOUT = "utc-day-v1"


def daily_partition_path(root: Path, value: date) -> Path:
    """Return the canonical UTC-day Parquet path below one dataset root."""
    return root / f"year={value:%Y}" / f"month={value:%m}" / f"day={value:%d}" / "part.parquet"


def calendar_days(start: date, end: date) -> list[date]:
    """Return every date in a half-open calendar interval."""
    return [start + timedelta(days=offset) for offset in range((end - start).days)]


SMA_WINDOWS = (10, 20, 50, 200)
MARKET_CALENDAR = xcals.get_calendar("XSTO")
SCHEMA = pa.schema(
    [
        pa.field("datetime", pa.timestamp("us", tz="UTC")),
        pa.field("ticker", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)
SMA_SCHEMA = pa.schema(
    [
        pa.field("datetime", pa.timestamp("us", tz="UTC")),
        pa.field("ticker", pa.string()),
        *(pa.field(f"sma{window}", pa.float64()) for window in SMA_WINDOWS),
    ]
)
SYMBOLOGY_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("isin", pa.string(), nullable=False),
        pa.field("cik", pa.string(), nullable=False),
        pa.field("company_name", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=False),
        pa.field("market_code", pa.string(), nullable=False),
    ]
)
MARKETS_SCHEMA = pa.schema(
    [
        pa.field("market_code", pa.string(), nullable=False),
        pa.field("weekday", pa.string(), nullable=False),
        pa.field("opens_at", pa.time32("s"), nullable=False),
        pa.field("closes_at", pa.time32("s"), nullable=False),
        pa.field("timezone", pa.string(), nullable=False),
    ]
)
NEWS_SCHEMA = pa.schema(
    [
        pa.field("datetime", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("source_publish_date", pa.string(), nullable=False),
        pa.field("source_symbol", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=False),
        pa.field("publisher", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("embedding", pa.list_(pa.float32(), NEWS_MODEL.dimension), nullable=False),
    ]
)
NEWS_SOURCE_COLUMNS = (
    "publish_date",
    "symbol",
    "title",
    "description",
    "publisher",
    "url",
    "source",
)


def parse_date(value: str) -> date:
    """Parse an ISO-8601 date used for an inclusive/exclusive date range."""
    return date.fromisoformat(value)


def trading_days(start: date, end: date) -> list[date]:
    """Return Nasdaq Stockholm (XSTO) sessions in ``[start, end)``."""
    if end <= start:
        return []
    sessions = MARKET_CALENDAR.sessions_in_range(start, end - timedelta(days=1))
    return [session.date() for session in sessions]


def session_minutes(days: list[date]) -> list[datetime]:
    """Create UTC minute-bar timestamps for Stockholm sessions from 09:00 through 17:29."""
    minutes: list[datetime] = []
    for trading_day in days:
        timestamp = datetime.combine(trading_day, MARKET_OPEN, MARKET_TIMEZONE)
        close_timestamp = datetime.combine(trading_day, MARKET_CLOSE, MARKET_TIMEZONE)
        while timestamp < close_timestamp:
            minutes.append(timestamp.astimezone(UTC))
            timestamp += timedelta(minutes=1)
    return minutes


def ticker_values(ticker_start: int, ticker_count: int) -> list[str]:
    """Return zero-padded synthetic tickers after validating their range."""
    if not 0 <= ticker_start <= 999 or not 1 <= ticker_count <= 1_000 - ticker_start:
        raise ValueError("ticker range must stay within 000 through 999")
    return [f"{number:03d}" for number in range(ticker_start, ticker_start + ticker_count)]


def assign_news_rows(
    row_count: int,
    tickers: list[str],
    timestamps: list[datetime],
) -> tuple[list[str], list[datetime]]:
    """Spread source rows evenly across synthetic tickers and trading minutes."""
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if not tickers:
        raise ValueError("at least one ticker is required")
    if row_count == 0:
        return [], []
    if not timestamps:
        raise ValueError("at least one trading minute is required")

    events_per_ticker = math.ceil(row_count / len(tickers))
    if events_per_ticker > len(timestamps):
        raise ValueError("news rows exceed the available unique ticker and trading-minute slots")

    assigned_tickers: list[str] = []
    assigned_timestamps: list[datetime] = []
    for row_index in range(row_count):
        ticker_index = row_index % len(tickers)
        event_index = row_index // len(tickers)
        timestamp_index = (
            0
            if events_per_ticker == 1
            else event_index * (len(timestamps) - 1) // (events_per_ticker - 1)
        )
        assigned_tickers.append(tickers[ticker_index])
        assigned_timestamps.append(timestamps[timestamp_index])
    return assigned_tickers, assigned_timestamps


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(values: Sequence[Any], name: str, start_row: int) -> list[str]:
    result = ["" if value is None else str(value).strip() for value in values]
    if name in {"title", "publish_date", "symbol", "url"}:
        missing = next((index for index, value in enumerate(result) if not value), None)
        if missing is not None:
            raise ValueError(f"news source row {start_row + missing} has empty {name}")
    return result


def _news_generation_identity(
    source_path: Path,
    start: date,
    end: date,
    ticker_start: int,
    ticker_count: int,
    row_count: int,
    model: pd.EmbeddingModelSpec,
    embedding_device: str,
) -> dict[str, Any]:
    return {
        "source_revision": NEWS_SOURCE_REVISION,
        "source_sha256": file_sha256(source_path),
        "source_rows": row_count,
        "model_fingerprint": model.fingerprint,
        "embedding_backend": model.backend,
        "embedding_device": embedding_device,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "ticker_start": ticker_start,
        "ticker_count": ticker_count,
        "partition_layout": DAILY_PARTITION_LAYOUT,
    }


def generate_news_dataset(
    source_path: Path,
    output: Path,
    start: date,
    end: date,
    ticker_start: int,
    ticker_count: int,
    overwrite: bool,
    batch_size: int = 1_024,
    embed_documents: Callable[[Sequence[str]], pa.Array[Any]] | None = None,
    verify_pinned_source: bool = True,
    embedding_backend: Literal["fastembed", "transformers"] = "fastembed",
    embedding_device: Literal["cpu", "cuda"] = "cpu",
    transformer_batch_size: int = 64,
) -> None:
    """Embed and spread a pinned news archive over synthetic market coordinates."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if transformer_batch_size <= 0:
        raise ValueError("transformer_batch_size must be positive")
    model = news_model(embedding_backend)
    if embedding_backend == "fastembed" and embedding_device != "cpu":
        raise ValueError("FastEmbed generation supports only embedding_device='cpu'")
    if not source_path.is_file():
        raise FileNotFoundError(f"News source file not found: {source_path}")
    parquet_file = pq.ParquetFile(source_path)
    row_count = parquet_file.metadata.num_rows

    tickers = ticker_values(ticker_start, ticker_count)
    timestamps = session_minutes(trading_days(start, end))
    assigned_tickers, assigned_timestamps = assign_news_rows(row_count, tickers, timestamps)
    identity = _news_generation_identity(
        source_path,
        start,
        end,
        ticker_start,
        ticker_count,
        row_count,
        model,
        embedding_device,
    )
    if verify_pinned_source and (
        identity["source_sha256"] != NEWS_SOURCE_SHA256 or row_count != NEWS_SOURCE_ROWS
    ):
        raise ValueError("Local news source does not match the pinned artifact")
    staging = output.parent / f".{output.name}-staging"
    completion_path = output / "_SUCCESS.json"
    if completion_path.is_file() and not overwrite:
        if json.loads(completion_path.read_text(encoding="utf-8")) == identity:
            shutil.rmtree(staging, ignore_errors=True)
            print(f"Skipping completed news dataset at {output}")
            return
        raise ValueError("Existing news dataset was generated with different settings")
    identity_path = staging / "identity.json"
    if overwrite:
        shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    vector_bytes = row_count * NEWS_MODEL.dimension * 4
    free_bytes = shutil.disk_usage(staging).free
    print(
        f"News vectors require at least {vector_bytes / 1024**3:.2f} GiB before "
        f"Parquet encoding; {free_bytes / 1024**3:.2f} GiB is free"
    )
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("News staging data was generated with different settings")
    else:
        identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")

    if embed_documents is None:
        if embedding_backend == "fastembed":
            provider: pd.TextEmbeddingProvider = pd.FastEmbedProvider(model)
        else:
            provider = pd.TransformersEmbeddingProvider(
                model,
                device=embedding_device,
                batch_size=transformer_batch_size,
            )
        provider.prepare()
        embed_documents = provider.embed_documents
    embed = embed_documents
    source_row = 0
    chunk_paths: list[Path] = []
    for batch_index, batch in enumerate(
        parquet_file.iter_batches(batch_size=batch_size, columns=list(NEWS_SOURCE_COLUMNS))
    ):
        batch_rows = batch.num_rows
        chunk_path = staging / f"chunk-{batch_index:06d}.parquet"
        chunk_paths.append(chunk_path)
        if chunk_path.is_file() and pq.ParquetFile(chunk_path).metadata.num_rows == batch_rows:
            source_row += batch_rows
            continue

        values = batch.to_pydict()
        columns = {
            name: _required_text(values[name], name, source_row) for name in NEWS_SOURCE_COLUMNS
        }
        documents = [
            f"{title}\n\n{description}" if description else title
            for title, description in zip(columns["title"], columns["description"], strict=True)
        ]
        embedding = embed(documents)
        if not embedding.type.equals(NEWS_SCHEMA.field("embedding").type):
            raise TypeError(
                f"embedding provider returned {embedding.type}; "
                f"expected {NEWS_SCHEMA.field('embedding').type}"
            )
        table = pa.table(
            {
                "datetime": assigned_timestamps[source_row : source_row + batch_rows],
                "ticker": assigned_tickers[source_row : source_row + batch_rows],
                "document_id": [
                    f"{NEWS_SOURCE_REVISION}:{index:010d}"
                    for index in range(source_row, source_row + batch_rows)
                ],
                "source_publish_date": columns["publish_date"],
                "source_symbol": columns["symbol"],
                "title": columns["title"],
                "description": columns["description"],
                "publisher": columns["publisher"],
                "url": columns["url"],
                "source": columns["source"],
                "embedding": embedding,
            },
            schema=NEWS_SCHEMA,
        )
        temporary_path = chunk_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(chunk_path)
        source_row += batch_rows
        print(f"Embedded {source_row:,}/{row_count:,} news rows")

    active_partition: date | None = None
    active_writer: pq.ParquetWriter | None = None
    active_temporary: Path | None = None
    try:
        for chunk_path in chunk_paths:
            table = pq.read_table(chunk_path)
            datetimes = table["datetime"].to_pylist()
            offset = 0
            while offset < table.num_rows:
                current_datetime = datetimes[offset]
                assert current_datetime is not None
                partition = current_datetime.date()
                end_offset = offset + 1
                while end_offset < table.num_rows:
                    candidate_datetime = datetimes[end_offset]
                    assert candidate_datetime is not None
                    if candidate_datetime.date() != partition:
                        break
                    end_offset += 1
                if partition != active_partition:
                    if active_writer is not None:
                        active_writer.close()
                        assert active_temporary is not None and active_partition is not None
                        active_temporary.replace(daily_partition_path(output, active_partition))
                    destination = daily_partition_path(output, partition)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    active_temporary = destination.with_suffix(".parquet.tmp")
                    active_temporary.unlink(missing_ok=True)
                    active_writer = pq.ParquetWriter(
                        active_temporary,
                        NEWS_SCHEMA,
                        compression="zstd",
                    )
                    active_partition = partition
                assert active_writer is not None
                active_writer.write_table(table.slice(offset, end_offset - offset))
                offset = end_offset
        if active_writer is not None:
            active_writer.close()
            active_writer = None
            assert active_temporary is not None and active_partition is not None
            active_temporary.replace(daily_partition_path(output, active_partition))
    except BaseException:
        if active_writer is not None:
            active_writer.close()
        if active_temporary is not None:
            active_temporary.unlink(missing_ok=True)
        raise

    empty_news = pa.Table.from_batches([], schema=NEWS_SCHEMA)
    for partition in calendar_days(start, end):
        destination = daily_partition_path(output, partition)
        if destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")
        pq.write_table(empty_news, temporary, compression="zstd")
        temporary.replace(destination)
    completion_path.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")
    shutil.rmtree(staging, ignore_errors=True)
    print(f"Wrote {row_count:,} embedded news rows to {output}")


def generate_symbology(tickers: list[str]) -> pa.Table:
    """Generate one deterministic symbology row for every ticker."""
    return pa.table(
        {
            "ticker": tickers,
            "isin": [f"SE{int(ticker):010d}" for ticker in tickers],
            "cik": [f"{int(ticker) + 1:010d}" for ticker in tickers],
            "company_name": [f"Example Company {ticker}" for ticker in tickers],
            "description": [f"Synthetic company record for ticker {ticker}." for ticker in tickers],
            "market_code": ["XSTO"] * len(tickers),
        },
        schema=SYMBOLOGY_SCHEMA,
    )


def generate_markets() -> pa.Table:
    """Generate market-hours rows without declaring or enforcing a primary key."""
    weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
    rows = [("XSTO", weekday, time(9, 0), time(17, 30), "Europe/Stockholm") for weekday in weekdays]
    rows.extend(
        ("XNYS", weekday, time(9, 30), time(16, 0), "America/New_York") for weekday in weekdays
    )
    return pa.Table.from_pylist(
        [
            {
                "market_code": market_code,
                "weekday": weekday,
                "opens_at": opens_at,
                "closes_at": closes_at,
                "timezone": timezone,
            }
            for market_code, weekday, opens_at, closes_at, timezone in rows
        ],
        schema=MARKETS_SCHEMA,
    )


def write_table_dataset(path: Path, table: pa.Table, overwrite: bool) -> None:
    """Atomically write a single-file table dataset."""
    if path.exists() and not overwrite:
        print(f"Skipping existing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".parquet.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Wrote {table.num_rows:,} rows to {path}")


def generate_bars(ticker: str, year: int, timestamps: list[datetime], seed: int) -> pa.Table:
    """Generate deterministic, internally consistent OHLCV bars for one partition."""
    generator = np.random.default_rng(np.random.SeedSequence([seed, int(ticker), year]))
    row_count = len(timestamps)

    starting_price = generator.uniform(20.0, 500.0)
    close = starting_price * np.exp(np.cumsum(generator.normal(0.0, 0.0015, row_count)))
    open_price = np.empty(row_count)
    open_price[0] = starting_price
    open_price[1:] = close[:-1] * np.exp(generator.normal(0.0, 0.0003, row_count - 1))

    upper_spread = generator.uniform(0.0, 0.002, row_count)
    lower_spread = generator.uniform(0.0, 0.002, row_count)
    high = np.maximum(open_price, close) * (1.0 + upper_spread)
    low = np.minimum(open_price, close) * (1.0 - lower_spread)
    volume = generator.integers(100, 1_000_001, row_count, dtype=np.int64)

    return pa.table(
        {
            "datetime": pa.array(timestamps, type=SCHEMA.field("datetime").type),
            "ticker": pa.array([ticker] * row_count, type=pa.string()),
            "open": pa.array(open_price),
            "high": pa.array(high),
            "low": pa.array(low),
            "close": pa.array(close),
            "volume": pa.array(volume),
        },
        schema=SCHEMA,
    )


def years_in_range(start: date, end: date) -> list[tuple[int, date, date]]:
    """Split an inclusive/exclusive date range into calendar-year ranges."""
    ranges: list[tuple[int, date, date]] = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, date(year, 1, 1))
        year_end = min(end, date(year + 1, 1, 1))
        if year_start < year_end:
            ranges.append((year, year_start, year_end))
    return ranges


def generate_dataset(
    output: Path,
    start: date,
    end: date,
    ticker_start: int,
    ticker_count: int,
    seed: int,
    overwrite: bool,
) -> None:
    """Write atomic UTC-day files while generating one bounded year at a time."""
    if end <= start:
        raise ValueError("end must be later than start")
    tickers = ticker_values(ticker_start, ticker_count)

    for year, year_start, year_end in years_in_range(start, end):
        days = calendar_days(year_start, year_end)
        market_days = trading_days(year_start, year_end)
        timestamps = session_minutes(market_days)
        offsets: dict[date, tuple[int, int]] = {}
        offset = 0
        for market_day in market_days:
            next_offset = offset + len(session_minutes([market_day]))
            offsets[market_day] = (offset, next_offset)
            offset = next_offset

        destinations = {day: daily_partition_path(output, day) for day in days}
        missing = {
            day: destination
            for day, destination in destinations.items()
            if overwrite or not destination.exists()
        }
        if not missing:
            print(f"Skipping {year}: all daily partitions already exist")
            continue

        writers: dict[date, pq.ParquetWriter] = {}
        temporary_paths: dict[date, Path] = {}
        try:
            for day, destination in missing.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".parquet.tmp")
                temporary.unlink(missing_ok=True)
                temporary_paths[day] = temporary
                writers[day] = pq.ParquetWriter(temporary, SCHEMA, compression="zstd")

            for ticker in tickers:
                bars = generate_bars(ticker, year, timestamps, seed)
                for market_day, (day_start, day_end) in offsets.items():
                    writer = writers.get(market_day)
                    if writer is not None:
                        writer.write_table(bars.slice(day_start, day_end - day_start))
        except BaseException:
            for writer in writers.values():
                writer.close()
            for temporary in temporary_paths.values():
                temporary.unlink(missing_ok=True)
            raise
        else:
            for writer in writers.values():
                writer.close()
            for day, temporary in temporary_paths.items():
                temporary.replace(destinations[day])

        print(
            f"Wrote {len(missing):,} daily partitions for {ticker_count:,} tickers "
            f"under {output / f'year={year}'}"
        )


def simple_moving_averages(close: np.ndarray, history: np.ndarray) -> dict[int, np.ndarray]:
    """Calculate SMAs, retaining only the preceding values needed by each window."""
    values = np.concatenate((history, close))
    history_length = len(history)
    averages: dict[int, np.ndarray] = {}
    for window in SMA_WINDOWS:
        result = np.full(len(close), np.nan)
        if len(values) >= window:
            cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
            all_averages = (cumulative[window:] - cumulative[:-window]) / window
            first_output = max(0, window - 1 - history_length)
            result[first_output:] = all_averages[-len(result) + first_output :]
        averages[window] = result
    return averages


def generate_sma_dataset(ohlcv_root: Path, output: Path, overwrite: bool) -> None:
    """Generate SMA10, SMA20, SMA50, and SMA200 datasets from daily OHLCV files."""
    close_history: dict[str, np.ndarray] = {}
    source_files = sorted(ohlcv_root.glob("year=*/month=*/day=*/part.parquet"))
    if not source_files:
        raise FileNotFoundError(f"No daily OHLCV files found under {ohlcv_root}")

    for source in source_files:
        relative_path = source.relative_to(ohlcv_root)
        destination = output / relative_path
        write_output = overwrite or not destination.exists()
        table = pq.read_table(source)
        tickers = table["ticker"].to_numpy(zero_copy_only=False)
        close = table["close"].to_numpy(zero_copy_only=False)
        output_tables: list[pa.Table] = []
        for ticker in np.unique(tickers):
            ticker_mask = tickers == ticker
            ticker_close = close[ticker_mask]
            history = close_history.get(ticker, np.array([], dtype=np.float64))
            if write_output:
                averages = simple_moving_averages(ticker_close, history)
                output_tables.append(
                    pa.table(
                        {
                            "datetime": table["datetime"].filter(pa.array(ticker_mask)),
                            "ticker": pa.array([ticker] * len(ticker_close), type=pa.string()),
                            **{
                                f"sma{window}": pa.array(averages[window]) for window in SMA_WINDOWS
                            },
                        },
                        schema=SMA_SCHEMA,
                    )
                )
            close_history[ticker] = np.concatenate((history, ticker_close))[-199:]

        if not write_output:
            print(f"Loaded history from existing {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")
        temporary.unlink(missing_ok=True)
        output_table = (
            pa.concat_tables(output_tables)
            if output_tables
            else pa.Table.from_batches([], schema=SMA_SCHEMA)
        )
        try:
            pq.write_table(output_table, temporary, compression="zstd", row_group_size=250_000)
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        print(f"Wrote {output_table.num_rows:,} SMA rows to {destination}")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/ohlcv"))
    parser.add_argument("--start", type=parse_date, default=date(2010, 1, 1))
    parser.add_argument("--end", type=parse_date, default=date(2025, 1, 1))
    parser.add_argument("--ticker-start", type=int, default=0)
    parser.add_argument(
        "--ticker-count",
        type=int,
        default=100,
        help="Number of sequential tickers to generate (default: 100, or 000 through 099).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--news-only",
        action="store_true",
        help="Generate only the embedded news family.",
    )
    parser.add_argument(
        "--news-source",
        type=Path,
        default=Path("source_data/dojo_stock_news.parquet"),
    )
    parser.add_argument("--news-output", type=Path, default=Path("data/news"))
    parser.add_argument("--embedding-batch-size", type=int, default=1_024)
    parser.add_argument(
        "--embedding-backend",
        choices=("fastembed", "transformers"),
        default="fastembed",
    )
    parser.add_argument(
        "--embedding-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--transformer-batch-size", type=int, default=64)
    parser.add_argument(
        "--generate-sma",
        action="store_true",
        help="Generate data/sma from the daily OHLCV Parquet files after generating OHLCV.",
    )
    parser.add_argument(
        "--sma-output",
        type=Path,
        default=Path("data/sma"),
        help="Directory for generated SMA feature files.",
    )
    parser.add_argument(
        "--symbology-output",
        type=Path,
        default=Path("data/symbols/data.parquet"),
        help="Path for the keyed symbology table.",
    )
    parser.add_argument(
        "--markets-output",
        type=Path,
        default=Path("data/markets/data.parquet"),
        help="Path for the non-keyed market-hours table.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    settings = arguments()
    tickers = ticker_values(settings.ticker_start, settings.ticker_count)
    if settings.news_only:
        generate_news_dataset(
            source_path=settings.news_source,
            output=settings.news_output,
            start=settings.start,
            end=settings.end,
            ticker_start=settings.ticker_start,
            ticker_count=settings.ticker_count,
            overwrite=settings.overwrite,
            batch_size=settings.embedding_batch_size,
            embedding_backend=settings.embedding_backend,
            embedding_device=settings.embedding_device,
            transformer_batch_size=settings.transformer_batch_size,
        )
    else:
        generate_dataset(
            output=settings.output,
            start=settings.start,
            end=settings.end,
            ticker_start=settings.ticker_start,
            ticker_count=settings.ticker_count,
            seed=settings.seed,
            overwrite=settings.overwrite,
        )
        if settings.generate_sma:
            generate_sma_dataset(settings.output, settings.sma_output, settings.overwrite)
        write_table_dataset(
            settings.symbology_output,
            generate_symbology(tickers),
            settings.overwrite,
        )
        write_table_dataset(
            settings.markets_output,
            generate_markets(),
            settings.overwrite,
        )
