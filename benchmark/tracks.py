"""Validated, reproducible benchmark tracks for release evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from statistics import median
from time import perf_counter

import duckdb
import pandas as pd
from pandas.testing import assert_frame_equal

import duckpd
from benchmark.generate import market_query
from benchmark.metrics import get_peak_rss_bytes

_COLD_CACHE_POLICY = "fresh_spawn_process+posix_fadvise_dontneed"


@dataclass(frozen=True)
class EngineResult:
    engine: str
    status: str
    planning_seconds: float | None
    execution_seconds: float | None
    peak_rss_bytes: int | None
    spill_bytes: int | None
    correct: bool | None
    detail: str | None = None
    source_bytes_read: int | None = None
    transfer_bytes: int | None = None
    cold_execution_seconds: float | None = None
    warm_execution_seconds: float | None = None
    cold_cache_policy: str | None = None


@dataclass(frozen=True)
class TrackResult:
    track: str
    rows: int
    engines: tuple[EngineResult, ...]


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _normalized(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.sort_values(keys).reset_index(drop=True)


def _evict_source_cache(paths: tuple[Path, ...]) -> None:
    """Evict source pages before a cold run using an explicit Linux policy."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("cold runs require POSIX_FADV_DONTNEED")
    for path in paths:
        with path.open("rb") as source:
            os.fsync(source.fileno())
            os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def _measure(
    engine: str,
    run: Callable[[], pd.DataFrame],
    expected: pd.DataFrame,
    keys: list[str],
    spill_directory: Path | None = None,
) -> EngineResult:
    try:
        cold_started = perf_counter()
        cold_result = run()
        cold_execution = perf_counter() - cold_started
        warm_started = perf_counter()
        warm_result = run()
        warm_execution = perf_counter() - warm_started
        for result in (cold_result, warm_result):
            assert_frame_equal(
                _normalized(result, keys),
                _normalized(expected, keys),
                check_dtype=False,
                rtol=1e-6,
                atol=1e-6,
            )
    except MemoryError as exc:
        return EngineResult(engine, "oom", None, None, None, None, False, str(exc))
    except Exception as exc:
        return EngineResult(
            engine,
            "failed",
            None,
            None,
            get_peak_rss_bytes(),
            None,
            False,
            f"{type(exc).__name__}: {exc}",
        )
    spill = _directory_bytes(spill_directory) if spill_directory is not None else 0
    return EngineResult(
        engine,
        "success",
        None,
        warm_execution,
        get_peak_rss_bytes(),
        spill,
        True,
        cold_execution_seconds=cold_execution,
        warm_execution_seconds=warm_execution,
        cold_cache_policy=_COLD_CACHE_POLICY,
    )


def _measure_duckpd(
    build: Callable[[duckpd.Session], duckpd.DataFrame],
    expected: pd.DataFrame,
    keys: list[str],
    spill_directory: Path,
) -> EngineResult:
    try:
        with duckpd.connect(temp_directory=spill_directory) as session:
            cold_started = perf_counter()
            cold_result = build(session).collect()
            cold_execution = perf_counter() - cold_started
            warm_started = perf_counter()
            warm_result = build(session).collect()
            warm_execution = perf_counter() - warm_started
            peak_rss_bytes = get_peak_rss_bytes()
            profile_frame = build(session)
            planning_started = perf_counter()
            profile_frame.explain("sql")
            planning = perf_counter() - planning_started
            profile = profile_frame.profile()
            source_bytes_read = profile.bytes_read
            transfer_bytes = profile.measured_transfer_bytes
        for result in (cold_result, warm_result):
            assert_frame_equal(
                _normalized(result, keys),
                _normalized(expected, keys),
                check_dtype=False,
                rtol=1e-6,
                atol=1e-6,
            )
    except MemoryError as exc:
        return EngineResult("duckpd", "oom", None, None, None, None, False, str(exc))
    except Exception as exc:
        return EngineResult(
            "duckpd",
            "failed",
            None,
            None,
            get_peak_rss_bytes(),
            _directory_bytes(spill_directory),
            False,
            f"{type(exc).__name__}: {exc}",
        )
    return EngineResult(
        "duckpd",
        "success",
        planning,
        warm_execution,
        peak_rss_bytes,
        _directory_bytes(spill_directory),
        True,
        source_bytes_read=source_bytes_read,
        transfer_bytes=transfer_bytes,
        cold_execution_seconds=cold_execution,
        warm_execution_seconds=warm_execution,
        cold_cache_policy=_COLD_CACHE_POLICY,
    )


def _unavailable(engine: str, module: str) -> EngineResult:
    available = importlib.util.find_spec(module) is not None
    detail = (
        "optional engine is installed but this validated track has no adapter"
        if available
        else "optional engine is not installed"
    )
    return EngineResult(
        engine,
        "unsupported" if available else "unavailable",
        None,
        None,
        None,
        None,
        None,
        detail,
    )


def _prepare(root: Path, rows: int) -> dict[str, Path]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    paths = {
        "lineitem": root / "lineitem.parquet",
        "orders": root / "orders.parquet",
        "customers": root / "customers.parquet",
        "ohlc": root / "ohlc.parquet",
    }
    con = duckdb.connect()
    con.sql(
        "SELECT CASE i % 3 WHEN 0 THEN 'A' WHEN 1 THEN 'N' ELSE 'R' END AS returnflag, "
        "CASE i % 2 WHEN 0 THEN 'F' ELSE 'O' END AS linestatus, "
        "(i % 50 + 1)::DOUBLE AS quantity, "
        "(i % 10000 + 100)::DOUBLE AS extendedprice, "
        "(i % 10)::DOUBLE / 100 AS discount, (i % 8)::DOUBLE / 100 AS tax "
        "FROM range(?) source(i)",
        params=[rows],
    ).write_parquet(str(paths["lineitem"]))
    customer_rows = max(10, rows // 10)
    con.sql(
        "SELECT i::BIGINT AS customer_id, "
        "CASE i % 4 WHEN 0 THEN 'east' WHEN 1 THEN 'west' "
        "WHEN 2 THEN 'north' ELSE 'south' END AS region "
        "FROM range(?) source(i)",
        params=[customer_rows],
    ).write_parquet(str(paths["customers"]))
    con.sql(
        "SELECT i::BIGINT AS order_id, (i % ?)::BIGINT AS customer_id, "
        "(i % 1000 + 1)::DOUBLE AS amount FROM range(?) source(i)",
        params=[customer_rows, rows],
    ).write_parquet(str(paths["orders"]))
    con.sql(market_query(rows)).write_parquet(str(paths["ohlc"]))
    con.close()
    return paths


def _tpch_expected(path: Path) -> pd.DataFrame:
    return duckdb.sql(
        "SELECT returnflag, linestatus, sum(quantity) AS sum_qty, "
        "sum(extendedprice) AS sum_base_price, "
        "sum(extendedprice * (1 - discount)) AS sum_disc_price, "
        "avg(quantity) AS avg_qty, count(*) AS count_order "
        "FROM read_parquet(?) GROUP BY returnflag, linestatus",
        params=[str(path)],
    ).df()


def _tpch_duckpd(session: duckpd.Session, path: Path) -> duckpd.DataFrame:
    frame = session.read_parquet(path)
    return (
        frame.assign(
            disc_price=lambda current: (
                current["extendedprice"] * (1 - current["discount"])
            )
        )
        .groupby(["returnflag", "linestatus"], as_index=False)
        .agg(
            sum_qty=("quantity", "sum"),
            sum_base_price=("extendedprice", "sum"),
            sum_disc_price=("disc_price", "sum"),
            avg_qty=("quantity", "mean"),
            count_order=("quantity", "size"),
        )
    )


def _join_expected(orders: Path, customers: Path) -> pd.DataFrame:
    return duckdb.sql(
        "SELECT c.region, sum(o.amount) AS revenue, count(*) AS order_count "
        "FROM read_parquet(?) o JOIN read_parquet(?) c USING (customer_id) "
        "GROUP BY c.region",
        params=[str(orders), str(customers)],
    ).df()


def _join_duckpd(
    session: duckpd.Session, orders: Path, customers: Path
) -> duckpd.DataFrame:
    return (
        session.read_parquet(orders)
        .merge(session.read_parquet(customers), on="customer_id")
        .groupby("region", as_index=False)
        .agg(revenue=("amount", "sum"), order_count=("order_id", "size"))
    )


def _ohlc_expected(path: Path) -> pd.DataFrame:
    return duckdb.sql(
        "SELECT ticker, avg(open) AS mean_open, max(high) AS max_high, "
        "min(low) AS min_low, avg(close) AS mean_close, count(*) AS bar_count "
        "FROM read_parquet(?) GROUP BY ticker",
        params=[str(path)],
    ).df()


def _ohlc_duckpd(session: duckpd.Session, path: Path) -> duckpd.DataFrame:
    return (
        session.read_parquet(path)
        .groupby("ticker", as_index=False)
        .agg(
            mean_open=("open", "mean"),
            max_high=("high", "max"),
            min_low=("low", "min"),
            mean_close=("close", "mean"),
            bar_count=("close", "size"),
        )
    )


_TRACK_KEYS = {
    "tpch_q1": ["returnflag", "linestatus"],
    "db_groupby_join": ["region"],
    "synthetic_ohlc": ["ticker"],
}


def _track_sources(name: str, paths: dict[str, Path]) -> tuple[Path, ...]:
    if name == "tpch_q1":
        return (paths["lineitem"],)
    if name == "db_groupby_join":
        return (paths["orders"], paths["customers"])
    return (paths["ohlc"],)


def _duckdb_track(name: str, paths: dict[str, Path]) -> pd.DataFrame:
    if name == "tpch_q1":
        return _tpch_expected(paths["lineitem"])
    if name == "db_groupby_join":
        return _join_expected(paths["orders"], paths["customers"])
    return _ohlc_expected(paths["ohlc"])


def _duckpd_track(
    name: str, session: duckpd.Session, paths: dict[str, Path]
) -> duckpd.DataFrame:
    if name == "tpch_q1":
        return _tpch_duckpd(session, paths["lineitem"])
    if name == "db_groupby_join":
        return _join_duckpd(session, paths["orders"], paths["customers"])
    return _ohlc_duckpd(session, paths["ohlc"])


def _pandas_track(name: str, paths: dict[str, Path]) -> pd.DataFrame:
    if name == "tpch_q1":
        return (
            pd.read_parquet(paths["lineitem"])
            .assign(
                disc_price=lambda frame: (
                    frame["extendedprice"] * (1 - frame["discount"])
                )
            )
            .groupby(["returnflag", "linestatus"], as_index=False)
            .agg(
                sum_qty=("quantity", "sum"),
                sum_base_price=("extendedprice", "sum"),
                sum_disc_price=("disc_price", "sum"),
                avg_qty=("quantity", "mean"),
                count_order=("quantity", "size"),
            )
        )
    if name == "db_groupby_join":
        return (
            pd.read_parquet(paths["orders"])
            .merge(pd.read_parquet(paths["customers"]), on="customer_id")
            .groupby("region", as_index=False)
            .agg(revenue=("amount", "sum"), order_count=("order_id", "size"))
        )
    return (
        pd.read_parquet(paths["ohlc"])
        .groupby("ticker", as_index=False)
        .agg(
            mean_open=("open", "mean"),
            max_high=("high", "max"),
            min_low=("low", "min"),
            mean_close=("close", "mean"),
            bar_count=("close", "size"),
        )
    )


def _isolated_worker(
    output: Connection,
    name: str,
    engine: str,
    paths: dict[str, Path],
    expected: pd.DataFrame,
    spill_directory: Path,
) -> None:
    try:
        _evict_source_cache(_track_sources(name, paths))
        keys = _TRACK_KEYS[name]
        if engine == "duckdb_sql":
            result = _measure(
                engine, lambda: _duckdb_track(name, paths), expected, keys
            )
        elif engine == "duckpd":
            result = _measure_duckpd(
                lambda session: _duckpd_track(name, session, paths),
                expected,
                keys,
                spill_directory,
            )
        else:
            result = _measure(
                engine, lambda: _pandas_track(name, paths), expected, keys
            )
    except BaseException as exc:
        result = EngineResult(
            engine,
            "failed",
            None,
            None,
            get_peak_rss_bytes(),
            None,
            False,
            f"{type(exc).__name__}: {exc}",
        )
    output.send(result)
    output.close()


def _measure_isolated(
    name: str,
    engine: str,
    paths: dict[str, Path],
    expected: pd.DataFrame,
    spill_directory: Path,
) -> EngineResult:
    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_worker,
        args=(sender, name, engine, paths, expected, spill_directory),
    )
    process.start()
    sender.close()
    process.join()
    if receiver.poll():
        result = receiver.recv()
        receiver.close()
        return result
    receiver.close()
    return EngineResult(
        engine,
        "failed",
        None,
        None,
        None,
        None,
        False,
        f"isolated benchmark process exited with code {process.exitcode}",
    )


def run_tracks(rows: int, root: Path) -> tuple[TrackResult, ...]:
    """Run validated TPC-H, db-benchmark-style, and OHLC tracks."""
    root.mkdir(parents=True, exist_ok=True)
    paths = _prepare(root, rows)
    results: list[TrackResult] = []
    for name in _TRACK_KEYS:
        expected = _duckdb_track(name, paths)
        spill = root / f"spill-{name}"
        spill.mkdir(exist_ok=True)
        engines = (
            _measure_isolated(name, "duckdb_sql", paths, expected, spill),
            _measure_isolated(name, "duckpd", paths, expected, spill),
            _measure_isolated(name, "pandas", paths, expected, spill),
            _unavailable("polars", "polars"),
            _unavailable("fireducks", "fireducks"),
        )
        results.append(TrackResult(name, rows, engines))
    return tuple(results)


def scorecard(results: tuple[TrackResult, ...]) -> dict[str, object]:
    """Aggregate explicit correctness, performance, memory, and support evidence."""
    engines = sorted({engine.engine for result in results for engine in result.engines})
    entries: list[dict[str, object]] = []
    for engine_name in engines:
        samples = [
            engine
            for result in results
            for engine in result.engines
            if engine.engine == engine_name
        ]
        successful = [sample for sample in samples if sample.status == "success"]
        cold = [
            value
            for sample in successful
            if (value := sample.cold_execution_seconds) is not None
        ]
        warm = [
            value
            for sample in successful
            if (value := sample.warm_execution_seconds) is not None
        ]
        rss = [
            value
            for sample in successful
            if (value := sample.peak_rss_bytes) is not None
        ]
        entries.append(
            {
                "engine": engine_name,
                "tracks": len(samples),
                "successful_tracks": len(successful),
                "correct_tracks": sum(sample.correct is True for sample in samples),
                "failed_tracks": sum(sample.status == "failed" for sample in samples),
                "oom_tracks": sum(sample.status == "oom" for sample in samples),
                "unsupported_tracks": sum(
                    sample.status == "unsupported" for sample in samples
                ),
                "unavailable_tracks": sum(
                    sample.status == "unavailable" for sample in samples
                ),
                "median_cold_seconds": median(cold) if cold else None,
                "median_warm_seconds": median(warm) if warm else None,
                "max_peak_rss_bytes": max(rss) if rss else None,
                "source_byte_measurements": sum(
                    sample.source_bytes_read is not None for sample in samples
                ),
                "transfer_byte_measurements": sum(
                    sample.transfer_bytes is not None for sample in samples
                ),
            }
        )
    return {
        "dimensions": {
            "correctness": "validated results retained per engine and track",
            "safety": "failed, OOM, unsupported, and unknown metrics are retained",
            "scale": {
                "rows_per_track": {result.track: result.rows for result in results}
            },
            "observability": [
                "planning_seconds",
                "peak_rss_bytes",
                "spill_bytes",
                "source_bytes_read",
                "transfer_bytes",
            ],
            "performance": [
                "cold_execution_seconds",
                "warm_execution_seconds",
                "cold_cache_policy",
            ],
            "portability": {
                "platform": platform.system(),
                "python": platform.python_version(),
            },
            "interoperability": {"engines": engines},
            "openness": {
                "format": "public JSON",
                "license": "MIT",
                "generator": "benchmark.tracks",
            },
        },
        "engines": entries,
    }


def results_json(results: tuple[TrackResult, ...]) -> str:
    return json.dumps([asdict(result) for result in results], indent=2)


def scorecard_json(results: tuple[TrackResult, ...]) -> str:
    return json.dumps(scorecard(results), indent=2)


def smoke(rows: int = 1_000) -> tuple[TrackResult, ...]:
    with tempfile.TemporaryDirectory(prefix="duckpd-tracks-") as directory:
        return run_tracks(rows, Path(directory))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run validated TPC-H, db-benchmark-style, and OHLC tracks."
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scorecard-output", type=Path)
    args = parser.parse_args()
    if args.data_dir is None:
        with tempfile.TemporaryDirectory(prefix="duckpd-tracks-") as directory:
            results = run_tracks(args.rows, Path(directory))
    else:
        results = run_tracks(args.rows, args.data_dir)
    output = results_json(results)
    if args.output is None:
        print(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    if args.scorecard_output is not None:
        args.scorecard_output.parent.mkdir(parents=True, exist_ok=True)
        args.scorecard_output.write_text(
            scorecard_json(results) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
