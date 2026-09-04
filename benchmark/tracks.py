"""Validated, reproducible benchmark tracks for release evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import duckdb
import pandas as pd
from pandas.testing import assert_frame_equal

import duckpd
from benchmark.generate import market_query
from benchmark.metrics import get_peak_rss_bytes


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


@dataclass(frozen=True)
class TrackResult:
    track: str
    rows: int
    engines: tuple[EngineResult, ...]


TrackSpec = tuple[
    str,
    pd.DataFrame,
    Callable[[], pd.DataFrame],
    list[str],
    Callable[[duckpd.Session], duckpd.DataFrame],
    Callable[[], pd.DataFrame],
]


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _normalized(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.sort_values(keys).reset_index(drop=True)


def _measure(
    engine: str,
    run: Callable[[], pd.DataFrame],
    expected: pd.DataFrame,
    keys: list[str],
    spill_directory: Path | None = None,
) -> EngineResult:
    started = perf_counter()
    try:
        result = run()
        execution = perf_counter() - started
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
        execution,
        get_peak_rss_bytes(),
        spill,
        True,
    )


def _measure_duckpd(
    build: Callable[[duckpd.Session], duckpd.DataFrame],
    expected: pd.DataFrame,
    keys: list[str],
    spill_directory: Path,
) -> EngineResult:
    try:
        with duckpd.connect(temp_directory=spill_directory) as session:
            frame = build(session)
            planning_started = perf_counter()
            frame.explain("sql")
            planning = perf_counter() - planning_started
            execution_started = perf_counter()
            result = frame.collect()
            execution = perf_counter() - execution_started
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
        execution,
        get_peak_rss_bytes(),
        _directory_bytes(spill_directory),
        True,
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


def run_tracks(rows: int, root: Path) -> tuple[TrackResult, ...]:
    """Run validated TPC-H, db-benchmark-style, and OHLC tracks."""
    root.mkdir(parents=True, exist_ok=True)
    paths = _prepare(root, rows)
    specifications: tuple[TrackSpec, ...] = (
        (
            "tpch_q1",
            _tpch_expected(paths["lineitem"]),
            lambda: _tpch_expected(paths["lineitem"]),
            ["returnflag", "linestatus"],
            lambda session: _tpch_duckpd(session, paths["lineitem"]),
            lambda: (
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
            ),
        ),
        (
            "db_groupby_join",
            _join_expected(paths["orders"], paths["customers"]),
            lambda: _join_expected(paths["orders"], paths["customers"]),
            ["region"],
            lambda session: _join_duckpd(session, paths["orders"], paths["customers"]),
            lambda: (
                pd.read_parquet(paths["orders"])
                .merge(pd.read_parquet(paths["customers"]), on="customer_id")
                .groupby("region", as_index=False)
                .agg(revenue=("amount", "sum"), order_count=("order_id", "size"))
            ),
        ),
        (
            "synthetic_ohlc",
            _ohlc_expected(paths["ohlc"]),
            lambda: _ohlc_expected(paths["ohlc"]),
            ["ticker"],
            lambda session: _ohlc_duckpd(session, paths["ohlc"]),
            lambda: (
                pd.read_parquet(paths["ohlc"])
                .groupby("ticker", as_index=False)
                .agg(
                    mean_open=("open", "mean"),
                    max_high=("high", "max"),
                    min_low=("low", "min"),
                    mean_close=("close", "mean"),
                    bar_count=("close", "size"),
                )
            ),
        ),
    )
    results: list[TrackResult] = []
    for name, expected, duckdb_run, keys, duckpd_build, pandas_run in specifications:
        spill = root / f"spill-{name}"
        spill.mkdir(exist_ok=True)
        engines = (
            _measure("duckdb_sql", duckdb_run, expected, keys),
            _measure_duckpd(duckpd_build, expected, keys, spill),
            _measure("pandas", pandas_run, expected, keys),
            _unavailable("polars", "polars"),
            _unavailable("fireducks", "fireducks"),
        )
        results.append(TrackResult(name, rows, engines))
    return tuple(results)


def results_json(results: tuple[TrackResult, ...]) -> str:
    return json.dumps([asdict(result) for result in results], indent=2)


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
    args = parser.parse_args()
    if args.data_dir is None:
        with tempfile.TemporaryDirectory(prefix="duckpd-tracks-") as directory:
            output = results_json(run_tracks(args.rows, Path(directory)))
    else:
        output = results_json(run_tracks(args.rows, args.data_dir))
    if args.output is None:
        print(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
