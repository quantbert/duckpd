"""Reproducible exhaustive vector-search benchmark against direct DuckDB SQL."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import duckdb

import duckpd
from benchmark.metrics import get_peak_rss_bytes


@dataclass(frozen=True)
class VectorBenchmarkResult:
    rows: int
    dimension: int
    k: int
    duckpd_seconds: float
    direct_duckdb_seconds: float
    duckpd_peak_rss_bytes: int
    duckpd_spill_bytes: int
    direct_parity: bool
    strategy: str = "exact"


@dataclass(frozen=True)
class VSSBenchmarkResult:
    rows: int
    dimension: int
    k: int
    metric: str
    index_creation_seconds: float
    approximate_seconds: float
    recall_at_k: float
    repeated_result_equal: bool
    peak_rss_bytes: int
    physical_index_verified: bool
    index_memory_accounted_by_duckdb_limit: bool = False


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_case(rows: int, dimension: int, k: int = 10) -> VectorBenchmarkResult:
    """Run one deterministic exhaustive-search case and validate direct SQL parity."""
    if rows <= 0 or dimension <= 0 or k <= 0:
        raise ValueError("rows, dimension, and k must be positive")
    with tempfile.TemporaryDirectory(prefix="duckpd-vector-spill-") as temporary:
        spill = Path(temporary)
        with duckpd.connect(temp_directory=spill) as session:
            session._connection.execute(
                f"CREATE TABLE vectors AS SELECT i::BIGINT AS id, "
                f"CAST(list_transform(range({dimension}), "
                f"x -> (((i * 31 + x * 17) % 1009)::FLOAT / 1009)) AS FLOAT[{dimension}]) "
                f"AS embedding FROM range({rows}) source(i)"
            )
            query = tuple((index * 13 % 101) / 101 for index in range(dimension))
            lazy = session.table("vectors").vector.search(
                query,
                column="embedding",
                metric="l2",
                k=min(k, rows),
                tie_breaker="id",
            )
            started = perf_counter()
            duckpd_result = lazy.collect()
            duckpd_seconds = perf_counter() - started
            duckpd_peak = get_peak_rss_bytes()

            query_expression = duckdb.ConstantExpression(list(query)).cast(f"FLOAT[{dimension}]")
            distance = duckdb.FunctionExpression(
                "array_distance",
                duckdb.ColumnExpression("embedding"),
                query_expression,
            )
            direct = session._connection.table("vectors").project(
                duckdb.ColumnExpression("id"),
                distance.alias("_distance"),
            )
            started = perf_counter()
            direct_result = (
                direct.sort(
                    duckdb.ColumnExpression("_distance").asc(),
                    duckdb.ColumnExpression("id").asc(),
                )
                .limit(min(k, rows))
                .df()
            )
            direct_seconds = perf_counter() - started

        parity = duckpd_result["id"].tolist() == direct_result["id"].tolist()
        if not parity:
            raise AssertionError("DuckPD exact vector results differ from direct DuckDB")
        return VectorBenchmarkResult(
            rows,
            dimension,
            min(k, rows),
            duckpd_seconds,
            direct_seconds,
            duckpd_peak,
            _directory_bytes(spill),
            True,
        )


def run_vss_case(rows: int, dimension: int, k: int = 10) -> VSSBenchmarkResult:
    """Qualify one optional in-memory VSS case against exact DuckPD retrieval."""
    if rows <= 0 or dimension <= 0 or k <= 0:
        raise ValueError("rows, dimension, and k must be positive")
    with duckpd.connect() as session:
        session._connection.execute(
            f"CREATE TABLE vectors AS SELECT i::BIGINT AS id, "
            f"CAST(list_transform(range({dimension}), "
            f"x -> (((i * 31 + x * 17) % 1009)::FLOAT / 1009)) AS FLOAT[{dimension}]) "
            f"AS embedding FROM range({rows}) source(i)"
        )
        query = tuple((index * 13 % 101) / 101 for index in range(dimension))
        exact = (
            session.table("vectors")
            .vector.search(
                query,
                column="embedding",
                metric="cosine",
                k=min(k, rows),
                tie_breaker="id",
            )
            .collect()
        )
        started = perf_counter()
        info = session.create_vector_index(
            table="vectors",
            column="embedding",
            name="vectors_embedding_hnsw",
            metric="cosine",
        )
        index_seconds = perf_counter() - started
        approximate = session.table("vectors").vector.search(
            query,
            column="embedding",
            metric="cosine",
            k=min(k, rows),
            mode="approximate",
        )
        physical = approximate.explain("physical")
        started = perf_counter()
        first = approximate.collect()
        approximate_seconds = perf_counter() - started
        second = approximate.collect()
        exact_ids = set(exact["id"].tolist())
        approximate_ids = set(first["id"].tolist())
        recall = len(exact_ids & approximate_ids) / len(exact_ids)
        return VSSBenchmarkResult(
            rows,
            dimension,
            min(k, rows),
            info.metric,
            index_seconds,
            approximate_seconds,
            recall,
            first["id"].tolist() == second["id"].tolist(),
            get_peak_rss_bytes(),
            "HNSW_INDEX_SCAN" in physical and info.name in physical,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 64, 384])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--vss",
        action="store_true",
        help="also qualify optional in-memory DuckDB HNSW retrieval",
    )
    args = parser.parse_args(argv)
    results = [
        asdict(run_case(rows, dimension, args.k))
        for rows in args.rows
        for dimension in args.dimensions
    ]
    payload: dict[str, object] = {"track": "vector_exact", "results": results}
    if args.vss:
        payload["vss"] = [
            asdict(run_vss_case(rows, dimension, args.k))
            for rows in args.rows
            for dimension in args.dimensions
        ]
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
