"""Benchmark DuckPD optimizer planning and validated execution on Linux."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from statistics import median
from time import perf_counter

import pyarrow as pa

import duckpd
from duckpd._logical import LogicalPlan
from duckpd.session import Session


def _duration(call: Callable[[], object]) -> float:
    started = perf_counter()
    call()
    return perf_counter() - started


def _derived(offset: int) -> Callable[[duckpd.DataFrame], duckpd.Series]:
    def derive(current: duckpd.DataFrame) -> duckpd.Series:
        return current["value"] + offset

    return derive


def _materialize(session: Session, plan: LogicalPlan) -> pa.Table:
    compiled = session._compiler.compile(plan, optimize=False)
    return session._compiler.project_visible(compiled, plan).relation.to_arrow_table()


def _timed_materialize(session: Session, plan: LogicalPlan) -> tuple[float, pa.Table]:
    started = perf_counter()
    result = _materialize(session, plan)
    return perf_counter() - started, result


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_seconds": median(values),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def _assert_equal(left: pa.Table, right: pa.Table) -> None:
    if not left.equals(right):
        raise AssertionError("Optimizer changed the benchmark result")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=250_000)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument(
        "--max-regression-ratio",
        type=float,
        default=1.5,
        help="Fail when optimized median execution exceeds this baseline ratio",
    )
    args = parser.parse_args()
    if args.iterations < 3:
        raise ValueError("--iterations must be at least 3")
    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    if args.max_regression_ratio < 1:
        raise ValueError("--max-regression-ratio must be at least 1")

    with duckpd.connect() as session:
        frame = session.sql(
            "SELECT i AS id, i % 1000 AS group_id, i * 3 AS value "
            f"FROM range({args.rows}) AS source(i)"
        )
        for position in range(12):
            frame = frame.assign(**{f"derived_{position}": _derived(position)})
        workload = frame[["id", "group_id", "value"]].sort_values(["group_id", "value"]).limit(1000)

        optimization_times = [
            _duration(lambda: session._compiler.optimize(workload._plan))
            for _ in range(args.iterations)
        ]
        optimization = session._compiler.optimize(workload._plan)
        assert session._compiler.optimize(optimization.plan).plan == optimization.plan

        # Warm both variants before measurement and prove semantic equivalence.
        baseline_warm = _materialize(session, workload._plan)
        optimized_warm = _materialize(session, optimization.plan)
        _assert_equal(baseline_warm, optimized_warm)

        baseline_times: list[float] = []
        optimized_times: list[float] = []
        for iteration in range(args.iterations):
            plans = (
                (("baseline", workload._plan), ("optimized", optimization.plan))
                if iteration % 2 == 0
                else (("optimized", optimization.plan), ("baseline", workload._plan))
            )
            results: dict[str, pa.Table] = {}
            for label, plan in plans:
                duration, result = _timed_materialize(session, plan)
                results[label] = result
                if label == "baseline":
                    baseline_times.append(duration)
                else:
                    optimized_times.append(duration)
            _assert_equal(results["baseline"], results["optimized"])

        changed_passes = [snapshot.name for snapshot in optimization.snapshots if snapshot.changed]
        ablations: dict[str, dict[str, object]] = {}
        for pass_name in changed_passes:
            ablated = session._compiler.optimize(
                workload._plan,
                disabled_passes=frozenset({pass_name}),
            )
            _assert_equal(optimized_warm, _materialize(session, ablated.plan))
            ablated_times: list[float] = []
            for _ in range(args.iterations):
                duration, result = _timed_materialize(session, ablated.plan)
                _assert_equal(optimized_warm, result)
                ablated_times.append(duration)
            ablations[pass_name] = {
                **_summary(ablated_times),
                "optimized_to_ablated_median_ratio": (
                    median(optimized_times) / median(ablated_times)
                ),
            }

        execution_ratio = median(optimized_times) / median(baseline_times)
        print(
            json.dumps(
                {
                    "system": platform.system(),
                    "machine": platform.machine(),
                    "rows": args.rows,
                    "iterations": args.iterations,
                    "optimizer": _summary(optimization_times),
                    "unoptimized_execution": _summary(baseline_times),
                    "optimized_execution": _summary(optimized_times),
                    "optimized_to_unoptimized_median_ratio": execution_ratio,
                    "changed_passes": changed_passes,
                    "ablations": ablations,
                    "results_equal": True,
                    "measurement_order": "alternating",
                    "warmup_runs_per_variant": 1,
                },
                indent=2,
            )
        )
        if execution_ratio > args.max_regression_ratio:
            raise SystemExit(
                "optimizer regression gate failed: "
                f"{execution_ratio:.3f} > {args.max_regression_ratio:.3f}"
            )


if __name__ == "__main__":
    main()
