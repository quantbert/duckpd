"""Behavioral tests for logical rewrites and optimizer observability."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import pandas as pd
from pandas.testing import assert_frame_equal

import duckpd
from duckpd._logical import (
    FilterPlan,
    LimitPlan,
    ParquetSource,
    ProjectPlan,
    ScanPlan,
    SourceKind,
    SourceProvenance,
    TopKPlan,
    UnionPlan,
)


def test_optimizer_is_named_idempotent_and_preserves_root_metadata() -> None:
    frame = duckpd.from_pandas(pd.DataFrame({"a": [3, 1, 2], "b": [30, 10, 20]}))
    plan = frame.assign(extra=lambda value: value["a"] + value["b"])[["a"]]._plan

    first = frame._session._compiler.optimize(plan)
    second = frame._session._compiler.optimize(first.plan)

    assert tuple(snapshot.name for snapshot in first.snapshots) == (
        "predicate_pushdown",
        "required_column_liveness",
        "limit_topk",
        "redundant_nodes",
    )
    assert first.plan == second.plan
    assert first.plan.metadata == plan.metadata
    assert all(
        snapshot.before.metadata == snapshot.after.metadata
        for snapshot in first.snapshots
    )
    assert any(
        snapshot.name == "required_column_liveness" and snapshot.changed
        for snapshot in first.snapshots
    )


def test_predicate_pushdown_preserves_results_and_metadata() -> None:
    source = pd.DataFrame({"a": [3, 1, 2], "b": [30, 10, 20]})
    frame = duckpd.from_pandas(source)
    projected = frame[["a", "b"]]
    filtered = projected[projected["a"] > 1]

    optimized = frame._session._compiler.optimize(filtered._plan)

    assert optimized.plan.metadata == filtered._plan.metadata
    assert isinstance(optimized.plan, (FilterPlan, ProjectPlan))
    assert_frame_equal(
        filtered.collect().reset_index(drop=True),
        source.loc[source["a"] > 1].reset_index(drop=True),
    )


def test_limit_rewrites_combine_ranges_and_form_topk() -> None:
    source = pd.DataFrame({"value": [5, 1, 4, 2, 3]})
    frame = duckpd.from_pandas(source)

    nested = frame.limit(4).limit(2, offset=1)
    nested_result = frame._session._compiler.optimize(nested._plan)
    assert isinstance(nested_result.plan, LimitPlan)
    assert nested_result.plan.count == 2
    assert nested_result.plan.offset == 1

    topk = frame.sort_values("value").limit(3)
    topk_result = frame._session._compiler.optimize(topk._plan)
    assert isinstance(topk_result.plan, TopKPlan)
    assert_frame_equal(
        topk.collect().reset_index(drop=True),
        source.sort_values("value").head(3).reset_index(drop=True),
    )


def test_optimizer_reports_common_subplans_without_implicit_cache() -> None:
    frame = duckpd.from_pandas(pd.DataFrame({"value": [1, 2, 3]})).assign(
        doubled=lambda value: value["value"] * 2
    )
    repeated = UnionPlan((frame._plan, frame._plan), frame._plan.metadata)

    result = frame._session._compiler.optimize(repeated)

    assert result.recommendations
    assert all(item["kind"] == "persist" for item in result.recommendations)
    assert all(item["occurrences"] == 2 for item in result.recommendations)


def test_explain_json_exports_rewrites_and_optimized_plan() -> None:
    frame = (
        duckpd.from_pandas(pd.DataFrame({"value": [2, 1]}))
        .sort_values("value")
        .limit(1)
    )
    execution_count = frame._session.execution_count

    data = json.loads(frame.explain("json"))

    assert data["plan"]["node"] == "TopKPlan"
    assert data["execution_boundaries"] == {
        "fallback_policy": "error",
        "fallback": [],
        "materialization": [],
    }
    assert [snapshot["name"] for snapshot in data["snapshots"]] == [
        "predicate_pushdown",
        "required_column_liveness",
        "limit_topk",
        "redundant_nodes",
    ]
    assert "DuckPD optimized logical plan:" in frame.explain("optimized")
    assert frame._session.execution_count == execution_count


def test_machine_readable_plans_redact_source_credentials() -> None:
    frame = duckpd.from_pandas(pd.DataFrame({"value": [1]}))
    uri = "https://user:password@example.com/data.parquet?token=secret#fragment"
    metadata = replace(
        frame._plan.metadata,
        provenance=SourceProvenance(
            SourceKind.PARQUET,
            (uri,),
            fingerprint="fingerprint",
        ),
    )
    plan = ScanPlan(ParquetSource((uri,)), metadata)

    exported = frame._session._executor.explain(plan, mode="json")

    assert "user" not in exported
    assert "password" not in exported
    assert "token" not in exported
    assert "secret" not in exported
    assert "https://example.com/data.parquet" in exported


def test_profile_separates_planning_and_execution_timings() -> None:
    frame = duckpd.from_pandas(pd.DataFrame({"value": range(100)}))

    profile = frame[frame["value"] > 50].profile()
    exported = profile.to_dict()
    duckpd_metrics = cast("dict[str, object]", exported["duckpd"])

    assert profile.planning_seconds >= 0
    assert profile.execution_seconds >= 0
    assert profile.optimization is not None
    assert duckpd_metrics["planning_seconds"] == profile.planning_seconds
    assert duckpd_metrics["execution_seconds"] == profile.execution_seconds
