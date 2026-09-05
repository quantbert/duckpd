"""Idempotent logical rewrites and machine-readable plan observability."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from time import perf_counter
from typing import cast
from uuid import UUID

from duckpd._logical import (
    BinaryExpression,
    CaseWhen,
    CastExpression,
    ColumnId,
    ColumnRef,
    CsvSource,
    Expression,
    FilterPlan,
    FunctionCall,
    JoinPlan,
    LimitPlan,
    LiteralValue,
    LogicalPlan,
    ParquetSource,
    ProjectPlan,
    RemoteTableSource,
    ScanPlan,
    SortPlan,
    SourceProvenance,
    SqlSource,
    TopKPlan,
    UnaryExpression,
    UnionPlan,
    sanitize_source_location,
)
from duckpd._metadata import after_filter, after_projection


@dataclass(frozen=True)
class RewriteSnapshot:
    """Result and cost of one named optimizer pass."""

    name: str
    changed: bool
    duration_seconds: float
    before: LogicalPlan
    after: LogicalPlan

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "changed": self.changed,
            "duration_seconds": self.duration_seconds,
            "before": plan_to_dict(self.before),
            "after": plan_to_dict(self.after),
        }


@dataclass(frozen=True)
class OptimizationResult:
    """Optimized plan, pass snapshots, and explicit persist recommendations."""

    plan: LogicalPlan
    snapshots: tuple[RewriteSnapshot, ...]
    recommendations: tuple[dict[str, object], ...]
    duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "duration_seconds": self.duration_seconds,
            "plan": plan_to_dict(self.plan),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "recommendations": list(self.recommendations),
        }


Rewrite = Callable[[LogicalPlan], LogicalPlan]


class LogicalOptimizer:
    """Apply a fixed, named, idempotent logical rewrite pipeline."""

    def __init__(self) -> None:
        self._passes: tuple[tuple[str, Rewrite], ...] = (
            ("predicate_pushdown", _predicate_pushdown),
            ("required_column_liveness", _required_column_liveness),
            ("limit_topk", _limit_and_topk),
            ("redundant_nodes", _remove_redundant_nodes),
        )

    @property
    def pass_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._passes)

    def optimize(
        self,
        plan: LogicalPlan,
        *,
        disabled_passes: frozenset[str] = frozenset(),
    ) -> OptimizationResult:
        unknown = disabled_passes - set(self.pass_names)
        if unknown:
            raise ValueError(f"Unknown optimizer passes: {sorted(unknown)}")
        started = perf_counter()
        current = plan
        snapshots: list[RewriteSnapshot] = []
        for name, rewrite in self._passes:
            if name in disabled_passes:
                continue
            before = current
            pass_started = perf_counter()
            current = _fixed_point(rewrite, current)
            snapshots.append(
                RewriteSnapshot(
                    name,
                    current != before,
                    perf_counter() - pass_started,
                    before,
                    current,
                )
            )
        recommendations = _common_subplan_recommendations(current)
        return OptimizationResult(
            current,
            tuple(snapshots),
            recommendations,
            perf_counter() - started,
        )


def plan_to_dict(plan: LogicalPlan) -> dict[str, object]:
    """Return a JSON-compatible representation of a logical plan."""
    value = _json_value(plan)
    if not isinstance(value, dict):
        raise AssertionError("Logical plan serialization did not produce an object")
    return cast("dict[str, object]", value)


def _fixed_point(rewrite: Rewrite, plan: LogicalPlan) -> LogicalPlan:
    current = plan
    for _ in range(64):
        rewritten = rewrite(current)
        if rewritten == current:
            return current
        current = rewritten
    raise AssertionError("Logical optimizer pass did not converge")


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (ParquetSource, CsvSource)):
        result: dict[str, object] = {
            "node": type(value).__name__,
            "paths": [sanitize_source_location(path) for path in value.paths],
        }
        for field in fields(value):
            if field.name != "paths":
                result[field.name] = _json_value(getattr(value, field.name))
        return result
    if isinstance(value, RemoteTableSource):
        return {
            "node": "RemoteTableSource",
            "engine": value.engine,
            "attachment": value.attachment,
            "schema": value.schema,
            "table": value.table,
            "location": sanitize_source_location(value.location),
            "capabilities": _json_value(value.capabilities),
            "unbounded_scan": value.unbounded_scan,
        }
    if isinstance(value, SqlSource):
        return {
            "node": "SqlSource",
            "query": "<redacted>",
            "fingerprint": hashlib.sha256(value.query.encode()).hexdigest(),
        }
    if isinstance(value, SourceProvenance):
        result = {
            "node": "SourceProvenance",
            "kind": value.kind.value,
            "locations": [
                sanitize_source_location(location) for location in value.locations
            ],
            "fingerprint": value.fingerprint,
            "writable": value.writable,
            "row_preserving": value.row_preserving,
            "transformations": list(value.transformations),
        }
        if value.capabilities is not None:
            result["capabilities"] = _json_value(value.capabilities)
        return result
    if isinstance(value, tuple):
        return [_json_value(item) for item in cast("tuple[object, ...]", value)]
    if isinstance(value, list):
        return [_json_value(item) for item in cast("list[object]", value)]
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {str(key): _json_value(item) for key, item in mapping.items()}
    if is_dataclass(value):
        result: dict[str, object] = {"node": type(value).__name__}
        result.update(
            {
                field.name: _json_value(getattr(value, field.name))
                for field in fields(value)
            }
        )
        return result
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _rewrite_tree(plan: LogicalPlan, local: Rewrite) -> LogicalPlan:
    if isinstance(plan, ScanPlan):
        return local(plan)
    if isinstance(plan, JoinPlan):
        rewritten = replace(
            plan,
            left=_rewrite_tree(plan.left, local),
            right=_rewrite_tree(plan.right, local),
        )
    elif isinstance(plan, UnionPlan):
        rewritten = replace(
            plan,
            inputs=tuple(_rewrite_tree(item, local) for item in plan.inputs),
        )
    else:
        rewritten = replace(plan, input=_rewrite_tree(plan.input, local))
    return local(rewritten)


def _predicate_pushdown(plan: LogicalPlan) -> LogicalPlan:
    def rewrite(node: LogicalPlan) -> LogicalPlan:
        if not isinstance(node, FilterPlan) or not isinstance(node.input, ProjectPlan):
            return node
        projection = node.input
        mapping: dict[ColumnId, ColumnId] = {}
        for item in projection.projections:
            if not isinstance(item.expression, ColumnRef):
                return node
            mapping[item.column.id] = item.expression.column_id
        predicate = _map_expression_columns(node.predicate, mapping)
        if predicate is None:
            return node
        filtered_input = FilterPlan(
            projection.input,
            predicate,
            after_filter(projection.input.metadata),
        )
        return ProjectPlan(filtered_input, projection.projections, node.metadata)

    return _rewrite_tree(plan, rewrite)


def _required_column_liveness(plan: LogicalPlan) -> LogicalPlan:
    def rewrite(node: LogicalPlan) -> LogicalPlan:
        if not isinstance(node, ProjectPlan):
            return node
        required: set[ColumnId] = set()
        for projection in node.projections:
            required.update(_expression_columns(projection.expression))
        if isinstance(node.input, ScanPlan):
            scan = node.input
            columns = tuple(
                column for column in scan.metadata.columns if column.id in required
            )
            if not columns or len(columns) == len(scan.metadata.columns):
                return node
            return replace(
                node,
                input=replace(
                    scan,
                    metadata=after_projection(scan.metadata, columns),
                ),
            )
        if not isinstance(node.input, ProjectPlan):
            return node
        inner = node.input
        retained = tuple(
            projection
            for projection in inner.projections
            if projection.column.id in required
        )
        if not retained or len(retained) == len(inner.projections):
            return node
        columns = tuple(projection.column for projection in retained)
        pruned = ProjectPlan(
            inner.input,
            retained,
            after_projection(inner.input.metadata, columns),
        )
        return replace(node, input=pruned)

    return _rewrite_tree(plan, rewrite)


def _limit_and_topk(plan: LogicalPlan) -> LogicalPlan:
    def rewrite(node: LogicalPlan) -> LogicalPlan:
        if not isinstance(node, LimitPlan):
            return node
        if isinstance(node.input, LimitPlan):
            inner = node.input
            remaining = max(0, inner.count - node.offset)
            return LimitPlan(
                inner.input,
                min(node.count, remaining),
                inner.offset + node.offset,
                node.metadata,
            )
        if isinstance(node.input, SortPlan):
            ordered = node.input
            return TopKPlan(
                ordered.input,
                ordered.keys,
                node.count,
                node.offset,
                node.metadata,
            )
        return node

    return _rewrite_tree(plan, rewrite)


def _remove_redundant_nodes(plan: LogicalPlan) -> LogicalPlan:
    def rewrite(node: LogicalPlan) -> LogicalPlan:
        if (
            isinstance(node, SortPlan)
            and isinstance(node.input, SortPlan)
            and node.keys == node.input.keys
        ):
            return replace(node, input=node.input.input)
        if isinstance(node, ProjectPlan) and _is_identity_projection(node):
            return replace(node.input, metadata=node.metadata)
        return node

    return _rewrite_tree(plan, rewrite)


def _is_identity_projection(plan: ProjectPlan) -> bool:
    if len(plan.projections) != len(plan.input.metadata.columns):
        return False
    input_by_id = {column.id: column for column in plan.input.metadata.columns}
    return all(
        isinstance(projection.expression, ColumnRef)
        and projection.expression.column_id == projection.column.id
        and input_by_id.get(projection.column.id) == projection.column
        for projection in plan.projections
    )


def _expression_columns(expression: Expression) -> set[ColumnId]:
    if isinstance(expression, ColumnRef):
        return {expression.column_id}
    if isinstance(expression, LiteralValue):
        return set()
    if isinstance(expression, (UnaryExpression, CastExpression)):
        return _expression_columns(expression.operand)
    if isinstance(expression, BinaryExpression):
        return _expression_columns(expression.left) | _expression_columns(
            expression.right
        )
    if isinstance(expression, CaseWhen):
        return (
            _expression_columns(expression.condition)
            | _expression_columns(expression.value)
            | _expression_columns(expression.otherwise)
        )
    if isinstance(expression, FunctionCall):
        refs: set[ColumnId] = set()
        for item in expression.arguments:
            refs.update(_expression_columns(item))
        return refs
    refs: set[ColumnId] = set()
    for item in expression.arguments:
        refs.update(_expression_columns(item))
    for key in expression.order_by:
        refs.update(_expression_columns(key.expression))
    for item in expression.partition_by:
        refs.update(_expression_columns(item))
    return refs


def _map_expression_columns(
    expression: Expression,
    mapping: dict[ColumnId, ColumnId],
) -> Expression | None:
    if isinstance(expression, ColumnRef):
        target = mapping.get(expression.column_id)
        return ColumnRef(target) if target is not None else None
    if isinstance(expression, LiteralValue):
        return expression
    if isinstance(expression, (UnaryExpression, CastExpression)):
        operand = _map_expression_columns(expression.operand, mapping)
        return replace(expression, operand=operand) if operand is not None else None
    if isinstance(expression, BinaryExpression):
        left = _map_expression_columns(expression.left, mapping)
        right = _map_expression_columns(expression.right, mapping)
        if left is None or right is None:
            return None
        return replace(expression, left=left, right=right)
    if isinstance(expression, CaseWhen):
        condition = _map_expression_columns(expression.condition, mapping)
        value = _map_expression_columns(expression.value, mapping)
        otherwise = _map_expression_columns(expression.otherwise, mapping)
        if condition is None or value is None or otherwise is None:
            return None
        return replace(
            expression,
            condition=condition,
            value=value,
            otherwise=otherwise,
        )
    if isinstance(expression, FunctionCall):
        arguments = tuple(
            _map_expression_columns(item, mapping) for item in expression.arguments
        )
        if any(item is None for item in arguments):
            return None
        return replace(expression, arguments=arguments)
    return None


def _common_subplan_recommendations(
    plan: LogicalPlan,
) -> tuple[dict[str, object], ...]:
    counts: dict[str, tuple[int, LogicalPlan]] = {}

    def visit(node: LogicalPlan) -> None:
        encoded = json.dumps(plan_to_dict(node), sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()[:16]
        count, _ = counts.get(fingerprint, (0, node))
        counts[fingerprint] = (count + 1, node)
        if isinstance(node, JoinPlan):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, UnionPlan):
            for item in node.inputs:
                visit(item)
        elif not isinstance(node, ScanPlan):
            visit(node.input)

    visit(plan)
    return tuple(
        {
            "kind": "persist",
            "fingerprint": fingerprint,
            "occurrences": count,
            "node": type(node).__name__,
            "message": (
                "Repeated logical subplan; consider explicit DataFrame.persist()."
            ),
        }
        for fingerprint, (count, node) in sorted(counts.items())
        if count > 1 and not isinstance(node, ScanPlan)
    )
