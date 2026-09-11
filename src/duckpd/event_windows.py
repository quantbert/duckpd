"""Planning validation for exact fixed-grid event windows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from duckpd._logical import Column, ColumnId, EventWindowPlan, Nullability
from duckpd._metadata import after_event_windows, find_column
from duckpd._reductions import is_numeric_type, is_timestamp_type, is_timezone_aware_type
from duckpd._temporal import fixed_duration_ns
from duckpd.errors import AlignmentError, UnsupportedOperationError
from duckpd.series_embeddings import SeriesWindowSpec

if TYPE_CHECKING:
    from duckpd.frame import DataFrame

EventAnchor = Literal["floor", "ceil"]
IncompleteEventWindowPolicy = Literal["null", "error"]


def _labels(value: str | Sequence[str], *, parameter: str) -> tuple[str, ...]:
    labels = (value,) if isinstance(value, str) else tuple(value)
    if not labels or any(type(label) is not str or not label for label in labels):
        raise ValueError(f"{parameter} must contain non-empty column labels")
    if len(labels) != len(set(labels)):
        raise ValueError(f"{parameter} must contain unique column labels")
    return labels


def _timestamp_contract(columns: tuple[Column, ...]) -> None:
    if any(not is_timestamp_type(column.duckdb_type.upper()) for column in columns):
        raise ValueError("event_windows timestamp columns must have timestamp dtypes")
    types = {column.duckdb_type.upper() for column in columns}
    timezones = {column.timezone for column in columns}
    awareness = {is_timezone_aware_type(column.duckdb_type.upper()) for column in columns}
    if len(types) != 1 or len(timezones) != 1 or len(awareness) != 1:
        raise ValueError(
            "event_windows timestamp and availability columns must have identical "
            "dtypes and timezone semantics"
        )
    timezone = columns[0].timezone
    if timezone is not None and timezone.casefold() not in {"utc", "etc/utc"}:
        raise UnsupportedOperationError("event_windows currently supports only UTC timestamps")


def plan_event_windows(
    observations: DataFrame,
    events: DataFrame,
    *,
    on: str,
    bar_label: Literal["start"],
    event_on: str,
    by: str | Sequence[str],
    event_id: str | Sequence[str],
    columns: Mapping[str, str],
    window: tuple[int, int],
    step: str,
    anchor: EventAnchor,
    available_at: str,
    event_available_at: str,
    incomplete: IncompleteEventWindowPolicy = "null",
    metadata_prefix: str = "window",
) -> EventWindowPlan:
    """Validate event-grid semantics and build a side-effect-free binary plan."""
    if observations._session is not events._session:
        raise AlignmentError("event_windows inputs must belong to the same session")
    if bar_label != "start":
        raise UnsupportedOperationError("event_windows currently requires bar_label='start'")
    if anchor not in {"floor", "ceil"}:
        raise ValueError("anchor must be 'floor' or 'ceil'")
    if incomplete not in {"null", "error"}:
        raise ValueError("incomplete must be 'null' or 'error'")
    if type(metadata_prefix) is not str or not metadata_prefix:
        raise ValueError("metadata_prefix must be a non-empty string")
    if (
        type(window) is not tuple
        or len(window) != 2
        or any(type(offset) is not int for offset in window)
        or window[0] >= window[1]
    ):
        raise ValueError("window must be a pair of integer offsets with start < end")
    step_ns = fixed_duration_ns(step, parameter="step")
    if step_ns % 1_000 != 0:
        raise UnsupportedOperationError("event_windows step must be representable in microseconds")

    by_labels = _labels(by, parameter="by")
    event_id_labels = _labels(event_id, parameter="event_id")
    observation_keys = tuple(find_column(observations._plan.metadata, label) for label in by_labels)
    event_keys = tuple(find_column(events._plan.metadata, label) for label in by_labels)
    for observation_key, event_key in zip(observation_keys, event_keys, strict=True):
        if observation_key.duckdb_type != event_key.duckdb_type:
            raise ValueError(f"event_windows key {observation_key.label!r} has incompatible dtypes")

    observation_time = find_column(observations._plan.metadata, on)
    event_time = find_column(events._plan.metadata, event_on)
    observation_availability = find_column(observations._plan.metadata, available_at)
    event_availability = find_column(events._plan.metadata, event_available_at)
    _timestamp_contract(
        (
            observation_time,
            event_time,
            observation_availability,
            event_availability,
        )
    )
    event_ids = tuple(find_column(events._plan.metadata, label) for label in event_id_labels)

    if not columns:
        raise ValueError("columns must be a non-empty output-to-source mapping")
    channel_mapping = dict(columns)
    if any(type(label) is not str or not label for label in channel_mapping):
        raise ValueError("columns output labels must be non-empty strings")
    if any(type(label) is not str or not label for label in channel_mapping.values()):
        raise ValueError("columns source labels must be non-empty strings")

    existing_labels = {column.label for column in events._plan.metadata.columns}
    metadata_labels = {
        "start": f"{metadata_prefix}_window_start",
        "end": f"{metadata_prefix}_window_end",
        "count": f"{metadata_prefix}_window_count",
        "complete": f"{metadata_prefix}_window_complete",
        "available": f"{metadata_prefix}_window_available_at",
    }
    requested_labels = [*channel_mapping, *metadata_labels.values()]
    collisions = sorted(existing_labels.intersection(requested_labels))
    if collisions:
        raise ValueError(f"event_windows output columns already exist: {collisions!r}")
    if len(requested_labels) != len(set(requested_labels)):
        raise ValueError("event_windows output and metadata labels must be unique")

    length = window[1] - window[0]
    window_spec = SeriesWindowSpec(
        length,
        sampling="fixed_grid",
        step=step,
        order_by=(observation_time.id,),
        partition_by=tuple(column.id for column in observation_keys),
        origin="event",
    )
    output_nullability = Nullability.NULLABLE if incomplete == "null" else Nullability.NON_NULL
    channels: list[tuple[ColumnId, Column]] = []
    for output_label, source_label in channel_mapping.items():
        source_column = find_column(observations._plan.metadata, source_label)
        if (
            not is_numeric_type(source_column.duckdb_type.upper())
            or source_column.duckdb_type.upper() == "BOOLEAN"
        ):
            raise UnsupportedOperationError(
                f"event_windows source column {source_label!r} must be numeric"
            )
        output = Column(
            ColumnId.create(),
            output_label,
            f"FLOAT[{length}]",
            nullable=output_nullability,
            series_window=window_spec,
        )
        channels.append((source_column.id, output))

    window_start = Column(
        ColumnId.create(),
        metadata_labels["start"],
        event_time.duckdb_type,
        nullable=Nullability.NON_NULL,
        timezone=event_time.timezone,
    )
    window_end = Column(
        ColumnId.create(),
        metadata_labels["end"],
        event_time.duckdb_type,
        nullable=Nullability.NON_NULL,
        timezone=event_time.timezone,
    )
    window_count = Column(
        ColumnId.create(),
        metadata_labels["count"],
        "BIGINT",
        nullable=Nullability.NON_NULL,
    )
    window_complete = Column(
        ColumnId.create(),
        metadata_labels["complete"],
        "BOOLEAN",
        nullable=Nullability.NON_NULL,
    )
    window_available = Column(
        ColumnId.create(),
        metadata_labels["available"],
        event_availability.duckdb_type,
        nullable=output_nullability,
        timezone=event_availability.timezone,
    )
    output_columns = (
        *events._plan.metadata.columns,
        *(output for _, output in channels),
        window_start,
        window_end,
        window_count,
        window_complete,
        window_available,
    )
    metadata = after_event_windows(events._plan.metadata, output_columns)
    return EventWindowPlan(
        observations=observations._plan,
        events=events._plan,
        observation_time=observation_time.id,
        event_time=event_time.id,
        observation_keys=tuple(column.id for column in observation_keys),
        event_keys=tuple(column.id for column in event_keys),
        event_ids=tuple(column.id for column in event_ids),
        channels=tuple(channels),
        start_offset=window[0],
        end_offset=window[1],
        step_ns=step_ns,
        anchor=anchor,
        observation_available_at=observation_availability.id,
        event_available_at=event_availability.id,
        incomplete=incomplete,
        bar_label=bar_label,
        window_start=window_start,
        window_end=window_end,
        window_count=window_count,
        window_complete=window_complete,
        window_available_at=window_available,
        metadata=metadata,
    )
