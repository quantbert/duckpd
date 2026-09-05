"""Feature store catalog models, validation, and resolution."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast


def parse_timestamp(value: Any) -> datetime:
    """Parse an aware ISO 8601 timestamp and normalize it to UTC."""
    if not isinstance(value, str):
        raise TypeError("timestamp must be an ISO 8601 string")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid ISO timestamp: {value}") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"Timestamp must include a timezone offset: {value}")
    return timestamp.astimezone(UTC)


def parse_availability_delay(value: Any, reference: str) -> timedelta:
    """Parse a non-negative, calendar-independent ISO 8601 duration."""
    if not isinstance(value, str):
        raise ValueError(f"Feature {reference} must define availability_delay")
    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?"
        r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?",
        value,
    )
    if match is None or value.endswith("T") or not any(part is not None for part in match.groups()):
        raise ValueError(f"Invalid availability_delay for {reference}: {value!r}")
    parts = match.groupdict(default="0")
    return timedelta(
        days=int(parts["days"]),
        hours=int(parts["hours"]),
        minutes=int(parts["minutes"]),
        seconds=float(parts["seconds"]),
    )


def validate_catalog(
    catalog: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate a catalog dictionary adhering to Catalog Version 1."""
    catalog_version = catalog.get("catalog_version")
    if catalog_version != 1:
        raise ValueError(f"Unsupported catalog version: {catalog_version!r}")

    dataset_entries_raw = catalog.get("datasets")
    if not isinstance(dataset_entries_raw, list) or not dataset_entries_raw:
        raise ValueError("Catalog must define at least one dataset")
    dataset_entries: list[object] = cast("list[object]", dataset_entries_raw)
    dataset_index: dict[str, dict[str, Any]] = {}
    for entry_raw in dataset_entries:
        if not isinstance(entry_raw, dict):
            raise ValueError("Catalog dataset entry must be a mapping")
        entry = cast("dict[str, Any]", entry_raw)
        name: Any = entry.get("name")
        kind: Any = entry.get("kind")
        if not isinstance(name, str) or not name:
            raise ValueError("Catalog datasets must have non-empty names")
        if name in dataset_index:
            raise ValueError(f"Duplicate catalog dataset: {name}")
        if kind not in ("timeseries", "table"):
            raise ValueError(f"Dataset {name!r} has invalid kind: {kind!r}")
        if name == "features":
            raise ValueError("Table dataset name 'features' is reserved")
        if kind == "timeseries":
            time_col: Any = entry.get("time_column")
            if not isinstance(time_col, str) or not time_col:
                raise ValueError(f"Timeseries dataset {name!r} requires time_column")
            series_keys_raw = entry.get("series_keys")
            if not isinstance(series_keys_raw, list):
                raise ValueError(f"Timeseries dataset {name!r} requires series_keys")
            series_keys = cast("list[Any]", series_keys_raw)
            if not all(isinstance(key, str) and key for key in series_keys):
                raise ValueError(f"Timeseries dataset {name!r} requires series_keys")
        dataset_index[name] = entry

    feature_entries_raw = catalog.get("features", {})
    if not isinstance(feature_entries_raw, dict):
        raise ValueError("Catalog features must be a mapping")
    feature_entries: dict[str, Any] = cast("dict[str, Any]", feature_entries_raw)

    feature_index: dict[str, dict[str, Any]] = {}
    for ref_key, entry_val in feature_entries.items():
        reference = str(ref_key)
        if not isinstance(entry_val, dict):
            raise ValueError(f"Catalog feature entry for {reference!r} must be a mapping")
        entry = cast("dict[str, Any]", entry_val)
        expected_reference = f"{entry.get('dataset')}:{entry.get('name')}"
        if reference != expected_reference:
            raise ValueError(
                f"Catalog feature {reference!r} must be keyed as {expected_reference!r}"
            )
        ds_name = str(entry.get("dataset"))
        dataset = dataset_index.get(ds_name)
        if dataset is None or dataset["kind"] != "timeseries":
            raise ValueError(f"Catalog feature {reference!r} must belong to a timeseries dataset")
        feature_index[reference] = entry

    return dataset_index, feature_index


def resolve_features(
    requested_features: Any,
    feature_entries: dict[str, dict[str, Any]],
    dataset_entries: dict[str, dict[str, Any]],
) -> list[tuple[str, str, str, str]]:
    """Resolve user-requested feature references, wildcards, and aliases.

    Returns
    -------
    list[tuple[str, str, str, str]]
        List of (output_alias, canonical_reference, dataset_name, physical_feature_name)
    """
    if not requested_features:
        raise ValueError("At least one feature must be requested")

    if not isinstance(requested_features, (Mapping, Sequence)) or isinstance(
        requested_features, (str, bytes)
    ):
        raise TypeError("features must be a sequence of references or an alias mapping")

    if isinstance(requested_features, Mapping):
        mapping = cast("Mapping[Any, Any]", requested_features)
        requested_items: list[tuple[str | None, str]] = [
            (str(k), str(v)) for k, v in mapping.items()
        ]
    else:
        seq = cast("Sequence[Any]", requested_features)
        requested_items = [(None, str(reference)) for reference in seq]

    expanded_items: list[tuple[str | None, str]] = []
    for requested_alias, requested_reference in requested_items:
        if not requested_reference:
            raise ValueError("Feature references must be non-empty strings")
        if requested_reference.endswith(":*"):
            if requested_alias is not None:
                raise ValueError("Feature wildcards cannot have output aliases")
            prefix = requested_reference[:-1]
            matches = [reference for reference in feature_entries if reference.startswith(prefix)]
            if not matches:
                raise ValueError(f"Unknown feature group: {requested_reference}")
            expanded_items.extend((None, reference) for reference in matches)
        else:
            expanded_items.append((requested_alias, requested_reference))

    resolved_features: list[tuple[str, str, str, str]] = []
    seen_requests: set[tuple[str, str]] = set()
    for requested_alias, requested_reference in expanded_items:
        reference = requested_reference
        if ":" not in reference:
            matches = [
                candidate
                for candidate, entry in feature_entries.items()
                if entry["name"] == reference
            ]
            if not matches:
                raise ValueError(f"Unknown feature: {reference}")
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous feature {reference!r}; use one of: {', '.join(matches)}"
                )
            reference = matches[0]
        if reference not in feature_entries:
            raise ValueError(
                f"Unknown feature: {reference}. Available: {', '.join(feature_entries)}"
            )

        entry = feature_entries[reference]
        output_name = requested_alias or entry["name"]
        if not isinstance(output_name, str) or not output_name:
            raise ValueError("Output aliases must be non-empty strings")
        dataset_entry = dataset_entries[entry["dataset"]]
        reserved_columns = {
            dataset_entry["time_column"],
            *dataset_entry["series_keys"],
        }
        if output_name in reserved_columns:
            raise ValueError(f"Output alias is reserved: {output_name}")
        request_key = (output_name, reference)
        if request_key not in seen_requests:
            resolved_features.append((output_name, reference, entry["dataset"], entry["name"]))
            seen_requests.add(request_key)

    output_names = [output_name for output_name, *_ in resolved_features]
    duplicate_outputs = sorted(name for name in set(output_names) if output_names.count(name) > 1)
    if duplicate_outputs:
        raise ValueError(
            f"Duplicate output columns: {', '.join(duplicate_outputs)}. "
            "Use an alias mapping to give them distinct names."
        )
    return resolved_features


def normalize_filters(
    filters: Any,
) -> dict[str, list[str]] | None:
    """Validate and normalize filter mappings."""
    if filters is None:
        return None
    if not isinstance(filters, Mapping):
        raise TypeError("filters must be a mapping of columns to string values")
    if not filters:
        raise ValueError("filters cannot be empty")
    filter_map = cast("Mapping[Any, Any]", filters)
    normalized_filters: dict[str, list[str]] = {}
    for column_raw, values in filter_map.items():
        column = str(column_raw)
        if not column:
            raise ValueError("filter columns must be non-empty strings")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"filter values for {column!r} must be a sequence")
        val_seq = cast("Sequence[Any]", values)
        normalized_values: list[str] = []
        for value in val_seq:
            if not isinstance(value, str):
                raise TypeError(f"filter values for {column!r} must be strings")
            normalized_values.append(value)
        if not normalized_values:
            raise ValueError(f"filter values for {column!r} cannot be empty")
        normalized_filters[column] = normalized_values
    return normalized_filters
