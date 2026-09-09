"""Feature store catalog models, validation, and resolution."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from duckpd.embeddings import EmbeddingModelSpec

_EMBEDDING_MODEL_FIELDS = frozenset(
    {
        "model",
        "revision",
        "dimension",
        "backend",
        "normalize",
        "pooling",
        "document_prefix",
        "query_prefix",
    }
)


def _validate_embedding_models(value: Any) -> dict[str, EmbeddingModelSpec]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Catalog embedding_models must be a mapping")
    models: dict[str, EmbeddingModelSpec] = {}
    raw_models = cast("Mapping[object, object]", value)
    for raw_name, raw_specification in raw_models.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("Catalog embedding model keys must be non-empty strings")
        if not isinstance(raw_specification, Mapping):
            raise ValueError(f"Embedding model {raw_name!r} must be a mapping")
        specification_data = dict(cast("Mapping[str, Any]", raw_specification))
        unknown = sorted(set(specification_data) - _EMBEDDING_MODEL_FIELDS)
        if unknown:
            raise ValueError(
                f"Embedding model {raw_name!r} has unknown fields: {', '.join(unknown)}"
            )
        backend = specification_data.get("backend", "fastembed")
        if backend not in {"fastembed", "transformers"}:
            raise ValueError(f"Embedding model {raw_name!r} has unsupported backend: {backend!r}")
        revision = specification_data.get("revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) is None:
            raise ValueError(
                f"Embedding model {raw_name!r} revision must be an immutable commit digest"
            )
        try:
            specification = EmbeddingModelSpec(**specification_data)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid embedding model {raw_name!r}: {error}") from None
        if backend == "fastembed" and (
            specification.pooling != "model-default" or not specification.normalize
        ):
            raise ValueError(
                f"Embedding model {raw_name!r} uses FastEmbed fields the backend cannot honor"
            )
        if backend == "transformers" and specification.pooling not in {"cls", "mean"}:
            raise ValueError(
                f"Embedding model {raw_name!r} Transformers pooling must be 'cls' or 'mean'"
            )
        models[raw_name] = specification
    return models


def _embedding_reference(
    declaration: Mapping[str, Any],
    *,
    owner: str,
    models: Mapping[str, EmbeddingModelSpec],
) -> str | None:
    raw_reference = declaration.get("embedding_model")
    if raw_reference is None:
        return None
    if not isinstance(raw_reference, str) or not raw_reference:
        raise ValueError(f"{owner} embedding_model must be a non-empty string")
    if raw_reference not in models:
        raise ValueError(f"{owner} references unknown embedding model {raw_reference!r}")
    return raw_reference


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
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, EmbeddingModelSpec],
]:
    """Validate the complete catalog version 1 schema."""
    catalog_version = catalog.get("catalog_version")
    if catalog_version != 1:
        raise ValueError(f"Unsupported catalog version: {catalog_version!r}")

    embedding_models = _validate_embedding_models(catalog.get("embedding_models"))
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
        columns_raw = entry.get("columns")
        if columns_raw is not None:
            if kind != "table" or not isinstance(columns_raw, Mapping):
                raise ValueError(f"Dataset {name!r} columns metadata must be a table mapping")
            normalized_columns: dict[str, dict[str, Any]] = {}
            columns = cast("Mapping[object, object]", columns_raw)
            for raw_column_name, raw_declaration in columns.items():
                if not isinstance(raw_column_name, str) or not raw_column_name:
                    raise ValueError(f"Dataset {name!r} column names must be non-empty strings")
                if not isinstance(raw_declaration, Mapping):
                    raise ValueError(
                        f"Dataset {name!r} column {raw_column_name!r} metadata must be a mapping"
                    )
                declaration = cast("Mapping[str, Any]", raw_declaration)
                unknown_column_fields = sorted(set(declaration) - {"embedding_model"})
                if unknown_column_fields:
                    raise ValueError(
                        f"Dataset {name!r} column {raw_column_name!r} has unknown fields: "
                        f"{', '.join(unknown_column_fields)}"
                    )
                embedding = _embedding_reference(
                    declaration,
                    owner=f"Dataset {name!r} column {raw_column_name!r}",
                    models=embedding_models,
                )
                normalized = dict(declaration)
                if embedding is not None:
                    normalized["embedding"] = embedding
                normalized_columns[raw_column_name] = normalized
            entry = {**entry, "columns": normalized_columns}
        dataset_index[name] = entry

    feature_entries_raw = catalog.get("features", {})
    if not isinstance(feature_entries_raw, dict):
        raise ValueError("Catalog features must be a mapping")
    feature_entries: dict[str, Any] = cast("dict[str, Any]", feature_entries_raw)

    feature_index: dict[str, dict[str, Any]] = {}
    physical_embeddings: dict[tuple[str, str], str] = {}
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
        embedding = _embedding_reference(
            entry,
            owner=f"Catalog feature {reference!r}",
            models=embedding_models,
        )
        normalized_entry = dict(entry)
        if embedding is not None:
            normalized_entry["embedding"] = embedding
            key = (entry["dataset"], entry["name"])
            existing = physical_embeddings.setdefault(key, embedding)
            if existing != embedding:
                raise ValueError(
                    f"Catalog feature {reference!r} conflicts with another embedding "
                    f"declaration for {key[0]}.{key[1]}"
                )
        feature_index[reference] = normalized_entry

    return dataset_index, feature_index, embedding_models


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
