"""Immutable contracts for time-series representations and typed queries."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from decimal import Decimal
from math import fsum, isfinite, sqrt
from typing import TYPE_CHECKING, Literal, cast

from duckpd._temporal import fixed_duration_ns

if TYPE_CHECKING:
    from duckpd.frame import DataFrame


SeriesSampling = Literal["observations", "fixed_grid"]
SeriesNormalization = Literal["none", "center", "zscore"]
ZeroScalePolicy = Literal["error", "null"]
SeriesNullPolicy = Literal["propagate", "error"]
SeriesMetadataOrigin = Literal["generated", "application_asserted", "sidecar", "table", "catalog"]
SeriesWindowOrigin = Literal["rolling", "event", "application_asserted", "catalog"]

_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _positive_integer(value: object, *, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _names(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    items = cast("tuple[object, ...]", value)
    if any(type(item) is not str or not item for item in items):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} must contain unique names")
    return cast("tuple[str, ...]", items)


def _canonical_duration(value: str | timedelta) -> str:
    nanoseconds = fixed_duration_ns(value, parameter="step")
    day, remainder = divmod(nanoseconds, 86_400_000_000_000)
    hour, remainder = divmod(remainder, 3_600_000_000_000)
    minute, remainder = divmod(remainder, 60_000_000_000)
    second, fractional = divmod(remainder, 1_000_000_000)
    date_part = f"{day}D" if day else ""
    time_parts = ""
    if hour:
        time_parts += f"{hour}H"
    if minute:
        time_parts += f"{minute}M"
    if second or fractional or not (date_part or time_parts):
        if fractional:
            fraction = f"{fractional:09d}".rstrip("0")
            time_parts += f"{second}.{fraction}S"
        else:
            time_parts += f"{second}S"
    return f"P{date_part}{'T' + time_parts if time_parts else ''}"


def _canonical_hash(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _strict_mapping(
    value: Mapping[str, object], *, fields: frozenset[str], owner: str
) -> dict[str, object]:
    data = dict(value)
    unknown = sorted(set(data) - fields)
    if unknown:
        raise ValueError(f"{owner} has unknown fields: {', '.join(unknown)}")
    return data


@dataclass(frozen=True)
class SeriesEmbeddingModelSpec:
    """Immutable identity and input contract for a learned series encoder."""

    model: str
    revision: str
    artifact_sha256: str
    backend: str
    dimension: int
    input_length: int
    input_channels: tuple[str, ...]
    input_normalization: str
    pooling: str
    adapter_revision: str

    def __post_init__(self) -> None:
        for field_name in (
            "model",
            "revision",
            "backend",
            "input_normalization",
            "pooling",
            "adapter_revision",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.revision.casefold() in {"main", "master", "latest", "head"}:
            raise ValueError("revision must identify an immutable model revision")
        if type(self.artifact_sha256) is not str or _SHA256.fullmatch(self.artifact_sha256) is None:
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        _positive_integer(self.dimension, field_name="dimension")
        _positive_integer(self.input_length, field_name="input_length")
        _names(self.input_channels, field_name="input_channels")

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return cast("dict[str, object]", asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SeriesEmbeddingModelSpec:
        fields = frozenset(cls.__dataclass_fields__)
        return cls(**_strict_mapping(value, fields=fields, owner="series embedding model"))  # type: ignore[arg-type]


@dataclass(frozen=True)
class SeriesRepresentationSpec:
    """Complete, canonical identity of one native or learned series space."""

    window: int
    channels: tuple[str, ...]
    sampling: SeriesSampling
    data_contract: str
    step: str | timedelta | None = None
    normalization: SeriesNormalization = "none"
    unit_norm: bool = False
    zero_scale: ZeroScalePolicy = "error"
    encoder: SeriesEmbeddingModelSpec | None = None

    def __post_init__(self) -> None:
        _positive_integer(self.window, field_name="window")
        _names(self.channels, field_name="channels")
        if self.sampling not in {"observations", "fixed_grid"}:
            raise ValueError("sampling must be 'observations' or 'fixed_grid'")
        if type(self.data_contract) is not str or not self.data_contract:
            raise ValueError("data_contract must be a non-empty versioned identifier")
        if self.normalization not in {"none", "center", "zscore"}:
            raise ValueError("normalization must be 'none', 'center', or 'zscore'")
        if type(self.unit_norm) is not bool:
            raise TypeError("unit_norm must be a boolean")
        if self.zero_scale not in {"error", "null"}:
            raise ValueError("zero_scale must be 'error' or 'null'")
        if self.step is None:
            if self.sampling == "fixed_grid":
                raise ValueError("fixed_grid sampling requires a positive fixed step")
        else:
            object.__setattr__(self, "step", _canonical_duration(self.step))
        if self.encoder is not None:
            if self.encoder.input_length != self.window:
                raise ValueError("encoder input_length must equal representation window")
            if self.encoder.input_channels != self.channels:
                raise ValueError("encoder input_channels must equal representation channels")

    @property
    def dimension(self) -> int:
        if self.encoder is not None:
            return self.encoder.dimension
        return self.window * len(self.channels)

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "window": self.window,
            "channels": list(self.channels),
            "sampling": self.sampling,
            "data_contract": self.data_contract,
            "step": self.step,
            "normalization": self.normalization,
            "unit_norm": self.unit_norm,
            "zero_scale": self.zero_scale,
            "encoder": self.encoder.to_dict() if self.encoder is not None else None,
            "output_type": "FLOAT",
            "dimension": self.dimension,
            "layout": "channel-major-oldest-first-v1" if self.encoder is None else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SeriesRepresentationSpec:
        fields = frozenset(
            {
                "schema_version",
                "window",
                "channels",
                "sampling",
                "data_contract",
                "step",
                "normalization",
                "unit_norm",
                "zero_scale",
                "encoder",
                "output_type",
                "dimension",
                "layout",
            }
        )
        data = _strict_mapping(value, fields=fields, owner="series representation")
        if data.pop("schema_version", None) != _SCHEMA_VERSION:
            raise ValueError(f"series representation schema_version must be {_SCHEMA_VERSION}")
        expected_dimension = data.pop("dimension", None)
        if data.pop("output_type", None) != "FLOAT":
            raise ValueError("series representation output_type must be 'FLOAT'")
        layout = data.pop("layout", None)
        raw_channels = data.get("channels")
        if not isinstance(raw_channels, Sequence) or isinstance(raw_channels, (str, bytes)):
            raise TypeError("channels must be an array of strings")
        data["channels"] = tuple(cast("Sequence[object]", raw_channels))
        raw_encoder = data.get("encoder")
        if raw_encoder is not None:
            if not isinstance(raw_encoder, Mapping):
                raise TypeError("encoder must be an object or null")
            encoder_data = cast("Mapping[str, object]", raw_encoder)
            data["encoder"] = SeriesEmbeddingModelSpec.from_dict(encoder_data)
            if layout is not None:
                raise ValueError("learned series representations must not define native layout")
        elif layout != "channel-major-oldest-first-v1":
            raise ValueError("native series representation layout is unsupported")
        result = cls(**data)  # type: ignore[arg-type]
        if expected_dimension != result.dimension:
            raise ValueError("series representation dimension does not match its contract")
        return result


@dataclass(frozen=True)
class SeriesColumnSpec:
    """Representation identity attached to one fixed-size vector column."""

    representation: SeriesRepresentationSpec
    origin: SeriesMetadataOrigin = field(default="generated", compare=False)

    def __post_init__(self) -> None:
        if self.origin not in {
            "generated",
            "application_asserted",
            "sidecar",
            "table",
            "catalog",
        }:
            raise ValueError("invalid series representation metadata origin")

    @property
    def fingerprint(self) -> str:
        return self.representation.fingerprint


@dataclass(frozen=True)
class SeriesWindowSpec:
    """Raw fixed-count window shape and sampling provenance."""

    window: int
    sampling: SeriesSampling = "observations"
    step: str | timedelta | None = None
    order_by: tuple[object, ...] = ()
    partition_by: tuple[object, ...] = ()
    origin: SeriesWindowOrigin = "rolling"

    def __post_init__(self) -> None:
        _positive_integer(self.window, field_name="window")
        if self.sampling not in {"observations", "fixed_grid"}:
            raise ValueError("sampling must be 'observations' or 'fixed_grid'")
        if self.step is None:
            if self.sampling == "fixed_grid":
                raise ValueError("fixed_grid sampling requires a positive fixed step")
        else:
            object.__setattr__(self, "step", _canonical_duration(self.step))
        if self.origin not in {"rolling", "event", "application_asserted", "catalog"}:
            raise ValueError("invalid series window metadata origin")


@dataclass(frozen=True)
class EmbeddedSeriesQuery:
    """Immutable pre-encoded values retaining their representation-space identity."""

    values: tuple[float, ...]
    representation_fingerprint: str

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("values must contain at least one number")
        if any(type(value) not in {int, float} or not isfinite(value) for value in self.values):
            raise ValueError("values must contain only finite numbers")
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))
        if (
            type(self.representation_fingerprint) is not str
            or _SHA256.fullmatch(self.representation_fingerprint) is None
        ):
            raise ValueError("representation_fingerprint must be a lowercase SHA-256 digest")


SeriesQuerySnapshot = tuple[tuple[str, tuple[float, ...]], ...]


def snapshot_series_query(
    query: object,
    representation: SeriesRepresentationSpec,
) -> SeriesQuerySnapshot:
    """Validate and freeze raw channel observations without representing them."""
    if not isinstance(query, Mapping):
        raise TypeError("query must be a mapping of channel names to numeric sequences")
    raw_query = dict(cast("Mapping[object, object]", query))
    if any(type(channel) is not str or not channel for channel in raw_query):
        raise TypeError("query keys must be non-empty channel names")
    expected = set(representation.channels)
    actual = set(cast("dict[str, object]", raw_query))
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "query must exactly match representation.channels; "
            f"missing={missing}, unexpected={unexpected}"
        )

    snapshot: list[tuple[str, tuple[float, ...]]] = []
    for channel in representation.channels:
        raw_values = raw_query[channel]
        if isinstance(raw_values, (str, bytes, bytearray)):
            raise TypeError(f"query channel {channel!r} must be a numeric sequence")
        try:
            values = tuple(cast("Sequence[object]", raw_values))
        except TypeError:
            raise TypeError(f"query channel {channel!r} must be a numeric sequence") from None
        if len(values) != representation.window:
            raise ValueError(
                f"query channel {channel!r} length {len(values)} does not match "
                f"representation window {representation.window}"
            )
        converted: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
                raise TypeError(f"query channel {channel!r} must contain only numeric scalars")
            number = float(value)
            if not isfinite(number):
                raise ValueError(f"query channel {channel!r} must contain only finite values")
            converted.append(number)
        snapshot.append((channel, tuple(converted)))
    return tuple(snapshot)


def embed_native_series_query(
    query: SeriesQuerySnapshot,
    representation: SeriesRepresentationSpec,
) -> EmbeddedSeriesQuery:
    """Apply the canonical native corpus recipe to one frozen query."""
    from duckpd.errors import UnsupportedOperationError

    if representation.encoder is not None:
        raise UnsupportedOperationError(
            "Native series query encoding requires a representation with encoder=None"
        )
    by_channel = dict(query)
    flattened: list[float] = []
    for channel in representation.channels:
        values = by_channel[channel]
        if representation.normalization == "none":
            normalized = values
        else:
            mean = fsum(values) / len(values)
            centered = tuple(value - mean for value in values)
            if representation.normalization == "center":
                normalized = centered
            else:
                scale = sqrt(fsum(value * value for value in centered) / len(centered))
                if scale == 0.0:
                    raise ValueError(
                        "series query representation has zero scale and cannot be encoded"
                    )
                normalized = tuple(value / scale for value in centered)
        flattened.extend(normalized)

    if representation.unit_norm:
        norm = sqrt(fsum(value * value for value in flattened))
        if norm == 0.0:
            raise ValueError("series query representation has zero norm and cannot be encoded")
        flattened = [value / norm for value in flattened]

    encoded: list[float] = []
    for value in flattened:
        try:
            float32 = struct.unpack("!f", struct.pack("!f", value))[0]
        except OverflowError:
            raise ValueError("series query representation produced a non-finite value") from None
        if not isfinite(float32):
            raise ValueError("series query representation produced a non-finite value")
        encoded.append(float32)
    return EmbeddedSeriesQuery(tuple(encoded), representation.fingerprint)


def series_embedding_model(
    model: str,
    *,
    revision: str,
    artifact_sha256: str,
    backend: str,
    dimension: int,
    input_length: int,
    input_channels: tuple[str, ...],
    input_normalization: str,
    pooling: str,
    adapter_revision: str,
) -> SeriesEmbeddingModelSpec:
    """Create a side-effect-free learned series encoder specification."""
    return SeriesEmbeddingModelSpec(
        model=model,
        revision=revision,
        artifact_sha256=artifact_sha256,
        backend=backend,
        dimension=dimension,
        input_length=input_length,
        input_channels=input_channels,
        input_normalization=input_normalization,
        pooling=pooling,
        adapter_revision=adapter_revision,
    )


def series_representation(
    *,
    window: int,
    channels: tuple[str, ...],
    sampling: SeriesSampling,
    data_contract: str,
    step: str | timedelta | None = None,
    normalization: SeriesNormalization = "none",
    unit_norm: bool = False,
    zero_scale: ZeroScalePolicy = "error",
    encoder: SeriesEmbeddingModelSpec | None = None,
) -> SeriesRepresentationSpec:
    """Create a canonical native or learned time-series representation contract."""
    return SeriesRepresentationSpec(
        window=window,
        channels=channels,
        sampling=sampling,
        data_contract=data_contract,
        step=step,
        normalization=normalization,
        unit_norm=unit_norm,
        zero_scale=zero_scale,
        encoder=encoder,
    )


def embed_series(
    frame: DataFrame,
    *,
    columns: Mapping[str, str],
    into: str,
    representation: SeriesRepresentationSpec,
    batch_size: int = 256,
    null_policy: SeriesNullPolicy = "propagate",
) -> DataFrame:
    """Append one native deterministic series representation lazily."""
    from duckpd._logical import (
        Column,
        ColumnId,
        Nullability,
        SeriesRepresentationPlan,
    )
    from duckpd._metadata import after_embedding, find_column
    from duckpd.errors import UnsupportedOperationError
    from duckpd.frame import DataFrame

    if representation.encoder is not None:
        raise UnsupportedOperationError(
            "embed_series() currently supports only native representations with encoder=None"
        )
    if not columns:
        raise TypeError("columns must be a non-empty channel-to-column mapping")
    channel_mapping = dict(columns)
    if any(type(channel) is not str or not channel for channel in channel_mapping):
        raise TypeError("columns mapping keys must be non-empty channel names")
    if any(type(label) is not str or not label for label in channel_mapping.values()):
        raise TypeError("columns mapping values must be non-empty column labels")
    expected_channels = set(representation.channels)
    actual_channels = set(channel_mapping)
    if actual_channels != expected_channels:
        missing = sorted(expected_channels - actual_channels)
        unexpected = sorted(actual_channels - expected_channels)
        raise ValueError(
            "columns must exactly match representation.channels; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if type(into) is not str or not into:
        raise ValueError("into must be a non-empty string label")
    try:
        find_column(frame._plan.metadata, into, include_hidden=True)
    except KeyError:
        pass
    else:
        raise ValueError(f"output column {into!r} already exists")
    _positive_integer(batch_size, field_name="batch_size")
    if null_policy not in {"propagate", "error"}:
        raise ValueError("null_policy must be 'propagate' or 'error'")

    resolved: list[tuple[str, Column]] = []
    window_contract: tuple[tuple[object, ...], tuple[object, ...]] | None = None
    for channel in representation.channels:
        column = find_column(frame._plan.metadata, channel_mapping[channel])
        match = re.fullmatch(r"(FLOAT|DOUBLE)\[(\d+)\]", column.duckdb_type)
        if match is None or int(match.group(2)) != representation.window:
            raise UnsupportedOperationError(
                f"channel {channel!r} requires FLOAT[{representation.window}] or "
                f"DOUBLE[{representation.window}]; column {column.label!r} is "
                f"{column.duckdb_type}"
            )
        specification = column.series_window
        if specification is None:
            raise UnsupportedOperationError(
                f"channel {channel!r} column {column.label!r} has no verified "
                "series-window metadata"
            )
        if (
            specification.window != representation.window
            or specification.sampling != representation.sampling
            or (
                representation.sampling == "fixed_grid"
                and specification.step != representation.step
            )
        ):
            raise UnsupportedOperationError(
                f"channel {channel!r} window metadata is incompatible with the "
                "requested representation"
            )
        current_contract = (specification.order_by, specification.partition_by)
        if window_contract is None:
            window_contract = current_contract
        elif current_contract != window_contract:
            raise UnsupportedOperationError(
                "All embed_series() channels must use the same row, order, and "
                "partition window contract"
            )
        resolved.append((channel, column))

    output = Column(
        ColumnId.create(),
        into,
        f"FLOAT[{representation.dimension}]",
        nullable=(
            Nullability.NULLABLE
            if null_policy == "propagate" or representation.zero_scale == "null"
            else Nullability.NON_NULL
        ),
        series=SeriesColumnSpec(representation),
    )
    metadata = after_embedding(frame._plan.metadata, output, operation="embed_series")
    return DataFrame(
        frame._session,
        SeriesRepresentationPlan(
            frame._plan,
            tuple((channel, column.id) for channel, column in resolved),
            output,
            representation,
            batch_size,
            null_policy,
            metadata,
        ),
    )
