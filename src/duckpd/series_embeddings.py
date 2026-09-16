"""Immutable contracts for time-series representations and typed queries."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from math import fsum, isfinite, sqrt
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

import pyarrow as pa

from duckpd._temporal import (
    cadence_nanoseconds,
    fixed_duration_ns,
    generate_series_time_features,
    resolve_series_timestamp,
    resolve_timesfm_frequency,
)
from duckpd.embeddings import (
    EmbeddingModelSpec,
    PreparedModelInfo,
    SeriesEmbeddingInputSpec,
)

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


@runtime_checkable
class SeriesEmbeddingProvider(Protocol):
    """Prepared batch provider for an ordered-window embedding model."""

    @property
    def specification(self) -> EmbeddingModelSpec: ...

    @property
    def thread_safe(self) -> bool: ...

    def prepare(self) -> PreparedModelInfo: ...

    def embed_windows(self, batch: pa.RecordBatch) -> pa.Array[Any]: ...


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
    encoder: EmbeddingModelSpec | None = None

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
            if not isinstance(self.encoder.input, SeriesEmbeddingInputSpec):
                raise ValueError("encoder input must be a series embedding input contract")
            if self.encoder.input.length != self.window:
                raise ValueError("encoder input length must equal representation window")
            if self.encoder.input.channels != self.channels:
                raise ValueError("encoder input channels must equal representation channels")
            if self.encoder.normalize != self.unit_norm:
                raise ValueError("encoder normalize must equal representation unit_norm")
            cadence = (
                self.encoder.input.frequency.cadence
                if self.encoder.input.frequency is not None
                else self.encoder.input.temporal.cadence
                if self.encoder.input.temporal is not None
                else None
            )
            if cadence is not None and self.sampling == "fixed_grid":
                if cadence.mode != "elapsed":
                    raise ValueError(
                        "fixed_grid representations require an elapsed series input cadence"
                    )
                assert self.step is not None
                if cadence_nanoseconds(cadence) != fixed_duration_ns(
                    self.step,
                    parameter="step",
                ):
                    raise ValueError(
                        "fixed_grid representation step must equal the series input cadence"
                    )

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
            data["encoder"] = EmbeddingModelSpec.from_dict(encoder_data)
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


def _series_schema_error(  # pyright: ignore[reportUnusedFunction]
    schema: object,
    columns: Sequence[tuple[str, SeriesRepresentationSpec]],
) -> str | None:
    """Validate catalog-bound physical series vectors without reading row values."""
    import pyarrow as pa

    arrow_schema = cast("Any", schema)
    for label, representation in columns:
        index = arrow_schema.get_field_index(label)
        if index < 0:
            return f"Catalog series column {label!r} is absent from feature data"
        dtype = arrow_schema.field(index).type
        if (
            not pa.types.is_fixed_size_list(dtype)
            or dtype.list_size != representation.dimension
            or not pa.types.is_float32(dtype.value_type)
        ):
            return (
                f"Catalog series column {label!r} requires fixed-size "
                f"float32[{representation.dimension}] feature data; found {dtype}"
            )
    return None


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


def _query_values_snapshot(
    values: object,
    *,
    owner: str,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{owner} must be a mapping of channel names to numeric sequences")
    raw = dict(cast("Mapping[object, object]", values))
    if any(type(channel) is not str or not channel for channel in raw):
        raise TypeError(f"{owner} keys must be non-empty channel names")
    snapshot: list[tuple[str, tuple[float, ...]]] = []
    for raw_channel, raw_values in raw.items():
        channel = cast("str", raw_channel)
        if isinstance(raw_values, (str, bytes, bytearray)):
            raise TypeError(f"{owner} channel {channel!r} must be a numeric sequence")
        try:
            items = tuple(cast("Sequence[object]", raw_values))
        except TypeError:
            raise TypeError(f"{owner} channel {channel!r} must be a numeric sequence") from None
        converted: list[float] = []
        for value in items:
            if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
                raise TypeError(f"{owner} channel {channel!r} must contain only numeric scalars")
            number = float(value)
            if not isfinite(number):
                raise ValueError(f"{owner} channel {channel!r} must contain only finite values")
            converted.append(number)
        snapshot.append((channel, tuple(converted)))
    return tuple(snapshot)


def _static_values_snapshot(
    values: object,
) -> tuple[tuple[str, float | int], ...]:
    if values is None:
        return ()
    if not isinstance(values, Mapping):
        raise TypeError("static must be a mapping of names to numeric scalars")
    result: list[tuple[str, float | int]] = []
    for raw_name, raw_value in cast("Mapping[object, object]", values).items():
        if type(raw_name) is not str or not raw_name:
            raise TypeError("static keys must be non-empty names")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, Decimal)):
            raise TypeError(f"static value {raw_name!r} must be a numeric scalar")
        number = float(raw_value)
        if not isfinite(number):
            raise ValueError(f"static value {raw_name!r} must be finite")
        result.append((raw_name, raw_value if type(raw_value) is int else number))
    return tuple(result)


@dataclass(frozen=True)
class SeriesQueryInput:
    """Immutable rich raw query including temporal and static coordinates."""

    values: tuple[tuple[str, tuple[float, ...]], ...]
    time: datetime | None = None
    series_start: datetime | None = None
    static: tuple[tuple[str, float | int], ...] = ()

    def __post_init__(self) -> None:
        if type(self.values) is not tuple or not self.values:
            raise ValueError("values must be a non-empty ordered tuple")
        if any(type(name) is not str or not name for name, _ in self.values):
            raise TypeError("values keys must be non-empty channel names")
        if len({name for name, _ in self.values}) != len(self.values):
            raise ValueError("values channel names must be unique")
        for name, values in self.values:
            raw_values = cast("tuple[object, ...]", values)
            if type(values) is not tuple or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                for value in raw_values
            ):
                raise ValueError(f"values channel {name!r} must contain finite floats")
        raw_time = cast("object", self.time)
        raw_start = cast("object", self.series_start)
        if raw_time is not None and not isinstance(raw_time, datetime):
            raise TypeError("time must be a datetime or None")
        if raw_start is not None and not isinstance(raw_start, datetime):
            raise TypeError("series_start must be a datetime or None")
        if type(self.static) is not tuple:
            raise TypeError("static must be an ordered tuple")
        if len({name for name, _ in self.static}) != len(self.static):
            raise ValueError("static names must be unique")
        for name, value in cast("tuple[tuple[object, object], ...]", self.static):
            if (
                type(name) is not str
                or not name
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError("static must contain unique names and finite numeric values")


def series_query(
    values: Mapping[str, Sequence[float]],
    *,
    time: datetime | None = None,
    series_start: datetime | None = None,
    static: Mapping[str, float | int] | None = None,
) -> SeriesQueryInput:
    """Validate and snapshot a rich raw series query."""
    channel_values = _query_values_snapshot(values, owner="values")
    if not channel_values:
        raise ValueError("values must contain at least one channel")
    raw_time = cast("object", time)
    raw_start = cast("object", series_start)
    if raw_time is not None and not isinstance(raw_time, datetime):
        raise TypeError("time must be a datetime or None")
    if raw_start is not None and not isinstance(raw_start, datetime):
        raise TypeError("series_start must be a datetime or None")
    return SeriesQueryInput(
        channel_values,
        time,
        series_start,
        _static_values_snapshot(static),
    )


@dataclass(frozen=True)
class SeriesQuerySnapshot:
    """Internal representation-ordered query coordinates."""

    channels: tuple[tuple[str, tuple[float, ...]], ...]
    time: datetime | None = None
    series_start: datetime | None = None
    static: tuple[tuple[str, float | int], ...] = ()


def snapshot_series_query(
    query: object,
    representation: SeriesRepresentationSpec,
) -> SeriesQuerySnapshot:
    """Validate and freeze raw channel observations without representing them."""
    if isinstance(query, SeriesQueryInput):
        query_values = query.values
        raw_time = query.time
        raw_series_start = query.series_start
        raw_static = query.static
    else:
        query_values = _query_values_snapshot(query, owner="query")
        raw_time = None
        raw_series_start = None
        raw_static = ()

    raw_query = dict(query_values)
    expected = set(representation.channels)
    actual = set(raw_query)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "query must exactly match representation.channels; "
            f"missing={missing}, unexpected={unexpected}"
        )
    channels: list[tuple[str, tuple[float, ...]]] = []
    for channel in representation.channels:
        values = raw_query[channel]
        if len(values) != representation.window:
            raise ValueError(
                f"query channel {channel!r} length {len(values)} does not match "
                f"representation window {representation.window}"
            )
        channels.append((channel, values))

    input_spec = (
        cast("SeriesEmbeddingInputSpec", representation.encoder.input)
        if representation.encoder is not None
        else None
    )
    temporal = input_spec.temporal if input_spec is not None else None
    static_specs = input_spec.static if input_spec is not None else ()
    if temporal is None:
        if raw_time is not None:
            raise ValueError("time is valid only for a representation with temporal input")
        if raw_series_start is not None:
            raise ValueError("series_start is valid only when age_log10 is declared")
        resolved_time = None
        resolved_start = None
    else:
        if raw_time is None:
            raise ValueError("this representation requires series_query(..., time=<datetime>)")
        resolved_time = resolve_series_timestamp(raw_time, temporal.timezone, field="time")
        requires_start = "age_log10" in temporal.features
        if requires_start and raw_series_start is None:
            raise ValueError("this representation requires series_start for age_log10")
        if not requires_start and raw_series_start is not None:
            raise ValueError("series_start is valid only when age_log10 is declared")
        resolved_start = (
            resolve_series_timestamp(
                raw_series_start,
                temporal.timezone,
                field="series_start",
            )
            if raw_series_start is not None
            else None
        )

    static_values = dict(raw_static)
    expected_static = {item.name for item in static_specs}
    actual_static = set(static_values)
    if actual_static != expected_static:
        if not expected_static and actual_static:
            raise ValueError("static values are valid only when declared by the representation")
        raise ValueError(
            "query static values must exactly match the representation; "
            f"missing={sorted(expected_static - actual_static)}, "
            f"unexpected={sorted(actual_static - expected_static)}"
        )
    ordered_static: list[tuple[str, float | int]] = []
    for specification in static_specs:
        value = static_values[specification.name]
        if specification.kind == "categorical":
            if type(value) is not int:
                raise TypeError(
                    f"static categorical value {specification.name!r} must be an integer"
                )
            assert specification.cardinality is not None
            if value < 0 or value >= specification.cardinality:
                raise ValueError(
                    f"static categorical value {specification.name!r} must be in "
                    f"[0, {specification.cardinality})"
                )
            ordered_static.append((specification.name, value))
        else:
            number = float(value)
            if not isfinite(number):
                raise ValueError(f"static real value {specification.name!r} must be finite")
            ordered_static.append((specification.name, number))
    return SeriesQuerySnapshot(
        tuple(channels),
        resolved_time,
        resolved_start,
        tuple(ordered_static),
    )


def _normalize_series_snapshot(
    query: SeriesQuerySnapshot,
    representation: SeriesRepresentationSpec,
    *,
    zero_scale_as_null: bool,
) -> SeriesQuerySnapshot | None:
    """Apply outer channel normalization identically for corpus and query rows."""
    normalized: list[tuple[str, tuple[float, ...]]] = []
    for channel, values in query.channels:
        if representation.normalization == "none":
            output = values
        else:
            mean = fsum(values) / len(values)
            centered = tuple(value - mean for value in values)
            if representation.normalization == "center":
                output = centered
            else:
                scale = sqrt(fsum(value * value for value in centered) / len(centered))
                if scale == 0.0:
                    if zero_scale_as_null and representation.zero_scale == "null":
                        return None
                    raise ValueError(
                        "series query representation has zero scale and cannot be encoded"
                    )
                output = tuple(value / scale for value in centered)
        normalized.append((channel, tuple(output)))
    return SeriesQuerySnapshot(
        tuple(normalized),
        query.time,
        query.series_start,
        query.static,
    )


def _finite_float32(value: float) -> float:
    try:
        converted = struct.unpack("!f", struct.pack("!f", value))[0]
    except OverflowError:
        raise ValueError("series representation produced a non-finite value") from None
    if not isfinite(converted):
        raise ValueError("series representation produced a non-finite value")
    return converted


def _fixed_list_array(
    values: Sequence[float | int],
    *,
    value_type: pa.DataType,
    width: int,
) -> pa.Array[Any]:
    child = pa.array(values, type=value_type)
    maker = cast("Any", pa.FixedSizeListArray)
    return cast("pa.Array[Any]", maker.from_arrays(child, width))


def make_series_provider_batch(
    rows: Sequence[SeriesQuerySnapshot],
    representation: SeriesRepresentationSpec,
) -> pa.RecordBatch:
    """Build the canonical complete-row, fixed-width Arrow provider boundary."""
    encoder = representation.encoder
    if encoder is None:
        raise ValueError("A learned representation requires an encoder")
    arrays: list[pa.Array[Any]] = []
    fields: list[Any] = []
    by_row = [dict(row.channels) for row in rows]
    input_spec = cast("SeriesEmbeddingInputSpec", encoder.input)
    for channel, role in zip(input_spec.channels, input_spec.roles, strict=True):
        flattened = [_finite_float32(value) for row in by_row for value in row[channel]]
        arrays.append(
            _fixed_list_array(flattened, value_type=pa.float32(), width=input_spec.length)
        )
        fields.append(
            pa.field(
                channel,
                pa.list_(pa.float32(), input_spec.length),
                nullable=False,
                metadata={b"duckpd.channel_role": role.encode()},
            )
        )

    metadata: dict[bytes | str, bytes | str] = {
        b"duckpd.model_fingerprint": encoder.fingerprint.encode(),
        b"duckpd.representation_fingerprint": representation.fingerprint.encode(),
        b"duckpd.mask_semantics": b"complete_rows_only-v1",
    }
    if input_spec.provider_abi is not None:
        metadata[b"duckpd.provider_abi"] = input_spec.provider_abi.encode()
    if input_spec.frequency is not None:
        metadata[b"duckpd.timesfm_frequency"] = str(
            resolve_timesfm_frequency(input_spec.frequency)
        ).encode()
    if input_spec.temporal is not None:
        feature_width = input_spec.temporal.width
        temporal_values: list[float] = []
        for row in rows:
            if row.time is None:
                raise ValueError("temporal provider rows require a time coordinate")
            features = generate_series_time_features(
                input_spec.temporal,
                anchor=row.time,
                series_start=row.series_start,
                length=input_spec.length,
            )
            temporal_values.extend(
                _finite_float32(value) for feature_row in features for value in feature_row
            )
        width = input_spec.length * feature_width
        arrays.append(_fixed_list_array(temporal_values, value_type=pa.float32(), width=width))
        fields.append(
            pa.field(
                "__duckpd_past_time_features",
                pa.list_(pa.float32(), width),
                nullable=False,
            )
        )
        metadata[b"duckpd.time_feature_shape"] = f"{input_spec.length},{feature_width}".encode()

    static_by_row = [dict(row.static) for row in rows]
    real_specs = tuple(item for item in input_spec.static if item.kind == "real")
    categorical_specs = tuple(item for item in input_spec.static if item.kind == "categorical")
    if real_specs:
        real_values = [
            _finite_float32(float(row[specification.name]))
            for row in static_by_row
            for specification in real_specs
        ]
        arrays.append(
            _fixed_list_array(real_values, value_type=pa.float32(), width=len(real_specs))
        )
        fields.append(
            pa.field(
                "__duckpd_static_real",
                pa.list_(pa.float32(), len(real_specs)),
                nullable=False,
            )
        )
    if categorical_specs:
        categorical_values = [
            int(row[specification.name])
            for row in static_by_row
            for specification in categorical_specs
        ]
        arrays.append(
            _fixed_list_array(
                categorical_values,
                value_type=pa.int64(),
                width=len(categorical_specs),
            )
        )
        fields.append(
            pa.field(
                "__duckpd_static_categorical",
                pa.list_(pa.int64(), len(categorical_specs)),
                nullable=False,
            )
        )
    schema = pa.schema(fields, metadata=metadata)
    record_batch = cast("Any", pa.RecordBatch)
    return cast("pa.RecordBatch", record_batch.from_arrays(arrays, schema=schema))


def _validate_series_embedding_array(
    output: object,
    model: EmbeddingModelSpec,
    *,
    expected_rows: int,
) -> pa.Array[Any]:
    """Reject provider output that does not exactly match its declared space."""
    if not isinstance(output, pa.FixedSizeListArray):
        raise TypeError("Series embedding provider must return a FixedSizeListArray")
    array = cast("Any", output)
    if len(array) != expected_rows:
        raise ValueError(
            f"Series embedding provider returned {len(array)} rows; expected {expected_rows}"
        )
    if array.type.list_size != model.dimension or array.type.value_type != pa.float32():
        raise ValueError(
            "Series embedding provider output must have type "
            f"FixedSizeList<float32>[{model.dimension}]"
        )
    if array.null_count or array.values.null_count:
        raise ValueError("Series embedding provider output must not contain nulls")
    rows = cast("list[list[float]]", array.to_pylist())
    if any(not isfinite(float(value)) for row in rows for value in row):
        raise ValueError("Series embedding provider output must contain only finite values")
    return cast("pa.Array[Any]", output)


def _finalize_series_embeddings(  # pyright: ignore[reportUnusedFunction]
    output: object,
    representation: SeriesRepresentationSpec,
    *,
    expected_rows: int,
    zero_scale_as_null: bool,
) -> tuple[tuple[float, ...] | None, ...]:
    encoder = representation.encoder
    if encoder is None:
        raise ValueError("A learned representation requires an encoder")
    validated = _validate_series_embedding_array(
        output,
        encoder,
        expected_rows=expected_rows,
    )
    rows: list[tuple[float, ...] | None] = []
    raw_rows = cast("list[list[float]]", validated.to_pylist())
    for raw_row in raw_rows:
        values = tuple(float(value) for value in raw_row)
        if representation.unit_norm:
            norm = sqrt(fsum(value * value for value in values))
            if norm == 0.0:
                if zero_scale_as_null and representation.zero_scale == "null":
                    rows.append(None)
                    continue
                raise ValueError("series query representation has zero norm and cannot be encoded")
            values = tuple(value / norm for value in values)
        rows.append(tuple(_finite_float32(value) for value in values))
    return tuple(rows)


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
    normalized = _normalize_series_snapshot(
        query,
        representation,
        zero_scale_as_null=False,
    )
    assert normalized is not None
    flattened = [value for _, values in normalized.channels for value in values]
    if representation.unit_norm:
        norm = sqrt(fsum(value * value for value in flattened))
        if norm == 0.0:
            raise ValueError("series query representation has zero norm and cannot be encoded")
        flattened = [value / norm for value in flattened]
    encoded = tuple(_finite_float32(value) for value in flattened)
    return EmbeddedSeriesQuery(encoded, representation.fingerprint)


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
    encoder: EmbeddingModelSpec | None = None,
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
    time: str | None = None,
    series_start: str | None = None,
    static_columns: Mapping[str, str] | None = None,
) -> DataFrame:
    """Append one native or prepared learned series representation lazily."""
    from duckpd._logical import (
        Column,
        ColumnId,
        Nullability,
        SeriesRepresentationPlan,
    )
    from duckpd._metadata import after_embedding, find_column
    from duckpd.errors import UnsupportedOperationError
    from duckpd.frame import DataFrame

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
    input_spec = (
        cast("SeriesEmbeddingInputSpec", representation.encoder.input)
        if representation.encoder is not None
        else None
    )
    temporal = input_spec.temporal if input_spec is not None else None
    if (time is None) != (temporal is None):
        if temporal is None:
            raise ValueError("time is valid only when temporal input is declared")
        raise ValueError("time is required when temporal input is declared")
    age_required = temporal is not None and "age_log10" in temporal.features
    if (series_start is None) != (not age_required):
        if age_required:
            raise ValueError("series_start is required when age_log10 is declared")
        raise ValueError("series_start is valid only when age_log10 is declared")
    static_specs = input_spec.static if input_spec is not None else ()
    raw_static_columns = cast("object", static_columns)
    if raw_static_columns is not None and not isinstance(raw_static_columns, Mapping):
        raise TypeError("static_columns must be a mapping or None")
    static_mapping = dict(static_columns or {})
    if any(type(name) is not str or not name for name in static_mapping):
        raise TypeError("static_columns keys must be non-empty names")
    if any(type(label) is not str or not label for label in static_mapping.values()):
        raise TypeError("static_columns values must be non-empty column labels")
    expected_static = {item.name for item in static_specs}
    actual_static = set(static_mapping)
    if actual_static != expected_static:
        raise ValueError(
            "static_columns must exactly match the representation; "
            f"missing={sorted(expected_static - actual_static)}, "
            f"unexpected={sorted(actual_static - expected_static)}"
        )

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
    resolved_time = find_column(frame._plan.metadata, time) if time is not None else None
    if resolved_time is not None and resolved_time.duckdb_type.upper() not in {
        "TIMESTAMP",
        "TIMESTAMPTZ",
        "TIMESTAMP WITH TIME ZONE",
    }:
        raise UnsupportedOperationError(
            f"time column {resolved_time.label!r} must be TIMESTAMP or TIMESTAMPTZ; "
            f"found {resolved_time.duckdb_type}"
        )
    resolved_start = (
        find_column(frame._plan.metadata, series_start) if series_start is not None else None
    )
    if resolved_start is not None and resolved_start.duckdb_type.upper() not in {
        "TIMESTAMP",
        "TIMESTAMPTZ",
        "TIMESTAMP WITH TIME ZONE",
    }:
        raise UnsupportedOperationError(
            f"series_start column {resolved_start.label!r} must be TIMESTAMP or TIMESTAMPTZ; "
            f"found {resolved_start.duckdb_type}"
        )
    resolved_static: list[tuple[str, Column]] = []
    integral_types = {
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
    }
    from duckpd._reductions import is_numeric_type

    for specification in static_specs:
        column = find_column(frame._plan.metadata, static_mapping[specification.name])
        dtype = column.duckdb_type.upper()
        if specification.kind == "categorical" and dtype not in integral_types:
            raise UnsupportedOperationError(
                f"static categorical column {column.label!r} must be integral; found "
                f"{column.duckdb_type}"
            )
        if specification.kind == "real" and (dtype == "BOOLEAN" or not is_numeric_type(dtype)):
            raise UnsupportedOperationError(
                f"static real column {column.label!r} must be numeric; found {column.duckdb_type}"
            )
        resolved_static.append((specification.name, column))

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
            input=frame._plan,
            channels=tuple((channel, column.id) for channel, column in resolved),
            time=resolved_time.id if resolved_time is not None else None,
            series_start=resolved_start.id if resolved_start is not None else None,
            static_columns=tuple((name, column.id) for name, column in resolved_static),
            output_column=output,
            representation=representation,
            batch_size=batch_size,
            null_policy=null_policy,
            metadata=metadata,
        ),
    )
