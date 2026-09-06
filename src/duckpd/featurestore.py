"""Native FeatureStore API for DuckPD."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from duckpd._feature_catalog import (
    normalize_filters,
    parse_availability_delay,
    parse_timestamp,
    resolve_features,
    validate_catalog,
)
from duckpd._feature_sources import (
    _rooted_path,
    _validated_relative_path,
    get_dataset_path_template,
)
from duckpd._logical import (
    AsOfJoinPlan,
    Column,
    ColumnId,
    ColumnRef,
    FeatureParquetSource,
    FrameMetadata,
    NamedExpression,
    Nullability,
    ProjectPlan,
    ScanPlan,
    SourceKind,
    SourceProvenance,
)
from duckpd._metadata import after_join, after_projection
from duckpd.series import Series

if TYPE_CHECKING:
    from duckpd.frame import DataFrame
    from duckpd.session import Session


@dataclass(frozen=True)
class SyncReport:
    """Summary of partition pre-warming operations."""

    partitions_synced: int
    bytes_written: int
    tables_synced: int


class FeatureStore:
    """A high-performance feature store backed by cataloged Parquet datasets and DuckPD."""

    def __init__(
        self,
        source: str | Path,
        *,
        session: Session | None = None,
        cache: str | Path | None = None,
        token: str | None = None,
        catalog_path: str | Path | None = None,
        features: Sequence[str] | Mapping[str, str] | None = None,
        start: str | None = None,
        end: str | None = None,
        filters: Mapping[str, Sequence[str]] | None = None,
        alignment: Literal["exact", "point_in_time"] | None = None,
        spine: str | None = None,
    ) -> None:
        from duckpd.io import _get_implicit_session

        self._session = session if session is not None else _get_implicit_session()
        self._source_raw = str(source).rstrip("/")
        self._cache_raw = str(cache) if cache is not None else None
        self._token = token
        self._catalog_path_raw = Path(catalog_path) if catalog_path is not None else None
        self.alignment = self._validate_alignment(alignment)
        self.spine = spine

        self._is_remote = self._source_raw.startswith(("hf://", "http://", "https://"))
        self._filesystem: Any = None
        self._filesystem_key: str | None = None

        if self._is_remote:
            if not self._source_raw.startswith("hf://"):
                raise ValueError("Remote feature store currently supports hf:// URIs")
            if not self._cache_raw:
                msg = "A local cache directory is required when using a remote feature store"
                raise ValueError(msg)
            self._source_path = None
            self._cache_path = Path(self._cache_raw).expanduser().resolve()
            self._init_remote_filesystem()
        else:
            self._source_path = Path(self._source_raw).expanduser().resolve()
            if not self._source_path.is_dir():
                msg = f"Feature store source directory not found: {self._source_path}"
                raise FileNotFoundError(msg)
            self._cache_path = (
                Path(self._cache_raw).expanduser().resolve() if self._cache_raw else None
            )

        # Load catalog
        self._catalog: dict[str, Any] = {}
        self._dataset_entries: dict[str, dict[str, Any]] = {}
        self._feature_entries: dict[str, dict[str, Any]] = {}
        self._load_catalog()

        # Selection state
        self._configured_features: Sequence[str] | Mapping[str, str] | None = None
        self.start: datetime | None = None
        self.end: datetime | None = None
        self.filters: dict[str, tuple[str, ...]] | None = None
        self._feature_selection: tuple[str, ...] = ()

        selection_arguments = (features, start, end)
        if any(v is not None for v in selection_arguments):
            if not all(v is not None for v in selection_arguments):
                raise ValueError("features, start, and end must be provided together")
            assert features is not None and start is not None and end is not None
            self._configure(
                features=features,
                start=start,
                end=end,
                filters=filters,
                alignment=self.alignment,
                spine=self.spine,
            )
        elif filters is not None:
            raise ValueError("filters requires features, start, and end")

    @staticmethod
    def _validate_alignment(
        alignment: Any,
    ) -> Literal["exact", "point_in_time"] | None:
        if alignment not in (None, "exact", "point_in_time"):
            raise ValueError("alignment must be 'exact' or 'point_in_time'")
        return alignment

    def _init_remote_filesystem(self) -> None:
        try:
            from huggingface_hub import HfFileSystem  # pyright: ignore[reportMissingImports]
        except ImportError:
            msg = (
                "huggingface-hub is required for remote Hugging Face feature store access. "
                "Install it with `pip install duckpd[featurestore]` or "
                "`pip install huggingface-hub`."
            )
            raise ImportError(msg) from None

        self._filesystem = HfFileSystem(token=self._token)
        self._filesystem_key = f"featurestore:{uuid4().hex}"
        self._session._registered_sources[self._filesystem_key] = self._filesystem
        from contextlib import suppress

        with suppress(Exception):
            self._session._connection.register_filesystem(self._filesystem)

    def _load_catalog(self) -> None:
        candidates: list[Path] = []
        if self._catalog_path_raw is not None:
            candidates.append(self._catalog_path_raw.expanduser().resolve())
        if self._source_path is not None:
            candidates.append(self._source_path / "catalog.json")
        if self._cache_path is not None:
            candidates.append(self._cache_path / "catalog.json")

        cat_file = next((p for p in candidates if p.is_file()), None)
        if cat_file is not None:
            catalog_text = cat_file.read_text(encoding="utf-8")
        elif self._is_remote and self._filesystem is not None:
            # Read from remote filesystem
            remote_cat_path = f"{self._source_raw.replace('hf://', '').rstrip('/')}/catalog.json"
            try:
                with self._filesystem.open(remote_cat_path, "r") as f:
                    catalog_text = f.read()
            except Exception as err:
                raise FileNotFoundError(f"Remote catalog not found: {remote_cat_path}") from err
        else:
            raise FileNotFoundError(f"Catalog not found in {self._source_raw}")

        catalog_data = json.loads(catalog_text)
        dataset_index, feature_index = validate_catalog(catalog_data)
        self._catalog = catalog_data
        self._dataset_entries = dataset_index
        self._feature_entries = feature_index

    def catalog(self) -> dict[str, Any]:
        """Return a defensive copy of the validated catalog."""
        return copy.deepcopy(self._catalog)

    @property
    def source(self) -> str:
        """Return the source location path or URI."""
        return str(self._source_path) if self._source_path else self._source_raw

    @property
    def session(self) -> Session:
        """Return the owning DuckPD session."""
        return self._session

    def table(self, name: Any) -> DataFrame:
        """Return a lazy scan of a registered catalog reference table."""
        if not isinstance(name, str) or not name:
            raise ValueError("table name must be a non-empty string")
        if name not in self._dataset_entries:
            raise ValueError(f"Unknown catalog dataset: {name!r}")
        entry = self._dataset_entries[name]
        if entry["kind"] != "table":
            raise ValueError(f"Dataset {name!r} is a {entry['kind']}, not a table")

        root_context = self._source_raw if self._is_remote else cast(Path, self._source_path)
        path_template = get_dataset_path_template(
            root_context,
            entry,
            fs=self._filesystem,
        )
        _validated_relative_path(path_template)
        if "{year}" in path_template:
            raise ValueError(f"Table dataset {name!r} cannot have year partition template")

        columns = self._inspect_table_columns(path_template)
        source = FeatureParquetSource(
            source_root=self._source_raw if self._is_remote else str(self._source_path),
            cache_root=str(self._cache_path) if self._is_remote else None,
            path_template=path_template,
            needed_columns=tuple(column.label for column in columns),
            filesystem_key=self._filesystem_key,
            table=True,
        )
        metadata = FrameMetadata(
            columns,
            provenance=SourceProvenance(
                SourceKind.FEATURE_STORE,
                (self._source_raw,),
            ),
        )
        from duckpd.frame import DataFrame

        return DataFrame(self._session, ScanPlan(source, metadata))

    def _inspect_table_columns(self, path_template: str) -> tuple[Column, ...]:
        """Read only Parquet schema metadata; table bytes remain uncached until execution."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._source_path is not None:
            table_path = _rooted_path(self._source_path, path_template)
            if not table_path.is_file():
                raise FileNotFoundError(f"Table file not found: {table_path}")
            schema = pq.ParquetFile(table_path).schema_arrow
        else:
            assert self._filesystem is not None
            assert self._cache_path is not None
            cached_path = _rooted_path(self._cache_path, path_template)
            if cached_path.is_file():
                schema = pq.ParquetFile(cached_path).schema_arrow
            else:
                remote_path = (
                    self._source_raw.replace("hf://", "").rstrip("/")
                    + "/"
                    + _validated_relative_path(path_template).as_posix()
                )
                with self._filesystem.open(remote_path, "rb") as remote_file:
                    schema = pq.ParquetFile(remote_file).schema_arrow

        empty = pa.Table.from_batches([], schema=schema)
        relation = self._session._connection.from_arrow(empty)
        return tuple(
            Column(ColumnId.create(), label, str(dtype))
            for label, dtype in zip(relation.columns, relation.types, strict=True)
        )

    def _configure(
        self,
        features: Sequence[str] | Mapping[str, str],
        start: str,
        end: str,
        filters: Mapping[str, Sequence[str]] | None,
        alignment: Literal["exact", "point_in_time"] | None,
        spine: str | None,
    ) -> None:
        if alignment is None:
            raise TypeError("alignment is required when features are configured")
        if alignment == "exact" and spine is not None:
            raise ValueError("spine is not supported with exact alignment")
        if alignment == "point_in_time":
            if spine is None:
                raise ValueError("spine dataset name is required with point_in_time alignment")
            if spine not in self._dataset_entries:
                raise ValueError(f"Unknown spine dataset: {spine!r}")
            if self._dataset_entries[spine]["kind"] != "timeseries":
                raise ValueError(f"Spine dataset {spine!r} must be a timeseries dataset")

        start_dt = parse_timestamp(start)
        end_dt = parse_timestamp(end)
        if end_dt <= start_dt:
            raise ValueError("end must be later than start")

        norm_filters = normalize_filters(filters)
        resolved = resolve_features(features, self._feature_entries, self._dataset_entries)

        self._configured_features = features
        self.start = start_dt
        self.end = end_dt
        self.filters = (
            None if norm_filters is None else {k: tuple(v) for k, v in norm_filters.items()}
        )
        self._feature_selection = tuple(ref for _, ref, _, _ in resolved)

    def features(
        self,
        features: Sequence[str] | Mapping[str, str] | None = None,
        *,
        start: str | None = None,
        end: str | None = None,
        filters: Mapping[str, Sequence[str]] | None = None,
        alignment: Literal["exact", "point_in_time"] | None = None,
        spine: str | None = None,
        columns: Sequence[str] | None = None,
        order_by: Sequence[str] | None = None,
    ) -> DataFrame:
        """Return the aligned feature selection as a lazy DataFrame."""
        if features is not None or start is not None or end is not None:
            # Sliced or overridden call
            eff_features = features if features is not None else self._configured_features
            eff_start = (
                start if start is not None else (self.start.isoformat() if self.start else None)
            )
            eff_end = end if end is not None else (self.end.isoformat() if self.end else None)
            eff_filters = filters if filters is not None else self.filters
            eff_alignment = alignment if alignment is not None else self.alignment
            eff_spine = spine if spine is not None else self.spine

            if eff_features is None or eff_start is None or eff_end is None:
                raise ValueError("features, start, and end must be configured or passed together")
            eff_align = self._validate_alignment(eff_alignment)
            if eff_align is None:
                raise TypeError("alignment is required")

            start_dt = parse_timestamp(eff_start)
            end_dt = parse_timestamp(eff_end)
            if end_dt <= start_dt:
                raise ValueError("end must be later than start")
            norm_filters = normalize_filters(eff_filters)
            resolved = resolve_features(eff_features, self._feature_entries, self._dataset_entries)
            align_mode = eff_align
            spine_name = eff_spine
        else:
            if not self._feature_selection or self.start is None or self.end is None:
                raise ValueError("FeatureStore has no configured feature selection")
            start_dt = self.start
            end_dt = self.end
            norm_filters = {k: list(v) for k, v in self.filters.items()} if self.filters else None
            resolved = resolve_features(
                self._configured_features or self._feature_selection,
                self._feature_entries,
                self._dataset_entries,
            )
            align_mode = cast(Literal["exact", "point_in_time"], self.alignment)
            spine_name = self.spine

        # Group features by dataset
        grouped: dict[str, list[tuple[str, str]]] = {}
        for out_name, _, dataset, feat_name in resolved:
            grouped.setdefault(dataset, []).append((out_name, feat_name))

        if align_mode == "exact":
            frame = self._build_exact_alignment(
                grouped,
                resolved,
                start_dt,
                end_dt,
                norm_filters,
            )
        else:
            assert spine_name is not None
            frame = self._build_pit_alignment(
                spine_name,
                grouped,
                resolved,
                start_dt,
                end_dt,
                norm_filters,
            )

        if columns is not None:
            if isinstance(columns, (str, bytes)):
                raise TypeError("columns must be a sequence of column names")
            frame = frame[list(columns)]

        if order_by is not None:
            if isinstance(order_by, (str, bytes)):
                raise TypeError("order_by must be a sequence of column names")
            frame = frame.sort_values(list(order_by))

        return frame

    def _timeseries_frame(
        self,
        dataset: str,
        start: datetime,
        end: datetime,
        needed_features: Sequence[str],
    ) -> DataFrame:
        """Build a typed deferred scan for one cataloged timeseries dataset."""
        entry = self._dataset_entries[dataset]
        root_context = self._source_raw if self._is_remote else cast(Path, self._source_path)
        path_template = get_dataset_path_template(
            root_context,
            entry,
            fs=self._filesystem,
        )
        _validated_relative_path(path_template)
        labels = list(
            dict.fromkeys(
                [
                    entry["time_column"],
                    *entry["series_keys"],
                    *needed_features,
                ]
            )
        )
        columns = tuple(Column(ColumnId.create(), label, "UNKNOWN") for label in labels)
        source = FeatureParquetSource(
            source_root=self._source_raw if self._is_remote else str(self._source_path),
            cache_root=str(self._cache_path) if self._is_remote else None,
            path_template=path_template,
            needed_columns=tuple(labels),
            start=start.isoformat(),
            end=end.isoformat(),
            min_time=entry.get("min_time"),
            max_time=entry.get("max_time"),
            filesystem_key=self._filesystem_key,
        )
        metadata = FrameMetadata(
            columns,
            provenance=SourceProvenance(
                SourceKind.FEATURE_STORE,
                (self._source_raw,),
            ),
        )
        from duckpd.frame import DataFrame

        return DataFrame(self._session, ScanPlan(source, metadata))

    @staticmethod
    def _project_feature_outputs(
        frame: DataFrame,
        keys: Sequence[str],
        features: Sequence[tuple[str, str]],
    ) -> DataFrame:
        """Project keys and feature aliases without collapsing repeated physical columns."""
        by_label = {column.label: column for column in frame._plan.metadata.visible_columns}
        projections: list[NamedExpression] = []
        output_columns: list[Column] = []
        for label in keys:
            column = by_label[label]
            projections.append(NamedExpression(column, ColumnRef(column.id)))
            output_columns.append(column)
        for output_name, physical_name in features:
            source_column = by_label[physical_name]
            output_column = Column(
                ColumnId.create(),
                output_name,
                source_column.duckdb_type,
                nullable=source_column.nullable,
                alias_of=source_column.id,
            )
            projections.append(NamedExpression(output_column, ColumnRef(source_column.id)))
            output_columns.append(output_column)

        metadata = after_projection(frame._plan.metadata, tuple(output_columns))
        from duckpd.frame import DataFrame

        return DataFrame(
            frame._session,
            ProjectPlan(frame._plan, tuple(projections), metadata),
        )

    def _build_exact_alignment(
        self,
        grouped: dict[str, list[tuple[str, str]]],
        resolved: list[tuple[str, str, str, str]],
        start: datetime,
        end: datetime,
        filters: dict[str, list[str]] | None,
    ) -> DataFrame:
        key_specs = {
            (
                self._dataset_entries[dataset]["time_column"],
                tuple(self._dataset_entries[dataset]["series_keys"]),
            )
            for dataset in grouped
        }
        if len(key_specs) != 1:
            raise ValueError(
                "Exact alignment requires compatible time_column and series_keys "
                "across all selected datasets"
            )
        time_column, series_keys = next(iter(key_specs))
        all_keys = [time_column, *series_keys]

        family_frames: list[DataFrame] = []
        for dataset, features in grouped.items():
            physical_names = list(dict.fromkeys(name for _, name in features))
            frame = self._timeseries_frame(dataset, start, end, physical_names)
            time_series = cast("Series", frame[time_column])
            frame = frame[(time_series >= start) & (time_series < end)]
            if filters is not None:
                for filter_column, filter_values in filters.items():
                    if filter_column not in all_keys:
                        raise ValueError(f"Unknown filter column: {filter_column!r}")
                    match_expression: Series | None = None
                    for value in filter_values:
                        condition = frame[filter_column] == value
                        match_expression = (
                            condition
                            if match_expression is None
                            else (match_expression | condition)
                        )
                    assert match_expression is not None
                    frame = frame[match_expression]
            family_frames.append(self._project_feature_outputs(frame, all_keys, features))

        result = family_frames[0]
        for next_frame in family_frames[1:]:
            result = result.merge(next_frame, on=all_keys, how="inner")

        return result[
            [
                *all_keys,
                *(output_name for output_name, _, _, _ in resolved),
            ]
        ]

    def _build_pit_alignment(
        self,
        spine_dataset: str,
        grouped: dict[str, list[tuple[str, str]]],
        resolved: list[tuple[str, str, str, str]],
        start: datetime,
        end: datetime,
        filters: dict[str, list[str]] | None,
    ) -> DataFrame:
        delays: dict[str, timedelta] = {}
        for _, reference, _, _ in resolved:
            feature_entry = self._feature_entries[reference]
            if feature_entry.get("lookahead_safe") is not True:
                raise ValueError(f"Feature {reference} is not marked lookahead_safe")
            delays[reference] = parse_availability_delay(
                feature_entry.get("availability_delay"),
                reference,
            )

        spine_entry = self._dataset_entries[spine_dataset]
        time_column = spine_entry["time_column"]
        series_keys = list(spine_entry["series_keys"])
        for dataset in grouped:
            entry = self._dataset_entries[dataset]
            if entry["time_column"] != time_column or list(entry["series_keys"]) != series_keys:
                raise ValueError(
                    "Point-in-time alignment requires compatible time_column and series_keys: "
                    f"{dataset} != {spine_dataset}"
                )

        spine = self._timeseries_frame(spine_dataset, start, end, ())
        spine_time = cast("Series", spine[time_column])
        spine = spine[(spine_time >= start) & (spine_time < end)]
        if filters is not None:
            for filter_column, filter_values in filters.items():
                if filter_column not in series_keys:
                    raise ValueError(f"Unknown filter column: {filter_column!r}")
                match_expression: Series | None = None
                for value in filter_values:
                    condition = spine[filter_column] == value
                    match_expression = (
                        condition if match_expression is None else (match_expression | condition)
                    )
                assert match_expression is not None
                spine = spine[match_expression]

        alignment_groups: dict[
            tuple[str, int],
            list[tuple[str, str]],
        ] = {}
        for output_name, reference, dataset, physical_name in resolved:
            delay_microseconds = delays[reference] // timedelta(microseconds=1)
            alignment_groups.setdefault((dataset, delay_microseconds), []).append(
                (output_name, physical_name)
            )

        result: DataFrame = spine
        for (dataset, delay_microseconds), features in alignment_groups.items():
            entry = self._dataset_entries[dataset]
            root_context = self._source_raw if self._is_remote else cast(Path, self._source_path)
            path_template = get_dataset_path_template(
                root_context,
                entry,
                fs=self._filesystem,
            )
            min_time = entry.get("min_time")
            if "{year}" in path_template and min_time is None:
                raise ValueError(
                    f"Point-in-time dataset {dataset!r} requires min_time "
                    "to resolve complete predecessor history"
                )
            history_start = (
                parse_timestamp(min_time)
                if min_time is not None
                else datetime.min.replace(tzinfo=start.tzinfo)
            )
            physical_names = list(dict.fromkeys(name for _, name in features))
            right = self._timeseries_frame(dataset, history_start, end, physical_names)
            right_time = cast("Series", right[time_column])
            right = right[right_time < end]
            right = self._project_feature_outputs(
                right,
                [time_column, *series_keys],
                features,
            )

            left_columns = {column.label: column for column in result._plan.metadata.columns}
            right_columns = {column.label: column for column in right._plan.metadata.columns}
            payload_columns = tuple(
                Column(
                    right_columns[output_name].id,
                    output_name,
                    right_columns[output_name].duckdb_type,
                    nullable=Nullability.NULLABLE,
                    alias_of=right_columns[output_name].alias_of,
                )
                for output_name, _ in features
            )
            metadata = after_join((*result._plan.metadata.columns, *payload_columns))
            plan = AsOfJoinPlan(
                left=result._plan,
                right=right._plan,
                left_time=left_columns[time_column].id,
                right_time=right_columns[time_column].id,
                left_keys=tuple(left_columns[key].id for key in series_keys),
                right_keys=tuple(right_columns[key].id for key in series_keys),
                delay_microseconds=delay_microseconds,
                metadata=metadata,
            )
            from duckpd.frame import DataFrame

            result = DataFrame(self._session, plan)

        return result[
            [
                time_column,
                *series_keys,
                *(output_name for output_name, _, _, _ in resolved),
            ]
        ]

    def feature_batches(
        self,
        frame: DataFrame | None = None,
        *,
        window: Any,
        start: str | None = None,
        end: str | None = None,
    ) -> Iterator[DataFrame]:
        """Yield consecutive time-windowed DataFrames."""
        if not isinstance(window, timedelta):
            raise TypeError("window must be a datetime.timedelta")
        if window <= timedelta():
            raise ValueError("window must be positive")

        start_dt = parse_timestamp(start) if start is not None else self.start
        end_dt = parse_timestamp(end) if end is not None else self.end
        if start_dt is None or end_dt is None:
            raise ValueError("FeatureStore start and end must be specified or configured")
        if end_dt <= start_dt:
            raise ValueError("end must be later than start")

        cursor = start_dt
        while cursor < end_dt:
            step_end = min(cursor + window, end_dt)
            if frame is not None:
                # Slicing existing frame on its time column
                # Identify time column from catalog
                time_col = next(iter(self._dataset_entries.values()))["time_column"]
                series_t = cast("Series", frame[time_col])
                cond = (series_t >= cursor) & (series_t < step_end)
                yield frame[cond]
            else:
                yield self.features(start=cursor.isoformat(), end=step_end.isoformat())
            cursor = step_end

    def sync(
        self,
        features: Sequence[str] | Mapping[str, str] | None = None,
        *,
        start: str | None = None,
        end: str | None = None,
        tables: Sequence[str] | None = None,
    ) -> SyncReport:
        """Pre-fetch and project remote partitions into local cache without executing analysis."""
        eff_features = features if features is not None else self._configured_features
        eff_start = start if start is not None else (self.start.isoformat() if self.start else None)
        eff_end = end if end is not None else (self.end.isoformat() if self.end else None)

        partitions_count = 0
        bytes_count = 0
        tables_count = 0

        # Sync tables if requested
        if tables is not None:
            for tbl_name in tables:
                if tbl_name not in self._dataset_entries:
                    raise ValueError(f"Unknown catalog dataset: {tbl_name!r}")
                tbl_entry = self._dataset_entries[tbl_name]
                if tbl_entry["kind"] != "table":
                    raise ValueError(f"Dataset {tbl_name!r} is not a table dataset")
                root_ctx = self._source_path if self._source_path else self._source_raw
                path_tmpl = get_dataset_path_template(root_ctx, tbl_entry, fs=self._filesystem)
                if self._source_path is None and self._cache_path is not None:
                    from duckpd._feature_sources import ensure_cached_table

                    cached_f = ensure_cached_table(
                        self._source_raw,
                        self._cache_path,
                        path_tmpl,
                        self._filesystem,
                    )
                    tables_count += 1
                    bytes_count += cached_f.stat().st_size
                elif self._source_path is not None:
                    tables_count += 1

        # Sync timeseries partitions if features specified
        if eff_features is not None:
            if eff_start is None or eff_end is None:
                raise ValueError("start and end must be specified or configured to sync features")
            start_dt = parse_timestamp(eff_start)
            end_dt = parse_timestamp(eff_end)
            if end_dt <= start_dt:
                raise ValueError("end must be later than start")

            resolved = resolve_features(eff_features, self._feature_entries, self._dataset_entries)
            grouped: dict[str, list[str]] = {}
            for _, _, ds, feat_name in resolved:
                if feat_name not in grouped.setdefault(ds, []):
                    grouped[ds].append(feat_name)

            from duckpd._feature_sources import materialize_feature_source

            for dataset, feature_names in grouped.items():
                frame = self._timeseries_frame(dataset, start_dt, end_dt, feature_names)
                source = cast(ScanPlan, frame._plan).source
                if not isinstance(source, FeatureParquetSource):
                    raise AssertionError("Feature sync expected a deferred feature source")
                paths = materialize_feature_source(
                    source,
                    self._session._connection,
                    self._filesystem,
                )
                partitions_count += len(paths)
                for path in paths:
                    path_file = Path(path)
                    if path_file.is_file():
                        bytes_count += path_file.stat().st_size

        return SyncReport(
            partitions_synced=partitions_count,
            bytes_written=bytes_count,
            tables_synced=tables_count,
        )
