"""Native FeatureStore API for DuckPD."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from duckpd._feature_catalog import (
    normalize_filters,
    parse_availability_delay,
    parse_timestamp,
    resolve_features,
    validate_catalog,
)
from duckpd._feature_sources import (
    get_dataset_path_template,
    resolve_partition_paths,
)
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

        if self._is_remote:
            if not self._source_raw.startswith("hf://"):
                raise ValueError("Remote feature store currently supports hf:// URIs")
            if not self._cache_raw:
                msg = "A local cache directory is required when using a remote feature store"
                raise ValueError(msg)
            self._source_path = None
            self._cache_path = Path(self._cache_raw).expanduser().resolve()
            self._cache_path.mkdir(parents=True, exist_ok=True)
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
        # Register filesystem with session DuckDB connection if not already registered
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
            # Cache local copy of catalog.json
            if self._cache_path is not None:
                (self._cache_path / "catalog.json").write_text(catalog_text, encoding="utf-8")
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
        """Return a registered catalog reference table as a lazy DataFrame."""
        if not isinstance(name, str) or not name:
            raise ValueError("table name must be a non-empty string")
        if name not in self._dataset_entries:
            raise ValueError(f"Unknown catalog dataset: {name!r}")
        entry = self._dataset_entries[name]
        if entry["kind"] != "table":
            raise ValueError(f"Dataset {name!r} is a {entry['kind']}, not a table")

        root_ctx = self._source_path if self._source_path else self._source_raw
        path_template = get_dataset_path_template(root_ctx, entry, fs=self._filesystem)
        if "{year}" in path_template:
            raise ValueError(f"Table dataset {name!r} cannot have year partition template")

        if self._source_path is not None:
            table_path = self._source_path / path_template
            if not table_path.is_file():
                raise FileNotFoundError(f"Table file not found for {name!r}: {table_path}")
            return self._session.read_parquet(str(table_path.resolve()))
        else:
            # Remote source: ensure cached locally
            assert self._cache_path is not None
            from duckpd._feature_sources import ensure_cached_table

            cached_file = ensure_cached_table(
                self._source_raw,
                self._cache_path,
                path_template,
                self._filesystem,
            )
            return self._session.read_parquet(str(cached_file.resolve()))

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

    def _resolve_paths_for_dataset(
        self,
        dataset: str,
        start: datetime,
        end: datetime,
        needed_features: list[str],
    ) -> list[str]:
        """Resolve local paths or fetch and project remote partitions into cache."""
        entry = self._dataset_entries[dataset]
        if self._source_path is not None:
            return resolve_partition_paths(self._source_path, entry, start, end)

        # Remote source: ensure cached
        assert self._cache_path is not None
        from duckpd._feature_sources import (
            ensure_cached_partition,
            get_dataset_path_template,
            partition_years_for_interval,
        )

        path_template = get_dataset_path_template(self._cache_path, entry, fs=self._filesystem)
        key_cols = [entry["time_column"], *entry["series_keys"]]
        all_needed = list(dict.fromkeys(key_cols + needed_features))

        paths: list[str] = []
        if "{year}" in path_template:
            years = partition_years_for_interval(entry, start, end)
            for year in years:
                rel_path = path_template.format(year=year)
                cached_file = ensure_cached_partition(
                    self._source_raw,
                    self._cache_path,
                    rel_path,
                    all_needed,
                    self._session._connection,
                )
                paths.append(str(cached_file.resolve()))
        else:
            cached_file = ensure_cached_partition(
                self._source_raw,
                self._cache_path,
                path_template,
                all_needed,
                self._session._connection,
            )
            paths.append(str(cached_file.resolve()))
        return paths

    def _build_exact_alignment(
        self,
        grouped: dict[str, list[tuple[str, str]]],
        resolved: list[tuple[str, str, str, str]],
        start: datetime,
        end: datetime,
        filters: dict[str, list[str]] | None,
    ) -> DataFrame:
        # Validate compatibility of timeseries keys across all datasets
        key_specs = {
            (
                self._dataset_entries[ds]["time_column"],
                tuple(self._dataset_entries[ds]["series_keys"]),
            )
            for ds in grouped
        }
        if len(key_specs) != 1:
            msg = (
                "Exact alignment requires compatible time_column and series_keys "
                "across all selected datasets"
            )
            raise ValueError(msg)
        time_column, series_keys = next(iter(key_specs))
        all_keys = [time_column, *series_keys]

        # Scan each dataset family
        family_frames: list[DataFrame] = []
        for dataset, feats in grouped.items():
            feat_names = [f_name for _, f_name in feats]
            paths = self._resolve_paths_for_dataset(dataset, start, end, feat_names)
            if not paths:
                raise FileNotFoundError(
                    f"No partition files found for dataset {dataset!r} between {start} and {end}"
                )
            df: Any = self._session.read_parquet(paths)
            # Filter time range [start, end)
            t_col: Any = df[time_column]
            df = df[(t_col >= start) & (t_col < end)]

            # Apply filters
            if filters is not None:
                for f_col, f_vals in filters.items():
                    if f_col in all_keys:
                        # Construct OR condition for matching values
                        match_expr: Any = None
                        for val in f_vals:
                            cond: Any = df[f_col] == val
                            match_expr = cond if match_expr is None else (match_expr | cond)
                        if match_expr is not None:
                            df = df[match_expr]

            # Project keys + requested features (with renaming to out_name)
            needed_cols = list(all_keys)
            rename_map: dict[str, str] = {}
            for out_name, feat_name in feats:
                if feat_name not in needed_cols:
                    needed_cols.append(feat_name)
                rename_map[feat_name] = out_name

            df = df[needed_cols]
            if rename_map:
                df = df.rename(columns=rename_map)
            family_frames.append(df)

        # Merge family frames on all_keys with inner join
        result_df = family_frames[0]
        for next_df in family_frames[1:]:
            result_df = result_df.merge(next_df, on=all_keys, how="inner")

        # Project final column ordering: keys + output features in resolution order
        final_cols = list(all_keys)
        for out_name, _, _, _ in resolved:
            if out_name not in final_cols:
                final_cols.append(out_name)

        return result_df[final_cols]

    def _build_pit_alignment(
        self,
        spine_dataset: str,
        grouped: dict[str, list[tuple[str, str]]],
        resolved: list[tuple[str, str, str, str]],
        start: datetime,
        end: datetime,
        filters: dict[str, list[str]] | None,
    ) -> DataFrame:
        # Check lookahead safety and availability delays
        delays: dict[str, timedelta] = {}
        for _, ref, _, _ in resolved:
            entry = self._feature_entries[ref]
            if entry.get("lookahead_safe") is not True:
                raise ValueError(f"Feature {ref} is not marked lookahead_safe")
            val = entry.get("availability_delay")
            if val is None:
                raise ValueError(f"Feature {ref} must define availability_delay")
            delays[ref] = parse_availability_delay(val, ref)

        spine_entry = self._dataset_entries[spine_dataset]
        spine_time_col = spine_entry["time_column"]
        spine_series_keys = list(spine_entry["series_keys"])

        for ds in grouped:
            entry = self._dataset_entries[ds]
            if (
                entry["time_column"] != spine_time_col
                or list(entry["series_keys"]) != spine_series_keys
            ):
                msg = (
                    "Point-in-time alignment requires compatible time_column and series_keys: "
                    f"{ds} != {spine_dataset}"
                )
                raise ValueError(msg)

        # Build native SQL query executed through session.sql()
        # This provides vectorized ASOF LEFT JOIN execution
        from duckpd._quoting import quote_identifier, quote_literal

        ctes: list[str] = []
        quoted_keys = [quote_identifier(k) for k in [spine_time_col, *spine_series_keys]]

        # Scan spine
        spine_feats = [
            f_name for ds, feats in grouped.items() if ds == spine_dataset for _, f_name in feats
        ]
        spine_paths = self._resolve_paths_for_dataset(spine_dataset, start, end, spine_feats)
        if not spine_paths:
            raise FileNotFoundError(f"No partitions found for spine {spine_dataset!r}")
        spine_path_sql = ", ".join(quote_literal(p) for p in spine_paths)

        spine_filters = (
            f"{quote_identifier(spine_time_col)} >= TIMESTAMPTZ {quote_literal(start.isoformat())} "
            f"AND {quote_identifier(spine_time_col)} < TIMESTAMPTZ {quote_literal(end.isoformat())}"
        )
        if filters:
            for f_col, f_vals in filters.items():
                if f_col in spine_series_keys:
                    val_list = ", ".join(quote_literal(v) for v in f_vals)
                    spine_filters += f" AND {quote_identifier(f_col)} IN ({val_list})"

        spine_cte = (
            f"spine AS (SELECT {', '.join(quoted_keys)} "
            f"FROM read_parquet([{spine_path_sql}]) WHERE {spine_filters})"
        )
        ctes.append(spine_cte)

        # Build each family CTE
        # To ensure predecessor rows are available for ASOF, scan from earliest necessary history
        max_delay = max(delays.values(), default=timedelta())
        history_start = start - max_delay - timedelta(days=366)

        dataset_aliases: dict[str, str] = {}
        for idx, (ds, feats) in enumerate(grouped.items()):
            ds_alias = f"family_{idx}"
            dataset_aliases[ds] = ds_alias
            feat_names = list({f_name for _, f_name in feats})
            ds_paths = self._resolve_paths_for_dataset(ds, history_start, end, feat_names)
            if not ds_paths:
                # If no history paths exist, fall back to start-end paths
                ds_paths = self._resolve_paths_for_dataset(ds, start, end, feat_names)
            if not ds_paths:
                raise FileNotFoundError(f"No partitions found for dataset {ds!r}")
            path_sql = ", ".join(quote_literal(p) for p in ds_paths)

            feat_names = list({f_name for _, f_name in feats})
            col_list = ", ".join(
                quote_identifier(c) for c in [spine_time_col, *spine_series_keys, *feat_names]
            )
            time_filter = (
                f"{quote_identifier(spine_time_col)} < TIMESTAMPTZ {quote_literal(end.isoformat())}"
            )
            family_cte = (
                f"{ds_alias} AS (SELECT {col_list} "
                f"FROM read_parquet([{path_sql}]) WHERE {time_filter})"
            )
            ctes.append(family_cte)

        # Group joins by (dataset, delay_microseconds)
        alignment_groups: dict[tuple[str, int], str] = {}
        for _, ref, ds, _ in resolved:
            d_us = delays[ref] // timedelta(microseconds=1)
            alignment_groups.setdefault((ds, d_us), f"aligned_{len(alignment_groups)}")

        select_cols = [f"spine.{k}" for k in quoted_keys]
        for out_name, ref, ds, feat_name in resolved:
            d_us = delays[ref] // timedelta(microseconds=1)
            group_alias = alignment_groups[(ds, d_us)]
            select_cols.append(
                f"{group_alias}.{quote_identifier(feat_name)} AS {quote_identifier(out_name)}"
            )

        joins: list[str] = []
        for (ds, d_us), g_alias in alignment_groups.items():
            ds_alias = dataset_aliases[ds]
            conds = [
                f"spine.{quote_identifier(k)} = {g_alias}.{quote_identifier(k)}"
                for k in spine_series_keys
            ]
            conds.append(
                f"spine.{quote_identifier(spine_time_col)} >= "
                f"{g_alias}.{quote_identifier(spine_time_col)} + INTERVAL {d_us} MICROSECOND"
            )
            joins.append(f"ASOF LEFT JOIN {ds_alias} AS {g_alias} ON {' AND '.join(conds)}")

        joins_sql = " ".join(joins)
        query = f"WITH {', '.join(ctes)} SELECT {', '.join(select_cols)} FROM spine {joins_sql}"
        return self._session.sql(query)

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

            for ds, feat_names in grouped.items():
                paths = self._resolve_paths_for_dataset(ds, start_dt, end_dt, feat_names)
                partitions_count += len(paths)
                for p in paths:
                    p_file = Path(p)
                    if p_file.is_file():
                        bytes_count += p_file.stat().st_size

        return SyncReport(
            partitions_synced=partitions_count,
            bytes_written=bytes_count,
            tables_synced=tables_count,
        )
