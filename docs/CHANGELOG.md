# Changelog

All notable changes to DuckPD will be documented in this file.

## Unreleased

### Added

- Initial `uv`-managed Python package and quality toolchain.
- Lazy pandas, Arrow, Parquet, DuckDB table, and read-only SQL sources.
- Typed scan, filter, project, sort, and limit logical plans.
- Lazy Series arithmetic, comparisons, boolean expressions, filtering,
  assignment, projection, sorting, and limiting.
- Explicit pandas and Arrow materialization, bounded previews, plan inspection,
  and direct Parquet output.
- Session resource configuration, source retention, execution counting, and
  clear closed-session errors.
- Immutable schema/index/order metadata with hidden-column preservation across
  projections and output boundaries.
- Lazy `set_index()` and `reset_index()`, plus source `index=` and `order_by=`
  declarations.
- Expression metadata for scalar, elementwise, and length-preserving behavior.
- Calibrated synthetic OHLC Parquet generator with smoke, 100 MB, 1 GB, and
  5 GB target-size presets.
- Self-documenting Make targets for development checks, builds, demo smoke
  runs, release validation, and cleanup.
- Package classifiers, project URLs, and PEP 561 typed-package metadata.
- Central documentation index with roadmap, design, decision, benchmark,
  changelog, and research sections.
- Explicit work-in-progress warning and a documented pre-`1.0` release policy.
- Make target for tested patch-version builds and manual PyPI publication.