# DuckPD documentation

This directory contains project planning, technical decisions, benchmarks, and
supporting research. User-facing installation and quick-start information stays
in the repository [README](../README.md), while contribution instructions stay
in [CONTRIBUTING.md](../CONTRIBUTING.md) so GitHub can surface them automatically.


## User guides

- [Getting started](GETTING_STARTED.md) — installation, execution boundaries,
  ordering, resource-bounded sessions, HTTP/S3/GCS Parquet, and read-only
  PostgreSQL/MySQL/SQLite attachments.
- [API Compatibility & Semantic Matrix](COMPATIBILITY.md) — public signatures,
  supported arguments, intentional deviations, and unsupported behavior.

## Project status

- [Implementation roadmap](roadmap.md) — product contract, phased work, testing
  strategy, and release criteria.
- [API Compatibility & Semantic Matrix](COMPATIBILITY.md) — comprehensive overview
  of supported methods, DuckPD extensions, intentional deviations, and behavioral
  guarantees versus pandas 3.0.
- [Changelog](CHANGELOG.md) — unreleased changes and reconstructed, untagged development milestones.
- [Release policy](RELEASES.md) — pre-`1.0` versioning, deprecation, and
  publishing rules.
- [Benchmark results](BENCHMARK.md) — reproducible DuckPD and pandas performance
  and memory comparisons.

## Architecture and design

- [Core execution contract](decisions/0001-core-contract.md) — accepted decision
  covering laziness, execution boundaries, sessions, indexes, and fallback.
- [Ordering, indexing, and session contract](decisions/0002-order-index-session-contract.md) — accepted decision
  covering hidden row identity, honest ordering guarantees, lazy `.loc`, and session isolation.
- [Project directives and relational architecture](decisions/0003-directives-and-architecture.md) — foundational
  engineering principles, execution invariants, and compiler layer boundaries.
- [Feature Store architecture](design/featurestore-architecture.md) — implemented
  catalog, point-in-time alignment, and partition-cache design.


## Research and references

- [Competitive landscape](references/competitive-landscape.md) — relevant ideas,
  compatibility risks, benchmark targets, and primary implementation sources.

## File conventions

`README.md`, `CONTRIBUTING.md`, and `LICENSE` remain at the repository root for
standard GitHub and packaging discovery. Detailed project documentation belongs
under `docs/`; accepted architecture decisions belong under `docs/decisions/`,
historical design material under `docs/design/`, and research under
`docs/references/`.
