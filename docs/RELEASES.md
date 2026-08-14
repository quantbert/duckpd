# Release policy

DuckPD is a work in progress. Releases before `1.0` are intended for evaluation,
experimentation, and feedback rather than production-critical workloads.

## Versioning

DuckPD uses [PEP 440](https://peps.python.org/pep-0440/) versions with a
Semantic Versioning-inspired policy:

- `0.MINOR.PATCH` identifies a release before API stability.
- Increment `MINOR` for features and for breaking API or semantic changes.
- Increment `PATCH` for compatible fixes, documentation corrections, and
  packaging fixes.
- Use alpha versions such as `0.1.0a1` for early public releases.
- Reserve `.devN` versions for unreleased development snapshots.
- After `1.0`, incompatible public API changes require a major-version bump.

While the version is below `1.0`, supported behavior may change between minor
releases. The documented compatibility scope and changelog are authoritative;
DuckPD does not promise compatibility with arbitrary pandas programs.

## Deprecations

Before `1.0`, deprecation periods are best effort. When practical, behavior is
announced as deprecated in documentation and the changelog before removal. A
breaking correction may be made without a deprecation period when existing
behavior is unsafe, silently materializes data, or produces incorrect results.

After `1.0`, public behavior should normally remain deprecated for at least one
minor release before removal.

## Release process

Each release should:

1. Pass the repository quality gate and package build.
2. Set the intended version in `pyproject.toml`.
3. Move relevant entries from `Unreleased` into a dated changelog section.
4. Be tagged as `v<version>`, for example `v0.1.0a1`.
5. Be published from that tag through PyPI Trusted Publishing.

Published distributions and Git tags are immutable. A mistake in a published
release is corrected with a new version rather than replacing an artifact or
moving its tag.

## Current minimum gate

Until stronger artifact checks are implemented, the minimum release gate is:

- Ruff lint and format checks;
- strict Pyright analysis;
- the complete pytest suite and coverage threshold;
- successful wheel and source-distribution builds;
- an explicit work-in-progress warning in the package README.

Planned clean-environment installation and artifact-content checks are tracked
in the [roadmap](roadmap.md).
