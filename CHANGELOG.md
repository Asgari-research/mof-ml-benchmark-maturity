# Changelog

All notable repository releases are documented here.

## [1.0.0] - 2026-08-24

### Added

- public processed benchmark input as `data/clean_data.zip`;
- dataset integrity manifest for the processed benchmark input;
- canonical `publication_data/` archive for manuscript and SI figure/table source data;
- deterministic revision-table generator and machine-readable robustness outputs;
- publication configuration and provenance/checksum metadata;
- repository and release-data validation scripts;
- GitHub Actions validation workflow;
- Zenodo and CFF metadata;
- explicit in-distribution validation scope and synchronized data-availability documentation.

### Changed

- synchronized README, data documentation, and reproducibility guidance with the public processed benchmark input;
- added PyYAML to the documented validation environment;
- aligned release metadata with Digital Discovery acceptance;
- retained the final publication top-5% overlap SD threshold of 0.01;
- documented the non-monotonic/local interpretation of the composite maturity diagnostic.

### Repository hygiene

- removed tracked Python cache artifacts;
- removed a redundant packaged copy of `publication_data/` when verified byte/tree-identical to the canonical archive;
- removed placeholder-only manuscript/SI asset directories;
- removed the one-off publication-threshold patch helper from the frozen research-software release.
