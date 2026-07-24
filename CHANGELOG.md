# Changelog

All notable repository releases are documented here.

## [1.0.0] - 2026-07-23

### Added

- canonical `publication_data/` archive for manuscript and SI figure/table source data;
- deterministic revision-table generator;
- post-hoc revision outputs with corrected four-seed partition sensitivity;
- checksum and provenance metadata;
- publication configuration file;
- repository validation script and GitHub Actions workflow;
- Zenodo and CFF metadata;
- explicit in-distribution scope and data-availability documentation.

### Changed

- corrected the manuscript title in repository metadata;
- replaced broad “external-test” and “screening readiness” language with fixed in-distribution held-out and first-pass filtering language;
- aligned the top-5% overlap SD threshold with the manuscript value of 0.01;
- documented the non-monotonic/local interpretation of the composite diagnostic;
- made final Figure 7 source data non-redundant.

### Removed from the public release

- duplicate root `README.MD`;
- private reviewer correspondence;
- internal revision plans and handoff documents;
- obsolete or incomplete revision helper outputs;
- redundant former Figure 7 panel source files.
