# v1.0.0 — Publication reproducibility release

This release prepares the public companion repository for the manuscript:

**Benchmark Maturity in MOF Adsorption Machine Learning: When Do Conclusions Become Scientifically Reliable?**

## Included

- full benchmark pipeline;
- Conda and pip environment specifications;
- canonical figure and table source-data archive;
- deterministic post-hoc revision-table generator;
- target-specific maturity summary;
- 81-setting cutoff-sensitivity grid;
- four-partition robustness tables;
- fixed-\(k\) screening summaries;
- top-1 consensus Wilson intervals;
- RF-versus-HGB pairwise summary;
- full-data pairwise stochasticity audit;
- checksums, provenance metadata, and automated validation;
- updated citation and Zenodo metadata.

## Scope

The numerical findings are specific to fixed in-distribution held-out partitions of the ARC–MOF-derived tabular benchmark. This release does not claim a universal sample-size law or general out-of-distribution performance.

## Data boundary

The release does not redistribute the raw ARC–MOF database or the processed benchmark input table. Public CSVs contain derived numerical figure/table source data and result summaries.

## Validation

Run:

```bash
python scripts/revision/generate_revision_tables.py
python scripts/validate_repository.py
```
