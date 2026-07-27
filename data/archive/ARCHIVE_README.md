# Manuscript-exact reproducibility archive

## Associated manuscript

**Title:** Benchmark Maturity in MOF Adsorption Machine Learning: When Do Conclusions Become Scientifically Reliable?  
**Journal:** Digital Discovery  
**Manuscript ID:** DD-ART-06-2026-000351  
**Code release tag:** `dd-art-06-2026-000351-rev1`

## Purpose

This archive preserves the manuscript-exact processed data identity, fixed in-distribution held-out partitions, metadata, environment, and principal machine-readable outputs needed to verify the revised analysis. It complements the public GitHub repository, which remains a lightweight code, documentation, and figure-regeneration package.

## Required archive components

```text
inputs/
  processed_modelling_table.*

splits/
  split_manifest.tsv
  <one immutable identifier file per target/test seed>
  <seed-specific nested-subset permutation or prefix manifest>

records/
  row_inclusion_exclusion.tsv
  preprocessing_record.md

metadata/
  column_dictionary.tsv
  target_dictionary.tsv
  environment-lock.yml
  software_versions.txt
  repository_commit.txt
  release_tag.txt

results/
  principal machine-readable result tables
  targeted_revision_outputs/
  figure_source_data/
  final_summary_manifests/

checksums/
  SHA256SUMS.txt
```

## Validation scope

The held-out rows are random in-distribution partitions of the same processed ARC-MOF-derived parent pool. This archive does not establish topology-disjoint, chemistry-disjoint, temporal, or general out-of-distribution performance.

## Source-data conditions

The underlying source data are derived from ARC-MOF. Archive contents must comply with the original ARC-MOF licence and citation requirements. If redistribution of the processed modelling table is restricted, provide a controlled reviewer-access version and document the access procedure accurately.

## Verification

1. Verify every file against `checksums/SHA256SUMS.txt`.
2. Confirm the repository commit and release tag.
3. Confirm that split identifier files are disjoint and match the recorded counts.
4. Confirm that nested subsets are derived from the archived seed-specific ordering.
5. Confirm environment identity before rerunning.
6. Compare regenerated principal tables and figure-source values with the archived outputs.

## Citation

Cite the associated manuscript, the tagged software release, the archive record, and the original ARC-MOF data source.
