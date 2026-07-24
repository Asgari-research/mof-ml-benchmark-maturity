# Output guide

## Full pipeline working directory

The main pipeline writes a generated working directory:

```text
small_data_mof_benchmark_outputs/
```

This directory may contain:

- logs;
- job checkpoints;
- seed-level predictions;
- model outputs;
- aggregated metrics;
- figure-data exports;
- figures;
- LaTeX tables;
- manifests.

The full working directory is intentionally excluded from Git because it can be large and may contain non-redistributable processed material.

## Curated publication archive

The repository tracks a curated, non-redundant numerical archive:

```text
publication_data/
```

### Figure source data

```text
publication_data/figure_source_data/main/
publication_data/figure_source_data/si/
```

The final Figure 7 is a single-panel synthesis/leaderboard figure. Redundant former panels are not duplicated in the canonical publication archive because their numerical data are already represented by earlier figures.

### Table source data

```text
publication_data/table_source_data/main/
publication_data/table_source_data/si/
```

This includes compact main-text tables and the larger machine-readable SI audit tables.

### Revision outputs

```text
publication_data/revision_outputs/
```

These tables were generated post-hoc from existing benchmark results and did not require model retraining.

### Metadata

```text
publication_data/metadata/
```

This folder records:

- file inventory;
- row and column counts;
- SHA-256 checksums;
- benchmark definition;
- provenance notes.

## Regeneration

Rebuild revision tables:

```bash
python scripts/revision/generate_revision_tables.py
```

Validate all tracked publication data:

```bash
python scripts/validate_repository.py
```
