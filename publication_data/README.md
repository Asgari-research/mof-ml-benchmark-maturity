# Publication source data

This directory is the canonical, checksum-tracked numerical archive for the manuscript and Supporting Information.

## Contents

- `figure_source_data/main/`: source CSVs for main-text figures.
- `figure_source_data/si/`: source CSVs for SI figures.
- `table_source_data/main/`: source CSVs for main-text tables.
- `table_source_data/si/`: source CSVs for SI tables and machine-readable audit outputs.
- `revision_outputs/`: post-hoc tables used in the major revision.
- `metadata/`: benchmark definition, provenance, and SHA-256 manifest.

## Final Figure 7

The final manuscript uses a single non-redundant Figure 7 panel. Its source data are stored as:

```text
figure_source_data/main/figure7_full_data_leaderboard.csv
```

The numerical data behind the removed redundant panels are already represented by the source files for Figures 2–4 and are not duplicated here.

## Regeneration

```bash
python scripts/revision/generate_revision_tables.py
python scripts/validate_repository.py
```

## Data boundary

These CSVs are derived figure/table source data and benchmark result summaries. They do not contain the raw ARC–MOF database or the processed benchmark input table.
