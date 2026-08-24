# Publication source data

This directory is the canonical checksum-tracked numerical archive for the manuscript and Supporting Information.

## Contents

- `figure_source_data/main/`: main-text figure source CSVs.
- `figure_source_data/si/`: SI figure source CSVs.
- `table_source_data/main/`: main-text table source CSVs.
- `table_source_data/si/`: SI table source data and machine-readable audit outputs.
- `revision_outputs/`: robustness/sensitivity outputs generated during manuscript revision.
- `metadata/`: benchmark definition, provenance, and SHA-256 manifest.

## Regeneration and validation

```bash
python scripts/revision/generate_revision_tables.py
python scripts/validate_repository.py
```

## Data boundary

These files are derived figure/table source data and benchmark result summaries.

The complete processed benchmark input is not duplicated in this folder; it is distributed once as:

```text
data/clean_data.zip
```

The original raw/unmodified ARC–MOF source archive is not duplicated in this repository.

Original ARC–MOF record:

https://doi.org/10.5281/zenodo.6908728
