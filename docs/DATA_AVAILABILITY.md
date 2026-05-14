# Data availability

The processed benchmark input table used in the manuscript is not included in this repository.

The workflow expects a local `clean_data.csv` file containing framework identifiers, geometric descriptors, grouped topology labels, and adsorption targets. An optional `geometric_properties.csv` file can be provided for geometric-descriptor consistency checks.

## Expected local files

Required:

```text
clean_data.csv
```

Optional:

```text
geometric_properties.csv
```

## Why the data are not redistributed

The benchmark input table is derived from ARC–MOF resources. This repository provides the code and workflow logic but does not redistribute raw ARC–MOF files or the processed benchmark table.

Users should obtain the original data from the official ARC–MOF source records and comply with the original data license, access conditions, and citation requirements.

## Reproducibility intent

This repository is designed so that users who have prepared the required local input tables can reproduce the benchmark workflow, generated outputs, tables, and figures.

The repository intentionally excludes:

- raw data files,
- processed benchmark input tables,
- model checkpoints,
- per-job prediction files,
- generated output folders.

This avoids redistributing data and keeps the GitHub repository lightweight.
