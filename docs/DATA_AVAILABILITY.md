# Data availability

## Source data

The adsorption and structural information used by this project is derived from ARC-MOF resources. The original ARC-MOF archive is available at:

```text
https://doi.org/10.5281/zenodo.6908728
```

Users must comply with the original ARC-MOF licence, access conditions, and citation requirements.

## Public repository contents

This GitHub repository contains code, environment files, documentation, figure-regeneration scripts, figure-level source CSV files, and manuscript/SI assets. It does not redistribute:

- raw ARC-MOF data;
- the exact processed modelling table used in the manuscript;
- large prediction files;
- model checkpoints;
- complete generated output folders.

## Expected local input files

Required:

```text
clean_data.csv
```

Optional:

```text
geometric_properties.csv
```

`clean_data.csv` must contain the framework identifiers, descriptor columns, grouped topology labels, and adsorption targets expected by the pipeline. `geometric_properties.csv` is used only for optional descriptor-consistency checks.

## Manuscript-exact reproducibility

A manuscript-exact reproducibility record should contain the processed modelling table, immutable split identifiers, inclusion/exclusion record, column dictionary, checksums, pinned environment, principal result tables, figure-source data, and targeted revision outputs. That record is separate from the lightweight public repository. Its permanent archive link must be synchronized with the manuscript, SI, response letter, repository README, and standalone Data Availability Statement when available.

## Reuse boundary

The code can be inspected and the publication figures can be regenerated from the included figure-level source files. A full end-to-end benchmark rerun requires locally prepared input data or access to the manuscript-exact archive. The random test partitions are in-distribution held-out partitions from the same processed parent pool; they are not independent external databases.
