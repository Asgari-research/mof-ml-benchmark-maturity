# Data availability

The public repository at

https://github.com/Asgari-research/mof-ml-benchmark-maturity

contains the custom benchmark software, processed study input, publication source data, machine-readable audit
outputs, environment/configuration files, and reproducibility documentation.

The frozen publication release is `v1.0.0`.

<!-- ZENODO_DOI_START -->
Accepted-manuscript release `v1.0.0` (version DOI): https://doi.org/10.5281/zenodo.22080792

Current/all-versions record DOI: https://doi.org/10.5281/zenodo.22080791
<!-- ZENODO_DOI_END -->

## Processed study input

The benchmark input used by the publication is:

```text
data/clean_data.zip
```

The archive contains `clean_data.csv`, a processed and modified ARC–MOF-derived table.

## Original source data

Underlying source data are from ARC–MOF:

https://doi.org/10.5281/zenodo.6908728

Raw/unmodified ARC–MOF files are not republished here. Users requiring original source files should obtain them
from the original ARC–MOF record and comply with its source licence/citation requirements.

## Machine-readable publication outputs

The canonical source-data archive is `publication_data/`.

It contains the numerical source data and audit outputs required to inspect manuscript/SI figures, tables,
performance summaries, ranking/screening diagnostics, feature-effect diagnostics, and robustness/sensitivity
analyses.

Large intermediate predictions and model checkpoints are omitted because they are regenerable.
