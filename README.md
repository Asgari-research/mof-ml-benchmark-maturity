# Benchmark Maturity in MOF Adsorption Machine Learning

Reproducible code, processed benchmark input, publication source data, and audit outputs for:

> **Benchmark Maturity in MOF Adsorption Machine Learning: When Do Conclusions Become Scientifically Reliable?**

## What this repository contains

This repository supports the accepted Digital Discovery study with:

- the benchmark and post-processing workflow;
- the processed ARC–MOF-derived benchmark input used by the workflow;
- fixed publication configuration and validation utilities;
- figure-regeneration code and source data;
- machine-readable figure/table data and robustness outputs;
- environment and provenance documentation.

## Validation scope

All reported numerical evaluations use fixed **in-distribution held-out partitions** drawn from the same processed
ARC–MOF-derived parent pool.

The results do not claim topology-disjoint, chemistry-disjoint, temporal, or general out-of-distribution validation.

## Repository structure

```text
config/                 publication configuration
data/                   processed benchmark input and provenance
docs/                   data availability, outputs, reproducibility
figure_regeneration/    figure code and source data
publication_data/       canonical machine-readable publication archive
scripts/                validation, DOI, and revision-analysis utilities
src/                    benchmark pipeline
```

## Processed benchmark input

The full study-specific ML input is distributed as:

```text
data/clean_data.zip
```

The ZIP contains `clean_data.csv`, a processed and modified ARC–MOF-derived table prepared for this study through
data cleaning, identifier normalization, descriptor selection, adsorption-target organization, and
machine-learning input preparation.

Extract it with:

```bash
python -m zipfile -e data/clean_data.zip data/
```

Its release identity is documented in:

```text
data/clean_data_manifest.json
```

## Original ARC–MOF source

The underlying source adsorption/structural data are available from ARC–MOF:

```text
https://doi.org/10.5281/zenodo.6908728
```

Raw/unmodified ARC–MOF files are not duplicated here. Users requiring original source files or optional
source-level descriptor checks should obtain those files from ARC–MOF and comply with its original licence and
citation requirements.

## Installation

### pip

```bash
python -m pip install -r requirements.txt
```

### Conda

```bash
conda env create -f environment.yml
conda activate mof-benchmark-maturity
```

## Run the benchmark

```bash
python src/small_data_mof_benchmark_pipeline.py --stage all
```

Model jobs only:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage run
```

Post-processing only:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage post
```

## Validate the publication release

```bash
python scripts/validate_release_data.py
python scripts/validate_repository.py
python -m compileall src scripts figure_regeneration
```

The same checks are run automatically by GitHub Actions.

## Regenerate publication figures

```bash
python figure_regeneration/draw_all_figures.py
```

## Publication source data

`publication_data/` is the canonical checksum-tracked machine-readable archive for manuscript/SI figure and table
source data and robustness analyses.

Large intermediate prediction caches and model checkpoints are not archived because they can be regenerated from
the processed input and documented workflow.

## Zenodo

<!-- ZENODO_DOI_START -->
The DOI for the frozen publication release `v1.0.0` will be inserted here after the manual Zenodo Software record is published.
<!-- ZENODO_DOI_END -->

## Citation

Please cite:

1. the associated Digital Discovery article when its article DOI is available;
2. the frozen Zenodo `v1.0.0` software archive;
3. the original ARC–MOF source record at `https://doi.org/10.5281/zenodo.6908728`.

See `CITATION.cff` for software citation metadata.

## Licence and data provenance

Custom software is released under the MIT License.

The processed benchmark input is derived from ARC–MOF and remains subject to the original source's licence and
citation requirements.
