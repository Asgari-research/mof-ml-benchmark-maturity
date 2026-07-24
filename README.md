# Benchmark Maturity in MOF Adsorption Machine Learning

[![License: MIT](https://img.shields.io/badge/Code%20license-MIT-blue.svg)](LICENSE)
[![Paper: ChemRxiv](https://img.shields.io/badge/Paper-ChemRxiv-8A2BE2.svg)](https://doi.org/10.26434/chemrxiv.15003455/v2)
[![Repository validation](https://github.com/Asgari-research/mof-ml-benchmark-maturity/actions/workflows/validate.yml/badge.svg)](https://github.com/Asgari-research/mof-ml-benchmark-maturity/actions/workflows/validate.yml)

Reproducibility code and publication source data for:

> **Benchmark Maturity in MOF Adsorption Machine Learning: When Do Conclusions Become Scientifically Reliable?**

The project asks when a metal–organic framework (MOF) adsorption benchmark becomes reliable enough to support scientific conclusions about descriptor families, model ordering, candidate prioritization, and feature-effect summaries—not only a competitive point estimate.

## Scope of the evidence

The reported numerical transition is specific to:

- an ARC–MOF-derived tabular adsorption pool;
- random, fixed **in-distribution held-out partitions**;
- nested training subsets;
- four lightweight descriptor families;
- Ridge, random forest (RF), histogram gradient boosting (HGB), and a shallow multilayer perceptron (MLP);
- the four adsorption targets listed below.

The repository does **not** claim that the reported sample-size transition transfers unchanged to topology-disjoint, chemistry-disjoint, graph-model, or otherwise out-of-distribution settings.

## Targets

Main manuscript target:

```text
uptake(mmol/g) CO2 at 0.15 bar
```

Additional Supporting Information targets:

```text
uptake(mmol/g) CO2 at 0.015 bar
uptake(mmol/g) methane at 5.8 bar
uptake(mmol/g) methane at 65 bar
```

## Repository map

```text
.
├── src/
│   └── small_data_mof_benchmark_pipeline.py
├── figure_regeneration/
│   ├── draw_all_figures.py
│   ├── source_data/
│   └── redrawn_figures/
├── publication_data/
│   ├── figure_source_data/
│   │   ├── main/
│   │   └── si/
│   ├── table_source_data/
│   │   ├── main/
│   │   └── si/
│   ├── revision_outputs/
│   └── metadata/
├── scripts/
│   ├── revision/
│   │   └── generate_revision_tables.py
│   ├── patch_publication_threshold.py
│   ├── update_zenodo_doi.py
│   └── validate_repository.py
├── config/
│   └── publication_v1.yml
├── data/
│   └── README.md
├── docs/
│   ├── DATA_AVAILABILITY.md
│   ├── OUTPUTS.md
│   ├── REPRODUCIBILITY.md
│   └── REVISION_ANALYSES.md
├── manuscript_assets/
│   └── README.md
├── supplementary_assets/
│   └── README.md
├── .github/workflows/validate.yml
├── CITATION.cff
├── .zenodo.json
├── CHANGELOG.md
├── CONTRIBUTING.md
├── environment.yml
├── requirements.txt
└── LICENSE
```

## What is public and what is not

### Public in this repository

- the main benchmark pipeline;
- environment specifications;
- figure-regeneration code;
- curated CSV source data underlying manuscript and SI figures;
- curated CSV source data underlying manuscript and SI tables;
- post-hoc revision tables;
- checksums and provenance metadata.

### Not redistributed

- raw ARC–MOF records;
- the processed benchmark input table (`clean_data.csv`);
- optional local geometry-check table (`geometric_properties.csv`);
- model checkpoints;
- large job-level prediction arrays;
- private reviewer correspondence or internal revision-planning documents.

The public CSV files are derived numerical source data for verification. They are not a redistribution of the underlying MOF database.

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate mof-benchmark-maturity
```

### pip

```bash
python -m pip install -r requirements.txt
```

## Full benchmark workflow

Place the required local input file beside the pipeline or update the configured path:

```text
clean_data.csv
```

Optional consistency-check input:

```text
geometric_properties.csv
```

Run all stages:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage all
```

Run model jobs only:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage run
```

Regenerate post-processing from saved checkpoints:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage post
```

The complete workflow writes to:

```text
small_data_mof_benchmark_outputs/
```

That generated working directory is intentionally excluded from Git tracking.

## Regenerate publication figures

The lightweight figure package does not retrain models and does not require the private benchmark input table:

```bash
python figure_regeneration/draw_all_figures.py
```

Canonical numerical source data for publication auditing are in `publication_data/`. The existing `figure_regeneration/source_data/` directory is retained as the plotting convenience layer used by `draw_all_figures.py`.

## Regenerate revision tables

```bash
python scripts/revision/generate_revision_tables.py
```

The script reads the curated public result tables and deterministically rebuilds:

- per-target first local composite-rule crossings;
- test-partition sensitivity;
- cutoff-sensitivity grid;
- top-1 consensus Wilson intervals;
- fixed-\(k\) screening summaries;
- RF-versus-HGB pairwise summaries;
- the full-data pairwise audit.

Outputs are written to:

```text
publication_data/revision_outputs/
```

## Validate the repository

```bash
python scripts/validate_repository.py
```

Validation checks include:

- required-file presence;
- CSV readability;
- publication-data checksums;
- expected row counts and headline revision values;
- composite-threshold consistency;
- absence of internal reviewer files;
- duplicate README detection;
- Python syntax compilation.

The same validation runs automatically through GitHub Actions.

## Benchmark definitions

The publication configuration is recorded in:

```text
config/publication_v1.yml
```

The strict composite diagnostic uses:

- top-two RMSE interval overlap;
- top-1 consensus \(\ge 0.80\);
- mean rank-Spearman versus full ordering \(\ge 0.90\);
- top-5% overlap SD \(\le 0.01\);
- top-5% enrichment SD \(\le 0.20\).

The flag is a diagnostic operating rule and may be non-monotonic because loss of top-two interval overlap at larger sample sizes can reflect stronger model separation rather than reduced reliability.

## Data availability

See [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) for the exact boundary between public derived outputs and non-redistributed inputs.

<!-- ZENODO_DOI_START -->
A DOI-bearing archive will be linked here after the first GitHub release is ingested by Zenodo.
<!-- ZENODO_DOI_END -->

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff). When using the benchmark, cite:

1. the associated manuscript/preprint;
2. the original ARC–MOF source publication and records;
3. the DOI-bearing software release once available.

## License

The code and documentation are released under the MIT License.

The MIT License does not automatically relicense third-party source data. Users must follow the original ARC–MOF access, license, and citation requirements.

## Maintainers

- Shayan Abaei
- Hosein Alimardani
- Mehrdad Asgari

Questions about the scientific benchmark should be raised through a GitHub issue so that answers remain visible and reusable.
