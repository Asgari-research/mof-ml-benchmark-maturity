# MOF Benchmark Maturity

Code companion repository for the manuscript:

**Benchmark Maturity in MOF Machine Learning: When Do Conclusions Become Scientifically Reliable?**

This repository contains the reproducibility files for a benchmark-maturity study in metal–organic framework (MOF) machine learning. The project asks when a MOF benchmark becomes stable enough to support scientific conclusions rather than only competitive numerical scores.

The workflow evaluates how benchmark conclusions change as training data volume increases. It uses a fixed external test set, nested training subsets, repeated retraining, lightweight descriptor families, and multiple stability diagnostics.

## Scientific aim

The central question is:

> When does a MOF machine-learning benchmark become scientifically reliable enough to support conclusions about descriptor value, model ranking, screening readiness, and feature-effect interpretation?

The manuscript studies this question using a tabular ARC–MOF-derived adsorption benchmark.

## Main benchmark design

The pipeline evaluates:

- fixed external train/test splits,
- nested training subsets,
- repeated subsampling,
- 16 descriptor–model pipelines,
- external-test regression metrics,
- ranking stability,
- screening reproducibility,
- pairwise probability of superiority,
- feature-effect convergence.

The main target in the manuscript is:

```text
uptake(mmol/g) CO2 at 0.15 bar
```

Additional targets used in the Supporting Information are:

```text
uptake(mmol/g) CO2 at 0.015 bar
uptake(mmol/g) methane at 5.8 bar
uptake(mmol/g) methane at 65 bar
```

## Repository structure

```text
README.md
LICENSE
CITATION.cff
requirements.txt
environment.yml
.gitignore

src/
  small_data_mof_benchmark_pipeline.py

data/
  README.md
  .gitkeep

docs/
  DATA_AVAILABILITY.md
  OUTPUTS.md
  REPRODUCIBILITY.md

manuscript_assets/
  README.md

supplementary_assets/
  README.md
```

## Main pipeline

The main script should be placed at:

```text
src/small_data_mof_benchmark_pipeline.py
```

The script is intentionally written as a single transparent workflow. It creates fixed external splits, nested training subsets, per-job checkpoints, metrics tables, figures, LaTeX tables, and machine-readable outputs.

## Data availability

The processed benchmark input data are **not included** in this repository.

The workflow expects the following local files:

```text
clean_data.csv
geometric_properties.csv
```

`clean_data.csv` is required. `geometric_properties.csv` is optional and is used only for geometric-descriptor consistency checks.

The underlying source data are derived from ARC–MOF resources. Users should obtain the source data from the original ARC–MOF records and comply with the original license and citation requirements.

This repository does not redistribute:

- raw ARC–MOF data,
- processed benchmark input tables,
- generated prediction files,
- generated model/checkpoint files,
- large output folders.

## Installation

Using `pip`:

```bash
pip install -r requirements.txt
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate mof-benchmark-maturity
```

## Running the workflow

Place the required input file(s) beside the pipeline script or adapt the `ProjectConfig` paths inside:

```text
src/small_data_mof_benchmark_pipeline.py
```

Run the complete workflow:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage all
```

Run model jobs only:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage run
```

Run only post-processing from saved checkpoints:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage post
```

## Resume behavior

The pipeline is checkpointed at the job level. If a run is interrupted, rerun the same command. Completed jobs are skipped automatically when checkpoint files are available.

## Generated outputs

The pipeline creates:

```text
small_data_mof_benchmark_outputs/
```

This folder contains logs, processed tables, job checkpoints, prediction files, metrics, figure-data exports, manuscript figures, SI figures, LaTeX tables, and summary manifests.

Generated outputs are intentionally excluded from Git tracking because they can be recreated from the code and local input files.



## Figure regeneration package

In addition to the full benchmark pipeline, this repository includes a lightweight figure-regeneration package:

```text
figure_regeneration/
```

This folder contains the source CSV files, plotting script, and regenerated PDF/PNG figures used for the final manuscript and Supporting Information. It is intended for figure-level reproducibility and manuscript auditing. Running this script does not require the full benchmark input table and does not retrain any models.

To regenerate the publication figures:

```bash
python figure_regeneration/draw_all_figures.py
```

The script reads figure-level CSV files from:

```text
figure_regeneration/source_data/
```

and writes regenerated figures to:

```text
figure_regeneration/redrawn_figures/
```

These figure-level CSV files are included because they provide the numerical audit trail for the published figures without redistributing the underlying ARC--MOF-derived benchmark table.

## Key generated files

Important generated machine-readable outputs include:

```text
si_all_job_metrics.csv
si_aggregated_performance.csv
si_ranking_stability.csv
si_screening_reproducibility.csv
descriptor_family_aggregation.csv
sample_efficiency.csv
target_difficulty.csv
si_feature_effect_convergence.csv
si_feature_effect_importances.csv
method_pareto_summary.csv
pairwise_superiority_*.csv
```

These files form the numerical audit trail behind the manuscript and Supporting Information.

## Scope and limitations

This repository is intended to reproduce the benchmark workflow after the required local input tables have been prepared. It does not redistribute the underlying MOF database or processed benchmark table.

The numerical sample-size transition reported in the manuscript is specific to this ARC–MOF-derived tabular adsorption benchmark and should not be interpreted as a universal rule for all MOF machine-learning tasks. The general framework—checking ranking stability, screening reproducibility, pairwise superiority, and feature-effect convergence—is intended to be reusable.

## Citation

If you use this repository, please cite:

1. the associated manuscript, once available;
2. the original ARC–MOF data source/publication;
3. this software repository.

See `CITATION.cff` for software citation metadata.

## License

The code and documentation in this repository are released under the MIT License.

Data are not redistributed. Users must consult the original data providers for data access, license, and citation requirements.
