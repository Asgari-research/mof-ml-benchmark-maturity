# Benchmark Maturity in MOF Adsorption Machine Learning

Code, documentation, figure-regeneration assets, and machine-readable audit outputs for the manuscript:

> **Benchmark Maturity in MOF Adsorption Machine Learning: When Do Conclusions Become Scientifically Reliable?**

The project evaluates when conclusions drawn from an ARC-MOF-derived tabular adsorption benchmark become reproducible enough for a stated in-distribution use. It audits predictive error, method ordering, pairwise superiority, screening consistency, and feature-effect convergence as training data are added.

## Evaluation scope

All numerical evaluations use random **in-distribution held-out test partitions** drawn from the same processed ARC-MOF-derived parent pool. The repository and manuscript do not claim topology-disjoint, chemistry-disjoint, temporal, or general out-of-distribution validation.

The main benchmark design uses:

- fixed in-distribution train/held-out partitions;
- nested training subsets;
- repeated subsampling;
- 16 descriptor-model pipelines;
- held-out regression metrics;
- ranking-stability diagnostics;
- screening-reproducibility diagnostics;
- pairwise frequencies of superiority;
- feature-effect convergence diagnostics.

The main target is CO2 uptake at 0.15 bar. The Supporting Information also reports CO2 uptake at 0.015 bar and methane uptake at 5.8 and 65 bar.

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

docs/
  DATA_AVAILABILITY.md
  OUTPUTS.md
  REPRODUCIBILITY.md

figure_regeneration/
  draw_all_figures.py
  source_data/
  redrawn_figures/

manuscript_assets/
supplementary_assets/
```

## Installation

Using pip:

```bash
pip install -r requirements.txt
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate mof-benchmark-maturity
```

## Input data

The full pipeline expects a locally prepared `clean_data.csv` file containing framework identifiers, geometric descriptors, grouped topology labels, and adsorption targets. An optional `geometric_properties.csv` file is used only for geometric-descriptor consistency checks.

The public GitHub repository does not redistribute raw ARC-MOF files, the manuscript's processed modelling table, large prediction files, model checkpoints, or complete generated output folders. Consult `docs/DATA_AVAILABILITY.md` before attempting a full rerun.

## Running the workflow

Run the complete benchmark and post-processing workflow:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage all
```

Run model jobs only:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage run
```

Run post-processing from saved checkpoints:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage post
```

The workflow is checkpointed at the job level. Re-running the same command skips completed jobs when valid checkpoint files are present.

## Generated outputs

The pipeline creates `small_data_mof_benchmark_outputs/`, containing logs, processed tables, job checkpoints, predictions, metrics, figure-data exports, manuscript figures, SI figures, LaTeX tables, and summary manifests. Generated folders are excluded from Git because they can be recreated from the documented workflow and the required local inputs.

The principal machine-readable outputs are described in `docs/OUTPUTS.md`.

## Figure regeneration

Publication figures can be regenerated without retraining models:

```bash
python figure_regeneration/draw_all_figures.py
```

The script reads the included figure-level CSV files from `figure_regeneration/source_data/` and writes figures to `figure_regeneration/redrawn_figures/`. These files provide a numerical audit trail for the published figures without redistributing the complete processed ARC-MOF-derived modelling table.

## Reproducibility boundary

The public repository supports code inspection, figure regeneration, and full reruns after the required local input table has been prepared. Exact reproduction of the submitted analysis additionally requires the manuscript-specific processed table, immutable split identifiers, inclusion/exclusion record, environment identity, and principal result files. The location of the manuscript-exact archive should be added here when its permanent record is available.

## Scope and interpretation

The numerical sample-size transition is specific to the targets, split design, model classes, descriptor space, and database coverage analysed in the manuscript. It is not a universal row-count rule. The transferable contribution is the joint audit of accuracy, method ordering, screening behaviour, and feature-effect reproducibility.

## Citation

Users should cite:

1. the associated manuscript when available;
2. the original ARC-MOF data publication and archive;
3. this software repository or its tagged release.

See `CITATION.cff` for software citation metadata.

## Licence

Code and documentation are released under the MIT License. Data are not licensed by this repository; users must follow the original data providers' access, licence, and citation requirements.
