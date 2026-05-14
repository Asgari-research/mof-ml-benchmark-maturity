# Reproducibility guide

This document explains how to reproduce the benchmark outputs after preparing the required local data files.

## 1. Prepare input files

Prepare the required local input file:

```text
clean_data.csv
```

Optional:

```text
geometric_properties.csv
```

Place these files beside the pipeline script or adjust the configuration in:

```text
src/small_data_mof_benchmark_pipeline.py
```

The script expects the following target columns:

```text
uptake(mmol/g) CO2 at 0.015 bar
uptake(mmol/g) CO2 at 0.15 bar
uptake(mmol/g) methane at 5.8 bar
uptake(mmol/g) methane at 65 bar
```

The expected identifier and topology columns are:

```text
filename
Crystalnet
```

## 2. Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate mof-benchmark-maturity
```

## 3. Run the full pipeline

```bash
python src/small_data_mof_benchmark_pipeline.py --stage all
```

This runs model jobs and post-processing.

## 4. Run only model jobs

```bash
python src/small_data_mof_benchmark_pipeline.py --stage run
```

## 5. Run only post-processing

```bash
python src/small_data_mof_benchmark_pipeline.py --stage post
```

This regenerates tables and figures from saved job outputs.

## 6. Resume interrupted runs

The workflow is checkpointed at the job level. If a run is interrupted, rerun the same command:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage all
```

Completed jobs are skipped automatically when checkpoint files are available.

## 7. Output folder

The main generated folder is:

```text
small_data_mof_benchmark_outputs/
```

This folder is intentionally ignored by Git.

## 8. Reproducibility checks

After running the workflow, check the following generated files:

```text
small_data_mof_benchmark_outputs/final_exports/project_summary.txt
small_data_mof_benchmark_outputs/final_exports/project_summary.json
```

Also inspect the core machine-readable outputs listed in `docs/OUTPUTS.md`.
