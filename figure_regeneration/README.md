# Figure regeneration package

This folder contains the source data, plotting code, and regenerated publication figures for the manuscript:

**Benchmark Maturity in MOF Machine Learning: When Do Conclusions Become Scientifically Reliable?**

The files in this folder are intended for figure-level reproducibility. They do **not** retrain the models and do **not** redistribute the processed benchmark input table. Instead, they redraw the manuscript and Supporting Information figures from the machine-readable CSV exports used to prepare the final plots.

## Contents

```text
figure_regeneration/
  draw_all_figures.py
  requirements.txt
  file_manifest.csv
  source_data/
    figure_data_main/
    figure_data_si/
  redrawn_figures/
    figures_main/
    figures_si/
    validation_report.csv
```

## How to run

From the repository root:

```bash
pip install -r requirements.txt
python figure_regeneration/draw_all_figures.py
```

The script reads the CSV files in `source_data/` and writes regenerated figures to `redrawn_figures/`.

## What this package does

The figure-regeneration script:

* redraws all main-text figures from source CSV files;
* redraws all Supporting Information figures from source CSV files;
* writes figures in both PDF and PNG format;
* creates a simple validation report confirming that the expected source CSV files can be read.

## What this package does not do

This package does not:

* retrain machine-learning models;
* recreate the original train/test splits;
* regenerate per-job prediction files;
* redistribute the raw ARC--MOF data;
* redistribute the processed benchmark input table.

Full benchmark reproduction requires the local input files described in the main repository README and in `docs/DATA_AVAILABILITY.md`.

## Data scope

The CSV files included here are figure-level source data. They contain the numerical values plotted in the manuscript and Supporting Information figures. They are included to make the final visual evidence traceable without redistributing the underlying ARC--MOF-derived benchmark table.
