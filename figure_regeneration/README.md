# MOF benchmark ultra-elegant figure package

This package redraws the manuscript and SI figures from the supplied machine-readable CSV source data.

## Contents

- `draw_all_mof_benchmark_figures_ultra_elegant.py` — standalone plotting script.
- `source_data/figure_data_main/` — exact source CSVs for main-text figures.
- `source_data/figure_data_si/` — exact source CSVs for SI figures.
- `redrawn_figures/figures_main/` — regenerated main figures in PNG and PDF.
- `redrawn_figures/figures_si/` — regenerated SI figures in PNG and PDF.
- `redrawn_figures/validation_report.csv` — simple source-data read/shape check.
- `file_manifest.csv` — package file manifest.

## How to run

```bash
pip install -r requirements.txt
python draw_all_mof_benchmark_figures_ultra_elegant.py
```

The script does not retrain models and does not change numerical values. It only redraws the figures from the CSV exports, with improved panel spacing, legend placement, annotation placement, and readability.
