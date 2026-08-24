# Reproducibility guide

This guide corresponds to publication release `v1.0.0`.

## 1. Install the environment

```bash
python -m pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate mof-benchmark-maturity
```

## 2. Extract the processed benchmark input

```bash
python -m zipfile -e data/clean_data.zip data/
```

The resulting file is:

```text
data/clean_data.csv
```

Its release checksum/schema record is:

```text
data/clean_data_manifest.json
```

Underlying ARC–MOF source data:

https://doi.org/10.5281/zenodo.6908728

## 3. Validate the release

```bash
python scripts/validate_release_data.py
python scripts/validate_repository.py
python -m compileall src scripts figure_regeneration
```

## 4. Run the benchmark

```bash
python src/small_data_mof_benchmark_pipeline.py --stage all
```

## 5. Regenerate figures without retraining

```bash
python figure_regeneration/draw_all_figures.py
```

## 6. Provenance for a verification rerun

Record:

- release tag (`v1.0.0`);
- exact Git commit SHA;
- `data/clean_data.zip` SHA-256;
- internal `clean_data.csv` SHA-256;
- Python/environment identity;
- publication configuration;
- checksums of compared outputs.

## Interpretation boundary

The reported held-out evaluation is in-distribution. It does not establish general out-of-distribution performance.
