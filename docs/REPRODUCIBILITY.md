# Reproducibility guide

This guide reproduces the benchmark workflow after the required local data have been prepared. It does not claim out-of-distribution or independent external validation; all reported test rows are from fixed in-distribution held-out partitions of the processed ARC-MOF-derived parent pool.

## 1. Prepare inputs

Required:

```text
clean_data.csv
```

Optional:

```text
geometric_properties.csv
```

Place the files at the configured local paths or update `ProjectConfig` in:

```text
src/small_data_mof_benchmark_pipeline.py
```

Expected target columns:

```text
uptake(mmol/g) CO2 at 0.015 bar
uptake(mmol/g) CO2 at 0.15 bar
uptake(mmol/g) methane at 5.8 bar
uptake(mmol/g) methane at 65 bar
```

Expected identifier/topology columns include:

```text
filename
Crystalnet
```

Consult the manuscript-exact archive's column dictionary before a verification rerun.

## 2. Create the environment

Using pip:

```bash
pip install -r requirements.txt
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate mof-benchmark-maturity
```

For manuscript-exact verification, use the archived pinned environment or lock file rather than resolving unconstrained package versions.

## 3. Run the benchmark

Full workflow:

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

## 4. Resume an interrupted run

The workflow is checkpointed at the job level. Re-run the same command; completed jobs are skipped when valid checkpoints exist.

## 5. Inspect outputs

The main output folder is:

```text
small_data_mof_benchmark_outputs/
```

Core machine-readable files are listed in `docs/OUTPUTS.md`. Also inspect:

```text
small_data_mof_benchmark_outputs/final_exports/project_summary.txt
small_data_mof_benchmark_outputs/final_exports/project_summary.json
```

## 6. Verify split identity

For manuscript-exact reproduction, do not create new random partitions. Use the archived immutable framework identifiers for the specified target and test seed. Confirm that:

- no framework identifier occurs in both the training pool and held-out partition;
- the held-out count matches the archived manifest;
- the nested subsets are prefixes of the archived seed-specific training-pool permutations;
- identical rows are supplied to all 16 pipelines at a given size and seed;
- model random states match the archived protocol.

## 7. Verify file identity

Run the checksum command appropriate to the platform, for example:

```bash
sha256sum -c SHA256SUMS.txt
```

A manuscript-exact rerun should record the repository commit SHA, release tag, environment identity, input checksums, and split-manifest checksums in its run manifest.

## 8. Regenerate figures without retraining

```bash
python figure_regeneration/draw_all_figures.py
```

This step uses included figure-source CSV files and does not retrain models. Compare regenerated figures and source-table checksums with the archived submission record.
