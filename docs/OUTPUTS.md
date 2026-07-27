# Output files

Running the full workflow creates:

```text
small_data_mof_benchmark_outputs/
```

The folder is excluded from Git because it can be regenerated from the required local inputs and documented code state.

## Main output folders

```text
logs/
checkpoints/
data_processed/
results/
manuscript_assets/
supplementary_assets/
final_exports/
```

## Principal machine-readable files

### Per-job and aggregated held-out metrics

```text
si_all_job_metrics.csv
si_aggregated_performance.csv
```

The first file stores per-job metrics. The second stores repeat means, standard deviations, descriptive intervals, and repeat counts.

### Ranking stability

```text
si_ranking_stability.csv
```

Contains modal-winner consensus and ordering agreement with the full-data ranking.

### Screening reproducibility

```text
si_screening_reproducibility.csv
fixed_k_screening_summary.csv
```

Contains elite-fraction and fixed-k overlap/enrichment summaries. Mean values quantify screening utility; across-repeat variation quantifies screening stability.

### Descriptor-family and target summaries

```text
descriptor_family_aggregation.csv
sample_efficiency.csv
target_difficulty.csv
```

### Feature-effect diagnostics

```text
si_feature_effect_convergence.csv
si_feature_effect_importances.csv
```

### Pairwise superiority

```text
pairwise_superiority_*.csv
rf_vs_hgb_pairwise.csv
full_data_pairwise_audit.csv
```

Pairwise values are empirical repeat frequencies evaluated on the fixed in-distribution held-out partition.

### Revision analyses

```text
per_target_maturity_summary.csv
cutoff_sensitivity_grid.csv
test_partition_alt_seed_curves.csv
top1_consensus_uncertainty.csv
group_ablation_seed_level.csv
group_ablation_summary.csv
stability_metric_uncertainty.csv
rf_low_n_sensitivity_seed_level.csv
rf_low_n_sensitivity_summary.csv
```

These files support the targeted analyses added during revision, including Tables S10-S19.

### Summary manifests

```text
project_summary.txt
project_summary.json
```

For manuscript-exact reproduction, archive the final copies of these files together with input, split, environment, and checksum manifests.
