# Output files

Running the pipeline creates:

```text
small_data_mof_benchmark_outputs/
```

This folder is intentionally ignored by Git because it can be regenerated.

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

## Important generated CSV files

### Per-job and aggregated metrics

```text
si_all_job_metrics.csv
```

Per-job metric table before aggregation. This is the main raw numerical audit trail.

```text
si_aggregated_performance.csv
```

Aggregated mean values, standard deviations, confidence intervals, and repeat counts.

### Ranking stability

```text
si_ranking_stability.csv
```

Top-1 consensus and rank-preservation summaries.

### Screening reproducibility

```text
si_screening_reproducibility.csv
```

Top-k overlap and elite-enrichment summaries, including the primary 5% elite-fraction metric and fixed-k variants.

### Descriptor-family summaries

```text
descriptor_family_aggregation.csv
```

Descriptor-family-level aggregation across targets and test partitions.

### Sample efficiency

```text
sample_efficiency.csv
```

Method-level sample-efficiency thresholds for recovering 50%, 80%, 90%, and 95% of attainable RMSE gain.

### Target difficulty

```text
target_difficulty.csv
```

Target-level summary comparing best full-data RMSE, rank correlation, and method spread across adsorption targets.

### Feature-effect convergence

```text
si_feature_effect_convergence.csv
```

Compact feature-effect convergence summary.

```text
si_feature_effect_importances.csv
```

Feature-level permutation-importance records.

### Final leaderboard / Pareto summary

```text
method_pareto_summary.csv
```

Full-data accuracy–ranking–screening trade-off summary.

### Pairwise superiority

```text
pairwise_superiority_*.csv
```

Pairwise probability-of-superiority matrices for selected targets, test seeds, and training sizes.

## Summary manifests

```text
project_summary.txt
project_summary.json
```

Human-readable and machine-readable summaries of the completed benchmark run.
