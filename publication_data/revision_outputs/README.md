# Revision outputs

These CSV files were generated post-hoc from the existing public benchmark result tables. No models were retrained.

Canonical generator:

```bash
python scripts/revision/generate_revision_tables.py
```

Files:

- `composite_rule_by_target_and_partition.csv`: all local composite-rule evaluations.
- `per_target_maturity_summary.csv`: first local crossing by target.
- `test_partition_sensitivity.csv`: first local crossing by held-out test seed.
- `cutoff_sensitivity_grid.csv`: 81 combinations of rule cutoffs.
- `top1_consensus_uncertainty.csv`: Wilson intervals for modal-winner consensus.
- `fixed_k_screening_summary.csv`: practical shortlist recovery.
- `rf_vs_hgb_pairwise.csv`: full-data RF/HGB empirical pairwise probabilities.
- `full_data_pairwise_audit.csv`: ordered full-data method-pair audit.
- `test_partition_alt_seed_curves.csv`: best-RMSE curves across test seeds 17, 29, 47, and 71.
