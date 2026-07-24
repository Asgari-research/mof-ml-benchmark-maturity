# Post-hoc revision analyses

The files in `publication_data/revision_outputs/` were generated from existing benchmark outputs. No models were retrained for these tables.

## Included analyses

- target-specific first local composite-rule crossing;
- sensitivity of the strict rule to 81 cutoff combinations;
- alternative held-out partition learning curves;
- alternative partition composite-rule crossings;
- fixed-\(k\) screening recovery for \(k=50,100,250,500\);
- RF-versus-HGB full-data empirical pairwise superiority;
- Wilson intervals for top-1 consensus;
- full-data pairwise stochasticity audit.

## Canonical generation

Run:

```bash
python scripts/revision/generate_revision_tables.py
```

The generator reads the public aggregate, ranking, screening, job-level, and pairwise tables and rewrites the revision-output CSVs deterministically.

## Headline validation targets

The repository validator checks that:

- the target-specific first crossings are 2000, 5000, 10000, and not reached, as appropriate;
- the cutoff grid has 81 combinations;
- first crossings in the grid occur at 5000 for 36 combinations and 10000 for 18 combinations, while 27 combinations are not satisfied;
- the largest best-RMSE spread across four held-out partitions is approximately 0.01037 mmol g^-1;
- RF beats HGB in every sampled full-data comparison for the four test seeds;
- the full-data audit contains 1680 ordered off-diagonal pairs;
- 84 ordered Ridge-versus-Ridge comparisons are present;
- no Ridge-only anomaly is detected.
