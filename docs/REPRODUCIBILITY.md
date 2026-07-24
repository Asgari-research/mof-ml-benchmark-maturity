# Reproducibility protocol

## Evaluation scope

All reported main benchmark results use fixed, random, in-distribution held-out partitions drawn from the same ARC–MOF-derived parent pool. They assess interpolation reliability and do not establish topology-disjoint or general out-of-distribution performance.

## Split and seed protocol

For each target and test seed:

1. A fixed held-out partition is created once.
2. The remaining rows form the training pool.
3. For each subsample seed, a random permutation of the training pool is generated.
4. Nested training subsets are prefixes of that permutation.
5. The same subset is used for every descriptor–model pipeline at the same training size and seed.
6. Model random states follow the benchmark pipeline's recorded seed logic.
7. At full training size, row sampling is identical across repeats; any remaining variation comes from stochastic learners.

## Primary target

```text
uptake(mmol/g) CO2 at 0.15 bar
```

Primary test seed:

```text
17
```

Primary subsample seeds:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

## Training sizes

```text
500, 1000, 2000, 5000, 10000, 20000, 40000, 210995
```

## Descriptor families

- geometry only;
- enriched interpretable geometry;
- topology only;
- geometry plus topology.

## Model families

- Ridge;
- RF;
- HGB;
- shallow MLP.

## Composite diagnostic

The publication configuration is stored in `config/publication_v1.yml`.

The baseline rule requires:

- overlapping top-two RMSE repeat intervals;
- top-1 consensus at least 0.80;
- mean rank-Spearman versus the full-data ordering at least 0.90;
- top-5% overlap SD no greater than 0.01;
- top-5% enrichment SD no greater than 0.20.

The first satisfying size is a local diagnostic crossing, not a guaranteed persistent or universal threshold.

## Important statistical interpretation

The primary target uses ten repeated subsample seeds, so empirical probabilities have resolution 0.1. Auxiliary targets and alternative partitions may use five repeats, giving resolution 0.2. Wilson intervals are therefore reported for consensus proportions.

## Commands

Full run:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage all
```

Post-processing only:

```bash
python src/small_data_mof_benchmark_pipeline.py --stage post
```

Revision tables:

```bash
python scripts/revision/generate_revision_tables.py
```

Repository validation:

```bash
python scripts/validate_repository.py
```
