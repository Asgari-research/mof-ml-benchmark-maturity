# Contributing

Thank you for helping improve the reproducibility of this benchmark.

## Reporting problems

Use a GitHub issue for:

- missing or inconsistent documentation;
- figure/table source-data mismatches;
- checksum failures;
- platform-specific installation problems;
- suspected bugs in the benchmark pipeline.

Do not attach private ARC–MOF data, processed benchmark tables, reviewer correspondence, credentials, or submission-system links to a public issue.

## Proposed changes

1. Create a branch from `main`.
2. Make the smallest coherent change.
3. Run:

   ```bash
   python scripts/revision/generate_revision_tables.py
   python scripts/validate_repository.py
   ```

4. Commit generated revision tables only when they are intentionally changed.
5. Open a pull request describing:
   - what changed;
   - why it changed;
   - which manuscript/figure/table claim is affected;
   - whether numerical results changed.

## Scientific integrity

- Preserve the fixed split and seed definitions unless the purpose of the change is explicitly to study an alternative design.
- Do not tune models on the held-out test partition.
- Preserve seed-level outputs behind aggregate claims.
- Report contradictory results rather than suppressing them.
- Distinguish empirical repeat frequencies from formal significance tests.
- Do not present correlated feature importance as causal mechanism.

## Code style

- Python 3.10 or newer;
- descriptive functions and type hints where practical;
- UTF-8 text files and LF line endings;
- no hard-coded private absolute paths in committed code.
