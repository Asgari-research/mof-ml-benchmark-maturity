# Publication outputs

The full benchmark creates a generated workspace that is intentionally not tracked because it can be regenerated
from the processed study input and the documented workflow.

The canonical publication archive is:

```text
publication_data/
```

It includes:

- held-out performance summaries;
- ranking-stability diagnostics;
- screening-reproducibility summaries;
- descriptor-family and target summaries;
- feature-effect diagnostics;
- pairwise superiority outputs;
- maturity/cutoff/test-partition sensitivity analyses;
- fixed-k screening and consensus uncertainty outputs;
- group-ablation and RF-sensitivity outputs;
- figure source data;
- table source data;
- metadata/checksum manifests.

For verification, record the release tag, commit SHA, environment, publication configuration,
`data/clean_data_manifest.json`, and checksums of the specific outputs being compared.
