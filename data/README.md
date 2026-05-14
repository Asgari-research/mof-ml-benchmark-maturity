# Data folder

This repository does not redistribute the benchmark input data.

The pipeline expects the following local files:

```text
clean_data.csv
geometric_properties.csv
```

`clean_data.csv` is required. `geometric_properties.csv` is optional and is used only for geometric-descriptor consistency checks.

## Expected required columns

The expected identifier column is:

```text
filename
```

The expected topology column is:

```text
Crystalnet
```

The expected target columns are:

```text
uptake(mmol/g) CO2 at 0.015 bar
uptake(mmol/g) CO2 at 0.15 bar
uptake(mmol/g) methane at 5.8 bar
uptake(mmol/g) methane at 65 bar
```

## Data policy

The processed benchmark table was derived from ARC–MOF resources. Users should obtain the source data from the original ARC–MOF records and comply with the original data license and citation requirements.

Large data files are intentionally excluded from GitHub.

Do not commit:

```text
clean_data.csv
geometric_properties.csv
*.pkl.gz
*.csv.gz
small_data_mof_benchmark_outputs/
```
