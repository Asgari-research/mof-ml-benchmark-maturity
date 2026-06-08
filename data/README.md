```markdown
# Data

This folder contains derived data and figure/table source data used for the benchmark-maturity study.

The file `clean_data.csv` is a processed and modified ARC–MOF-derived table prepared for this study. It was generated from ARC–MOF source files and associated geometric-property information through cleaning, identifier normalization, descriptor selection, target-column organization, and preparation of machine-learning-ready inputs. It should therefore be treated as a derived dataset, not as the original ARC–MOF release.

The original ARC–MOF source data are available from Zenodo:

https://doi.org/10.5281/zenodo.6908728

ARC–MOF is distributed under the Creative Commons Attribution 4.0 International license. Any ARC–MOF-derived files included in this repository, including `clean_data.csv` and any processed geometric-property tables, are provided with attribution to the original ARC–MOF dataset and associated publication. Users should cite the original ARC–MOF dataset and paper when using these derived files.

Files in this folder may include:

- `clean_data.csv`: processed ARC–MOF-derived benchmark input table used by the workflow.
- `geometric_properties.csv`: ARC–MOF-derived or ARC–MOF-associated geometric-property table, where included for descriptor consistency checks or regeneration.

The repository provides the scripts, environment files, documentation, and machine-readable outputs needed to regenerate the reported figures and tables. Users who wish to reproduce the full workflow should check that the required input files are present and should respect the license and citation requirements of the original ARC–MOF source data.

These derived files are shared to support transparency, reproducibility, and reuse of the present benchmark analysis. They should not be interpreted as an independent replacement for the original ARC–MOF database.
```
