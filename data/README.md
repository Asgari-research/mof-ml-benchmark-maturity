```markdown
# Data

This folder contains the data files used to support the benchmark-maturity study.

The file `clean_data.csv` is included in this folder. It is a processed and modified ARC–MOF-derived benchmark table prepared for this study. It was generated from ARC–MOF source data through data cleaning, identifier normalization, descriptor selection, adsorption-target organization, and preparation of machine-learning-ready inputs. It should therefore be treated as a derived dataset, not as the original ARC–MOF release.

The original ARC–MOF source data are available from Zenodo:

https://doi.org/10.5281/zenodo.6908728

ARC–MOF is distributed under the Creative Commons Attribution 4.0 International license. Because `clean_data.csv` is derived from ARC–MOF, users should cite the original ARC–MOF dataset and associated publication when using this file.

The geometric-property file `geometric_properties.csv` is not redistributed in this folder. Users who need this file for descriptor-consistency checks or full workflow regeneration should download it directly from the original ARC–MOF Zenodo record listed above.

Files in this folder:

- `clean_data.csv`: processed ARC–MOF-derived benchmark input table used by the workflow.
- `README.md`: this data-description and attribution file.

Additional files needed for full regeneration, such as `geometric_properties.csv`, should be obtained from the original ARC–MOF source record and placed locally according to the workflow instructions.

This repository provides the scripts, environment files, documentation, and machine-readable outputs needed to regenerate the reported figures and tables. Users should respect the license and citation requirements of the original ARC–MOF source data.

The derived file included here is shared to support transparency, reproducibility, and reuse of the present benchmark analysis. It should not be interpreted as an independent replacement for the original ARC–MOF database.
```
