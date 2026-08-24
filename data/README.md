# Processed benchmark data

## `clean_data.zip`

`clean_data.zip` contains `clean_data.csv`, the processed and modified ARC–MOF-derived benchmark table used by the
publication workflow.

The table was prepared through data cleaning, identifier normalization, descriptor selection,
adsorption-target organization, and machine-learning input preparation.

It is a **derived study dataset**, not an independent replacement for ARC–MOF.

Original source:

https://doi.org/10.5281/zenodo.6908728

## Extract

```bash
python -m zipfile -e data/clean_data.zip data/
```

## Integrity and schema

The release preparation generates:

```text
data/clean_data_manifest.json
```

which records:

- archive SHA-256;
- internal CSV SHA-256;
- archive/internal file sizes;
- data-row count;
- column count;
- column names;
- source DOI and provenance note.

## Data boundary

Raw/unmodified ARC–MOF source files are not duplicated in this repository.

Optional source-level files such as `geometric_properties.csv` should be obtained from the original ARC–MOF record
when required.

Users should cite ARC–MOF and follow the original source licence/citation requirements.
