# Data availability

## Public materials

This repository provides:

- benchmark and post-processing source code;
- environment specifications;
- curated CSV source data underlying manuscript figures;
- curated CSV source data underlying manuscript and Supporting Information tables;
- post-hoc revision outputs;
- file-level checksums and provenance metadata.

The canonical public numerical archive is:

```text
publication_data/
```

## Materials not redistributed

The repository does not redistribute:

- raw ARC–MOF records;
- the processed benchmark input table (`clean_data.csv`);
- the optional local geometry-check table (`geometric_properties.csv`);
- model checkpoints;
- large prediction arrays or job checkpoint folders;
- private peer-review correspondence;
- internal revision-planning documents.

The underlying database must be obtained from its original providers and used under their access, license, and citation conditions.

## Reproducibility boundary

Two levels of reproducibility are supported.

### Figure/table-level reproduction

This level is fully public. Users can inspect the numerical values underlying the figures and tables, verify checksums, regenerate post-hoc revision tables, and redraw publication figures without the private processed benchmark input.

### Full model-level reproduction

This level requires users to prepare the local benchmark input table from the original ARC–MOF resources. The repository documents the expected filenames, targets, descriptors, splits, fixed model settings, and commands, but does not redistribute the prepared input table.

## Derived-data licensing

The repository MIT License applies to code and documentation. It does not relicense third-party source records. The public CSV files are derived numerical outputs supplied for scholarly verification; users must cite the associated paper and the original ARC–MOF sources and must respect any applicable upstream terms.

## DOI archive

<!-- ZENODO_DOI_START -->
A DOI-bearing software archive will be added after the first GitHub release is processed by Zenodo.
<!-- ZENODO_DOI_END -->
