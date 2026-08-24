#!/usr/bin/env python3
"""Validate the public repository before a GitHub release."""
from __future__ import annotations

import hashlib
import json
import math
import py_compile
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml


EXPECTED_REQUIRED = [
    "README.md",
    "CITATION.cff",
    ".zenodo.json",
    "config/publication_v1.yml",
    "publication_data/metadata/file_manifest.csv",
    "publication_data/revision_outputs/per_target_maturity_summary.csv",
    "publication_data/revision_outputs/test_partition_sensitivity.csv",
    "publication_data/revision_outputs/cutoff_sensitivity_grid.csv",
    "publication_data/revision_outputs/test_partition_alt_seed_curves.csv",
    "publication_data/revision_outputs/fixed_k_screening_summary.csv",
    "publication_data/revision_outputs/rf_vs_hgb_pairwise.csv",
    "publication_data/revision_outputs/top1_consensus_uncertainty.csv",
    "publication_data/revision_outputs/full_data_pairwise_audit.csv",
]

FORBIDDEN_NAME_PARTS = [
    "Revision_Handoff",
    "Revision_Action_Plan",
    "Response_to_Reviewers",
    "Decision_Record",
    "reviewer_response",
]

PYTHON_PATHS = [
    "src/small_data_mof_benchmark_pipeline.py",
    "figure_regeneration/draw_all_figures.py",
    "scripts/revision/generate_revision_tables.py",
    "scripts/patch_publication_threshold.py",
    "scripts/update_zenodo_doi.py",
    "scripts/validate_repository.py",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def check_required(root: Path) -> None:
    missing = [p for p in EXPECTED_REQUIRED if not (root / p).exists()]
    if missing:
        fail("Missing required files:\n" + "\n".join(missing))


def check_duplicate_readmes(root: Path) -> None:
    root_readmes = [p.name for p in root.iterdir() if p.is_file() and p.name.lower() == "readme.md"]
    if root_readmes != ["README.md"]:
        fail(
            "Keep exactly one root README named README.md. "
            f"Found: {root_readmes}. Delete README.MD or other case duplicates."
        )


def check_forbidden_files(root: Path) -> None:
    offenders = []
    for p in root.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        if any(token.lower() in p.name.lower() for token in FORBIDDEN_NAME_PARTS):
            offenders.append(str(p.relative_to(root)))
    if offenders:
        fail("Internal reviewer/planning files must not be public:\n" + "\n".join(offenders))


def check_manifest(root: Path) -> None:
    pub = root / "publication_data"
    manifest = pd.read_csv(pub / "metadata" / "file_manifest.csv")
    required_cols = {"relative_path", "size_bytes", "sha256"}
    if not required_cols.issubset(manifest.columns):
        fail(f"Manifest lacks columns {sorted(required_cols - set(manifest.columns))}")
    for row in manifest.itertuples(index=False):
        path = pub / row.relative_path
        if not path.exists():
            fail(f"Manifest entry missing on disk: {row.relative_path}")
        raw = path.read_bytes()

        # The publication manifest records canonical LF bytes.
        # Git may check text files out as CRLF on Windows, so normalize
        # line endings before checksum and byte-size validation.
        canonical = raw.replace(b"\r\n", b"\n") if b"\x00" not in raw else raw

        digest = hashlib.sha256(canonical).hexdigest()
        if digest != row.sha256:
            fail(f"Checksum mismatch: {row.relative_path}")
        if len(canonical) != int(row.size_bytes):
            fail(f"Size mismatch: {row.relative_path}")
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            if not pd.isna(row.rows) and int(row.rows) != len(frame):
                fail(f"Row-count mismatch: {row.relative_path}")
            if not pd.isna(row.columns) and int(row.columns) != len(frame.columns):
                fail(f"Column-count mismatch: {row.relative_path}")


def check_csvs(root: Path) -> None:
    for path in root.joinpath("publication_data").rglob("*.csv"):
        try:
            pd.read_csv(path)
        except Exception as exc:
            fail(f"Unreadable CSV {path.relative_to(root)}: {exc}")


def check_revision_values(root: Path) -> None:
    out = root / "publication_data" / "revision_outputs"

    per_target = pd.read_csv(out / "per_target_maturity_summary.csv")
    got = dict(zip(per_target["target_col"], per_target["first_n_satisfying_base_rule"]))
    expected = {
        "uptake(mmol/g) CO2 at 0.015 bar": math.nan,
        "uptake(mmol/g) CO2 at 0.15 bar": 5000.0,
        "uptake(mmol/g) methane at 5.8 bar": 2000.0,
        "uptake(mmol/g) methane at 65 bar": 10000.0,
    }
    if set(got) != set(expected):
        fail(f"Unexpected target set in per-target summary: {set(got)}")
    for target, value in expected.items():
        actual = got[target]
        if math.isnan(value):
            if not pd.isna(actual):
                fail(f"Expected no crossing for {target}, found {actual}")
        elif float(actual) != value:
            fail(f"Unexpected crossing for {target}: {actual} != {value}")

    partitions = pd.read_csv(out / "test_partition_sensitivity.csv")
    part_map = dict(zip(partitions["test_seed"].astype(int), partitions["first_n_satisfying_base_rule"]))
    if part_map != {17: 5000.0, 29: 10000.0, 47: 2000.0, 71: 5000.0}:
        fail(f"Unexpected test-partition crossings: {part_map}")

    grid = pd.read_csv(out / "cutoff_sensitivity_grid.csv")
    if len(grid) != 81:
        fail(f"Cutoff grid must contain 81 rows, found {len(grid)}")
    counts = grid["first_pass_n"].value_counts(dropna=False)
    if int(counts.get(5000.0, 0)) != 36:
        fail(f"Expected 36 grid crossings at 5000; found {counts.to_dict()}")
    if int(counts.get(10000.0, 0)) != 18:
        fail(f"Expected 18 grid crossings at 10000; found {counts.to_dict()}")
    if int(grid["first_pass_n"].isna().sum()) != 27:
        fail(f"Expected 27 unsatisfied grid settings; found {grid['first_pass_n'].isna().sum()}")

    alt = pd.read_csv(out / "test_partition_alt_seed_curves.csv")
    max_spread = float(alt["rmse_range_across_seeds"].max())
    if not math.isclose(max_spread, 0.0103685802720873, rel_tol=0, abs_tol=1e-12):
        fail(f"Unexpected maximum test-partition RMSE spread: {max_spread}")

    rf_hgb = pd.read_csv(out / "rf_vs_hgb_pairwise.csv")
    if len(rf_hgb) != 4 or not (rf_hgb["P(rf_beats_hgb)"] == 1.0).all():
        fail("RF-vs-HGB summary must contain four seeds with P(RF beats HGB)=1.0.")

    audit = pd.read_csv(out / "full_data_pairwise_audit.csv")
    if len(audit) != 1680:
        fail(f"Expected 1680 ordered off-diagonal pairs; found {len(audit)}")
    if int(audit["ridge_only"].astype(bool).sum()) != 84:
        fail(f"Expected 84 ordered Ridge-only pairs; found {audit['ridge_only'].astype(bool).sum()}")
    if int(audit["anomalous"].astype(bool).sum()) != 0:
        fail("Ridge-only pairwise audit contains anomalies.")

    fixed = pd.read_csv(out / "fixed_k_screening_summary.csv")
    if list(fixed["n_train"]) != sorted(fixed["n_train"].tolist()):
        fail("Fixed-k summary must be sorted by training size.")
    full = fixed[fixed["n_train"] == 210995]
    if len(full) != 1:
        fail("Fixed-k summary lacks the full-data row.")
    full = full.iloc[0]
    if not math.isclose(float(full["topk_overlap_k50_mean"]), 0.478, abs_tol=1e-12):
        fail("Unexpected full-data top-50 overlap.")
    if not math.isclose(float(full["topk_overlap_k500_mean"]), 0.4574, abs_tol=1e-12):
        fail("Unexpected full-data top-500 overlap.")


def check_config(root: Path) -> None:
    config = yaml.safe_load((root / "config" / "publication_v1.yml").read_text(encoding="utf-8"))
    topk = float(config["composite_rule"]["top5_overlap_sd_max"])
    if topk != 0.01:
        fail(f"Publication config top-5 overlap SD threshold must be 0.01, found {topk}")

    pipeline = root / "src" / "small_data_mof_benchmark_pipeline.py"
    if pipeline.exists():
        text = pipeline.read_text(encoding="utf-8")
        matches = re.findall(
            r"stability_topk_std_threshold\s*(?::\s*float\s*)?=\s*([0-9.]+)",
            text,
        )
        if len(matches) != 1:
            fail(
                "Expected exactly one stability_topk_std_threshold assignment "
                f"in the pipeline; found {matches}"
            )
        if float(matches[0]) != 0.01:
            fail(
                "Pipeline/manuscript mismatch: "
                f"stability_topk_std_threshold={matches[0]}, expected 0.01."
            )


def check_metadata(root: Path) -> None:
    cff = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    for key in ["cff-version", "message", "title", "authors", "version", "repository-code"]:
        if key not in cff:
            fail(f"CITATION.cff missing key: {key}")
    if cff["cff-version"] != "1.2.0":
        fail("CITATION.cff must use schema version 1.2.0.")
    json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))


def check_python(root: Path) -> None:
    for rel in PYTHON_PATHS:
        path = root / rel
        if path.exists():
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                fail(f"Python syntax error in {rel}: {exc}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checks = [
        ("required files", check_required),
        ("single README", check_duplicate_readmes),
        ("no internal files", check_forbidden_files),
        ("metadata", check_metadata),
        ("publication manifest", check_manifest),
        ("CSV readability", check_csvs),
        ("revision values", check_revision_values),
        ("publication configuration", check_config),
        ("Python syntax", check_python),
    ]
    for name, function in checks:
        function(root)
        print(f"PASS: {name}")
    print("\nRepository validation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nVALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
