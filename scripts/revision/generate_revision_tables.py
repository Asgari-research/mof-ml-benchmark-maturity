#!/usr/bin/env python3
"""Regenerate post-hoc revision tables from the public publication-data archive.

No model is trained by this script. It reads existing seed-level and aggregated
benchmark outputs and deterministically writes reviewer-support CSV files.

Usage
-----
python scripts/revision/generate_revision_tables.py
python scripts/revision/generate_revision_tables.py --root /path/to/repository
"""
from __future__ import annotations

import argparse
import itertools
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MAIN_TARGET = "uptake(mmol/g) CO2 at 0.15 bar"
FULL_N = 210_995
BASE_CUTS = {
    "top1": 0.80,
    "rank": 0.90,
    "topk_sd": 0.01,
    "enrich_sd": 0.20,
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def intervals_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    return not (a_hi < b_lo or b_hi < a_lo)


def load_inputs(root: Path) -> dict[str, object]:
    si = root / "publication_data" / "table_source_data" / "si"
    fig_si = root / "publication_data" / "figure_source_data" / "si"
    required = {
        "aggregated": si / "si_aggregated_performance.csv",
        "ranking": si / "si_ranking_stability.csv",
        "screening": si / "si_screening_reproducibility.csv",
        "jobs": si / "si_all_job_metrics.csv",
        "alt_curve": fig_si / "si1_alt_test_robustness.csv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required public inputs:\n" + "\n".join(missing))
    return {
        "aggregated": pd.read_csv(required["aggregated"]),
        "ranking": pd.read_csv(required["ranking"]),
        "screening": pd.read_csv(required["screening"]),
        "jobs": pd.read_csv(required["jobs"]),
        "alt_curve": pd.read_csv(required["alt_curve"]),
        "pairwise_dir": si,
    }


def build_composite_rows(
    aggregated: pd.DataFrame,
    ranking: pd.DataFrame,
    screening: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["suite_name", "target_col", "test_seed", "n_train"]
    for keys, sub in aggregated.groupby(group_cols, sort=True):
        suite, target, test_seed, n_train = keys
        sub = sub.sort_values(["rmse_mean", "method_label"], kind="mergesort")
        if len(sub) < 2:
            continue
        best = sub.iloc[0]
        second = sub.iloc[1]

        r = ranking[
            (ranking["suite_name"] == suite)
            & (ranking["target_col"] == target)
            & (ranking["test_seed"] == test_seed)
            & (ranking["n_train"] == n_train)
        ]
        s = screening[
            (screening["suite_name"] == suite)
            & (screening["target_col"] == target)
            & (screening["test_seed"] == test_seed)
            & (screening["n_train"] == n_train)
            & (screening["method_label"] == best["method_label"])
        ]
        if len(r) != 1 or len(s) != 1:
            raise ValueError(
                f"Expected one ranking and one screening row for {keys}; "
                f"found ranking={len(r)}, screening={len(s)}."
            )
        r0 = r.iloc[0]
        s0 = s.iloc[0]
        overlap = intervals_overlap(
            float(best["rmse_ci_low"]),
            float(best["rmse_ci_high"]),
            float(second["rmse_ci_low"]),
            float(second["rmse_ci_high"]),
        )
        conditions = bool(
            overlap
            and float(r0["top1_consensus_probability"]) >= BASE_CUTS["top1"]
            and float(r0["mean_rank_spearman_vs_full"]) >= BASE_CUTS["rank"]
            and float(s0["topk_overlap_frac_0p050_std"]) <= BASE_CUTS["topk_sd"]
            and float(s0["elite_enrichment_0p050_std"]) <= BASE_CUTS["enrich_sd"]
        )
        rows.append(
            {
                "suite_name": suite,
                "target_col": target,
                "test_seed": int(test_seed),
                "n_train": int(n_train),
                "best_method_label": best["method_label"],
                "best_rmse_mean": float(best["rmse_mean"]),
                "second_best_method_label": second["method_label"],
                "top2_rmse_ci_overlap": overlap,
                "top1_consensus_probability": float(r0["top1_consensus_probability"]),
                "mean_rank_spearman_vs_full": float(r0["mean_rank_spearman_vs_full"]),
                "topk_overlap_frac_0p050_std_best": float(
                    s0["topk_overlap_frac_0p050_std"]
                ),
                "elite_enrichment_0p050_std_best": float(
                    s0["elite_enrichment_0p050_std"]
                ),
                "conditions_met_here": conditions,
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def first_crossing(sub: pd.DataFrame, flag_col: str = "conditions_met_here") -> float:
    passed = sub.loc[sub[flag_col].astype(bool)].sort_values("n_train")
    return float(passed["n_train"].iloc[0]) if len(passed) else math.nan


def make_per_target(composite: pd.DataFrame) -> pd.DataFrame:
    primary = composite[composite["suite_name"] == "primary"]
    rows = []
    for target, sub in primary.groupby("target_col", sort=True):
        rows.append(
            {
                "target_col": target,
                "first_n_satisfying_base_rule": first_crossing(sub),
                "n_training_sizes": int(sub["n_train"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def make_partition_sensitivity(composite: pd.DataFrame) -> pd.DataFrame:
    main = composite[composite["target_col"] == MAIN_TARGET]
    rows = []
    for seed, sub in main.groupby("test_seed", sort=True):
        rows.append(
            {
                "test_seed": int(seed),
                "first_n_satisfying_base_rule": first_crossing(sub),
                "n_rows": int(len(sub)),
                "n_repeats": int(
                    10 if int(seed) == 17 else 5
                ),
            }
        )
    return pd.DataFrame(rows)


def make_cutoff_grid(composite: pd.DataFrame) -> pd.DataFrame:
    main = composite[
        (composite["target_col"] == MAIN_TARGET)
        & (composite["test_seed"] == 17)
    ].sort_values("n_train")
    rows = []
    for top1, rank_cut, topk_sd, enrich_sd in itertools.product(
        [0.70, 0.80, 0.90],
        [0.85, 0.90, 0.95],
        [0.01, 0.02, 0.05],
        [0.10, 0.20, 0.30],
    ):
        flag = (
            main["top2_rmse_ci_overlap"].astype(bool)
            & (main["top1_consensus_probability"] >= top1)
            & (main["mean_rank_spearman_vs_full"] >= rank_cut)
            & (main["topk_overlap_frac_0p050_std_best"] <= topk_sd)
            & (main["elite_enrichment_0p050_std_best"] <= enrich_sd)
        )
        passed = main.loc[flag, "n_train"]
        rows.append(
            {
                "top1_cut": top1,
                "rank_cut": rank_cut,
                "topk_sd_cut": topk_sd,
                "enrich_sd_cut": enrich_sd,
                "first_pass_n": float(passed.iloc[0]) if len(passed) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def make_top1_uncertainty(jobs: pd.DataFrame) -> pd.DataFrame:
    required = [
        "suite_name", "target_col", "test_seed", "n_train",
        "subsample_seed", "descriptor_family", "model_name", "rmse",
    ]
    missing = [c for c in required if c not in jobs.columns]
    if missing:
        raise ValueError(f"Job table missing columns: {missing}")
    df = jobs.copy()
    df["method"] = (
        df["descriptor_family"].astype(str) + " | " + df["model_name"].astype(str)
    )
    rows = []
    group_cols = ["suite_name", "target_col", "test_seed", "n_train"]
    for keys, sub in df.groupby(group_cols, sort=True):
        winners = (
            sub.sort_values(["subsample_seed", "rmse", "method"], kind="mergesort")
            .groupby("subsample_seed", sort=True)
            .head(1)["method"]
        )
        counts = winners.value_counts()
        if counts.empty:
            continue
        modal = str(counts.index[0])
        k = int(counts.iloc[0])
        n = int(len(winners))
        lo, hi = wilson_ci(k, n)
        rows.append(
            {
                "suite_name": keys[0],
                "target_col": keys[1],
                "test_seed": int(keys[2]),
                "n_train": int(keys[3]),
                "modal_winner": modal,
                "modal_count": k,
                "n_seeds": n,
                "top1_consensus": k / n,
                "wilson_lo": lo,
                "wilson_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def make_fixed_k(jobs: pd.DataFrame) -> pd.DataFrame:
    main = jobs[
        (jobs["suite_name"] == "primary")
        & (jobs["target_col"] == MAIN_TARGET)
        & (jobs["test_seed"] == 17)
    ].copy()
    means = (
        main.groupby(["n_train", "descriptor_family", "model_name"], as_index=False)
        ["rmse"].mean()
    )
    best = (
        means.sort_values(
            ["n_train", "rmse", "descriptor_family", "model_name"],
            kind="mergesort",
        )
        .groupby("n_train", sort=True)
        .head(1)
    )
    metrics = [
        c for c in main.columns
        if c.startswith("topk_overlap_k") or c.startswith("elite_enrichment_k")
    ]
    rows = []
    for _, b in best.iterrows():
        sub = main[
            (main["n_train"] == b["n_train"])
            & (main["descriptor_family"] == b["descriptor_family"])
            & (main["model_name"] == b["model_name"])
        ]
        row: dict[str, object] = {
            "n_train": int(b["n_train"]),
            "best_method": f"{b['descriptor_family']} | {b['model_name']}",
        }
        for c in metrics:
            row[c + "_mean"] = float(sub[c].mean())
            row[c + "_sd"] = float(sub[c].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_train").reset_index(drop=True)


def pairwise_files(pairwise_dir: Path) -> list[Path]:
    return sorted(pairwise_dir.glob("pairwise_superiority__*__n_210995.csv"))


def parse_test_seed(name: str) -> int:
    m = re.search(r"__testseed_(\d+)__", name)
    if not m:
        raise ValueError(f"Cannot parse test seed from {name}")
    return int(m.group(1))


def make_rf_hgb(pairwise_dir: Path) -> pd.DataFrame:
    rows = []
    for p in pairwise_files(pairwise_dir):
        if "CO2_at_0.15_bar" not in p.name:
            continue
        frame = pd.read_csv(p).set_index("method_label")
        rf = "geometry_plus_topology | rf"
        hgb = "geometry_plus_topology | hgb"
        if rf not in frame.index or hgb not in frame.columns:
            raise ValueError(f"Expected RF/HGB labels not found in {p.name}")
        rows.append(
            {
                "test_seed": parse_test_seed(p.name),
                "source_file": p.name,
                "P(rf_beats_hgb)": float(frame.loc[rf, hgb]),
                "P(hgb_beats_rf)": float(frame.loc[hgb, rf]),
            }
        )
    return pd.DataFrame(rows).sort_values("test_seed").reset_index(drop=True)


def make_pairwise_audit(pairwise_dir: Path) -> pd.DataFrame:
    rows = []
    for p in pairwise_files(pairwise_dir):
        frame = pd.read_csv(p).set_index("method_label")
        for method_a in frame.index:
            for method_b in frame.columns:
                if method_a == method_b:
                    continue
                prob = float(frame.loc[method_a, method_b])
                ridge_only = (
                    method_a.endswith(" | ridge") and method_b.endswith(" | ridge")
                )
                anomalous = ridge_only and prob not in (0.0, 0.5, 1.0)
                rows.append(
                    {
                        "method_a": method_a,
                        "method_b": method_b,
                        "prob": prob,
                        "source_file": p.name,
                        "ridge_only": ridge_only,
                        "anomalous": anomalous,
                    }
                )
    return pd.DataFrame(rows)


def make_alt_curves(aggregated: pd.DataFrame) -> pd.DataFrame:
    main = aggregated[aggregated["target_col"] == MAIN_TARGET]
    best = (
        main.groupby(["test_seed", "n_train"], as_index=False)["rmse_mean"].min()
        .pivot(index="n_train", columns="test_seed", values="rmse_mean")
        .sort_index()
    )
    required = [17, 29, 47, 71]
    missing = [s for s in required if s not in best.columns]
    if missing:
        raise ValueError(f"Alternative-partition data missing test seeds {missing}")
    out = pd.DataFrame({"n_train": best.index.astype(int)})
    for seed in required:
        out[f"best_rmse_seed_{seed}"] = best[seed].to_numpy()
    cols = [f"best_rmse_seed_{s}" for s in required]
    out["rmse_range_across_seeds"] = out[cols].max(axis=1) - out[cols].min(axis=1)
    out["rmse_mean_across_seeds"] = out[cols].mean(axis=1)
    return out.reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    print(f"wrote {path.relative_to(path.parents[2])}: {len(df)} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (default: inferred from script location).",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    data = load_inputs(root)
    out = root / "publication_data" / "revision_outputs"
    out.mkdir(parents=True, exist_ok=True)

    composite = build_composite_rows(
        data["aggregated"], data["ranking"], data["screening"]
    )
    write_csv(composite, out / "composite_rule_by_target_and_partition.csv")
    write_csv(make_per_target(composite), out / "per_target_maturity_summary.csv")
    write_csv(
        make_partition_sensitivity(composite),
        out / "test_partition_sensitivity.csv",
    )
    write_csv(make_cutoff_grid(composite), out / "cutoff_sensitivity_grid.csv")
    write_csv(
        make_top1_uncertainty(data["jobs"]),
        out / "top1_consensus_uncertainty.csv",
    )
    write_csv(make_fixed_k(data["jobs"]), out / "fixed_k_screening_summary.csv")
    write_csv(
        make_rf_hgb(data["pairwise_dir"]),
        out / "rf_vs_hgb_pairwise.csv",
    )
    write_csv(
        make_pairwise_audit(data["pairwise_dir"]),
        out / "full_data_pairwise_audit.csv",
    )
    write_csv(
        make_alt_curves(data["aggregated"]),
        out / "test_partition_alt_seed_curves.csv",
    )


if __name__ == "__main__":
    main()
