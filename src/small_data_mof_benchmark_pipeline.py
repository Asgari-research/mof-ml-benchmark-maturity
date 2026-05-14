
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small-Data MOF Benchmark Design pipeline
=======================================

This standalone Python script implements a full, resume-friendly benchmarking
pipeline for the "Small-Data MOF Benchmark Design" project.

Scientific question
-------------------
How much training data is needed before a MOF benchmark result becomes stable
enough to support a scientific claim?

Core design choices implemented here
------------------------------------
1. A fixed external test set is created and held constant.
2. Nested subsamples of the remaining training pool are created.
3. The same descriptor-family / model combinations are trained at each sample
   size and random seed under identical preprocessing.
4. Performance, uncertainty, ranking stability, screening reproducibility,
   pairwise probability-of-superiority, and interpretation stability are all
   quantified and saved.
5. The script is checkpointed at the job level, so interrupted runs can be
   resumed without losing completed work.

Expected input files
--------------------
Place the following CSV files in the SAME folder as this script before running:
    - clean_data.csv
    - geometric_properties.csv  (optional; used only for consistency checks)

Outputs
-------
The script creates a project-style output tree containing:
    - logs
    - processed data
    - per-job checkpoints
    - per-job prediction files
    - compiled metrics tables
    - manuscript figures (PNG/PDF)
    - SI figures (PNG/PDF)
    - companion CSV / pickle figure-data exports containing the exact plotted values
    - CSV / pickle / LaTeX tables
    - summary JSON manifests

Visual Studio Code friendliness
-------------------------------
The script is intentionally written as a single, heavily commented file with
clear sections and helper functions, so it is easy to inspect, run, and modify
inside Visual Studio Code.

Author note
-----------
The code below is designed to be scientifically transparent rather than overly
"clever". Many intermediate files are saved on purpose so that later manuscript
editing or figure updates do NOT require re-running the full pipeline.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import textwrap
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# SECTION 1. CONFIGURATION
# =============================================================================

@dataclass
class ProjectConfig:
    """
    All main configuration settings live here so the script can be edited
    comfortably in VS Code without hunting through the file.
    """
    # -------------------------------
    # File names expected beside script
    # -------------------------------
    clean_data_filename: str = "clean_data.csv"
    geometric_properties_filename: str = "geometric_properties.csv"

    # -------------------------------
    # Output root
    # -------------------------------
    output_root: str = "small_data_mof_benchmark_outputs"

    # -------------------------------
    # Main scientific choices
    # -------------------------------
    # This follows the LaTeX methodology / manuscript blueprint default.
    main_target: str = "uptake(mmol/g) CO2 at 0.15 bar"

    # Other targets automatically go to robustness / SI analyses.
    all_targets: Tuple[str, ...] = (
        "uptake(mmol/g) CO2 at 0.015 bar",
        "uptake(mmol/g) CO2 at 0.15 bar",
        "uptake(mmol/g) methane at 5.8 bar",
        "uptake(mmol/g) methane at 65 bar",
    )

    id_col: str = "filename"
    topology_col: str = "Crystalnet"

    # -------------------------------
    # External test split controls
    # -------------------------------
    test_fraction: float = 0.20
    primary_test_seed: int = 17
    alternative_test_seeds: Tuple[int, ...] = (29, 47, 71)

    # -------------------------------
    # Nested sample sizes
    # The script will automatically keep only those <= training-pool size,
    # then append "full" = all remaining training data.
    # -------------------------------
    requested_sample_sizes: Tuple[int, ...] = (500, 1000, 2000, 5000, 10000, 20000, 40000)

    # -------------------------------
    # Repeats / seeds
    # -------------------------------
    main_target_seeds: int = 10
    si_target_seeds: int = 5

    # Separate seeds used for robustness alt-test-set experiments.
    robustness_seeds: int = 5

    # -------------------------------
    # Screening metrics
    # -------------------------------
    # top-k can be defined by fraction or by absolute counts.
    # We save both to give the manuscript flexibility.
    topk_fraction: float = 0.05
    topk_absolute_list: Tuple[int, ...] = (50, 100, 250, 500)

    # -------------------------------
    # Descriptor family controls
    # -------------------------------
    topology_min_count: int = 50  # rare labels grouped into "__OTHER__"

    # -------------------------------
    # Model controls
    # These are intentionally fixed defaults rather than a hyperparameter search,
    # because the paper's focus is benchmark maturity and data sufficiency,
    # not leaderboard optimization.
    # -------------------------------
    ridge_alpha: float = 1.0

    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = 16
    rf_min_samples_leaf: int = 2
    rf_n_jobs: int = 2

    hgb_learning_rate: float = 0.05
    hgb_max_depth: int = 8
    hgb_max_iter: int = 250
    hgb_l2_regularization: float = 0.0

    mlp_hidden_layer_sizes: Tuple[int, ...] = (128, 64)
    mlp_alpha: float = 1e-4
    mlp_learning_rate_init: float = 5e-4
    mlp_max_iter: int = 300
    mlp_early_stopping: bool = True

    # -------------------------------
    # Stable conclusion threshold logic
    # These are pragmatic thresholds used to build the paper's guidance table.
    # -------------------------------
    stability_top1_consensus_threshold: float = 0.80
    stability_rankcorr_threshold: float = 0.90
    stability_topk_std_threshold: float = 0.05
    stability_elite_std_threshold: float = 0.20

    # -------------------------------
    # Figure 5 pairwise superiority selected sizes
    # "full" will be resolved later.
    # -------------------------------
    pairwise_selected_sizes: Tuple[int, ...] = (1000, 10000)

    # -------------------------------
    # Figure 6 interpretation stability
    # Selected sizes for feature-effect convergence.
    # "full" will be resolved later.
    # -------------------------------
    feature_effect_sizes: Tuple[int, ...] = (500, 2000, 10000)

    # Number of seeds to use for Figure 6 reruns (to control cost).
    feature_effect_seeds: int = 5

    # Number of top features to display in rank convergence panels.
    feature_effect_topn: int = 15

    # -------------------------------
    # Resume behaviour
    # -------------------------------
    resume_if_available: bool = True
    overwrite_existing_figures: bool = True
    overwrite_existing_tables: bool = True

    # -------------------------------
    # Miscellaneous
    # -------------------------------
    random_state_base: int = 12345
    verbose_every_n_jobs: int = 20

    # -------------------------------
    # Figure / visualization controls
    # -------------------------------
    top_methods_to_highlight_main_curve: int = 6
    top_methods_to_highlight_si_curve: int = 6
    export_svg_figures: bool = False


CONFIG = ProjectConfig()


# =============================================================================
# SECTION 2. BASIC HELPERS
# =============================================================================

def slugify(text: str) -> str:
    """
    Convert a human-readable label into a filesystem-friendly slug.
    """
    text = str(text)
    text = text.replace("%", "pct")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def ensure_dir(path: Path) -> Path:
    """
    Create a directory if it does not already exist and return the same Path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj, path: Path) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def load_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_dataframe(df: pd.DataFrame, csv_path: Path, pkl_path: Optional[Path] = None) -> None:
    """
    Save a DataFrame in both CSV and compressed pickle formats.
    """
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)
    if pkl_path is not None:
        save_pickle(df, pkl_path)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def latex_escape(text: str) -> str:
    """
    Minimal LaTeX escaping for table export.
    """
    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def df_to_latex_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    index: bool = False,
    float_format: str = "{:.4f}",
) -> None:
    """
    Save a DataFrame as a simple LaTeX table file.
    """
    ensure_dir(path.parent)

    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == object:
            df_copy[col] = df_copy[col].map(latex_escape)

    latex_str = df_copy.to_latex(
        index=index,
        escape=False,
        caption=caption,
        label=label,
        float_format=lambda x: float_format.format(x) if isinstance(x, (float, np.floating)) else str(x),
        longtable=False,
    )
    write_text(path, latex_str)


def ci95(series: pd.Series) -> Tuple[float, float, float]:
    """
    Return mean, lower CI, upper CI using normal approximation.
    This is sufficient for manuscript-level uncertainty summaries.
    """
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    mean_val = float(s.mean())
    if len(s) == 1:
        return mean_val, mean_val, mean_val
    se = float(s.std(ddof=1)) / math.sqrt(len(s))
    delta = 1.96 * se
    return mean_val, mean_val - delta, mean_val + delta


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman correlation that safely handles constant vectors.
    """
    try:
        corr, _ = spearmanr(y_true, y_pred)
        if pd.isna(corr):
            return np.nan
        return float(corr)
    except Exception:
        return np.nan


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_stratify_bins(y: pd.Series, n_bins: int = 10) -> pd.Series:
    """
    Create stratification bins for a continuous target.

    This helps keep the fixed external test set reasonably representative.
    """
    y = pd.Series(y).astype(float)
    unique_count = y.nunique(dropna=True)
    bins = min(n_bins, unique_count)
    if bins < 2:
        return pd.Series(np.zeros(len(y), dtype=int), index=y.index)
    try:
        return pd.qcut(y.rank(method="first"), q=bins, labels=False, duplicates="drop")
    except Exception:
        return pd.Series(np.zeros(len(y), dtype=int), index=y.index)


def top_k_count_from_fraction(n: int, fraction: float) -> int:
    """
    Convert a top-k fraction into an absolute count with sensible bounds.
    """
    return max(1, int(round(n * fraction)))


def topk_overlap_fraction(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Overlap between predicted top-k and true top-k, normalized to [0,1].
    """
    k = min(k, len(y_true))
    pred_top = np.argsort(-y_pred)[:k]
    true_top = np.argsort(-y_true)[:k]
    overlap = len(set(pred_top).intersection(set(true_top)))
    return overlap / float(k)


def elite_enrichment_factor(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Enrichment factor for recovering the true top-k (elite set) when selecting
    the predicted top-k set.

    Since the selected set size equals the elite set size, the random baseline
    expected hit rate is k/N. Enrichment factor = observed precision / random precision.
    """
    n = len(y_true)
    k = min(k, n)
    pred_top = np.argsort(-y_pred)[:k]
    true_top = set(np.argsort(-y_true)[:k])
    hits = sum(i in true_top for i in pred_top)
    observed_precision = hits / float(k)
    random_precision = k / float(n)
    if random_precision == 0:
        return np.nan
    return observed_precision / random_precision


def jaccard_similarity(items_a: Iterable, items_b: Iterable) -> float:
    """
    Jaccard similarity between two iterables.
    """
    a, b = set(items_a), set(items_b)
    union = a.union(b)
    if len(union) == 0:
        return np.nan
    return len(a.intersection(b)) / float(len(union))


def current_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_plot_style() -> None:
    """Apply a cleaner, publication-oriented matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.8,
        "axes.titlesize": 16,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "font.size": 11.5,
        "figure.dpi": 140,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,
    })


def pretty_method_label(label: str) -> str:
    """Human-friendlier method label for legends and tables."""
    left, right = [x.strip() for x in str(label).split("|")] if "|" in str(label) else (str(label), "")
    left = left.replace("geometry_plus_topology", "Geometry + topology")
    left = left.replace("enriched_interpretable", "Enriched interpretable")
    left = left.replace("geometry_only", "Geometry only")
    left = left.replace("topology_only", "Topology only")
    right = right.upper()
    return f"{left} | {right}" if right else left


def build_method_style_map(methods: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """Deterministically assign colors, linestyles, and markers to methods."""
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Dark2").colors)
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
    style_map = {}
    for i, method in enumerate(methods):
        style_map[method] = {
            "color": colors[i % len(colors)],
            "linestyle": linestyles[(i // len(colors)) % len(linestyles)],
            "marker": markers[i % len(markers)],
        }
    return style_map


def add_panel_label(ax, label: str) -> None:
    ax.text(0.01, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=13)


def coerce_numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =============================================================================
# SECTION 3. OUTPUT PATH MANAGEMENT
# =============================================================================

def build_output_paths(root: Path) -> Dict[str, Path]:
    """
    Create and return a dictionary of important output folders.
    """
    paths = {
        "root": root,
        "logs": root / "logs",
        "checkpoints": root / "checkpoints",
        "job_metrics": root / "checkpoints" / "job_metrics",
        "job_predictions": root / "checkpoints" / "job_predictions",
        "job_feature_effects": root / "checkpoints" / "job_feature_effects",
        "split_cache": root / "checkpoints" / "split_cache",
        "manifests": root / "checkpoints" / "manifests",
        "data_processed": root / "data_processed",
        "results": root / "results",
        "metrics_tables": root / "results" / "metrics_tables",
        "performance_tables": root / "results" / "performance_tables",
        "ranking_tables": root / "results" / "ranking_tables",
        "screening_tables": root / "results" / "screening_tables",
        "pairwise_tables": root / "results" / "pairwise_tables",
        "feature_effect_tables": root / "results" / "feature_effect_tables",
        "additional_analysis_tables": root / "results" / "additional_analysis_tables",
        "manuscript_assets": root / "manuscript_assets",
        "main_figures": root / "manuscript_assets" / "figures_main",
        "main_figure_data": root / "manuscript_assets" / "figure_data_main",
        "main_tables": root / "manuscript_assets" / "tables_main",
        "main_figures_extra": root / "manuscript_assets" / "figures_additional",
        "supplementary_assets": root / "supplementary_assets",
        "si_figures": root / "supplementary_assets" / "figures_si",
        "si_figure_data": root / "supplementary_assets" / "figure_data_si",
        "si_tables": root / "supplementary_assets" / "tables_si",
        "si_figures_extra": root / "supplementary_assets" / "figures_additional_si",
        "all_figure_data": root / "results" / "figure_data_all",
        "final_exports": root / "final_exports",
    }
    for p in paths.values():
        if isinstance(p, Path):
            ensure_dir(p)
    return paths


def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Configure a logger that writes both to screen and to file.
    """
    ensure_dir(log_dir)
    logger = logging.getLogger("small_data_mof_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_dir / "run_log.txt", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# =============================================================================
# SECTION 4. DATA LOADING, CLEANING, AND FEATURE ENGINEERING
# =============================================================================

def normalize_filename_series(s: pd.Series) -> pd.Series:
    """
    Normalize filenames so clean_data.csv and geometric_properties.csv can be compared.
    """
    s = s.astype(str).str.replace(".cif", "", regex=False)
    return s


def load_input_data(script_dir: Path, cfg: ProjectConfig, paths: Dict[str, Path], logger: logging.Logger) -> pd.DataFrame:
    """
    Load clean_data.csv, perform essential cleaning, engineer features, and save
    the processed master table for reuse.
    """
    processed_path = paths["data_processed"] / "processed_master_table.pkl.gz"
    processed_csv_path = paths["data_processed"] / "processed_master_table.csv"

    if cfg.resume_if_available and processed_path.exists():
        logger.info("Loading previously processed master table from checkpoint.")
        try:
            df = load_pickle(processed_path)
            return df
        except Exception:
            logger.warning("Processed master-table checkpoint could not be read. Rebuilding it from the raw CSV.")

    clean_path = script_dir / cfg.clean_data_filename
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {clean_path}\n"
            f"Please place {cfg.clean_data_filename} in the same folder as this script."
        )

    logger.info("Reading clean_data.csv ...")
    df = pd.read_csv(clean_path)

    # Remove accidental index column if present.
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if cfg.id_col not in df.columns:
        raise KeyError(f"ID column '{cfg.id_col}' not found in clean_data.csv")

    # Normalize filenames for consistency.
    df[cfg.id_col] = normalize_filename_series(df[cfg.id_col])

    # Ensure topology exists.
    if cfg.topology_col not in df.columns:
        raise KeyError(f"Topology column '{cfg.topology_col}' not found in clean_data.csv")

    # Keep only rows with the required target columns present.
    required_columns = [cfg.id_col, cfg.topology_col] + list(cfg.all_targets)
    missing_required = [c for c in required_columns if c not in df.columns]
    if missing_required:
        raise KeyError(f"Missing required columns: {missing_required}")

    # Basic numeric coercion.
    for col in df.columns:
        if col not in [cfg.id_col, cfg.topology_col]:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    # Drop duplicate IDs if any.
    if df[cfg.id_col].duplicated().any():
        dup_count = int(df[cfg.id_col].duplicated().sum())
        logger.warning(f"Found {dup_count} duplicated IDs. Keeping first occurrence.")
        df = df.drop_duplicates(subset=[cfg.id_col], keep="first")

    # Engineer lightweight interpretable features.
    eps = 1e-9
    if "Di" in df.columns and "Df" in df.columns:
        df["lcd_pld_ratio"] = df["Di"] / (df["Df"] + eps)
        df["cavity_window_gap"] = df["Di"] - df["Df"]
        df["log_pld_plus1"] = np.log1p(np.clip(df["Df"], a_min=0, a_max=None))
        df["log_lcd_plus1"] = np.log1p(np.clip(df["Di"], a_min=0, a_max=None))
    if "ASA" in df.columns and "AVA" in df.columns:
        df["sa_pv_ratio"] = df["ASA"] / (df["AVA"] + eps)
    if "AVAf" in df.columns and "Density" in df.columns:
        df["vf_density_ratio"] = df["AVAf"] / (df["Density"] + eps)

    # Group rare topologies to keep modelling tractable and robust.
    topo_counts = df[cfg.topology_col].fillna("__MISSING__").astype(str).value_counts()
    common_topos = set(topo_counts[topo_counts >= cfg.topology_min_count].index)
    df["topology_label"] = (
        df[cfg.topology_col]
        .fillna("__MISSING__")
        .astype(str)
        .map(lambda x: x if x in common_topos else "__OTHER__")
    )

    # Save processed version.
    logger.info("Saving processed master table.")
    save_dataframe(df, processed_csv_path, processed_path)

    # Optional cross-check against geometric_properties.csv.
    geo_path = script_dir / cfg.geometric_properties_filename
    if geo_path.exists():
        try:
            crosscheck_geometric_properties(df, geo_path, cfg, paths, logger)
        except Exception as exc:
            logger.warning(f"Geometric-property cross-check failed, but pipeline will continue. Reason: {exc}")

    return df


def crosscheck_geometric_properties(
    df_clean: pd.DataFrame,
    geo_path: Path,
    cfg: ProjectConfig,
    paths: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """
    Optional consistency report comparing overlapping columns between clean_data.csv
    and geometric_properties.csv.

    This is not needed for the core paper but is useful documentation.
    """
    logger.info("Running optional consistency cross-check with geometric_properties.csv ...")
    df_geo = pd.read_csv(geo_path)

    unnamed_cols = [c for c in df_geo.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        df_geo = df_geo.drop(columns=unnamed_cols)

    if cfg.id_col not in df_geo.columns:
        if "filename" in df_geo.columns:
            pass
        else:
            logger.warning("No comparable filename column found in geometric_properties.csv. Skipping cross-check.")
            return

    df_geo[cfg.id_col] = normalize_filename_series(df_geo[cfg.id_col])

    overlap_cols = [
        c for c in ["UC_volume", "Density", "ASA", "AVA", "AVAf", "POAVA", "Di", "Df", "Dif"]
        if c in df_clean.columns and c in df_geo.columns
    ]
    if not overlap_cols:
        logger.warning("No overlapping geometric columns found for cross-check.")
        return

    merged = df_clean[[cfg.id_col] + overlap_cols].merge(
        df_geo[[cfg.id_col] + overlap_cols], on=cfg.id_col, suffixes=("_clean", "_geo"), how="inner"
    )

    report_rows = []
    for col in overlap_cols:
        x = merged[f"{col}_clean"].astype(float)
        y = merged[f"{col}_geo"].astype(float)
        abs_diff = (x - y).abs()
        row = {
            "column": col,
            "n_overlap": len(merged),
            "max_abs_diff": float(abs_diff.max()) if len(abs_diff) else np.nan,
            "mean_abs_diff": float(abs_diff.mean()) if len(abs_diff) else np.nan,
            "median_abs_diff": float(abs_diff.median()) if len(abs_diff) else np.nan,
        }
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows).sort_values("column")
    save_dataframe(
        report_df,
        paths["data_processed"] / "geometric_consistency_report.csv",
        paths["data_processed"] / "geometric_consistency_report.pkl.gz",
    )
    df_to_latex_table(
        report_df,
        paths["si_tables"] / "geometric_consistency_report.tex",
        caption="Optional consistency check between clean\\_data.csv and geometric\\_properties.csv.",
        label="tab:geom_consistency",
    )
    logger.info("Geometric-properties cross-check report saved.")


def get_descriptor_family_map(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """
    Define descriptor families used throughout the project.

    The families are intentionally interpretable and lightweight.
    """
    # Core geometry features repeatedly used in earlier ARC-MOF style work.
    geometry_core = [c for c in ["Density", "ASA", "AVA", "AVAf", "Df", "Di"] if c in df.columns]

    # Enriched interpretable family: still tabular and lightweight.
    enriched_numeric = [c for c in [
        "UC_volume", "Density", "ASA", "vASA", "NASA", "AVA", "AVAf", "POAVA", "Df", "Di", "Dif",
        "lcd_pld_ratio", "cavity_window_gap", "sa_pv_ratio", "vf_density_ratio",
        "log_pld_plus1", "log_lcd_plus1"
    ] if c in df.columns]

    families = {
        "geometry_only": {
            "numeric": geometry_core,
            "categorical": [],
            "description": "Six lightweight pore/geometry descriptors."
        },
        "enriched_interpretable": {
            "numeric": enriched_numeric,
            "categorical": [],
            "description": "Expanded lightweight geometry/interpretable descriptor set."
        },
        "topology_only": {
            "numeric": [],
            "categorical": ["topology_label"],
            "description": "Grouped topology label only."
        },
        "geometry_plus_topology": {
            "numeric": enriched_numeric,
            "categorical": ["topology_label"],
            "description": "Enriched interpretable descriptors plus grouped topology."
        },
    }
    return families


# =============================================================================
# SECTION 5. MODEL DEFINITIONS
# =============================================================================

def build_onehot_encoder():
    """
    Compatibility wrapper for OneHotEncoder across scikit-learn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_model_catalog(cfg: ProjectConfig) -> Dict[str, Dict]:
    """
    Fixed model definitions.

    These are intentionally moderate, stable choices. This avoids confounding
    the paper's sample-size question with aggressive hyperparameter searches.
    """
    return {
        "ridge": {
            "estimator": Ridge(alpha=cfg.ridge_alpha, random_state=None),
            "type": "linear",
            "description": f"Ridge regression (alpha={cfg.ridge_alpha}).",
            "needs_scaling": True,
        },
        "rf": {
            "estimator": RandomForestRegressor(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                min_samples_leaf=cfg.rf_min_samples_leaf,
                n_jobs=cfg.rf_n_jobs,
                random_state=None,
            ),
            "type": "tree",
            "description": (
                f"Random forest (n_estimators={cfg.rf_n_estimators}, "
                f"max_depth={cfg.rf_max_depth}, min_samples_leaf={cfg.rf_min_samples_leaf})."
            ),
            "needs_scaling": False,
        },
        "hgb": {
            "estimator": HistGradientBoostingRegressor(
                learning_rate=cfg.hgb_learning_rate,
                max_depth=cfg.hgb_max_depth,
                max_iter=cfg.hgb_max_iter,
                l2_regularization=cfg.hgb_l2_regularization,
                random_state=None,
            ),
            "type": "tree",
            "description": (
                f"HistGradientBoostingRegressor (learning_rate={cfg.hgb_learning_rate}, "
                f"max_depth={cfg.hgb_max_depth}, max_iter={cfg.hgb_max_iter})."
            ),
            "needs_scaling": False,
        },
        "mlp": {
            "estimator": MLPRegressor(
                hidden_layer_sizes=cfg.mlp_hidden_layer_sizes,
                alpha=cfg.mlp_alpha,
                learning_rate_init=cfg.mlp_learning_rate_init,
                max_iter=cfg.mlp_max_iter,
                early_stopping=cfg.mlp_early_stopping,
                random_state=None,
            ),
            "type": "neural",
            "description": (
                f"MLPRegressor (hidden_layers={cfg.mlp_hidden_layer_sizes}, alpha={cfg.mlp_alpha}, "
                f"learning_rate_init={cfg.mlp_learning_rate_init}, max_iter={cfg.mlp_max_iter})."
            ),
            "needs_scaling": True,
        },
    }


def build_pipeline(
    model_name: str,
    model_info: Dict,
    numeric_features: List[str],
    categorical_features: List[str],
    seed: int,
) -> Pipeline:
    """
    Build a preprocessing + model pipeline for a given descriptor family and model.
    """
    estimator = clone(model_info["estimator"])

    # Inject random_state into models that support it.
    if hasattr(estimator, "random_state"):
        setattr(estimator, "random_state", seed)

    transformers = []

    if numeric_features:
        if model_info["needs_scaling"]:
            transformers.append(("num", StandardScaler(), numeric_features))
        else:
            transformers.append(("num", "passthrough", numeric_features))

    if categorical_features:
        transformers.append(("cat", build_onehot_encoder(), categorical_features))

    if not transformers:
        raise ValueError("At least one numeric or categorical feature must be provided.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    return pipe


# =============================================================================
# SECTION 6. SPLITS AND JOB REGISTRY
# =============================================================================

def get_target_seed_count(target_name: str, cfg: ProjectConfig) -> int:
    return cfg.main_target_seeds if target_name == cfg.main_target else cfg.si_target_seeds


def build_fixed_external_split(
    df: pd.DataFrame,
    target_col: str,
    test_seed: int,
    cfg: ProjectConfig,
    cache_path: Path,
    logger: logging.Logger,
) -> Dict[str, np.ndarray]:
    """
    Build or load the fixed external train/test split for one target and one test seed.
    """
    if cfg.resume_if_available and cache_path.exists():
        try:
            return load_pickle(cache_path)
        except Exception:
            if cache_path.exists():
                cache_path.unlink()


    valid_mask = df[target_col].notna()
    idx_valid = np.where(valid_mask.values)[0]
    y_valid = df.loc[valid_mask, target_col].astype(float)
    strat_bins = regression_stratify_bins(y_valid, n_bins=10)

    idx_train, idx_test = train_test_split(
        idx_valid,
        test_size=cfg.test_fraction,
        random_state=test_seed,
        stratify=strat_bins,
    )

    split_info = {
        "idx_train_pool": np.array(sorted(idx_train), dtype=int),
        "idx_test": np.array(sorted(idx_test), dtype=int),
        "target_col": target_col,
        "test_seed": test_seed,
    }
    save_pickle(split_info, cache_path)
    logger.info(f"Cached fixed external split for target='{target_col}', test_seed={test_seed}.")
    return split_info


def build_nested_subsample_index_map(
    train_pool_indices: np.ndarray,
    requested_sizes: Sequence[int],
    seed: int,
    cache_path: Path,
    cfg: ProjectConfig,
) -> Dict[int, np.ndarray]:
    """
    Build nested subsamples by drawing one random ordering of the training pool
    and taking prefixes.

    This ensures n=500 is a subset of n=1000, which is a subset of n=2000, etc.
    """
    if cfg.resume_if_available and cache_path.exists():
        try:
            return load_pickle(cache_path)
        except Exception:
            if cache_path.exists():
                cache_path.unlink()


    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_pool_indices)

    nested = {}
    for n in requested_sizes:
        nested[int(n)] = np.array(sorted(perm[:n]), dtype=int)

    save_pickle(nested, cache_path)
    return nested


def resolve_sample_sizes(train_pool_size: int, cfg: ProjectConfig) -> List[int]:
    """
    Keep only feasible requested sizes and append the full training pool size.
    """
    sizes = sorted(set([int(n) for n in cfg.requested_sample_sizes if int(n) < int(train_pool_size)]))
    sizes.append(int(train_pool_size))
    return sizes


def build_job_id(
    suite_name: str,
    target_col: str,
    test_seed: int,
    subsample_seed: int,
    n_train: int,
    descriptor_family: str,
    model_name: str,
) -> str:
    """
    Build a unique job identifier for checkpointing.
    """
    parts = [
        suite_name,
        slugify(target_col),
        f"testseed_{test_seed}",
        f"subseed_{subsample_seed}",
        f"n_{n_train}",
        descriptor_family,
        model_name,
    ]
    return "__".join(parts)


def build_job_manifest(df: pd.DataFrame, cfg: ProjectConfig, logger: logging.Logger, paths: Dict[str, Path]) -> pd.DataFrame:
    """
    Create the full list of benchmark jobs.

    This manifest is central for:
    - transparent bookkeeping
    - resume capability
    - progress counting
    """
    manifest_path = paths["manifests"] / "job_manifest.csv"
    manifest_pkl = paths["manifests"] / "job_manifest.pkl.gz"

    if cfg.resume_if_available and manifest_path.exists() and manifest_pkl.exists():
        logger.info("Loading existing job manifest.")
        return load_pickle(manifest_pkl)

    families = get_descriptor_family_map(df)
    models = get_model_catalog(cfg)

    manifest_rows = []

    # 1) Primary fixed-test suite for all targets.
    for target_col in cfg.all_targets:
        target_seed_count = get_target_seed_count(target_col, cfg)
        split_cache_path = paths["split_cache"] / f"split__{slugify(target_col)}__testseed_{cfg.primary_test_seed}.pkl.gz"
        split_info = build_fixed_external_split(df, target_col, cfg.primary_test_seed, cfg, split_cache_path, logger)
        train_pool_size = len(split_info["idx_train_pool"])
        sample_sizes = resolve_sample_sizes(train_pool_size, cfg)

        logger.info(f"Adding primary-suite jobs for target='{target_col}' with train-pool size {train_pool_size:,}.")
        for subsample_seed in range(target_seed_count):
            for n_train in sample_sizes:
                for descriptor_family in families.keys():
                    for model_name in models.keys():
                        job_id = build_job_id(
                            suite_name="primary",
                            target_col=target_col,
                            test_seed=cfg.primary_test_seed,
                            subsample_seed=subsample_seed,
                            n_train=n_train,
                            descriptor_family=descriptor_family,
                            model_name=model_name,
                        )
                        manifest_rows.append({
                            "job_id": job_id,
                            "suite_name": "primary",
                            "target_col": target_col,
                            "is_main_target": target_col == cfg.main_target,
                            "test_seed": cfg.primary_test_seed,
                            "subsample_seed": subsample_seed,
                            "n_train": int(n_train),
                            "descriptor_family": descriptor_family,
                            "model_name": model_name,
                            "job_metrics_path": str(paths["job_metrics"] / f"{job_id}.json"),
                            "job_predictions_path": str(paths["job_predictions"] / f"{job_id}.csv.gz"),
                        })

    # 2) Robustness suite: alternative external test sets for the MAIN target only.
    for alt_test_seed in cfg.alternative_test_seeds:
        split_cache_path = paths["split_cache"] / f"split__{slugify(cfg.main_target)}__testseed_{alt_test_seed}.pkl.gz"
        split_info = build_fixed_external_split(df, cfg.main_target, alt_test_seed, cfg, split_cache_path, logger)
        train_pool_size = len(split_info["idx_train_pool"])
        sample_sizes = resolve_sample_sizes(train_pool_size, cfg)

        logger.info(f"Adding robustness-suite jobs for main target with alternative test_seed={alt_test_seed} and train-pool size {train_pool_size:,}.")
        for subsample_seed in range(cfg.robustness_seeds):
            for n_train in sample_sizes:
                for descriptor_family in families.keys():
                    for model_name in models.keys():
                        job_id = build_job_id(
                            suite_name="robustness_alt_test",
                            target_col=cfg.main_target,
                            test_seed=alt_test_seed,
                            subsample_seed=subsample_seed,
                            n_train=n_train,
                            descriptor_family=descriptor_family,
                            model_name=model_name,
                        )
                        manifest_rows.append({
                            "job_id": job_id,
                            "suite_name": "robustness_alt_test",
                            "target_col": cfg.main_target,
                            "is_main_target": True,
                            "test_seed": alt_test_seed,
                            "subsample_seed": subsample_seed,
                            "n_train": int(n_train),
                            "descriptor_family": descriptor_family,
                            "model_name": model_name,
                            "job_metrics_path": str(paths["job_metrics"] / f"{job_id}.json"),
                            "job_predictions_path": str(paths["job_predictions"] / f"{job_id}.csv.gz"),
                        })

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["suite_name", "target_col", "test_seed", "subsample_seed", "n_train", "descriptor_family", "model_name"]
    ).reset_index(drop=True)

    save_dataframe(manifest, manifest_path, manifest_pkl)
    logger.info(f"Job manifest created with {len(manifest):,} jobs.")
    return manifest


# =============================================================================
# SECTION 7. CORE TRAINING / EVALUATION LOGIC
# =============================================================================

def prepare_xy(
    df: pd.DataFrame,
    row_idx: np.ndarray,
    target_col: str,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Slice X and y for a particular job.
    """
    use_cols = list(numeric_features) + list(categorical_features)
    X = df.iloc[row_idx][use_cols].copy()
    y = df.iloc[row_idx][target_col].astype(float).values
    return X, y


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: ProjectConfig,
) -> Dict[str, float]:
    """
    Compute all per-job evaluation metrics used in the manuscript and SI.
    """
    result = {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": safe_spearman(y_true, y_pred),
    }

    # Fraction-based top-k metrics
    k_frac = top_k_count_from_fraction(len(y_true), cfg.topk_fraction)
    result[f"topk_overlap_frac_{cfg.topk_fraction:.3f}".replace(".", "p")] = topk_overlap_fraction(y_true, y_pred, k_frac)
    result[f"elite_enrichment_{cfg.topk_fraction:.3f}".replace(".", "p")] = elite_enrichment_factor(y_true, y_pred, k_frac)

    # Absolute top-k metrics
    for k in cfg.topk_absolute_list:
        kk = min(k, len(y_true))
        result[f"topk_overlap_k{kk}"] = topk_overlap_fraction(y_true, y_pred, kk)
        result[f"elite_enrichment_k{kk}"] = elite_enrichment_factor(y_true, y_pred, kk)

    return result


def run_single_job(
    job_row: pd.Series,
    df: pd.DataFrame,
    cfg: ProjectConfig,
    paths: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """
    Execute one benchmark job if needed.

    Resume logic:
    If the job metrics file already exists and resume_if_available=True, the job
    is skipped automatically.
    """
    job_id = job_row["job_id"]
    metrics_path = Path(job_row["job_metrics_path"])
    predictions_path = Path(job_row["job_predictions_path"])

    if cfg.resume_if_available and metrics_path.exists() and predictions_path.exists():
        return

    target_col = job_row["target_col"]
    suite_name = job_row["suite_name"]
    test_seed = int(job_row["test_seed"])
    subsample_seed = int(job_row["subsample_seed"])
    n_train = int(job_row["n_train"])
    descriptor_family = job_row["descriptor_family"]
    model_name = job_row["model_name"]

    families = get_descriptor_family_map(df)
    models = get_model_catalog(cfg)

    family_info = families[descriptor_family]
    model_info = models[model_name]

    split_cache_path = paths["split_cache"] / f"split__{slugify(target_col)}__testseed_{test_seed}.pkl.gz"
    split_info = build_fixed_external_split(df, target_col, test_seed, cfg, split_cache_path, logger)

    # Nested indices cache depends on suite, target, test seed, and subsample seed.
    nested_cache = paths["split_cache"] / (
        f"nested__{suite_name}__{slugify(target_col)}__testseed_{test_seed}"
        f"__subseed_{subsample_seed}.pkl.gz"
    )
    sample_sizes = resolve_sample_sizes(len(split_info["idx_train_pool"]), cfg)
    nested_map = build_nested_subsample_index_map(split_info["idx_train_pool"], sample_sizes, subsample_seed, nested_cache, cfg)

    idx_train = nested_map[n_train]
    idx_test = split_info["idx_test"]

    numeric_features = family_info["numeric"]
    categorical_features = family_info["categorical"]

    # Prepare train/test matrices.
    X_train, y_train = prepare_xy(df, idx_train, target_col, numeric_features, categorical_features)
    X_test, y_test = prepare_xy(df, idx_test, target_col, numeric_features, categorical_features)

    # Basic missing-value handling:
    # Lightweight approach: numeric median imputation and categorical fill.
    # We do it explicitly before the pipeline so saved predictions remain simple.
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)
        else:
            X_train[col] = X_train[col].fillna("__MISSING__")
            X_test[col] = X_test[col].fillna("__MISSING__")

    pipeline = build_pipeline(
        model_name=model_name,
        model_info=model_info,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        seed=subsample_seed + 1000 * test_seed + 17,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = evaluate_predictions(y_test, y_pred, cfg)
    metrics.update({
        "job_id": job_id,
        "suite_name": suite_name,
        "target_col": target_col,
        "test_seed": test_seed,
        "subsample_seed": subsample_seed,
        "n_train": int(n_train),
        "descriptor_family": descriptor_family,
        "model_name": model_name,
        "n_test": int(len(y_test)),
        "timestamp_finished": current_timestamp(),
    })

    prediction_df = pd.DataFrame({
        "job_id": job_id,
        "suite_name": suite_name,
        "target_col": target_col,
        "test_seed": test_seed,
        "subsample_seed": subsample_seed,
        "n_train": int(n_train),
        "descriptor_family": descriptor_family,
        "model_name": model_name,
        "row_index": idx_test,
        "material_id": df.iloc[idx_test][CONFIG.id_col].values,
        "y_true": y_test,
        "y_pred": y_pred,
    })

    # Rank positions in descending adsorption performance.
    prediction_df["rank_true_desc"] = prediction_df["y_true"].rank(method="first", ascending=False)
    prediction_df["rank_pred_desc"] = prediction_df["y_pred"].rank(method="first", ascending=False)
    prediction_df["abs_error"] = (prediction_df["y_true"] - prediction_df["y_pred"]).abs()

    ensure_dir(metrics_path.parent)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    ensure_dir(predictions_path.parent)
    prediction_df.to_csv(predictions_path, index=False, compression="gzip")


# =============================================================================
# SECTION 8. EXECUTE JOBS WITH RESUME SUPPORT
# =============================================================================

def run_all_jobs(
    manifest: pd.DataFrame,
    df: pd.DataFrame,
    cfg: ProjectConfig,
    paths: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """
    Iterate through all jobs and execute missing ones.
    """
    total = len(manifest)
    done = 0
    start_time = time.time()

    for i, (_, row) in enumerate(manifest.iterrows(), start=1):
        metrics_path = Path(row["job_metrics_path"])
        predictions_path = Path(row["job_predictions_path"])
        already_done = cfg.resume_if_available and metrics_path.exists() and predictions_path.exists()
        if already_done:
            done += 1
        else:
            run_single_job(row, df, cfg, paths, logger)
            done += 1

        if (i % cfg.verbose_every_n_jobs == 0) or (i == total):
            elapsed = time.time() - start_time
            logger.info(f"Progress: {done:,}/{total:,} jobs completed ({100*done/total:.1f}%). Elapsed: {elapsed/60:.1f} min")

    logger.info("All benchmark jobs completed or already available.")


# =============================================================================
# SECTION 9. LOAD JOB OUTPUTS AND BUILD MASTER RESULT TABLES
# =============================================================================

def load_all_job_metrics(manifest: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Compile all per-job JSON metrics into one master table.
    """
    rows = []
    for _, row in manifest.iterrows():
        metrics_path = Path(row["job_metrics_path"])
        if metrics_path.exists():
            rows.append(load_json(metrics_path))
    if not rows:
        raise RuntimeError("No job metrics found. Run the benchmark jobs first.")
    df_metrics = pd.DataFrame(rows)

    logger.info(f"Loaded {len(df_metrics):,} completed job metrics.")
    return df_metrics


def load_prediction_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip")


def add_method_label_columns(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined labels that are convenient for ranking and plotting.
    """
    df = df_metrics.copy()
    df["method_label"] = df["descriptor_family"] + " | " + df["model_name"]
    return df


def aggregate_performance(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mean / CI performance for each method at each sample size.
    """
    df = add_method_label_columns(df_metrics)
    metric_cols = [c for c in df.columns if c.startswith(("rmse", "mae", "r2", "spearman", "topk_overlap", "elite_enrichment"))]

    group_cols = ["suite_name", "target_col", "test_seed", "n_train", "descriptor_family", "model_name", "method_label"]
    rows = []

    for keys, sub in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_repeats"] = len(sub)

        for metric in metric_cols:
            mean_val, low, high = ci95(sub[metric])
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_std"] = float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["suite_name", "target_col", "test_seed", "n_train", "rmse_mean"])


# =============================================================================
# SECTION 10. RANKING STABILITY ANALYSES
# =============================================================================

def compute_seedwise_rankings(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Rank methods within each target / test seed / sample size / subsample seed
    using RMSE as the primary criterion (lower is better).
    """
    df = add_method_label_columns(df_metrics)
    rows = []

    for keys, sub in df.groupby(["suite_name", "target_col", "test_seed", "n_train", "subsample_seed"]):
        sub = sub.sort_values(["rmse", "mae", "descriptor_family", "model_name"]).reset_index(drop=True)
        sub["rank_rmse"] = np.arange(1, len(sub) + 1)
        for _, r in sub.iterrows():
            row = dict(zip(["suite_name", "target_col", "test_seed", "n_train", "subsample_seed"], keys))
            row["method_label"] = r["method_label"]
            row["rank_rmse"] = int(r["rank_rmse"])
            row["rmse"] = r["rmse"]
            rows.append(row)

    return pd.DataFrame(rows)


def compute_ranking_stability(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Quantify:
    - probability each method is rank 1
    - top-1 consensus probability
    - average Spearman correlation of seedwise rankings vs full-data mean ranking
    """
    df = add_method_label_columns(df_metrics)
    rank_df = compute_seedwise_rankings(df_metrics)

    rows = []

    # First compute a reference ranking at full data for each suite/target/test_seed.
    full_sizes = df.groupby(["suite_name", "target_col", "test_seed"])["n_train"].max().reset_index()
    ref_maps = {}

    for _, fs in full_sizes.iterrows():
        suite_name, target_col, test_seed, n_full = fs["suite_name"], fs["target_col"], fs["test_seed"], int(fs["n_train"])
        sub = df[
            (df["suite_name"] == suite_name) &
            (df["target_col"] == target_col) &
            (df["test_seed"] == test_seed) &
            (df["n_train"] == n_full)
        ]
        summary = sub.groupby("method_label")["rmse"].mean().sort_values()
        ref_maps[(suite_name, target_col, test_seed)] = summary.index.tolist()

    for keys, sub in rank_df.groupby(["suite_name", "target_col", "test_seed", "n_train"]):
        suite_name, target_col, test_seed, n_train = keys
        method_probs = sub[sub["rank_rmse"] == 1]["method_label"].value_counts(normalize=True).to_dict()
        top1_consensus = max(method_probs.values()) if method_probs else np.nan

        # Mean Spearman correlation of seedwise rankings vs full-data mean ranking.
        ref_order = ref_maps[(suite_name, target_col, test_seed)]
        ref_rank = {m: i + 1 for i, m in enumerate(ref_order)}
        seed_corrs = []

        for seed, seed_sub in sub.groupby("subsample_seed"):
            current_rank = seed_sub.set_index("method_label")["rank_rmse"].to_dict()
            common_methods = [m for m in ref_order if m in current_rank]
            if len(common_methods) >= 2:
                a = [ref_rank[m] for m in common_methods]
                b = [current_rank[m] for m in common_methods]
                corr = safe_spearman(np.array(a), np.array(b))
                seed_corrs.append(corr)

        row = {
            "suite_name": suite_name,
            "target_col": target_col,
            "test_seed": test_seed,
            "n_train": int(n_train),
            "top1_consensus_probability": top1_consensus,
            "mean_rank_spearman_vs_full": float(np.nanmean(seed_corrs)) if seed_corrs else np.nan,
        }

        # Also save method-specific probability of being rank 1.
        for method_label, prob in method_probs.items():
            row[f"p_rank1__{slugify(method_label)}"] = prob

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["suite_name", "target_col", "test_seed", "n_train"])


# =============================================================================
# SECTION 11. SCREENING REPRODUCIBILITY ANALYSES
# =============================================================================

def compute_screening_reproducibility(df_metrics: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """
    Summarize screening-relevant metrics across seeds for each method and sample size.
    """
    df = add_method_label_columns(df_metrics)
    overlap_cols = [c for c in df.columns if c.startswith("topk_overlap")]
    elite_cols = [c for c in df.columns if c.startswith("elite_enrichment")]

    group_cols = ["suite_name", "target_col", "test_seed", "n_train", "method_label", "descriptor_family", "model_name"]
    rows = []

    for keys, sub in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_repeats"] = len(sub)

        for col in overlap_cols + elite_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["suite_name", "target_col", "test_seed", "n_train", "method_label"])


# =============================================================================
# SECTION 12. STABLE-CONCLUSION THRESHOLD TABLE
# =============================================================================

def intervals_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    """
    Whether two confidence intervals overlap.
    """
    return max(a_low, b_low) <= min(a_high, b_high)


def compute_stable_conclusion_table(
    agg_perf: pd.DataFrame,
    ranking_stability: pd.DataFrame,
    screening: pd.DataFrame,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    """
    Define a practical benchmark-maturity threshold table.

    The criterion is deliberately transparent:
    1. Top-2 RMSE confidence intervals overlap.
    2. Top-1 consensus probability is high enough.
    3. Mean rank correlation vs full-data ranking is high enough.
    4. Screening metrics for the best-mean-RMSE method are sufficiently stable.
    """
    rows = []

    # Focus on the primary fixed external test suite.
    agg_primary = agg_perf[agg_perf["suite_name"] == "primary"].copy()
    rank_primary = ranking_stability[ranking_stability["suite_name"] == "primary"].copy()
    screen_primary = screening[screening["suite_name"] == "primary"].copy()

    overlap_metric = f"topk_overlap_frac_{cfg.topk_fraction:.3f}".replace(".", "p")
    elite_metric = f"elite_enrichment_{cfg.topk_fraction:.3f}".replace(".", "p")

    for (target_col, test_seed), sub_all in agg_primary.groupby(["target_col", "test_seed"]):
        sizes = sorted(sub_all["n_train"].unique())
        per_n_rows = []

        for n_train in sizes:
            sub = sub_all[sub_all["n_train"] == n_train].sort_values("rmse_mean").reset_index(drop=True)
            best = sub.iloc[0]
            second = sub.iloc[1] if len(sub) > 1 else sub.iloc[0]
            ci_overlap = intervals_overlap(
                best["rmse_ci_low"], best["rmse_ci_high"],
                second["rmse_ci_low"], second["rmse_ci_high"]
            )

            rank_row = rank_primary[
                (rank_primary["target_col"] == target_col) &
                (rank_primary["test_seed"] == test_seed) &
                (rank_primary["n_train"] == n_train)
            ].iloc[0]

            best_method_label = best["method_label"]
            screen_row = screen_primary[
                (screen_primary["target_col"] == target_col) &
                (screen_primary["test_seed"] == test_seed) &
                (screen_primary["n_train"] == n_train) &
                (screen_primary["method_label"] == best_method_label)
            ]
            if len(screen_row) == 0:
                overlap_std = np.nan
                elite_std = np.nan
            else:
                screen_row = screen_row.iloc[0]
                overlap_std = screen_row.get(f"{overlap_metric}_std", np.nan)
                elite_std = screen_row.get(f"{elite_metric}_std", np.nan)

            conditions_met = (
                bool(ci_overlap) and
                float(rank_row["top1_consensus_probability"]) >= cfg.stability_top1_consensus_threshold and
                float(rank_row["mean_rank_spearman_vs_full"]) >= cfg.stability_rankcorr_threshold and
                float(overlap_std) <= cfg.stability_topk_std_threshold and
                float(elite_std) <= cfg.stability_elite_std_threshold
            )

            per_n_rows.append({
                "target_col": target_col,
                "test_seed": test_seed,
                "n_train": int(n_train),
                "best_method_label": best_method_label,
                "best_rmse_mean": best["rmse_mean"],
                "second_best_method_label": second["method_label"],
                "top2_rmse_ci_overlap": ci_overlap,
                "top1_consensus_probability": float(rank_row["top1_consensus_probability"]),
                "mean_rank_spearman_vs_full": float(rank_row["mean_rank_spearman_vs_full"]),
                f"{overlap_metric}_std_best": overlap_std,
                f"{elite_metric}_std_best": elite_std,
                "conditions_met_here": conditions_met,
            })

        # "Threshold" means first n after which all larger n values also satisfy the rule.
        threshold_n = np.nan
        threshold_method = None
        for i, row in enumerate(per_n_rows):
            tail = per_n_rows[i:]
            if all(r["conditions_met_here"] for r in tail):
                threshold_n = row["n_train"]
                threshold_method = row["best_method_label"]
                break

        for row in per_n_rows:
            row["benchmark_maturity_threshold_n"] = threshold_n
            row["threshold_method_label"] = threshold_method
            row["is_at_or_above_threshold"] = (row["n_train"] >= threshold_n) if not pd.isna(threshold_n) else False
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["target_col", "test_seed", "n_train"])


# =============================================================================
# SECTION 13. PAIRWISE PROBABILITY OF SUPERIORITY
# =============================================================================

def compute_pairwise_probability_superiority(
    df_metrics: pd.DataFrame,
    selected_sizes: Sequence[int],
) -> Dict[Tuple[str, int, int], pd.DataFrame]:
    """
    For selected sample sizes, compute P(method A beats method B) using the
    empirical fraction of seeds where A has lower RMSE than B.

    Returns a dictionary keyed by:
        (target_col, test_seed, n_train)
    """
    df = add_method_label_columns(df_metrics)
    results = {}

    for (target_col, test_seed, n_train), sub in df.groupby(["target_col", "test_seed", "n_train"]):
        if n_train not in selected_sizes:
            continue

        pivot = sub.pivot_table(
            index="subsample_seed",
            columns="method_label",
            values="rmse",
            aggfunc="first",
        )
        methods = list(pivot.columns)
        matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)

        for a in methods:
            for b in methods:
                pair = pivot[[a, b]].dropna()
                if len(pair) == 0:
                    matrix.loc[a, b] = np.nan
                elif a == b:
                    matrix.loc[a, b] = 0.5
                else:
                    matrix.loc[a, b] = float((pair[a] < pair[b]).mean())

        results[(target_col, test_seed, int(n_train))] = matrix

    return results


# =============================================================================
# SECTION 14. FEATURE-EFFECT / INTERPRETATION STABILITY (FIGURE 6)
# =============================================================================

def identify_best_full_data_method(agg_perf: pd.DataFrame, target_col: str, test_seed: int) -> pd.Series:
    """
    Identify the best mean-RMSE method at the full sample size for a given target.
    """
    sub = agg_perf[
        (agg_perf["suite_name"] == "primary") &
        (agg_perf["target_col"] == target_col) &
        (agg_perf["test_seed"] == test_seed)
    ].copy()
    full_n = int(sub["n_train"].max())
    best = sub[sub["n_train"] == full_n].sort_values("rmse_mean").iloc[0]
    return best


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract transformed feature names from a fitted ColumnTransformer.
    """
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue

        if transformer == "passthrough":
            feature_names.extend(list(cols))
        elif hasattr(transformer, "get_feature_names_out"):
            try:
                names = transformer.get_feature_names_out(cols)
            except Exception:
                names = transformer.get_feature_names_out()
            feature_names.extend(list(names))
        else:
            # Fallback
            if isinstance(cols, (list, tuple, np.ndarray)):
                feature_names.extend(list(cols))
            else:
                feature_names.append(str(cols))
    return [str(x) for x in feature_names]


def run_feature_effect_jobs(
    df: pd.DataFrame,
    agg_perf: pd.DataFrame,
    cfg: ProjectConfig,
    paths: Dict[str, Path],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Refit the best full-data method at selected sample sizes and compute
    permutation importances on the fixed external test set.

    This provides the data for Figure 6:
        "Convergence of feature effects / SHAP-like rank ordering"
    without requiring an additional SHAP dependency.
    """
    out_csv = paths["feature_effect_tables"] / "feature_effect_importances_all.csv"
    out_pkl = paths["feature_effect_tables"] / "feature_effect_importances_all.pkl.gz"

    if cfg.resume_if_available and out_csv.exists() and out_pkl.exists():
        logger.info("Loading existing feature-effect summary.")
        return load_pickle(out_pkl)

    families = get_descriptor_family_map(df)
    models = get_model_catalog(cfg)

    best = identify_best_full_data_method(agg_perf, cfg.main_target, cfg.primary_test_seed)
    best_family = best["descriptor_family"]
    best_model = best["model_name"]

    logger.info(f"Feature-effect analysis will use best full-data main-text method: {best_family} | {best_model}")

    split_cache_path = paths["split_cache"] / f"split__{slugify(cfg.main_target)}__testseed_{cfg.primary_test_seed}.pkl.gz"
    split_info = load_pickle(split_cache_path)
    train_pool_size = len(split_info["idx_train_pool"])
    sample_sizes = resolve_sample_sizes(train_pool_size, cfg)

    selected_sizes = sorted(set([n for n in cfg.feature_effect_sizes if n in sample_sizes] + [sample_sizes[-1]]))

    rows = []

    for subsample_seed in range(cfg.feature_effect_seeds):
        nested_cache = paths["split_cache"] / (
            f"nested__primary__{slugify(cfg.main_target)}__testseed_{cfg.primary_test_seed}"
            f"__subseed_{subsample_seed}.pkl.gz"
        )
        nested_map = build_nested_subsample_index_map(split_info["idx_train_pool"], sample_sizes, subsample_seed, nested_cache, cfg)

        for n_train in selected_sizes:
            ckpt_path = paths["job_feature_effects"] / (
                f"feature_effect__{slugify(cfg.main_target)}__testseed_{cfg.primary_test_seed}"
                f"__subseed_{subsample_seed}__n_{n_train}"
                f"__{best_family}__{best_model}.json"
            )

            if cfg.resume_if_available and ckpt_path.exists():
                part = pd.DataFrame(load_json(ckpt_path)["rows"])
                rows.extend(part.to_dict(orient="records"))
                continue

            family_info = families[best_family]
            model_info = models[best_model]

            idx_train = nested_map[n_train]
            idx_test = split_info["idx_test"]

            X_train, y_train = prepare_xy(
                df, idx_train, cfg.main_target,
                family_info["numeric"], family_info["categorical"]
            )
            X_test, y_test = prepare_xy(
                df, idx_test, cfg.main_target,
                family_info["numeric"], family_info["categorical"]
            )

            for col in X_train.columns:
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    med = X_train[col].median()
                    X_train[col] = X_train[col].fillna(med)
                    X_test[col] = X_test[col].fillna(med)
                else:
                    X_train[col] = X_train[col].fillna("__MISSING__")
                    X_test[col] = X_test[col].fillna("__MISSING__")

            pipe = build_pipeline(
                best_model, model_info,
                family_info["numeric"], family_info["categorical"],
                seed=subsample_seed + 1000 * cfg.primary_test_seed + 17,
            )
            pipe.fit(X_train, y_train)

            # Permutation importance on the fixed external test set.
            perm = permutation_importance(
                pipe, X_test, y_test,
                scoring="neg_root_mean_squared_error",
                n_repeats=10,
                random_state=subsample_seed + 77,
                n_jobs=1,
            )

            feature_names = get_transformed_feature_names(pipe.named_steps["preprocessor"])
            importances_mean = perm.importances_mean
            importances_std = perm.importances_std

            part_rows = []
            for feat, imp_mean, imp_std in zip(feature_names, importances_mean, importances_std):
                part_rows.append({
                    "target_col": cfg.main_target,
                    "test_seed": cfg.primary_test_seed,
                    "subsample_seed": subsample_seed,
                    "n_train": int(n_train),
                    "descriptor_family": best_family,
                    "model_name": best_model,
                    "feature": str(feat),
                    "importance_mean": float(imp_mean),
                    "importance_std": float(imp_std),
                })

            save_json({"rows": part_rows}, ckpt_path)
            rows.extend(part_rows)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("Feature-effect analysis produced no rows.")
    save_dataframe(out, out_csv, out_pkl)
    return out


def compute_feature_rank_convergence(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each sample-size feature ranking to the full-data feature ranking.
    """
    rows = []
    full_n = int(feature_df["n_train"].max())

    # Reference mean importance ranking at full n.
    ref = (
        feature_df[feature_df["n_train"] == full_n]
        .groupby("feature")["importance_mean"]
        .mean()
        .sort_values(ascending=False)
    )
    ref_rank = {feat: i + 1 for i, feat in enumerate(ref.index)}

    for n_train, sub_n in feature_df.groupby("n_train"):
        by_seed = []
        for seed, seed_sub in sub_n.groupby("subsample_seed"):
            rank_sub = seed_sub.sort_values("importance_mean", ascending=False)
            seed_rank = {feat: i + 1 for i, feat in enumerate(rank_sub["feature"].tolist())}
            common = [feat for feat in ref.index if feat in seed_rank]
            if len(common) >= 2:
                a = [ref_rank[f] for f in common]
                b = [seed_rank[f] for f in common]
                corr = safe_spearman(np.array(a), np.array(b))
            else:
                corr = np.nan

            top10_ref = list(ref.index[:10])
            top10_seed = rank_sub["feature"].tolist()[:10]
            jacc = jaccard_similarity(top10_ref, top10_seed)

            by_seed.append({
                "n_train": int(n_train),
                "subsample_seed": seed,
                "rank_spearman_vs_full": corr,
                "top10_jaccard_vs_full": jacc,
            })

        by_seed_df = pd.DataFrame(by_seed)
        rows.append({
            "n_train": int(n_train),
            "rank_spearman_vs_full_mean": float(by_seed_df["rank_spearman_vs_full"].mean()),
            "rank_spearman_vs_full_std": float(by_seed_df["rank_spearman_vs_full"].std(ddof=1)) if len(by_seed_df) > 1 else 0.0,
            "top10_jaccard_vs_full_mean": float(by_seed_df["top10_jaccard_vs_full"].mean()),
            "top10_jaccard_vs_full_std": float(by_seed_df["top10_jaccard_vs_full"].std(ddof=1)) if len(by_seed_df) > 1 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("n_train")


# =============================================================================
# SECTION 15. ADDITIONAL ANALYSES TO STRENGTHEN THE MANUSCRIPT
# =============================================================================


def compute_descriptor_family_aggregation(agg_perf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance at descriptor-family level by averaging across models."""
    rows = []
    group_cols = ["suite_name", "target_col", "test_seed", "n_train", "descriptor_family"]
    for keys, sub in agg_perf.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_models"] = int(sub["model_name"].nunique())
        for metric in ["rmse", "mae", "r2", "spearman"]:
            mean_col = f"{metric}_mean"
            row[f"{metric}_family_mean"] = float(sub[mean_col].mean())
            row[f"{metric}_family_best"] = float(sub[mean_col].min()) if metric in ["rmse", "mae"] else float(sub[mean_col].max())
            row[f"{metric}_family_std_across_models"] = float(sub[mean_col].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def compute_sample_efficiency_table(agg_perf: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """For each method, determine the smallest n reaching fractions of full-data performance."""
    sub = agg_perf[(agg_perf["suite_name"] == "primary") & (agg_perf["test_seed"] == cfg.primary_test_seed)].copy()
    rows = []
    for (target_col, method_label), grp in sub.groupby(["target_col", "method_label"]):
        grp = grp.sort_values("n_train")
        full_rmse = float(grp.loc[grp["n_train"].idxmax(), "rmse_mean"])
        initial_rmse = float(grp.iloc[0]["rmse_mean"])
        improvement = initial_rmse - full_rmse
        if improvement <= 0:
            thresholds = {0.5: np.nan, 0.8: np.nan, 0.9: np.nan, 0.95: np.nan}
            achieved_fraction = np.zeros(len(grp))
        else:
            achieved_fraction = (initial_rmse - grp["rmse_mean"]) / improvement
            thresholds = {}
            for frac in [0.5, 0.8, 0.9, 0.95]:
                met = grp.loc[achieved_fraction >= frac, "n_train"]
                thresholds[frac] = int(met.iloc[0]) if len(met) else np.nan
        row = {
            "target_col": target_col,
            "method_label": method_label,
            "descriptor_family": grp.iloc[0]["descriptor_family"],
            "model_name": grp.iloc[0]["model_name"],
            "initial_rmse": initial_rmse,
            "full_rmse": full_rmse,
            "absolute_rmse_gain": improvement,
            "n_to_50pct_gain": thresholds[0.5],
            "n_to_80pct_gain": thresholds[0.8],
            "n_to_90pct_gain": thresholds[0.9],
            "n_to_95pct_gain": thresholds[0.95],
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_col", "full_rmse"])


def compute_target_difficulty_table(agg_perf: pd.DataFrame, ranking_stability: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Summarize target difficulty and maturity behaviour."""
    rows = []
    sub = agg_perf[(agg_perf["suite_name"] == "primary") & (agg_perf["test_seed"] == cfg.primary_test_seed)].copy()
    rs = ranking_stability[(ranking_stability["suite_name"] == "primary") & (ranking_stability["test_seed"] == cfg.primary_test_seed)].copy()
    for target_col, grp in sub.groupby("target_col"):
        full_n = int(grp["n_train"].max())
        full_grp = grp[grp["n_train"] == full_n].copy()
        best_full = full_grp.sort_values("rmse_mean").iloc[0]
        spread = float(full_grp["rmse_mean"].max() - full_grp["rmse_mean"].min())
        rs_full = rs[(rs["target_col"] == target_col) & (rs["n_train"] == full_n)].iloc[0]
        rows.append({
            "target_col": target_col,
            "full_n": full_n,
            "best_method_label": best_full["method_label"],
            "best_full_rmse": float(best_full["rmse_mean"]),
            "best_full_r2": float(best_full["r2_mean"]),
            "best_full_spearman": float(best_full["spearman_mean"]),
            "method_spread_rmse_at_full": spread,
            "top1_consensus_probability_at_full": float(rs_full["top1_consensus_probability"]),
            "mean_rank_spearman_vs_full_at_full": float(rs_full["mean_rank_spearman_vs_full"]),
        })
    return pd.DataFrame(rows).sort_values("best_full_rmse")


def compute_method_pareto_summary(agg_perf: pd.DataFrame, ranking_stability: pd.DataFrame, screening: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Combine accuracy, stability, and screening performance into a pragmatic summary table."""
    main = agg_perf[(agg_perf["suite_name"] == "primary") & (agg_perf["target_col"] == cfg.main_target) & (agg_perf["test_seed"] == cfg.primary_test_seed)].copy()
    rs = ranking_stability[(ranking_stability["suite_name"] == "primary") & (ranking_stability["target_col"] == cfg.main_target) & (ranking_stability["test_seed"] == cfg.primary_test_seed)].copy()
    sc = screening[(screening["suite_name"] == "primary") & (screening["target_col"] == cfg.main_target) & (screening["test_seed"] == cfg.primary_test_seed)].copy()
    full_n = int(main["n_train"].max())
    main = main[main["n_train"] == full_n].copy()
    rs = rs[rs["n_train"] == full_n].copy()
    metric_overlap = f"topk_overlap_frac_{cfg.topk_fraction:.3f}".replace(".", "p")
    score_rows = []
    rmse_rank = main["rmse_mean"].rank(method="min", ascending=True)
    r2_rank = main["r2_mean"].rank(method="min", ascending=False)
    sp_rank = main["spearman_mean"].rank(method="min", ascending=False)
    for i, (_, row) in enumerate(main.iterrows()):
        m = row["method_label"]
        rs_row = rs.iloc[0] if len(rs) else None
        sc_row = sc[sc["method_label"] == m]
        sc_row = sc_row[sc_row["n_train"] == full_n]
        score_rows.append({
            "method_label": m,
            "descriptor_family": row["descriptor_family"],
            "model_name": row["model_name"],
            "rmse_mean": float(row["rmse_mean"]),
            "r2_mean": float(row["r2_mean"]),
            "spearman_mean": float(row["spearman_mean"]),
            "rmse_rank": float(rmse_rank.loc[row.name]),
            "r2_rank": float(r2_rank.loc[row.name]),
            "spearman_rank": float(sp_rank.loc[row.name]),
            "top1_consensus_probability": float(rs_row["top1_consensus_probability"]) if rs_row is not None else np.nan,
            "screening_topk_overlap_mean": float(sc_row.iloc[0].get(f"{metric_overlap}_mean", np.nan)) if len(sc_row) else np.nan,
        })
    out = pd.DataFrame(score_rows)
    out["aggregate_rank_score"] = out[["rmse_rank", "r2_rank", "spearman_rank"]].mean(axis=1)
    return out.sort_values(["aggregate_rank_score", "rmse_mean"])


def compute_additional_analyses(agg_perf: pd.DataFrame, ranking_stability: pd.DataFrame, screening: pd.DataFrame, cfg: ProjectConfig) -> Dict[str, pd.DataFrame]:
    return {
        "descriptor_family_aggregation": compute_descriptor_family_aggregation(agg_perf),
        "sample_efficiency": compute_sample_efficiency_table(agg_perf, cfg),
        "target_difficulty": compute_target_difficulty_table(agg_perf, ranking_stability, cfg),
        "method_pareto_summary": compute_method_pareto_summary(agg_perf, ranking_stability, screening, cfg),
    }


def make_additional_analysis_figures(agg_perf: pd.DataFrame, ranking_stability: pd.DataFrame, additional: Dict[str, pd.DataFrame], cfg: ProjectConfig, paths: Dict[str, Path]) -> None:
    """Create extra, more interpretive figures to strengthen the manuscript/SI."""
    # Figure A1: descriptor-family learning curves (family-level mean across models)
    fam = additional["descriptor_family_aggregation"]
    fam_main = fam[(fam["suite_name"] == "primary") & (fam["target_col"] == cfg.main_target) & (fam["test_seed"] == cfg.primary_test_seed)].copy()
    fam_order = (
        fam_main[fam_main["n_train"] == fam_main["n_train"].max()]
        .sort_values("rmse_family_mean")["descriptor_family"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    style_map = build_method_style_map(fam_order)
    for fam_name in fam_order:
        sub = fam_main[fam_main["descriptor_family"] == fam_name].sort_values("n_train")
        style = style_map[fam_name]
        ax.plot(sub["n_train"], sub["rmse_family_mean"], label=pretty_method_label(fam_name.replace("_", " ")).replace(" |", ""), color=style["color"], marker=style["marker"])
        ax.fill_between(sub["n_train"], sub["rmse_family_mean"] - sub["rmse_family_std_across_models"], sub["rmse_family_mean"] + sub["rmse_family_std_across_models"], alpha=0.12, color=style["color"])
    ax.set_xscale("log")
    ax.set_xlabel("Training size (log scale)")
    ax.set_ylabel("Family-level mean RMSE across models")
    ax.set_title("Additional analysis. Descriptor-family learning curves")
    ax.legend(frameon=False)
    save_figure_dataframes({"descriptor_family_learning_curve": fam_main}, "Additional_descriptor_family_learning_curves", paths["si_figure_data"], paths["all_figure_data"])
    save_figure(fig, paths["si_figures_extra"] / "Additional_descriptor_family_learning_curves")

    # Figure A2: sample efficiency heatmap for reaching 90% of attainable gain
    eff = additional["sample_efficiency"]
    eff_main = eff[eff["target_col"] == cfg.main_target].copy()
    if len(eff_main):
        methods = eff_main.sort_values("full_rmse")["method_label"].tolist()
        value_col = "n_to_90pct_gain"
        vals = eff_main.set_index("method_label").loc[methods, value_col].values.reshape(-1, 1)
        fig, ax = plt.subplots(figsize=(6, max(6, 0.35 * len(methods))))
        im = ax.imshow(vals, aspect="auto", interpolation="nearest")
        ax.set_xticks([0])
        ax.set_xticklabels(["n to reach 90%\nof attainable gain"])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([pretty_method_label(m) for m in methods], fontsize=9)
        ax.set_title("Additional analysis. Sample-efficiency heatmap")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        eff_export = eff_main[["method_label", "descriptor_family", "model_name", "initial_rmse", "full_rmse", "absolute_rmse_gain", "n_to_50pct_gain", "n_to_80pct_gain", "n_to_90pct_gain", "n_to_95pct_gain"]].copy()
        save_figure_dataframes({"sample_efficiency_values": eff_export}, "Additional_sample_efficiency_heatmap", paths["si_figure_data"], paths["all_figure_data"])
        save_figure(fig, paths["si_figures_extra"] / "Additional_sample_efficiency_heatmap")

    # Figure A3: target difficulty summary
    td = additional["target_difficulty"].copy().sort_values("best_full_rmse")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(td["best_full_rmse"], td["best_full_spearman"], s=90)
    for _, r in td.iterrows():
        ax.annotate(r["target_col"].replace("uptake(mmol/g) ", ""), (r["best_full_rmse"], r["best_full_spearman"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Best full-data RMSE")
    ax.set_ylabel("Best full-data Spearman")
    ax.set_title("Additional analysis. Relative difficulty of the four adsorption targets")
    save_figure_dataframes({"target_difficulty_values": td}, "Additional_target_difficulty_map", paths["si_figure_data"], paths["all_figure_data"])
    save_figure(fig, paths["si_figures_extra"] / "Additional_target_difficulty_map")

# =============================================================================
# SECTION 15. SAVE TABLES FOR MANUSCRIPT AND SI
# =============================================================================

def build_descriptor_model_definition_tables(
    df: pd.DataFrame,
    cfg: ProjectConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create Table 1-style definitions of descriptor families and models.
    """
    families = get_descriptor_family_map(df)
    models = get_model_catalog(cfg)

    family_rows = []
    for family_name, info in families.items():
        features = info["numeric"] + info["categorical"]
        family_rows.append({
            "descriptor_family": family_name,
            "n_features_before_encoding": len(features),
            "numeric_features": ", ".join(info["numeric"]) if info["numeric"] else "-",
            "categorical_features": ", ".join(info["categorical"]) if info["categorical"] else "-",
            "description": info["description"],
        })
    family_df = pd.DataFrame(family_rows)

    model_rows = []
    for model_name, info in models.items():
        model_rows.append({
            "model_name": model_name,
            "model_type": info["type"],
            "scaling_used": bool(info["needs_scaling"]),
            "description": info["description"],
        })
    model_df = pd.DataFrame(model_rows)

    return family_df, model_df


def build_sample_size_schedule_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Create a table summarizing the nested sample-size schedule and number of repeats.
    """
    rows = []
    for (suite_name, target_col, test_seed, n_train), sub in manifest.groupby(["suite_name", "target_col", "test_seed", "n_train"]):
        rows.append({
            "suite_name": suite_name,
            "target_col": target_col,
            "test_seed": test_seed,
            "n_train": int(n_train),
            "n_repeats": int(sub["subsample_seed"].nunique()),
            "n_methods": int(sub["descriptor_family"].nunique() * sub["model_name"].nunique()),
            "n_jobs": len(sub),
        })
    return pd.DataFrame(rows).sort_values(["suite_name", "target_col", "test_seed", "n_train"])


def build_external_test_performance_summary(
    agg_perf: pd.DataFrame,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    """
    Build the main performance summary table at selected sample sizes.
    """
    main = agg_perf[
        (agg_perf["suite_name"] == "primary") &
        (agg_perf["target_col"] == cfg.main_target) &
        (agg_perf["test_seed"] == cfg.primary_test_seed)
    ].copy()

    sizes = sorted(main["n_train"].unique())
    selected = sorted(set([sizes[0], sizes[min(2, len(sizes)-1)], sizes[min(4, len(sizes)-1)], sizes[-1]]))

    sub = main[main["n_train"].isin(selected)].copy()
    sub = sub.sort_values(["n_train", "rmse_mean"])
    keep_cols = [
        "n_train", "descriptor_family", "model_name", "method_label",
        "rmse_mean", "rmse_ci_low", "rmse_ci_high",
        "mae_mean", "r2_mean", "spearman_mean",
    ]
    return sub[keep_cols].reset_index(drop=True)


def save_all_tables(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    df_metrics: pd.DataFrame,
    agg_perf: pd.DataFrame,
    ranking_stability: pd.DataFrame,
    screening: pd.DataFrame,
    stable_table: pd.DataFrame,
    pairwise_results: Dict[Tuple[str, int, int], pd.DataFrame],
    feature_df: pd.DataFrame,
    feature_conv: pd.DataFrame,
    additional_analysis: Dict[str, pd.DataFrame],
    cfg: ProjectConfig,
    paths: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """
    Save all manuscript and SI tables in CSV, pickle, and LaTeX-friendly formats.
    """
    family_df, model_df = build_descriptor_model_definition_tables(df, cfg)
    schedule_df = build_sample_size_schedule_table(manifest)
    perf_summary_df = build_external_test_performance_summary(agg_perf, cfg)

    tables_to_save = {
        "table_1a_descriptor_family_definitions": family_df,
        "table_1b_model_definitions": model_df,
        "table_2_sample_size_schedule": schedule_df,
        "table_3_stable_conclusion_threshold": stable_table,
        "table_4_external_test_performance_summary": perf_summary_df,
        "all_job_metrics": df_metrics,
        "aggregated_performance": agg_perf,
        "ranking_stability": ranking_stability,
        "screening_reproducibility": screening,
        "feature_effect_importances_all": feature_df,
        "feature_effect_convergence_summary": feature_conv,
    }

    for name, table in tables_to_save.items():
        save_dataframe(
            table,
            paths["results"] / f"{name}.csv",
            paths["results"] / f"{name}.pkl.gz",
        )

    for name, table in additional_analysis.items():
        save_dataframe(
            table,
            paths["additional_analysis_tables"] / f"{name}.csv",
            paths["additional_analysis_tables"] / f"{name}.pkl.gz",
        )
        save_dataframe(
            table,
            paths["si_tables"] / f"{name}.csv",
            paths["si_tables"] / f"{name}.pkl.gz",
        )

    # Save main manuscript tables
    main_tables = {
        "table1a_descriptor_families": family_df,
        "table1b_model_definitions": model_df,
        "table2_sample_size_schedule": schedule_df[schedule_df["suite_name"] == "primary"],
        "table3_stable_conclusion_threshold": stable_table[stable_table["target_col"] == cfg.main_target],
        "table4_external_test_performance_summary": perf_summary_df,
    }
    captions = {
        "table1a_descriptor_families": "Descriptor-family definitions used in the small-data MOF benchmark.",
        "table1b_model_definitions": "Model definitions used in the small-data MOF benchmark.",
        "table2_sample_size_schedule": "Sample-size schedule and number of repeated runs.",
        "table3_stable_conclusion_threshold": "Benchmark-maturity threshold table for the main target.",
        "table4_external_test_performance_summary": "External-test performance summary at selected training sizes for the main target.",
    }
    labels = {
        "table1a_descriptor_families": "tab:descriptor_families",
        "table1b_model_definitions": "tab:model_definitions",
        "table2_sample_size_schedule": "tab:sample_size_schedule",
        "table3_stable_conclusion_threshold": "tab:stability_threshold",
        "table4_external_test_performance_summary": "tab:external_test_summary",
    }

    for name, table in main_tables.items():
        save_dataframe(
            table,
            paths["main_tables"] / f"{name}.csv",
            paths["main_tables"] / f"{name}.pkl.gz",
        )
        df_to_latex_table(
            table,
            paths["main_tables"] / f"{name}.tex",
            caption=captions[name],
            label=labels[name],
        )

    # Save SI tables
    si_tables = {
        "si_all_job_metrics": df_metrics,
        "si_aggregated_performance": agg_perf,
        "si_ranking_stability": ranking_stability,
        "si_screening_reproducibility": screening,
        "si_feature_effect_importances": feature_df,
        "si_feature_effect_convergence": feature_conv,
    }
    for name, table in si_tables.items():
        save_dataframe(
            table,
            paths["si_tables"] / f"{name}.csv",
            paths["si_tables"] / f"{name}.pkl.gz",
        )

    # Pairwise superiority matrices: save each one separately.
    for (target_col, test_seed, n_train), mat in pairwise_results.items():
        stem = f"pairwise_superiority__{slugify(target_col)}__testseed_{test_seed}__n_{n_train}"
        save_dataframe(mat.reset_index().rename(columns={"index": "method_label"}), paths["pairwise_tables"] / f"{stem}.csv")
        save_dataframe(mat.reset_index().rename(columns={"index": "method_label"}), paths["si_tables"] / f"{stem}.csv")

    logger.info("All tables saved.")


# =============================================================================
# SECTION 16. FIGURE HELPERS
# =============================================================================


def save_figure(fig: plt.Figure, base_path_no_ext: Path) -> None:
    """
    Save each figure as both PNG and PDF for manuscript flexibility.
    """
    ensure_dir(base_path_no_ext.parent)
    fig.savefig(str(base_path_no_ext) + ".png", dpi=400, bbox_inches="tight")
    fig.savefig(str(base_path_no_ext) + ".pdf", bbox_inches="tight")
    if CONFIG.export_svg_figures:
        fig.savefig(str(base_path_no_ext) + ".svg", bbox_inches="tight")
    plt.close(fig)


def save_axis_as_figure(
    source_ax: plt.Axes,
    base_path_no_ext: Path,
    figsize: Optional[Tuple[float, float]] = None,
    keep_legend: bool = True,
) -> None:
    """
    Save a single axis as its own standalone figure.

    This is useful when the manuscript uses a composite figure, but the user
    also wants each panel separately for manual editing later.
    """
    fig0 = source_ax.figure
    bbox = source_ax.get_tightbbox(fig0.canvas.get_renderer()).expanded(1.04, 1.08)
    bbox_inches = bbox.transformed(fig0.dpi_scale_trans.inverted())

    ensure_dir(base_path_no_ext.parent)
    fig0.savefig(str(base_path_no_ext) + ".png", dpi=400, bbox_inches=bbox_inches)
    fig0.savefig(str(base_path_no_ext) + ".pdf", bbox_inches=bbox_inches)
    if CONFIG.export_svg_figures:
        fig0.savefig(str(base_path_no_ext) + ".svg", bbox_inches=bbox_inches)



def save_figure_dataframes(
    dataframes: Dict[str, pd.DataFrame],
    figure_stem: str,
    primary_dir: Path,
    secondary_dir: Optional[Path] = None,
) -> None:
    """
    Save one or more DataFrames containing the exact numerical values plotted in a figure.

    Each DataFrame is written as:
        <figure_stem>__<dataset_name>.csv
        <figure_stem>__<dataset_name>.pkl.gz

    Parameters
    ----------
    dataframes
        Mapping from a short dataset name to the DataFrame containing the
        numbers behind the plotted figure.
    figure_stem
        Base figure name, for example "Figure2_learning_curves_main_target".
    primary_dir
        Main destination directory (for example manuscript_assets/figure_data_main).
    secondary_dir
        Optional mirror directory where the same files are also saved. This is
        useful for keeping one publication-facing location and one central
        results location.
    """
    for dataset_name, df_plot in dataframes.items():
        stem = f"{figure_stem}__{dataset_name}"
        csv_path = primary_dir / f"{stem}.csv"
        pkl_path = primary_dir / f"{stem}.pkl.gz"
        save_dataframe(df_plot, csv_path, pkl_path)
        if secondary_dir is not None:
            save_dataframe(df_plot, secondary_dir / f"{stem}.csv", secondary_dir / f"{stem}.pkl.gz")


def choose_best_method_at_each_n(agg_perf: pd.DataFrame, suite_name: str, target_col: str, test_seed: int) -> pd.DataFrame:
    """
    Return the best mean-RMSE method at each sample size.
    """
    sub = agg_perf[
        (agg_perf["suite_name"] == suite_name) &
        (agg_perf["target_col"] == target_col) &
        (agg_perf["test_seed"] == test_seed)
    ].copy()
    idx = sub.groupby("n_train")["rmse_mean"].idxmin()
    return sub.loc[idx].sort_values("n_train").reset_index(drop=True)


def make_workflow_figure(cfg: ProjectConfig, paths: Dict[str, Path]) -> None:
    """
    Figure 1: Benchmark-maturity workflow schematic.

    This is intentionally created inside Python so the project is self-contained.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    boxes = [
        (0.03, 0.35, 0.14, 0.30, "Load\nclean_data.csv"),
        (0.21, 0.35, 0.14, 0.30, "Choose main target\nand fixed\nexternal test set"),
        (0.39, 0.35, 0.14, 0.30, "Build nested\ntraining subsamples\n(n = 500 ... full)"),
        (0.57, 0.35, 0.14, 0.30, "Train repeated\nmodel/descriptor\ncombinations"),
        (0.75, 0.35, 0.14, 0.30, "Quantify\nperformance,\nuncertainty,\nstability"),
    ]

    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=12)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i+1][0]
        y = boxes[i][1] + boxes[i][3] / 2
        ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="->", lw=2))

    ax.text(
        0.5, 0.12,
        "Output: learning curves, ranking stability, screening reproducibility, pairwise superiority, and feature-effect convergence.",
        ha="center", va="center", fontsize=12
    )
    ax.set_title("Figure 1. Benchmark-maturity workflow", fontsize=16, pad=20)

    workflow_df = pd.DataFrame([
        {"step_order": 1, "box_label": "Load clean_data.csv", "role": "Input data loading"},
        {"step_order": 2, "box_label": "Choose main target and fixed external test set", "role": "Problem definition and held-out evaluation"},
        {"step_order": 3, "box_label": "Build nested training subsamples (n = 500 ... full)", "role": "Sample-size design"},
        {"step_order": 4, "box_label": "Train repeated model/descriptor combinations", "role": "Repeated benchmark execution"},
        {"step_order": 5, "box_label": "Quantify performance, uncertainty, stability", "role": "Post-processing and scientific conclusions"},
    ])
    save_figure_dataframes(
        {"workflow_steps": workflow_df},
        "Figure1_benchmark_maturity_workflow",
        paths["main_figure_data"],
        paths["all_figure_data"],
    )
    save_figure(fig, paths["main_figures"] / "Figure1_benchmark_maturity_workflow")


def make_learning_curve_figure(
    agg_perf: pd.DataFrame,
    cfg: ProjectConfig,
    paths: Dict[str, Path],
) -> None:
    """
    Figure 2: Learning curves with confidence bands for the main target.
    """
    main = agg_perf[
        (agg_perf["suite_name"] == "primary") &
        (agg_perf["target_col"] == cfg.main_target) &
        (agg_perf["test_seed"] == cfg.primary_test_seed)
    ].copy()

    methods = main["method_label"].unique().tolist()
    methods_sorted = (
        main[main["n_train"] == main["n_train"].max()]
        .sort_values("rmse_mean")["method_label"]
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(13, 8))

    for method in methods_sorted:
        sub = main[main["method_label"] == method].sort_values("n_train")
        ax.plot(sub["n_train"], sub["rmse_mean"], marker="o", linewidth=1.8, label=method)
        ax.fill_between(sub["n_train"], sub["rmse_ci_low"], sub["rmse_ci_high"], alpha=0.12)

    ax.set_xscale("log")
    ax.set_xlabel("Training sample size (log scale)")
    ax.set_ylabel("External-test RMSE")
    ax.set_title(f"Figure 2. Learning curves with 95% confidence bands\nMain target: {cfg.main_target}")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)

    learning_curve_df = main[
        [
            "suite_name", "target_col", "test_seed", "n_train", "descriptor_family", "model_name", "method_label",
            "rmse_mean", "rmse_ci_low", "rmse_ci_high", "rmse_std",
            "mae_mean", "mae_ci_low", "mae_ci_high",
            "r2_mean", "r2_ci_low", "r2_ci_high",
            "spearman_mean", "spearman_ci_low", "spearman_ci_high",
        ]
    ].sort_values(["method_label", "n_train"]).reset_index(drop=True)
    save_figure_dataframes(
        {"learning_curve_values": learning_curve_df},
        "Figure2_learning_curves_main_target",
        paths["main_figure_data"],
        paths["all_figure_data"],
    )
    save_figure(fig, paths["main_figures"] / "Figure2_learning_curves_main_target")



def make_ranking_stability_figure(
    ranking_stability: pd.DataFrame,
    agg_perf: pd.DataFrame,
    cfg: ProjectConfig,
    paths: Dict[str, Path],
) -> None:
    """
    Figure 3: Method-ranking stability phase map.

    Panel A: probability each method is ranked first at each n.
    Panel B: average rank correlation vs full-data ranking.
    """
    rs = ranking_stability[
        (ranking_stability["suite_name"] == "primary") &
        (ranking_stability["target_col"] == cfg.main_target) &
        (ranking_stability["test_seed"] == cfg.primary_test_seed)
    ].copy()

    main = agg_perf[
        (agg_perf["suite_name"] == "primary") &
        (agg_perf["target_col"] == cfg.main_target) &
        (agg_perf["test_seed"] == cfg.primary_test_seed)
    ].copy()
    methods_order = (
        main[main["n_train"] == main["n_train"].max()]
        .sort_values("rmse_mean")["method_label"]
        .tolist()
    )

    matrix = []
    labels = []
    for method in methods_order:
        c = f"p_rank1__{slugify(method)}"
        labels.append(method)
        if c in rs.columns:
            matrix.append(rs.sort_values("n_train")[c].fillna(0).values)
        else:
            matrix.append(np.zeros(len(rs)))
    matrix = np.array(matrix)

    fig = plt.figure(figsize=(15.2, 9.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.25, 1.0], hspace=0.42)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(matrix, aspect="auto", interpolation="nearest")
    ax0.set_yticks(range(len(labels)))
    ax0.set_yticklabels(labels, fontsize=8)
    ax0.set_xticks(range(len(rs)))
    ax0.set_xticklabels(rs.sort_values("n_train")["n_train"].tolist(), rotation=45)
    ax0.set_title("Panel A. Probability a method is ranked first", pad=10)
    ax0.set_xlabel("Training size", labelpad=14)
    ax0.set_ylabel("Method")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.022, pad=0.03)
    cbar.set_label("Probability")

    ax1 = fig.add_subplot(gs[1, 0])
    rs_sorted = rs.sort_values("n_train")
    ax1.plot(rs_sorted["n_train"], rs_sorted["top1_consensus_probability"], marker="o", label="Top-1 consensus probability")
    ax1.plot(rs_sorted["n_train"], rs_sorted["mean_rank_spearman_vs_full"], marker="s", label="Mean rank Spearman vs full")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Training size (log scale)")
    ax1.set_ylabel("Stability")
    ax1.set_title("Panel B. Ranking preservation metrics", pad=10)
    ax1.legend(frameon=False, loc="lower left")

    ax0.tick_params(axis="x", pad=10)
    ax1.tick_params(axis="x", pad=6)
    fig.suptitle("Figure 3. Method-ranking stability phase map", fontsize=16, y=0.982)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.11, right=0.965)

    rank1_prob_long = []
    rs_sorted_for_export = rs.sort_values("n_train").reset_index(drop=True)
    for _, row in rs_sorted_for_export.iterrows():
        for method in methods_order:
            col = f"p_rank1__{slugify(method)}"
            rank1_prob_long.append({
                "n_train": int(row["n_train"]),
                "method_label": method,
                "p_rank1": float(row[col]) if col in row.index and pd.notna(row[col]) else 0.0,
            })
    rank1_prob_df = pd.DataFrame(rank1_prob_long)
    stability_summary_df = rs_sorted_for_export[[
        "suite_name", "target_col", "test_seed", "n_train",
        "top1_consensus_probability", "mean_rank_spearman_vs_full"
    ]].copy()
    save_figure_dataframes(
        {
            "panelA_rank1_probabilities": rank1_prob_df,
            "panelB_stability_summary": stability_summary_df,
        },
        "Figure3_ranking_stability_phase_map",
        paths["main_figure_data"],
        paths["all_figure_data"],
    )

    # Save separate panels for manual editing.
    fig.canvas.draw()
    save_axis_as_figure(ax0, paths["main_figures"] / "Figure3_ranking_stability_phase_map__PanelA")
    save_axis_as_figure(ax1, paths["main_figures"] / "Figure3_ranking_stability_phase_map__PanelB")
    save_figure(fig, paths["main_figures"] / "Figure3_ranking_stability_phase_map")




def make_screening_reproducibility_figure(
    screening: pd.DataFrame,
    agg_perf: pd.DataFrame,
    cfg: ProjectConfig,
    paths: Dict[str, Path],
) -> None:
    """
    Figure 4: Screening reproducibility for the best method at each n.
    """
    main_best = choose_best_method_at_each_n(agg_perf, "primary", cfg.main_target, cfg.primary_test_seed)
    screen_main = screening[
        (screening["suite_name"] == "primary") &
        (screening["target_col"] == cfg.main_target) &
        (screening["test_seed"] == cfg.primary_test_seed)
    ].copy()

    metric_overlap = f"topk_overlap_frac_{cfg.topk_fraction:.3f}".replace(".", "p")
    metric_elite = f"elite_enrichment_{cfg.topk_fraction:.3f}".replace(".", "p")

    plot_rows = []
    for _, r in main_best.iterrows():
        sub = screen_main[
            (screen_main["n_train"] == r["n_train"]) &
            (screen_main["method_label"] == r["method_label"])
        ]
        if len(sub) == 0:
            continue
        s = sub.iloc[0]
        plot_rows.append({
            "n_train": int(r["n_train"]),
            "method_label": r["method_label"],
            "topk_overlap_mean": s[f"{metric_overlap}_mean"],
            "topk_overlap_std": s[f"{metric_overlap}_std"],
            "elite_enrichment_mean": s[f"{metric_elite}_mean"],
            "elite_enrichment_std": s[f"{metric_elite}_std"],
        })
    plot_df = pd.DataFrame(plot_rows).sort_values("n_train")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(plot_df["n_train"], plot_df["topk_overlap_mean"], marker="o")
    axes[0].fill_between(
        plot_df["n_train"],
        plot_df["topk_overlap_mean"] - plot_df["topk_overlap_std"],
        plot_df["topk_overlap_mean"] + plot_df["topk_overlap_std"],
        alpha=0.15,
    )
    axes[0].set_ylabel("Top-k overlap")
    axes[0].set_title("Panel A. Recovery of true elite candidates")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(plot_df["n_train"], plot_df["elite_enrichment_mean"], marker="s")
    axes[1].fill_between(
        plot_df["n_train"],
        plot_df["elite_enrichment_mean"] - plot_df["elite_enrichment_std"],
        plot_df["elite_enrichment_mean"] + plot_df["elite_enrichment_std"],
        alpha=0.15,
    )
    axes[1].set_ylabel("Elite enrichment factor")
    axes[1].set_xlabel("Training size")
    axes[1].set_title("Panel B. Screening enrichment stability")
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xscale("log")

    fig.suptitle("Figure 4. Screening reproducibility versus sample size", fontsize=16)

    screening_plot_export = plot_df.copy()
    screening_plot_export["topk_overlap_lower"] = screening_plot_export["topk_overlap_mean"] - screening_plot_export["topk_overlap_std"]
    screening_plot_export["topk_overlap_upper"] = screening_plot_export["topk_overlap_mean"] + screening_plot_export["topk_overlap_std"]
    screening_plot_export["elite_enrichment_lower"] = screening_plot_export["elite_enrichment_mean"] - screening_plot_export["elite_enrichment_std"]
    screening_plot_export["elite_enrichment_upper"] = screening_plot_export["elite_enrichment_mean"] + screening_plot_export["elite_enrichment_std"]
    save_figure_dataframes(
        {"screening_reproducibility_values": screening_plot_export},
        "Figure4_screening_reproducibility",
        paths["main_figure_data"],
        paths["all_figure_data"],
    )
    fig.canvas.draw()
    save_axis_as_figure(axes[0], paths["main_figures"] / "Figure4_screening_reproducibility__PanelA")
    save_axis_as_figure(axes[1], paths["main_figures"] / "Figure4_screening_reproducibility__PanelB")
    save_figure(fig, paths["main_figures"] / "Figure4_screening_reproducibility")



def make_pairwise_superiority_figure(
    pairwise_results: Dict[Tuple[str, int, int], pd.DataFrame],
    agg_perf: pd.DataFrame,
    cfg: ProjectConfig,
    paths: Dict[str, Path],
) -> None:
    """
    Figure 5: Pairwise probability-of-superiority matrix at selected sizes.

    Layout note:
    A dedicated colorbar axis is used so the probability bar stays flush to the
    far right instead of drifting into the last heatmap panel.
    """
    main_sub = agg_perf[
        (agg_perf["suite_name"] == "primary") &
        (agg_perf["target_col"] == cfg.main_target) &
        (agg_perf["test_seed"] == cfg.primary_test_seed)
    ].copy()
    full_n = int(main_sub["n_train"].max())
    selected_sizes = sorted(set([n for n in cfg.pairwise_selected_sizes if n in main_sub["n_train"].unique()] + [full_n]))

    n_panels = len(selected_sizes)
    fig = plt.figure(figsize=(7.4 * n_panels + 0.9, 7.0))
    gs = fig.add_gridspec(
        1, n_panels + 1,
        width_ratios=[1] * n_panels + [0.06],
        wspace=0.32,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    cax = fig.add_subplot(gs[0, -1])

    methods_order = (
        main_sub[main_sub["n_train"] == full_n]
        .sort_values("rmse_mean")["method_label"]
        .tolist()
    )

    ims = []
    for ax, n_train in zip(axes, selected_sizes):
        mat = pairwise_results.get((cfg.main_target, cfg.primary_test_seed, int(n_train)))
        if mat is None:
            ax.axis("off")
            continue

        mat = mat.loc[methods_order, methods_order]
        im = ax.imshow(mat.values.astype(float), vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        ims.append(im)
        ax.set_title(f"n = {n_train}", pad=9)
        ax.set_xticks(range(len(methods_order)))
        ax.set_xticklabels(methods_order, rotation=90, fontsize=7)
        ax.set_yticks(range(len(methods_order)))
        ax.set_yticklabels(methods_order, fontsize=7)
        ax.tick_params(axis="y", pad=4)
        ax.tick_params(axis="x", pad=3)
        ax.set_anchor('W')

    fig.suptitle(
        "Figure 5. Pairwise probability-of-superiority matrix\nP(method A has lower RMSE than method B)",
        fontsize=15,
        y=0.975,
    )
    if ims:
        cbar = fig.colorbar(ims[-1], cax=cax)
        cbar.set_label("Probability")
    else:
        cax.axis("off")
    fig.subplots_adjust(left=0.065, right=0.965, bottom=0.20, top=0.86)

    pairwise_long_rows = []
    for n_train in selected_sizes:
        mat = pairwise_results.get((cfg.main_target, cfg.primary_test_seed, int(n_train)))
        if mat is None:
            continue
        mat = mat.copy()
        for row_method in mat.index:
            for col_method in mat.columns:
                pairwise_long_rows.append({
                    "target_col": cfg.main_target,
                    "test_seed": cfg.primary_test_seed,
                    "n_train": int(n_train),
                    "row_method": row_method,
                    "column_method": col_method,
                    "probability_row_beats_column": float(mat.loc[row_method, col_method]) if pd.notna(mat.loc[row_method, col_method]) else np.nan,
                })
    pairwise_long_df = pd.DataFrame(pairwise_long_rows)
    save_figure_dataframes(
        {"pairwise_superiority_long": pairwise_long_df},
        "Figure5_pairwise_probability_superiority",
        paths["main_figure_data"],
        paths["all_figure_data"],
    )

    # Save separate heatmap panels for manual editing.
    fig.canvas.draw()
    for ax, n_train in zip(axes, selected_sizes):
        save_axis_as_figure(ax, paths["main_figures"] / f"Figure5_pairwise_probability_superiority__n_{n_train}")
    save_figure(fig, paths["main_figures"] / "Figure5_pairwise_probability_superiority")



def make_feature_effect_convergence_figure(
    feature_df: pd.DataFrame,
    feature_conv: pd.DataFrame,
    cfg: ProjectConfig,
    paths: Dict[str, Path],
) -> None:
    """
    Figure 6: Convergence of feature effects / interpretation stability.

    Panel A: Spearman correlation of feature rankings vs full-data ranking.
    Panel B: Jaccard similarity of top-10 feature sets vs full-data ranking.
    Panel C: Heatmap of mean importance ranks for top features across n.
    """
    full_n = int(feature_df["n_train"].max())
    feature_mean = (
        feature_df.groupby(["n_train", "feature"])["importance_mean"]
        .mean()
        .reset_index()
    )

    full_ref = (
        feature_mean[feature_mean["n_train"] == full_n]
        .sort_values("importance_mean", ascending=False)
        .head(cfg.feature_effect_topn)["feature"]
        .tolist()
    )

    rank_matrix_rows = []
    sizes_sorted = sorted(feature_mean["n_train"].unique())
    for feat in full_ref:
        row = []
        for n_train in sizes_sorted:
            sub = feature_mean[feature_mean["n_train"] == n_train].sort_values("importance_mean", ascending=False)
            rank_map = {f: i + 1 for i, f in enumerate(sub["feature"].tolist())}
            row.append(rank_map.get(feat, np.nan))
        rank_matrix_rows.append(row)
    rank_matrix = np.array(rank_matrix_rows)

    fig = plt.figure(figsize=(14, 10.3))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2], hspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(feature_conv["n_train"], feature_conv["rank_spearman_vs_full_mean"], marker="o")
    ax1.fill_between(
        feature_conv["n_train"],
        feature_conv["rank_spearman_vs_full_mean"] - feature_conv["rank_spearman_vs_full_std"],
        feature_conv["rank_spearman_vs_full_mean"] + feature_conv["rank_spearman_vs_full_std"],
        alpha=0.15,
    )
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Spearman")
    ax1.set_title("Panel A. Rank correlation of feature effects vs full-data ranking")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(feature_conv["n_train"], feature_conv["top10_jaccard_vs_full_mean"], marker="s")
    ax2.fill_between(
        feature_conv["n_train"],
        feature_conv["top10_jaccard_vs_full_mean"] - feature_conv["top10_jaccard_vs_full_std"],
        feature_conv["top10_jaccard_vs_full_mean"] + feature_conv["top10_jaccard_vs_full_std"],
        alpha=0.15,
    )
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Jaccard")
    ax2.set_title("Panel B. Top-10 feature-set similarity vs full-data ranking")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2, 0])
    im = ax3.imshow(rank_matrix, aspect="auto", interpolation="nearest")
    ax3.set_yticks(range(len(full_ref)))
    ax3.set_yticklabels(full_ref, fontsize=8)
    ax3.set_xticks(range(len(sizes_sorted)))
    ax3.set_xticklabels(sizes_sorted, rotation=45)
    ax3.set_xlabel("Training size")
    ax3.set_ylabel("Reference top features")
    ax3.set_title("Panel C. Mean importance-rank trajectory of top full-data features")
    fig.colorbar(im, ax=ax3, fraction=0.02, pad=0.02, label="Rank (lower is more important)")

    fig.suptitle("Figure 6. Convergence of feature effects with growing data", fontsize=16)

    feature_conv_export = feature_conv.copy()
    feature_conv_export["rank_spearman_lower"] = feature_conv_export["rank_spearman_vs_full_mean"] - feature_conv_export["rank_spearman_vs_full_std"]
    feature_conv_export["rank_spearman_upper"] = feature_conv_export["rank_spearman_vs_full_mean"] + feature_conv_export["rank_spearman_vs_full_std"]
    feature_conv_export["top10_jaccard_lower"] = feature_conv_export["top10_jaccard_vs_full_mean"] - feature_conv_export["top10_jaccard_vs_full_std"]
    feature_conv_export["top10_jaccard_upper"] = feature_conv_export["top10_jaccard_vs_full_mean"] + feature_conv_export["top10_jaccard_vs_full_std"]

    rank_matrix_long_rows = []
    for feat, row_vals in zip(full_ref, rank_matrix):
        for n_train, rank_val in zip(sizes_sorted, row_vals):
            rank_matrix_long_rows.append({
                "feature": feat,
                "n_train": int(n_train),
                "mean_importance_rank": float(rank_val) if pd.notna(rank_val) else np.nan,
            })
    rank_matrix_long_df = pd.DataFrame(rank_matrix_long_rows)

    feature_mean_export = feature_mean.sort_values(["feature", "n_train"]).reset_index(drop=True)
    save_figure_dataframes(
        {
            "panelA_panelB_convergence_summary": feature_conv_export,
            "panelC_rank_matrix_long": rank_matrix_long_df,
            "all_feature_importance_means": feature_mean_export,
        },
        "Figure6_feature_effect_convergence",
        paths["main_figure_data"],
        paths["all_figure_data"],
    )
    fig.canvas.draw()
    save_axis_as_figure(ax1, paths["main_figures"] / "Figure6_feature_effect_convergence__PanelA")
    save_axis_as_figure(ax2, paths["main_figures"] / "Figure6_feature_effect_convergence__PanelB")
    save_axis_as_figure(ax3, paths["main_figures"] / "Figure6_feature_effect_convergence__PanelC")
    save_figure(fig, paths["main_figures"] / "Figure6_feature_effect_convergence")



def make_main_conclusion_figure(
    agg_perf: pd.DataFrame,
    ranking_stability: pd.DataFrame,
    screening: pd.DataFrame,
    stable_table: pd.DataFrame,
    additional_analysis: Dict[str, pd.DataFrame],
    cfg: ProjectConfig,
    paths: Dict[str, Path],
) -> None:
    """
    New main-text synthesis figure with stronger visual conclusions.

    Panel A: best-achieved RMSE vs sample size with benchmark-maturity threshold.
    Panel B: ranking-stability metrics vs sample size.
    Panel C: screening reproducibility metrics for the best method at each n.
    Panel D: full-data leaderboard balancing accuracy and correlation.

    In Panel D, colors encode descriptor family and marker shapes encode model
    type. Per-point text annotations are intentionally removed for readability.
    Small deterministic offsets are applied so nearly overlapping methods do not
    collapse into one strange composite symbol.
    """
    main = agg_perf[
        (agg_perf["suite_name"] == "primary") &
        (agg_perf["target_col"] == cfg.main_target) &
        (agg_perf["test_seed"] == cfg.primary_test_seed)
    ].copy()
    rs = ranking_stability[
        (ranking_stability["suite_name"] == "primary") &
        (ranking_stability["target_col"] == cfg.main_target) &
        (ranking_stability["test_seed"] == cfg.primary_test_seed)
    ].copy()
    sc = screening[
        (screening["suite_name"] == "primary") &
        (screening["target_col"] == cfg.main_target) &
        (screening["test_seed"] == cfg.primary_test_seed)
    ].copy()
    stable_main = stable_table[stable_table["target_col"] == cfg.main_target].copy()
    pareto = additional_analysis["method_pareto_summary"].copy()

    best_each_n = choose_best_method_at_each_n(agg_perf, "primary", cfg.main_target, cfg.primary_test_seed)
    best_each_n = best_each_n.sort_values("n_train").reset_index(drop=True)
    rs = rs.sort_values("n_train").reset_index(drop=True)
    stable_main = stable_main.sort_values("n_train").reset_index(drop=True)

    threshold_vals = stable_main["benchmark_maturity_threshold_n"].dropna().unique().tolist()
    threshold_n = int(threshold_vals[0]) if len(threshold_vals) else None

    metric_overlap = f"topk_overlap_frac_{cfg.topk_fraction:.3f}".replace(".", "p")
    metric_elite = f"elite_enrichment_{cfg.topk_fraction:.3f}".replace(".", "p")

    screen_rows = []
    for _, r in best_each_n.iterrows():
        sub = sc[(sc["n_train"] == r["n_train"]) & (sc["method_label"] == r["method_label"])]
        if len(sub) == 0:
            continue
        s = sub.iloc[0]
        screen_rows.append({
            "n_train": int(r["n_train"]),
            "best_method_label": r["method_label"],
            "topk_overlap_mean": float(s.get(f"{metric_overlap}_mean", np.nan)),
            "topk_overlap_std": float(s.get(f"{metric_overlap}_std", np.nan)),
            "elite_enrichment_mean": float(s.get(f"{metric_elite}_mean", np.nan)),
            "elite_enrichment_std": float(s.get(f"{metric_elite}_std", np.nan)),
        })
    screen_df = pd.DataFrame(screen_rows).sort_values("n_train").reset_index(drop=True)

    family_color_map = {
        "geometry_only": "#4C78A8",
        "enriched_interpretable": "#F58518",
        "topology_only": "#54A24B",
        "geometry_plus_topology": "#B279A2",
    }
    family_legend_labels = {
        "geometry_only": "Geometry only",
        "enriched_interpretable": "Enriched interpretable",
        "topology_only": "Topology only",
        "geometry_plus_topology": "Geometry + topology",
    }
    model_marker_map = {
        "ridge": "o",
        "rf": "s",
        "hgb": "^",
        "mlp": "D",
    }
    model_legend_labels = {
        "ridge": "Ridge",
        "rf": "RF",
        "hgb": "HGB",
        "mlp": "MLP",
    }

    fig = plt.figure(figsize=(16.6, 11.0))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22)

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(best_each_n["n_train"], best_each_n["rmse_mean"], color="#1f77b4", marker="o", linewidth=2.8)
    ax1.fill_between(
        best_each_n["n_train"],
        best_each_n["rmse_ci_low"],
        best_each_n["rmse_ci_high"],
        color="#1f77b4",
        alpha=0.12,
    )
    if threshold_n is not None:
        ax1.axvline(threshold_n, color="#D62728", linestyle="--", linewidth=2.0)
        ax1.axvspan(threshold_n, best_each_n["n_train"].max(), color="#2CA02C", alpha=0.08)
        thr_y = float(best_each_n.loc[best_each_n["n_train"] >= threshold_n, "rmse_mean"].iloc[0])
        ax1.annotate(
            f"Maturity threshold\nn ≈ {threshold_n:,}",
            xy=(threshold_n, thr_y),
            xytext=(12, 18),
            textcoords="offset points",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75"),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="0.35"),
        )
    ax1.set_xscale("log")
    ax1.set_xlabel("Training size (log scale)")
    ax1.set_ylabel("Best-achieved RMSE")
    ax1.set_title("Panel A. Accuracy improves rapidly, then enters a mature regime")
    add_panel_label(ax1, "A")

    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(rs["n_train"], rs["top1_consensus_probability"], marker="o", linewidth=2.3, label="Top-1 consensus")
    ax2.plot(rs["n_train"], rs["mean_rank_spearman_vs_full"], marker="s", linewidth=2.3, label="Rank Spearman vs full")
    if threshold_n is not None:
        ax2.axvline(threshold_n, color="#D62728", linestyle="--", linewidth=2.0)
        ax2.axhline(cfg.stability_top1_consensus_threshold, color="0.45", linestyle=":", linewidth=1.2)
        ax2.axhline(cfg.stability_rankcorr_threshold, color="0.65", linestyle=":", linewidth=1.2)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.04)
    ax2.set_xlabel("Training size (log scale)")
    ax2.set_ylabel("Stability metric")
    ax2.set_title("Panel B. Method ranking becomes highly reproducible with data")
    ax2.legend(frameon=False, loc="lower right")
    add_panel_label(ax2, "B")

    # Panel C
    ax3 = fig.add_subplot(gs[1, 0])
    if len(screen_df):
        ax3.plot(screen_df["n_train"], screen_df["topk_overlap_mean"], marker="o", linewidth=2.3, label="Top-k overlap")
        ax3.fill_between(
            screen_df["n_train"],
            screen_df["topk_overlap_mean"] - screen_df["topk_overlap_std"],
            screen_df["topk_overlap_mean"] + screen_df["topk_overlap_std"],
            alpha=0.12,
        )
        ax3b = ax3.twinx()
        ax3b.plot(screen_df["n_train"], screen_df["elite_enrichment_mean"], marker="s", linewidth=2.3, color="#FF7F0E", label="Elite enrichment")
        ax3b.fill_between(
            screen_df["n_train"],
            screen_df["elite_enrichment_mean"] - screen_df["elite_enrichment_std"],
            screen_df["elite_enrichment_mean"] + screen_df["elite_enrichment_std"],
            color="#FF7F0E",
            alpha=0.10,
        )
        if threshold_n is not None:
            ax3.axvline(threshold_n, color="#D62728", linestyle="--", linewidth=2.0)
        ax3.set_xscale("log")
        ax3.set_xlabel("Training size (log scale)")
        ax3.set_ylabel("Top-k overlap")
        ax3b.set_ylabel("Elite enrichment factor")
        ax3.set_title("Panel C. Screening outcomes also stabilize, not just RMSE")
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower right")
    add_panel_label(ax3, "C")

    # Panel D
    ax4 = fig.add_subplot(gs[1, 1])
    pareto_plot = pareto.copy()
    pareto_plot["pretty_label"] = pareto_plot["method_label"].map(pretty_method_label)

    # Small deterministic offsets prevent overlapping markers from forming odd composite symbols.
    marker_offsets = {
        "ridge": (-0.004, 0.0000),
        "rf": (-0.0015, 0.0018),
        "hgb": (0.0015, -0.0018),
        "mlp": (0.004, 0.0000),
    }
    plot_x, plot_y = [], []
    for _, row in pareto_plot.iterrows():
        dx, dy = marker_offsets.get(row["model_name"], (0.0, 0.0))
        x = float(row["rmse_mean"]) + dx
        y = float(row["spearman_mean"]) + dy
        plot_x.append(x)
        plot_y.append(y)
        ax4.scatter(
            x,
            y,
            s=115,
            alpha=0.95,
            color=family_color_map.get(row["descriptor_family"], "#777777"),
            marker=model_marker_map.get(row["model_name"], "o"),
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
        )

    ax4.set_xlabel("Full-data RMSE")
    ax4.set_ylabel("Full-data Spearman")
    ax4.set_title("Panel D. The final leaderboard reveals the best accuracy–ranking trade-off")
    add_panel_label(ax4, "D")
    ax4.set_xlim(min(plot_x) - 0.015, max(plot_x) + 0.015)
    ax4.set_ylim(min(plot_y) - 0.012, max(plot_y) + 0.012)

    from matplotlib.lines import Line2D
    family_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=family_color_map[k], markeredgecolor='black',
               markeredgewidth=0.45, markersize=8.5, label=family_legend_labels[k])
        for k in ["geometry_only", "enriched_interpretable", "topology_only", "geometry_plus_topology"]
        if k in pareto_plot["descriptor_family"].unique()
    ]
    model_handles = [
        Line2D([0], [0], marker=model_marker_map[k], color='0.25',
               linestyle='none', markerfacecolor='0.25', markersize=8.5, label=model_legend_labels[k])
        for k in ["ridge", "rf", "hgb", "mlp"]
        if k in pareto_plot["model_name"].unique()
    ]

    leg1 = ax4.legend(handles=family_handles, title="Descriptor family", frameon=True,
                      facecolor="white", edgecolor="0.85", framealpha=0.96,
                      fontsize=8.6, title_fontsize=9.0, loc="lower left", bbox_to_anchor=(0.02, 0.03))
    ax4.add_artist(leg1)
    ax4.legend(handles=model_handles, title="Model", frameon=True,
               facecolor="white", edgecolor="0.85", framealpha=0.96,
               fontsize=8.6, title_fontsize=9.0, loc="lower right", bbox_to_anchor=(0.98, 0.03))

    fig.suptitle(
        "Figure 7. When does a small-data MOF benchmark become scientifically reliable?",
        fontsize=18,
        y=0.985,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08)

    export_best = best_each_n[[
        "n_train", "method_label", "descriptor_family", "model_name",
        "rmse_mean", "rmse_ci_low", "rmse_ci_high", "r2_mean", "spearman_mean"
    ]].copy()
    if threshold_n is not None:
        export_best["benchmark_maturity_threshold_n"] = threshold_n

    export_rank = rs[[
        "n_train", "top1_consensus_probability", "mean_rank_spearman_vs_full"
    ]].copy()
    if threshold_n is not None:
        export_rank["benchmark_maturity_threshold_n"] = threshold_n

    export_screen = screen_df.copy()
    if threshold_n is not None and len(export_screen):
        export_screen["benchmark_maturity_threshold_n"] = threshold_n

    export_pareto = pareto.copy()
    export_pareto["plot_x"] = plot_x
    export_pareto["plot_y"] = plot_y
    save_figure_dataframes(
        {
            "panelA_best_curve": export_best,
            "panelB_ranking_stability": export_rank,
            "panelC_screening_stability": export_screen,
            "panelD_full_data_leaderboard": export_pareto,
        },
        "Figure7_main_conclusion_synthesis",
        paths["main_figure_data"],
        paths["all_figure_data"],
    )

    # Save separate panels for manual editing.
    fig.canvas.draw()
    save_axis_as_figure(ax1, paths["main_figures"] / "Figure7_main_conclusion_synthesis__PanelA")
    save_axis_as_figure(ax2, paths["main_figures"] / "Figure7_main_conclusion_synthesis__PanelB")
    save_axis_as_figure(ax3, paths["main_figures"] / "Figure7_main_conclusion_synthesis__PanelC")
    save_axis_as_figure(ax4, paths["main_figures"] / "Figure7_main_conclusion_synthesis__PanelD")
    save_figure(fig, paths["main_figures"] / "Figure7_main_conclusion_synthesis")



def make_si_target_learning_curves(
    agg_perf: pd.DataFrame,
    cfg: ProjectConfig,
    paths: Dict[str, Path],
) -> None:
    """
    SI figure: one learning-curve plot for each robustness target.
    """
    for target_col in cfg.all_targets:
        sub = agg_perf[
            (agg_perf["suite_name"] == "primary") &
            (agg_perf["target_col"] == target_col) &
            (agg_perf["test_seed"] == cfg.primary_test_seed)
        ].copy()

        methods_sorted = (
            sub[sub["n_train"] == sub["n_train"].max()]
            .sort_values("rmse_mean")["method_label"]
            .tolist()
        )

        style_map = build_method_style_map(methods_sorted)
        highlight = set(methods_sorted[:cfg.top_methods_to_highlight_si_curve])
        fig, ax = plt.subplots(figsize=(13.2, 8))
        for method in methods_sorted:
            s = sub[sub["method_label"] == method].sort_values("n_train")
            style = style_map[method]
            ax.plot(
                s["n_train"], s["rmse_mean"], marker=style["marker"], linestyle=style["linestyle"],
                color=style["color"], linewidth=2.3 if method in highlight else 1.2, alpha=1.0 if method in highlight else 0.40,
                label=pretty_method_label(method)
            )

        ax.set_xscale("log")
        ax.set_xlabel("Training size (log scale)")
        ax.set_ylabel("External-test RMSE")
        ax.set_title(f"SI learning curves: {target_col}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)

        learning_curve_df = sub[
            [
                "suite_name", "target_col", "test_seed", "n_train", "descriptor_family", "model_name", "method_label",
                "rmse_mean", "rmse_ci_low", "rmse_ci_high", "rmse_std",
                "mae_mean", "mae_ci_low", "mae_ci_high",
                "r2_mean", "r2_ci_low", "r2_ci_high",
                "spearman_mean", "spearman_ci_low", "spearman_ci_high",
            ]
        ].sort_values(["method_label", "n_train"]).reset_index(drop=True)
        figure_stem = f"SI_learning_curves__{slugify(target_col)}"
        save_figure_dataframes(
            {"learning_curve_values": learning_curve_df},
            figure_stem,
            paths["si_figure_data"],
            paths["all_figure_data"],
        )
        save_figure(fig, paths["si_figures"] / figure_stem)


def make_si_alt_test_robustness_figure(
    agg_perf: pd.DataFrame,
    cfg: ProjectConfig,
    paths: Dict[str, Path],
) -> None:
    """
    SI figure: compare best-achieved curves across alternative external test sets.
    """
    all_sub = agg_perf[
        (agg_perf["suite_name"].isin(["primary", "robustness_alt_test"])) &
        (agg_perf["target_col"] == cfg.main_target)
    ].copy()

    rows = []
    for (suite_name, test_seed, n_train), sub in all_sub.groupby(["suite_name", "test_seed", "n_train"]):
        best = sub.sort_values("rmse_mean").iloc[0]
        rows.append({
            "suite_name": suite_name,
            "test_seed": test_seed,
            "n_train": int(n_train),
            "best_rmse_mean": best["rmse_mean"],
            "best_method_label": best["method_label"],
        })
    best_df = pd.DataFrame(rows).sort_values(["test_seed", "n_train"])

    fig, ax = plt.subplots(figsize=(11.5, 7))
    seed_styles = build_method_style_map([str(s) for s in sorted(best_df["test_seed"].unique())])
    for test_seed, sub in best_df.groupby("test_seed"):
        sty = seed_styles[str(test_seed)]
        label = f"test_seed={test_seed}"
        ax.plot(sub["n_train"], sub["best_rmse_mean"], marker=sty["marker"], linestyle=sty["linestyle"], color=sty["color"], linewidth=2.4, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Training size (log scale)")
    ax.set_ylabel("Best method mean RMSE")
    ax.set_title("SI robustness to alternative external test sets")
    ax.grid(True, alpha=0.3)
    ax.legend()

    best_df_export = best_df.copy().sort_values(["test_seed", "n_train"]).reset_index(drop=True)
    save_figure_dataframes(
        {"best_curve_values": best_df_export},
        "SI_alt_test_robustness",
        paths["si_figure_data"],
        paths["all_figure_data"],
    )
    save_figure(fig, paths["si_figures"] / "SI_alt_test_robustness")


def make_all_figures(
    agg_perf: pd.DataFrame,
    ranking_stability: pd.DataFrame,
    screening: pd.DataFrame,
    stable_table: pd.DataFrame,
    pairwise_results: Dict[Tuple[str, int, int], pd.DataFrame],
    feature_df: pd.DataFrame,
    feature_conv: pd.DataFrame,
    additional_analysis: Dict[str, pd.DataFrame],
    cfg: ProjectConfig,
    paths: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """
    Produce all main-text and SI figures.
    """
    logger.info("Creating Figure 1 ...")
    make_workflow_figure(cfg, paths)

    logger.info("Creating Figure 2 ...")
    make_learning_curve_figure(agg_perf, cfg, paths)

    logger.info("Creating Figure 3 ...")
    make_ranking_stability_figure(ranking_stability, agg_perf, cfg, paths)

    logger.info("Creating Figure 4 ...")
    make_screening_reproducibility_figure(screening, agg_perf, cfg, paths)

    logger.info("Creating Figure 5 ...")
    make_pairwise_superiority_figure(pairwise_results, agg_perf, cfg, paths)

    logger.info("Creating Figure 6 ...")
    make_feature_effect_convergence_figure(feature_df, feature_conv, cfg, paths)

    logger.info("Creating Figure 7 ...")
    make_main_conclusion_figure(agg_perf, ranking_stability, screening, stable_table, additional_analysis, cfg, paths)

    logger.info("Creating SI figures ...")
    make_si_target_learning_curves(agg_perf, cfg, paths)
    make_si_alt_test_robustness_figure(agg_perf, cfg, paths)

    logger.info("Creating additional strengthening figures ...")
    make_additional_analysis_figures(agg_perf, ranking_stability, additional_analysis, cfg, paths)

    logger.info("All figures saved.")


# =============================================================================
# SECTION 17. SUMMARY EXPORTS / MANIFESTS
# =============================================================================

def save_summary_files(
    manifest: pd.DataFrame,
    df_metrics: pd.DataFrame,
    agg_perf: pd.DataFrame,
    stable_table: pd.DataFrame,
    cfg: ProjectConfig,
    paths: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    """
    Save compact summary JSON files and human-readable notes for later use.
    """
    summary = {
        "timestamp": current_timestamp(),
        "main_target": cfg.main_target,
        "n_total_jobs": int(len(manifest)),
        "n_completed_jobs": int(len(df_metrics)),
        "targets": list(cfg.all_targets),
        "primary_test_seed": cfg.primary_test_seed,
        "alternative_test_seeds": list(cfg.alternative_test_seeds),
        "output_root": str(paths["root"]),
    }

    # Best full-data methods by target
    best_methods = {}
    for target_col, sub in agg_perf[
        (agg_perf["suite_name"] == "primary") &
        (agg_perf["test_seed"] == cfg.primary_test_seed)
    ].groupby("target_col"):
        full_n = int(sub["n_train"].max())
        best = sub[sub["n_train"] == full_n].sort_values("rmse_mean").iloc[0]
        best_methods[target_col] = {
            "n_train_full": full_n,
            "best_method_label": best["method_label"],
            "rmse_mean": float(best["rmse_mean"]),
            "mae_mean": float(best["mae_mean"]),
            "r2_mean": float(best["r2_mean"]),
            "spearman_mean": float(best["spearman_mean"]),
        }
    summary["best_full_data_methods"] = best_methods

    # Benchmark maturity threshold summary for main target.
    main_stable = stable_table[stable_table["target_col"] == cfg.main_target].copy()
    if len(main_stable):
        threshold_n = main_stable["benchmark_maturity_threshold_n"].dropna()
        summary["main_target_maturity_threshold_n"] = int(threshold_n.iloc[0]) if len(threshold_n) else None

    save_json(summary, paths["final_exports"] / "project_summary.json")

    notes = []
    notes.append("Small-Data MOF Benchmark Design - Project Summary")
    notes.append("=" * 60)
    notes.append(f"Generated: {summary['timestamp']}")
    notes.append(f"Main target: {cfg.main_target}")
    notes.append(f"Completed jobs: {summary['n_completed_jobs']:,} / {summary['n_total_jobs']:,}")
    notes.append("")
    notes.append("Best full-data methods by target:")
    for target_col, info in best_methods.items():
        notes.append(
            f"  - {target_col}: {info['best_method_label']} "
            f"(RMSE={info['rmse_mean']:.4f}, R2={info['r2_mean']:.4f}, Spearman={info['spearman_mean']:.4f})"
        )
    write_text(paths["final_exports"] / "project_summary.txt", "\n".join(notes))
    logger.info("Summary manifests written.")


# =============================================================================
# SECTION 18. MAIN DRIVER
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small-Data MOF Benchmark Design pipeline"
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "run", "post"],
        help="all = run jobs + post-processing; run = jobs only; post = tables/figures only from saved results"
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    set_plot_style()
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    paths = build_output_paths(script_dir / CONFIG.output_root)
    logger = setup_logging(paths["logs"])

    logger.info("=" * 80)
    logger.info("Starting Small-Data MOF Benchmark Design pipeline")
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Output root: {paths['root']}")
    logger.info(f"Stage: {args.stage}")
    logger.info("=" * 80)

    # -------------------------------
    # Load and process input data
    # -------------------------------
    df = load_input_data(script_dir, CONFIG, paths, logger)
    logger.info(f"Processed master table shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    # -------------------------------
    # Job manifest
    # -------------------------------
    manifest = build_job_manifest(df, CONFIG, logger, paths)

    # -------------------------------
    # Stage 1: run missing jobs
    # -------------------------------
    if args.stage in ["all", "run"]:
        run_all_jobs(manifest, df, CONFIG, paths, logger)

    # -------------------------------
    # Stage 2: post-processing
    # -------------------------------
    if args.stage in ["all", "post"]:
        logger.info("Loading saved job metrics for post-processing ...")
        df_metrics = load_all_job_metrics(manifest, logger)
        agg_perf = aggregate_performance(df_metrics)
        ranking_stability = compute_ranking_stability(df_metrics)
        screening = compute_screening_reproducibility(df_metrics, CONFIG)

        # Resolve selected pairwise sizes including full.
        main_sub = agg_perf[
            (agg_perf["suite_name"] == "primary") &
            (agg_perf["target_col"] == CONFIG.main_target) &
            (agg_perf["test_seed"] == CONFIG.primary_test_seed)
        ]
        full_n = int(main_sub["n_train"].max())
        selected_pairwise_sizes = sorted(set([n for n in CONFIG.pairwise_selected_sizes if n in main_sub["n_train"].unique()] + [full_n]))
        pairwise_results = compute_pairwise_probability_superiority(df_metrics, selected_pairwise_sizes)

        stable_table = compute_stable_conclusion_table(agg_perf, ranking_stability, screening, CONFIG)
        additional_analysis = compute_additional_analyses(agg_perf, ranking_stability, screening, CONFIG)

        # Figure 6 / interpretation stability.
        feature_df = run_feature_effect_jobs(df, agg_perf, CONFIG, paths, logger)
        feature_conv = compute_feature_rank_convergence(feature_df)

        # Save tables and figures.
        save_all_tables(
            df, manifest, df_metrics, agg_perf, ranking_stability, screening,
            stable_table, pairwise_results, feature_df, feature_conv, additional_analysis,
            CONFIG, paths, logger
        )

        make_all_figures(
            agg_perf, ranking_stability, screening, stable_table,
            pairwise_results, feature_df, feature_conv, additional_analysis,
            CONFIG, paths, logger
        )

        save_summary_files(manifest, df_metrics, agg_perf, stable_table, CONFIG, paths, logger)

    logger.info("Pipeline finished successfully.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
