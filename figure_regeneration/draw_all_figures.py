
from __future__ import annotations

from pathlib import Path
import math
import re
import textwrap
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
MAIN_DATA = ROOT / "source_data" / "figure_data_main"
SI_DATA = ROOT / "source_data" / "figure_data_si"
OUT_MAIN = ROOT / "redrawn_figures" / "figures_main"
OUT_SI = ROOT / "redrawn_figures" / "figures_si"
for p in (OUT_MAIN, OUT_SI):
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Global visual design
# -----------------------------------------------------------------------------
DPI = 250
BLUE = "#0B3B75"
TEAL = "#087D8F"
CYAN = "#1687C9"
PURPLE = "#6D4BB1"
ORANGE = "#D68500"
GREEN = "#4E8B3D"
GREY = "#6D7783"
LIGHT_BG = "#F8FBFF"
GRID = "#D9E2EC"
TEXT = "#0A2540"

FAMILY_COLORS = {
    "geometry_plus_topology": "#007C91",      # deep teal
    "enriched_interpretable": "#7E57C2",     # purple
    "geometry_only": "#D98A00",              # amber
    "topology_only": "#6E6E6E",              # neutral grey
}
FAMILY_LABELS = {
    "geometry_plus_topology": "Geometry + topology",
    "enriched_interpretable": "Enriched geometry",
    "geometry_only": "Geometry only",
    "topology_only": "Topology only",
}
MODEL_MARKERS = {"rf": "o", "hgb": "s", "mlp": "D", "ridge": "^"}
MODEL_LABELS = {"rf": "RF", "hgb": "HGB", "mlp": "MLP", "ridge": "Ridge"}
MODEL_LINES = {"rf": "-", "hgb": "--", "mlp": "-.", "ridge": ":"}

METHOD_ABBR = {
    "geometry_plus_topology | rf": "G+T | RF",
    "geometry_plus_topology | hgb": "G+T | HGB",
    "geometry_plus_topology | mlp": "G+T | MLP",
    "geometry_plus_topology | ridge": "G+T | Ridge",
    "enriched_interpretable | rf": "Enriched | RF",
    "enriched_interpretable | hgb": "Enriched | HGB",
    "enriched_interpretable | mlp": "Enriched | MLP",
    "enriched_interpretable | ridge": "Enriched | Ridge",
    "geometry_only | rf": "Geometry | RF",
    "geometry_only | hgb": "Geometry | HGB",
    "geometry_only | mlp": "Geometry | MLP",
    "geometry_only | ridge": "Geometry | Ridge",
    "topology_only | rf": "Topology | RF",
    "topology_only | hgb": "Topology | HGB",
    "topology_only | mlp": "Topology | MLP",
    "topology_only | ridge": "Topology | Ridge",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13.0,
        "axes.labelsize": 11.0,
        "xtick.labelsize": 9.6,
        "ytick.labelsize": 9.6,
        "legend.fontsize": 8.9,
        "figure.titlesize": 18,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#20364D",
        "axes.linewidth": 0.95,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.55,
        "grid.linewidth": 0.7,
        "lines.linewidth": 2.3,
        "lines.markersize": 5.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.constrained_layout.use": False,
    })


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(str(path) + ".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(Path(str(path) + ".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def family_from_method(method: str) -> str:
    return str(method).split(" | ")[0]


def model_from_method(method: str) -> str:
    return str(method).split(" | ")[1] if " | " in str(method) else ""


def pretty_method(method: str) -> str:
    return METHOD_ABBR.get(str(method), str(method).replace("geometry_plus_topology", "G+T").replace("enriched_interpretable", "Enriched").replace("geometry_only", "Geometry").replace("topology_only", "Topology").replace(" | rf", " | RF").replace(" | hgb", " | HGB").replace(" | mlp", " | MLP").replace(" | ridge", " | Ridge"))


def pretty_target(target: str) -> str:
    s = str(target).replace("uptake(mmol/g) ", "")
    s = s.replace("CO2", r"CO$_2$").replace("CH4", r"CH$_4$")
    return s


def nice_n(v: int | float | str, full: int | None = None) -> str:
    try:
        vv = int(v)
    except Exception:
        return str(v)
    if full is not None and vv == int(full):
        return "full"
    if vv >= 1000 and vv < 100000:
        return f"{vv//1000}k"
    return str(vv)


def set_log_ticks(ax, xs: Iterable[int], full: int | None = None) -> None:
    xs = sorted([int(x) for x in xs])
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([nice_n(x, full=full) for x in xs])
    ax.tick_params(axis="x", rotation=0)


def panel_label(ax, label: str, x=-0.085, y=1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, va="top", ha="left",
            fontsize=15, fontweight="bold", color=TEXT)


def add_subtle_card(ax, pad: float = 0.012) -> None:
    # Soft panel outline in axes coordinates, behind all content.
    rect = FancyBboxPatch((-pad, -pad), 1 + 2 * pad, 1 + 2 * pad,
                          boxstyle="round,pad=0.014,rounding_size=0.018",
                          fc="white", ec="#D8E3F0", lw=0.8,
                          transform=ax.transAxes, zorder=-10, clip_on=False)
    ax.add_patch(rect)


def legend_outside_right(ax, ncol=1, title=None, fontsize=8.3, **kwargs):
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.012, 0.5), ncol=ncol,
                    frameon=True, title=title, fontsize=fontsize, borderaxespad=0.5, **kwargs)
    leg.get_frame().set_edgecolor("#CFD8E3")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.96)
    return leg


def add_maturity_band(ax, xmin=5000, xmax=10000, text=True, color="#7DAE68", alpha=0.12):
    ax.axvspan(xmin, xmax, color=color, alpha=alpha, zorder=0)
    if text:
        ymin, ymax = ax.get_ylim()
        ax.text(math.sqrt(xmin * xmax), ymax - 0.06*(ymax-ymin), "practical\nmaturity",
                ha="center", va="top", fontsize=9.2, color="#3F6F30",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BFD7B3", alpha=0.94))


# -----------------------------------------------------------------------------
# Figure 1
# -----------------------------------------------------------------------------
def draw_workflow_figure():
    df = read_csv(MAIN_DATA / "Figure1_benchmark_maturity_workflow__workflow_steps.csv")
    fig = plt.figure(figsize=(16.2, 4.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Soft background
    ax.add_patch(Rectangle((0, 0), 1, 1, fc="#FBFDFF", ec="none", zorder=-100))

    steps = [
        ("1", "Load clean dataset", "tabular MOF data", "▱▱"),
        ("2", "Choose target +\nfixed external test set", "constant holdout difficulty", "◎"),
        ("3", "Build nested\ntraining subsets", "n = 500 → full", "▣"),
        ("4", "Train descriptor–\nmodel pipelines", "repeated seeds", "⚙"),
        ("5", "Quantify maturity", "performance, uncertainty,\nstability", "▤"),
    ]
    xs = np.linspace(0.10, 0.90, 5)
    y = 0.61
    w = 0.15
    h = 0.52
    colors = ["#1C5EAD", "#168B92", "#1687C9", "#354DA3", "#0B3B75"]
    for i, ((num, title, subtitle, icon), x, c) in enumerate(zip(steps, xs, colors)):
        # main card with shadow
        shadow = FancyBboxPatch((x-w/2+0.006, y-h/2-0.006), w, h,
                                boxstyle="round,pad=0.018,rounding_size=0.02",
                                fc="#DCE7F5", ec="none", alpha=0.38, zorder=0)
        ax.add_patch(shadow)
        card = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.018,rounding_size=0.02",
                              fc="white", ec=c, lw=1.35, zorder=1)
        ax.add_patch(card)
        # number bubble
        ax.add_patch(Circle((x, y+h/2+0.010), 0.027, fc=c, ec="white", lw=1.4, zorder=3))
        ax.text(x, y+h/2+0.010, num, ha="center", va="center", color="white",
                fontsize=16, fontweight="bold", zorder=4)
        # Icon in a small rounded square
        icon_box = FancyBboxPatch((x-0.042, y+0.075), 0.084, 0.095,
                                  boxstyle="round,pad=0.016,rounding_size=0.016",
                                  fc="#F3F8FF", ec="#DCEBFA", lw=0.9, zorder=2)
        ax.add_patch(icon_box)
        ax.text(x, y+0.121, icon, ha="center", va="center", fontsize=26, color=c, fontweight="bold", zorder=4)
        ax.text(x, y-0.015, title, ha="center", va="center", fontsize=12.5,
                color="#062554", fontweight="bold", linespacing=1.05)
        ax.plot([x-0.014, x+0.014], [y-0.105, y-0.105], color=c, lw=1.0, alpha=0.85)
        ax.text(x, y-0.195, subtitle, ha="center", va="center", fontsize=9.8,
                color="#3A4655", linespacing=1.14)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x+w/2+0.018, y), (xs[i+1]-w/2-0.018, y),
                                         arrowstyle="-|>", mutation_scale=20, lw=2.4,
                                         color="#062554", alpha=0.95))

    # Output band
    band = FancyBboxPatch((0.035, 0.09), 0.93, 0.16,
                          boxstyle="round,pad=0.018,rounding_size=0.022",
                          fc="#F4F9FF", ec="#B8CEE8", lw=1.0)
    ax.add_patch(band)
    ax.text(0.075, 0.17, "Outputs", ha="center", va="center", fontsize=13.5, color="white",
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.45", fc=BLUE, ec=BLUE))
    outputs = [
        ("Learning curves", "⌁"),
        ("Ranking stability", "▥"),
        ("Screening\nreproducibility", "✓"),
        ("Pairwise\nsuperiority", "⚖"),
        ("Feature-effect\nconvergence", "◎"),
    ]
    oxs = np.linspace(0.23, 0.88, len(outputs))
    for j, (lab, sym) in enumerate(outputs):
        ax.text(oxs[j]-0.035, 0.17, sym, ha="center", va="center", fontsize=26, color=colors[j%len(colors)])
        ax.text(oxs[j]+0.005, 0.17, lab, ha="left", va="center", fontsize=11.2,
                color="#103B73", fontweight="bold", linespacing=1.05)
        if j > 0:
            ax.plot([oxs[j]-0.10, oxs[j]-0.10], [0.115, 0.225], color="#AAC4E2", lw=1.0)
    savefig(fig, OUT_MAIN / "Figure1_benchmark_maturity_workflow")


# -----------------------------------------------------------------------------
# Figure 2
# -----------------------------------------------------------------------------
def draw_learning_curves(df: pd.DataFrame, outpath: Path, title: str, focused: bool = True):
    full_n = int(df["n_train"].max())
    methods_order = (df[df["n_train"] == full_n]
                     .sort_values("rmse_mean")["method_label"].tolist())
    fig_w = 13.4 if focused else 13.8
    fig, ax = plt.subplots(figsize=(fig_w, 7.4))
    add_subtle_card(ax)
    for rank, method in enumerate(methods_order):
        sub = df[df["method_label"] == method].sort_values("n_train")
        fam = family_from_method(method)
        mod = model_from_method(method)
        color = FAMILY_COLORS.get(fam, GREY)
        marker = MODEL_MARKERS.get(mod, "o")
        ls = MODEL_LINES.get(mod, "-")
        lw = 2.9 if rank < 4 else 1.75
        alpha = 0.98 if rank < 8 else 0.58
        z = 5 if rank < 4 else 2
        x = sub["n_train"].to_numpy()
        y = sub["rmse_mean"].to_numpy()
        ax.plot(x, y, marker=marker, linestyle=ls, color=color, lw=lw, alpha=alpha,
                label=pretty_method(method), zorder=z, markeredgecolor="white", markeredgewidth=0.65)
        if rank < 6:
            ax.fill_between(x, sub["rmse_ci_low"].to_numpy(), sub["rmse_ci_high"].to_numpy(),
                            color=color, alpha=0.10, linewidth=0, zorder=1)
    set_log_ticks(ax, df["n_train"].unique(), full=full_n)
    ax.set_xlabel("Training sample size")
    ax.set_ylabel("External-test RMSE")
    ax.set_title(title, loc="left", color=TEXT, fontweight="bold", pad=12)
    ax.margins(x=0.03, y=0.08)
    add_maturity_band(ax)
    # Move legend completely outside the data region.
    leg = legend_outside_right(ax, ncol=1, title="Pipeline", fontsize=7.7)
    if leg.get_title(): leg.get_title().set_fontweight("bold")
    # Caption-like guide below axis
    ax.text(0.01, -0.18, "Line colour = descriptor family; marker/line pattern = model class. Confidence bands shown for leading pipelines.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.1, color="#4B5B6B")
    fig.subplots_adjust(left=0.08, right=0.78, top=0.90, bottom=0.18)
    savefig(fig, outpath)


def draw_figure2():
    df = read_csv(MAIN_DATA / "Figure2_learning_curves_main_target__learning_curve_values.csv")
    draw_learning_curves(df, OUT_MAIN / "Figure2_learning_curves_main_target",
                         "Figure 2. Learning curves for CO$_2$ uptake at 0.15 bar")


# -----------------------------------------------------------------------------
# Figure 3
# -----------------------------------------------------------------------------
def draw_figure3():
    rank = read_csv(MAIN_DATA / "Figure3_ranking_stability_phase_map__panelA_rank1_probabilities.csv")
    summ = read_csv(MAIN_DATA / "Figure3_ranking_stability_phase_map__panelB_stability_summary.csv")
    full_n = int(summ["n_train"].max())
    # order by first nonzero p then method; better visual: methods with any rank1 first, then others by family/model
    pivot = rank.pivot_table(index="method_label", columns="n_train", values="p_rank1", fill_value=0)
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    order = pivot.max(axis=1).sort_values(ascending=False).index.tolist()
    # keep non-winning methods grouped underneath
    pivot = pivot.loc[order]
    pretty_labels = [pretty_method(m) for m in pivot.index]

    fig = plt.figure(figsize=(12.6, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.2], hspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0]); add_subtle_card(ax1)
    im = ax1.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=1)
    ax1.set_yticks(np.arange(len(pretty_labels))); ax1.set_yticklabels(pretty_labels, fontsize=8.2)
    ax1.set_xticks(np.arange(len(pivot.columns))); ax1.set_xticklabels([nice_n(x, full_n) for x in pivot.columns])
    ax1.set_title("a  Probability that each pipeline ranks first", loc="left", fontweight="bold", color=TEXT)
    ax1.set_xlabel("Training sample size")
    ax1.set_ylabel("Pipeline")
    # readable cell values only for nonzero
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if val > 0.0:
                ax1.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.6,
                         color="white" if val > 0.45 else TEXT, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.022, pad=0.015)
    cbar.set_label("Rank-1 probability")

    ax2 = fig.add_subplot(gs[1, 0]); add_subtle_card(ax2)
    summ = summ.sort_values("n_train")
    ax2.plot(summ["n_train"], summ["top1_consensus_probability"], color=TEAL, marker="o", lw=2.5, label="Top-1 consensus")
    ax2.plot(summ["n_train"], summ["mean_rank_spearman_vs_full"], color=PURPLE, marker="s", lw=2.5, label="Rank Spearman vs full")
    ax2.axhline(0.80, color=TEAL, ls=":", lw=1.2, alpha=0.75)
    ax2.axhline(0.90, color=PURPLE, ls=":", lw=1.2, alpha=0.75)
    add_maturity_band(ax2, text=False)
    set_log_ticks(ax2, summ["n_train"].unique(), full=full_n)
    ax2.set_ylim(0.52, 1.035)
    ax2.set_title("b  Ranking preservation against the full-data ordering", loc="left", fontweight="bold", color=TEXT)
    ax2.set_xlabel("Training sample size")
    ax2.set_ylabel("Stability metric")
    ax2.legend(loc="lower right", frameon=True)
    fig.suptitle("Figure 3. Method-ranking stability phase map", y=0.985, fontweight="bold", color=TEXT)
    fig.subplots_adjust(left=0.18, right=0.93, top=0.92, bottom=0.08)
    savefig(fig, OUT_MAIN / "Figure3_ranking_stability_phase_map")


# -----------------------------------------------------------------------------
# Figure 4
# -----------------------------------------------------------------------------
def draw_figure4():
    df = read_csv(MAIN_DATA / "Figure4_screening_reproducibility__screening_reproducibility_values.csv").sort_values("n_train")
    full_n = int(df["n_train"].max())
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), sharex=False)
    for ax in axes: add_subtle_card(ax)
    x = df["n_train"].to_numpy()
    axes[0].plot(x, df["topk_overlap_mean"], marker="o", color=TEAL, lw=2.7, label="Mean top-k overlap")
    axes[0].fill_between(x, df["topk_overlap_lower"], df["topk_overlap_upper"], color=TEAL, alpha=0.16)
    axes[0].set_ylabel("Top-k overlap fraction")
    axes[0].set_title("a  Elite-candidate overlap", loc="left", fontweight="bold", color=TEXT)
    axes[1].plot(x, df["elite_enrichment_mean"], marker="s", color=ORANGE, lw=2.7, label="Mean elite enrichment")
    axes[1].fill_between(x, df["elite_enrichment_lower"], df["elite_enrichment_upper"], color=ORANGE, alpha=0.16)
    axes[1].set_ylabel("Elite enrichment factor")
    axes[1].set_title("b  Enrichment over random selection", loc="left", fontweight="bold", color=TEXT)
    for ax in axes:
        add_maturity_band(ax)
        set_log_ticks(ax, x, full=full_n)
        ax.set_xlabel("Training sample size")
        ax.margins(x=0.04, y=0.12)
        ax.legend(loc="lower right", frameon=True)
    fig.suptitle("Figure 4. Screening reproducibility versus sample size", y=1.02, fontweight="bold", color=TEXT)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.16, wspace=0.25)
    savefig(fig, OUT_MAIN / "Figure4_screening_reproducibility")


# -----------------------------------------------------------------------------
# Figure 5
# -----------------------------------------------------------------------------
def draw_figure5():
    df = read_csv(MAIN_DATA / "Figure5_pairwise_probability_superiority__pairwise_superiority_long.csv")
    sizes = sorted(df["n_train"].unique())
    # ordering from final size: use row means probability as proxy, descending wins
    final = df[df["n_train"] == max(sizes)]
    order = final.groupby("row_method")["probability_row_beats_column"].mean().sort_values(ascending=False).index.tolist()
    labels = [pretty_method(m) for m in order]
    fig = plt.figure(figsize=(5.2*len(sizes)+0.8, 6.8))
    gs = fig.add_gridspec(1, len(sizes)+1, width_ratios=[1]*len(sizes)+[0.055], wspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(sizes))]
    cax = fig.add_subplot(gs[0, -1])
    ims = []
    for ax, n in zip(axes, sizes):
        sub = df[df["n_train"] == n].pivot(index="row_method", columns="column_method", values="probability_row_beats_column")
        sub = sub.reindex(index=order, columns=order)
        im = ax.imshow(sub.to_numpy(), cmap="RdBu_r", vmin=0, vmax=1, aspect="equal", interpolation="nearest")
        ims.append(im)
        ax.set_title(f"n = {nice_n(n, max(sizes))}", fontweight="bold", color=TEXT, pad=8)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7.3)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=7.3)
        # diagonal line for orientation
        ax.plot([-0.5, len(order)-0.5], [-0.5, len(order)-0.5], color="white", lw=1.2, alpha=0.72)
        ax.tick_params(length=0)
    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.set_label("P(row beats column)")
    fig.suptitle("Figure 5. Pairwise probability of superiority", y=0.995, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.02, "Rows and columns are ordered by final-size superiority. Values > 0.5 indicate that the row pipeline more often has lower RMSE.",
             ha="center", fontsize=9.0, color="#4B5B6B")
    fig.subplots_adjust(left=0.09, right=0.965, top=0.87, bottom=0.24)
    savefig(fig, OUT_MAIN / "Figure5_pairwise_probability_superiority")


# -----------------------------------------------------------------------------
# Figure 6
# -----------------------------------------------------------------------------
def draw_figure6():
    conv = read_csv(MAIN_DATA / "Figure6_feature_effect_convergence__panelA_panelB_convergence_summary.csv").sort_values("n_train")
    rank_long = read_csv(MAIN_DATA / "Figure6_feature_effect_convergence__panelC_rank_matrix_long.csv")
    full_n = int(conv["n_train"].max())
    features = (rank_long[rank_long["n_train"] == full_n]
                .sort_values("mean_importance_rank")["feature"].tolist())
    sizes = sorted(rank_long["n_train"].unique())
    mat = rank_long.pivot(index="feature", columns="n_train", values="mean_importance_rank").reindex(index=features, columns=sizes)

    fig = plt.figure(figsize=(15.2, 8.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.34], height_ratios=[1, 1], wspace=0.47, hspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0]); add_subtle_card(ax1)
    ax2 = fig.add_subplot(gs[1, 0]); add_subtle_card(ax2)
    ax3 = fig.add_subplot(gs[:, 1]); add_subtle_card(ax3)

    x = conv["n_train"].to_numpy()
    ax1.plot(x, conv["rank_spearman_vs_full_mean"], marker="o", color=TEAL, lw=2.8)
    ax1.fill_between(x, conv["rank_spearman_lower"], conv["rank_spearman_upper"], color=TEAL, alpha=0.16)
    ax1.set_title("a  Feature-rank convergence", loc="left", fontweight="bold", color=TEXT)
    ax1.set_ylabel("Spearman vs full")
    ax1.set_ylim(0.34, 1.04)
    ax1.axhline(0.80, color=TEAL, ls=":", lw=1.2, alpha=0.75)
    set_log_ticks(ax1, x, full=full_n)
    ax1.set_xlabel("Training sample size")
    add_maturity_band(ax1, xmin=5000, xmax=10000, text=False)

    ax2.plot(x, conv["top10_jaccard_vs_full_mean"], marker="s", color=PURPLE, lw=2.8)
    ax2.fill_between(x, conv["top10_jaccard_lower"], conv["top10_jaccard_upper"], color=PURPLE, alpha=0.16)
    ax2.set_title("b  Top-feature set stability", loc="left", fontweight="bold", color=TEXT)
    ax2.set_ylabel("Top-10 Jaccard")
    ax2.set_ylim(0.26, 1.04)
    ax2.axhline(0.67, color=PURPLE, ls=":", lw=1.2, alpha=0.75)
    set_log_ticks(ax2, x, full=full_n)
    ax2.set_xlabel("Training sample size")
    add_maturity_band(ax2, xmin=5000, xmax=10000, text=False)

    im = ax3.imshow(mat.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis_r",
                    vmin=1, vmax=max(18, np.nanmax(mat.to_numpy())))
    ax3.set_title("c  Rank trajectories of leading descriptors", loc="left", fontweight="bold", color=TEXT)
    ax3.set_yticks(np.arange(len(features)))
    ax3.set_yticklabels([f.replace("_", " ") for f in features], fontsize=9.1)
    ax3.set_xticks(np.arange(len(sizes)))
    ax3.set_xticklabels([nice_n(n, full_n) for n in sizes])
    ax3.set_xlabel("Training sample size")
    ax3.set_ylabel("Descriptor", labelpad=10)
    cbar = fig.colorbar(im, ax=ax3, fraction=0.034, pad=0.025)
    cbar.set_label("Mean importance rank\n(lower = more important)")
    # Annotate key full-data descriptors unobtrusively
    ax3.text(1.03, 0.01, "Ranks are computed from\nsource-data permutation\nimportance summaries.", transform=ax3.transAxes,
             ha="left", va="bottom", fontsize=8.6, color="#4B5B6B")
    fig.suptitle("Figure 6. Feature-effect interpretation matures with the benchmark", y=0.992, fontweight="bold", color=TEXT)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.91, bottom=0.10)
    savefig(fig, OUT_MAIN / "Figure6_feature_effect_convergence")


# -----------------------------------------------------------------------------
# Figure 7
# -----------------------------------------------------------------------------
def draw_figure7():
    a = read_csv(MAIN_DATA / "Figure7_main_conclusion_synthesis__panelA_best_curve.csv").sort_values("n_train")
    b = read_csv(MAIN_DATA / "Figure7_main_conclusion_synthesis__panelB_ranking_stability.csv").sort_values("n_train")
    c = read_csv(MAIN_DATA / "Figure7_main_conclusion_synthesis__panelC_screening_stability.csv").sort_values("n_train")
    d = read_csv(MAIN_DATA / "Figure7_main_conclusion_synthesis__panelD_full_data_leaderboard.csv")
    full_n = int(a["n_train"].max())

    fig = plt.figure(figsize=(16.4, 10.2))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0]); add_subtle_card(ax1)
    ax2 = fig.add_subplot(gs[0, 1]); add_subtle_card(ax2)
    ax3 = fig.add_subplot(gs[1, 0]); add_subtle_card(ax3)
    ax4 = fig.add_subplot(gs[1, 1]); add_subtle_card(ax4)

    # Panel A
    x = a["n_train"].to_numpy()
    ax1.plot(x, a["rmse_mean"], color=TEAL, marker="o", lw=2.8, markeredgecolor="white", markeredgewidth=0.7)
    ax1.fill_between(x, a["rmse_ci_low"], a["rmse_ci_high"], color=TEAL, alpha=0.14)
    add_maturity_band(ax1, xmin=5000, xmax=10000)
    set_log_ticks(ax1, x, full=full_n)
    ax1.set_ylabel("Best RMSE")
    ax1.set_xlabel("Training sample size")
    ax1.set_title("a  Accuracy frontier", loc="left", fontweight="bold", color=TEXT)
    ax1.annotate("steady error reduction\ncontinues beyond stable regime",                 xy=(40000, float(a.loc[a["n_train"]==40000, "rmse_mean"].iloc[0])),
                 xytext=(13500, 0.462), textcoords="data",
                 arrowprops=dict(arrowstyle="->", lw=1.25, color="#1E3A5F", connectionstyle="arc3,rad=-0.14"),
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#C8D6E5", alpha=0.97),
                 fontsize=9.2, color="#1E3A5F")
    ax1.margins(x=0.04, y=0.10)

    # Panel B
    ax2.plot(b["n_train"], b["top1_consensus_probability"], color=TEAL, marker="o", lw=2.6, label="Top-1 consensus")
    ax2.plot(b["n_train"], b["mean_rank_spearman_vs_full"], color=PURPLE, marker="s", lw=2.6, label="Rank Spearman vs full")
    add_maturity_band(ax2, xmin=5000, xmax=10000)
    ax2.axhline(0.80, color=TEAL, lw=1.0, ls=":", alpha=0.8)
    ax2.axhline(0.90, color=PURPLE, lw=1.0, ls=":", alpha=0.8)
    set_log_ticks(ax2, b["n_train"].unique(), full=full_n)
    ax2.set_ylim(0.55, 1.035)
    ax2.set_ylabel("Stability metric")
    ax2.set_xlabel("Training sample size")
    ax2.set_title("b  Ranking reliability", loc="left", fontweight="bold", color=TEXT)
    ax2.legend(loc="lower right", frameon=True)

    # Panel C
    ax3.plot(c["n_train"], c["topk_overlap_mean"], color=ORANGE, marker="o", lw=2.6, label="Top-k overlap")
    ax3.fill_between(c["n_train"], c["topk_overlap_mean"]-c["topk_overlap_std"], c["topk_overlap_mean"]+c["topk_overlap_std"], color=ORANGE, alpha=0.14)
    ax3b = ax3.twinx()
    ax3b.plot(c["n_train"], c["elite_enrichment_mean"], color=PURPLE, marker="s", lw=2.6, label="Elite enrichment")
    ax3b.fill_between(c["n_train"], c["elite_enrichment_mean"]-c["elite_enrichment_std"], c["elite_enrichment_mean"]+c["elite_enrichment_std"], color=PURPLE, alpha=0.12)
    add_maturity_band(ax3, xmin=5000, xmax=10000)
    set_log_ticks(ax3, c["n_train"].unique(), full=full_n)
    ax3.set_xlim(350, full_n*1.15)  # avoids the leftmost purple point/error bar being clipped
    ax3.set_ylabel("Top-k overlap fraction", color=ORANGE)
    ax3b.set_ylabel("Elite enrichment", color=PURPLE)
    ax3.tick_params(axis="y", colors=ORANGE); ax3b.tick_params(axis="y", colors=PURPLE)
    # adjust y limits to show all error bars clearly and remove empty visual space
    ax3.set_ylim(max(0.38, (c["topk_overlap_mean"]-c["topk_overlap_std"]).min()-0.02), (c["topk_overlap_mean"]+c["topk_overlap_std"]).max()+0.035)
    ax3b.set_ylim((c["elite_enrichment_mean"]-c["elite_enrichment_std"]).min()-0.45, (c["elite_enrichment_mean"]+c["elite_enrichment_std"]).max()+0.55)
    ax3.set_xlabel("Training sample size")
    ax3.set_title("c  Screening stability", loc="left", fontweight="bold", color=TEXT)
    lines1, labs1 = ax3.get_legend_handles_labels(); lines2, labs2 = ax3b.get_legend_handles_labels()
    leg = ax3.legend(lines1+lines2, labs1+labs2, loc="lower right", frameon=True)
    leg.get_frame().set_alpha(0.96)

    # Panel D
    # Use source plot_x/plot_y if provided for deterministic offsets; otherwise raw metrics.
    d = d.copy()
    if "plot_x" not in d.columns: d["plot_x"] = d["rmse_mean"]
    if "plot_y" not in d.columns: d["plot_y"] = d["spearman_mean"]
    for _, r in d.iterrows():
        fam = r["descriptor_family"]; mod = r["model_name"]
        size = 128 if r["aggregate_rank_score"] <= 3 else 88
        ax4.scatter(r["plot_x"], r["plot_y"], s=size, color=FAMILY_COLORS.get(fam, GREY),
                    marker=MODEL_MARKERS.get(mod, "o"), edgecolor="white", linewidth=0.8, alpha=0.96, zorder=4)
    best = d.sort_values(["rmse_mean", "spearman_mean"], ascending=[True, False]).iloc[0]
    ax4.scatter(best["plot_x"], best["plot_y"], s=540, facecolors="none", edgecolors="#9EE6EF", linewidths=5, alpha=0.50, zorder=3)
    ax4.scatter(best["plot_x"], best["plot_y"], s=250, facecolors="none", edgecolors=TEAL, linewidths=2.2, alpha=0.9, zorder=5)
    ax4.annotate("Best full-data balance\nGeometry + topology\nRMSE = 0.443, ρ = 0.911",
                 xy=(best["plot_x"], best["plot_y"]), xytext=(0.575, 0.902), textcoords="data",
                 arrowprops=dict(arrowstyle="->", lw=1.3, color="#1E3A5F", connectionstyle="arc3,rad=0.18"),
                 bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#AFC5DD", alpha=0.97),
                 fontsize=9.4, color="#0B2D5C")
    ax4.text(0.704, 0.678, "topology-only\nplateau", fontsize=9.0, color="#555555")
    ax4.set_xlabel("Full-data RMSE (lower is better)")
    ax4.set_ylabel("Full-data Spearman ρ (higher is better)")
    ax4.set_title("d  Full-data trade-off", loc="left", fontweight="bold", color=TEXT)
    ax4.set_xlim(d["plot_x"].min()-0.018, d["plot_x"].max()+0.018)
    ax4.set_ylim(d["plot_y"].min()-0.035, d["plot_y"].max()+0.026)
    # Separate legend from data: place under panel D, not over points.

    
    # Separate legends from data and stack them below panel D.
    # This avoids overlap between the descriptor-family and model legends.
    fam_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=FAMILY_COLORS[f],
            markeredgecolor="white",
            markersize=9,
            label=FAMILY_LABELS[f],
        )
        for f in FAMILY_LABELS
    ]
    
    model_handles = [
        Line2D(
            [0], [0],
            marker=MODEL_MARKERS[m],
            color="#27384A",
            linestyle="None",
            markersize=8,
            label=MODEL_LABELS[m],
        )
        for m in MODEL_LABELS
    ]
    
    leg1 = ax4.legend(
        handles=fam_handles,
        title="Descriptor family",
        loc="upper center",
        bbox_to_anchor=(0.50, -0.19),
        ncol=2,
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        columnspacing=1.2,
        handletextpad=0.55,
        borderaxespad=0.0,
    )
    
    ax4.add_artist(leg1)
    
    leg2 = ax4.legend(
        handles=model_handles,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.50, -0.43),
        ncol=4,
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        columnspacing=1.1,
        handletextpad=0.55,
        borderaxespad=0.0,
    )
    
    leg1.get_frame().set_alpha(0.96)
    leg2.get_frame().set_alpha(0.96)

    fig.suptitle("Figure 7. Multidimensional evidence for benchmark maturity", y=0.992, fontweight="bold", color=TEXT)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.92, bottom=0.23)
    savefig(fig, OUT_MAIN / "Figure7_main_conclusion_synthesis")


# -----------------------------------------------------------------------------
# SI figures
# -----------------------------------------------------------------------------
def draw_si_alt_test():
    p = SI_DATA / "SI_alt_test_robustness__best_curve_values.csv"
    if not p.exists(): return
    df = read_csv(p).sort_values(["test_seed", "n_train"])
    full_n = int(df["n_train"].max())
    fig, ax = plt.subplots(figsize=(9.8, 5.6)); add_subtle_card(ax)
    colors = [TEAL, PURPLE, ORANGE, BLUE, GREEN]
    for i, (seed, sub) in enumerate(df.groupby("test_seed")):
        ax.plot(sub["n_train"], sub["best_rmse_mean"], marker="o", color=colors[i%len(colors)], lw=2.3,
                label=f"test seed {seed}", alpha=0.96)
    add_maturity_band(ax)
    set_log_ticks(ax, df["n_train"].unique(), full=full_n)
    ax.set_xlabel("Training sample size")
    ax.set_ylabel("Best-achieved RMSE")
    ax.set_title("SI Figure 1. Robustness across fixed external test sets", loc="left", fontweight="bold", color=TEXT)
    ax.legend(loc="upper right", frameon=True)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.14)
    savefig(fig, OUT_SI / "SI_alt_test_robustness")


def draw_si_learning_curves():
    for p in sorted(SI_DATA.glob("SI_learning_curves__*__learning_curve_values.csv")):
        df = read_csv(p)
        # infer target from data
        target = df["target_col"].iloc[0] if "target_col" in df.columns and len(df) else p.stem
        stem = p.name.replace("__learning_curve_values.csv", "")
        draw_learning_curves(df, OUT_SI / stem, f"SI. Learning curves for {pretty_target(target)}", focused=False)


def draw_si_descriptor_family():
    p = SI_DATA / "Additional_descriptor_family_learning_curves__descriptor_family_learning_curve.csv"
    if not p.exists(): return
    df = read_csv(p)
    # main target only if target_col exists
    if "target_col" in df.columns:
        target = df["target_col"].iloc[0]
    full_n = int(df["n_train"].max())
    fig, ax = plt.subplots(figsize=(10.2, 6.0)); add_subtle_card(ax)
    val_col = "rmse_family_mean" if "rmse_family_mean" in df.columns else "rmse_mean"
    std_col = "rmse_family_std_across_models" if "rmse_family_std_across_models" in df.columns else None
    for fam, sub in df.groupby("descriptor_family"):
        sub = sub.sort_values("n_train")
        color = FAMILY_COLORS.get(fam, GREY)
        ax.plot(sub["n_train"], sub[val_col], marker="o", color=color, lw=2.6, label=FAMILY_LABELS.get(fam, fam))
        if std_col and std_col in sub.columns:
            ax.fill_between(sub["n_train"], sub[val_col]-sub[std_col], sub[val_col]+sub[std_col], color=color, alpha=0.12)
    add_maturity_band(ax)
    set_log_ticks(ax, df["n_train"].unique(), full=full_n)
    ax.set_xlabel("Training sample size")
    ax.set_ylabel("Family-level RMSE")
    ax.set_title("SI. Descriptor-family learning curves", loc="left", fontweight="bold", color=TEXT)
    ax.legend(loc="upper right", frameon=True)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.14)
    savefig(fig, OUT_SI / "Additional_descriptor_family_learning_curves")


def draw_si_sample_efficiency():
    p = SI_DATA / "Additional_sample_efficiency_heatmap__sample_efficiency_values.csv"
    if not p.exists(): return
    df = read_csv(p)
    col = "n_to_90pct_gain"
    if col not in df.columns: return
    df = df.sort_values("full_rmse" if "full_rmse" in df.columns else col)
    vals = df[col].replace({np.nan: np.nan}).astype(float).to_numpy().reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(7.0, max(5.4, 0.34*len(df)+1.2))); add_subtle_card(ax)
    im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap="YlGnBu")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels([pretty_method(m) for m in df["method_label"]], fontsize=8.2)
    ax.set_xticks([0]); ax.set_xticklabels(["n to reach 90%\nof attainable gain"])
    for i, v in enumerate(vals[:,0]):
        ax.text(0, i, nice_n(v, int(np.nanmax(vals))), ha="center", va="center", fontsize=8.0,
                color="white" if np.isfinite(v) and v > np.nanmedian(vals) else TEXT, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.035)
    cbar.set_label("Training size")
    ax.set_title("SI. Sample-efficiency heatmap", loc="left", fontweight="bold", color=TEXT)
    fig.subplots_adjust(left=0.32, right=0.90, top=0.90, bottom=0.08)
    savefig(fig, OUT_SI / "Additional_sample_efficiency_heatmap")


def draw_si_target_difficulty():
    p = SI_DATA / "Additional_target_difficulty_map__target_difficulty_values.csv"
    if not p.exists(): return
    df = read_csv(p)
    xcol = "best_full_rmse" if "best_full_rmse" in df.columns else df.columns[0]
    ycol = "best_full_spearman" if "best_full_spearman" in df.columns else df.columns[1]
    fig, ax = plt.subplots(figsize=(8.6, 5.6)); add_subtle_card(ax)
    colors = [TEAL, PURPLE, ORANGE, BLUE]
    for i, (_, r) in enumerate(df.iterrows()):
        ax.scatter(r[xcol], r[ycol], s=130, color=colors[i%len(colors)], edgecolor="white", linewidth=0.9, zorder=3)
        lab = str(r.get("target_col", r.get("target", "target"))).replace("uptake(mmol/g) ", "")
        ax.annotate(lab, (r[xcol], r[ycol]), xytext=(7, 7), textcoords="offset points", fontsize=8.8,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#D0DAE6", alpha=0.92))
    ax.set_xlabel("Best full-data RMSE")
    ax.set_ylabel("Best full-data Spearman")
    ax.set_title("SI. Relative difficulty of adsorption targets", loc="left", fontweight="bold", color=TEXT)
    ax.margins(x=0.15, y=0.12)
    fig.subplots_adjust(left=0.11, right=0.96, top=0.88, bottom=0.14)
    savefig(fig, OUT_SI / "Additional_target_difficulty_map")


# -----------------------------------------------------------------------------
# Validation and manifest
# -----------------------------------------------------------------------------
def make_validation_report():
    rows = []
    for p in list(MAIN_DATA.glob("*.csv")) + list(SI_DATA.glob("*.csv")):
        try:
            df = pd.read_csv(p)
            rows.append({
                "source_file": str(p.relative_to(ROOT)),
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": int(df.isna().sum().sum()),
                "numeric_columns": int(sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)),
                "status": "read_ok",
            })
        except Exception as e:
            rows.append({"source_file": str(p.relative_to(ROOT)), "rows": np.nan, "columns": np.nan,
                         "missing_values": np.nan, "numeric_columns": np.nan, "status": f"error: {e}"})
    out = ROOT / "redrawn_figures" / "validation_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def make_file_manifest():
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_file():
            rows.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size})
    pd.DataFrame(rows).to_csv(ROOT / "file_manifest.csv", index=False)


def main():
    set_style()
    draw_workflow_figure()
    draw_figure2()
    draw_figure3()
    draw_figure4()
    draw_figure5()
    draw_figure6()
    draw_figure7()
    draw_si_alt_test()
    draw_si_learning_curves()
    draw_si_descriptor_family()
    draw_si_sample_efficiency()
    draw_si_target_difficulty()
    make_validation_report()
    make_file_manifest()
    print("Done. Figures written to", ROOT / "redrawn_figures")

if __name__ == "__main__":
    main()
