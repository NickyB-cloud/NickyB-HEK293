#!/usr/bin/env python3
"""
HEK293_mix_vs_purified_GitHub.py
Plotting/statistics script for comparing ONE lipid A preparation (the
unfractionated "mix") against its individual Rf-band fractions in the HEK293
TLR reporter assay, human vs mouse cells, at several timepoints.

Input: raw-absorbance tidy table from parse_HEK293_mix_vs_purified_GitHub.py (columns: Time,
Species, Plate, Wavelength, Strain, Rf, Dose, Replicate, Absorbance).
compute_fold() converts absorbance to fold change over the geometric mean of
the "Neg" (unstimulated) wells of the same Species x Time (x Plate).

Usage:
    python HEK293_mix_vs_purified_GitHub.py HEK293_mix_vs_purified_tidy.csv

Generates a timestamped PDF with:
  1) All strains per species × timepoint (log2 Fold-Change vs Rf)
     → ANOVA + Tukey across Rf positions (mix vs individual bands)
  2) Per-strain Human vs Mouse comparison (log2 Fold-Change vs Rf)
     → ANOVA + Tukey Human vs Mouse (within strain × timepoint × Rf)

Y-axis is log2; tick labels show the original linear fold-change values.
"""

import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
from datetime import datetime

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")

# ── Constants ────────────────────────────────────────────────────────────────
# EDIT FOR YOUR DATA: strain IDs (must match the Strain column of your tidy
# table), positive-control label, dose labels, colors and axis settings.
# XPOS_RF / X_TICK_LABELS below hard-code this preparation's Rf bands.

PGIN_STRAINS    = ["381", "1587", "1773"]
ECOLI_STRAINS   = ["EC"]
ALL_STRAINS     = PGIN_STRAINS + ECOLI_STRAINS
POS_CTRL_STRAIN = "Pos_cont."
POS_CTRL_COLOR  = "#9467bd"

DOSE_ORDER   = ["1ug", "10ug"]
DOSE_MARKERS = {"1ug": "o", "10ug": "s"}

STRAIN_COLORS = {
    "381":  "#1f77b4",
    "1587": "#ff7f0e",
    "1773": "#2ca02c",
    "EC":   "#d62728",
}

SPECIES_COLORS = {"human": "red", "mouse": "blue"}
SPECIES_LABELS = {"human": "Human", "mouse": "Mouse"}

SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]

# ── Y-axis (log2) settings ───────────────────────────────────────────────────
# Tick positions are in log2 space; labels show the original linear values.
LOG2_YTICKS_LINEAR = [1, 2, 4, 8, 16, 32, 64]   # linear fold-change values
LOG2_YTICKS        = [np.log2(v) for v in LOG2_YTICKS_LINEAR]
LOG2_YLIM          = (np.log2(0.8), np.log2(64))  # ceiling matches bar hard-cap


def setup_log2_yaxis(ax):
    ax.set_yscale("linear")          # we store log2 values manually
    ax.set_ylim(*LOG2_YLIM)
    ax.set_yticks(LOG2_YTICKS)
    ax.set_yticklabels([str(v) for v in LOG2_YTICKS_LINEAR])
    ax.set_ylabel("Fold-Change from Unstimulated Control (log₂ scale)")
    # Light horizontal gridlines at each tick for readability
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def to_log2(values):
    """Convert an array of fold-change values to log2, dropping non-positive."""
    x = np.array(values, dtype=float)
    x = x[x > 0]
    return np.log2(x)


# ── X-axis layout ─────────────────────────────────────────────────────────────
#   0 = Pos ctrl
#   1 = 1587 Lipid A  ("mix")
#   --- x-break at 2 ---
#   3 = Rf 0.68
#   4 = Rf 0.80
#   5 = Rf 0.86

XPOS_POS_CTRL    = 0
XPOS_LIPID_A_MIX = 1
XPOS_RF          = {0.68: 3, 0.80: 4, 0.86: 5}
RF_BAND_KEYS_STR = {"68": 0.68, "80": 0.80, "86": 0.86}

X_TICK_POSITIONS = [0, 1, 3, 4, 5]
X_TICK_LABELS    = ["Pos ctrl", "1587\nLipid A", "0.68", "0.80", "0.86"]

BREAK_X     = 2.0
BREAK_WIDTH = 0.35
BREAK_YREL  = -0.10


# ── X-axis helpers ────────────────────────────────────────────────────────────

def get_rf_xpos(rf_str):
    if rf_str == "mix":
        return XPOS_LIPID_A_MIX
    mapped = RF_BAND_KEYS_STR.get(rf_str)
    if mapped is not None:
        return XPOS_RF[mapped]
    return None


def draw_x_break(ax, x=BREAK_X, yrel=BREAK_YREL, w=BREAK_WIDTH, d=0.015):
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for sign in (-1, 1):
        xs = [x + sign * w * 0.6 - d, x + sign * w * 0.6 + d]
        ys = [yrel - 0.03, yrel + 0.03]
        ax.plot(xs, ys, transform=trans, color="black", lw=1.5, clip_on=False)
    ax.axvspan(x - w, x + w,
               ymin=yrel - 0.015, ymax=yrel + 0.015,
               transform=trans, color="white", zorder=10, clip_on=False)


def draw_rf_bracket(ax, x_left=XPOS_RF[0.68], x_right=XPOS_RF[0.86],
                    yrel=-0.22, label="Rf band"):
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    cx  = (x_left + x_right) / 2.0
    tip = yrel - 0.05
    arm = yrel
    lx = [x_left, x_left, cx, cx];  ly = [arm, arm - 0.01, arm - 0.01, tip]
    rx = [cx, cx, x_right, x_right]; ry = [tip, arm - 0.01, arm - 0.01, arm]
    for xs, ys in [(lx, ly), (rx, ry)]:
        ax.plot(xs, ys, transform=trans, color="black", lw=1.2, clip_on=False)
    ax.text(cx, tip - 0.03, label, transform=trans,
            ha="center", va="top", fontsize=12, clip_on=False)


def setup_xaxis(ax):
    ax.set_xticks(X_TICK_POSITIONS)
    ax.set_xticklabels(X_TICK_LABELS, fontsize=12)
    ax.set_xlim(-0.6, 5.6)
    draw_x_break(ax)
    draw_rf_bracket(ax)


# ── Statistics helpers ────────────────────────────────────────────────────────

def pvalue_to_stars(p):
    for threshold, stars in SIG_LEVELS:
        if p < threshold:
            return stars
    return ""


def run_anova_tukey(groups: dict):
    valid = {k: np.array(v, dtype=float) for k, v in groups.items() if len(v) >= 2}
    if len(valid) < 2:
        return None, None
    _, anova_p = stats.f_oneway(*valid.values())
    all_vals   = np.concatenate(list(valid.values()))
    all_labels = np.concatenate([[k] * len(v) for k, v in valid.items()])
    tukey      = pairwise_tukeyhsd(all_vals, all_labels, alpha=0.05)
    tukey_df   = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0],
    )
    tukey_df.columns = [c.replace(" ", "_").replace("-", "_") for c in tukey_df.columns]
    if "p_adj" not in tukey_df.columns:
        raise KeyError(f"p_adj not found: {tukey_df.columns}")
    tukey_df["stars"] = tukey_df["p_adj"].apply(pvalue_to_stars)
    return anova_p, tukey_df


def draw_significance_bars(ax, bar_specs, y_start=None, step=None):
    """
    Draw significance bars using a collision-aware shelf algorithm:
    each bar is placed on the lowest y-level where it does not
    horizontally overlap any previously placed bar.
    """
    sig_specs = [(min(x1, x2), max(x1, x2), s) for x1, x2, s in bar_specs if s]
    if not sig_specs:
        return

    # Sort widest-first so wide bars anchor low and narrow bars nest above them
    sig_specs.sort(key=lambda t: -(t[1] - t[0]))

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    if step is None:
        step = yrange * 0.055   # vertical gap between bar levels

    BAR_CEIL = np.log2(64)

    if y_start is None:
        data_max = max(
            (c.get_offsets()[:, 1].max()
             for c in ax.collections if len(c.get_offsets())),
            default=ymin + yrange * 0.5,
        )
        y_start = data_max + step * 0.4

    # Shelf: list of (x_left, x_right, y) for placed bars
    placed = []

    for x1, x2, stars in sig_specs:
        # Find the lowest shelf where this bar fits without x-overlap
        # Two bars overlap if their x-ranges intersect (with a small margin)
        margin = 0.15
        y = y_start
        while True:
            conflict = any(
                not (x2 + margin <= px1 or x1 - margin >= px2)
                for px1, px2, py in placed
                if abs(py - y) < step * 0.9
            )
            if not conflict:
                break
            y += step

        if y > BAR_CEIL:
            continue   # skip rather than collide with title

        placed.append((x1, x2, y))
        ax.plot([x1, x1, x2, x2], [y, y + step * 0.3, y + step * 0.3, y],
                lw=1.0, color="black")
        ax.text((x1 + x2) / 2, y + step * 0.35, stars,
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(ymin, min(ymax, BAR_CEIL))


# ── Data loading / fold-change ────────────────────────────────────────────────

def compute_min_n(df):
    counts = df.groupby(["Strain", "Species", "Dose", "Rf"])["FoldChange"].count()
    return int(counts.min()) if len(counts) else np.nan


def geomean(values):
    x = np.array(values, dtype=float); x = x[x > 0]
    return np.exp(np.log(x).mean()) if len(x) else np.nan


def geomean_and_gsem(values):
    x = np.array(values, dtype=float); x = x[x > 0]
    if len(x) == 0:
        return np.nan, np.nan
    logs = np.log(x)
    gm   = np.exp(logs.mean())
    if len(x) == 1:
        return gm, 0.0
    gsem_factor = np.exp(logs.std(ddof=1) / np.sqrt(len(x)))
    return gm, gm * (gsem_factor - 1.0)


def load_data(path):
    df = pd.read_csv(path)
    df["Dose"] = df["Dose"].fillna("1ug")
    df["Dose"] = pd.Categorical(df["Dose"], categories=DOSE_ORDER, ordered=True)
    df["Rf"]   = df["Rf"].astype(str).str.strip().str.lower()

    def normalise_rf(val):
        if val == "mix":
            return "mix"
        try:
            f = float(val)
            return str(int(round(f * 100))) if f < 1 else str(int(round(f)))
        except ValueError:
            return val

    df["Rf"]      = df["Rf"].apply(normalise_rf)
    df["Species"] = df["Species"].str.lower()
    return df


def compute_fold(df):
    group_keys = ["Species", "Time"]
    if "Plate" in df.columns and df["Plate"].notna().any():
        group_keys.insert(1, "Plate")
    neg = (
        df[df["Strain"] == "Neg"]
        .groupby(group_keys)["Absorbance"]
        .apply(geomean)
        .rename("NegGeoMean")
        .reset_index()
    )
    df = df.merge(neg, on=group_keys, how="left")
    df["FoldChange"] = df["Absorbance"] / df["NegGeoMean"]
    return df


# ── Shared scatter/errorbar helper ────────────────────────────────────────────

def scatter_with_mean(ax, xi, pts, color, marker, alpha_pts=0.35):
    """Plot individual points (log2) and geomean ± gSEM on a single axes."""
    if len(pts) == 0:
        return
    log2_pts = to_log2(pts)
    if len(log2_pts) == 0:
        return

    ax.scatter([xi] * len(log2_pts), log2_pts, color=color, marker=marker,
               alpha=alpha_pts, s=25, zorder=3)

    gm, yerr = geomean_and_gsem(pts)
    if np.isnan(gm):
        return
    log2_gm   = np.log2(gm)
    # Convert gSEM to log2 space (asymmetric, but gSEM is small so symmetric ok)
    log2_yerr = np.log2(gm + yerr) - log2_gm if yerr > 0 else 0.0

    ax.errorbar(xi, log2_gm, yerr=log2_yerr, fmt="none",
                ecolor=color, elinewidth=1.2, capsize=3, zorder=5)
    ax.scatter(xi, log2_gm, color=color, marker=marker,
               s=80, edgecolor="black", zorder=6)


def make_fig(title):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.subplots_adjust(bottom=0.25, top=0.82)
    ax.set_title(title, fontsize=12)
    return fig, ax


# ── Plot type 1 — All strains per species × time ─────────────────────────────

def plot_all_strains(df, pdf):
    for sp in sorted(df["Species"].unique()):
        for tm in sorted(df["Time"].unique()):
            sub = df[(df["Species"] == sp) & (df["Time"] == tm)]
            if sub.empty:
                continue

            title = (f"{SPECIES_LABELS.get(sp, sp)} TLR4/MD2 activity to\n"
                     f"$\\it{{1\\ µg}}$ of Lipid A at {tm}H")
            fig, ax = make_fig(title)

            rf_vals = [r for r in ["mix", "68", "80", "86"]
                       if r in sub["Rf"].unique()]

            # Pos ctrl
            pos_sub = sub[sub["Strain"] == POS_CTRL_STRAIN]
            if not pos_sub.empty:
                scatter_with_mean(ax, XPOS_POS_CTRL,
                                  pos_sub["FoldChange"].dropna().values,
                                  POS_CTRL_COLOR, "o")

            # Strain data
            strain_sub = sub[sub["Strain"].isin(ALL_STRAINS)]
            for strain in ALL_STRAINS:
                s = strain_sub[strain_sub["Strain"] == strain]
                if s.empty:
                    continue
                color = STRAIN_COLORS[strain]
                for dose in DOSE_ORDER:
                    sd = s[s["Dose"] == dose]
                    if sd.empty:
                        continue
                    marker = DOSE_MARKERS[dose]
                    for rf in rf_vals:
                        pts = sd[sd["Rf"] == rf]["FoldChange"].dropna().values
                        xi  = get_rf_xpos(rf)
                        if xi is None or len(pts) == 0:
                            continue
                        scatter_with_mean(ax, xi, pts, color, marker)

            # Statistics (run on log2 values)
            rf_groups = {}
            for rf in rf_vals:
                pts = strain_sub[strain_sub["Rf"] == rf]["FoldChange"].dropna().values
                log2_pts = to_log2(pts)
                if len(log2_pts) >= 2:
                    rf_groups[rf] = log2_pts

            anova_p, tukey_df = run_anova_tukey(rf_groups)
            if tukey_df is not None and anova_p is not None and anova_p < 0.05:
                bar_specs = []
                for _, row in tukey_df.iterrows():
                    if not row["stars"]:
                        continue
                    x1 = get_rf_xpos(row["group1"])
                    x2 = get_rf_xpos(row["group2"])
                    if x1 is None or x2 is None:
                        continue
                    bar_specs.append((min(x1, x2), max(x1, x2), row["stars"]))
                draw_significance_bars(ax, bar_specs)

            setup_log2_yaxis(ax)
            setup_xaxis(ax)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            min_n = compute_min_n(strain_sub)
            ax.text(0.02, 0.02, f"N = {min_n}", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ── Plot type 2 — Per-strain Human vs Mouse ──────────────────────────────────

def plot_species_comparison(df, pdf):
    for strain in ALL_STRAINS:
        for tm in sorted(df["Time"].unique()):
            strain_sub = df[(df["Strain"] == strain) & (df["Time"] == tm)]
            pos_sub    = df[(df["Strain"] == POS_CTRL_STRAIN) & (df["Time"] == tm)]
            if strain_sub.empty:
                continue

            title = (f"Strain {strain} — Human vs Mouse\n"
                     f"$\\it{{1\\ µg}}$ of Lipid A at {tm}H")
            fig, ax = make_fig(title)

            rf_vals = [r for r in ["mix", "68", "80", "86"]
                       if r in strain_sub["Rf"].unique()]

            # Pos ctrl — split by species with species color and offset
            for sp in ["human", "mouse"]:
                ps     = pos_sub[pos_sub["Species"] == sp]
                offset = -0.10 if sp == "mouse" else 0.10
                if not ps.empty:
                    scatter_with_mean(ax, XPOS_POS_CTRL + offset,
                                      ps["FoldChange"].dropna().values,
                                      SPECIES_COLORS[sp], "o")

            # Human / Mouse
            for sp in ["human", "mouse"]:
                s = strain_sub[strain_sub["Species"] == sp]
                if s.empty:
                    continue
                color  = SPECIES_COLORS[sp]
                offset = -0.10 if sp == "mouse" else 0.10
                for dose in DOSE_ORDER:
                    sd = s[s["Dose"] == dose]
                    if sd.empty:
                        continue
                    marker = DOSE_MARKERS[dose]
                    for rf in rf_vals:
                        pts = sd[sd["Rf"] == rf]["FoldChange"].dropna().values
                        xi  = get_rf_xpos(rf)
                        if xi is None or len(pts) == 0:
                            continue
                        scatter_with_mean(ax, xi + offset, pts, color, marker)

            # Statistics (run on log2 values)
            bar_specs = []
            for rf in rf_vals:
                groups = {}
                for sp in ["human", "mouse"]:
                    pts = strain_sub[
                        (strain_sub["Species"] == sp) & (strain_sub["Rf"] == rf)
                    ]["FoldChange"].dropna().values
                    log2_pts = to_log2(pts)
                    if len(log2_pts) >= 1:
                        groups[sp] = log2_pts
                anova_p, tukey_df = run_anova_tukey(groups)
                if tukey_df is None or anova_p is None or anova_p >= 0.05:
                    continue
                for _, row in tukey_df.iterrows():
                    if not row["stars"]:
                        continue
                    xi = get_rf_xpos(rf)
                    if xi is None:
                        continue
                    bar_specs.append((xi - 0.10, xi + 0.10, row["stars"]))
            draw_significance_bars(ax, bar_specs)

            setup_log2_yaxis(ax)
            setup_xaxis(ax)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            handles = [
                Line2D([0], [0], marker="o", color=SPECIES_COLORS[sp],
                       linestyle="None", markersize=8, label=SPECIES_LABELS[sp])
                for sp in ["human", "mouse"]
            ]
            ax.legend(handles=handles, fontsize=8, frameon=False)

            min_n = compute_min_n(strain_sub)
            ax.text(0.02, 0.02, f"N = {min_n}", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = load_data(sys.argv[1] if len(sys.argv) > 1 else "HEK293_mix_vs_purified_tidy.csv")
    df = compute_fold(df)
    df = df[df["Strain"].isin(ALL_STRAINS + [POS_CTRL_STRAIN])]

    outname = f"HEK293_mix_vs_purified_{timestamp}.pdf"
    with PdfPages(outname) as pdf:
        plot_all_strains(df, pdf)
        plot_species_comparison(df, pdf)

    print(f"Saved {outname}")


if __name__ == "__main__":
    main()