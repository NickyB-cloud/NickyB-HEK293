#!/usr/bin/env python3
"""
HEK293_exploratory_GitHub.py
HEK293 TLR reporter (SEAP/NF-kB) dose-response plots from a tidy table with
pre-computed fold change (columns: Species, Wavelength, Strain, Rf, Dose,
Replicate, FoldChange). Writes a timestamped multi-page PDF of exploratory
plot types; HEK293_figures_stacked_GitHub.py makes the composite figures.

Usage:
    python HEK293_exploratory_GitHub.py HEK_tidy_dose.csv

Details:
  - FoldChange already computed (no Absorbance/Neg normalisation step)
  - No Time column — single timepoint data
  - Dynamic Rf x-positions (all unique Rf values sorted)
  - Log2 y-axis with linear fold-change tick labels
  - X-axis break separating Pos_ctrl from Rf bands
  - Per-strain and per-dose x-offsets to prevent crowding
  - Plot type 3: x-axis grouped by strain, subaxis by Rf
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

PGIN_STRAINS    = ["381", "1587", "1773"]
ECOLI_STRAINS   = ["EC"]
ALL_STRAINS     = PGIN_STRAINS + ECOLI_STRAINS
POS_CTRL_STRAIN = "Pos_cont."
POS_CTRL_COLOR  = "#9467bd"

DOSE_ORDER   = ["1ug", "10ug"]
DOSE_MARKERS = {"1ug": "o", "10ug": "s"}
DOSE_LABELS  = {"1ug": "1 µg/mL", "10ug": "10 µg/mL"}

STRAIN_COLORS = {
    "381":  "#1f77b4",
    "1587": "#ff7f0e",
    "1773": "#2ca02c",
    "EC":   "#d62728",
}

# Horizontal offsets for plot type 1 — spreads 4 strains × 2 doses at crowded Rf positions
STRAIN_XOFFSETS = {"381": -0.22, "1587": -0.07, "1773": 0.07, "EC": 0.22}
DOSE_XOFFSETS   = {"1ug": -0.07, "10ug": 0.07}

SPECIES_COLORS   = {"human": "red", "mouse": "blue"}
SPECIES_LABELS   = {"human": "Human", "mouse": "Mouse"}
# Horizontal offsets for plot type 2 — spreads 2 species × 2 doses
SPECIES_XOFFSETS = {"human": 0.12, "mouse": -0.12}

SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]

# ── Y-axis (log2) ─────────────────────────────────────────────────────────────
# Extra headroom above log2(64) keeps significance bars from kissing the title.

LOG2_YTICKS_LINEAR = [1, 2, 4, 8, 16, 32, 64]
LOG2_YTICKS        = [np.log2(v) for v in LOG2_YTICKS_LINEAR]
LOG2_YLIM          = (np.log2(0.8), np.log2(64) + 0.5)
BAR_CEIL           = np.log2(64)          # sig bars never exceed this


def setup_log2_yaxis(ax):
    ax.set_yscale("linear")
    ax.set_ylim(*LOG2_YLIM)
    ax.set_yticks(LOG2_YTICKS)
    ax.set_yticklabels([str(v) for v in LOG2_YTICKS_LINEAR])
    ax.set_ylabel("Fold-Change from Unstimulated Control (log₂ scale)")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def to_log2(values):
    x = np.array(values, dtype=float)
    x = x[x > 0]
    return np.log2(x)


# ── X-axis layout ─────────────────────────────────────────────────────────────
# x = 0       : Pos_ctrl
# break ~x = 1
# x = 2, 3, … : Rf bands (plots 1 & 2) or strain-grouped Rf bands (plot 3)

XPOS_POS_CTRL = 0
BREAK_X       = 1.0
BREAK_WIDTH   = 0.35
BREAK_YREL    = -0.10


def build_rf_xpos(rf_vals_sorted):
    """Flat Rf positions starting at x=2 (plots 1 & 2)."""
    return {rf: i + 2 for i, rf in enumerate(rf_vals_sorted)}


def build_strain_rf_xpos(df, strains):
    """
    Grouped positions for plot 3: each strain gets a consecutive block,
    blocks separated by a 1.5-unit gap.

    Returns
    -------
    xpos           : {(strain, rf): x}
    strain_centers : {strain: center_x}  used for group labels
    """
    xpos           = {}
    strain_centers = {}
    x              = 2.0
    STRAIN_GAP     = 1.5

    for strain in strains:
        rf_vals = sorted(
            [r for r in df[df["Strain"] == strain]["Rf"].dropna().unique()],
            key=lambda r: float(r) if r != "mix" else -1.0,
        )
        if not rf_vals:
            continue
        start_x = x
        for rf in rf_vals:
            xpos[(strain, rf)] = x
            x += 1.0
        strain_centers[strain] = (start_x + x - 1.0) / 2.0
        x += STRAIN_GAP

    return xpos, strain_centers


def draw_x_break(ax, x=BREAK_X, yrel=BREAK_YREL, w=BREAK_WIDTH, d=0.015):
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for sign in (-1, 1):
        xs = [x + sign * w * 0.6 - d, x + sign * w * 0.6 + d]
        ys = [yrel - 0.03, yrel + 0.03]
        ax.plot(xs, ys, transform=trans, color="black", lw=1.5, clip_on=False)
    ax.axvspan(x - w, x + w,
               ymin=yrel - 0.015, ymax=yrel + 0.015,
               transform=trans, color="white", zorder=10, clip_on=False)


def rf_display_label(rf_str):
    try:
        return f"{int(rf_str) / 100:.2f}"
    except (ValueError, TypeError):
        return rf_str


def setup_xaxis(ax, rf_xpos):
    """Standard flat x-axis for plot types 1 & 2."""
    rf_sorted      = sorted(rf_xpos, key=rf_xpos.get)
    tick_positions = [XPOS_POS_CTRL] + [rf_xpos[r] for r in rf_sorted]
    tick_labels    = ["Pos ctrl"] + [rf_display_label(r) for r in rf_sorted]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlim(-0.6, max(rf_xpos.values()) + 0.6)
    draw_x_break(ax)
    ax.set_xlabel("Rf value")


def setup_xaxis_by_strain(ax, xpos, strain_centers):
    """
    X-axis for plot type 3.
    Rf values as tick labels; strain names drawn as bold group labels below.
    """
    sorted_items   = sorted(xpos.items(), key=lambda kv: kv[1])
    tick_positions = [XPOS_POS_CTRL] + [v for _, v in sorted_items]
    tick_labels    = ["Pos\nctrl"] + [rf_display_label(rf) for (_, rf), _ in sorted_items]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlim(-0.6, max(xpos.values()) + 0.6)
    draw_x_break(ax, yrel=-0.08)

    # Strain group labels centred below each block, in bold strain colour
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for strain, cx in strain_centers.items():
        ax.text(cx, -0.20, strain, transform=trans,
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=STRAIN_COLORS.get(strain, "black"), clip_on=False)


# ── Statistics ────────────────────────────────────────────────────────────────

def pvalue_to_stars(p):
    for threshold, stars in SIG_LEVELS:
        if p < threshold:
            return stars
    return ""


def run_anova_tukey(groups):
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
    sig_specs = [(min(x1, x2), max(x1, x2), s) for x1, x2, s in bar_specs if s]
    if not sig_specs:
        return
    sig_specs.sort(key=lambda t: -(t[1] - t[0]))
    ymin, ymax = ax.get_ylim()
    yrange     = ymax - ymin
    if step is None:
        step = yrange * 0.055
    if y_start is None:
        data_max = max(
            (c.get_offsets()[:, 1].max()
             for c in ax.collections if len(c.get_offsets())),
            default=ymin + yrange * 0.5,
        )
        y_start = data_max + step * 0.4
    placed = []
    for x1, x2, stars in sig_specs:
        margin = 0.15
        y      = y_start
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
            continue
        placed.append((x1, x2, y))
        ax.plot([x1, x1, x2, x2], [y, y + step * 0.3, y + step * 0.3, y],
                lw=1.0, color="black")
        ax.text((x1 + x2) / 2, y + step * 0.35, stars,
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(ymin, min(ymax, BAR_CEIL))


# ── Data helpers ──────────────────────────────────────────────────────────────

def compute_min_n(df):
    counts = df.groupby(["Strain", "Species", "Dose", "Rf"])["FoldChange"].count()
    return int(counts.min()) if len(counts) else np.nan


def geomean_and_gsem(values):
    x = np.array(values, dtype=float)
    x = x[x > 0]
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
    df["Rf"]   = df["Rf"].astype(str).str.strip()

    def normalise_rf(val):
        if pd.isna(val):
            return None
        val = str(val).strip()
        if val.lower() in ("nan", ""):
            return None
        if val.lower() == "mix":
            return "mix"
        try:
            f = float(val)
            return str(int(round(f * 100))) if f < 1 else str(int(round(f)))
        except ValueError:
            return val

    df["Rf"]      = df["Rf"].apply(normalise_rf)
    df["Species"] = df["Species"].str.lower()
    return df


# ── Scatter / errorbar helper ─────────────────────────────────────────────────

def scatter_with_mean(ax, xi, pts, color, marker, alpha_pts=0.35):
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
    log2_yerr = np.log2(gm + yerr) - log2_gm if yerr > 0 else 0.0
    ax.errorbar(xi, log2_gm, yerr=log2_yerr, fmt="none",
                ecolor=color, elinewidth=1.2, capsize=3, zorder=5)
    ax.scatter(xi, log2_gm, color=color, marker=marker,
               s=80, edgecolor="black", zorder=6)


def make_fig(title, n_xpos, bottom=0.20):
    width = max(7.5, n_xpos * 0.9 + 2.5)
    fig, ax = plt.subplots(figsize=(width, 6.0))
    fig.subplots_adjust(bottom=bottom, top=0.83)
    ax.set_title(title, fontsize=12, pad=14)
    return fig, ax


# ── Plot type 1 — All strains per species ─────────────────────────────────────

def plot_all_strains(df, pdf):
    strain_df = df[df["Strain"].isin(ALL_STRAINS)]
    rf_vals_sorted = sorted(
        [r for r in strain_df["Rf"].dropna().unique()],
        key=lambda r: float(r) if r != "mix" else -1.0,
    )
    rf_xpos = build_rf_xpos(rf_vals_sorted)

    for sp in sorted(df["Species"].unique()):
        sub = df[df["Species"] == sp]
        if sub.empty:
            continue

        fig, ax = make_fig(
            f"{SPECIES_LABELS.get(sp, sp)} TLR4/MD2 Fold-Change by Rf",
            len(rf_vals_sorted) + 1,
        )

        # Pos ctrl — centred, no dose/strain offset
        pos_sub = sub[sub["Strain"] == POS_CTRL_STRAIN]
        if not pos_sub.empty:
            scatter_with_mean(ax, XPOS_POS_CTRL,
                              pos_sub["FoldChange"].dropna().values,
                              POS_CTRL_COLOR, "o")

        # All strains with per-strain + per-dose offsets
        for strain in ALL_STRAINS:
            s = sub[sub["Strain"] == strain]
            if s.empty:
                continue
            color      = STRAIN_COLORS[strain]
            strain_off = STRAIN_XOFFSETS.get(strain, 0.0)
            for dose in DOSE_ORDER:
                sd = s[s["Dose"] == dose]
                if sd.empty:
                    continue
                marker   = DOSE_MARKERS[dose]
                dose_off = DOSE_XOFFSETS.get(dose, 0.0)
                for rf in rf_vals_sorted:
                    pts = sd[sd["Rf"] == rf]["FoldChange"].dropna().values
                    xi  = rf_xpos.get(rf)
                    if xi is None or len(pts) == 0:
                        continue
                    scatter_with_mean(ax, xi + strain_off + dose_off, pts, color, marker)

        # Stats: compare Rf positions pooled across strains
        strain_sub = sub[sub["Strain"].isin(ALL_STRAINS)]
        rf_groups  = {}
        for rf in rf_vals_sorted:
            pts      = strain_sub[strain_sub["Rf"] == rf]["FoldChange"].dropna().values
            log2_pts = to_log2(pts)
            if len(log2_pts) >= 2:
                rf_groups[rf] = log2_pts
        anova_p, tukey_df = run_anova_tukey(rf_groups)
        if tukey_df is not None and anova_p is not None and anova_p < 0.05:
            bar_specs = []
            for _, row in tukey_df.iterrows():
                if not row["stars"]:
                    continue
                x1 = rf_xpos.get(row["group1"])
                x2 = rf_xpos.get(row["group2"])
                if x1 is None or x2 is None:
                    continue
                bar_specs.append((min(x1, x2), max(x1, x2), row["stars"]))
            draw_significance_bars(ax, bar_specs)

        setup_log2_yaxis(ax)
        setup_xaxis(ax, rf_xpos)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        present_strains = [s for s in ALL_STRAINS if s in sub["Strain"].values]
        handles = (
            [Line2D([0], [0], marker=DOSE_MARKERS[d], color="gray",
                    linestyle="None", markersize=8, label=DOSE_LABELS[d])
             for d in DOSE_ORDER]
            + [Line2D([0], [0], marker="o", color=STRAIN_COLORS[s],
                      linestyle="None", markersize=8, label=s)
               for s in present_strains]
        )
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")

        min_n = compute_min_n(strain_sub)
        ax.text(0.02, 0.02, f"N = {min_n}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)


# ── Plot type 2 — Per-strain Human vs Mouse ───────────────────────────────────

def plot_species_comparison(df, pdf):
    pos_sub = df[df["Strain"] == POS_CTRL_STRAIN]

    for strain in ALL_STRAINS:
        strain_sub = df[df["Strain"] == strain]
        if strain_sub.empty:
            continue

        rf_vals_sorted = sorted(
            [r for r in strain_sub["Rf"].dropna().unique()],
            key=lambda r: float(r) if r != "mix" else -1.0,
        )
        rf_xpos = build_rf_xpos(rf_vals_sorted)

        fig, ax = make_fig(f"Strain {strain} — Human vs Mouse", len(rf_vals_sorted) + 1)

        # Pos ctrl — split by species with species color and offset
        for sp in ["human", "mouse"]:
            ps = pos_sub[pos_sub["Species"] == sp]
            if ps.empty:
                continue
            scatter_with_mean(ax, XPOS_POS_CTRL + SPECIES_XOFFSETS[sp],
                              ps["FoldChange"].dropna().values,
                              SPECIES_COLORS[sp], "o")

        # Human / Mouse with per-species + per-dose offsets
        for sp in ["human", "mouse"]:
            s = strain_sub[strain_sub["Species"] == sp]
            if s.empty:
                continue
            color   = SPECIES_COLORS[sp]
            sp_off  = SPECIES_XOFFSETS[sp]
            for dose in DOSE_ORDER:
                sd = s[s["Dose"] == dose]
                if sd.empty:
                    continue
                marker   = DOSE_MARKERS[dose]
                dose_off = DOSE_XOFFSETS.get(dose, 0.0)
                for rf in rf_vals_sorted:
                    pts = sd[sd["Rf"] == rf]["FoldChange"].dropna().values
                    xi  = rf_xpos.get(rf)
                    if xi is None or len(pts) == 0:
                        continue
                    scatter_with_mean(ax, xi + sp_off + dose_off, pts, color, marker)

        # Stats: human vs mouse per Rf
        bar_specs = []
        for rf in rf_vals_sorted:
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
                xi = rf_xpos.get(rf)
                if xi is None:
                    continue
                bar_specs.append((xi - 0.10, xi + 0.10, row["stars"]))
        draw_significance_bars(ax, bar_specs)

        setup_log2_yaxis(ax)
        setup_xaxis(ax, rf_xpos)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        present_species = [sp for sp in ["human", "mouse"]
                           if sp in strain_sub["Species"].values]
        handles = (
            [Line2D([0], [0], marker=DOSE_MARKERS[d], color="gray",
                    linestyle="None", markersize=8, label=DOSE_LABELS[d])
             for d in DOSE_ORDER]
            + [Line2D([0], [0], marker="o", color=SPECIES_COLORS[sp],
                      linestyle="None", markersize=8, label=SPECIES_LABELS[sp])
               for sp in present_species]
        )
        ax.legend(handles=handles, fontsize=8, frameon=False)

        min_n = compute_min_n(strain_sub)
        ax.text(0.02, 0.02, f"N = {min_n}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)


# ── Plot type 3 — Per-species, x-axis grouped by strain then Rf ───────────────

def plot_by_strain(df, pdf):
    """
    One plot per species.
    Layout: Pos_ctrl | break | [strain-A Rf bands] | gap | [strain-B Rf bands] | …
    Strain names appear as bold coloured group labels below the Rf tick labels.
    Statistics compare Rf bands within each strain.
    """
    pos_sub   = df[df["Strain"] == POS_CTRL_STRAIN]
    strain_df = df[df["Strain"].isin(ALL_STRAINS)]

    for sp in sorted(df["Species"].unique()):
        sub        = strain_df[strain_df["Species"] == sp]
        pos_sp_sub = pos_sub[pos_sub["Species"] == sp]
        if sub.empty:
            continue

        xpos, strain_centers = build_strain_rf_xpos(sub, ALL_STRAINS)
        if not xpos:
            continue

        max_xpos = max(xpos.values())
        fig, ax  = make_fig(
            f"{SPECIES_LABELS.get(sp, sp)} TLR4/MD2 — By Strain",
            max_xpos + 1,
            bottom=0.28,   # extra room for rotated Rf ticks + strain group labels
        )

        # Pos ctrl
        if not pos_sp_sub.empty:
            scatter_with_mean(ax, XPOS_POS_CTRL,
                              pos_sp_sub["FoldChange"].dropna().values,
                              POS_CTRL_COLOR, "o")

        # Per strain × Rf with dose offset
        for strain in ALL_STRAINS:
            s = sub[sub["Strain"] == strain]
            if s.empty:
                continue
            color = STRAIN_COLORS[strain]
            for dose in DOSE_ORDER:
                sd       = s[s["Dose"] == dose]
                if sd.empty:
                    continue
                marker   = DOSE_MARKERS[dose]
                dose_off = DOSE_XOFFSETS.get(dose, 0.0)
                for (st, rf), xi in xpos.items():
                    if st != strain:
                        continue
                    pts = sd[sd["Rf"] == rf]["FoldChange"].dropna().values
                    if len(pts) == 0:
                        continue
                    scatter_with_mean(ax, xi + dose_off, pts, color, marker)

        # Stats: compare Rf bands within each strain
        bar_specs = []
        for strain in ALL_STRAINS:
            s = sub[sub["Strain"] == strain]
            if s.empty:
                continue
            rf_keys = sorted(
                [rf for (st, rf) in xpos if st == strain],
                key=lambda r: float(r) if r != "mix" else -1.0,
            )
            if len(rf_keys) < 2:
                continue
            groups = {}
            for rf in rf_keys:
                pts      = s[s["Rf"] == rf]["FoldChange"].dropna().values
                log2_pts = to_log2(pts)
                if len(log2_pts) >= 2:
                    groups[rf] = log2_pts
            anova_p, tukey_df = run_anova_tukey(groups)
            if tukey_df is None or anova_p is None or anova_p >= 0.05:
                continue
            for _, row in tukey_df.iterrows():
                if not row["stars"]:
                    continue
                x1 = xpos.get((strain, row["group1"]))
                x2 = xpos.get((strain, row["group2"]))
                if x1 is None or x2 is None:
                    continue
                bar_specs.append((min(x1, x2), max(x1, x2), row["stars"]))
        draw_significance_bars(ax, bar_specs)

        setup_log2_yaxis(ax)
        setup_xaxis_by_strain(ax, xpos, strain_centers)
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        present_strains = [s for s in ALL_STRAINS if s in sub["Strain"].values]
        handles = (
            [Line2D([0], [0], marker=DOSE_MARKERS[d], color="gray",
                    linestyle="None", markersize=8, label=DOSE_LABELS[d])
             for d in DOSE_ORDER]
            + [Line2D([0], [0], marker="o", color=STRAIN_COLORS[s],
                      linestyle="None", markersize=8, label=s)
               for s in present_strains]
        )
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")

        min_n = compute_min_n(sub)
        ax.text(0.02, 0.02, f"N = {min_n}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = load_data(sys.argv[1] if len(sys.argv) > 1 else "HEK_tidy_dose.csv")
    df = df[df["Strain"].isin(ALL_STRAINS + [POS_CTRL_STRAIN])]

    outname = f"HEK293_exploratory_{timestamp}.pdf"
    with PdfPages(outname) as pdf:
        plot_all_strains(df, pdf)
        plot_species_comparison(df, pdf)
        plot_by_strain(df, pdf)

    print(f"Saved {outname}")


if __name__ == "__main__":
    main()
