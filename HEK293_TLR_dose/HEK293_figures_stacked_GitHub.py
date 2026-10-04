#!/usr/bin/env python3
"""
HEK293_figures_stacked_GitHub.py  —  composite HEK293 TLR dose-response figures
from a tidy fold-change table (columns: Species, Wavelength, Strain, Rf,
Dose, Replicate, FoldChange).

Usage:
    python HEK293_figures_stacked_GitHub.py HEK_tidy_dose.csv
Output: HEK293_figures_stacked.pdf  (two pages)

Points = replicates (log2 axis, linear tick labels); marker = geometric
mean ± geometric SEM. Stats: one-way ANOVA + Tukey HSD on log2 fold change
(across Rf bands within a strain; human vs mouse within strain × Rf).
Stars: * P<0.05, ** P<0.01, *** P<0.001 (Tukey-adjusted).

  Figure 1  All strains per species, stacked (Human above, Mouse below).
            One shared legend.

  Figure 2  Human vs Mouse per strain, 2×2 grid.
            One shared legend.
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

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")

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

# Horizontal offsets for strain panels — spreads 4 strains × 2 doses at crowded Rf positions
STRAIN_XOFFSETS = {"381": -0.22, "1587": -0.07, "1773": 0.07, "EC": 0.22}
DOSE_XOFFSETS   = {"1ug": -0.07, "10ug": 0.07}

SPECIES_COLORS   = {"human": "red", "mouse": "blue"}
SPECIES_LABELS   = {"human": "Human", "mouse": "Mouse"}
# Horizontal offsets for species panels — spreads 2 species × 2 doses
SPECIES_XOFFSETS = {"human": 0.12, "mouse": -0.12}

SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]

# ── Y-axis (log2) ─────────────────────────────────────────────────────────────
# Extra headroom above log2(64) keeps significance bars from kissing the title.

LOG2_YTICKS_LINEAR = [1, 2, 4, 8, 16, 32, 64]
LOG2_YTICKS        = [np.log2(v) for v in LOG2_YTICKS_LINEAR]
LOG2_YLIM          = (np.log2(0.8), np.log2(64) + 0.5)
BAR_CEIL           = np.log2(64)          # sig bars never exceed this

# ── X-axis layout ─────────────────────────────────────────────────────────────
# x = 0       : Pos_ctrl
# break ~x = 1
# x = 2, 3, … : Rf bands

XPOS_POS_CTRL = 0
BREAK_X       = 1.0
BREAK_WIDTH   = 0.35

# ── Figure layout ─────────────────────────────────────────────────────────────

PANEL_LABELS = "ABCDEFGH"

# ── Data utilities ────────────────────────────────────────────────────────────

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


def load_data(path):
    df = pd.read_csv(path)
    df["Dose"]    = df["Dose"].fillna("1ug")
    df["Dose"]    = pd.Categorical(df["Dose"], categories=DOSE_ORDER, ordered=True)
    df["Rf"]      = df["Rf"].astype(str).str.strip().apply(normalise_rf)
    df["Species"] = df["Species"].str.lower()
    return df


def build_rf_xpos(rf_vals_sorted):
    return {rf: i + 2 for i, rf in enumerate(rf_vals_sorted)}


def compute_min_n(df):
    counts = df.groupby(["Strain", "Species", "Dose", "Rf"])["FoldChange"].count()
    return int(counts.min()) if len(counts) else np.nan

# ── Math helpers ──────────────────────────────────────────────────────────────

def to_log2(values):
    x = np.array(values, dtype=float)
    x = x[x > 0]
    return np.log2(x)


def geomean_and_gsem(values):
    x = np.array(values, dtype=float)
    x = x[x > 0]
    if len(x) == 0:
        return np.nan, np.nan
    logs = np.log(x)
    gm   = np.exp(logs.mean())
    if len(x) == 1:
        return gm, 0.0
    return gm, gm * (np.exp(logs.std(ddof=1) / np.sqrt(len(x))) - 1.0)


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
    tukey_df["stars"] = tukey_df["p_adj"].apply(pvalue_to_stars)
    return anova_p, tukey_df

# ── Axes drawing primitives ───────────────────────────────────────────────────

def scatter_with_mean(ax, xi, pts, color, marker):
    if len(pts) == 0:
        return
    log2_pts = to_log2(pts)
    if len(log2_pts) == 0:
        return
    ax.scatter([xi] * len(log2_pts), log2_pts, color=color, marker=marker,
               alpha=0.35, s=20, zorder=3)
    gm, yerr = geomean_and_gsem(pts)
    if np.isnan(gm):
        return
    log2_gm   = np.log2(gm)
    log2_yerr = np.log2(gm + yerr) - log2_gm if yerr > 0 else 0.0
    ax.errorbar(xi, log2_gm, yerr=log2_yerr, fmt="none",
                ecolor=color, elinewidth=1.2, capsize=3, zorder=5)
    ax.scatter(xi, log2_gm, color=color, marker=marker,
               s=65, edgecolor="black", zorder=6)


def setup_log2_yaxis(ax):
    ax.set_yscale("linear")
    ax.set_ylim(*LOG2_YLIM)
    ax.set_yticks(LOG2_YTICKS)
    ax.set_yticklabels([str(v) for v in LOG2_YTICKS_LINEAR], fontsize=8)
    ax.set_ylabel("Fold-Change (log₂)", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def draw_x_break(ax, yrel=-0.10):
    x, w, d = BREAK_X, BREAK_WIDTH, 0.015
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for sign in (-1, 1):
        xs = [x + sign * w * 0.6 - d, x + sign * w * 0.6 + d]
        ys = [yrel - 0.03, yrel + 0.03]
        ax.plot(xs, ys, transform=trans, color="black", lw=1.5, clip_on=False)
    ax.axvspan(x - w, x + w, ymin=yrel - 0.015, ymax=yrel + 0.015,
               transform=trans, color="white", zorder=10, clip_on=False)


def rf_display_label(rf_str):
    try:
        return f"{int(rf_str) / 100:.2f}"
    except (ValueError, TypeError):
        return rf_str


def setup_xaxis(ax, rf_xpos):
    rf_sorted = sorted(rf_xpos, key=rf_xpos.get)
    ax.set_xticks([XPOS_POS_CTRL] + [rf_xpos[r] for r in rf_sorted])
    ax.set_xticklabels(["Pos ctrl"] + [rf_display_label(r) for r in rf_sorted], fontsize=8)
    ax.set_xlim(-0.6, max(rf_xpos.values()) + 0.6)
    draw_x_break(ax)
    ax.set_xlabel("Rf", fontsize=9)


def draw_significance_bars(ax, bar_specs):
    sig_specs = [(min(x1, x2), max(x1, x2), s) for x1, x2, s in bar_specs if s]
    if not sig_specs:
        return
    sig_specs.sort(key=lambda t: -(t[1] - t[0]))
    ymin, ymax = ax.get_ylim()
    step       = (ymax - ymin) * 0.055
    data_max   = max(
        (c.get_offsets()[:, 1].max() for c in ax.collections if len(c.get_offsets())),
        default=ymin + (ymax - ymin) * 0.5,
    )
    y_start = data_max + step * 0.4
    placed  = []
    for x1, x2, stars in sig_specs:
        y = y_start
        while any(
            not (x2 + 0.15 <= px1 or x1 - 0.15 >= px2)
            for px1, px2, py in placed if abs(py - y) < step * 0.9
        ):
            y += step
        if y > BAR_CEIL:
            continue
        placed.append((x1, x2, y))
        ax.plot([x1, x1, x2, x2], [y, y + step * 0.3, y + step * 0.3, y],
                lw=1.0, color="black")
        ax.text((x1 + x2) / 2, y + step * 0.35, stars,
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(ymin, min(ymax, BAR_CEIL))

# ── Panel drawing functions ───────────────────────────────────────────────────

def draw_all_strains_panel(ax, df, sp, rf_xpos, rf_vals_sorted):
    """All-strains panel for one species onto a pre-existing axes."""
    sub = df[df["Species"] == sp]

    pos_sub = sub[sub["Strain"] == POS_CTRL_STRAIN]
    if not pos_sub.empty:
        scatter_with_mean(ax, XPOS_POS_CTRL,
                          pos_sub["FoldChange"].dropna().values, POS_CTRL_COLOR, "o")

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
            dose_off = DOSE_XOFFSETS.get(dose, 0.0)
            for rf in rf_vals_sorted:
                pts = sd[sd["Rf"] == rf]["FoldChange"].dropna().values
                xi  = rf_xpos.get(rf)
                if xi is None or len(pts) == 0:
                    continue
                scatter_with_mean(ax, xi + strain_off + dose_off,
                                  pts, color, DOSE_MARKERS[dose])

    strain_sub = sub[sub["Strain"].isin(ALL_STRAINS)]
    rf_groups  = {
        rf: to_log2(strain_sub[strain_sub["Rf"] == rf]["FoldChange"].dropna().values)
        for rf in rf_vals_sorted
    }
    rf_groups = {k: v for k, v in rf_groups.items() if len(v) >= 2}
    anova_p, tukey_df = run_anova_tukey(rf_groups)
    if tukey_df is not None and anova_p is not None and anova_p < 0.05:
        bar_specs = [
            (rf_xpos[r["group1"]], rf_xpos[r["group2"]], r["stars"])
            for _, r in tukey_df.iterrows()
            if r["stars"] and r["group1"] in rf_xpos and r["group2"] in rf_xpos
        ]
        draw_significance_bars(ax, bar_specs)

    setup_log2_yaxis(ax)
    setup_xaxis(ax, rf_xpos)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    min_n = compute_min_n(strain_sub)
    ax.text(0.02, 0.04, f"N = {min_n}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8)


def draw_species_panel(ax, df, strain, rf_xpos, rf_vals_sorted, pos_sub):
    """Human-vs-mouse panel for one strain onto a pre-existing axes."""
    strain_sub = df[df["Strain"] == strain]

    # Pos ctrl — split by species with species color and offset
    for sp in ["human", "mouse"]:
        ps = pos_sub[pos_sub["Species"] == sp]
        if ps.empty:
            continue
        scatter_with_mean(ax, XPOS_POS_CTRL + SPECIES_XOFFSETS[sp],
                          ps["FoldChange"].dropna().values, SPECIES_COLORS[sp], "o")

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
            dose_off = DOSE_XOFFSETS.get(dose, 0.0)
            for rf in rf_vals_sorted:
                pts = sd[sd["Rf"] == rf]["FoldChange"].dropna().values
                xi  = rf_xpos.get(rf)
                if xi is None or len(pts) == 0:
                    continue
                scatter_with_mean(ax, xi + sp_off + dose_off,
                                  pts, color, DOSE_MARKERS[dose])

    bar_specs = []
    for rf in rf_vals_sorted:
        groups = {}
        for sp in ["human", "mouse"]:
            log2_pts = to_log2(strain_sub[
                (strain_sub["Species"] == sp) & (strain_sub["Rf"] == rf)
            ]["FoldChange"].dropna().values)
            if len(log2_pts) >= 1:
                groups[sp] = log2_pts
        anova_p, tukey_df = run_anova_tukey(groups)
        if tukey_df is None or anova_p is None or anova_p >= 0.05:
            continue
        xi = rf_xpos.get(rf)
        if xi is None:
            continue
        for _, row in tukey_df.iterrows():
            if row["stars"]:
                bar_specs.append((xi - 0.10, xi + 0.10, row["stars"]))
    draw_significance_bars(ax, bar_specs)

    setup_log2_yaxis(ax)
    setup_xaxis(ax, rf_xpos)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    min_n = compute_min_n(strain_sub)
    ax.text(0.02, 0.04, f"N = {min_n}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8)

# ── Figure 1 — All strains, stacked ──────────────────────────────────────────

def make_figure1(df, pdf):
    strain_df      = df[df["Strain"].isin(ALL_STRAINS)]
    species_list   = sorted(strain_df["Species"].dropna().unique())
    rf_vals_sorted = sorted(
        [r for r in strain_df["Rf"].dropna().unique()],
        key=lambda r: float(r) if r != "mix" else -1.0,
    )
    rf_xpos = build_rf_xpos(rf_vals_sorted)
    n       = len(species_list)

    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n + 0.8), squeeze=False)
    fig.suptitle(
        "Figure 1.  HEK293 TLR4/MD2 Fold-Change Response Across Rf Fractions",
        fontsize=13, fontweight="bold",
    )

    subtitles = {"human": "Human TLR4/MD2", "mouse": "Mouse TLR4/MD2"}
    for i, sp in enumerate(species_list):
        ax = axes[i, 0]
        draw_all_strains_panel(ax, df, sp, rf_xpos, rf_vals_sorted)
        ax.set_title(
            f"{PANEL_LABELS[i]}.  {subtitles.get(sp, SPECIES_LABELS.get(sp, sp))}",
            fontsize=11, loc="left", pad=8, fontweight="bold",
        )

    # Shared legend
    present_strains = [s for s in ALL_STRAINS if s in df["Strain"].values]
    legend_handles  = (
        [Line2D([0], [0], marker=DOSE_MARKERS[d], color="gray", linestyle="None",
                markersize=8, label=DOSE_LABELS.get(d, d))
         for d in DOSE_ORDER]
        + [Line2D([0], [0], marker="o", color=POS_CTRL_COLOR, linestyle="None",
                  markersize=8, label="Positive control")]
        + [Line2D([0], [0], marker="o", color=STRAIN_COLORS[s], linestyle="None",
                  markersize=8, label=f"Strain {s}")
           for s in present_strains]
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 1 done")


# ── Figure 2 — Per-strain species comparison, 2×2 ────────────────────────────

def make_figure2(df, pdf):
    present_strains = [s for s in ALL_STRAINS if s in df["Strain"].values]
    pos_sub         = df[df["Strain"] == POS_CTRL_STRAIN]
    n               = len(present_strains)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6.5), squeeze=False)
    fig.suptitle(
        "Figure 2.  HEK293 TLR4/MD2 Human vs Mouse Comparison by Strain",
        fontsize=13, fontweight="bold",
    )

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_visible(False)
            continue

        strain     = present_strains[i]
        strain_sub = df[df["Strain"] == strain]
        rf_vals    = sorted(
            [r for r in strain_sub["Rf"].dropna().unique()],
            key=lambda r: float(r) if r != "mix" else -1.0,
        )
        rf_xpos = build_rf_xpos(rf_vals)

        draw_species_panel(ax, df, strain, rf_xpos, rf_vals, pos_sub)
        ax.set_title(
            f"{PANEL_LABELS[i]}.  Strain {strain}",
            fontsize=11, loc="left", pad=8, fontweight="bold",
        )

    # Shared legend
    present_species = [sp for sp in ["human", "mouse"] if sp in df["Species"].values]
    legend_handles  = (
        [Line2D([0], [0], marker=DOSE_MARKERS[d], color="gray", linestyle="None",
                markersize=8, label=DOSE_LABELS.get(d, d))
         for d in DOSE_ORDER]
        + [Line2D([0], [0], marker="o", color=SPECIES_COLORS[sp], linestyle="None",
                  markersize=8, label=SPECIES_LABELS[sp])
           for sp in present_species]
    )
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 2 done")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    df = load_data(sys.argv[1] if len(sys.argv) > 1 else "HEK_tidy_dose.csv")
    df = df[df["Strain"].isin(ALL_STRAINS + [POS_CTRL_STRAIN])]

    outname = "HEK293_figures_stacked.pdf"
    with PdfPages(outname) as pdf:
        make_figure1(df, pdf)
        make_figure2(df, pdf)

    print(f"\nSaved {outname}")


if __name__ == "__main__":
    main()
