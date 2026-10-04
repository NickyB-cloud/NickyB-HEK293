#!/usr/bin/env python3
"""
HEK293_flex_NB.py
Plots the tidy table made by Parse_Flex_NB.py (HEK293 reporter cells,
absorbance or fluorescence, fold change over the Negative_Control).

Usage:
    python HEK293_flex_NB.py combined_tidy.csv
Output: HEK293_flex_<timestamp>.pdf
    all experiments together (points = experiment means), then its table
    one graph per experiment (points = wells), each followed by its table
    Tables: one row per sample x dose with N and the P value (+ stars) vs
    Negative_Control / Positive_Control when those stats are switched on.

X-axis: Positive_Control | break | each strain's Rf values under a bracket
labelled with the strain (numbers first, then names). Names without a second
part get a single tick. All doses on one plot, one marker shape per dose.
The Negative_Control is not drawn (it is 1 by definition).

Stats are on log2 fold change, per dose, and need n >= 3 per group:
    gray stars (row above the plot)   each item vs Negative_Control (Dunnett)
    red stars  (row above the gray)   each item vs Positive_Control (Dunnett)
    blue bars + stars                 items within a strain (ANOVA + Tukey)
The Pool/Match settings for both controls are read from Parse_Flex_NB.py.
On page 1, n = number of experiments (each experiment is averaged first).
Stars: * P<0.05, ** P<0.01, *** P<0.001.
"""
import os
import sys
import importlib.util
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from datetime import datetime

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")

#----------------------------------
#----------Statistics--------------
#----------------------------------
# Comment a line out (#) to turn that set of stars off.

Stats_vs_Negative = True
#Stats_vs_Positive = True
#Stats_Within_Strain = True

#--------------------------------
#---------Constants--------------
#--------------------------------

POS_CTRL = "Positive_Control"
NEG_CTRL = "Negative_Control"
POS_CTRL_COLOR = "#9467bd"

STRAIN_COLORS = plt.get_cmap("tab10").colors
DOSE_MARKERS  = ["o", "s", "^", "D", "v", "P", "X", "*"]

GRAY_STAR = "#555555"
RED_STAR  = "#c00000"
BLUE_STAR = "#1f4fbf"
SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
MIN_N      = 3

XPOS_POS_CTRL = 0
BREAK_X       = 1.0
BREAK_WIDTH   = 0.35
BREAK_YREL    = -0.10
STRAIN_GAP    = 0.8        # extra space between strain brackets

# Star rows above the data (axes fraction): one thin line per dose, highest
# dose on top; gray block (vs negative) below the red block (vs positive)
SUBROW_H  = 0.032
BLOCK_GAP = 0.02
TOP_PAD   = 0.01


#--------------------------------
#---------Settings from parser---
#--------------------------------

def load_parser_settings():
    """Read the Pool/Match control settings from Parse_Flex_NB.py (same folder)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Parse_Flex_NB.py")
    spec = importlib.util.spec_from_file_location("Parse_Flex_NB", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return settings_from(mod)


def settings_from(mod):
    """
    Negative: Pool (all Negative_Control wells) or Match (same dose).
    Positive: Pool (all Positive_Control wells as one group, compared with
              every dose) or Match (same dose).
    """
    neg_match = getattr(mod, "Match_Negative_Control", False) is True
    neg_pool  = getattr(mod, "Pool_Negative_Control", False) is True
    if neg_match == neg_pool:
        raise ValueError("Parse_Flex_NB.py: set exactly one of "
                         "Pool_Negative_Control / Match_Negative_Control")

    pos_match = getattr(mod, "Match_Positive_Control", False) is True
    pos_pool  = getattr(mod, "Pool_Positive_Control", False) is True
    if pos_match and pos_pool:
        raise ValueError("Parse_Flex_NB.py: set only one of "
                         "Pool_Positive_Control / Match_Positive_Control")
    pos_mode = "match" if pos_match else "pool"

    return ("match" if neg_match else "pool"), pos_mode


#--------------------------------
#---------Sorting / x-axis-------
#--------------------------------

def sort_key(v):
    """Numbers first (by value), then names (alphabetical)."""
    try:
        return (0, float(v), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(v).lower())


def build_xaxis(df):
    """
    Returns
    xpos     {(strain, rf): x}
    ticks    [(x, label)]
    brackets [(strain, x_left, x_right)]  strains with Rf values only
    colors   {strain: color}
    """
    items  = df[~df["Name"].isin([POS_CTRL, NEG_CTRL])]
    strains = sorted(items["Strain"].unique(), key=sort_key)
    colors  = {s: STRAIN_COLORS[i % len(STRAIN_COLORS)] for i, s in enumerate(strains)}
    xpos, ticks, brackets = {}, [(XPOS_POS_CTRL, "Pos ctrl")], []
    x = 2.0
    for strain in strains:
        rfs = sorted(items.loc[items["Strain"] == strain, "Rf"].unique(), key=sort_key)
        start = x
        for rf in rfs:
            xpos[(strain, rf)] = x
            ticks.append((x, rf if rf != "" else strain))
            x += 1.0
        if any(rf != "" for rf in rfs):
            brackets.append((strain, start, x - 1.0))
        x += STRAIN_GAP
    return xpos, ticks, brackets, colors


def dose_offsets(doses):
    n = len(doses)
    if n == 1:
        return {doses[0]: 0.0}
    span = min(0.5, 0.15 * (n - 1))
    return {d: -span / 2 + span * i / (n - 1) for i, d in enumerate(doses)}


def draw_x_break(ax):
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for sign in (-1, 1):
        xs = [BREAK_X + sign * BREAK_WIDTH * 0.6 - 0.015,
              BREAK_X + sign * BREAK_WIDTH * 0.6 + 0.015]
        ax.plot(xs, [BREAK_YREL - 0.03, BREAK_YREL + 0.03],
                transform=trans, color="black", lw=1.5, clip_on=False)
    ax.axvspan(BREAK_X - BREAK_WIDTH, BREAK_X + BREAK_WIDTH,
               ymin=BREAK_YREL - 0.015, ymax=BREAK_YREL + 0.015,
               transform=trans, color="white", zorder=10, clip_on=False)


def draw_bracket(ax, x_left, x_right, label, color, yrel=-0.16):
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    pad = 0.3 if x_left == x_right else 0.0
    xl, xr = x_left - pad, x_right + pad
    cx, tip = (xl + xr) / 2.0, yrel - 0.04
    ax.plot([xl, xl, xr, xr], [yrel, yrel - 0.01, yrel - 0.01, yrel],
            transform=trans, color="black", lw=1.2, clip_on=False)
    ax.plot([cx, cx], [yrel - 0.01, tip], transform=trans,
            color="black", lw=1.2, clip_on=False)
    ax.text(cx, tip - 0.02, label, transform=trans, ha="center", va="top",
            fontsize=11, fontweight="bold", color=color, clip_on=False)


def labels_overlap(ax, pad_px=4):
    """True if any two neighbouring x tick labels touch (measured as drawn)."""
    renderer = ax.figure.canvas.get_renderer()
    boxes = sorted((l.get_window_extent(renderer) for l in ax.get_xticklabels()
                    if l.get_text()), key=lambda b: b.x0)
    return any(b1.x1 + pad_px > b2.x0 for b1, b2 in zip(boxes, boxes[1:]))


def lowest_label_y(ax):
    """Bottom of the lowest x tick label, in axes fraction."""
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    return min(inv.transform((0, l.get_window_extent(renderer).y0))[1]
               for l in ax.get_xticklabels() if l.get_text())


def setup_xaxis(ax, ticks, brackets, colors):
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([l for _, l in ticks], fontsize=10)
    ax.set_xlim(-0.6, max(t for t, _ in ticks) + 0.6)
    # Angle ALL labels only if any two would overlap; brackets drop below them
    if labels_overlap(ax):
        for l in ax.get_xticklabels():
            l.set_rotation(45)
            l.set_horizontalalignment("right")
            l.set_rotation_mode("anchor")
    draw_x_break(ax)
    yrel = min(-0.16, lowest_label_y(ax) - 0.03)
    for strain, xl, xr in brackets:
        draw_bracket(ax, xl, xr, strain, colors[strain], yrel=yrel)


#--------------------------------
#---------Y-axis (log2, adjusts)-
#--------------------------------

def fmt_linear(v):
    return f"{v:g}"


def setup_log2_yaxis(ax, ymin, ymax):
    lo, hi = int(np.floor(ymin)), int(np.ceil(ymax))
    if hi - lo < 2:
        hi = lo + 2
    ticks = list(range(lo, hi + 1))
    ax.set_yticks(ticks)
    ax.set_yticklabels([fmt_linear(2.0 ** t) for t in ticks])
    ax.set_ylabel("Fold change over Negative_Control (log₂ scale)")
    ax.axhline(0, color="black", lw=0.6, ls=":", zorder=1)       # 1-fold
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    return lo - 0.3, hi


#--------------------------------
#---------Statistics-------------
#--------------------------------

def stars(p):
    for threshold, s in SIG_LEVELS:
        if p < threshold:
            return s
    return ""


def log2_values(sub):
    v = sub["FoldChange"].dropna().values.astype(float)
    return np.log2(v[v > 0])


def per_experiment_means(df):
    """One row per Experiment x Name x Dose: mean of log2 fold change."""
    d = df[df["FoldChange"] > 0].copy()
    d["log2FC"] = np.log2(d["FoldChange"])
    keys = ["Experiment", "Name", "Strain", "Rf", "Dose", "DoseValue"]
    m = d.groupby(keys, as_index=False, dropna=False)["log2FC"].mean()   # keep "pooled" (no number)
    m["FoldChange"] = 2.0 ** m["log2FC"]
    return m.drop(columns="log2FC")


def control_values(df, ctrl, dose, mode):
    """log2 values of a control for one dose under pool / match mode."""
    c = df[df["Name"] == ctrl]
    if mode == "match":
        c = c[c["Dose"] == dose]
    if mode == "pool" and "Well" not in c.columns:
        # experiment means (page 1): pool the doses within each experiment so
        # the control has one value per experiment, not one per dose
        c = c[c["FoldChange"] > 0]
        return np.log2(c["FoldChange"]).groupby(c["Experiment"]).mean().values
    return log2_values(c)


def dunnett_vs_control(df, items, dose, ctrl, mode):
    """{(strain, rf): p} for every item with n >= MIN_N at this dose."""
    ctrl_vals = control_values(df, ctrl, dose, mode)
    if len(ctrl_vals) < MIN_N:
        return {}
    keys, samples = [], []
    for (strain, rf), g in items[items["Dose"] == dose].groupby(["Strain", "Rf"]):
        v = log2_values(g)
        if len(v) >= MIN_N:
            keys.append((strain, rf))
            samples.append(v)
    if not samples:
        return {}
    res = stats.dunnett(*samples, control=ctrl_vals, rng=0)   # fixed seed: same P every run
    return dict(zip(keys, res.pvalue))


def tukey_within_strain(items, dose):
    """[(strain, rf1, rf2, p)] for strains whose ANOVA at this dose is P < 0.05."""
    out = []
    d = items[items["Dose"] == dose]
    for strain, s in d.groupby("Strain"):
        groups = {rf: log2_values(g) for rf, g in s.groupby("Rf")}
        groups = {rf: v for rf, v in groups.items() if len(v) >= MIN_N}
        if len(groups) < 2:
            continue
        _, p_anova = stats.f_oneway(*groups.values())
        if not p_anova < 0.05:
            continue
        vals   = np.concatenate(list(groups.values()))
        labels = np.concatenate([[rf] * len(v) for rf, v in groups.items()])
        t = pairwise_tukeyhsd(vals, labels, alpha=0.05)
        tdf = pd.DataFrame(t._results_table.data[1:], columns=t._results_table.data[0])
        for _, r in tdf.iterrows():
            out.append((strain, str(r["group1"]), str(r["group2"]), float(r["p-adj"])))
    return out


#--------------------------------
#---------Drawing----------------
#--------------------------------

def geomean_and_gsem(values):
    x = np.array(values, dtype=float)
    x = x[x > 0]
    if len(x) == 0:
        return np.nan, 0.0
    logs = np.log(x)
    gm = np.exp(logs.mean())
    if len(x) == 1:
        return gm, 0.0
    return gm, gm * (np.exp(logs.std(ddof=1) / np.sqrt(len(x))) - 1.0)


def scatter_with_mean(ax, x, values, color, marker, mean_marker=None):
    v = np.array(values, dtype=float)
    v = v[v > 0]
    if len(v) == 0:
        return
    ax.scatter([x] * len(v), np.log2(v), color=color, marker=marker,
               alpha=0.35, s=25, zorder=3)
    gm, err = geomean_and_gsem(v)
    up = np.log2(gm + err) - np.log2(gm) if err > 0 else 0.0
    ax.errorbar(x, np.log2(gm), yerr=up, fmt="none", ecolor=color,
                elinewidth=1.2, capsize=3, zorder=5)
    if mean_marker == "_":                         # pooled control: horizontal bar
        ax.scatter(x, np.log2(gm), color=color, marker="_", s=500,
                   linewidths=3, zorder=6)
    else:
        ax.scatter(x, np.log2(gm), color=color, marker=marker, s=80,
                   edgecolor="black", zorder=6)


def draw_blue_bars(ax, bar_specs, y_start, step):
    """Stack blue Tukey bars from y_start up; returns the highest y used."""
    placed, top = [], y_start
    for x1, x2, s in sorted(bar_specs, key=lambda b: b[1] - b[0]):
        y = y_start
        while any(not (x2 + 0.1 <= px1 or x1 - 0.1 >= px2) and abs(py - y) < step * 0.9
                  for px1, px2, py in placed):
            y += step
        placed.append((x1, x2, y))
        ax.plot([x1, x1, x2, x2], [y, y + step * 0.3, y + step * 0.3, y],
                lw=1.0, color=BLUE_STAR)
        ax.text((x1 + x2) / 2, y + step * 0.32, s, ha="center", va="bottom",
                fontsize=9, color=BLUE_STAR)
        top = max(top, y + step)
    return top


def draw_panel(ax, df, stats_df, title, neg_mode, pos_mode, n_label):
    """
    df        values to plot (wells, or experiment means on page 1)
    stats_df  values the tests use (same as df here)
    """
    xpos, ticks, brackets, colors = build_xaxis(df)
    # a pooled positive control carries its own label, not one of the plate doses
    dose_src = df[df["Name"] != POS_CTRL] if pos_mode == "pool" else df
    doses = sorted(dose_src["Dose"].dropna().unique(),
                   key=lambda d: (df.loc[df["Dose"] == d, "DoseValue"].iloc[0], d))
    markers = {d: DOSE_MARKERS[i % len(DOSE_MARKERS)] for i, d in enumerate(doses)}
    offs    = dose_offsets(doses)

    # Positive control: Match = one marker per dose; Pool = one mean (bar)
    pos = df[df["Name"] == POS_CTRL]
    if pos_mode == "pool":
        v = 2.0 ** control_values(df, POS_CTRL, None, "pool")
        scatter_with_mean(ax, XPOS_POS_CTRL, v, POS_CTRL_COLOR, "o", mean_marker="_")
    else:
        for d in doses:
            v = pos.loc[pos["Dose"] == d, "FoldChange"].values
            scatter_with_mean(ax, XPOS_POS_CTRL + offs[d], v, POS_CTRL_COLOR, markers[d])

    items = df[~df["Name"].isin([POS_CTRL, NEG_CTRL])]
    for (strain, rf), g in items.groupby(["Strain", "Rf"]):
        for d in doses:
            v = g.loc[g["Dose"] == d, "FoldChange"].values
            scatter_with_mean(ax, xpos[(strain, rf)] + offs[d], v,
                              colors[strain], markers[d])

    # y range from the data, then room for blue bars and the star rows
    allv = df.loc[df["Name"] != NEG_CTRL, "FoldChange"]
    allv = np.log2(allv[allv > 0])
    ymin, ymax = setup_log2_yaxis(ax, min(allv.min(), 0.0), max(allv.max(), 1.0))
    step = (ymax - ymin) * 0.06
    content_top = ymax

    stat_items = stats_df[~stats_df["Name"].isin([POS_CTRL, NEG_CTRL])]
    if globals().get("Stats_Within_Strain") is True:
        specs = []
        for d in doses:
            for strain, rf1, rf2, p in tukey_within_strain(stat_items, d):
                s = stars(p)
                if s:
                    specs.append((xpos[(strain, rf1)] + offs[d],
                                  xpos[(strain, rf2)] + offs[d], s))
        if specs:
            content_top = draw_blue_bars(ax, specs, ymax + step * 0.3, step)

    blocks = []
    if globals().get("Stats_vs_Negative") is True:
        blocks.append((GRAY_STAR, NEG_CTRL, neg_mode))
    if globals().get("Stats_vs_Positive") is True:
        blocks.append((RED_STAR, POS_CTRL, pos_mode))
    block_h = len(doses) * SUBROW_H
    content_frac = 1.0 - TOP_PAD - len(blocks) * (block_h + BLOCK_GAP)
    if not blocks:
        content_frac = 0.95

    # data + blue bars occupy the bottom content_frac of the axes; star rows above
    ax.set_ylim(ymin, ymin + (content_top - ymin) / content_frac)
    xt = ax.get_xaxis_transform()

    results = {}                                   # {ctrl: {(strain, rf, dose): p}}
    for b, (color, ctrl, mode) in enumerate(blocks):
        results[ctrl] = {}
        bottom = content_frac + BLOCK_GAP + b * (block_h + BLOCK_GAP)
        for i, d in enumerate(doses):              # doses low -> high = bottom -> top
            y_row = bottom + (i + 0.5) * SUBROW_H
            # dose key at the left of each line (above the positive control)
            ax.plot([XPOS_POS_CTRL], [y_row], transform=xt, marker=markers[d],
                    markersize=4, linestyle="None", color=color, clip_on=False,
                    zorder=7)
            for key, p in dunnett_vs_control(stats_df, stat_items, d, ctrl, mode).items():
                results[ctrl][(key[0], key[1], d)] = p
                s = stars(p)
                if s:
                    ax.text(xpos[key] + offs[d], y_row, s, transform=xt,
                            ha="center", va="center", fontsize=8,
                            fontweight="bold", color=color, clip_on=False)

    setup_xaxis(ax, ticks, brackets, colors)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=12, pad=12)
    ax.text(0.01, 0.01, n_label, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=8)

    handles = [Line2D([0], [0], marker=markers[d], color="gray", linestyle="None",
                      markersize=8, label=d) for d in doses]
    if pos_mode == "pool":
        handles.append(Line2D([0], [0], marker="_", color=POS_CTRL_COLOR, linestyle="None",
                              markersize=16, markeredgewidth=3,
                              label=("Positive control (pooled)" if pooled_label(df) == "pooled"
                                     else f"Positive control ({pooled_label(df)}, pooled)")))
    else:
        handles.append(Line2D([0], [0], marker="o", color=POS_CTRL_COLOR,
                              linestyle="None", markersize=8, label="Positive control"))
    star_note = []
    if globals().get("Stats_vs_Negative") is True:
        star_note.append(Line2D([0], [0], color="none", label="* vs Negative_Control"))
    if globals().get("Stats_vs_Positive") is True:
        star_note.append(Line2D([0], [0], color="none", label="* vs Positive_Control"))
    if globals().get("Stats_Within_Strain") is True:
        star_note.append(Line2D([0], [0], color="none", label="* within strain"))
    leg = ax.legend(handles=handles + star_note, fontsize=8, frameon=False,
                    loc="upper left", bbox_to_anchor=(1.01, 1.0))
    note_colors = ([GRAY_STAR] if globals().get("Stats_vs_Negative") is True else []) \
        + ([RED_STAR] if globals().get("Stats_vs_Positive") is True else []) \
        + ([BLUE_STAR] if globals().get("Stats_Within_Strain") is True else [])
    for text, c in zip(leg.get_texts()[len(handles):], note_colors):
        text.set_color(c)
    return results, doses


def n_range(counts, unit):
    if len(counts) == 0:
        return f"n = 0 {unit}"
    lo, hi = int(counts.min()), int(counts.max())
    return f"n = {lo} {unit}" if lo == hi else f"n = {lo}\u2013{hi} {unit}"


def group_counts(df):
    items = df[~df["Name"].isin([POS_CTRL, NEG_CTRL]) & (df["FoldChange"] > 0)]
    return items.groupby(["Strain", "Rf", "Dose"])["FoldChange"].count()


TABLE_ROWS_PER_PAGE = 28


def pooled_label(df):
    """Dose label of a pooled positive control, e.g. '1 ug/ml', else 'pooled'."""
    labels = df.loc[df["Name"] == POS_CTRL, "Dose"].dropna().unique()
    return labels[0] if len(labels) == 1 else "pooled"


def positive_control_rows(data, doses, neg_mode, pos_mode, ctrls):
    """
    Positive_Control row(s) for the top of the table: one per dose (Match) or
    one "pooled" row (Pool), with N and its P value vs Negative_Control.
    """
    groups = []
    if pos_mode == "pool":
        groups.append((pooled_label(data), control_values(data, POS_CTRL, None, "pool"),
                       control_values(data, NEG_CTRL, None, "pool")))
    else:
        for d in doses:
            groups.append((d, control_values(data, POS_CTRL, d, "match"),
                           control_values(data, NEG_CTRL, d, neg_mode)))
    rows = []
    for label, pos_vals, neg_vals in groups:
        n = len(pos_vals)
        if n == 0:
            continue
        row = [POS_CTRL, "-", label, str(n)]
        for c in ctrls:
            if c == POS_CTRL:
                row.append("-")
            elif n < MIN_N or len(neg_vals) < MIN_N:
                row.append(f"n < {MIN_N}")
            else:
                p = float(stats.dunnett(pos_vals, control=neg_vals, rng=0).pvalue[0])
                row.append(f"{p:.4f} {stars(p)}".strip())
        rows.append(row)
    return rows


def draw_tables(pdf, title, counts, results, doses, unit, data, neg_mode, pos_mode):
    """Table page(s): Test | Sub | Dose | N | vs Neg | vs Pos; Positive_Control first."""
    ctrls = [c for c in (NEG_CTRL, POS_CTRL) if c in results]
    short  = {NEG_CTRL: "Neg_ctrl", POS_CTRL: "Pos_ctrl"}
    header = ["Test", "Sub", "Dose", "N"] + [short[c] for c in ctrls]
    dose_rank = {d: i for i, d in enumerate(doses)}
    keys = sorted(counts.index, key=lambda k: (sort_key(k[0]), sort_key(k[1]),
                                               dose_rank.get(k[2], 99)))
    rows = positive_control_rows(data, doses, neg_mode, pos_mode, ctrls)
    for strain, rf, dose in keys:
        n = int(counts[(strain, rf, dose)])
        row = [strain, rf if rf != "" else "-", dose, str(n)]
        for c in ctrls:
            p = results[c].get((strain, rf, dose))
            if n < MIN_N:
                row.append(f"n < {MIN_N}")
            elif p is None:
                row.append("-")
            else:
                row.append(f"{p:.4f} {stars(p)}".strip())
        rows.append(row)

    chunks = [rows[i:i + TABLE_ROWS_PER_PAGE]
              for i in range(0, len(rows), TABLE_ROWS_PER_PAGE)] or [[]]
    for i, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        part = f" (part {i + 1} of {len(chunks)})" if len(chunks) > 1 else ""
        ax.set_title(f"{title}{part}", fontsize=12, pad=12, loc="left")
        if chunk:
            t = ax.table(cellText=chunk, colLabels=header, loc="upper center",
                         cellLoc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(9)
            t.scale(1, 1.4)
            for (r, c), cell in t.get_celld().items():
                if r == 0:
                    cell.set_text_props(fontweight="bold")
                    cell.set_facecolor("#eeeeee")
                elif c >= 4 and "*" in cell.get_text().get_text():
                    cell.get_text().set_color(RED_STAR if header[c] == "Pos_ctrl"
                                              else GRAY_STAR)
                    cell.get_text().set_fontweight("bold")
        ax.text(0.0, 0.0, "P values: Dunnett's test on log2 fold change, per dose. "
                "* P<0.05, ** P<0.01, *** P<0.001.",
                transform=ax.transAxes, fontsize=8, va="bottom")
        pdf.savefig(fig)
        plt.close(fig)


def new_figure(n_x):
    fig, ax = plt.subplots(figsize=(max(8.0, 0.75 * n_x + 4.0), 6.5))
    fig.subplots_adjust(bottom=0.26, top=0.88, right=0.82)
    return fig, ax


#--------------------------------
#---------Run--------------------
#--------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "combined_tidy.csv"
    df = pd.read_csv(path, dtype={"Strain": str, "Rf": str, "Dose": str})
    df["Strain"] = df["Strain"].fillna("")
    df["Rf"]     = df["Rf"].fillna("")
    neg_mode, pos_mode = load_parser_settings()

    exps = sorted(df["Experiment"].unique(), key=sort_key)
    n_x  = len(build_xaxis(df)[1])
    out  = f"HEK293_flex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    with PdfPages(out) as pdf:
        # Page 1: all experiments, each experiment averaged first (n = experiments)
        means  = per_experiment_means(df)
        counts = group_counts(means)
        title  = f"All experiments ({len(exps)}): points = experiment means"
        fig, ax = new_figure(n_x)
        results, doses = draw_panel(ax, means, means, title, neg_mode, pos_mode,
                                    n_range(counts, "experiments"))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        draw_tables(pdf, title, counts, results, doses, "experiments",
                    means, neg_mode, pos_mode)

        # One page per experiment (n = wells)
        for exp in exps:
            sub    = df[df["Experiment"] == exp]
            counts = group_counts(sub)
            title  = f"{exp}: points = wells"
            fig, ax = new_figure(n_x)
            results, doses = draw_panel(ax, sub, sub, title, neg_mode, pos_mode,
                                        n_range(counts, "wells"))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            draw_tables(pdf, title, counts, results, doses, "wells",
                        sub, neg_mode, pos_mode)

    print(f"Saved {out}")


if __name__ == "__main__":
    main()
