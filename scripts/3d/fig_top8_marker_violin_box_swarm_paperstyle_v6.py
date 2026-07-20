#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REAL-DATA ONLY
Top Raman markers: paper-style violin + box + jitter + nonparametric statistics (v6 paper-style).

This version is designed for the LUAD 3D Raman project.

Input files (searched recursively below --input-dir)
-----------------------------------------------------
- source_embedding_coordinates.csv
- source_true_pixel_robust_z_features.csv

Figure layout
-------------
- Top N markers (default 12), arranged 4 panels per row
- Violin + box + jittered pixel values
- Nature-style muted palette
- No global Kruskal-Wallis q-value printed inside panels by default
- Pairwise statistics use Dunn's post-hoc test after Kruskal-Wallis,
  with Benjamini-Hochberg FDR correction across ALL 21 group-pairs per marker.
- To keep panels readable, only the strongest FDR-significant comparisons
  are drawn as brackets (default: up to 3), but all 21 comparisons per marker
  are exported to CSV.

Important interpretation
------------------------
The source values are pixel-level Raman features. The figure is descriptive
of pixel-level distributions. Do not interpret p-values as independent
patient-level replication unless a specimen/layer-level sensitivity analysis
is added separately.

PowerShell example
------------------
py .\scripts\3d\fig_top8_marker_violin_box_swarm_paperstyle_v6.py --input-dir ".\3dresults" --out-dir ".\3dresults\fig_top12_marker_violin_paperstyle" --top-n 12 --cols 4 --palette cream-mint-pink --pairwise-display top2
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import kruskal, rankdata, norm

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.titleweight"] = "bold"

GROUP_ORDER = ["CA", "LEP", "ACN", "PAP", "MP", "CGP", "SOL"]
GROUP_FULL = {
    "CA": "Cancer-adjacent",
    "LEP": "Lepidic",
    "ACN": "Acinar",
    "PAP": "Papillary",
    "MP": "Micropapillary",
    "CGP": "Complex glands",
    "SOL": "Solid",
}
GROUP_ALIASES = {
    "ca": "CA", "cancer-adjacent": "CA", "cancer adjacent": "CA", "adjacent": "CA", "normal": "CA", "pneumonia": "CA",
    "lep": "LEP", "lepidic": "LEP", "tiebi": "LEP",
    "acn": "ACN", "acinar": "ACN", "xianpao": "ACN",
    "pap": "PAP", "papillary": "PAP",
    "mp": "MP", "micropapillary": "MP", "micropap": "MP",
    "cgp": "CGP", "complex glands": "CGP", "complex_glands": "CGP", "fzxt": "CGP",
    "sol": "SOL", "solid": "SOL", "soild": "SOL",
}

# High-end Nature-like palette: restrained, warm/cool balanced, publication-friendly.
NATURE_PALETTE = {
    "CA":  "#8B98A5",  # blue-grey
    "LEP": "#3978A8",  # ocean blue
    "ACN": "#148C83",  # teal
    "PAP": "#77A968",  # sage green
    "MP":  "#D5A43D",  # muted ochre
    "CGP": "#CE7B50",  # terracotta
    "SOL": "#B84D5C",  # muted crimson
}

ALT_PALETTES = {
    "nature": NATURE_PALETTE,
    "nature-soft": {
        "CA": "#A7B4BD", "LEP": "#71A6CB", "ACN": "#55AD9E", "PAP": "#9CC58B",
        "MP": "#E6C66B", "CGP": "#E8A77F", "SOL": "#D98B98",
    },
    "morandi": {
        "CA": "#939DA5", "LEP": "#739DC0", "ACN": "#639D91", "PAP": "#91AD79",
        "MP": "#CDB36C", "CGP": "#D39377", "SOL": "#BE777B",
    },
    # Reference-style soft pastel palette, matched to the example you sent:
    # cool blue/green on the left and soft sand/pink on the right.
    "ref-pastel": {
        "CA":  "#AEBFCC",  # dusty blue-grey
        "LEP": "#8DBFE3",  # soft sky blue
        "ACN": "#79CDB8",  # aqua green
        "PAP": "#93D28D",  # pastel green
        "MP":  "#E7CC86",  # pale sand
        "CGP": "#EDB78A",  # soft apricot
        "SOL": "#E8A0A4",  # dusty pink
    },
    # slightly deeper version if you want a bit more contrast
    "ref-pastel-deep": {
        "CA":  "#98ACBC",
        "LEP": "#6EAAD7",
        "ACN": "#5BBDA6",
        "PAP": "#74C56F",
        "MP":  "#DDBA63",
        "CGP": "#E39E6A",
        "SOL": "#DC7F88",
    },
    # Minimal palette in the exact spirit of your example:
    # cream + light green + light pink, with very soft transitions.
    "cream-mint-pink": {
        "CA":  "#B9C7D3",  # mist blue-grey
        "LEP": "#98C8E8",  # pale sky blue
        "ACN": "#9FD8C3",  # mint green
        "PAP": "#B7DEAA",  # light green
        "MP":  "#EED9A7",  # butter cream
        "CGP": "#F2C8A7",  # soft peach cream
        "SOL": "#E8B1B3",  # light pink
    },
    # Slightly brighter minimalist variant
    "cream-mint-pink-bright": {
        "CA":  "#AEBECC",
        "LEP": "#86BEE4",
        "ACN": "#89D1B7",
        "PAP": "#A8D690",
        "MP":  "#E8CC84",
        "CGP": "#EEB88F",
        "SOL": "#E39A9F",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-style top Raman marker violin plot with Dunn post-hoc statistics")
    parser.add_argument("--input-dir", required=True, help="Root directory searched recursively for the two source CSVs")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--top-n", type=int, default=8, help="Number of top markers to plot")
    parser.add_argument("--cols", type=int, default=4, help="Panels per row")
    parser.add_argument("--points-per-group", type=int, default=160, help="Maximum jitter points shown per group")
    parser.add_argument("--min-group-size", type=int, default=20, help="Minimum finite values per group during marker ranking")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--group-col", type=str, default="auto", help="Group/pathology column; default detects group/pathology/state/label/class")
    parser.add_argument("--palette", choices=list(ALT_PALETTES.keys()), default="cream-mint-pink")
    parser.add_argument("--pairwise-display", choices=["top3", "top2", "all", "none"], default="top2",
                        help="Brackets displayed per panel. All 21 pairwise Dunn tests are always exported.")
    parser.add_argument("--show-global-q", action="store_true", help="Print global Kruskal-Wallis BH-FDR q inside each panel (default off)")
    parser.add_argument("--title", default="Top 8 discriminative Raman markers across LUAD growth patterns")
    parser.add_argument("--subtitle", default="Paper-style violin + box + jitter | strongest FDR-significant Dunn comparisons shown")
    parser.add_argument("--out-name", default="Fig_top8_marker_violin_paperstyle")
    parser.add_argument("--show-legend", action="store_true", help="Show bottom color legend. Default is off for cleaner paper-style output")
    return parser.parse_args()


def clean_text(x: str) -> str:
    return " ".join(str(x).replace("_", " ").split()).strip()


def canonical_group(x: str) -> str | None:
    return GROUP_ALIASES.get(clean_text(x).lower())


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[valid] = tmp
    return out


def find_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct
    hits = sorted(root.rglob(filename), key=lambda p: (len(p.parts), str(p)))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return hits[0]


def detect_group_col(df: pd.DataFrame, user_value: str) -> str:
    if user_value != "auto":
        if user_value not in df.columns:
            raise KeyError(f"Specified --group-col '{user_value}' not found. Available columns: {list(df.columns)}")
        return user_value
    for col in df.columns:
        if col.lower() in {"group", "pathology", "state", "label", "class"}:
            return col
    raise KeyError("Could not auto-detect group/pathology column in source_embedding_coordinates.csv")


def load_sources(input_dir: Path, group_col_arg: str) -> Tuple[pd.Series, pd.DataFrame]:
    coord_path = find_file(input_dir, "source_embedding_coordinates.csv")
    feature_path = find_file(input_dir, "source_true_pixel_robust_z_features.csv")

    coords = pd.read_csv(coord_path)
    raw_features = pd.read_csv(feature_path)
    if len(coords) != len(raw_features):
        raise RuntimeError(f"Row mismatch: coordinates={len(coords)}; features={len(raw_features)}")

    group_col = detect_group_col(coords, group_col_arg)
    groups = coords[group_col].astype(str).map(canonical_group)
    keep = groups.notna().values

    groups = groups.loc[keep].reset_index(drop=True)
    raw_features = raw_features.loc[keep].reset_index(drop=True)

    features = raw_features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_columns = [
        c for c in features.columns
        if np.isfinite(features[c].values).sum() >= 20 and np.nanstd(features[c].values.astype(float)) > 1e-10
    ]
    features = features[valid_columns]
    if features.shape[1] < 12:
        raise RuntimeError("Too few valid marker columns were found in source_true_pixel_robust_z_features.csv")

    return groups, features


def rank_top_markers(groups: pd.Series, features: pd.DataFrame, min_group_size: int, top_n: int) -> pd.DataFrame:
    rows = []
    for marker in features.columns:
        arrays = []
        medians = {}
        valid = True
        for group in GROUP_ORDER:
            values = pd.to_numeric(features.loc[groups == group, marker], errors="coerce").dropna().values
            if len(values) < min_group_size:
                valid = False
                break
            arrays.append(values)
            medians[group] = float(np.median(values))
        if not valid:
            continue
        try:
            h_value, p_value = kruskal(*arrays)
        except Exception:
            continue
        median_range = float(max(medians.values()) - min(medians.values()))
        rows.append({
            "marker": marker,
            "kruskal_H": float(h_value),
            "kruskal_p": float(p_value),
            "median_range": median_range,
            **{f"median_{g}": medians[g] for g in GROUP_ORDER},
        })

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        raise RuntimeError("No marker passed the minimum group-size requirement.")
    stats_df["kruskal_BH_FDR_q"] = bh_fdr(stats_df["kruskal_p"].values)
    stats_df = stats_df.sort_values(
        ["kruskal_H", "median_range", "kruskal_BH_FDR_q"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return stats_df.head(top_n).copy()


def dunn_pairwise_tests(data_by_group: List[np.ndarray], marker: str) -> pd.DataFrame:
    """Dunn test with tie correction, followed by BH-FDR across the 21 group pairs for this marker."""
    clean_groups = []
    for values in data_by_group:
        arr = np.asarray(values, dtype=float)
        clean_groups.append(arr[np.isfinite(arr)])

    sizes = np.asarray([len(v) for v in clean_groups], dtype=float)
    total_n = int(sizes.sum())
    if total_n < 3 or np.any(sizes < 2):
        return pd.DataFrame()

    pooled = np.concatenate(clean_groups)
    ranks = rankdata(pooled, method="average")

    mean_ranks = []
    start = 0
    for n_i in sizes.astype(int):
        mean_ranks.append(float(np.mean(ranks[start:start + n_i])))
        start += n_i
    mean_ranks = np.asarray(mean_ranks, dtype=float)

    # Tie correction used in Dunn's rank-based comparison.
    _, counts = np.unique(pooled, return_counts=True)
    tie_sum = float(np.sum(counts**3 - counts))
    denom = float(total_n**3 - total_n)
    tie_correction = 1.0 - tie_sum / denom if denom > 0 else 1.0
    tie_correction = max(tie_correction, 1e-12)
    base_var = total_n * (total_n + 1.0) / 12.0 * tie_correction

    rows = []
    for i, j in itertools.combinations(range(len(GROUP_ORDER)), 2):
        se = math.sqrt(base_var * (1.0 / sizes[i] + 1.0 / sizes[j]))
        if se <= 0 or not np.isfinite(se):
            continue
        z = (mean_ranks[i] - mean_ranks[j]) / se
        p = float(2.0 * norm.sf(abs(z)))
        med_i = float(np.median(clean_groups[i]))
        med_j = float(np.median(clean_groups[j]))
        # Effect used only to prioritize visible labels, not for significance.
        robust_scale = float(np.subtract(*np.percentile(pooled, [75, 25])))
        if robust_scale <= 1e-12:
            robust_scale = float(np.std(pooled))
        if robust_scale <= 1e-12:
            robust_scale = 1.0
        standardized_median_difference = abs(med_i - med_j) / robust_scale
        rows.append({
            "marker": marker,
            "group_1": GROUP_ORDER[i],
            "group_2": GROUP_ORDER[j],
            "index_1": i,
            "index_2": j,
            "n_1": int(sizes[i]),
            "n_2": int(sizes[j]),
            "mean_rank_1": mean_ranks[i],
            "mean_rank_2": mean_ranks[j],
            "Dunn_z": float(z),
            "Dunn_p": p,
            "median_1": med_i,
            "median_2": med_j,
            "standardized_median_difference": standardized_median_difference,
        })

    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["Dunn_BH_FDR_q"] = bh_fdr(output["Dunn_p"].values)
    output["significance"] = output["Dunn_BH_FDR_q"].map(stars_from_q)
    return output


def stars_from_q(q_value: float) -> str:
    if not np.isfinite(q_value):
        return "ns"
    if q_value < 1e-4:
        return "****"
    if q_value < 1e-3:
        return "***"
    if q_value < 1e-2:
        return "**"
    if q_value < 0.05:
        return "*"
    return "ns"


def choose_display_pairs(pairwise_df: pd.DataFrame, display: str) -> List[dict]:
    if display == "none" or pairwise_df.empty:
        return []

    significant = pairwise_df.loc[pairwise_df["Dunn_BH_FDR_q"] < 0.05].copy()
    if significant.empty:
        return []

    # Display priority: strongest adjusted evidence, then largest robust median separation.
    significant = significant.sort_values(
        ["Dunn_BH_FDR_q", "standardized_median_difference"],
        ascending=[True, False],
    )

    if display == "top2":
        significant = significant.head(2)
    elif display == "top3":
        significant = significant.head(3)
    # display == all retains every significant pair.

    return significant.to_dict(orient="records")


def robust_ylim(data_by_group: List[np.ndarray], n_brackets: int) -> Tuple[float, float, float]:
    pooled = np.concatenate([np.asarray(x, dtype=float)[np.isfinite(x)] for x in data_by_group if len(x) > 0])
    if pooled.size == 0:
        return -1.0, 1.0, 0.2
    lo = float(np.nanmin(pooled))
    hi = float(np.nanmax(pooled))
    span = hi - lo
    if not np.isfinite(span) or span <= 1e-12:
        span = 1.0
    lower_pad = span * 0.08
    upper_pad = span * (0.16 + 0.095 * max(0, n_brackets - 1))
    return lo - lower_pad, hi + upper_pad, span


def add_sig_brackets(ax, pairs: List[dict], base_y: float, step: float) -> None:
    # Short comparisons are lower; long spans are placed higher to prevent crossings.
    ordered = sorted(pairs, key=lambda row: (abs(row["index_2"] - row["index_1"]), row["Dunn_BH_FDR_q"]))
    for level, row in enumerate(ordered):
        x1 = row["index_1"] + 1
        x2 = row["index_2"] + 1
        if x1 > x2:
            x1, x2 = x2, x1
        y = base_y + level * step
        h = step * 0.28
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#454545", lw=1.1, clip_on=False, zorder=10)
        ax.text((x1 + x2) / 2, y + h + step * 0.06, row["significance"],
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#303030", zorder=11)


def draw_marker_panel(
    ax,
    data_by_group: List[np.ndarray],
    marker: str,
    global_q: float,
    palette: Dict[str, str],
    rng: np.random.Generator,
    points_per_group: int,
    pairwise_df: pd.DataFrame,
    pairwise_display: str,
    show_global_q: bool,
) -> None:
    positions = np.arange(1, len(GROUP_ORDER) + 1)
    visible_pairs = choose_display_pairs(pairwise_df, pairwise_display)
    y_min, y_max, span = robust_ylim(data_by_group, len(visible_pairs))

    # Publication-style white panel with subtle baseline.
    ax.set_facecolor("white")
    ax.axhline(0, color="#E9E9E9", lw=0.9, zorder=0)

    violins = ax.violinplot(
        data_by_group,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, body in enumerate(violins["bodies"]):
        group = GROUP_ORDER[i]
        body.set_facecolor(palette[group])
        body.set_edgecolor(palette[group])
        body.set_linewidth(1.0)
        body.set_alpha(0.42)

    box = ax.boxplot(
        data_by_group,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#2F2F2F", linewidth=1.35),
        whiskerprops=dict(color="#7A7A7A", linewidth=0.95),
        capprops=dict(color="#7A7A7A", linewidth=0.95),
        boxprops=dict(edgecolor="#525252", linewidth=0.95),
    )
    for patch in box["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.92)

    for i, raw_values in enumerate(data_by_group):
        values = np.asarray(raw_values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if values.size > points_per_group:
            values = values[rng.choice(values.size, size=points_per_group, replace=False)]
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        ax.scatter(
            np.full(values.size, positions[i]) + jitter,
            values,
            s=7.0,
            color=palette[GROUP_ORDER[i]],
            alpha=0.43,
            linewidths=0,
            zorder=3,
        )

    ax.set_ylim(y_min, y_max)
    ax.set_title(marker, fontsize=13.3, pad=7, fontweight="bold", color="#222222")

    if show_global_q:
        ax.text(
            0.02, 0.98,
            f"Global q={global_q:.2e}",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9.0, color="#5B5B5B",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.80),
        )

    if visible_pairs:
        base_y = y_max - span * (0.12 + 0.08 * (len(visible_pairs) - 1))
        add_sig_brackets(ax, visible_pairs, base_y=base_y, step=span * 0.068)

    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_ORDER, fontsize=10.6)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=9.2, colors="#4B4B4B")
    ax.grid(axis="y", linestyle="-", linewidth=0.6, color="#EFEFEF", alpha=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.spines["bottom"].set_color("#BDBDBD")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

def make_figure(
    groups: pd.Series,
    features: pd.DataFrame,
    top_markers: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Path, Path, Path, pd.DataFrame]:
    palette = ALT_PALETTES[args.palette]
    markers = top_markers["marker"].tolist()
    global_q_map = dict(zip(top_markers["marker"], top_markers["kruskal_BH_FDR_q"]))

    nrows = math.ceil(len(markers) / args.cols)
    fig_w = 4.55 * args.cols
    fig_h = 3.72 * nrows + 0.92
    fig, axes = plt.subplots(nrows=nrows, ncols=args.cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor("white")
    rng = np.random.default_rng(args.random_seed)

    all_pairwise = []
    for index, marker in enumerate(markers):
        row = index // args.cols
        col = index % args.cols
        ax = axes[row][col]

        values_by_group = [
            pd.to_numeric(features.loc[groups == group, marker], errors="coerce").dropna().values
            for group in GROUP_ORDER
        ]
        pairwise_df = dunn_pairwise_tests(values_by_group, marker=marker)
        if not pairwise_df.empty:
            all_pairwise.append(pairwise_df)

        draw_marker_panel(
            ax=ax,
            data_by_group=values_by_group,
            marker=marker,
            global_q=float(global_q_map.get(marker, np.nan)),
            palette=palette,
            rng=rng,
            points_per_group=args.points_per_group,
            pairwise_df=pairwise_df,
            pairwise_display=args.pairwise_display,
            show_global_q=args.show_global_q,
        )
        if col == 0:
            ax.set_ylabel("Robust z-score", fontsize=10.8, color="#333333")
        else:
            ax.set_ylabel("")

    for index in range(len(markers), nrows * args.cols):
        axes[index // args.cols][index % args.cols].axis("off")

    fig.suptitle(args.title, fontsize=20.5, fontweight="bold", y=0.988)
    fig.text(0.5, 0.958, args.subtitle, ha="center", va="center", fontsize=10.4, color="#666666")

    if args.show_legend:
        group_handles = [Patch(facecolor=palette[group], edgecolor="none", alpha=0.90, label=group) for group in GROUP_ORDER]
        fig.legend(
            handles=group_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.032),
            ncol=len(GROUP_ORDER),
            frameon=False,
            fontsize=10.0,
            handlelength=1.4,
            columnspacing=1.4,
        )

    if args.pairwise_display != "none":
        fig.text(
            0.5, 0.014,
            "Only the strongest FDR-significant pairwise differences are annotated in each panel.",
            ha="center", va="center", fontsize=8.9, color="#707070",
        )

    bottom_margin = 0.072 if args.show_legend else 0.052
    plt.subplots_adjust(left=0.060, right=0.992, top=0.902, bottom=bottom_margin, hspace=0.42, wspace=0.25)

    out_png = output_dir / f"{args.out_name}.png"
    out_pdf = output_dir / f"{args.out_name}.pdf"
    out_svg = output_dir / f"{args.out_name}.svg"
    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    all_pairwise_df = pd.concat(all_pairwise, ignore_index=True) if all_pairwise else pd.DataFrame()
    return out_png, out_pdf, out_svg, all_pairwise_df

def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups, features = load_sources(input_dir, args.group_col)
    top_markers = rank_top_markers(groups, features, args.min_group_size, args.top_n)

    out_png, out_pdf, out_svg, pairwise_df = make_figure(
        groups=groups,
        features=features,
        top_markers=top_markers,
        output_dir=out_dir,
        args=args,
    )

    top_stats_path = out_dir / f"{args.out_name}_top_marker_statistics.csv"
    pairwise_path = out_dir / f"{args.out_name}_all_pairwise_Dunn_BH_FDR.csv"
    top_markers.to_csv(top_stats_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved SVG: {out_svg}")
    print(f"Saved top-marker statistics: {top_stats_path}")
    print(f"Saved all pairwise Dunn statistics: {pairwise_path}")
    print("Top markers:")
    for i, marker in enumerate(top_markers["marker"], 1):
        print(f"  {i:02d}. {marker}")


if __name__ == "__main__":
    main()
