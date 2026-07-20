#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Ridgeline / ridgeplot figure for real 3D Raman data.

Suitable use in this project
----------------------------
1. Marker mode: show the distribution of top discriminative Raman markers
   across LUAD pathology groups.
2. Module mode: show the distribution of Raman module scores across groups.

Input (searched recursively under --input-dir)
---------------------------------------------
- source_embedding_coordinates.csv
- source_true_pixel_robust_z_features.csv

Output
------
- PNG / PDF / SVG ridgeplot figure
- CSV ranking/stat table used for plotting

Typical usage
-------------
py .\scripts\3d\fig_ridgeplot_raman_realdata.py --input-dir ".\3dresults\fig01_3d_pca_umap_clean_twopanel_v7_2000pixels" --out-dir ".\3dresults\fig_ridgeplot" --mode marker --top-n 6
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, kruskal

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

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

# paper-like pastel palette; coherent with user's preferred cream/mint/pink feeling
GROUP_COLORS = {
    "CA":  "#AEBCC8",  # dusty blue-gray
    "LEP": "#97C7EC",  # soft sky
    "ACN": "#77C9B7",  # mint teal
    "PAP": "#A8D39A",  # sage green
    "MP":  "#E8CF8A",  # cream gold
    "CGP": "#E8B07B",  # apricot
    "SOL": "#E59A9A",  # dusty pink
}

MODULE_KEYWORDS = {
    "Central metabolism": ["central", "metab"],
    "Lipid metabolism": ["lipid"],
    "ECM / stromal remodeling": ["ecm", "stromal"],
    "Immune / protein": ["immune", "protein"],
    "Stress / glycoxidation": ["stress", "glycox"],
    "Amino acid / nitrogen": ["amino", "nitrogen"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-style ridgeline plot for real 3D Raman data")
    parser.add_argument("--input-dir", required=True, help="Root folder searched recursively for source CSV files")
    parser.add_argument("--out-dir", required=True, help="Output folder")
    parser.add_argument("--mode", choices=["marker", "module"], default="marker", help="Plot top markers or module scores")
    parser.add_argument("--top-n", type=int, default=6, help="Top N markers to plot in marker mode")
    parser.add_argument("--cols", type=int, default=3, help="Subplot columns")
    parser.add_argument("--min-group-size", type=int, default=20, help="Minimum non-missing values per group")
    parser.add_argument("--group-col", default="auto", help="Pathology/group column name; default auto-detect")
    parser.add_argument("--single-feature", default="", help="Optional: plot only one feature (exact column name) in either mode")
    parser.add_argument("--points-per-group", type=int, default=90, help="Subsampled scatter points per ridge")
    parser.add_argument("--kde-points", type=int, default=300, help="Number of x-grid points for KDE")
    parser.add_argument("--ridge-height", type=float, default=0.82, help="Vertical height scaling for ridges")
    parser.add_argument("--bandwidth-adjust", type=float, default=1.0, help="KDE bandwidth multiplier")
    parser.add_argument("--x-pad-frac", type=float, default=0.08, help="Extra padding fraction added to x-range")
    parser.add_argument("--title", default="Ridgeline distributions of Raman features across LUAD growth patterns", help="Figure title")
    parser.add_argument("--subtitle", default="Real 3D Raman pixel-level data | ridgeline density distributions across pathology groups", help="Figure subtitle")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def clean_text(x: str) -> str:
    return " ".join(str(x).replace("_", " ").split()).strip()


def canonical_group(x: str) -> str | None:
    return GROUP_ALIASES.get(clean_text(x).lower(), None)


def bh_fdr(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
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
    for c in df.columns:
        if c.lower() in {"group", "pathology", "state", "label", "class"}:
            return c
    raise KeyError("Could not auto-detect pathology/group column in source_embedding_coordinates.csv")


def load_sources(input_dir: Path, group_col_arg: str) -> Tuple[pd.Series, pd.DataFrame]:
    coord_path = find_file(input_dir, "source_embedding_coordinates.csv")
    feat_path = find_file(input_dir, "source_true_pixel_robust_z_features.csv")

    coords = pd.read_csv(coord_path)
    feats = pd.read_csv(feat_path)
    group_col = detect_group_col(coords, group_col_arg)
    groups = coords[group_col].astype(str).map(canonical_group)
    keep = groups.notna().values
    groups = groups.loc[keep].reset_index(drop=True)
    feats = feats.loc[keep].reset_index(drop=True)

    numeric = feats.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    valid_cols = [
        c for c in numeric.columns
        if np.isfinite(numeric[c].values).sum() >= 20 and np.nanstd(numeric[c].values.astype(float)) > 1e-10
    ]
    numeric = numeric[valid_cols]
    if numeric.shape[1] == 0:
        raise RuntimeError("No valid numeric feature columns found.")
    return groups, numeric


def infer_modules(features: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=features.index)
    lowered = {c: clean_text(c).lower() for c in features.columns}
    for module_name, keywords in MODULE_KEYWORDS.items():
        matched = [c for c, lc in lowered.items() if all(k in lc for k in keywords)]
        if not matched:
            matched = [c for c, lc in lowered.items() if any(k in lc for k in keywords)]
        if matched:
            out[module_name] = features[matched].mean(axis=1, skipna=True)
    if out.shape[1] < 4:
        raise RuntimeError(
            "Could not infer enough module columns from the feature table. "
            "Please check that your feature columns include module-score columns."
        )
    return out


def rank_features(groups: pd.Series, table: pd.DataFrame, min_group_size: int) -> pd.DataFrame:
    rows = []
    for col in table.columns:
        arrays = []
        medians = {}
        ok = True
        for g in GROUP_ORDER:
            vals = pd.to_numeric(table.loc[groups == g, col], errors="coerce").dropna().values
            medians[g] = float(np.nanmedian(vals)) if len(vals) > 0 else np.nan
            if len(vals) < min_group_size:
                ok = False
                break
            arrays.append(vals)
        if not ok:
            continue
        try:
            h, p = kruskal(*arrays)
        except Exception:
            continue
        rows.append({
            "feature": col,
            "H": float(h),
            "p_value": float(p),
            "median_range": float(np.nanmax(list(medians.values())) - np.nanmin(list(medians.values()))),
            **{f"median_{g}": medians[g] for g in GROUP_ORDER},
        })
    stat = pd.DataFrame(rows)
    if stat.empty:
        raise RuntimeError("No feature passed the minimum group-size filter.")
    stat["fdr_q"] = bh_fdr(stat["p_value"].values)
    stat = stat.sort_values(["H", "median_range", "fdr_q"], ascending=[False, False, True]).reset_index(drop=True)
    return stat


def choose_features(groups: pd.Series, features: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    if args.mode == "module":
        table = infer_modules(features)
    else:
        table = features.copy()

    ranking = rank_features(groups, table, min_group_size=args.min_group_size)

    if args.single_feature:
        if args.single_feature not in table.columns:
            raise KeyError(f"Requested --single-feature '{args.single_feature}' not found. Available examples: {list(table.columns[:12])}")
        selected = [args.single_feature]
        ranking = ranking[ranking["feature"] == args.single_feature].copy()
        if ranking.empty:
            ranking = pd.DataFrame([{"feature": args.single_feature, "H": np.nan, "p_value": np.nan, "fdr_q": np.nan, "median_range": np.nan}])
    else:
        selected = ranking["feature"].head(args.top_n).tolist()
    return table, selected, ranking


def kde_on_grid(values: np.ndarray, x_grid: np.ndarray, bandwidth_adjust: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros_like(x_grid)
    if len(np.unique(values)) < 2:
        # tiny Gaussian bump around the constant value
        mu = float(np.nanmean(values))
        sigma = max(0.08, np.nanstd(values) + 0.05)
        y = np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
        return y / max(y.max(), 1e-12)

    kde = gaussian_kde(values)
    kde.set_bandwidth(kde.factor * bandwidth_adjust)
    y = kde.evaluate(x_grid)
    return y / max(np.nanmax(y), 1e-12)


def make_panel(ax: plt.Axes,
               feature_name: str,
               groups: pd.Series,
               table: pd.DataFrame,
               rng: np.random.Generator,
               args: argparse.Namespace,
               stat_row: pd.Series | None):
    data_by_group: Dict[str, np.ndarray] = {}
    pooled = []
    for g in GROUP_ORDER:
        vals = pd.to_numeric(table.loc[groups == g, feature_name], errors="coerce").dropna().values.astype(float)
        data_by_group[g] = vals
        if len(vals):
            pooled.append(vals)
    pooled_vals = np.concatenate(pooled) if pooled else np.array([0.0, 1.0])
    x_min, x_max = np.nanpercentile(pooled_vals, [1, 99])
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min, x_max = np.nanmin(pooled_vals), np.nanmax(pooled_vals)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min, x_max = -1.0, 1.0
    pad = (x_max - x_min) * args.x_pad_frac + 1e-6
    x_grid = np.linspace(x_min - pad, x_max + pad, args.kde_points)

    n_groups = len(GROUP_ORDER)
    baseline_positions = np.arange(n_groups, 0, -1)  # CA at top

    for idx, g in enumerate(GROUP_ORDER):
        y0 = baseline_positions[idx]
        vals = data_by_group[g]

        ax.hlines(y0, x_grid.min(), x_grid.max(), color="#D2D2D2", linewidth=0.8, zorder=1)

        if len(vals) == 0:
            continue

        dens = kde_on_grid(vals, x_grid, bandwidth_adjust=args.bandwidth_adjust)
        ridge = y0 + dens * args.ridge_height
        color = GROUP_COLORS[g]

        ax.fill_between(x_grid, y0, ridge, color=color, alpha=0.82, linewidth=0.0, zorder=2)
        ax.plot(x_grid, ridge, color=color, linewidth=1.5, zorder=3)

        # median marker
        med = float(np.nanmedian(vals))
        ax.scatter([med], [y0 + args.ridge_height * 0.52], s=18, color="white", edgecolor=color, linewidth=1.0, zorder=4)

        # sparse points along the baseline area, similar to publication ridge plots
        points = vals.copy()
        if len(points) > args.points_per_group:
            pick = rng.choice(len(points), size=args.points_per_group, replace=False)
            points = points[pick]
        y_jitter = y0 + rng.uniform(0.02, 0.16, size=len(points))
        ax.scatter(points, y_jitter, s=7, color=color, alpha=0.45, edgecolor="none", zorder=2.5)

    ax.set_yticks(baseline_positions)
    ax.set_yticklabels(GROUP_ORDER, fontsize=10, fontweight="bold")
    ax.set_ylim(0.5, n_groups + 1.0)
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_xlabel("Robust z-score", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.7)
    ax.set_axisbelow(True)

    pretty_name = feature_name.replace("_", " ")
    if stat_row is not None and pd.notna(stat_row.get("H", np.nan)):
        ax.set_title(f"{pretty_name}\nKruskal–Wallis p={stat_row['p_value']:.2e}", fontsize=11.5, fontweight="bold", pad=8)
    else:
        ax.set_title(pretty_name, fontsize=11.5, fontweight="bold", pad=8)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["bottom"].set_linewidth(0.8)


def make_figure(groups: pd.Series, table: pd.DataFrame, selected: List[str], ranking: pd.DataFrame, args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(selected)
    cols = max(1, args.cols)
    rows = math.ceil(n / cols)

    fig_w = 5.1 * cols
    fig_h = 3.3 * rows + 1.4
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.flatten()
    rng = np.random.default_rng(args.random_seed)

    rank_lookup = {r["feature"]: r for _, r in ranking.iterrows()}
    for i, feat in enumerate(selected):
        make_panel(axes_flat[i], feat, groups, table, rng, args, rank_lookup.get(feat, None))

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    handles = [Line2D([0], [0], color=GROUP_COLORS[g], lw=4, label=f"{g} ({GROUP_FULL[g]})") for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.02))

    fig.text(0.5, 0.985, args.title, ha="center", va="top", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.958, args.subtitle, ha="center", va="top", fontsize=10.5, color="#555555")

    plt.subplots_adjust(top=0.89, bottom=0.14, left=0.08, right=0.985, hspace=0.42, wspace=0.22)

    stem = f"Fig_ridgeplot_{args.mode}_{len(selected)}features"
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    out_svg = out_dir / f"{stem}.svg"
    fig.savefig(out_png, dpi=450, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    stat_out = out_dir / f"{stem}_ranking.csv"
    ranking.to_csv(stat_out, index=False, encoding="utf-8-sig")
    return out_png, out_pdf, out_svg, stat_out


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups, features = load_sources(input_dir, args.group_col)
    table, selected, ranking = choose_features(groups, features, args)

    if args.mode == "module":
        default_title = "Ridgeline distributions of Raman molecular programs across LUAD growth patterns"
        default_subtitle = "Real 3D Raman pixel-level module scores | stacked density distributions by pathology state"
    else:
        default_title = "Ridgeline distributions of top Raman markers across LUAD growth patterns"
        default_subtitle = "Real 3D Raman pixel-level marker values | stacked density distributions by pathology state"

    # overwrite default title only if user did not set it explicitly in CLI
    if args.title == "Ridgeline distributions of Raman features across LUAD growth patterns":
        args.title = default_title
    if args.subtitle == "Real 3D Raman pixel-level data | ridgeline density distributions across pathology groups":
        args.subtitle = default_subtitle

    out_png, out_pdf, out_svg, stat_out = make_figure(groups, table, selected, ranking, args)
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved SVG: {out_svg}")
    print(f"Saved ranking/stat table: {stat_out}")
    print(f"Selected features: {selected}")


if __name__ == "__main__":
    main()
