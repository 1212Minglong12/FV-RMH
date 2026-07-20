#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fig07: global Raman marker heatmap + dot plot summary.

Reference-style figure:
  A. Clustered marker-by-sampled-pixel heatmap
     Columns are pixel observations grouped by pathology stage.
     Rows are Raman markers clustered by Euclidean distance.
     Values are robust z-score marker activities.

  B. Dot plot summary
     Dot color = average marker activity in each pathology group
     Dot size  = fraction of pixels with positive marker activity

Input folder:
  Fig01 v7 output folder containing:
    source_embedding_coordinates.csv
    source_true_pixel_robust_z_features.csv

Run:
cd "FV-RMH"

python ".\\scripts\\3d\\fig07_3d_marker_heatmap_dotplot.py" ^
  --input-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_2000pixels" ^
  --out-dir "3dresults\\fig07_3d_marker_heatmap_dotplot_2000pixels"

For 1500-pixels-per-layer:
python ".\\scripts\\3d\\fig07_3d_marker_heatmap_dotplot.py" ^
  --input-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_1500pixels" ^
  --out-dir "3dresults\\fig07_3d_marker_heatmap_dotplot_1500pixels"

Options:
  --top-n 43                 Number of markers shown. Default uses all available markers unless >80.
  --columns-per-group 160    Number of sampled pixel columns per group in heatmap.
  --dot-top-n 43             Number of markers shown in dot plot.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

GROUP_ORDER = [
    "Cancer-adjacent",
    "Lepidic",
    "Acinar",
    "Papillary",
    "Micropapillary",
    "Complex glands",
    "Solid",
]
GROUP_SHORT = {
    "Cancer-adjacent": "CA",
    "Lepidic": "LEP",
    "Acinar": "ACN",
    "Papillary": "PAP",
    "Micropapillary": "MP",
    "Complex glands": "CGP",
    "Solid": "SOL",
}
GROUP_COLORS = {
    "Cancer-adjacent": "#7A7A7A",
    "Lepidic": "#4C78A8",
    "Acinar": "#54A24B",
    "Papillary": "#F58518",
    "Micropapillary": "#E45756",
    "Complex glands": "#B279A2",
    "Solid": "#C92D39",
}


def clean_marker_label(x: str) -> str:
    x = str(x).replace("_", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def load_inputs(input_dir: Path):
    emb_path = input_dir / "source_embedding_coordinates.csv"
    feature_path = input_dir / "source_true_pixel_robust_z_features.csv"

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing {feature_path}")

    coords = pd.read_csv(emb_path)
    features = pd.read_csv(feature_path)

    if len(coords) != len(features):
        raise RuntimeError(
            f"Embedding rows ({len(coords)}) and feature rows ({len(features)}) do not match."
        )

    keep = coords["group"].isin(GROUP_ORDER).values
    coords = coords.loc[keep].reset_index(drop=True)
    features = features.loc[keep].reset_index(drop=True)

    valid_cols = [
        c for c in features.columns
        if np.nanstd(features[c].values.astype(float)) > 1e-10
    ]
    features = features[valid_cols].replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)

    return coords, features


def marker_ranking(features: pd.DataFrame, coords: pd.DataFrame):
    """
    Rank markers by cross-group variance of group median activities.
    """
    rows = []
    for marker in features.columns:
        group_medians = []
        for group in GROUP_ORDER:
            idx = coords["group"].astype(str).values == group
            group_medians.append(np.nanmedian(features.loc[idx, marker].values.astype(float)))
        group_medians = np.asarray(group_medians, dtype=float)
        rows.append({
            "marker": marker,
            "between_group_median_sd": float(np.nanstd(group_medians)),
            "max_group_median": float(np.nanmax(group_medians)),
            "min_group_median": float(np.nanmin(group_medians)),
            "range_group_median": float(np.nanmax(group_medians) - np.nanmin(group_medians)),
            "top_group": GROUP_ORDER[int(np.nanargmax(group_medians))],
            "bottom_group": GROUP_ORDER[int(np.nanargmin(group_medians))],
        })
    return pd.DataFrame(rows).sort_values(
        ["between_group_median_sd", "range_group_median"],
        ascending=False
    ).reset_index(drop=True)


def stratified_column_sample(coords: pd.DataFrame, n_per_group: int, random_state=1):
    rng = np.random.default_rng(random_state)
    selected = []
    for group in GROUP_ORDER:
        idx = np.where(coords["group"].astype(str).values == group)[0]
        if len(idx) == 0:
            continue
        n_take = min(int(n_per_group), len(idx))
        selected.extend(rng.choice(idx, size=n_take, replace=False).tolist())
    return np.asarray(selected, dtype=int)


def row_cluster_order(matrix: np.ndarray):
    """
    matrix: markers x samples
    """
    if matrix.shape[0] <= 2:
        return np.arange(matrix.shape[0])
    try:
        dist = pdist(matrix, metric="euclidean")
        Z = linkage(dist, method="ward")
        return leaves_list(Z)
    except Exception:
        return np.arange(matrix.shape[0])


def build_heatmap_matrix(features, coords, markers, columns_per_group, random_state):
    selected_cols = stratified_column_sample(coords, columns_per_group, random_state=random_state)
    sampled_features = features.iloc[selected_cols][markers].copy()
    sampled_meta = coords.iloc[selected_cols].copy().reset_index(drop=True)

    # Sort columns by group, then by layer if present, then UMAP1.
    sampled_meta["group"] = pd.Categorical(sampled_meta["group"], categories=GROUP_ORDER, ordered=True)
    if "layer" not in sampled_meta.columns:
        sampled_meta["layer"] = 0
    order = sampled_meta.sort_values(["group", "layer", "umap_1"]).index.values
    sampled_meta = sampled_meta.iloc[order].reset_index(drop=True)
    sampled_features = sampled_features.iloc[order].reset_index(drop=True)

    # Matrix rows are markers, columns are sampled pixels.
    mat = sampled_features.T.values.astype(float)

    # Clip for display only.
    mat = np.clip(mat, -2.5, 2.5)

    order_rows = row_cluster_order(mat)
    ordered_markers = [markers[i] for i in order_rows]
    mat = mat[order_rows, :]

    return mat, ordered_markers, sampled_meta


def compute_dot_summary(features, coords, markers):
    rows = []
    for marker in markers:
        for group in GROUP_ORDER:
            idx = coords["group"].astype(str).values == group
            vals = features.loc[idx, marker].values.astype(float)
            rows.append({
                "marker": marker,
                "group": group,
                "mean_activity": float(np.nanmean(vals)),
                "median_activity": float(np.nanmedian(vals)),
                "fraction_positive": float(np.mean(vals > 0)),
                "n_pixels": int(np.sum(idx)),
            })
    dot = pd.DataFrame(rows)
    dot["group"] = pd.Categorical(dot["group"], categories=GROUP_ORDER, ordered=True)
    return dot


def save_multi(fig, out_dir: Path, stem: str, dpi=600, pad=0.02):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=pad)


def draw_group_color_bar(ax, sampled_meta):
    group_vals = sampled_meta["group"].astype(str).values
    color_lookup = {g: GROUP_COLORS[g] for g in GROUP_ORDER}
    colors = [color_lookup[g] for g in group_vals]
    rgb = np.array([plt.matplotlib.colors.to_rgb(c) for c in colors]).reshape(1, len(colors), 3)
    ax.imshow(rgb, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_heatmap(ax, mat, markers, show_y_labels=True):
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_xticks([])
    if show_y_labels:
        ax.set_yticks(np.arange(len(markers)))
        ax.set_yticklabels([clean_marker_label(m) for m in markers], fontsize=5.5)
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", length=0, pad=2)
    else:
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def draw_group_boundaries(ax, sampled_meta):
    groups = sampled_meta["group"].astype(str).values
    starts = []
    last = None
    for i, g in enumerate(groups):
        if g != last:
            starts.append((i, g))
            last = g
    for pos, _ in starts[1:]:
        ax.axvline(pos - 0.5, color="black", lw=0.35, alpha=0.6)


def draw_dendrogram(ax, mat):
    if mat.shape[0] <= 2:
        ax.axis("off")
        return
    try:
        Z = linkage(pdist(mat, metric="euclidean"), method="ward")
        dendrogram(
            Z,
            orientation="left",
            no_labels=True,
            color_threshold=0,
            above_threshold_color="#6A6A6A",
            link_color_func=lambda k: "#6A6A6A",
            ax=ax,
        )
        ax.invert_yaxis()
        ax.axis("off")
    except Exception:
        ax.axis("off")


def draw_dotplot(ax, dot, markers):
    # marker order top to bottom as heatmap order.
    y_lookup = {m: i for i, m in enumerate(markers)}
    x_lookup = {g: i for i, g in enumerate(GROUP_ORDER)}

    xs, ys, colors, sizes = [], [], [], []
    for _, row in dot.iterrows():
        marker = row["marker"]
        if marker not in y_lookup:
            continue
        xs.append(x_lookup[str(row["group"])])
        ys.append(y_lookup[marker])
        colors.append(float(row["mean_activity"]))
        sizes.append(18 + 90 * float(row["fraction_positive"]))

    sc = ax.scatter(
        xs, ys,
        c=colors,
        s=sizes,
        cmap="coolwarm",
        vmin=-1.2,
        vmax=1.2,
        linewidths=0.25,
        edgecolor="#666666",
        alpha=0.88,
    )
    ax.set_xlim(-0.6, len(GROUP_ORDER) - 0.4)
    ax.set_ylim(len(markers) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_SHORT[g] for g in GROUP_ORDER], fontsize=8, rotation=0)
    ax.set_yticks(np.arange(len(markers)))
    ax.set_yticklabels([clean_marker_label(m) for m in markers], fontsize=5.5)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="x", color="#EEEEEE", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return sc


def add_dot_size_legend(ax):
    sizes = [0.25, 0.50, 0.75, 1.00]
    handles = [
        ax.scatter([], [], s=18 + 90 * s, color="#777777", alpha=0.75, edgecolor="#666666", linewidths=0.25)
        for s in sizes
    ]
    labels = [f"{int(s*100)}%" for s in sizes]
    leg = ax.legend(
        handles,
        labels,
        title="Fraction\npositive",
        frameon=False,
        fontsize=7,
        title_fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.03, 0.92),
        borderaxespad=0,
    )
    ax.add_artist(leg)


def make_main_figure(mat, markers, sampled_meta, dot, out_dir, with_text=True):
    # Wider main heatmap + right dotplot, reference style.
    fig = plt.figure(figsize=(14.5, 8.3), constrained_layout=True)
    gs = fig.add_gridspec(
        3, 4,
        width_ratios=[0.35, 3.9, 0.18, 2.05],
        height_ratios=[0.10, 4.8, 0.18],
        wspace=0.03,
        hspace=0.02,
    )

    ax_bar = fig.add_subplot(gs[0, 1])
    draw_group_color_bar(ax_bar, sampled_meta)

    ax_den = fig.add_subplot(gs[1, 0])
    draw_dendrogram(ax_den, mat)

    ax_heat = fig.add_subplot(gs[1, 1])
    im = draw_heatmap(ax_heat, mat, markers, show_y_labels=True)
    draw_group_boundaries(ax_heat, sampled_meta)

    ax_gap = fig.add_subplot(gs[1, 2])
    ax_gap.axis("off")

    ax_dot = fig.add_subplot(gs[1, 3])
    sc = draw_dotplot(ax_dot, dot, markers)
    add_dot_size_legend(ax_dot)

    if with_text:
        # Group labels above color bar.
        group_values = sampled_meta["group"].astype(str).values
        for group in GROUP_ORDER:
            idx = np.where(group_values == group)[0]
            if len(idx):
                center = (idx.min() + idx.max()) / 2
                ax_bar.text(
                    center, -0.7, GROUP_SHORT[group],
                    ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color=GROUP_COLORS[group],
                )

        ax_heat.set_title("A. Global Raman marker activity heatmap", loc="left", fontsize=11.5, fontweight="bold", pad=8)
        ax_dot.set_title("B. Dynamic shifts of Raman markers", loc="left", fontsize=11.5, fontweight="bold", pad=8)
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.018, pad=0.01)
        cbar.set_label("Z-score", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        cbar2 = fig.colorbar(sc, ax=ax_dot, fraction=0.035, pad=0.02)
        cbar2.set_label("Average activity", fontsize=8)
        cbar2.ax.tick_params(labelsize=7)
        fig.suptitle(
            "Data overview and Raman feature validation across 3D LUAD growth patterns",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
    else:
        ax_heat.set_title("")
        ax_dot.set_title("")

    stem = "Fig07_3D_marker_heatmap_dotplot" + ("" if with_text else "_no_text")
    save_multi(fig, out_dir, stem)
    plt.close(fig)


def export_single_panels(mat, markers, sampled_meta, dot, out_dir):
    for mode in ["with_text", "no_text"]:
        with_text = mode == "with_text"
        folder = out_dir / f"single_panels_{mode}"
        folder.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(8.8, 8.2), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, width_ratios=[0.35, 4.0], height_ratios=[0.10, 4.8, 0.18])
        ax_bar = fig.add_subplot(gs[0, 1])
        draw_group_color_bar(ax_bar, sampled_meta)
        ax_den = fig.add_subplot(gs[1, 0])
        draw_dendrogram(ax_den, mat)
        ax_heat = fig.add_subplot(gs[1, 1])
        im = draw_heatmap(ax_heat, mat, markers, show_y_labels=with_text)
        draw_group_boundaries(ax_heat, sampled_meta)
        if with_text:
            ax_heat.set_title("Global Raman marker activity heatmap", loc="left", fontsize=11.5, fontweight="bold", pad=8)
            fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.012).set_label("Z-score", fontsize=8)
        save_multi(fig, folder, f"panel_A_heatmap_{mode}")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 8.2), constrained_layout=True)
        sc = draw_dotplot(ax, dot, markers)
        if with_text:
            ax.set_title("Dynamic shifts of Raman markers", loc="left", fontsize=11.5, fontweight="bold", pad=8)
            add_dot_size_legend(ax)
            fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label("Average activity", fontsize=8)
        save_multi(fig, folder, f"panel_B_dotplot_{mode}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="3D Raman marker heatmap and dot plot.")
    parser.add_argument("--input-dir", required=True, help="Fig01 v7 result folder.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=0, help="0 = all markers if <=80, otherwise top 80.")
    parser.add_argument("--dot-top-n", type=int, default=0, help="0 = same as heatmap marker set.")
    parser.add_argument("--columns-per-group", type=int, default=160)
    parser.add_argument("--random-state", type=int, default=1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords, features = load_inputs(input_dir)
    ranking = marker_ranking(features, coords)

    if args.top_n and args.top_n > 0:
        n_heat = min(args.top_n, len(ranking))
    else:
        n_heat = len(ranking) if len(ranking) <= 80 else 80
    markers = ranking.head(n_heat)["marker"].tolist()

    mat, ordered_markers, sampled_meta = build_heatmap_matrix(
        features,
        coords,
        markers,
        columns_per_group=args.columns_per_group,
        random_state=args.random_state,
    )

    if args.dot_top_n and args.dot_top_n > 0:
        dot_markers = ranking.head(min(args.dot_top_n, len(ranking)))["marker"].tolist()
        # Keep heatmap order where possible, then add extra ranked markers.
        dot_markers = [m for m in ordered_markers if m in dot_markers] + [m for m in dot_markers if m not in ordered_markers]
    else:
        dot_markers = ordered_markers

    dot = compute_dot_summary(features, coords, dot_markers)

    ranking.to_csv(out_dir / "source_marker_dynamic_ranking.csv", index=False)
    sampled_meta.to_csv(out_dir / "source_heatmap_sampled_pixel_metadata.csv", index=False)
    pd.DataFrame(mat, index=ordered_markers).to_csv(out_dir / "source_heatmap_matrix_markers_by_pixels.csv")
    dot.to_csv(out_dir / "source_dotplot_marker_group_summary.csv", index=False)

    make_main_figure(mat, dot_markers if dot_markers == ordered_markers else ordered_markers, sampled_meta, dot, out_dir, with_text=True)
    make_main_figure(mat, ordered_markers, sampled_meta, dot, out_dir, with_text=False)
    export_single_panels(mat, ordered_markers, sampled_meta, dot, out_dir)

    print("Done.")
    print(f"Output folder: {out_dir}")
    print(f"Markers shown: {len(ordered_markers)}")
    print(f"Heatmap columns: {mat.shape[1]}")
    print("Main figure: Fig07_3D_marker_heatmap_dotplot.png")


if __name__ == "__main__":
    main()
