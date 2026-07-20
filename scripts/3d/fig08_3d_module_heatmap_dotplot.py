#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fig08: systemic reprogramming of Raman molecular modules.

Reference-style figure:
  A. Module-level heatmap
     Rows = broad Raman molecular modules
     Columns = balanced pixel observations grouped by pathology group
     Values = module activity z-score

  B. Module dot plot
     Dot color = average module activity
     Dot size  = fraction of pixels with positive module activity

Input folder:
  Fig01 v7 output folder containing:
    source_embedding_coordinates.csv
    source_true_pixel_robust_z_features.csv

This script does NOT need SHAP and does NOT need Fig03/Fig04 outputs.

Run:
cd "FV-RMH"

python ".\\scripts\\3d\\fig08_3d_module_heatmap_dotplot.py" ^
  --input-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_2000pixels" ^
  --out-dir "3dresults\\fig08_3d_module_heatmap_dotplot_2000pixels"

For 1500 pixels/layer:
python ".\\scripts\\3d\\fig08_3d_module_heatmap_dotplot.py" ^
  --input-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_1500pixels" ^
  --out-dir "3dresults\\fig08_3d_module_heatmap_dotplot_1500pixels"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist

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

MODULE_ORDER = [
    "Central metabolism",
    "Lipid metabolism",
    "ECM/stromal remodeling",
    "Immune/protein",
    "Stress/glycoxidation",
    "Amino acid/nitrogen",
    "Other",
]
MODULE_LABELS = {
    "Central metabolism": "Central metabolism",
    "Lipid metabolism": "Lipid metabolism",
    "ECM/stromal remodeling": "ECM / stromal remodeling",
    "Immune/protein": "Immune / protein",
    "Stress/glycoxidation": "Stress / glycoxidation",
    "Amino acid/nitrogen": "Amino acid / nitrogen",
    "Other": "Other",
}


def normalize_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def infer_marker_module(marker: str) -> str:
    key = normalize_name(marker)
    central = [
        "lactate", "pyruvate", "phosphoenolpyruvic", "3phosphoglycerate",
        "phosphoglycerate", "glucose", "glycol", "citrate", "fumarate",
        "malate", "succinate", "acetylcoa", "coa", "atp", "nad", "fadh",
        "tca", "oxaloacetate", "mitochond"
    ]
    lipid = [
        "lipid", "cholesterol", "cholesteryl", "cholesterolester",
        "fattyacid", "palmit", "oleic", "linole", "triglyceride",
        "phospholipid", "sphingo", "ceramide", "monounsaturated",
        "saturatedlipid", "unsaturated", "sphinosine", "sphingosine"
    ]
    ecm = [
        "collagen", "elastin", "fibronectin", "laminin", "vitronectin",
        "tenascin", "versican", "syndecan", "matrix", "ecm", "acta2",
        "actin", "vimentin", "cathepsin", "mmp", "integrin", "hyaluron",
        "periostin", "thrombospondin", "spp1"
    ]
    immune = [
        "pd1", "pdl1", "b7h3", "sting", "cd", "hla", "mhc",
        "immun", "cytokine", "interferon", "tnf", "il", "protein",
        "histone", "albumin", "keratin", "cytokeratin", "ck"
    ]
    stress = [
        "glycoxidation", "glycation", "oxid", "ros", "glutathione",
        "slactoylglutathione", "lactoyl", "gsh", "gssg", "carbonyl",
        "mda", "4hne", "stress", "hypoxia", "nrf2", "hif"
    ]
    amino = [
        "alanine", "arginine", "asparagine", "aspartate", "cysteine",
        "glutamate", "glutamine", "glycine", "histidine", "isoleucine",
        "leucine", "lysine", "methionine", "phenylalanine", "proline",
        "serine", "threonine", "tryptophan", "tyrosine", "valine",
        "amino", "urea", "nitrogen", "creatine", "taurine"
    ]
    if any(k in key for k in central):
        return "Central metabolism"
    if any(k in key for k in lipid):
        return "Lipid metabolism"
    if any(k in key for k in ecm):
        return "ECM/stromal remodeling"
    if any(k in key for k in immune):
        return "Immune/protein"
    if any(k in key for k in stress):
        return "Stress/glycoxidation"
    if any(k in key for k in amino):
        return "Amino acid/nitrogen"
    return "Other"


def load_inputs(input_dir: Path):
    emb_path = input_dir / "source_embedding_coordinates.csv"
    feat_path = input_dir / "source_true_pixel_robust_z_features.csv"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}")

    coords = pd.read_csv(emb_path)
    features = pd.read_csv(feat_path)

    if len(coords) != len(features):
        raise RuntimeError(f"Embedding rows ({len(coords)}) and feature rows ({len(features)}) do not match.")

    keep = coords["group"].isin(GROUP_ORDER).values
    coords = coords.loc[keep].reset_index(drop=True)
    features = features.loc[keep].reset_index(drop=True)

    valid = [c for c in features.columns if np.nanstd(features[c].values.astype(float)) > 1e-10]
    features = features[valid].replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)

    if features.shape[1] < 3:
        raise RuntimeError("Too few valid Raman marker features.")
    return coords, features


def build_module_scores(features: pd.DataFrame):
    marker_module = pd.DataFrame({
        "marker": features.columns,
        "module": [infer_marker_module(c) for c in features.columns],
    })

    raw = pd.DataFrame(index=features.index)
    for module in MODULE_ORDER:
        cols = marker_module.loc[marker_module["module"] == module, "marker"].tolist()
        cols = [c for c in cols if c in features.columns]
        if cols:
            raw[module] = features[cols].mean(axis=1)
        else:
            raw[module] = np.nan

    # Fill any missing module score with 0 for visualization.
    raw = raw.fillna(0.0)

    # Robust z-score per module for heatmap.
    z = pd.DataFrame(index=raw.index)
    for module in MODULE_ORDER:
        vals = raw[module].values.astype(float)
        med = np.nanmedian(vals)
        q1, q3 = np.nanpercentile(vals, [25, 75])
        scale = q3 - q1
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.nanstd(vals)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        z[module] = np.clip((vals - med) / scale, -2.5, 2.5)

    return marker_module, raw, z


def stratified_sample(coords: pd.DataFrame, n_per_group: int, random_state=1):
    rng = np.random.default_rng(random_state)
    selected = []
    for group in GROUP_ORDER:
        idx = np.where(coords["group"].astype(str).values == group)[0]
        if idx.size == 0:
            continue
        n_take = min(int(n_per_group), idx.size)
        selected.extend(rng.choice(idx, size=n_take, replace=False).tolist())
    return np.asarray(selected, dtype=int)


def cluster_module_order(mat: np.ndarray):
    # mat: modules x columns
    if mat.shape[0] <= 2:
        return np.arange(mat.shape[0])
    try:
        Z = linkage(pdist(mat, metric="euclidean"), method="ward")
        return leaves_list(Z)
    except Exception:
        return np.arange(mat.shape[0])


def build_heatmap_data(coords, z_scores, n_per_group, random_state):
    idx = stratified_sample(coords, n_per_group=n_per_group, random_state=random_state)
    meta = coords.iloc[idx].copy().reset_index(drop=True)
    values = z_scores.iloc[idx].copy().reset_index(drop=True)

    meta["group"] = pd.Categorical(meta["group"], categories=GROUP_ORDER, ordered=True)
    if "layer" not in meta.columns:
        meta["layer"] = 0
    order = meta.sort_values(["group", "layer", "umap_1"]).index.values
    meta = meta.iloc[order].reset_index(drop=True)
    values = values.iloc[order].reset_index(drop=True)

    mat = values[MODULE_ORDER].T.values.astype(float)
    order_rows = cluster_module_order(mat)
    modules_ordered = [MODULE_ORDER[i] for i in order_rows]
    mat = mat[order_rows, :]
    return mat, modules_ordered, meta


def dot_summary(coords, raw_scores, modules):
    rows = []
    for module in modules:
        for group in GROUP_ORDER:
            idx = coords["group"].astype(str).values == group
            vals = raw_scores.loc[idx, module].values.astype(float)
            rows.append({
                "module": module,
                "group": group,
                "average_activity": float(np.nanmean(vals)),
                "median_activity": float(np.nanmedian(vals)),
                "fraction_positive": float(np.mean(vals > 0)),
                "n_pixels": int(np.sum(idx)),
            })
    out = pd.DataFrame(rows)
    # Scale color range to robust z-like range for dotplot.
    vals = out["average_activity"].values.astype(float)
    lo, hi = np.nanpercentile(vals, [2, 98])
    if np.isfinite(hi - lo) and hi > lo:
        out["average_activity_scaled"] = np.clip((vals - np.nanmedian(vals)) / (hi - lo) * 2.5, -1.5, 1.5)
    else:
        out["average_activity_scaled"] = 0.0
    return out


def save_multi(fig, out_dir: Path, stem: str, dpi=600, pad=0.02):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=pad)


def draw_group_bar(ax, meta):
    groups = meta["group"].astype(str).values
    colors = [GROUP_COLORS[g] for g in groups]
    rgb = np.array([plt.matplotlib.colors.to_rgb(c) for c in colors]).reshape(1, len(colors), 3)
    ax.imshow(rgb, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


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
            above_threshold_color="#666666",
            link_color_func=lambda k: "#666666",
            ax=ax,
        )
        ax.invert_yaxis()
    except Exception:
        pass
    ax.axis("off")


def draw_heatmap(ax, mat, modules, show_labels=True):
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-2.2, vmax=2.2, interpolation="nearest")
    ax.set_xticks([])
    if show_labels:
        ax.set_yticks(np.arange(len(modules)))
        ax.set_yticklabels([MODULE_LABELS[m] for m in modules], fontsize=8, fontweight="bold")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", length=0, pad=5)
    else:
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def draw_group_boundaries(ax, meta):
    groups = meta["group"].astype(str).values
    last = groups[0]
    for i, g in enumerate(groups):
        if g != last:
            ax.axvline(i - 0.5, color="black", lw=0.55, alpha=0.75)
            last = g


def draw_group_labels(ax, meta):
    groups = meta["group"].astype(str).values
    for group in GROUP_ORDER:
        idx = np.where(groups == group)[0]
        if idx.size:
            center = (idx.min() + idx.max()) / 2
            ax.text(center, -0.55, GROUP_SHORT[group], ha="center", va="center",
                    fontsize=8, fontweight="bold", color=GROUP_COLORS[group])


def draw_dotplot(ax, dot, modules):
    x_lookup = {g: i for i, g in enumerate(GROUP_ORDER)}
    y_lookup = {m: i for i, m in enumerate(modules)}
    xs, ys, colors, sizes = [], [], [], []
    for _, row in dot.iterrows():
        module = row["module"]
        if module not in y_lookup:
            continue
        xs.append(x_lookup[str(row["group"])])
        ys.append(y_lookup[module])
        colors.append(float(row["average_activity_scaled"]))
        sizes.append(20 + 150 * float(row["fraction_positive"]))

    sc = ax.scatter(xs, ys, c=colors, s=sizes, cmap="coolwarm", vmin=-1.5, vmax=1.5,
                    edgecolor="#555555", linewidths=0.35, alpha=0.90)
    ax.set_xlim(-0.6, len(GROUP_ORDER) - 0.4)
    ax.set_ylim(len(modules) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_SHORT[g] for g in GROUP_ORDER], fontsize=8, rotation=0)
    ax.set_yticks(np.arange(len(modules)))
    ax.set_yticklabels([MODULE_LABELS[m] for m in modules], fontsize=8, fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="x", color="#EEEEEE", lw=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return sc


def add_size_legend(ax):
    sizes = [0.25, 0.50, 0.75, 1.00]
    handles = [
        ax.scatter([], [], s=20 + 150 * s, color="#333333", edgecolor="#555555", linewidths=0.35)
        for s in sizes
    ]
    labels = [f"{int(s*100)}%" for s in sizes]
    leg = ax.legend(
        handles, labels, title="Fraction\npositive",
        frameon=False, fontsize=7, title_fontsize=7,
        loc="center left", bbox_to_anchor=(1.03, 0.28), borderaxespad=0
    )
    ax.add_artist(leg)


def make_main_figure(mat, modules, meta, dot, out_dir, with_text=True):
    fig = plt.figure(figsize=(12.5, 8.1), constrained_layout=True)
    gs = fig.add_gridspec(
        4, 4,
        width_ratios=[0.33, 3.65, 0.24, 2.10],
        height_ratios=[0.12, 2.55, 0.18, 2.10],
        wspace=0.04, hspace=0.05,
    )

    ax_bar = fig.add_subplot(gs[0, 1])
    draw_group_bar(ax_bar, meta)

    ax_den = fig.add_subplot(gs[1, 0])
    draw_dendrogram(ax_den, mat)

    ax_heat = fig.add_subplot(gs[1, 1])
    im = draw_heatmap(ax_heat, mat, modules, show_labels=True)
    draw_group_boundaries(ax_heat, meta)

    ax_gap = fig.add_subplot(gs[1:, 2])
    ax_gap.axis("off")

    ax_dot = fig.add_subplot(gs[3, 1])
    sc = draw_dotplot(ax_dot, dot, modules)

    ax_legend = fig.add_subplot(gs[3, 3])
    ax_legend.axis("off")

    if with_text:
        draw_group_labels(ax_bar, meta)
        ax_heat.set_title("A. Module-level Raman activity heatmap", loc="left", fontsize=11.5, fontweight="bold", pad=8)
        ax_dot.set_title("B. Systemic reprogramming of molecular modules", loc="left", fontsize=11.5, fontweight="bold", pad=8)

        cbar1 = fig.colorbar(im, ax=ax_heat, fraction=0.022, pad=0.015)
        cbar1.set_label("Z-score", fontsize=8)
        cbar1.ax.tick_params(labelsize=7)

        cbar2 = fig.colorbar(sc, ax=ax_legend, fraction=0.35, pad=0.01)
        cbar2.set_label("Average activity", fontsize=8)
        cbar2.ax.tick_params(labelsize=7)

        add_size_legend(ax_dot)

        fig.suptitle(
            "Systemic enrichment and depletion of Raman molecular modules across LUAD growth patterns",
            fontsize=14, fontweight="bold", y=1.02,
        )

    stem = "Fig08_3D_module_heatmap_dotplot" + ("" if with_text else "_no_text")
    save_multi(fig, out_dir, stem)
    plt.close(fig)


def export_single_panels(mat, modules, meta, dot, out_dir):
    for mode in ["with_text", "no_text"]:
        wt = mode == "with_text"
        folder = out_dir / f"single_panels_{mode}"
        folder.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(8.3, 4.6), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[0.35, 4.0], height_ratios=[0.12, 3.0])
        ax_bar = fig.add_subplot(gs[0, 1])
        draw_group_bar(ax_bar, meta)
        ax_den = fig.add_subplot(gs[1, 0])
        draw_dendrogram(ax_den, mat)
        ax_heat = fig.add_subplot(gs[1, 1])
        im = draw_heatmap(ax_heat, mat, modules, show_labels=wt)
        draw_group_boundaries(ax_heat, meta)
        if wt:
            draw_group_labels(ax_bar, meta)
            ax_heat.set_title("Module-level Raman activity heatmap", loc="left", fontsize=11.5, fontweight="bold", pad=8)
            fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.015).set_label("Z-score", fontsize=8)
        save_multi(fig, folder, f"panel_A_module_heatmap_{mode}")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.8, 4.6), constrained_layout=True)
        sc = draw_dotplot(ax, dot, modules)
        if wt:
            ax.set_title("Systemic reprogramming of molecular modules", loc="left", fontsize=11.5, fontweight="bold", pad=8)
            add_size_legend(ax)
            fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label("Average activity", fontsize=8)
        save_multi(fig, folder, f"panel_B_module_dotplot_{mode}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Module heatmap and dotplot for 3D Raman data.")
    parser.add_argument("--input-dir", required=True, help="Fig01 v7 output folder.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--columns-per-group", type=int, default=180)
    parser.add_argument("--random-state", type=int, default=1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords, features = load_inputs(input_dir)
    marker_module, raw_scores, z_scores = build_module_scores(features)
    mat, modules, meta = build_heatmap_data(coords, z_scores, args.columns_per_group, args.random_state)
    dot = dot_summary(coords, raw_scores, modules)

    marker_module.to_csv(out_dir / "source_marker_module_assignment.csv", index=False)
    raw_scores.to_csv(out_dir / "source_pixel_module_scores_raw.csv", index=False)
    z_scores.to_csv(out_dir / "source_pixel_module_scores_z.csv", index=False)
    pd.DataFrame(mat, index=modules).to_csv(out_dir / "source_module_heatmap_matrix_modules_by_pixels.csv")
    meta.to_csv(out_dir / "source_module_heatmap_sampled_pixel_metadata.csv", index=False)
    dot.to_csv(out_dir / "source_module_dotplot_summary.csv", index=False)

    make_main_figure(mat, modules, meta, dot, out_dir, with_text=True)
    make_main_figure(mat, modules, meta, dot, out_dir, with_text=False)
    export_single_panels(mat, modules, meta, dot, out_dir)

    print("Done.")
    print(f"Output folder: {out_dir}")
    print(f"Modules shown: {len(modules)}")
    print(f"Heatmap columns: {mat.shape[1]}")
    print("Main figure: Fig08_3D_module_heatmap_dotplot.png")


if __name__ == "__main__":
    main()
