#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fig03: dominant metabolic/module flow projection on 3D Raman UMAP space.

This script draws a reference-style "dominant metabolic module" UMAP figure.

Input
-----
Use the output folder from:
    fig01_3d_pca_umap_clean_twopanel_v7.py

Required files in --input-dir:
    source_embedding_coordinates.csv
    source_true_pixel_robust_z_features.csv
    source_common_marker_list.csv  (optional but recommended)

What it does
------------
1. Reads UMAP coordinates and pixel-level Raman marker features.
2. Assigns each marker to a biological/metabolic module using a built-in marker-module dictionary.
3. For each pixel, calculates module scores from the marker z-score matrix.
4. Assigns the dominant module to each pixel.
5. Produces:
   a. UMAP panels faceted by pathology group, colored by dominant module.
   b. A module composition stacked bar chart.
   c. Optional no-text panels for later figure assembly.

Recommended command
-------------------
cd "FV-RMH"

python ".\\scripts\\3d\\fig03_3d_dominant_module_umap_flow.py" ^
  --input-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_1500pixels" ^
  --out-dir "3dresults\\fig03_3d_dominant_module_umap_flow_1500pixels"

For 2000-pixels-per-layer output:
python ".\\scripts\\3d\\fig03_3d_dominant_module_umap_flow.py" ^
  --input-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_2000pixels" ^
  --out-dir "3dresults\\fig03_3d_dominant_module_umap_flow_2000pixels"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
GROUP_TITLES = {
    "Cancer-adjacent": "Cancer-adjacent",
    "Lepidic": "Lepidic",
    "Acinar": "Acinar",
    "Papillary": "Papillary",
    "Micropapillary": "Micropapillary",
    "Complex glands": "Complex glands",
    "Solid": "Solid",
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
MODULE_COLORS = {
    "Central metabolism": "#F58518",
    "Lipid metabolism": "#4C78A8",
    "ECM/stromal remodeling": "#E45756",
    "Immune/protein": "#72B7B2",
    "Stress/glycoxidation": "#B279A2",
    "Amino acid/nitrogen": "#54A24B",
    "Other": "#9D9487",
}


# ---------------------------------------------------------------------
# Marker module mapping
# ---------------------------------------------------------------------
def normalize_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def infer_marker_module(marker: str) -> str:
    """
    Rule-based mapping from Raman marker names to broad biological modules.

    This is intentionally conservative. Unknown markers are assigned to Other.
    You can edit these keyword lists if a marker should be moved to another module.
    """
    key = normalize_name(marker)

    central_keywords = [
        "lactate", "pyruvate", "phosphoenolpyruvic", "3phosphoglycerate",
        "phosphoglycerate", "glucose", "glycol", "citrate", "fumarate",
        "malate", "succinate", "acetylcoa", "coa", "atp", "nad", "fadh",
        "tca", "oxaloacetate", "mitochond"
    ]
    lipid_keywords = [
        "lipid", "cholesterol", "cholesteryl", "cholesterolester",
        "fattyacid", "palmit", "oleic", "linole", "triglyceride",
        "phospholipid", "sphingo", "ceramide", "monounsaturated",
        "saturatedlipid", "unsaturated"
    ]
    ecm_keywords = [
        "collagen", "elastin", "fibronectin", "laminin", "vitronectin",
        "tenascin", "versican", "syndecan", "matrix", "ecm", "acta2",
        "actin", "vimentin", "cathepsin", "mmp", "integrin", "hyaluron",
        "periostin", "thrombospondin"
    ]
    immune_keywords = [
        "pd1", "pdl1", "b7h3", "sting", "cd", "hla", "mhc",
        "immun", "cytokine", "interferon", "tnf", "il", "protein",
        "histone", "albumin", "keratin", "cytokeratin", "ck"
    ]
    stress_keywords = [
        "glycoxidation", "glycation", "oxid", "ros", "glutathione",
        "slactoylglutathione", "lactoyl", "gsh", "gssg", "carbonyl",
        "mda", "4hne", "stress", "hypoxia", "nrf2", "hif"
    ]
    amino_keywords = [
        "alanine", "arginine", "asparagine", "aspartate", "cysteine",
        "glutamate", "glutamine", "glycine", "histidine", "isoleucine",
        "leucine", "lysine", "methionine", "phenylalanine", "proline",
        "serine", "threonine", "tryptophan", "tyrosine", "valine",
        "amino", "urea", "nitrogen", "creatine", "taurine"
    ]

    if any(k in key for k in central_keywords):
        return "Central metabolism"
    if any(k in key for k in lipid_keywords):
        return "Lipid metabolism"
    if any(k in key for k in ecm_keywords):
        return "ECM/stromal remodeling"
    if any(k in key for k in immune_keywords):
        return "Immune/protein"
    if any(k in key for k in stress_keywords):
        return "Stress/glycoxidation"
    if any(k in key for k in amino_keywords):
        return "Amino acid/nitrogen"
    return "Other"


def build_marker_module_table(feature_cols):
    rows = []
    for marker in feature_cols:
        module = infer_marker_module(marker)
        rows.append({"marker": marker, "module": module})
    return pd.DataFrame(rows)


def compute_module_scores(features: pd.DataFrame, marker_module: pd.DataFrame):
    score_df = pd.DataFrame(index=features.index)

    for module in MODULE_ORDER:
        markers = marker_module.loc[marker_module["module"] == module, "marker"].tolist()
        markers = [m for m in markers if m in features.columns]
        if markers:
            score_df[module] = features[markers].mean(axis=1)
        else:
            score_df[module] = np.nan

    # Dominant module is the module with highest mean z-score in that pixel.
    valid_modules = [m for m in MODULE_ORDER if score_df[m].notna().any()]
    if not valid_modules:
        raise RuntimeError("No valid marker-module score could be calculated.")

    dominant = score_df[valid_modules].idxmax(axis=1)
    score_df["dominant_module"] = dominant
    score_df["dominant_score"] = score_df[valid_modules].max(axis=1)
    return score_df


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def get_umap_limits(coords: pd.DataFrame):
    x = coords["umap_1"].values.astype(float)
    y = coords["umap_2"].values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    xmin, xmax = np.percentile(x[ok], [0.2, 99.8])
    ymin, ymax = np.percentile(y[ok], [0.2, 99.8])
    dx = (xmax - xmin) * 0.06 + 1e-9
    dy = (ymax - ymin) * 0.06 + 1e-9
    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


def style_axis(ax, limits, with_text=True, show_xlabel=False, show_ylabel=False):
    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    if with_text:
        ax.tick_params(axis="both", labelsize=7, width=0.7, length=2.5)
        ax.set_xlabel("UMAP1" if show_xlabel else "", fontsize=8)
        ax.set_ylabel("UMAP2" if show_ylabel else "", fontsize=8)
        if not show_xlabel:
            ax.set_xticklabels([])
        if not show_ylabel:
            ax.set_yticklabels([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def plot_background(ax, df):
    if len(df) > 60000:
        bg = df.sample(n=60000, random_state=1)
    else:
        bg = df
    ax.scatter(
        bg["umap_1"],
        bg["umap_2"],
        s=0.45,
        color="#D6D6D6",
        alpha=0.20,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )


def plot_group_module_panel(ax, df, group, limits, with_text=True, show_xlabel=False, show_ylabel=False):
    plot_background(ax, df)
    sub = df[df["group"].astype(str) == group].copy()

    # Draw module colors in a fixed order to keep visual consistency.
    for module in MODULE_ORDER:
        ss = sub[sub["dominant_module"].astype(str) == module]
        if ss.empty:
            continue
        ax.scatter(
            ss["umap_1"],
            ss["umap_2"],
            s=0.95,
            color=MODULE_COLORS[module],
            alpha=0.80,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    if with_text:
        ax.set_title(GROUP_TITLES[group], fontsize=8.5, fontweight="bold", pad=4)

    style_axis(ax, limits, with_text=with_text, show_xlabel=show_xlabel, show_ylabel=show_ylabel)


def plot_composition_bar(ax, comp: pd.DataFrame, with_text=True):
    x = np.arange(len(GROUP_ORDER))
    bottom = np.zeros(len(GROUP_ORDER), dtype=float)

    for module in MODULE_ORDER:
        vals = []
        for group in GROUP_ORDER:
            row = comp[(comp["group"] == group) & (comp["dominant_module"] == module)]
            vals.append(float(row["fraction"].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=MODULE_COLORS[module],
            edgecolor="white",
            linewidth=0.4,
            width=0.75,
            label=module,
        )
        bottom += np.array(vals)

    if with_text:
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_SHORT[g] for g in GROUP_ORDER], rotation=0, fontsize=8)
        ax.set_ylabel("Dominant module fraction", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            frameon=False,
            fontsize=7.2,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            columnspacing=1.0,
            handlelength=1.0,
        )
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def save_multi(fig, out_dir: Path, stem: str, dpi=600, pad=0.02):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=pad)


def make_main_figure(df: pd.DataFrame, comp: pd.DataFrame, out_dir: Path, with_text=True):
    limits = get_umap_limits(df)

    n_groups = len(GROUP_ORDER)
    fig = plt.figure(figsize=(2.08 * n_groups, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, n_groups, height_ratios=[1.0, 0.42])

    axes = []
    for j, group in enumerate(GROUP_ORDER):
        ax = fig.add_subplot(gs[0, j])
        axes.append(ax)
        plot_group_module_panel(
            ax,
            df,
            group,
            limits,
            with_text=with_text,
            show_xlabel=True,
            show_ylabel=(j == 0),
        )

    ax_bar = fig.add_subplot(gs[1, :])
    plot_composition_bar(ax_bar, comp, with_text=with_text)

    if with_text:
        axes[0].text(-0.28, 1.14, "a", transform=axes[0].transAxes, fontsize=13, fontweight="bold")
        ax_bar.text(-0.025, 1.12, "b", transform=ax_bar.transAxes, fontsize=13, fontweight="bold")
        fig.suptitle(
            "Dominant Raman molecular modules projected onto 3D UMAP states",
            fontsize=13.5,
            fontweight="bold",
            y=1.02,
        )

    stem = "Fig03_3D_dominant_module_UMAP_flow"
    if not with_text:
        stem += "_no_text"
    save_multi(fig, out_dir, stem)
    plt.close(fig)


def export_single_panels(df: pd.DataFrame, comp: pd.DataFrame, out_dir: Path):
    limits = get_umap_limits(df)

    for mode in ["with_text", "no_text"]:
        with_text = mode == "with_text"
        pdir = out_dir / f"single_panels_{mode}"
        pdir.mkdir(parents=True, exist_ok=True)

        for group in GROUP_ORDER:
            fig, ax = plt.subplots(figsize=(4.0, 3.8))
            plot_group_module_panel(ax, df, group, limits, with_text=with_text, show_xlabel=with_text, show_ylabel=with_text)
            fig.tight_layout()
            save_multi(fig, pdir, f"{GROUP_SHORT[group]}_dominant_module_umap_{mode}")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.0, 3.0))
        plot_composition_bar(ax, comp, with_text=with_text)
        fig.tight_layout()
        save_multi(fig, pdir, f"dominant_module_composition_{mode}")
        plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Dominant metabolic/module UMAP projection for 3D Raman data.")
    parser.add_argument("--input-dir", required=True, help="Fig01 v7 output folder containing source_embedding_coordinates.csv and feature matrix.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--embedding-csv", default=None)
    parser.add_argument("--feature-csv", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = Path(args.embedding_csv) if args.embedding_csv else input_dir / "source_embedding_coordinates.csv"
    feature_path = Path(args.feature_csv) if args.feature_csv else input_dir / "source_true_pixel_robust_z_features.csv"

    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {feature_path}")

    coords = pd.read_csv(embedding_path)
    features = pd.read_csv(feature_path)

    if len(coords) != len(features):
        raise RuntimeError(
            f"Embedding rows ({len(coords)}) and feature rows ({len(features)}) do not match."
        )

    marker_module = build_marker_module_table(features.columns)
    module_scores = compute_module_scores(features, marker_module)

    plot_df = coords.copy()
    plot_df["dominant_module"] = module_scores["dominant_module"].values
    plot_df["dominant_score"] = module_scores["dominant_score"].values
    plot_df = plot_df[plot_df["group"].isin(GROUP_ORDER)].copy()
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=GROUP_ORDER, ordered=True)

    comp = (
        plot_df.groupby(["group", "dominant_module"], observed=True)
        .size()
        .reset_index(name="n_pixels")
    )
    totals = plot_df.groupby("group", observed=True).size().rename("total_pixels").reset_index()
    comp = comp.merge(totals, on="group", how="left")
    comp["fraction"] = comp["n_pixels"] / comp["total_pixels"]

    # Save source outputs.
    marker_module.to_csv(out_dir / "source_marker_module_assignment.csv", index=False)
    module_scores.to_csv(out_dir / "source_pixel_module_scores.csv", index=False)
    plot_df.to_csv(out_dir / "source_embedding_with_dominant_module.csv", index=False)
    comp.to_csv(out_dir / "source_dominant_module_composition.csv", index=False)

    make_main_figure(plot_df, comp, out_dir, with_text=True)
    make_main_figure(plot_df, comp, out_dir, with_text=False)
    export_single_panels(plot_df, comp, out_dir)

    print("Done.")
    print(f"Output folder: {out_dir}")
    print(f"Pixels: {len(plot_df)}")
    print(f"Markers: {features.shape[1]}")
    print("Main figure: Fig03_3D_dominant_module_UMAP_flow.png")
    print("Please review source_marker_module_assignment.csv and adjust module rules if needed.")


if __name__ == "__main__":
    main()
