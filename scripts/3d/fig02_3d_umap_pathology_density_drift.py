#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fig02: pathology-specific UMAP density drift panels for 3D Raman data.

This script mimics a reference-style UMAP figure:
  Row A: each pathology group highlighted on a shared grey UMAP background.
  Row B: 2D KDE / density contours for each pathology group on the same background.

Input:
  The output folder from fig01_3d_pca_umap_clean_twopanel_v7.py, especially:
  source_embedding_coordinates.csv

Recommended:
  Use the 1500- or 2000-pixels-per-layer Fig01 result folder.

Run:
cd "FV-RMH"

python ".\\scripts\\3d\\fig02_3d_umap_pathology_density_drift.py" ^
  --input-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_1500pixels" ^
  --out-dir "3dresults\\fig02_3d_umap_pathology_density_drift_1500pixels"

For a 2000-pixels-per-layer Fig01 output:
python ".\\scripts\\3d\\fig02_3d_umap_pathology_density_drift.py" ^
  --input-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_2000pixels" ^
  --out-dir "3dresults\\fig02_3d_umap_pathology_density_drift_2000pixels"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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
GROUP_TITLE = {
    "Cancer-adjacent": "Cancer-adjacent",
    "Lepidic": "Lepidic",
    "Acinar": "Acinar",
    "Papillary": "Papillary",
    "Micropapillary": "Micropapillary",
    "Complex glands": "Complex glands",
    "Solid": "Solid",
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


def load_embedding(input_dir: Path, embedding_csv: str | None = None) -> pd.DataFrame:
    path = Path(embedding_csv) if embedding_csv else input_dir / "source_embedding_coordinates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Embedding CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"umap_1", "umap_2", "group"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Embedding CSV is missing required columns: {missing}")

    df = df[df["group"].isin(GROUP_ORDER)].copy()
    if df.empty:
        raise RuntimeError("No rows found for expected groups.")
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    return df


def get_limits(df: pd.DataFrame, xcol="umap_1", ycol="umap_2"):
    x = df[xcol].values.astype(float)
    y = df[ycol].values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    xmin, xmax = np.percentile(x[ok], [0.2, 99.8])
    ymin, ymax = np.percentile(y[ok], [0.2, 99.8])
    dx = (xmax - xmin) * 0.06 + 1e-9
    dy = (ymax - ymin) * 0.06 + 1e-9
    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


def density_grid(x, y, limits, bins=240, smooth=3.0):
    xmin, xmax, ymin, ymax = limits
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    hist, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
    )
    hist = gaussian_filter(hist, sigma=smooth)
    if np.max(hist) > 0:
        hist = hist / np.max(hist)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    return xcenters, ycenters, hist.T


def style_umap_axis(ax, limits, with_text=True, show_ylabel=False, show_xlabel=False):
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
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)


def plot_background(ax, df, background_alpha=0.11, background_size=0.55):
    if len(df) > 60000:
        bg = df.sample(n=60000, random_state=1)
    else:
        bg = df
    ax.scatter(
        bg["umap_1"],
        bg["umap_2"],
        s=background_size,
        color="#BDBDBD",
        alpha=background_alpha,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )


def plot_highlight_panel(ax, df, group, limits, with_text=True, show_ylabel=False):
    plot_background(ax, df)
    sub = df[df["group"].astype(str) == group]

    ax.scatter(
        sub["umap_1"],
        sub["umap_2"],
        s=0.90,
        color=GROUP_COLORS[group],
        alpha=0.78,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    if with_text:
        ax.set_title(GROUP_TITLE[group], fontsize=8.5, fontweight="bold", pad=4)

    style_umap_axis(ax, limits, with_text=with_text, show_ylabel=show_ylabel, show_xlabel=False)


def plot_contour_panel(ax, df, group, limits, levels=7, with_text=True, show_ylabel=False, show_xlabel=True):
    plot_background(ax, df, background_alpha=0.10, background_size=0.50)
    sub = df[df["group"].astype(str) == group]

    xcenters, ycenters, density = density_grid(
        sub["umap_1"],
        sub["umap_2"],
        limits,
        bins=235,
        smooth=3.6,
    )
    positive = density[density > 0]

    if positive.size >= 20:
        qs = np.linspace(0.50, 0.94, levels)
        contour_levels = np.unique(np.quantile(positive, qs))
        if contour_levels.size >= 2:
            ax.contour(
                xcenters,
                ycenters,
                density,
                levels=contour_levels,
                colors=[GROUP_COLORS[group]],
                linewidths=0.9,
                alpha=0.95,
                zorder=3,
            )
        else:
            ax.scatter(sub["umap_1"], sub["umap_2"], s=0.8, color=GROUP_COLORS[group], alpha=0.5, linewidths=0)
    else:
        ax.scatter(sub["umap_1"], sub["umap_2"], s=0.8, color=GROUP_COLORS[group], alpha=0.5, linewidths=0)

    style_umap_axis(ax, limits, with_text=with_text, show_ylabel=show_ylabel, show_xlabel=show_xlabel)


def save_multi(fig, out_dir: Path, stem: str, dpi=600, pad=0.02):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=pad)


def make_main_figure(df: pd.DataFrame, out_dir: Path, with_text=True):
    limits = get_limits(df)

    n_groups = len(GROUP_ORDER)
    fig_w = 2.10 * n_groups
    fig_h = 4.75

    fig, axes = plt.subplots(
        2,
        n_groups,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
    )

    for j, group in enumerate(GROUP_ORDER):
        plot_highlight_panel(
            axes[0, j],
            df,
            group,
            limits,
            with_text=with_text,
            show_ylabel=(j == 0),
        )
        plot_contour_panel(
            axes[1, j],
            df,
            group,
            limits,
            with_text=with_text,
            show_ylabel=(j == 0),
            show_xlabel=True,
        )

    if with_text:
        axes[0, 0].text(-0.28, 1.14, "a", transform=axes[0, 0].transAxes, fontsize=13, fontweight="bold")
        axes[1, 0].text(-0.28, 1.10, "b", transform=axes[1, 0].transAxes, fontsize=13, fontweight="bold")
        fig.suptitle(
            "Pathology-associated molecular landscape and density drift in 3D Raman UMAP space",
            fontsize=13.5,
            fontweight="bold",
            y=1.02,
        )

    stem = "Fig02_3D_UMAP_pathology_density_drift"
    if not with_text:
        stem += "_no_text"
    save_multi(fig, out_dir, stem)
    plt.close(fig)


def export_single_panels(df: pd.DataFrame, out_dir: Path):
    limits = get_limits(df)

    for mode in ["with_text", "no_text"]:
        with_text = mode == "with_text"
        panel_dir = out_dir / f"single_panels_{mode}"
        panel_dir.mkdir(parents=True, exist_ok=True)

        for group in GROUP_ORDER:
            safe = GROUP_SHORT[group]

            fig, ax = plt.subplots(figsize=(4.0, 3.8))
            plot_highlight_panel(ax, df, group, limits, with_text=with_text, show_ylabel=with_text)
            fig.tight_layout()
            save_multi(fig, panel_dir, f"{safe}_highlight_scatter_{mode}")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(4.0, 3.8))
            plot_contour_panel(ax, df, group, limits, with_text=with_text, show_ylabel=with_text, show_xlabel=with_text)
            fig.tight_layout()
            save_multi(fig, panel_dir, f"{safe}_density_contour_{mode}")
            plt.close(fig)


def write_summary(df: pd.DataFrame, out_dir: Path):
    rows = []
    for group in GROUP_ORDER:
        sub = df[df["group"].astype(str) == group]
        rows.append({
            "group": group,
            "short_label": GROUP_SHORT[group],
            "n_pixels": int(len(sub)),
            "umap1_median": float(np.nanmedian(sub["umap_1"])) if len(sub) else np.nan,
            "umap2_median": float(np.nanmedian(sub["umap_2"])) if len(sub) else np.nan,
            "umap1_mean": float(np.nanmean(sub["umap_1"])) if len(sub) else np.nan,
            "umap2_mean": float(np.nanmean(sub["umap_2"])) if len(sub) else np.nan,
        })
    pd.DataFrame(rows).to_csv(out_dir / "source_density_panel_group_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Draw 3D Raman UMAP pathology density-drift panels.")
    parser.add_argument("--input-dir", required=True, help="Folder containing source_embedding_coordinates.csv from Fig01 v7.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--embedding-csv", default=None, help="Optional direct path to source_embedding_coordinates.csv.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_embedding(input_dir, embedding_csv=args.embedding_csv)
    write_summary(df, out_dir)

    make_main_figure(df, out_dir, with_text=True)
    make_main_figure(df, out_dir, with_text=False)
    export_single_panels(df, out_dir)

    print("Done.")
    print(f"Input rows: {len(df)}")
    print(f"Output folder: {out_dir}")
    print("Main figure: Fig02_3D_UMAP_pathology_density_drift.png")


if __name__ == "__main__":
    main()
