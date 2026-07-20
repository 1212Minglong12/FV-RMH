#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fig11 v2: pathology-ordered pseudotime trajectory and Raman metabolic cascade.

Upgrades relative to the original version:
  1) Smooth spline-style trajectory curve instead of a hard polyline.
  2) More fluid / high-tech heatmap rendering (2D smoothing + custom colormap).
  3) Same underlying pseudotime logic: pixels are still projected to an
     ordered centroid trajectory, so the method remains unchanged.

Recommended command:
cd "FV-RMH"

py ".\\3dscripts\\fig11_3d_spatiotemporal_trajectory_pseudotime_v2_curve_tech.py" ^
  --input-dir ".\\3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_2000pixels" ^
  --out-dir ".\\3dresults\\fig11_3d_spatiotemporal_trajectory_pseudotime_2000pixels_curve_tech" ^
  --top-n 40 --bins 120 --smooth-sigma 2.4
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.spatial.distance import pdist

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

TECH_CMAP = LinearSegmentedColormap.from_list(
    "tech_flux",
    [
        "#24104f",  # deep indigo
        "#3b0ca3",  # violet
        "#4361ee",  # electric blue
        "#4cc9f0",  # cyan
        "#72efdd",  # aqua
        "#ffd166",  # warm gold
        "#ef476f",  # pink-red
    ],
    N=256,
)

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


def clean_label(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).replace("_", " ")).strip()


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
        raise RuntimeError(
            f"Embedding rows ({len(coords)}) and feature rows ({len(features)}) do not match."
        )

    keep = coords["group"].isin(GROUP_ORDER).values
    coords = coords.loc[keep].reset_index(drop=True)
    features = features.loc[keep].reset_index(drop=True)

    valid = [c for c in features.columns if np.nanstd(features[c].values.astype(float)) > 1e-10]
    features = features[valid].replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    return coords, features


def robust_limits(coords):
    x = coords["umap_1"].values.astype(float)
    y = coords["umap_2"].values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    xmin, xmax = np.nanpercentile(x[ok], [0.2, 99.8])
    ymin, ymax = np.nanpercentile(y[ok], [0.2, 99.8])
    dx = (xmax - xmin) * 0.06 + 1e-9
    dy = (ymax - ymin) * 0.06 + 1e-9
    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


def compute_centroid_polyline(coords: pd.DataFrame, selected_groups: list[str]):
    centers = []
    used_groups = []
    for group in selected_groups:
        idx = coords["group"].astype(str).values == group
        if np.sum(idx) < 10:
            continue
        x = np.nanmedian(coords.loc[idx, "umap_1"].values.astype(float))
        y = np.nanmedian(coords.loc[idx, "umap_2"].values.astype(float))
        centers.append([x, y])
        used_groups.append(group)

    if len(centers) < 2:
        raise RuntimeError("At least two groups with sufficient pixels are required for trajectory.")

    return np.asarray(centers, dtype=float), used_groups


def project_points_to_polyline(points: np.ndarray, polyline: np.ndarray):
    n_segments = len(polyline) - 1
    seg_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(seg_lengths)]
    total = cumulative[-1] if cumulative[-1] > 0 else 1.0

    best_dist = np.full(points.shape[0], np.inf)
    best_time = np.zeros(points.shape[0], dtype=float)
    best_proj = np.zeros_like(points)
    best_seg = np.zeros(points.shape[0], dtype=int)

    for s in range(n_segments):
        a = polyline[s]
        b = polyline[s + 1]
        v = b - a
        denom = float(np.dot(v, v))
        if denom <= 1e-12:
            continue
        w = points - a
        t = np.clip((w @ v) / denom, 0.0, 1.0)
        proj = a + np.outer(t, v)
        dist = np.linalg.norm(points - proj, axis=1)
        better = dist < best_dist
        best_dist[better] = dist[better]
        best_proj[better] = proj[better]
        best_seg[better] = s
        best_time[better] = (cumulative[s] + t[better] * seg_lengths[s]) / total * 100.0
    return best_time, best_proj, best_seg, best_dist


def smooth_polyline(polyline: np.ndarray, n_points=420, smooth=0.8):
    if len(polyline) < 2:
        return polyline

    try:
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))]
        if d[-1] <= 1e-12:
            return polyline

        u = d / d[-1]
        x = polyline[:, 0].astype(float)
        y = polyline[:, 1].astype(float)
        k = min(3, len(polyline) - 1)

        tck, _ = splprep([x, y], u=u, s=smooth, k=k)
        u_new = np.linspace(0, 1, n_points)
        xs, ys = splev(u_new, tck)
        return np.c_[xs, ys]

    except Exception:
        lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(lengths)]
        if cum[-1] <= 1e-12:
            return polyline
        t = cum / cum[-1]
        grid = np.linspace(0, 1, n_points)
        xs = np.interp(grid, t, polyline[:, 0])
        ys = np.interp(grid, t, polyline[:, 1])
        return np.c_[xs, ys]


def rank_dynamic_markers(features: pd.DataFrame, pseudotime: np.ndarray, bins=70):
    binned = np.linspace(0, 100, bins + 1)
    rows = []
    for marker in features.columns:
        vals = features[marker].values.astype(float)
        medians = []
        for lo, hi in zip(binned[:-1], binned[1:]):
            idx = (pseudotime >= lo) & (pseudotime < hi)
            if np.sum(idx) >= 20:
                medians.append(np.nanmedian(vals[idx]))
            else:
                medians.append(np.nan)
        medians = pd.Series(medians).interpolate(limit_direction="both").fillna(0).values
        smoothed = gaussian_filter1d(medians.astype(float), sigma=1.6)
        rows.append({
            "marker": marker,
            "dynamic_range": float(np.nanmax(smoothed) - np.nanmin(smoothed)),
            "dynamic_sd": float(np.nanstd(smoothed)),
            "early_mean": float(np.nanmean(smoothed[:max(3, bins // 5)])),
            "late_mean": float(np.nanmean(smoothed[-max(3, bins // 5):])),
            "late_minus_early": float(
                np.nanmean(smoothed[-max(3, bins // 5):]) - np.nanmean(smoothed[:max(3, bins // 5)])
            ),
        })
    return pd.DataFrame(rows).sort_values(["dynamic_range", "dynamic_sd"], ascending=False).reset_index(drop=True)


def pseudotime_heatmap(
    features: pd.DataFrame,
    pseudotime: np.ndarray,
    markers: list[str],
    bins=100,
    smooth_sigma=1.8,
    row_smooth_sigma=0.75,
):
    edges = np.linspace(0, 100, bins + 1)
    matrix = []
    for marker in markers:
        vals = features[marker].values.astype(float)
        profile = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = (pseudotime >= lo) & (pseudotime < hi)
            if np.sum(idx) >= 15:
                profile.append(np.nanmedian(vals[idx]))
            else:
                profile.append(np.nan)
        profile = pd.Series(profile).interpolate(limit_direction="both").fillna(0).values.astype(float)
        profile = gaussian_filter1d(profile, sigma=smooth_sigma)
        lo, hi = np.nanpercentile(profile, [2, 98])
        if np.isfinite(hi - lo) and hi > lo:
            profile = np.clip((profile - lo) / (hi - lo), 0, 1)
        else:
            profile = np.zeros_like(profile)
        matrix.append(profile)

    mat = np.vstack(matrix)

    if len(markers) > 2:
        try:
            order = leaves_list(linkage(pdist(mat, metric="euclidean"), method="ward"))
            mat = mat[order]
            markers = [markers[i] for i in order]
        except Exception:
            pass

    # display-layer 2D smoothing for a more fluid look
    mat = gaussian_filter(mat, sigma=(row_smooth_sigma, 1.15))
    mat = np.clip(mat, 0, 1)

    centers = (edges[:-1] + edges[1:]) / 2
    return mat, markers, centers


def draw_group_umap(ax, coords, trajectory_curve, selected_groups, limits):
    bg = coords.sample(n=70000, random_state=2) if len(coords) > 70000 else coords
    ax.scatter(bg["umap_1"], bg["umap_2"], s=0.45, color="#D9D9D9", alpha=0.20, linewidths=0, rasterized=True)
    for group in selected_groups:
        idx = coords["group"].astype(str).values == group
        sub = coords.loc[idx]
        if sub.empty:
            continue
        ax.scatter(
            sub["umap_1"], sub["umap_2"],
            s=0.85,
            color=GROUP_COLORS.get(group, "#777777"),
            alpha=0.62,
            linewidths=0,
            rasterized=True,
            label=GROUP_SHORT.get(group, group),
        )

    # soft glow underlay
    ax.plot(
        trajectory_curve[:, 0], trajectory_curve[:, 1],
        color="#000000", lw=5.0, alpha=0.10,
        solid_capstyle="round", zorder=4,
    )
    # main smooth trajectory
    ax.plot(
        trajectory_curve[:, 0], trajectory_curve[:, 1],
        color="#111111", lw=1.8, alpha=0.96,
        solid_capstyle="round", zorder=5,
    )
    # start / end markers
    ax.scatter(
        trajectory_curve[0, 0], trajectory_curve[0, 1],
        s=38, color="white", edgecolor="#111111", linewidth=0.9, zorder=6,
    )
    ax.scatter(
        trajectory_curve[-1, 0], trajectory_curve[-1, 1],
        s=38, color="#111111", edgecolor="white", linewidth=0.4, zorder=6,
    )
    # arrow at the end
    if len(trajectory_curve) >= 8:
        ax.annotate(
            "",
            xy=(trajectory_curve[-1, 0], trajectory_curve[-1, 1]),
            xytext=(trajectory_curve[-8, 0], trajectory_curve[-8, 1]),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#111111"),
            zorder=7,
        )

    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("A. Trajectory by pathology-ordered states", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Group", title_fontsize=7)


def draw_pseudotime_umap(ax, coords, pseudotime, trajectory_curve, limits):
    sc = ax.scatter(
        coords["umap_1"], coords["umap_2"],
        c=pseudotime,
        cmap="magma",
        s=0.75,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
        vmin=0,
        vmax=100,
    )
    ax.plot(
        trajectory_curve[:, 0], trajectory_curve[:, 1],
        color="#000000", lw=4.5, alpha=0.10,
        solid_capstyle="round", zorder=4,
    )
    ax.plot(
        trajectory_curve[:, 0], trajectory_curve[:, 1],
        color="#111111", lw=1.6, alpha=0.92,
        solid_capstyle="round", zorder=5,
    )
    if len(trajectory_curve) >= 8:
        ax.annotate(
            "",
            xy=(trajectory_curve[-1, 0], trajectory_curve[-1, 1]),
            xytext=(trajectory_curve[-8, 0], trajectory_curve[-8, 1]),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#111111"),
            zorder=7,
        )
    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("B. Pathology-ordered pseudotime score", loc="left", fontsize=10.5, fontweight="bold")
    return sc


def draw_pseudotime_heatmap(ax, mat, markers, centers):
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=TECH_CMAP,
        vmin=0,
        vmax=1,
        interpolation="bicubic",
    )
    try:
        ax.contour(
            mat,
            levels=np.linspace(0.18, 0.90, 6),
            colors="white",
            linewidths=0.22,
            alpha=0.16,
        )
    except Exception:
        pass

    ax.set_xlabel("Pseudotime trajectory score (0 to 100)", fontsize=9)
    ax.set_ylabel("Dynamic Raman markers", fontsize=9)
    ax.set_xticks(np.linspace(0, len(centers) - 1, 6))
    ax.set_xticklabels([f"{int(x)}" for x in np.linspace(0, 100, 6)], fontsize=8)
    if len(markers) <= 55:
        ax.set_yticks(np.arange(len(markers)))
        ax.set_yticklabels([clean_label(m) for m in markers], fontsize=4.8)
    else:
        ax.set_yticks([])
    ax.set_title("C. Pseudotime-ordered Raman metabolic cascade", loc="left", fontsize=10.5, fontweight="bold")
    return im


def save_multi(fig, out_dir: Path, stem: str, dpi=600, pad=0.02):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=pad)


def make_main_figure(coords, pseudotime, trajectory_curve, used_groups, heatmap_mat, heatmap_markers, centers, out_dir, with_text=True):
    limits = robust_limits(coords)
    fig = plt.figure(figsize=(13.5, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.32], width_ratios=[1, 1], hspace=0.13, wspace=0.17)
    ax_a = fig.add_subplot(gs[0, 0])
    draw_group_umap(ax_a, coords, trajectory_curve, used_groups, limits)
    ax_b = fig.add_subplot(gs[0, 1])
    sc = draw_pseudotime_umap(ax_b, coords, pseudotime, trajectory_curve, limits)
    cbar = fig.colorbar(sc, ax=ax_b, fraction=0.035, pad=0.02)
    cbar.set_label("Pseudotime", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax_c = fig.add_subplot(gs[1, :])
    im = draw_pseudotime_heatmap(ax_c, heatmap_mat, heatmap_markers, centers)
    cbar2 = fig.colorbar(im, ax=ax_c, fraction=0.018, pad=0.012)
    cbar2.set_label("Relative abundance", fontsize=8)
    cbar2.ax.tick_params(labelsize=7)
    if with_text:
        fig.suptitle(
            "Pathology-ordered Raman trajectory and metabolic cascade across LUAD growth-pattern states",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
    stem = "Fig11_3D_spatiotemporal_trajectory_pseudotime_v2_curve_tech" + ("" if with_text else "_no_text")
    save_multi(fig, out_dir, stem)
    plt.close(fig)


def export_single_panels(coords, pseudotime, trajectory_curve, used_groups, heatmap_mat, heatmap_markers, centers, out_dir):
    limits = robust_limits(coords)
    folder = out_dir / "single_panels"
    folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    draw_group_umap(ax, coords, trajectory_curve, used_groups, limits)
    save_multi(fig, folder, "panel_A_group_trajectory_umap_v2_curve_tech")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    sc = draw_pseudotime_umap(ax, coords, pseudotime, trajectory_curve, limits)
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label("Pseudotime", fontsize=8)
    save_multi(fig, folder, "panel_B_pseudotime_umap_v2_curve_tech")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    im = draw_pseudotime_heatmap(ax, heatmap_mat, heatmap_markers, centers)
    fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012).set_label("Relative abundance", fontsize=8)
    save_multi(fig, folder, "panel_C_pseudotime_marker_heatmap_v2_curve_tech")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Spatiotemporal UMAP trajectory and pseudotime marker heatmap.")
    parser.add_argument("--input-dir", required=True, help="Fig01 v7 output folder.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selected-groups", nargs="+", default=GROUP_ORDER, help="Ordered groups defining the trajectory.")
    parser.add_argument("--top-n", type=int, default=40, help="Top dynamic markers shown in heatmap.")
    parser.add_argument("--bins", type=int, default=120, help="Number of pseudotime bins.")
    parser.add_argument("--smooth-sigma", type=float, default=2.4, help="Gaussian smoothing sigma for marker trajectories.")
    args = parser.parse_args()

    selected_groups = [g for g in args.selected_groups if g in GROUP_ORDER]
    if len(selected_groups) < 2:
        raise RuntimeError("Need at least two valid selected groups.")

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords, features = load_inputs(input_dir)
    centers, used_groups = compute_centroid_polyline(coords, selected_groups)
    points = coords[["umap_1", "umap_2"]].values.astype(float)
    pseudotime, projected, nearest_segment, distance = project_points_to_polyline(points, centers)
    trajectory_curve = smooth_polyline(centers, n_points=420, smooth=0.8)

    ranking = rank_dynamic_markers(features, pseudotime, bins=args.bins)
    top_markers = ranking.head(min(args.top_n, len(ranking)))["marker"].tolist()
    heatmap_mat, heatmap_markers, bin_centers = pseudotime_heatmap(
        features,
        pseudotime,
        top_markers,
        bins=args.bins,
        smooth_sigma=args.smooth_sigma,
        row_smooth_sigma=0.75,
    )

    meta = coords.copy()
    meta["pseudotime"] = pseudotime
    meta["trajectory_projected_umap1"] = projected[:, 0]
    meta["trajectory_projected_umap2"] = projected[:, 1]
    meta["nearest_trajectory_segment"] = nearest_segment
    meta["trajectory_distance"] = distance

    centers_df = pd.DataFrame(centers, columns=["centroid_umap1", "centroid_umap2"])
    centers_df["group"] = used_groups
    centers_df["trajectory_order"] = np.arange(len(used_groups))

    meta.to_csv(out_dir / "source_pseudotime_pixel_metadata.csv", index=False)
    centers_df.to_csv(out_dir / "source_trajectory_group_centroids.csv", index=False)
    ranking.to_csv(out_dir / "source_pseudotime_marker_dynamic_ranking.csv", index=False)
    pd.DataFrame(heatmap_mat, index=heatmap_markers).to_csv(out_dir / "source_pseudotime_heatmap_matrix.csv")
    pd.DataFrame({"pseudotime_bin_center": bin_centers}).to_csv(out_dir / "source_pseudotime_bin_centers.csv", index=False)

    make_main_figure(coords, pseudotime, trajectory_curve, used_groups, heatmap_mat, heatmap_markers, bin_centers, out_dir, with_text=True)
    make_main_figure(coords, pseudotime, trajectory_curve, used_groups, heatmap_mat, heatmap_markers, bin_centers, out_dir, with_text=False)
    export_single_panels(coords, pseudotime, trajectory_curve, used_groups, heatmap_mat, heatmap_markers, bin_centers, out_dir)

    summary = (
        "Fig11 pseudotime trajectory summary (v2 curve+tech)\n"
        "==================================================\n"
        f"Input folder: {input_dir}\n"
        f"Trajectory groups: {', '.join(used_groups)}\n"
        f"Pixel observations: {len(coords)}\n"
        f"Markers used for ranking: {features.shape[1]}\n"
        f"Top dynamic markers plotted: {len(heatmap_markers)}\n"
        f"Pseudotime bins: {args.bins}\n\n"
        "Pseudotime was inferred by projecting pixel-level UMAP states onto a polyline connecting ordered group centroids. "
        "This is a pathology-ordered trajectory score, not chronological time or direct lineage inference.\n"
        "Visual rendering was upgraded with a smoothed spline-like display curve and a fluid heatmap style; the underlying projection logic was not changed.\n"
    )
    (out_dir / "pseudotime_methods_summary.txt").write_text(summary, encoding="utf-8")

    print("Done.")
    print(f"Output folder: {out_dir}")
    print(f"Trajectory groups: {', '.join(used_groups)}")
    print(f"Pixel observations: {len(coords)}")
    print(f"Top dynamic markers plotted: {len(heatmap_markers)}")
    print("Main figure: Fig11_3D_spatiotemporal_trajectory_pseudotime_v2_curve_tech.png")


if __name__ == "__main__":
    main()
