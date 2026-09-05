# -*- coding: utf-8 -*-
"""
Raman 3D reference-free validation V5.3C FINAL
==============================================

Formal no-reference-spectrum validation pipeline for volumetric Raman data.

Key features
------------
1) NO external reference spectra are required.
2) All available optical layers contribute to ONE common model.
3) The same PCA / NMF / cluster models are mapped back to every layer.
4) A-style publication palette: deep navy -> teal -> pale gold.
5) Numerical layer caches are saved, so figures can be recolored/redrawn later.
6) Resume mode can continue mapping after an interrupted run.
7) Publication-focused outputs are generated automatically.
8) UMAP30 / t-SNE / binary cluster masks are optional exploratory outputs.

Recommended manuscript interpretation
-------------------------------------
RF1, RF2, ... are reference-free spectral states/components, not specific molecules.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.ndimage import minimum_filter1d, gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec

from sklearn.decomposition import PCA, NMF
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap
except Exception:
    umap = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------
ORDER = ["NAT", "LEP", "ACN", "PAP", "MP", "CGP", "SOL"]
GROUP_MAP = {
    "pneumonia": "NAT",
    "nat": "NAT",
    "healthy": "NAT",
    "peritumoral": "NAT",
    "tiebi": "LEP",
    "lepidic": "LEP",
    "xianpao": "ACN",
    "acinar": "ACN",
    "papillary": "PAP",
    "micropapillary": "MP",
    "micropap": "MP",
    "fzxt": "CGP",
    "complexglands": "CGP",
    "complex_glands": "CGP",
    "solid": "SOL",
    "soild": "SOL",
}
PATTERN = re.compile(r"^(.*?)[_-]layer(\d+)\.mat$", re.I)


# ---------------------------------------------------------------------
# A palette: deep navy -> teal -> pale gold
# ---------------------------------------------------------------------
def cmap_a_seq():
    # C palette: Midnight -> Cyan/Teal -> Coral/Peach
    colors = ["#091A33", "#173C64", "#1E7F91", "#49C0B3", "#E68D88", "#FFD0B8"]
    cm = LinearSegmentedColormap.from_list("C_seq", colors, N=256)
    cm.set_bad("#FFFFFF")
    return cm


def cmap_a_div():
    # C-style diverging map for signed statistics
    colors = ["#173C64", "#49C0B3", "#F7F6F3", "#F2B2A5", "#E68D88", "#B55263"]
    cm = LinearSegmentedColormap.from_list("C_div", colors, N=256)
    cm.set_bad("#FFFFFF")
    return cm


SEQ = cmap_a_seq()
DIV = cmap_a_div()

def cmap_umap_pretty():
    # Blue -> cyan -> magenta/pink, matching the reference UMAP30 aesthetic.
    colors = ["#080B4A", "#093B99", "#0078C8", "#12BED0", "#72D7E5", "#D95CAF", "#F5A7D2", "#FFD6EA"]
    cm = LinearSegmentedColormap.from_list("UMAP_pretty", colors, N=256)
    cm.set_bad("#F4F4F4")
    return cm

UMAP_PRETTY = cmap_umap_pretty()

GROUP_COLORS = {
    "NAT": "#091A33",
    "LEP": "#173C64",
    "ACN": "#1E7F91",
    "PAP": "#49C0B3",
    "MP":  "#C87482",
    "CGP": "#E68D88",
    "SOL": "#B55263",
}


def cluster_colors(n: int):
    # C-style categorical palette for clusters/t-SNE.
    base = [
        "#091A33", "#102B4B", "#173C64", "#1A506F", "#1D657D",
        "#1E7F91", "#25939C", "#33A7A6", "#49C0B3", "#6ACCC1",
        "#8BD5CC", "#A7DDD5", "#C0E4DC", "#D6EAE4", "#F0E2DD",
        "#F3CFC6", "#F0B8AE", "#EFA296", "#E68D88", "#DB7A7F",
        "#CF6876", "#C15B6C", "#B55263", "#9E485A", "#85404F",
        "#6D3A48", "#5B3947", "#493948", "#3A394A", "#2C3749",
    ]
    if n <= len(base):
        return base[:n]
    reps = math.ceil(n / len(base))
    return (base * reps)[:n]



# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------
# File discovery / MAT loading
# ---------------------------------------------------------------------
def normalize_group(raw: str):
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return GROUP_MAP.get(key) or GROUP_MAP.get(key.replace("_", ""))


def discover(folder: Path):
    rows = []
    for p in folder.glob("*.mat"):
        m = PATTERN.match(p.name)
        if not m:
            continue
        g = normalize_group(m.group(1))
        if g:
            rows.append((p, m.group(1), g, int(m.group(2))))
    rows.sort(key=lambda x: (ORDER.index(x[2]), x[3], x[0].name.lower()))
    return rows


def collect_numeric(obj, out):
    if obj is None:
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype != object and np.issubdtype(obj.dtype, np.number):
            out.append(np.asarray(obj))
            return
        for v in obj.flat:
            collect_numeric(v, out)
        return
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if not k.startswith("_"):
                collect_numeric(v, out)


def load_mat(path: Path):
    try:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as e:
        raise RuntimeError(
            f"{path.name}: MATLAB v7.3/HDF5 file detected. "
            f"This file needs an HDF5 loader; the current dataset previously worked with scipy.io.loadmat."
        ) from e

    if "data_struct" not in m:
        raise ValueError(f"{path.name}: missing variable 'data_struct'.")

    ds = m["data_struct"]
    X = np.asarray(ds.data, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{path.name}: data_struct.data must be 2-D, got {X.shape}.")

    numeric = []
    collect_numeric(ds.axisscale, numeric)
    axis = None
    for a in numeric:
        arr = np.asarray(a).squeeze()
        if arr.ndim == 1 and arr.size == X.shape[1]:
            axis = arr.astype(np.float32)
            break
    if axis is None:
        raise ValueError(f"{path.name}: Raman axis not found.")

    ims_raw = np.asarray(ds.imagesize).squeeze()
    if ims_raw.size < 2:
        raise ValueError(f"{path.name}: invalid imagesize.")
    ims = (int(ims_raw.flat[0]), int(ims_raw.flat[1]))
    if ims[0] * ims[1] != X.shape[0]:
        raise ValueError(
            f"{path.name}: imagesize {ims} does not match {X.shape[0]} spectra."
        )
    return X, axis, ims


# ---------------------------------------------------------------------
# Spectral preprocessing
# ---------------------------------------------------------------------
def select_fingerprint(X, axis, low=600.0, high=1800.0):
    mask = (axis >= low) & (axis <= high)
    if mask.sum() < 10:
        raise ValueError("Too few Raman bands in the requested fingerprint region.")
    return X[:, mask], axis[mask]


def common_axis_from_all(source_axes, low=600.0, high=1800.0, step=4.0):
    mins, maxs = [], []
    for ax in source_axes:
        ax = np.asarray(ax, dtype=np.float32).ravel()
        seg = ax[(ax >= low) & (ax <= high)]
        if seg.size < 2:
            raise ValueError(f"A file does not cover {low:.0f}-{high:.0f} cm^-1.")
        mins.append(float(seg.min()))
        maxs.append(float(seg.max()))

    overlap_min = max(mins)
    overlap_max = min(maxs)
    start = math.ceil(overlap_min / step) * step
    stop = math.floor(overlap_max / step) * step
    if stop <= start:
        raise RuntimeError(
            f"No common Raman overlap. Shared range={overlap_min:.2f}-{overlap_max:.2f} cm^-1."
        )
    grid = np.arange(start, stop + step * 0.5, step, dtype=np.float32)
    info = {
        "requested_min_cm-1": low,
        "requested_max_cm-1": high,
        "shared_overlap_min_cm-1": overlap_min,
        "shared_overlap_max_cm-1": overlap_max,
        "grid_min_cm-1": float(grid.min()),
        "grid_max_cm-1": float(grid.max()),
        "grid_step_cm-1": float(step),
        "grid_bands": int(grid.size),
    }
    return grid, info


def resample_spectra(X, source_axis, target_axis):
    X = np.asarray(X, dtype=np.float32)
    source_axis = np.asarray(source_axis, dtype=np.float32).ravel()
    target_axis = np.asarray(target_axis, dtype=np.float32).ravel()

    if source_axis[0] > source_axis[-1]:
        source_axis = source_axis[::-1]
        X = X[:, ::-1]

    if target_axis.min() < source_axis.min() - 0.5 or target_axis.max() > source_axis.max() + 0.5:
        raise ValueError(
            f"Source axis {source_axis.min():.2f}-{source_axis.max():.2f} does not cover "
            f"target {target_axis.min():.2f}-{target_axis.max():.2f}."
        )

    f = interp1d(
        source_axis,
        X,
        axis=1,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    return np.asarray(f(target_axis), dtype=np.float32)


def preprocess(X):
    """
    Reference-free preprocessing used consistently for training and mapping:
    - Savitzky-Golay smoothing
    - smooth minimum-envelope baseline
    - non-negative clipping
    - area normalization
    """
    X = np.asarray(X, dtype=np.float32)

    if X.shape[1] >= 9:
        X = savgol_filter(X, 9, 2, axis=1, mode="interp").astype(np.float32)

    baseline = minimum_filter1d(X, size=31, axis=1, mode="nearest")
    baseline = gaussian_filter1d(baseline, sigma=4, axis=1, mode="nearest")
    Y = X - baseline
    Y[~np.isfinite(Y)] = 0.0
    Y[Y < 0] = 0.0

    area = np.sum(Y, axis=1, keepdims=True)
    area[area <= 1e-12] = 1.0
    return (Y / area).astype(np.float32)


def valid_mask(X):
    X = np.asarray(X, dtype=np.float32)
    inten = np.nanmean(np.abs(X), axis=1)
    var = np.nanstd(X, axis=1)
    return (
        np.isfinite(inten)
        & np.isfinite(var)
        & (inten > np.nanquantile(inten, 0.10))
        & (var > np.nanquantile(var, 0.10))
    )


def tomap(v, ims):
    return np.asarray(v).reshape(ims, order="F")


def place_valid(values, valid_mask_flat, fill=np.nan, dtype=np.float32):
    values = np.asarray(values)
    out = np.full(valid_mask_flat.size, fill, dtype=dtype)
    out[np.where(valid_mask_flat)[0]] = values
    return out


def robust_limits(arrays, lo=1.0, hi=99.0):
    vals = []
    for a in arrays:
        x = np.asarray(a).ravel()
        x = x[np.isfinite(x)]
        if x.size:
            if x.size > 250000:
                idx = np.linspace(0, x.size - 1, 250000, dtype=int)
                x = x[idx]
            vals.append(x)
    if not vals:
        return 0.0, 1.0
    x = np.concatenate(vals)
    vmin = float(np.percentile(x, lo))
    vmax = float(np.percentile(x, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return vmin, vmax


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def savefig(path: Path, dpi=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_mean_spectra(axis0, mean_spectra, outpath):
    plt.figure(figsize=(10, 6))
    for g in ORDER:
        if g in mean_spectra:
            y = mean_spectra[g]
            y = y / max(np.max(y), 1e-12)
            plt.plot(axis0, y, lw=2.1, label=g, color=GROUP_COLORS[g])
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Normalized mean intensity")
    plt.title("Mean Raman spectra by pathological group")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    savefig(outpath)


def plot_pca_loadings(axis0, loadings, evr, outpath):
    plt.figure(figsize=(10, 6))
    n = min(5, loadings.shape[0])
    for i in range(n):
        plt.plot(axis0, loadings[i], lw=2, label=f"PC{i+1} ({100*evr[i]:.1f}%)")
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("PCA loading")
    plt.title("Reference-free PCA spectral loadings")
    plt.legend(frameon=False)
    plt.tight_layout()
    savefig(outpath)


def plot_group_scatter(xy, labels, title, xlabel, ylabel, outpath):
    plt.figure(figsize=(8, 7))
    for g in ORDER:
        m = labels == g
        if np.any(m):
            plt.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.35, label=g, color=GROUP_COLORS[g])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False, markerscale=2)
    plt.tight_layout()
    savefig(outpath)


def plot_nmf_spectra(axis0, components, outpath):
    plt.figure(figsize=(10, 6))
    colors = cluster_colors(components.shape[0])
    for i in range(components.shape[0]):
        y = components[i] / max(np.max(components[i]), 1e-12)
        plt.plot(axis0, y, lw=2.1, label=f"RF{i+1}", color=colors[i])
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Normalized intensity")
    plt.title("Reference-free extracted Raman components")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    savefig(outpath)


def plot_heatmap(df, title, outpath, cmap=SEQ, vmin=None, vmax=None, fmt=None):
    arr = df.values.astype(float)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    if fmt:
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                if np.isfinite(arr[r, c]):
                    ax.text(c, r, fmt.format(arr[r, c]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    savefig(outpath)


def plot_rep_maps(component_number, group_maps, outpath, vmin, vmax):
    groups = [g for g in ORDER if g in group_maps]
    fig, axes = plt.subplots(1, len(groups), figsize=(2.8 * len(groups), 3.5))
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        item = group_maps[g]
        ax.imshow(item["map"], cmap=SEQ, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"{g}\nLayer {item['layer']}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"RF{component_number}: representative-layer spatial distribution", fontsize=14)
    cax = fig.add_axes([0.92, 0.18, 0.014, 0.60])
    sm = plt.cm.ScalarMappable(cmap=SEQ)
    sm.set_clim(vmin, vmax)
    fig.colorbar(sm, cax=cax)
    fig.subplots_adjust(right=0.90, wspace=0.08)
    savefig(outpath)


def plot_all_layers(group, component_number, layer_maps, outpath, vmin, vmax):
    layers = sorted(layer_maps)
    ncols = 4
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.25*ncols, 3.05*nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, layer in zip(axes, layers):
        ax.imshow(layer_maps[layer], cmap=SEQ, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.axis("off")
    for ax in axes[len(layers):]:
        ax.axis("off")
    fig.suptitle(f"{group}: RF{component_number} across all optical layers", fontsize=14)
    cax = fig.add_axes([0.92, 0.16, 0.014, 0.66])
    sm = plt.cm.ScalarMappable(cmap=SEQ)
    sm.set_clim(vmin, vmax)
    fig.colorbar(sm, cax=cax)
    fig.subplots_adjust(right=0.90, hspace=0.12, wspace=0.08)
    savefig(outpath)


def plot_cluster_layers(group, layer_maps, k, outpath):
    colors = cluster_colors(k)
    cm = ListedColormap(colors)
    cm.set_bad("#FFFFFF")
    layers = sorted(layer_maps)
    ncols = 4
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.25*ncols, 3.05*nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, layer in zip(axes, layers):
        ax.imshow(layer_maps[layer], cmap=cm, vmin=-0.5, vmax=k-0.5, aspect="equal")
        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.axis("off")
    for ax in axes[len(layers):]:
        ax.axis("off")
    fig.suptitle(f"{group}: dominant reference-free spectral clusters", fontsize=14)
    fig.subplots_adjust(hspace=0.12, wspace=0.08)
    savefig(outpath)


def plot_representative_cluster_map(group, layer, clmap, k, outpath):
    """Single representative spectral-cluster map, analogous to a chemical cluster map."""
    colors = cluster_colors(k)
    cm = ListedColormap(colors)
    cm.set_bad("#FFFFFF")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    im = ax.imshow(clmap, cmap=cm, vmin=-0.5, vmax=k-0.5, aspect="equal")
    ax.set_title(f"{group}: Reference-free Spectral Cluster Map (Layer {layer})", fontsize=13)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, ticks=np.arange(k))
    cb.set_label("Spectral cluster")
    fig.tight_layout()
    savefig(outpath)


def plot_umap_cluster_projection(xy, labels, k, outpath):
    """2-D UMAP projection colored by the PCA-space spectral-cluster assignments."""
    colors = cluster_colors(k)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for i in range(k):
        m = labels == i
        if np.any(m):
            ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.55, color=colors[i], label=str(i), linewidths=0)
    ax.set_xlabel("UMAP Axis 1")
    ax.set_ylabel("UMAP Axis 2")
    ax.set_title("UMAP Projection of Reference-free Spectral Clusters")
    if k <= 12:
        ax.legend(title="Cluster", frameon=False, ncol=2, markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    savefig(outpath)


def plot_cluster_summary_triptych(group, layer, clmap, umap_xy, umap_labels,
                                  axis0, cluster_means, k, outpath):
    """
    Composite figure in the same logical style as:
    spatial cluster map + UMAP cluster projection + cluster spectra.
    """
    colors = cluster_colors(k)
    cm = ListedColormap(colors)
    cm.set_bad("#FFFFFF")

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.05], hspace=0.30, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(clmap, cmap=cm, vmin=-0.5, vmax=k-0.5, aspect="equal")
    ax1.set_title(f"{group}: Spectral Cluster Map (Layer {layer})")
    ax1.set_xlabel("X pixel")
    ax1.set_ylabel("Y pixel")

    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(k):
        m = umap_labels == i
        if np.any(m):
            ax2.scatter(umap_xy[m, 0], umap_xy[m, 1], s=5, alpha=0.52,
                        color=colors[i], linewidths=0)
    ax2.set_title("UMAP Projection")
    ax2.set_xlabel("UMAP Axis 1")
    ax2.set_ylabel("UMAP Axis 2")

    ax3 = fig.add_subplot(gs[1, :])
    for i in range(k):
        y = cluster_means[i]
        y = y / max(np.max(y), 1e-12)
        ax3.plot(axis0, y, lw=1.6, color=colors[i], label=f"Cluster {i}")
    ax3.set_title("Cluster Spectra (Preprocessed)")
    ax3.set_xlabel("Raman shift (cm$^{-1}$)")
    ax3.set_ylabel("Normalized mean intensity")
    if k <= 12:
        ax3.legend(frameon=False, ncol=5, fontsize=8)

    fig.suptitle("Reference-free Spectral Cluster Summary", fontsize=16, y=0.98)
    savefig(outpath)


def plot_umap30_pretty(group, layer, maps30, outpath):
    """
    Publication-style UMAP30 panel:
    - each UMAP dimension has its own robust 2–98% scale;
    - blue/cyan/pink colormap;
    - narrow individual colorbars;
    - masked pixels shown as neutral light gray.
    Individual panels are therefore optimized for morphology, not cross-dimension
    quantitative comparison.
    """
    n = maps30.shape[0]
    ncols = 6
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 15), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        ax = axes[i]
        vmin, vmax = robust_limits([maps30[i]], 2, 98)
        im = ax.imshow(maps30[i], cmap=UMAP_PRETTY, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"UMAP {i+1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.038, pad=0.018)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{group}: Spatial Distribution of {n} UMAP Dimensions (Layer {layer})",
                 fontsize=17)
    savefig(outpath)

def plot_cluster_mean_spectra(axis0, means, outpath):
    colors = cluster_colors(means.shape[0])
    plt.figure(figsize=(10.5, 6.5))
    for i in range(means.shape[0]):
        y = means[i] / max(np.max(means[i]), 1e-12)
        plt.plot(axis0, y, lw=1.7, label=f"Cluster {i}", color=colors[i])
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Normalized mean intensity")
    plt.title("Mean Raman spectra of reference-free spectral clusters")
    plt.legend(frameon=False, ncol=2 if means.shape[0] <= 12 else 3, fontsize=8)
    plt.tight_layout()
    savefig(outpath)


def plot_tsne(xy, labels, k, outpath):
    colors = cluster_colors(k)
    plt.figure(figsize=(8.2, 7.2))
    for i in range(k):
        m = labels == i
        if np.any(m):
            plt.scatter(xy[m, 0], xy[m, 1], s=9, alpha=0.5, color=colors[i], label=f"C{i}")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.title(f"t-SNE visualization of {k} reference-free spectral clusters")
    if k <= 12:
        plt.legend(frameon=False, ncol=2, markerscale=1.5)
    plt.tight_layout()
    savefig(outpath)


def plot_cluster_masks(group, layer, clmap, k, outpath):
    ncols = 5
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(k):
        ax = axes[i]
        binary = np.where(np.isfinite(clmap), clmap == i, np.nan)
        cm = LinearSegmentedColormap.from_list("mask", ["#FFFFFF", "#173C64"])
        cm.set_bad("#FFFFFF")
        ax.imshow(binary.astype(float), cmap=cm, vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"Cluster {i}", fontsize=9)
        ax.axis("off")
    for ax in axes[k:]:
        ax.axis("off")
    fig.suptitle(f"{group}: spectral-cluster masks (Layer {layer})", fontsize=14)
    fig.tight_layout()
    savefig(outpath)


def plot_umap30(group, layer, maps30, outpath):
    # Use the enhanced reference-style UMAP30 renderer.
    plot_umap30_pretty(group, layer, maps30, outpath)


def plot_workflow(outpath):
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    boxes = [
        ("Raw volumetric\nRaman spectra", 0.03),
        ("Common Raman grid\n+ preprocessing", 0.23),
        ("Reference-free\nPCA / NMF / clustering", 0.45),
        ("Spatial remapping\nto each optical layer", 0.68),
        ("Depth continuity\nquantification", 0.87),
    ]
    for text, x in boxes:
        ax.text(
            x, 0.5, text, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F1F2", edgecolor="#1E7F91", linewidth=1.5)
        )
    for x1, x2 in [(0.115,0.155),(0.335,0.375),(0.565,0.605),(0.78,0.815)]:
        ax.annotate("", xy=(x2,0.5), xytext=(x1,0.5),
                    arrowprops=dict(arrowstyle="->", color="#173C64", lw=1.8))
    ax.text(0.5, 0.08, "No purified molecular reference spectra are supplied", ha="center",
            color="#B55263", fontsize=10)
    savefig(outpath)


# ---------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------
def layer_cache_path(out: Path, group: str, layer: int):
    return out / "CACHE" / "layers" / group / f"{group}_layer{layer}.npz"


def save_layer_cache(path: Path, group: str, layer: int, ims, component_maps, cluster_map):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        group=np.array(group),
        layer=np.array(layer, dtype=np.int16),
        ims=np.asarray(ims, dtype=np.int32),
        component_maps=np.asarray(component_maps, dtype=np.float32),
        cluster_map=np.asarray(cluster_map, dtype=np.float32),
    )


def load_layer_cache(path: Path):
    z = np.load(path, allow_pickle=False)
    return {
        "group": str(z["group"].item()),
        "layer": int(z["layer"].item()),
        "ims": tuple(int(x) for x in z["ims"].ravel()[:2]),
        "component_maps": np.asarray(z["component_maps"], dtype=np.float32),
        "cluster_map": np.asarray(z["cluster_map"], dtype=np.float32),
    }


def load_all_cached_states(out: Path):
    states = {}
    for p in sorted((out / "CACHE" / "layers").glob("*/*.npz")):
        st = load_layer_cache(p)
        states[(st["group"], st["layer"])] = st
    return states


# ---------------------------------------------------------------------
# Derived summaries and redraw
# ---------------------------------------------------------------------
def representative_layers(records):
    reps = {}
    for g in ORDER:
        ls = sorted({r[3] for r in records if r[2] == g})
        if ls:
            reps[g] = ls[len(ls)//2]
    return reps


def compute_continuity(states, ncomp):
    rows = []
    for g in ORDER:
        layers = sorted(layer for gg, layer in states if gg == g)
        for c in range(ncomp):
            for a, b in zip(layers[:-1], layers[1:]):
                A = states[(g, a)]["component_maps"][c].ravel()
                B = states[(g, b)]["component_maps"][c].ravel()
                m = np.isfinite(A) & np.isfinite(B)
                if m.sum() < 20 or np.nanstd(A[m]) <= 1e-12 or np.nanstd(B[m]) <= 1e-12:
                    rho = np.nan
                else:
                    rho = float(spearmanr(A[m], B[m]).correlation)
                rows.append({
                    "group": g,
                    "component": f"RF{c+1}",
                    "layer_a": a,
                    "layer_b": b,
                    "spearman_rho": rho,
                })
    return pd.DataFrame(rows)


def continuity_summary(continuity_df, components):
    rows = []
    for g in ORDER:
        row = {}
        for c in components:
            s = continuity_df.loc[
                (continuity_df["group"] == g) &
                (continuity_df["component"] == f"RF{c+1}"),
                "spearman_rho"
            ]
            row[f"RF{c+1}"] = float(np.nanmean(s)) if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=ORDER)


def component_global_limits(states, c):
    arrays = [st["component_maps"][c] for st in states.values()]
    return robust_limits(arrays, 1, 99)


def redraw_from_cache(out: Path, records, ncomp, clusters, focus_component, focus_group, logger):
    logger("Redrawing publication figures from numerical cache...")
    states = load_all_cached_states(out)
    if not states:
        raise RuntimeError("No layer numerical cache found. A full run is required first.")

    reps = representative_layers(records)
    focus_idx = focus_component - 1
    if not (0 <= focus_idx < ncomp):
        raise ValueError(f"--focus-component must be between 1 and {ncomp}")

    # shared component limits
    comp_limits = {c: component_global_limits(states, c) for c in range(ncomp)}

    # all component representative maps
    for c in range(ncomp):
        gm = {}
        for g in ORDER:
            if g in reps and (g, reps[g]) in states:
                gm[g] = {
                    "layer": reps[g],
                    "map": states[(g, reps[g])]["component_maps"][c],
                }
        if gm:
            plot_rep_maps(
                c+1, gm,
                out / "03_Representative_Maps" / f"RF{c+1:02d}_representative_groups_A.png",
                *comp_limits[c]
            )

    # all-layer focus maps + cluster maps
    for g in ORDER:
        layers = sorted(layer for gg, layer in states if gg == g)
        if not layers:
            continue
        maps = {layer: states[(g, layer)]["component_maps"][focus_idx] for layer in layers}
        plot_all_layers(
            g, focus_component, maps,
            out / "04_Depth_Continuity" / f"{g}_RF{focus_component:02d}_all_layers_A.png",
            *comp_limits[focus_idx]
        )
        clmaps = {layer: states[(g, layer)]["cluster_map"] for layer in layers}
        plot_cluster_layers(
            g, clmaps, clusters,
            out / "05_Clusters" / f"{g}_dominant_clusters_all_layers_A.png"
        )

    # Single representative spectral-cluster maps (reference figure style).
    reps = representative_layers(records)
    for g in ORDER:
        if g in reps and (g, reps[g]) in states:
            plot_representative_cluster_map(
                g, reps[g], states[(g, reps[g])]["cluster_map"], clusters,
                out / "05_Clusters" / f"{g}_representative_spectral_cluster_map_C.png"
            )

    continuity = compute_continuity(states, ncomp)
    continuity.to_csv(out / "04_Depth_Continuity" / "adjacent_layer_continuity.csv",
                      index=False, encoding="utf-8-sig")
    cont_focus = continuity_summary(continuity, list(range(ncomp)))
    cont_focus.to_csv(out / "04_Depth_Continuity" / "adjacent_layer_continuity_mean.csv",
                      encoding="utf-8-sig")
    plot_heatmap(
        cont_focus,
        "Adjacent-layer spatial continuity (mean Spearman ρ)",
        out / "04_Depth_Continuity" / "adjacent_layer_continuity_heatmap_A.png",
        cmap=DIV, vmin=-1, vmax=1, fmt="{:.2f}"
    )

    # publication-focus copy
    pub = out / "07_Publication_Figure2"
    pub.mkdir(parents=True, exist_ok=True)
    plot_workflow(pub / "Fig2A_reference_free_workflow.png")

    # Focus component across groups
    gm = {}
    for g in ORDER:
        if g in reps and (g, reps[g]) in states:
            gm[g] = {"layer": reps[g], "map": states[(g, reps[g])]["component_maps"][focus_idx]}
    plot_rep_maps(
        focus_component, gm,
        pub / f"Fig2D_RF{focus_component:02d}_representative_groups.png",
        *comp_limits[focus_idx]
    )

    # Focus group through depth
    if focus_group in ORDER:
        layers = sorted(layer for gg, layer in states if gg == focus_group)
        if layers:
            maps = {layer: states[(focus_group, layer)]["component_maps"][focus_idx] for layer in layers}
            plot_all_layers(
                focus_group, focus_component, maps,
                pub / f"Fig2E_{focus_group}_RF{focus_component:02d}_all_layers.png",
                *comp_limits[focus_idx]
            )

    plot_heatmap(
        cont_focus,
        "Adjacent-layer spatial continuity (mean Spearman ρ)",
        pub / "Fig2F_adjacent_layer_continuity.png",
        cmap=DIV, vmin=-1, vmax=1, fmt="{:.2f}"
    )

    logger("Cache redraw completed.")


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="V5.2 FINAL no-reference-spectrum volumetric Raman pipeline."
    )
    ap.add_argument("--input", required=True, help="Folder containing raw *_layerN.mat files.")
    ap.add_argument("--output", default="RF_V5_2_FINAL_results", help="Output directory name/path.")
    ap.add_argument("--pixels-per-layer", type=int, default=1000, help="Balanced training pixels sampled per layer.")
    ap.add_argument("--nmf-components", type=int, default=6, help="Number of reference-free NMF components.")
    ap.add_argument("--clusters", type=int, default=10, help="Number of publication spectral clusters.")
    ap.add_argument("--pca-components", type=int, default=15, help="PCA dimensions retained.")
    ap.add_argument("--umap-sample", type=int, default=12000, help="Sample size for 2-D UMAP.")
    ap.add_argument("--tsne-sample", type=int, default=5000, help="Sample size for exploratory t-SNE.")
    ap.add_argument("--step", type=float, default=4.0, help="Common Raman-grid step (cm^-1).")
    ap.add_argument("--focus-component", type=int, default=2, help="RF component highlighted in publication Figure 2.")
    ap.add_argument("--focus-group", default="ACN", choices=ORDER, help="Group highlighted for depth-continuity panel.")
    ap.add_argument("--exploratory", action="store_true", help="Also run t-SNE, UMAP30, cluster masks.")
    ap.add_argument("--html3d", action="store_true", help="Also generate 3-D HTML for focus component.")
    ap.add_argument("--resume", action="store_true", help="Reuse saved global models/cache and continue interrupted mapping.")
    ap.add_argument("--redraw-only", action="store_true", help="Do not reanalyse spectra; redraw A-style maps from cache.")
    ap.add_argument("--skip-umap2", action="store_true", help="Skip 2-D UMAP to shorten runtime.")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    out_arg = Path(args.output)
    out = out_arg.resolve() if out_arg.is_absolute() else (inp / out_arg).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger = Logger(out / "RUN_LOG.txt")

    try:
        records = discover(inp)
        if not records:
            raise RuntimeError(f"No recognized *_layerN.mat files found directly under: {inp}")

        logger("V5.2 FINAL started")
        logger(f"Input: {inp}")
        logger(f"Output: {out}")
        logger(f"Recognized raw layer files: {len(records)}")

        inv = pd.DataFrame([
            {"file": r[0].name, "raw_group": r[1], "group": r[2], "layer": r[3]}
            for r in records
        ])
        inv.to_csv(out / "dataset_inventory.csv", index=False, encoding="utf-8-sig")

        for g in ORDER:
            ls = sorted(inv.loc[inv["group"] == g, "layer"].tolist())
            if ls:
                logger(f"{g}: {len(ls)} layers -> {ls}")

        if args.redraw_only:
            redraw_from_cache(
                out, records, args.nmf_components, args.clusters,
                args.focus_component, args.focus_group, logger
            )
            logger("Finished redraw-only.")
            return

        model_dir = out / "CACHE" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "global_models.joblib"
        plot_cache_file = out / "CACHE" / "global_plot_data.npz"
        meta_file = out / "CACHE" / "metadata.json"

        rng = np.random.default_rng(0)

        # -------------------------------------------------------------
        # Either load saved global models or fit once.
        # -------------------------------------------------------------
        if args.resume and model_file.exists() and meta_file.exists():
            logger("Resume mode: loading saved global models.")
            bundle = joblib.load(model_file)
            pca = bundle["pca"]
            nmf_model = bundle["nmf"]
            km = bundle["km"]
            common_axis = bundle["common_axis"]
            mean_spectra = bundle["mean_spectra"]
            group_abundance_df = bundle["group_abundance_df"]
            cluster_mean_spectra = bundle["cluster_mean_spectra"]
            logger("Global models loaded.")
        else:
            native_samples = []
            native_axes = []
            train_groups = []
            axis_rows = []

            for path, raw, g, layer in records:
                X, axis, ims = load_mat(path)
                Xs, shift = select_fingerprint(X, axis)
                vm = valid_mask(Xs)
                ids = np.where(vm)[0]
                n = min(args.pixels_per_layer, ids.size)
                if n < 50:
                    raise RuntimeError(f"{path.name}: only {n} valid tissue pixels.")
                ids = rng.choice(ids, size=n, replace=False)
                native_samples.append(np.asarray(Xs[ids], dtype=np.float32))
                native_axes.append(np.asarray(shift, dtype=np.float32))
                train_groups.extend([g] * n)
                axis_rows.append({
                    "file": path.name, "group": g, "layer": layer,
                    "native_min": float(shift.min()),
                    "native_max": float(shift.max()),
                    "native_bands": int(shift.size),
                    "sampled_pixels": int(n),
                })
                logger(
                    f"sampled {g} layer {layer}: {n} "
                    f"(native range {shift.min():.2f}-{shift.max():.2f} cm^-1)"
                )

            common_axis, axis_info = common_axis_from_all(
                native_axes, low=600.0, high=1800.0, step=args.step
            )
            logger(
                f"Common Raman grid: {common_axis.size} bands, "
                f"{common_axis.min():.1f}-{common_axis.max():.1f} cm^-1"
            )
            pd.DataFrame(axis_rows).to_csv(out / "native_axis_report.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([axis_info]).to_csv(out / "common_axis_summary.csv", index=False, encoding="utf-8-sig")

            train_blocks = []
            for i, (Xn, ax) in enumerate(zip(native_samples, native_axes), start=1):
                Y = preprocess(resample_spectra(Xn, ax, common_axis))
                train_blocks.append(Y)
                if i % 10 == 0 or i == len(native_samples):
                    logger(f"aligned/preprocessed {i}/{len(native_samples)} layer files")

            Xtrain = np.concatenate(train_blocks, axis=0).astype(np.float32)
            train_groups = np.asarray(train_groups)
            logger(f"Training matrix: {Xtrain.shape}")

            # Mean spectra
            mean_spectra = {}
            for g in ORDER:
                m = train_groups == g
                if np.any(m):
                    mean_spectra[g] = Xtrain[m].mean(axis=0).astype(np.float32)

            # PCA
            npca = min(args.pca_components, Xtrain.shape[1], Xtrain.shape[0]-1)
            pca = PCA(n_components=npca, random_state=0)
            Xpca = pca.fit_transform(Xtrain)

            # NMF on a large balanced subset for runtime stability
            nmf_n = min(50000, Xtrain.shape[0])
            nmf_idx = rng.choice(Xtrain.shape[0], size=nmf_n, replace=False)
            nmf_model = NMF(
                n_components=args.nmf_components,
                init="nndsvda",
                random_state=0,
                max_iter=1000,
                tol=1e-4,
            )
            nmf_model.fit(Xtrain[nmf_idx])
            logger(f"NMF fitted on {nmf_n} sampled spectra")

            H_all = nmf_model.transform(Xtrain)
            group_rows = []
            for g in ORDER:
                m = train_groups == g
                if np.any(m):
                    group_rows.append(np.nanmedian(H_all[m], axis=0))
                else:
                    group_rows.append(np.full(args.nmf_components, np.nan))
            group_abundance_df = pd.DataFrame(
                group_rows, index=ORDER, columns=[f"RF{i+1}" for i in range(args.nmf_components)]
            )

            # Publication clustering in PCA space
            km = MiniBatchKMeans(
                n_clusters=args.clusters,
                random_state=0,
                batch_size=4096,
                n_init=20,
            )
            cl_train = km.fit_predict(Xpca)

            cluster_mean_spectra = np.zeros((args.clusters, Xtrain.shape[1]), dtype=np.float32)
            for c in range(args.clusters):
                m = cl_train == c
                if np.any(m):
                    cluster_mean_spectra[c] = Xtrain[m].mean(axis=0)

            # Global plot data
            pca_plot_idx = rng.choice(Xpca.shape[0], size=min(20000, Xpca.shape[0]), replace=False)
            pca_xy = Xpca[pca_plot_idx, :2].astype(np.float32)
            pca_labels = train_groups[pca_plot_idx]

            umap_xy = np.empty((0, 2), dtype=np.float32)
            umap_labels = np.empty((0,), dtype="<U3")
            if not args.skip_umap2 and umap is not None:
                n_u = min(args.umap_sample, Xpca.shape[0])
                uidx = rng.choice(Xpca.shape[0], size=n_u, replace=False)
                reducer2 = umap.UMAP(
                    n_neighbors=25,
                    min_dist=0.18,
                    n_components=2,
                    random_state=0,
                    transform_seed=0,
                    low_memory=True,
                    n_jobs=1,
                )
                umap_xy = reducer2.fit_transform(Xpca[uidx]).astype(np.float32)
                umap_labels = train_groups[uidx].astype("<U3")
                logger(f"2-D UMAP completed on {n_u} sampled spectra")

            # Save models BEFORE layer mapping, so interrupted mapping can resume.
            bundle = {
                "pca": pca,
                "nmf": nmf_model,
                "km": km,
                "common_axis": common_axis,
                "mean_spectra": mean_spectra,
                "group_abundance_df": group_abundance_df,
                "cluster_mean_spectra": cluster_mean_spectra,
            }
            joblib.dump(bundle, model_file)

            np.savez_compressed(
                plot_cache_file,
                common_axis=common_axis,
                pca_loadings=pca.components_.astype(np.float32),
                pca_evr=pca.explained_variance_ratio_.astype(np.float32),
                pca_xy=pca_xy,
                pca_labels=pca_labels.astype("<U3"),
                umap_xy=umap_xy,
                umap_labels=umap_labels,
                nmf_components=nmf_model.components_.astype(np.float32),
                cluster_mean_spectra=cluster_mean_spectra,
            )
            meta = {
                "nmf_components": args.nmf_components,
                "clusters": args.clusters,
                "pca_components": npca,
                "common_axis_bands": int(common_axis.size),
                "created": datetime.now().isoformat(),
            }
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger("Global models/cache saved before layer mapping.")

            # Global figures now, before layer mapping
            plot_mean_spectra(common_axis, mean_spectra, out / "01_Global" / "mean_raman_spectra_by_group_A.png")
            plot_pca_loadings(
                common_axis, pca.components_, pca.explained_variance_ratio_,
                out / "01_Global" / "reference_free_pca_loadings_A.png"
            )
            plot_group_scatter(
                pca_xy, pca_labels,
                "Integrated raw-spectrum PCA", "PC1", "PC2",
                out / "01_Global" / "integrated_raw_spectrum_PCA_A.png"
            )
            if umap_xy.size:
                plot_group_scatter(
                    umap_xy, umap_labels,
                    "Raw-spectrum UMAP by pathological group", "UMAP1", "UMAP2",
                    out / "01_Global" / "raw_spectrum_UMAP_A.png"
                )
            plot_nmf_spectra(
                common_axis, nmf_model.components_,
                out / "02_ReferenceFree_Components" / "NMF_extracted_spectra_A.png"
            )
            group_abundance_df.to_csv(
                out / "02_ReferenceFree_Components" / "group_component_abundance.csv",
                encoding="utf-8-sig"
            )
            plot_heatmap(
                group_abundance_df,
                "Descriptive abundance of reference-free spectral components",
                out / "02_ReferenceFree_Components" / "group_component_abundance_heatmap_A.png",
                cmap=SEQ
            )
            plot_cluster_mean_spectra(
                common_axis, cluster_mean_spectra,
                out / "05_Clusters" / "cluster_mean_spectra_A.png"
            )

            # Explicitly release big training arrays before full-layer mapping.
            del Xtrain, Xpca, H_all, train_blocks, native_samples, native_axes

        # -------------------------------------------------------------
        # Map all layers, caching each layer numerically.
        # -------------------------------------------------------------
        for path, raw, g, layer in records:
            cache_path = layer_cache_path(out, g, layer)
            if args.resume and cache_path.exists():
                logger(f"cache exists, skipped mapping {g} layer {layer}")
                continue

            X, axis, ims = load_mat(path)
            Xs, shift = select_fingerprint(X, axis)
            vm = valid_mask(Xs)
            if vm.sum() < 50:
                raise RuntimeError(f"{path.name}: too few valid tissue pixels after masking.")

            Xvalid = Xs[vm]
            Y = preprocess(resample_spectra(Xvalid, shift, common_axis))
            pcs = pca.transform(Y)
            comps = nmf_model.transform(Y)
            cl = km.predict(pcs)

            comp_maps = []
            for c in range(args.nmf_components):
                flat = place_valid(comps[:, c], vm, fill=np.nan, dtype=np.float32)
                comp_maps.append(tomap(flat, ims))
            comp_maps = np.asarray(comp_maps, dtype=np.float32)

            cl_flat = place_valid(cl.astype(np.float32), vm, fill=np.nan, dtype=np.float32)
            cl_map = tomap(cl_flat, ims)

            save_layer_cache(cache_path, g, layer, ims, comp_maps, cl_map)
            logger(f"mapped + cached {g} layer {layer}")

        # -------------------------------------------------------------
        # Read cache, compute continuity and generate all A-style maps.
        # -------------------------------------------------------------
        states = load_all_cached_states(out)
        expected = {(r[2], r[3]) for r in records}
        missing = sorted(expected - set(states))
        if missing:
            raise RuntimeError(f"Mapping cache incomplete. Missing layers: {missing[:20]}")

        redraw_from_cache(
            out, records, args.nmf_components, args.clusters,
            args.focus_component, args.focus_group, logger
        )

        # -------------------------------------------------------------
        # Ensure global figures exist in resume mode.
        # -------------------------------------------------------------
        if plot_cache_file.exists():
            z = np.load(plot_cache_file, allow_pickle=False)
            axis0 = z["common_axis"]
            plot_mean_spectra(axis0, mean_spectra, out / "01_Global" / "mean_raman_spectra_by_group_A.png")
            plot_pca_loadings(
                axis0, z["pca_loadings"], z["pca_evr"],
                out / "01_Global" / "reference_free_pca_loadings_A.png"
            )
            plot_group_scatter(
                z["pca_xy"], z["pca_labels"],
                "Integrated raw-spectrum PCA", "PC1", "PC2",
                out / "01_Global" / "integrated_raw_spectrum_PCA_A.png"
            )
            if z["umap_xy"].size:
                plot_group_scatter(
                    z["umap_xy"], z["umap_labels"],
                    "Raw-spectrum UMAP by pathological group", "UMAP1", "UMAP2",
                    out / "01_Global" / "raw_spectrum_UMAP_A.png"
                )
            plot_nmf_spectra(
                axis0, z["nmf_components"],
                out / "02_ReferenceFree_Components" / "NMF_extracted_spectra_A.png"
            )
            plot_cluster_mean_spectra(
                axis0, z["cluster_mean_spectra"],
                out / "05_Clusters" / "cluster_mean_spectra_A.png"
            )

        # Group abundance plots/copies
        group_abundance_df.to_csv(
            out / "02_ReferenceFree_Components" / "group_component_abundance.csv",
            encoding="utf-8-sig"
        )
        plot_heatmap(
            group_abundance_df,
            "Descriptive abundance of reference-free spectral components",
            out / "02_ReferenceFree_Components" / "group_component_abundance_heatmap_A.png",
            cmap=SEQ
        )

        pub = out / "07_Publication_Figure2"
        pub.mkdir(parents=True, exist_ok=True)
        plot_nmf_spectra(
            common_axis, nmf_model.components_,
            pub / "Fig2B_reference_free_NMF_spectra.png"
        )
        plot_heatmap(
            group_abundance_df,
            "Descriptive abundance of reference-free spectral components",
            pub / "Fig2C_group_component_abundance.png",
            cmap=SEQ
        )

        # -------------------------------------------------------------
        # Optional exploratory analyses.
        # -------------------------------------------------------------
        if args.exploratory:
            logger("Exploratory mode: starting cluster-colored UMAP / t-SNE / UMAP30 / cluster maps.")
            # Recreate training PCA sample from models is not possible without Xtrain,
            # so use cached representative-layer spectra for UMAP30 and a balanced
            # re-sample from raw layers.
            ex_blocks, ex_groups = [], []
            for path, raw, g, layer in records:
                X, axis, ims = load_mat(path)
                Xs, shift = select_fingerprint(X, axis)
                vm = valid_mask(Xs)
                ids = np.where(vm)[0]
                n = min(250, ids.size)
                ids = rng.choice(ids, size=n, replace=False)
                Y = preprocess(resample_spectra(Xs[ids], shift, common_axis))
                ex_blocks.append(pca.transform(Y))
                ex_groups.extend([g] * n)
            Xex = np.concatenate(ex_blocks, axis=0).astype(np.float32)

            # 2-D UMAP projection colored by the same spectral-cluster labels.
            if umap is not None:
                n_uc = min(args.umap_sample, Xex.shape[0])
                ucidx = rng.choice(Xex.shape[0], size=n_uc, replace=False)
                reducer_cluster2 = umap.UMAP(
                    n_neighbors=25, min_dist=0.18, n_components=2,
                    random_state=0, transform_seed=0, low_memory=True, n_jobs=1
                )
                uc_xy = reducer_cluster2.fit_transform(Xex[ucidx])
                uc_labels = km.predict(Xex[ucidx])
                plot_umap_cluster_projection(
                    uc_xy, uc_labels, args.clusters,
                    out / "08_Exploratory" / "UMAP_projection_spectral_clusters_C.png"
                )
                logger(f"cluster-colored 2-D UMAP completed on {n_uc} spectra")
            else:
                uc_xy, uc_labels = None, None

            n_ts = min(args.tsne_sample, Xex.shape[0])
            tidx = rng.choice(Xex.shape[0], size=n_ts, replace=False)
            ts = TSNE(
                n_components=2, perplexity=35, init="pca",
                learning_rate="auto", random_state=0, max_iter=1000
            )
            txy = ts.fit_transform(Xex[tidx])
            tlab = km.predict(Xex[tidx])
            plot_tsne(txy, tlab, args.clusters, out / "08_Exploratory" / "tSNE_clusters_C.png")
            logger(f"t-SNE completed on {n_ts} spectra")

            if umap is not None:
                n_u = min(args.umap_sample, Xex.shape[0])
                uidx = rng.choice(Xex.shape[0], size=n_u, replace=False)
                reducer30 = umap.UMAP(
                    n_neighbors=25, min_dist=0.18, n_components=30,
                    random_state=0, transform_seed=0, low_memory=True, n_jobs=1
                )
                reducer30.fit(Xex[uidx])
                logger(f"30-D UMAP fitted on {n_u} spectra")

                reps = representative_layers(records)
                for g in ORDER:
                    if g not in reps:
                        continue
                    rec = next(r for r in records if r[2] == g and r[3] == reps[g])
                    X, axis, ims = load_mat(rec[0])
                    Xs, shift = select_fingerprint(X, axis)
                    vm = valid_mask(Xs)
                    Y = preprocess(resample_spectra(Xs[vm], shift, common_axis))
                    pcs = pca.transform(Y)
                    emb = reducer30.transform(pcs)
                    maps30 = []
                    for d in range(30):
                        flat = place_valid(emb[:, d], vm, fill=np.nan, dtype=np.float32)
                        maps30.append(tomap(flat, ims))
                    maps30 = np.asarray(maps30, dtype=np.float32)
                    plot_umap30(
                        g, reps[g], maps30,
                        out / "08_Exploratory" / f"{g}_UMAP30_layer{reps[g]}_C.png"
                    )
                    st = states[(g, reps[g])]
                    plot_cluster_masks(
                        g, reps[g], st["cluster_map"], args.clusters,
                        out / "08_Exploratory" / f"{g}_cluster_masks_layer{reps[g]}_C.png"
                    )

                    if g == args.focus_group and uc_xy is not None:
                        plot_cluster_summary_triptych(
                            g, reps[g], st["cluster_map"], uc_xy, uc_labels,
                            common_axis, cluster_mean_spectra, args.clusters,
                            out / "08_Exploratory" / f"{g}_spectral_cluster_summary_triptych_C.png"
                        )

                    logger(f"saved exploratory maps for {g}")

        # -------------------------------------------------------------
        # Optional 3-D HTML for focus component.
        # -------------------------------------------------------------
        if args.html3d and go is not None:
            c = args.focus_component - 1
            for g in ORDER:
                layers = sorted(layer for gg, layer in states if gg == g)
                xs, ys, zs, vs = [], [], [], []
                for layer in layers:
                    arr = states[(g, layer)]["component_maps"][c]
                    finite = np.isfinite(arr)
                    if not finite.any():
                        continue
                    thr = np.nanpercentile(arr[finite], 90)
                    yy, xx = np.where(finite & (arr >= thr))
                    vv = arr[yy, xx]
                    if vv.size > 1800:
                        idx = np.linspace(0, vv.size - 1, 1800, dtype=int)
                        yy, xx, vv = yy[idx], xx[idx], vv[idx]
                    xs.append(xx)
                    ys.append(yy)
                    zs.append(np.full(xx.shape, layer))
                    vs.append(vv)
                if xs:
                    fig = go.Figure(data=[go.Scatter3d(
                        x=np.concatenate(xs),
                        y=np.concatenate(ys),
                        z=np.concatenate(zs),
                        mode="markers",
                        marker=dict(
                            size=2,
                            opacity=0.45,
                            color=np.concatenate(vs),
                            colorscale=[
                                [0.00, "#091A33"],
                                [0.25, "#173C64"],
                                [0.50, "#1E7F91"],
                                [0.70, "#49C0B3"],
                                [0.88, "#E68D88"],
                                [1.00, "#FFD0B8"],
                            ],
                            showscale=True,
                        )
                    )])
                    fig.update_layout(
                        title=f"{g}: RF{args.focus_component} 3D reconstruction",
                        scene=dict(
                            xaxis_title="X pixel",
                            yaxis_title="Y pixel",
                            zaxis_title="Optical layer",
                            aspectmode="data",
                        )
                    )
                    hp = out / "09_3D_HTML" / f"{g}_RF{args.focus_component:02d}_3D.html"
                    hp.parent.mkdir(parents=True, exist_ok=True)
                    fig.write_html(hp, include_plotlyjs="cdn")
                    logger(f"saved 3D HTML {g}")

        # -------------------------------------------------------------
        # Final summary
        # -------------------------------------------------------------
        summary = {
            "version": "V5.3C FINAL",
            "reference_spectra_used": False,
            "recognized_layer_files": len(records),
            "groups": [g for g in ORDER if any(r[2] == g for r in records)],
            "focus_component": f"RF{args.focus_component}",
            "focus_group": args.focus_group,
            "cache_enabled": True,
            "resume_supported": True,
            "redraw_only_supported": True,
            "exploratory_outputs_default": False,
            "interpretation": (
                "RF components are reference-free spectral states. "
                "They should not be assigned to specific molecular identities without external evidence."
            ),
        }
        (out / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        logger("FINISHED SUCCESSFULLY")
        logger(f"Results: {out}")

    except Exception as e:
        logger(f"FATAL ERROR: {type(e).__name__}: {e}")
        with (out / "ERROR_TRACEBACK.txt").open("w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        logger("Full traceback saved to ERROR_TRACEBACK.txt")
        raise


if __name__ == "__main__":
    main()
