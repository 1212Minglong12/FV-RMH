# -*- coding: utf-8 -*-
"""
Raman 2D reference-free mapping V1.1 FINAL
========================================

Designed for ONE-LAYER-PER-SAMPLE Raman hyperspectral data.

Example input folder:
C:/Users/mingl/Desktop/lungcancer/2d

Recognized filename examples:
High-ChenZhiFang_layer1.mat
High-HeJinQuan_layer1.mat
Middle-CaiShuYing_layer1.mat
Low-ChenYangSong_layer1.mat
Normal-SomeName_layer1.mat   # optional

Core design
-----------
- Every subject/sample contributes the same maximum number of pixels to model fitting.
- ONE common PCA/NMF/cluster model is fitted across all recognized 2D samples.
- The SAME model is mapped back to every valid pixel of every sample.
- Statistical comparisons are performed at SAMPLE level, not pixel level.
- Numerical map caches are saved so colors/layout can be redrawn without rerunning analysis.
- No purified molecular reference spectra are required.

Interpretation
--------------
RF1, RF2, ... are reference-free Raman spectral states/components.
They are NOT specific molecular identities unless independently validated.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.ndimage import minimum_filter1d, gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import kruskal, mannwhitneyu

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec

from sklearn.decomposition import PCA, NMF
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap.umap_ as umap
except Exception:
    umap = None


# =====================================================================
# Labels / filenames
# =====================================================================
GROUP_ORDER = ["NAT", "WD", "MD", "PD"]
KNOWN_GROUPS = set(GROUP_ORDER)

# Article-consistent mapping for the 2D validation cohort:
# Normal -> NAT (tumour-adjacent reference tissue)
# High   -> WD  (well differentiated)
# Middle -> MD  (moderately differentiated)
# Low    -> PD  (poorly differentiated)
RAW_TO_ARTICLE_GROUP = {
    "normal": "NAT",
    "high": "WD",
    "middle": "MD",
    "low": "PD",
}
ARTICLE_GROUP_LABEL = {
    "NAT": "NAT",
    "WD": "Well differentiated",
    "MD": "Moderately differentiated",
    "PD": "Poorly differentiated",
}

FILE_RE = re.compile(
    r"^(Normal|High|Middle|Low)[-_](.+?)[_-]layer(\d+)\.mat$",
    re.I
)


# =====================================================================
# Palettes
# =====================================================================
def make_seq_cmap(style="A"):
    style = style.upper()
    palettes = {
        # V5.2 FINAL-like: deep navy -> teal -> pale gold
        "A": ["#081A3A", "#123A64", "#176B83", "#159B9C",
              "#50BFA8", "#B7D99A", "#F0CF71"],
        # Midnight -> cyan/teal -> coral/peach
        "C": ["#091A33", "#173C64", "#1E7F91", "#49C0B3",
              "#E68D88", "#FFD0B8"],
    }
    colors = palettes.get(style, palettes["A"])
    cm = LinearSegmentedColormap.from_list(f"seq_{style}", colors, N=256)
    cm.set_bad("#FFFFFF")
    return cm


def make_div_cmap(style="A"):
    style = style.upper()
    palettes = {
        "A": ["#123A64", "#2B7F91", "#75C7BC", "#F7F6EF",
              "#D9D692", "#C29A4C", "#8B5B1F"],
        "C": ["#173C64", "#49C0B3", "#F7F6F3",
              "#F2B2A5", "#E68D88", "#B55263"],
    }
    colors = palettes.get(style, palettes["A"])
    cm = LinearSegmentedColormap.from_list(f"div_{style}", colors, N=256)
    cm.set_bad("#FFFFFF")
    return cm


def group_colors(style="A"):
    if style.upper() == "C":
        return {
            "NAT": "#444444",
            "WD": "#173C64",
            "MD": "#49C0B3",
            "PD": "#E68D88",
        }
    return {
        "NAT": "#555555",
        "WD": "#176B83",
        "MD": "#50BFA8",
        "PD": "#C29A4C",
    }


def categorical_colors(n, style="A"):
    if style.upper() == "C":
        base = [
            "#091A33", "#173C64", "#1E7F91", "#2D9AA4", "#49C0B3",
            "#78D2C6", "#A4DDD4", "#D5E7E2", "#F0B8AE", "#E68D88",
            "#CF6876", "#B55263", "#8C4758", "#6B3F50", "#493949",
        ]
    else:
        base = [
            "#081A3A", "#143B61", "#1A5578", "#1B708A", "#168B95",
            "#199E9A", "#35B09E", "#58C0A1", "#7BCDA0", "#9DD69B",
            "#BDDA91", "#D7D886", "#E7CF77", "#D19846", "#A66E37",
        ]
    if n <= len(base):
        return base[:n]
    reps = math.ceil(n / len(base))
    return (base * reps)[:n]


# =====================================================================
# Logging
# =====================================================================
class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# =====================================================================
# Discovery / MAT loading
# =====================================================================
def discover(folder: Path):
    rows = []
    for p in folder.glob("*.mat"):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        raw_group = m.group(1).lower()
        group = RAW_TO_ARTICLE_GROUP[raw_group]
        subject = m.group(2)
        layer = int(m.group(3))
        rows.append({
            "path": p,
            "file": p.name,
            "group": group,
            "raw_group": m.group(1).capitalize(),
            "subject": subject,
            "layer": layer,
            "sample_id": f"{group}-{subject}",
        })

    available_groups = [g for g in GROUP_ORDER if any(r["group"] == g for r in rows)]
    order_map = {g: i for i, g in enumerate(available_groups)}
    rows.sort(key=lambda r: (order_map.get(r["group"], 99), r["subject"].lower(), r["layer"]))
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
            f"{path.name}: appears to be MATLAB v7.3/HDF5. "
            f"Please send one such file so a dedicated HDF5 loader can be added."
        ) from e

    if "data_struct" not in m:
        # Conservative fallback: find the first struct-like object with data/imagesize.
        ds = None
        for key, obj in m.items():
            if key.startswith("__"):
                continue
            if hasattr(obj, "data") and hasattr(obj, "imagesize"):
                ds = obj
                break
        if ds is None:
            raise ValueError(f"{path.name}: variable 'data_struct' not found.")
    else:
        ds = m["data_struct"]

    X = np.asarray(ds.data, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{path.name}: data must be 2-D [pixels x bands], got {X.shape}.")

    numeric = []
    collect_numeric(ds.axisscale, numeric)
    axis = None
    for a in numeric:
        x = np.asarray(a).squeeze()
        if x.ndim == 1 and x.size == X.shape[1]:
            axis = x.astype(np.float32)
            break
    if axis is None:
        raise ValueError(f"{path.name}: Raman axis not found in data_struct.axisscale.")

    ims_raw = np.asarray(ds.imagesize).squeeze()
    if ims_raw.size < 2:
        raise ValueError(f"{path.name}: imagesize is invalid.")
    ims = (int(ims_raw.flat[0]), int(ims_raw.flat[1]))
    if ims[0] * ims[1] != X.shape[0]:
        raise ValueError(
            f"{path.name}: imagesize {ims} gives {ims[0]*ims[1]} pixels, "
            f"but data has {X.shape[0]} spectra."
        )

    return X, axis, ims


# =====================================================================
# Spectral preprocessing
# =====================================================================
def select_region(X, axis, low=600.0, high=1800.0):
    mask = (axis >= low) & (axis <= high)
    if mask.sum() < 20:
        raise ValueError(
            f"Too few Raman bands inside {low:.0f}-{high:.0f} cm^-1."
        )
    return X[:, mask], axis[mask]


def make_common_axis(source_axes, low=600.0, high=1800.0, step=4.0):
    mins, maxs = [], []
    for ax in source_axes:
        ax = np.asarray(ax).ravel()
        seg = ax[(ax >= low) & (ax <= high)]
        if seg.size < 2:
            raise ValueError("At least one sample does not adequately cover the fingerprint region.")
        mins.append(float(seg.min()))
        maxs.append(float(seg.max()))

    overlap_min = max(mins)
    overlap_max = min(maxs)
    start = math.ceil(overlap_min / step) * step
    stop = math.floor(overlap_max / step) * step

    if stop <= start:
        raise RuntimeError(
            f"No common Raman overlap remains. Shared range: "
            f"{overlap_min:.2f}-{overlap_max:.2f} cm^-1."
        )

    grid = np.arange(start, stop + 0.5*step, step, dtype=np.float32)
    info = {
        "requested_min_cm-1": low,
        "requested_max_cm-1": high,
        "overlap_min_cm-1": overlap_min,
        "overlap_max_cm-1": overlap_max,
        "grid_min_cm-1": float(grid.min()),
        "grid_max_cm-1": float(grid.max()),
        "grid_step_cm-1": float(step),
        "grid_bands": int(grid.size),
    }
    return grid, info


def resample_spectra(X, source_axis, target_axis):
    source_axis = np.asarray(source_axis, dtype=np.float32).ravel()
    target_axis = np.asarray(target_axis, dtype=np.float32).ravel()
    X = np.asarray(X, dtype=np.float32)

    if source_axis[0] > source_axis[-1]:
        source_axis = source_axis[::-1]
        X = X[:, ::-1]

    if target_axis.min() < source_axis.min()-0.5 or target_axis.max() > source_axis.max()+0.5:
        raise ValueError(
            f"Source Raman range {source_axis.min():.2f}-{source_axis.max():.2f} "
            f"does not cover common grid {target_axis.min():.2f}-{target_axis.max():.2f}."
        )

    f = interp1d(
        source_axis, X, axis=1, kind="linear",
        bounds_error=False, fill_value="extrapolate",
        assume_sorted=True
    )
    return np.asarray(f(target_axis), dtype=np.float32)


def preprocess(X):
    """
    Same analysis logic as the successful 3D V5.2-style pipeline:
    1. Savitzky-Golay smoothing
    2. smooth minimum-envelope baseline correction
    3. non-negative clipping
    4. area normalization
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


def valid_mask(X, q=0.05):
    """
    Conservative per-sample tissue mask.
    q=0.05 removes only the weakest ~5% tails for both mean intensity and variance.
    """
    X = np.asarray(X, dtype=np.float32)
    inten = np.nanmean(np.abs(X), axis=1)
    var = np.nanstd(X, axis=1)

    finite = np.isfinite(inten) & np.isfinite(var)
    if finite.sum() < 50:
        return finite

    iq = np.nanquantile(inten[finite], q)
    vq = np.nanquantile(var[finite], q)
    return finite & (inten > iq) & (var > vq)


def tomap(v, ims, order="F"):
    return np.asarray(v).reshape(ims, order=order)


def place_valid(values, valid_flat, fill=np.nan, dtype=np.float32):
    out = np.full(valid_flat.size, fill, dtype=dtype)
    out[np.where(valid_flat)[0]] = np.asarray(values, dtype=dtype)
    return out


def robust_limits(arrays, lo=1, hi=99):
    vals = []
    for a in arrays:
        x = np.asarray(a).ravel()
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)
    if not vals:
        return 0.0, 1.0
    x = np.concatenate(vals)
    vmin = float(np.percentile(x, lo))
    vmax = float(np.percentile(x, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(x))
        vmax = float(np.nanmax(x))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return vmin, vmax


# =====================================================================
# Statistics
# =====================================================================
def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return q
    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m+1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    back = np.empty_like(adj)
    back[order] = adj
    q[idx] = back
    return q


def sample_level_stats(sample_df, groups_present, ncomp):
    omnibus_rows = []
    pair_rows = []

    for c in range(1, ncomp+1):
        col = f"RF{c}"
        arrays = [
            sample_df.loc[sample_df["group"] == g, col].dropna().values
            for g in groups_present
        ]
        valid_arrays = [a for a in arrays if len(a) > 0]

        if len(valid_arrays) >= 2:
            try:
                H, p = kruskal(*valid_arrays)
            except ValueError:
                H, p = np.nan, np.nan
        else:
            H, p = np.nan, np.nan

        omnibus_rows.append({
            "component": col,
            "test": "Kruskal-Wallis",
            "H": H,
            "p": p,
        })

        for i in range(len(groups_present)):
            for j in range(i+1, len(groups_present)):
                g1, g2 = groups_present[i], groups_present[j]
                a = sample_df.loc[sample_df["group"] == g1, col].dropna().values
                b = sample_df.loc[sample_df["group"] == g2, col].dropna().values
                if len(a) >= 2 and len(b) >= 2:
                    try:
                        U, p2 = mannwhitneyu(a, b, alternative="two-sided")
                    except ValueError:
                        U, p2 = np.nan, np.nan
                else:
                    U, p2 = np.nan, np.nan
                pair_rows.append({
                    "component": col,
                    "group1": g1,
                    "group2": g2,
                    "U": U,
                    "p": p2,
                })

    omni = pd.DataFrame(omnibus_rows)
    omni["q_BH"] = bh_fdr(omni["p"].values)

    pair = pd.DataFrame(pair_rows)
    if len(pair):
        pair["q_BH_within_all_pairs"] = bh_fdr(pair["p"].values)
    return omni, pair


# =====================================================================
# Plot helpers
# =====================================================================
def savefig(path: Path, dpi=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_mean_spectra(axis0, mean_spectra, groups_present, colors, outpath):
    plt.figure(figsize=(10, 6))
    for g in groups_present:
        y = mean_spectra[g]
        y = y / max(np.max(y), 1e-12)
        plt.plot(axis0, y, lw=2.1, label=g, color=colors[g])
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Normalized mean intensity")
    plt.title("Mean Raman spectra across 2D validation groups")
    plt.legend(frameon=False)
    plt.tight_layout()
    savefig(outpath)


def plot_pca_loadings(axis0, pca, outpath):
    plt.figure(figsize=(10, 6))
    n = min(5, pca.components_.shape[0])
    for i in range(n):
        plt.plot(
            axis0, pca.components_[i], lw=2,
            label=f"PC{i+1} ({100*pca.explained_variance_ratio_[i]:.1f}%)"
        )
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("PCA loading")
    plt.title("Reference-free PCA spectral loadings")
    plt.legend(frameon=False)
    plt.tight_layout()
    savefig(outpath)


def plot_group_scatter(xy, labels, groups_present, colors, title, xlabel, ylabel, outpath):
    plt.figure(figsize=(8, 7))
    for g in groups_present:
        m = labels == g
        if np.any(m):
            plt.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.35,
                        color=colors[g], label=g, linewidths=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False, markerscale=2)
    plt.tight_layout()
    savefig(outpath)


def plot_nmf_spectra(axis0, components, style, outpath):
    cols = categorical_colors(components.shape[0], style)
    plt.figure(figsize=(10, 6))
    for i in range(components.shape[0]):
        y = components[i] / max(np.max(components[i]), 1e-12)
        plt.plot(axis0, y, lw=2.1, color=cols[i], label=f"RF{i+1}")
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Normalized intensity")
    plt.title("Reference-free extracted Raman components")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    savefig(outpath)


def plot_heatmap(df, title, cmap, outpath, annotate=False):
    arr = df.values.astype(float)
    fig, ax = plt.subplots(figsize=(max(7.2, 1.0*df.shape[1]+3), max(4.5, 0.45*df.shape[0]+2)))
    im = ax.imshow(arr, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    if annotate:
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                if np.isfinite(arr[r, c]):
                    ax.text(c, r, f"{arr[r,c]:.3g}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    savefig(outpath)


def plot_component_boxplots(sample_df, groups_present, ncomp, colors, outpath):
    ncols = 3
    nrows = math.ceil(ncomp / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows))
    axes = np.atleast_1d(axes).ravel()

    rng = np.random.default_rng(1)
    for c in range(1, ncomp+1):
        ax = axes[c-1]
        data = [sample_df.loc[sample_df["group"] == g, f"RF{c}"].dropna().values for g in groups_present]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, g in zip(bp["boxes"], groups_present):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.35)

        for xidx, (vals, g) in enumerate(zip(data, groups_present), start=1):
            if len(vals):
                jitter = rng.normal(0, 0.045, size=len(vals))
                ax.scatter(np.full(len(vals), xidx)+jitter, vals, s=24,
                           color=colors[g], alpha=0.85, linewidths=0)

        ax.set_xticks(np.arange(1, len(groups_present)+1))
        ax.set_xticklabels(groups_present, rotation=20)
        ax.set_title(f"RF{c}")
        ax.set_ylabel("Sample-level median component score")

    for ax in axes[ncomp:]:
        ax.axis("off")

    fig.suptitle("Sample-level reference-free component abundance", fontsize=14)
    fig.tight_layout()
    savefig(outpath)


def plot_one_sample_component_grid(sample_id, maps, cmap, outpath, limits):
    ncomp = maps.shape[0]
    ncols = 3
    nrows = math.ceil(ncomp/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.7*nrows))
    axes = np.atleast_1d(axes).ravel()
    for c in range(ncomp):
        vmin, vmax = limits[c]
        im = axes[c].imshow(maps[c], cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        axes[c].set_title(f"RF{c+1}")
        axes[c].axis("off")
        fig.colorbar(im, ax=axes[c], fraction=0.04, pad=0.02)
    for ax in axes[ncomp:]:
        ax.axis("off")
    fig.suptitle(f"{sample_id}: reference-free spatial maps", fontsize=14)
    fig.tight_layout()
    savefig(outpath)


def plot_representative_groups(focus_component, reps, states, cmap, limit, outpath):
    groups = [g for g in GROUP_ORDER if g in reps]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2*len(groups), 3.7))
    axes = np.atleast_1d(axes)
    vmin, vmax = limit
    for ax, g in zip(axes, groups):
        sid = reps[g]
        arr = states[sid]["component_maps"][focus_component-1]
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"{g}\n{ARTICLE_GROUP_LABEL[g]}\n{states[sid]['subject']}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"RF{focus_component}: automatically selected representative 2D samples", fontsize=14)
    cax = fig.add_axes([0.92, 0.18, 0.014, 0.60])
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin, vmax)
    fig.colorbar(sm, cax=cax)
    fig.subplots_adjust(right=0.90, wspace=0.08)
    savefig(outpath)


def plot_cluster_map(sample_id, clmap, clusters, style, outpath):
    cm = ListedColormap(categorical_colors(clusters, style))
    cm.set_bad("#FFFFFF")
    fig, ax = plt.subplots(figsize=(6.5, 5.3))
    im = ax.imshow(clmap, cmap=cm, vmin=-0.5, vmax=clusters-0.5, aspect="equal")
    ax.set_title(f"{sample_id}: Reference-free Spectral Cluster Map")
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, ticks=np.arange(clusters))
    cb.set_label("Spectral cluster")
    fig.tight_layout()
    savefig(outpath)


def plot_cluster_mean_spectra(axis0, means, style, outpath):
    cols = categorical_colors(means.shape[0], style)
    plt.figure(figsize=(10.5, 6.5))
    for i in range(means.shape[0]):
        y = means[i] / max(np.max(means[i]), 1e-12)
        plt.plot(axis0, y, lw=1.7, color=cols[i], label=f"Cluster {i}")
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Normalized mean intensity")
    plt.title("Mean Raman spectra of reference-free spectral clusters")
    plt.legend(frameon=False, ncol=2 if means.shape[0] <= 12 else 3, fontsize=8)
    plt.tight_layout()
    savefig(outpath)


def plot_workflow(outpath):
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    boxes = [
        ("Raw 2D Raman\none layer / subject", 0.04),
        ("Common Raman grid\n+ preprocessing", 0.25),
        ("One common\nPCA / NMF / clustering model", 0.49),
        ("Map back to every\nsample pixel", 0.73),
        ("Sample-level\nstatistics", 0.92),
    ]
    for text, x in boxes:
        ax.text(
            x, 0.5, text, ha="center", va="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#EEF6F2",
                      edgecolor="#176B83", linewidth=1.4)
        )
    for x1, x2 in [(0.13,0.17),(0.36,0.40),(0.60,0.64),(0.82,0.85)]:
        ax.annotate("", xy=(x2,0.5), xytext=(x1,0.5),
                    arrowprops=dict(arrowstyle="->", color="#123A64", lw=1.7))
    ax.text(0.5, 0.08, "No purified molecular reference spectra are supplied",
            ha="center", fontsize=10, color="#8B5B1F")
    savefig(outpath)


# =====================================================================
# Caching
# =====================================================================
def sample_cache_path(out, sample_id):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    return out / "CACHE" / "samples" / f"{safe}.npz"


def save_sample_cache(path, meta, component_maps, cluster_map, component_summary, cluster_props):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        group=np.array(meta["group"]),
        subject=np.array(meta["subject"]),
        sample_id=np.array(meta["sample_id"]),
        source_file=np.array(meta["file"]),
        imagesize=np.asarray(meta["ims"], dtype=np.int32),
        component_maps=np.asarray(component_maps, dtype=np.float32),
        cluster_map=np.asarray(cluster_map, dtype=np.float32),
        component_summary=np.asarray(component_summary, dtype=np.float32),
        cluster_props=np.asarray(cluster_props, dtype=np.float32),
    )


def load_sample_cache(path):
    z = np.load(path, allow_pickle=False)
    return {
        "group": str(z["group"].item()),
        "subject": str(z["subject"].item()),
        "sample_id": str(z["sample_id"].item()),
        "file": str(z["source_file"].item()),
        "ims": tuple(int(x) for x in z["imagesize"].ravel()[:2]),
        "component_maps": np.asarray(z["component_maps"], dtype=np.float32),
        "cluster_map": np.asarray(z["cluster_map"], dtype=np.float32),
        "component_summary": np.asarray(z["component_summary"], dtype=np.float32),
        "cluster_props": np.asarray(z["cluster_props"], dtype=np.float32),
    }


def load_all_states(out):
    states = {}
    for p in sorted((out/"CACHE"/"samples").glob("*.npz")):
        st = load_sample_cache(p)
        states[st["sample_id"]] = st
    return states


# =====================================================================
# Representative sample selection
# =====================================================================
def choose_representatives(sample_df, groups_present, ncomp):
    """
    Automatic, non-cherry-picked representative:
    sample whose standardized RF abundance vector is closest to the group median.
    """
    reps = {}
    rfcols = [f"RF{i}" for i in range(1, ncomp+1)]

    Xall = sample_df[rfcols].values.astype(float)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(Xall)
    zdf = pd.DataFrame(Xz, index=sample_df.index, columns=rfcols)

    for g in groups_present:
        idx = sample_df.index[sample_df["group"] == g].tolist()
        if not idx:
            continue
        G = zdf.loc[idx].values
        target = np.nanmedian(G, axis=0)
        d = np.linalg.norm(G-target, axis=1)
        reps[g] = sample_df.loc[idx[int(np.nanargmin(d))], "sample_id"]
    return reps


# =====================================================================
# Redrawing from cache
# =====================================================================
def redraw(out, records, args, logger):
    states = load_all_states(out)
    if not states:
        raise RuntimeError("No numerical sample cache found. Run the full analysis first.")

    style = args.palette.upper()
    seq = make_seq_cmap(style)
    colors = group_colors(style)
    groups_present = [g for g in GROUP_ORDER if any(st["group"] == g for st in states.values())]

    # sample-level tables
    sample_rows = []
    for sid, st in states.items():
        row = {
            "sample_id": sid,
            "group": st["group"],
            "subject": st["subject"],
            "file": st["file"],
        }
        for c in range(args.nmf_components):
            row[f"RF{c+1}"] = float(st["component_summary"][c])
        for k in range(args.clusters):
            row[f"Cluster{k}_fraction"] = float(st["cluster_props"][k])
        sample_rows.append(row)

    sample_df = pd.DataFrame(sample_rows)
    order_map = {g:i for i,g in enumerate(groups_present)}
    sample_df["_ord"] = sample_df["group"].map(order_map)
    sample_df = sample_df.sort_values(["_ord","subject"]).drop(columns="_ord").reset_index(drop=True)
    sample_df.to_csv(out/"02_SampleLevel"/"sample_level_RF_and_cluster_summary.csv",
                     index=False, encoding="utf-8-sig")

    rfcols = [f"RF{i}" for i in range(1, args.nmf_components+1)]

    # group medians, only descriptive
    group_med = sample_df.groupby("group", sort=False)[rfcols].median()
    group_med = group_med.reindex([g for g in groups_present if g in group_med.index])
    group_med.to_csv(out/"02_SampleLevel"/"group_median_RF_abundance.csv", encoding="utf-8-sig")
    plot_heatmap(
        group_med, "Sample-level reference-free components across NAT and differentiation groups",
        seq, out/"02_SampleLevel"/f"group_median_RF_heatmap_{style}.png", annotate=True
    )

    plot_component_boxplots(
        sample_df, groups_present, args.nmf_components, colors,
        out/"02_SampleLevel"/f"sample_level_RF_boxplots_NAT_WD_MD_PD_{style}.png"
    )

    omni, pair = sample_level_stats(sample_df, groups_present, args.nmf_components)
    omni.to_csv(out/"02_SampleLevel"/"RF_KruskalWallis_sample_level.csv",
                index=False, encoding="utf-8-sig")
    pair.to_csv(out/"02_SampleLevel"/"RF_pairwise_MannWhitney_sample_level.csv",
                index=False, encoding="utf-8-sig")

    # component shared limits across all samples
    limits = {}
    for c in range(args.nmf_components):
        limits[c] = robust_limits([st["component_maps"][c] for st in states.values()], 1, 99)

    # individual sample map grids
    for sid, st in states.items():
        plot_one_sample_component_grid(
            sid, st["component_maps"], seq,
            out/"03_Sample_Maps"/st["group"]/f"{sid}_RF_maps_{style}.png",
            limits
        )

    reps = choose_representatives(sample_df, groups_present, args.nmf_components)
    (out/"04_Representative_Maps"/"representative_samples.json").write_text(
        json.dumps(reps, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plot_representative_groups(
        args.focus_component, reps, states, seq, limits[args.focus_component-1],
        out/"04_Representative_Maps"/f"RF{args.focus_component:02d}_representative_groups_{style}.png"
    )

    # cluster maps for automatically selected representative samples
    for g, sid in reps.items():
        st = states[sid]
        plot_cluster_map(
            sid, st["cluster_map"], args.clusters, style,
            out/"05_Clusters"/f"{g}_representative_cluster_map_{style}.png"
        )

    # cluster proportions sample-level heatmap
    cluster_cols = [f"Cluster{k}_fraction" for k in range(args.clusters)]
    cluster_df = sample_df.set_index("sample_id")[cluster_cols]
    plot_heatmap(
        cluster_df, "Sample-level spectral-cluster composition",
        seq, out/"05_Clusters"/f"sample_cluster_fraction_heatmap_{style}.png"
    )

    # publication candidates
    pub = out/"07_Publication_Candidates"
    plot_workflow(pub/"Panel_A_2D_reference_free_workflow.png")

    plot_representative_groups(
        args.focus_component, reps, states, seq, limits[args.focus_component-1],
        pub/f"Panel_D_RF{args.focus_component:02d}_representative_groups_{style}.png"
    )
    plot_component_boxplots(
        sample_df, groups_present, args.nmf_components, colors,
        pub/f"Panel_E_sample_level_RF_boxplots_NAT_WD_MD_PD_{style}.png"
    )

    logger(f"Redraw completed with palette {style}.")
    return sample_df, reps


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Robust 2D one-layer-per-sample reference-free Raman mapping."
    )
    ap.add_argument("--input", required=True, help="Folder containing High-/Middle-/Low- *_layer1.mat files.")
    ap.add_argument("--output", default="RF_2D_V1_1_FINAL_results", help="Output folder name/path.")
    ap.add_argument("--pixels-per-sample", type=int, default=1000,
                    help="Maximum balanced training pixels sampled from each subject.")
    ap.add_argument("--nmf-components", type=int, default=6)
    ap.add_argument("--clusters", type=int, default=10)
    ap.add_argument("--pca-components", type=int, default=15)
    ap.add_argument("--umap-sample", type=int, default=12000)
    ap.add_argument("--step", type=float, default=4.0)
    ap.add_argument("--mask-quantile", type=float, default=0.05)
    ap.add_argument("--focus-component", type=int, default=2)
    ap.add_argument("--palette", choices=["A","C","a","c"], default="A")
    ap.add_argument("--reshape-order", choices=["F","C"], default="F",
                    help="MATLAB pixel-to-image reshape order; default F matches prior successful 3D pipeline.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--redraw-only", action="store_true")
    ap.add_argument("--skip-umap2", action="store_true")
    ap.add_argument("--exploratory", action="store_true",
                    help="Add t-SNE and UMAP30 representative panels. Not required for core paper figures.")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    out_arg = Path(args.output)
    out = out_arg.resolve() if out_arg.is_absolute() else (inp/out_arg).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Create all fixed result folders up front so CSV/text writes cannot fail
    # simply because a parent directory does not yet exist.
    for sub in [
        "00_QC",
        "01_Global",
        "02_SampleLevel",
        "03_Sample_Maps",
        "04_Representative_Maps",
        "05_Clusters",
        "07_Publication_Candidates",
        "08_Exploratory",
        "CACHE/models",
        "CACHE/samples",
    ]:
        (out/sub).mkdir(parents=True, exist_ok=True)

    logger = Logger(out/"RUN_LOG.txt")

    try:
        records = discover(inp)
        if not records:
            raise RuntimeError(
                "No recognized 2D files found.\n"
                "Expected filenames such as High-ChenZhiFang_layer1.mat, "
                "Middle-Name_layer1.mat, Low-Name_layer1.mat."
            )

        # Keep only one layer per sample. If duplicate sample IDs exist, stop rather than silently guessing.
        seen = {}
        dup = []
        for r in records:
            sid = r["sample_id"]
            if sid in seen:
                dup.append((sid, seen[sid]["file"], r["file"]))
            else:
                seen[sid] = r
        if dup:
            msg = "\n".join([f"{a}: {b} AND {c}" for a,b,c in dup[:20]])
            raise RuntimeError(
                "More than one MAT file was found for the same subject/sample. "
                "This 2D pipeline expects one layer per subject.\nDuplicates:\n" + msg
            )

        groups_present = [g for g in GROUP_ORDER if any(r["group"] == g for r in records)]
        logger("Raman 2D reference-free V1 FINAL started.")
        logger(f"Input folder: {inp}")
        logger(f"Recognized samples: {len(records)}")
        for g in groups_present:
            n = sum(r["group"] == g for r in records)
            logger(f"{g} ({ARTICLE_GROUP_LABEL[g]}): {n} samples")

        # The manuscript's 32-specimen 2D validation set is:
        # NAT=5, WD=9, MD=11, PD=7. We report a warning rather than abort,
        # so the same code can still be used if the cohort is later expanded.
        expected_article_counts = {"NAT": 5, "WD": 9, "MD": 11, "PD": 7}
        observed_counts = {g: sum(r["group"] == g for r in records) for g in GROUP_ORDER}
        logger(f"Observed article-group counts: {observed_counts}")
        if observed_counts != expected_article_counts:
            logger(
                "QC WARNING: counts differ from the manuscript 32-specimen validation set "
                f"{expected_article_counts}. Analysis will continue with the detected files."
            )
        else:
            logger("QC PASS: group counts match the manuscript 32-specimen validation set.")

        inv = pd.DataFrame([{
            "file": r["file"],
            "raw_group": r["raw_group"],
            "group": r["group"],
            "article_group": ARTICLE_GROUP_LABEL[r["group"]],
            "subject": r["subject"],
            "sample_id": r["sample_id"],
            "layer": r["layer"],
        } for r in records])
        inv.to_csv(out/"dataset_inventory.csv", index=False, encoding="utf-8-sig")

        model_dir = out/"CACHE"/"models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir/"global_models.joblib"
        plot_cache = out/"CACHE"/"global_plot_data.npz"

        if args.redraw_only:
            redraw(out, records, args, logger)
            logger("FINISHED REDRAW-ONLY.")
            return

        rng = np.random.default_rng(0)
        style = args.palette.upper()
        seq = make_seq_cmap(style)
        colors = group_colors(style)

        # -------------------------------------------------------------
        # Global model fit or resume.
        # -------------------------------------------------------------
        if args.resume and model_file.exists():
            logger("Resume mode: loading global models.")
            bundle = joblib.load(model_file)
            common_axis = bundle["common_axis"]
            pca = bundle["pca"]
            nmf_model = bundle["nmf"]
            km = bundle["km"]
            mean_spectra = bundle["mean_spectra"]
            cluster_mean_spectra = bundle["cluster_mean_spectra"]
            logger("Global models loaded.")
        else:
            native_samples = []
            native_axes = []
            training_groups = []
            training_sample_ids = []
            axis_rows = []

            for r in records:
                X, axis, ims = load_mat(r["path"])
                Xs, shift = select_region(X, axis)
                vm = valid_mask(Xs, q=args.mask_quantile)
                ids = np.where(vm)[0]

                if ids.size < 50:
                    raise RuntimeError(
                        f"{r['file']}: only {ids.size} valid pixels after masking. "
                        f"Try --mask-quantile 0.02 if this is a real tissue scan."
                    )

                n = min(args.pixels_per_sample, ids.size)
                ids = rng.choice(ids, size=n, replace=False)

                native_samples.append(np.asarray(Xs[ids], dtype=np.float32))
                native_axes.append(np.asarray(shift, dtype=np.float32))
                training_groups.extend([r["group"]]*n)
                training_sample_ids.extend([r["sample_id"]]*n)

                axis_rows.append({
                    "file": r["file"],
                    "group": r["group"],
                    "subject": r["subject"],
                    "native_min": float(shift.min()),
                    "native_max": float(shift.max()),
                    "native_bands": int(shift.size),
                    "valid_pixels": int(vm.sum()),
                    "sampled_training_pixels": int(n),
                    "imagesize_row": int(ims[0]),
                    "imagesize_col": int(ims[1]),
                })
                logger(
                    f"sampled {r['sample_id']}: {n} pixels "
                    f"(range {shift.min():.2f}-{shift.max():.2f} cm^-1, image {ims})"
                )

            common_axis, axis_info = make_common_axis(
                native_axes, 600.0, 1800.0, args.step
            )
            logger(
                f"Common Raman grid: {common_axis.size} bands, "
                f"{common_axis.min():.1f}-{common_axis.max():.1f} cm^-1"
            )
            pd.DataFrame(axis_rows).to_csv(
                out/"00_QC"/"sample_axis_and_size_report.csv",
                index=False, encoding="utf-8-sig"
            )
            pd.DataFrame([axis_info]).to_csv(
                out/"00_QC"/"common_axis_summary.csv",
                index=False, encoding="utf-8-sig"
            )

            blocks = []
            for i, (Xn, ax) in enumerate(zip(native_samples, native_axes), start=1):
                Y = preprocess(resample_spectra(Xn, ax, common_axis))
                blocks.append(Y)
                if i % 5 == 0 or i == len(native_samples):
                    logger(f"aligned/preprocessed {i}/{len(native_samples)} samples")

            Xtrain = np.concatenate(blocks, axis=0).astype(np.float32)
            training_groups = np.asarray(training_groups)
            training_sample_ids = np.asarray(training_sample_ids)
            logger(f"Training matrix: {Xtrain.shape}")

            # group mean spectra from balanced training pixels
            mean_spectra = {}
            for g in groups_present:
                m = training_groups == g
                mean_spectra[g] = Xtrain[m].mean(axis=0).astype(np.float32)

            # PCA
            npca = min(args.pca_components, Xtrain.shape[1], Xtrain.shape[0]-1)
            pca = PCA(n_components=npca, random_state=0)
            Xpca = pca.fit_transform(Xtrain)

            # NMF
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
            logger(f"NMF fitted on {nmf_n} spectra")

            # clusters
            km = MiniBatchKMeans(
                n_clusters=args.clusters,
                random_state=0,
                batch_size=4096,
                n_init=20,
            )
            cl_train = km.fit_predict(Xpca)

            cluster_mean_spectra = np.zeros(
                (args.clusters, Xtrain.shape[1]), dtype=np.float32
            )
            for k in range(args.clusters):
                m = cl_train == k
                if np.any(m):
                    cluster_mean_spectra[k] = Xtrain[m].mean(axis=0)

            # plots from sampled training matrix
            scatter_idx = rng.choice(
                Xpca.shape[0], size=min(20000, Xpca.shape[0]), replace=False
            )
            pca_xy = Xpca[scatter_idx, :2].astype(np.float32)
            pca_labels = training_groups[scatter_idx].astype("<U10")

            umap_xy = np.empty((0,2), dtype=np.float32)
            umap_labels = np.empty((0,), dtype="<U10")
            if not args.skip_umap2 and umap is not None:
                n_u = min(args.umap_sample, Xpca.shape[0])
                uidx = rng.choice(Xpca.shape[0], size=n_u, replace=False)
                reducer2 = umap.UMAP(
                    n_neighbors=25, min_dist=0.18, n_components=2,
                    random_state=0, transform_seed=0,
                    low_memory=True, n_jobs=1
                )
                umap_xy = reducer2.fit_transform(Xpca[uidx]).astype(np.float32)
                umap_labels = training_groups[uidx].astype("<U10")
                logger(f"2-D UMAP completed on {n_u} spectra")

            bundle = {
                "common_axis": common_axis,
                "pca": pca,
                "nmf": nmf_model,
                "km": km,
                "mean_spectra": mean_spectra,
                "cluster_mean_spectra": cluster_mean_spectra,
            }
            joblib.dump(bundle, model_file)

            np.savez_compressed(
                plot_cache,
                common_axis=common_axis,
                pca_loadings=pca.components_.astype(np.float32),
                pca_evr=pca.explained_variance_ratio_.astype(np.float32),
                pca_xy=pca_xy,
                pca_labels=pca_labels,
                umap_xy=umap_xy,
                umap_labels=umap_labels,
                nmf_components=nmf_model.components_.astype(np.float32),
                cluster_mean_spectra=cluster_mean_spectra,
            )
            logger("Global model/cache saved before full sample mapping.")

            plot_mean_spectra(
                common_axis, mean_spectra, groups_present, colors,
                out/"01_Global"/f"mean_raman_spectra_by_group_{style}.png"
            )
            plot_pca_loadings(
                common_axis, pca,
                out/"01_Global"/f"PCA_spectral_loadings_{style}.png"
            )
            plot_group_scatter(
                pca_xy, pca_labels, groups_present, colors,
                "Integrated raw-spectrum PCA", "PC1", "PC2",
                out/"01_Global"/f"PCA_embedding_by_group_{style}.png"
            )
            if umap_xy.size:
                plot_group_scatter(
                    umap_xy, umap_labels, groups_present, colors,
                    "Raw-spectrum UMAP across differentiation groups", "UMAP1", "UMAP2",
                    out/"01_Global"/f"UMAP_embedding_by_group_{style}.png"
                )
            plot_nmf_spectra(
                common_axis, nmf_model.components_, style,
                out/"01_Global"/f"NMF_extracted_reference_free_spectra_{style}.png"
            )
            plot_cluster_mean_spectra(
                common_axis, cluster_mean_spectra, style,
                out/"05_Clusters"/f"cluster_mean_spectra_{style}.png"
            )

            del Xtrain, Xpca, blocks, native_samples, native_axes

        # -------------------------------------------------------------
        # Map back to each complete 2D sample.
        # -------------------------------------------------------------
        for r in records:
            cp = sample_cache_path(out, r["sample_id"])
            if args.resume and cp.exists():
                logger(f"cache exists, skipped mapping {r['sample_id']}")
                continue

            X, axis, ims = load_mat(r["path"])
            Xs, shift = select_region(X, axis)
            vm = valid_mask(Xs, q=args.mask_quantile)

            if vm.sum() < 50:
                raise RuntimeError(
                    f"{r['file']}: too few valid pixels in full mapping."
                )

            Y = preprocess(resample_spectra(Xs[vm], shift, common_axis))
            pcs = pca.transform(Y)
            comps = nmf_model.transform(Y)
            cl = km.predict(pcs)

            comp_maps = []
            comp_summary = []
            for c in range(args.nmf_components):
                full = place_valid(comps[:,c], vm, fill=np.nan)
                m2d = tomap(full, ims, order=args.reshape_order)
                comp_maps.append(m2d)
                comp_summary.append(float(np.nanmedian(comps[:,c])))

            cl_full = place_valid(cl.astype(np.float32), vm, fill=np.nan)
            cl_map = tomap(cl_full, ims, order=args.reshape_order)

            props = []
            for k in range(args.clusters):
                props.append(float(np.mean(cl == k)))

            meta = dict(r)
            meta["ims"] = ims
            save_sample_cache(
                cp, meta, np.asarray(comp_maps), cl_map,
                np.asarray(comp_summary), np.asarray(props)
            )
            logger(f"mapped + cached {r['sample_id']}")

        states = load_all_states(out)
        if len(states) != len(records):
            raise RuntimeError(
                f"Cache count {len(states)} != recognized sample count {len(records)}."
            )

        sample_df, reps = redraw(out, records, args, logger)

        # Ensure global plots also exist after resume.
        if plot_cache.exists():
            z = np.load(plot_cache, allow_pickle=False)
            plot_mean_spectra(
                common_axis, mean_spectra, groups_present, colors,
                out/"01_Global"/f"mean_raman_spectra_by_group_{style}.png"
            )
            plot_pca_loadings(
                common_axis, pca,
                out/"01_Global"/f"PCA_spectral_loadings_{style}.png"
            )
            plot_group_scatter(
                z["pca_xy"], z["pca_labels"], groups_present, colors,
                "Integrated raw-spectrum PCA", "PC1", "PC2",
                out/"01_Global"/f"PCA_embedding_by_group_{style}.png"
            )
            if z["umap_xy"].size:
                plot_group_scatter(
                    z["umap_xy"], z["umap_labels"], groups_present, colors,
                    "Raw-spectrum UMAP across differentiation groups", "UMAP1", "UMAP2",
                    out/"01_Global"/f"UMAP_embedding_by_group_{style}.png"
                )
            plot_nmf_spectra(
                common_axis, nmf_model.components_, style,
                out/"01_Global"/f"NMF_extracted_reference_free_spectra_{style}.png"
            )
            plot_cluster_mean_spectra(
                common_axis, cluster_mean_spectra, style,
                out/"05_Clusters"/f"cluster_mean_spectra_{style}.png"
            )

        # Publication copies of global plots
        pub = out/"07_Publication_Candidates"
        plot_nmf_spectra(
            common_axis, nmf_model.components_, style,
            pub/f"Panel_B_NMF_reference_free_spectra_{style}.png"
        )

        rfcols = [f"RF{i}" for i in range(1,args.nmf_components+1)]
        group_med = sample_df.groupby("group", sort=False)[rfcols].median()
        group_med = group_med.reindex(groups_present)
        plot_heatmap(
            group_med, "Sample-level reference-free components across NAT and differentiation groups",
            seq, pub/f"Panel_C_group_median_RF_heatmap_{style}.png", annotate=True
        )

        # -------------------------------------------------------------
        # Optional exploratory figures
        # -------------------------------------------------------------
        if args.exploratory:
            logger("Exploratory mode: t-SNE + representative UMAP30.")

            # Balanced re-sampling from each subject so subjects remain equally weighted.
            ex_blocks = []
            ex_groups = []
            for r in records:
                X, axis, ims = load_mat(r["path"])
                Xs, shift = select_region(X, axis)
                vm = valid_mask(Xs, q=args.mask_quantile)
                ids = np.where(vm)[0]
                n = min(200, ids.size)
                ids = rng.choice(ids, size=n, replace=False)
                Y = preprocess(resample_spectra(Xs[ids], shift, common_axis))
                ex_blocks.append(pca.transform(Y))
                ex_groups.extend([r["group"]]*n)

            Xex = np.concatenate(ex_blocks, axis=0)
            ex_groups = np.asarray(ex_groups)

            n_ts = min(5000, Xex.shape[0])
            tidx = rng.choice(Xex.shape[0], size=n_ts, replace=False)
            ts = TSNE(
                n_components=2, perplexity=35, init="pca",
                learning_rate="auto", random_state=0, max_iter=1000
            )
            txy = ts.fit_transform(Xex[tidx])
            plot_group_scatter(
                txy, ex_groups[tidx], groups_present, colors,
                "t-SNE of balanced 2D Raman spectra", "t-SNE1", "t-SNE2",
                out/"08_Exploratory"/f"tSNE_by_group_{style}.png"
            )
            logger(f"t-SNE completed on {n_ts} spectra")

            if umap is not None:
                n_u = min(args.umap_sample, Xex.shape[0])
                uidx = rng.choice(Xex.shape[0], size=n_u, replace=False)
                reducer30 = umap.UMAP(
                    n_neighbors=25, min_dist=0.18, n_components=30,
                    random_state=0, transform_seed=0,
                    low_memory=True, n_jobs=1
                )
                reducer30.fit(Xex[uidx])
                logger(f"30-D UMAP fitted on {n_u} spectra")

                reps = choose_representatives(sample_df, groups_present, args.nmf_components)
                pretty = LinearSegmentedColormap.from_list(
                    "umap_pretty",
                    ["#080B4A","#093B99","#0078C8","#12BED0",
                     "#72D7E5","#D95CAF","#F5A7D2","#FFD6EA"],
                    N=256
                )
                pretty.set_bad("#F4F4F4")

                for g, sid in reps.items():
                    r = next(rr for rr in records if rr["sample_id"] == sid)
                    X, axis, ims = load_mat(r["path"])
                    Xs, shift = select_region(X, axis)
                    vm = valid_mask(Xs, q=args.mask_quantile)
                    Y = preprocess(resample_spectra(Xs[vm], shift, common_axis))
                    pcs = pca.transform(Y)
                    emb = reducer30.transform(pcs)

                    maps30 = []
                    for d in range(30):
                        full = place_valid(emb[:,d], vm, fill=np.nan)
                        maps30.append(tomap(full, ims, order=args.reshape_order))
                    maps30 = np.asarray(maps30)

                    ncols = 6
                    nrows = 5
                    fig, axes = plt.subplots(nrows,ncols,figsize=(20,15),constrained_layout=True)
                    axes = axes.ravel()
                    for d in range(30):
                        vmin,vmax = robust_limits([maps30[d]],2,98)
                        im = axes[d].imshow(maps30[d],cmap=pretty,vmin=vmin,vmax=vmax,aspect="equal")
                        axes[d].set_title(f"UMAP {d+1}",fontsize=9)
                        axes[d].axis("off")
                        fig.colorbar(im,ax=axes[d],fraction=0.038,pad=0.018)
                    fig.suptitle(f"{sid}: Spatial Distribution of 30 UMAP Dimensions",fontsize=16)
                    savefig(out/"08_Exploratory"/f"{sid}_UMAP30_pretty.png")

        summary = {
            "version": "Raman 2D reference-free V1.1 FINAL",
            "input": str(inp),
            "recognized_samples": len(records),
            "groups": groups_present,
            "article_group_mapping": {
                "Normal": "NAT",
                "High": "WD (well differentiated)",
                "Middle": "MD (moderately differentiated)",
                "Low": "PD (poorly differentiated)"
            },
            "one_layer_per_sample": True,
            "reference_spectra_used": False,
            "sample_level_statistics": True,
            "palette": style,
            "reshape_order": args.reshape_order,
            "focus_component": f"RF{args.focus_component}",
            "note": (
                "RF components are reference-free spectral states. "
                "Inferential statistics use one summary value per subject/sample, not pixels."
            ),
        }
        (out/"RUN_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger("FINISHED SUCCESSFULLY")
        logger(f"Results: {out}")

    except Exception as e:
        logger(f"FATAL ERROR: {type(e).__name__}: {e}")
        with (out/"ERROR_TRACEBACK.txt").open("w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        logger("Full traceback saved to ERROR_TRACEBACK.txt")
        raise


if __name__ == "__main__":
    main()
