from pathlib import Path
import re
import json
import shutil
import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore")

# ============================================================
# Fig05 v3: balanced spacing version
# Fixes:
#   1) Right-side gap reduced to a reasonable size, but still prevents overlap
#   2) Left dendrogram area widened a bit so branch lines do not feel crowded
#   3) Colorbars moved moderately right, not too far
#
# Run:
#   cd FV-RMH
#   python scripts\2d\fig05_data_overview_heatmap_dotplot_v3_balanced_spacing.py --data-dir data_2d
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8

GROUP_ALIASES = {
    "Normal": "Normal",
    "NAT": "Normal",
    "Healthy": "Normal",
    "Control": "Normal",
    "High": "High",
    "WD": "High",
    "Well": "High",
    "WellDifferentiated": "High",
    "Well_Differentiated": "High",
    "Middle": "Middle",
    "Moderate": "Middle",
    "MD": "Middle",
    "Mid": "Middle",
    "ModeratelyDifferentiated": "Middle",
    "Moderately_Differentiated": "Middle",
    "Low": "Low",
    "Poor": "Low",
    "PD": "Low",
    "PoorlyDifferentiated": "Low",
    "Poorly_Differentiated": "Low"
}

GROUP_DISPLAY = {
    "Normal": "Normal",
    "High": "High differentiated",
    "Middle": "Moderately differentiated",
    "Low": "Poorly differentiated"
}

GROUP_COLORS = {
    "Normal": "#84C6A6",
    "High": "#7EB6FF",
    "Middle": "#F2B38F",
    "Low": "#C97C8A"
}


def find_mat_files(data_dir: Path):
    return sorted([p for p in data_dir.rglob("*.mat") if p.is_file()])


def parse_filename(path: Path):
    stem = path.stem
    group_terms = sorted(GROUP_ALIASES.keys(), key=len, reverse=True)
    group_regex = "|".join([re.escape(g) for g in group_terms])
    pattern = rf"^(.+)-({group_regex})-(.+?)_layer(\d+)$"
    m = re.match(pattern, stem)
    if m is None:
        return None
    return {
        "marker": m.group(1),
        "raw_group": m.group(2),
        "group": GROUP_ALIASES.get(m.group(2), m.group(2)),
        "patient": m.group(3),
        "layer": int(m.group(4)),
        "file": str(path)
    }


def load_mat_numeric_array(mat_path: Path):
    try:
        mat = sio.loadmat(mat_path)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if "coeffVector" in mat:
            arr = mat["coeffVector"]
        else:
            numeric_arrays = []
            for k in keys:
                v = mat[k]
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                    numeric_arrays.append((k, v))
            if not numeric_arrays:
                raise ValueError("No numeric array found")
            _, arr = max(numeric_arrays, key=lambda kv: kv[1].size)
        arr = np.asarray(arr).squeeze().astype(float).flatten()
        arr[~np.isfinite(arr)] = np.nan
        return arr
    except NotImplementedError:
        try:
            import h5py
        except ImportError:
            raise ImportError("This may be MATLAB v7.3. Install h5py: python -m pip install h5py")
        with h5py.File(mat_path, "r") as f:
            if "coeffVector" in f:
                arr = np.array(f["coeffVector"])
            else:
                candidates = []
                for k in f.keys():
                    try:
                        arr_tmp = np.array(f[k])
                        if np.issubdtype(arr_tmp.dtype, np.number):
                            candidates.append((k, arr_tmp))
                    except Exception:
                        pass
                if not candidates:
                    raise ValueError("No numeric array found in v7.3 mat")
                _, arr = max(candidates, key=lambda kv: kv[1].size)
        arr = np.asarray(arr).squeeze().astype(float).flatten()
        arr[~np.isfinite(arr)] = np.nan
        return arr


def robust_clip_preserve_shape(x, lower_q=0.005, upper_q=0.995):
    x = np.asarray(x, dtype=float).copy()
    valid = np.isfinite(x)
    if valid.sum() == 0:
        return x
    lo = np.nanquantile(x[valid], lower_q)
    hi = np.nanquantile(x[valid], upper_q)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        x[valid] = np.clip(x[valid], lo, hi)
    x[~valid] = np.nan
    return x


def fill_nonfinite_matrix(X):
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = np.nan
    if X.size == 0:
        return X
    col_medians = np.nanmedian(X, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
    inds = np.where(~np.isfinite(X))
    if len(inds[0]) > 0:
        X[inds] = np.take(col_medians, inds[1])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def build_sample_marker_dict(meta_df: pd.DataFrame, target_layer=1, allowed_groups=None):
    sample_dict = {}
    recognized = []
    skipped = []
    for _, row in meta_df.iterrows():
        if allowed_groups is not None and row["group"] not in allowed_groups:
            skipped.append({**row.to_dict(), "reason": "group_filtered"})
            continue
        if target_layer is not None and int(row["layer"]) != int(target_layer):
            skipped.append({**row.to_dict(), "reason": "layer_filtered"})
            continue
        try:
            vec = load_mat_numeric_array(Path(row["file"]))
        except Exception as e:
            skipped.append({**row.to_dict(), "reason": f"load_failed:{e}"})
            continue
        vec = robust_clip_preserve_shape(vec)
        if np.isfinite(vec).sum() == 0:
            skipped.append({**row.to_dict(), "reason": "all_values_missing"})
            continue
        key = (row["patient"], row["group"], int(row["layer"]))
        sample_dict.setdefault(key, {})[row["marker"]] = vec
        recognized.append(row.to_dict())
    return sample_dict, pd.DataFrame(recognized), pd.DataFrame(skipped)


def assemble_sample_matrix(sample_dict, min_markers_per_pixel=10, min_nonzero_features=3):
    all_markers = sorted({m for d in sample_dict.values() for m in d.keys()})
    per_sample = []
    for (patient, group, layer), mdict in sample_dict.items():
        lengths = [len(v) for v in mdict.values() if len(v) > 0]
        if not lengths:
            continue
        n = min(lengths)
        if n < 50:
            continue
        X = np.full((n, len(all_markers)), np.nan, dtype=float)
        for j, marker in enumerate(all_markers):
            if marker in mdict:
                X[:, j] = mdict[marker][:n]
        finite_count = np.sum(np.isfinite(X), axis=1)
        nonzero_count = np.sum(np.nan_to_num(X, nan=0.0) > 0, axis=1)
        keep = (finite_count >= min_markers_per_pixel) & (nonzero_count >= min_nonzero_features)
        X = X[keep]
        if X.shape[0] < 50:
            continue
        X = fill_nonfinite_matrix(X)
        per_sample.append({
            "sample_id": f"{patient}_{group}_layer{layer}",
            "patient": patient,
            "group": group,
            "layer": layer,
            "matrix": X
        })
    return per_sample, all_markers


def make_superpoints(X, target_n=120, random_state=42):
    X = fill_nonfinite_matrix(X)
    if X.shape[0] <= target_n:
        return X
    n_clusters = min(target_n, X.shape[0])
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=2048, n_init=5)
    labels = km.fit_predict(X)
    centers = np.zeros((n_clusters, X.shape[1]), dtype=float)
    for i in range(n_clusters):
        mask = labels == i
        if np.any(mask):
            centers[i] = X[mask].mean(axis=0)
    return fill_nonfinite_matrix(centers)


def prepare_superpoint_matrices(per_sample, all_markers, target_superpoints_per_sample=120, random_state=42):
    matrices = []
    meta_rows = []
    for item in per_sample:
        X = fill_nonfinite_matrix(item["matrix"])
        X = np.log1p(np.maximum(X, 0))
        X = make_superpoints(X, target_n=target_superpoints_per_sample, random_state=random_state)
        matrices.append(X)
        for _ in range(X.shape[0]):
            meta_rows.append({
                "sample_id": item["sample_id"],
                "patient": item["patient"],
                "group": item["group"],
                "layer": item["layer"]
            })
    if not matrices:
        raise ValueError("No superpoint matrices created")
    return pd.DataFrame(np.vstack(matrices), columns=all_markers), pd.DataFrame(meta_rows)


def balanced_column_sample(X_df, meta_df, groups, max_per_group=90, random_state=42):
    rng = np.random.default_rng(random_state)
    keep_idx = []
    for g in groups:
        idx = meta_df.index[meta_df["group"] == g].to_numpy()
        chosen = idx if len(idx) <= max_per_group else rng.choice(idx, size=max_per_group, replace=False)
        keep_idx.extend(chosen.tolist())
    keep_idx = np.array(sorted(keep_idx))
    return X_df.iloc[keep_idx].reset_index(drop=True), meta_df.iloc[keep_idx].reset_index(drop=True)


def row_zscore(mat):
    mat = np.asarray(mat, dtype=float)
    mu = np.nanmean(mat, axis=1, keepdims=True)
    sd = np.nanstd(mat, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    z = (mat - mu) / sd
    return np.clip(z, -2.0, 2.0)


def compute_row_order(heat_mat):
    row_dist = pdist(heat_mat, metric="euclidean")
    Z = linkage(row_dist, method="average")
    d = dendrogram(Z, no_plot=True)
    return Z, d["leaves"]


def compute_dotplot_tables(X_df, meta_df, groups):
    mean_table = []
    pct_table = []
    for g in groups:
        Xg = X_df.loc[meta_df["group"] == g]
        mean_vals = Xg.mean(axis=0)
        pct_vals = (Xg > 0).mean(axis=0) * 100.0
        mean_table.append(pd.DataFrame({"marker": X_df.columns, "group": g, "mean_expression": mean_vals.values}))
        pct_table.append(pd.DataFrame({"marker": X_df.columns, "group": g, "percent_expressed": pct_vals.values}))
    mean_df = pd.concat(mean_table, ignore_index=True)
    pct_df = pd.concat(pct_table, ignore_index=True)
    mean_wide = mean_df.pivot(index="marker", columns="group", values="mean_expression")
    pct_wide = pct_df.pivot(index="marker", columns="group", values="percent_expressed")

    arr = mean_wide.values
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    mean_z = mean_wide.copy()
    mean_z.iloc[:, :] = np.clip((arr - mu) / sd, -2.0, 2.0)
    return mean_df, pct_df, mean_wide, pct_wide, mean_z


def get_group_boundaries(meta_sorted, groups):
    out = []
    for g in groups:
        idx = np.where(meta_sorted["group"].values == g)[0]
        if len(idx) > 0:
            out.append((g, idx.min(), idx.max()))
    return out


def plot_heatmap_only(X_heat, meta_heat, groups, row_order, out_dir: Path):
    X_sorted = X_heat.copy()
    X_sorted["__group__"] = meta_heat["group"].values
    X_sorted = X_sorted.sort_values("__group__", key=lambda s: s.map({g: i for i, g in enumerate(groups)}))
    meta_sorted = meta_heat.loc[X_sorted.index].reset_index(drop=True)
    X_sorted = X_sorted.drop(columns="__group__").reset_index(drop=True)

    heat_mat = row_zscore(X_sorted.T.values)[row_order, :]
    markers_ordered = X_sorted.columns[row_order].tolist()
    group_codes = meta_sorted["group"].map({g: i for i, g in enumerate(groups)}).values
    group_boundaries = get_group_boundaries(meta_sorted, groups)
    Z, _ = compute_row_order(row_zscore(X_sorted.T.values))

    fig = plt.figure(figsize=(15.8, 18))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.8, 8.8, 1.2],   # balanced gap
        height_ratios=[0.28, 10],
        wspace=0.04, hspace=0.03
    )

    ax_top = fig.add_subplot(gs[0, 1])
    ax_den = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_gap = fig.add_subplot(gs[:, 2])
    ax_gap.axis("off")

    cmap_top = ListedColormap([GROUP_COLORS[g] for g in groups])
    ax_top.imshow(group_codes[np.newaxis, :], aspect="auto", cmap=cmap_top, vmin=0, vmax=len(groups)-1)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for g, start, end in group_boundaries:
        ax_top.text((start + end) / 2, -0.65, GROUP_DISPLAY[g], ha="center", va="bottom", fontsize=8.8, fontweight="bold")
        ax_top.axvline(start - 0.5, color="#666666", linewidth=0.6)
    ax_top.axvline(len(group_codes) - 0.5, color="#666666", linewidth=0.6)
    for side in ["top", "right", "left", "bottom"]:
        ax_top.spines[side].set_visible(False)

    dendrogram(
        Z, orientation="left", no_labels=True, color_threshold=0,
        above_threshold_color="#8F8F8F", ax=ax_den
    )
    ax_den.invert_yaxis()
    ax_den.set_xticks([])
    ax_den.set_yticks([])
    ax_den.margins(x=0.08)  # reduce crowding of lines
    for line in ax_den.get_lines():
        line.set_linewidth(0.55)
        line.set_alpha(0.9)
    for side in ["top", "right", "bottom", "left"]:
        ax_den.spines[side].set_visible(False)

    im = ax_heat.imshow(heat_mat, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2, interpolation="nearest", origin="upper")
    ax_heat.set_xticks([])
    ax_heat.set_yticks(np.arange(len(markers_ordered)))
    ax_heat.set_yticklabels(markers_ordered, fontsize=4.3)
    ax_heat.yaxis.tick_right()
    ax_heat.tick_params(axis="y", length=0, pad=1.6)
    for _, start, end in group_boundaries:
        ax_heat.axvline(start - 0.5, color="#808080", linewidth=0.8)
    ax_heat.axvline(len(group_codes) - 0.5, color="#808080", linewidth=0.8)

    cax = fig.add_axes([0.925, 0.47, 0.012, 0.16])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Z-score", fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Global expression heatmap of metabolic features", fontsize=14, fontweight="bold", y=0.995)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"Fig5_heatmap_only.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_dotplot_only(mean_z, pct_wide, groups, row_order, out_dir: Path):
    markers_ordered = mean_z.index[row_order].tolist()
    mean_plot = mean_z.loc[markers_ordered, groups]
    pct_plot = pct_wide.loc[markers_ordered, groups]

    fig_h = max(15, len(markers_ordered) * 0.11)
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    xs, ys, sizes, colors = [], [], [], []
    size_min, size_max = 8, 90

    for yi, marker in enumerate(markers_ordered):
        for xi, g in enumerate(groups):
            xs.append(xi)
            ys.append(yi)
            pct = float(pct_plot.loc[marker, g])
            sizes.append(size_min + (pct / 100.0) * (size_max - size_min))
            colors.append(float(mean_plot.loc[marker, g]))

    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="coolwarm", vmin=-2, vmax=2, edgecolors="none", alpha=0.95)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(len(markers_ordered) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([GROUP_DISPLAY[g] for g in groups], fontsize=9)
    ax.set_yticks(np.arange(len(markers_ordered)))
    ax.set_yticklabels(markers_ordered, fontsize=4.7)
    ax.set_title("Dynamic shifts of metabolic features", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="x", linestyle=(0, (2, 3)), linewidth=0.5, alpha=0.35)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.030, pad=0.03)
    cbar.set_label("Average expression\n(Z-score)", fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=7)

    legend_sizes = [25, 50, 75, 100]
    handles = [ax.scatter([], [], s=size_min + (s / 100.0) * (size_max - size_min), c="#7A7A7A") for s in legend_sizes]
    ax.legend(handles, [str(s) for s in legend_sizes], title="Percent expressed",
              frameon=False, bbox_to_anchor=(1.30, 0.58), loc="center left", fontsize=7, title_fontsize=8)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"Fig5_dotplot_only.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_combined_figure(X_heat, meta_heat, mean_z, pct_wide, groups, row_order, out_dir: Path):
    X_sorted = X_heat.copy()
    X_sorted["__group__"] = meta_heat["group"].values
    X_sorted = X_sorted.sort_values("__group__", key=lambda s: s.map({g: i for i, g in enumerate(groups)}))
    meta_sorted = meta_heat.loc[X_sorted.index].reset_index(drop=True)
    X_sorted = X_sorted.drop(columns="__group__").reset_index(drop=True)

    heat_mat = row_zscore(X_sorted.T.values)[row_order, :]
    markers_ordered = X_sorted.columns[row_order].tolist()
    group_codes = meta_sorted["group"].map({g: i for i, g in enumerate(groups)}).values
    group_boundaries = get_group_boundaries(meta_sorted, groups)
    mean_plot = mean_z.loc[markers_ordered, groups]
    pct_plot = pct_wide.loc[markers_ordered, groups]
    Z, _ = compute_row_order(row_zscore(X_sorted.T.values))

    fig = plt.figure(figsize=(21.8, 19))
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[1.85, 8.5, 1.15, 4.1, 0.55],   # balanced, not too wide
        height_ratios=[0.30, 10],
        wspace=0.04, hspace=0.03
    )

    ax_top = fig.add_subplot(gs[0, 1])
    ax_den = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_gap1 = fig.add_subplot(gs[:, 2])
    ax_dot = fig.add_subplot(gs[1, 3])
    ax_gap2 = fig.add_subplot(gs[:, 4])
    ax_gap1.axis("off")
    ax_gap2.axis("off")

    cmap_top = ListedColormap([GROUP_COLORS[g] for g in groups])
    ax_top.imshow(group_codes[np.newaxis, :], aspect="auto", cmap=cmap_top, vmin=0, vmax=len(groups)-1)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for g, start, end in group_boundaries:
        ax_top.text((start + end) / 2, -0.70, GROUP_DISPLAY[g], ha="center", va="bottom", fontsize=9.3, fontweight="bold")
        ax_top.axvline(start - 0.5, color="#666666", linewidth=0.6)
    ax_top.axvline(len(group_codes) - 0.5, color="#666666", linewidth=0.6)
    for side in ["top", "right", "left", "bottom"]:
        ax_top.spines[side].set_visible(False)

    dendrogram(
        Z, orientation="left", no_labels=True, color_threshold=0,
        above_threshold_color="#8F8F8F", ax=ax_den
    )
    ax_den.invert_yaxis()
    ax_den.set_xticks([])
    ax_den.set_yticks([])
    ax_den.margins(x=0.08)
    for line in ax_den.get_lines():
        line.set_linewidth(0.55)
        line.set_alpha(0.9)
    for side in ["top", "right", "bottom", "left"]:
        ax_den.spines[side].set_visible(False)

    im = ax_heat.imshow(heat_mat, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2, interpolation="nearest", origin="upper")
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    for _, start, end in group_boundaries:
        ax_heat.axvline(start - 0.5, color="#808080", linewidth=0.8)
    ax_heat.axvline(len(group_codes) - 0.5, color="#808080", linewidth=0.8)

    xs, ys, sizes, colors = [], [], [], []
    size_min, size_max = 7, 72
    for yi, marker in enumerate(markers_ordered):
        for xi, g in enumerate(groups):
            xs.append(xi)
            ys.append(yi)
            pct = float(pct_plot.loc[marker, g])
            sizes.append(size_min + (pct / 100.0) * (size_max - size_min))
            colors.append(float(mean_plot.loc[marker, g]))

    sc = ax_dot.scatter(xs, ys, s=sizes, c=colors, cmap="coolwarm", vmin=-2, vmax=2, edgecolors="none", alpha=0.95)
    ax_dot.set_xlim(-0.5, len(groups) - 0.5)
    ax_dot.set_ylim(len(markers_ordered) - 0.5, -0.5)
    ax_dot.set_xticks(np.arange(len(groups)))
    ax_dot.set_xticklabels([GROUP_DISPLAY[g] for g in groups], fontsize=9)
    ax_dot.set_yticks(np.arange(len(markers_ordered)))
    ax_dot.set_yticklabels(markers_ordered, fontsize=4.25)
    ax_dot.yaxis.tick_left()
    ax_dot.set_title("Dynamic shifts of metabolic features", fontsize=12, fontweight="bold", pad=6)
    ax_dot.tick_params(axis="both", length=0, pad=2.0)
    ax_dot.grid(axis="x", linestyle=(0, (2, 3)), linewidth=0.45, alpha=0.32)
    for side in ["top", "right"]:
        ax_dot.spines[side].set_visible(False)

    cax_heat = fig.add_axes([0.914, 0.49, 0.010, 0.14])
    cbar1 = fig.colorbar(im, cax=cax_heat)
    cbar1.set_label("Heatmap\nZ-score", fontsize=8, labelpad=7)
    cbar1.ax.tick_params(labelsize=7)

    cax_dot = fig.add_axes([0.937, 0.49, 0.010, 0.14])
    cbar2 = fig.colorbar(sc, cax=cax_dot)
    cbar2.set_label("Average expression\n(Z-score)", fontsize=8, labelpad=7)
    cbar2.ax.tick_params(labelsize=7)

    legend_sizes = [25, 50, 75, 100]
    handles = [ax_dot.scatter([], [], s=size_min + (s / 100.0) * (size_max - size_min), c="#7A7A7A") for s in legend_sizes]
    ax_dot.legend(handles, [str(s) for s in legend_sizes], title="Percent expressed",
                  frameon=False, bbox_to_anchor=(1.40, 0.54), loc="center left", fontsize=7, title_fontsize=8)

    fig.suptitle("Data overview and feature validation", fontsize=15, fontweight="bold", y=0.995)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"Fig5_data_overview_heatmap_dotplot.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Data overview heatmap + dotplot from nested 2D Raman .mat files")
    parser.add_argument("--data-dir", type=str, default="data_2d")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--target-layer", type=int, default=1)
    parser.add_argument("--target-superpoints-per-sample", type=int, default=120)
    parser.add_argument("--heatmap-max-per-group", type=int, default=90)
    parser.add_argument("--min-markers-per-pixel", type=int, default=10)
    parser.add_argument("--min-nonzero-features", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    script_name = Path(__file__).stem
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / script_name
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = ["Normal", "High", "Middle", "Low"]

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    mat_files = find_mat_files(data_dir)
    parsed = []
    skipped_parse = []
    for p in mat_files:
        info = parse_filename(p)
        if info is None:
            skipped_parse.append({"file": str(p), "reason": "filename_not_recognized"})
        else:
            parsed.append(info)
    meta_df = pd.DataFrame(parsed)
    pd.DataFrame(skipped_parse).to_csv(out_dir / "skipped_unrecognized_files.csv", index=False, encoding="utf-8-sig")

    if meta_df.empty:
        raise SystemExit("No recognized files. Example filename: CD98-High-PatientB_layer1.mat")

    sample_dict, recognized_df, skipped_df = build_sample_marker_dict(meta_df, target_layer=args.target_layer, allowed_groups=groups)
    recognized_df.to_csv(out_dir / "recognized_input_files.csv", index=False, encoding="utf-8-sig")
    if not skipped_df.empty:
        skipped_df.to_csv(out_dir / "skipped_after_loading.csv", index=False, encoding="utf-8-sig")
    if recognized_df.empty:
        raise SystemExit("No valid files after filtering.")

    per_sample, all_markers = assemble_sample_matrix(
        sample_dict,
        min_markers_per_pixel=args.min_markers_per_pixel,
        min_nonzero_features=args.min_nonzero_features
    )
    pd.DataFrame({"marker": all_markers}).to_csv(out_dir / "used_markers.csv", index=False, encoding="utf-8-sig")
    if len(per_sample) < 2:
        raise SystemExit("Too few valid samples after assembling matrices.")

    X_all, meta_all = prepare_superpoint_matrices(
        per_sample, all_markers,
        target_superpoints_per_sample=args.target_superpoints_per_sample,
        random_state=args.random_state
    )

    X_heat, meta_heat = balanced_column_sample(
        X_all, meta_all, groups=groups,
        max_per_group=args.heatmap_max_per_group,
        random_state=args.random_state
    )

    heat_for_order = row_zscore(X_heat.T.values)
    _, row_order = compute_row_order(heat_for_order)
    mean_df, pct_df, mean_wide, pct_wide, mean_z = compute_dotplot_tables(X_all, meta_all, groups)

    sample_summary = meta_all.groupby(["group", "patient"]).size().reset_index(name="n_superpoints")
    sample_summary.to_csv(out_dir / "sample_superpoint_summary.csv", index=False, encoding="utf-8-sig")
    meta_heat.to_csv(out_dir / "heatmap_selected_columns_metadata.csv", index=False, encoding="utf-8-sig")
    mean_df.to_csv(out_dir / "dotplot_mean_expression_long.csv", index=False, encoding="utf-8-sig")
    pct_df.to_csv(out_dir / "dotplot_percent_expressed_long.csv", index=False, encoding="utf-8-sig")
    mean_wide.to_csv(out_dir / "dotplot_mean_expression_wide.csv", encoding="utf-8-sig")
    pct_wide.to_csv(out_dir / "dotplot_percent_expressed_wide.csv", encoding="utf-8-sig")
    mean_z.to_csv(out_dir / "dotplot_mean_expression_zscore_wide.csv", encoding="utf-8-sig")

    plot_heatmap_only(X_heat, meta_heat, groups, row_order, out_dir)
    plot_dotplot_only(mean_z, pct_wide, groups, row_order, out_dir)
    plot_combined_figure(X_heat, meta_heat, mean_z, pct_wide, groups, row_order, out_dir)

    params = vars(args)
    params["script_name"] = script_name
    params["groups_present"] = groups
    params["n_mat_files_found"] = len(mat_files)
    params["n_valid_samples"] = len(per_sample)
    params["n_total_superpoints"] = len(X_all)
    params["n_heatmap_columns"] = len(X_heat)
    params["n_features"] = len(all_markers)
    with open(out_dir / "run_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    readme = [
        "Data overview and feature validation. v3 balanced-spacing version.",
        f"Script: {script_name}.py",
        "",
        "Main figure:",
        "  Fig5_data_overview_heatmap_dotplot.png / pdf / svg",
        "",
        "Single figures:",
        "  Fig5_heatmap_only.*",
        "  Fig5_dotplot_only.*",
        "",
        "Main updates:",
        "  1) right spacing reduced to a more reasonable level",
        "  2) label-bar overlap still avoided",
        "  3) left dendrogram width increased slightly to reduce line crowding"
    ]
    (out_dir / "00_README.txt").write_text("\n".join(readme), encoding="utf-8")

    try:
        shutil.copy2(Path(__file__), out_dir / f"{script_name}.py")
    except Exception:
        pass

    print("Done.")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
