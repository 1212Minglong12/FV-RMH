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

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8

GROUP_ALIASES = {
    "Normal": "Normal","NAT": "Normal","Healthy": "Normal","Control": "Normal",
    "High": "High","WD": "High","Well": "High","WellDifferentiated": "High","Well_Differentiated": "High",
    "Middle": "Middle","Moderate": "Middle","MD": "Middle","Mid": "Middle","ModeratelyDifferentiated": "Middle","Moderately_Differentiated": "Middle",
    "Low": "Low","Poor": "Low","PD": "Low","PoorlyDifferentiated": "Low","Poorly_Differentiated": "Low"
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

MODULE_ORDER = [
    "Extracellular Matrix","Amino Acids","Interleukins","Calcification",
    "Infection & Immunity","Hormones","Carotenoids","Other Metabolites",
    "Glycolipid Metabolism","Vitamins","Repair & Cytokines","Oxidative Stress"
]

MODULE_RULES = {
    "Extracellular Matrix": ["acta2","collagen","laminin","elastin","fibronectin","syndecan","versican","decorin","biglycan","vitronectin"],
    "Amino Acids": ["aspartic","proline","methionine","glutamine","asparagine","alanine","glycine","serine","tryptophan","tyrosine","valine","leucine","isoleucine","lysine","histidine","threonine"],
    "Interleukins": ["interleukin","il_","il-","cxcl","ccl","tnf","ifn"],
    "Calcification": ["calcium","hydroxyapatite","phosphate","mineral","alkalinephosphatase","alkalinephosphatasealpl"],
    "Infection & Immunity": ["cd68","cd31","cd98","b7h3","b7-h3","b7_h3","pdl1","cd4","cd8","immune","lysozyme","ccl7"],
    "Hormones": ["estrogen","progesterone","androgen","cortisol","thyroid","hormone","cyp"],
    "Carotenoids": ["carotene","betacarotene","beta-carotene"],
    "Other Metabolites": ["formaldehyde","fumarate","nicotinamide","succinate","citrate","malate","bmp4","aspergillus","butyricacid"],
    "Glycolipid Metabolism": ["acetyl","fructose","glucose","phosphoglycerate","lipid","glycolipid","cholesterol","fatty","triglyceride","sphing","glycerol","nadh","pyruvic","ketoglutaric","sphinosine"],
    "Vitamins": ["vitamin","pyridoxine","albumin"],
    "Repair & Cytokines": ["repair","cytokine","wound","healing","tgfb","tgf-b","tgf","cdkn1a","cdh1","titf1"],
    "Oxidative Stress": ["superoxidedismutase","sod","oxidative","ros","glutathione","cml","pentosidine"]
}

def normalize_marker_name(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def assign_module_to_marker(marker_name):
    m = normalize_marker_name(marker_name)
    matched = []
    for module, kws in MODULE_RULES.items():
        for kw in kws:
            if normalize_marker_name(kw) in m:
                matched.append(module)
                break
    if not matched:
        return "Other Metabolites"
    priority = {m: i for i, m in enumerate(MODULE_ORDER)}
    matched = sorted(matched, key=lambda x: priority.get(x, 999))
    return matched[0]

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
    return {"marker": m.group(1), "raw_group": m.group(2), "group": GROUP_ALIASES.get(m.group(2), m.group(2)),
            "patient": m.group(3), "layer": int(m.group(4)), "file": str(path)}

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
        import h5py
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

def build_sample_marker_dict(meta_df, target_layer=1, allowed_groups=None):
    sample_dict, recognized, skipped = {}, [], []
    for _, row in meta_df.iterrows():
        if allowed_groups is not None and row["group"] not in allowed_groups:
            skipped.append({**row.to_dict(), "reason": "group_filtered"}); continue
        if target_layer is not None and int(row["layer"]) != int(target_layer):
            skipped.append({**row.to_dict(), "reason": "layer_filtered"}); continue
        try:
            vec = load_mat_numeric_array(Path(row["file"]))
        except Exception as e:
            skipped.append({**row.to_dict(), "reason": f"load_failed:{e}"}); continue
        vec = robust_clip_preserve_shape(vec)
        if np.isfinite(vec).sum() == 0:
            skipped.append({**row.to_dict(), "reason": "all_values_missing"}); continue
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
        per_sample.append({"sample_id": f"{patient}_{group}_layer{layer}", "patient": patient, "group": group, "layer": layer, "matrix": X})
    return per_sample, all_markers

def make_superpoints(X, target_n=110, random_state=42):
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

def prepare_superpoint_matrices(per_sample, all_markers, target_superpoints_per_sample=110, random_state=42):
    mats, meta_rows = [], []
    for item in per_sample:
        X = fill_nonfinite_matrix(item["matrix"])
        X = np.log1p(np.maximum(X, 0))
        X = make_superpoints(X, target_n=target_superpoints_per_sample, random_state=random_state)
        mats.append(X)
        for _ in range(X.shape[0]):
            meta_rows.append({"sample_id": item["sample_id"], "patient": item["patient"], "group": item["group"], "layer": item["layer"]})
    if not mats:
        raise ValueError("No superpoint matrices created")
    return pd.DataFrame(np.vstack(mats), columns=all_markers), pd.DataFrame(meta_rows)

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
    return np.clip((mat - mu) / sd, -2.0, 2.0)

def compute_row_order(heat_mat):
    row_dist = pdist(heat_mat, metric="euclidean")
    Z = linkage(row_dist, method="average")
    d = dendrogram(Z, no_plot=True)
    return Z, d["leaves"]

def compute_module_scores(X_df):
    marker_to_module = {m: assign_module_to_marker(m) for m in X_df.columns}
    mapping_df = pd.DataFrame({"marker": list(marker_to_module.keys()), "module": list(marker_to_module.values())})
    score_df = pd.DataFrame(index=X_df.index)
    counts = {}
    for module in MODULE_ORDER:
        cols = [m for m, mod in marker_to_module.items() if mod == module]
        counts[module] = len(cols)
        if len(cols) == 0:
            score_df[module] = 0.0
        elif len(cols) == 1:
            score_df[module] = X_df[cols[0]].values
        else:
            score_df[module] = X_df[cols].mean(axis=1).values
    count_df = pd.DataFrame({"module": list(counts.keys()), "n_markers": list(counts.values())})
    return score_df, mapping_df, count_df

def compute_module_dotplot_tables(module_df, meta_df, groups):
    mean_table, pct_table = [], []
    for g in groups:
        Xg = module_df.loc[meta_df["group"] == g]
        mean_vals = Xg.mean(axis=0)
        pct_vals = (Xg > 0).mean(axis=0) * 100.0
        mean_table.append(pd.DataFrame({"module": module_df.columns, "group": g, "mean_expression": mean_vals.values}))
        pct_table.append(pd.DataFrame({"module": module_df.columns, "group": g, "percent_expressed": pct_vals.values}))
    mean_df = pd.concat(mean_table, ignore_index=True)
    pct_df = pd.concat(pct_table, ignore_index=True)
    mean_wide = mean_df.pivot(index="module", columns="group", values="mean_expression")
    pct_wide = pct_df.pivot(index="module", columns="group", values="percent_expressed")
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

def plot_heatmap_only(module_heat, meta_heat, groups, row_order, out_dir):
    X_sorted = module_heat.copy()
    X_sorted["__group__"] = meta_heat["group"].values
    X_sorted = X_sorted.sort_values("__group__", key=lambda s: s.map({g: i for i, g in enumerate(groups)}))
    meta_sorted = meta_heat.loc[X_sorted.index].reset_index(drop=True)
    X_sorted = X_sorted.drop(columns="__group__").reset_index(drop=True)

    heat_mat = row_zscore(X_sorted.T.values)[row_order, :]
    modules_ordered = X_sorted.columns[row_order].tolist()
    group_codes = meta_sorted["group"].map({g: i for i, g in enumerate(groups)}).values
    group_boundaries = get_group_boundaries(meta_sorted, groups)
    Z, _ = compute_row_order(row_zscore(X_sorted.T.values))

    fig = plt.figure(figsize=(14.2, 6.7))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.7, 8.8, 1.2, 0.45], height_ratios=[0.35, 6], wspace=0.05, hspace=0.03)
    ax_top = fig.add_subplot(gs[0, 1]); ax_den = fig.add_subplot(gs[1, 0]); ax_heat = fig.add_subplot(gs[1, 1]); ax_pad = fig.add_subplot(gs[:, 2]); ax_pad.axis("off"); cax = fig.add_subplot(gs[1, 3])

    cmap_top = ListedColormap([GROUP_COLORS[g] for g in groups])
    ax_top.imshow(group_codes[np.newaxis, :], aspect="auto", cmap=cmap_top, vmin=0, vmax=len(groups)-1)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for g, start, end in group_boundaries:
        ax_top.text((start + end) / 2, -0.75, GROUP_DISPLAY[g], ha="center", va="bottom", fontsize=9.0, fontweight="bold")
        ax_top.axvline(start - 0.5, color="#666666", linewidth=0.7)
    ax_top.axvline(len(group_codes) - 0.5, color="#666666", linewidth=0.7)
    for side in ["top","right","left","bottom"]:
        ax_top.spines[side].set_visible(False)

    dendrogram(Z, orientation="left", no_labels=True, color_threshold=0, above_threshold_color="#909090", ax=ax_den)
    ax_den.invert_yaxis()
    ax_den.set_xticks([]); ax_den.set_yticks([]); ax_den.margins(x=0.10)
    for side in ["top","right","bottom","left"]:
        ax_den.spines[side].set_visible(False)

    im = ax_heat.imshow(heat_mat, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2, interpolation="nearest", origin="upper")
    ax_heat.set_xticks([])
    ax_heat.set_yticks(np.arange(len(modules_ordered)))
    ax_heat.set_yticklabels(modules_ordered, fontsize=8.5, fontweight="bold")
    ax_heat.yaxis.tick_right()
    ax_heat.tick_params(axis="y", length=0, pad=2)
    for _, start, end in group_boundaries:
        ax_heat.axvline(start - 0.5, color="#808080", linewidth=0.9)
    ax_heat.axvline(len(group_codes) - 0.5, color="#808080", linewidth=0.9)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Z-score", fontsize=8, labelpad=7)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Metabolic pathway enrichment heatmap", fontsize=13.5, fontweight="bold", y=0.995)
    for ext in ["png","pdf","svg"]:
        fig.savefig(out_dir / f"Fig6_heatmap_only.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

def plot_dotplot_only(mean_z, pct_wide, row_order, groups, out_dir):
    modules_ordered = mean_z.index[row_order].tolist()
    mean_plot = mean_z.loc[modules_ordered, groups]
    pct_plot = pct_wide.loc[modules_ordered, groups]

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    xs, ys, sizes, colors = [], [], [], []
    size_min, size_max = 16, 150
    for yi, module in enumerate(modules_ordered):
        for xi, g in enumerate(groups):
            xs.append(xi); ys.append(yi)
            pct = float(pct_plot.loc[module, g])
            sizes.append(size_min + (pct / 100.0) * (size_max - size_min))
            colors.append(float(mean_plot.loc[module, g]))

    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="coolwarm", vmin=-2, vmax=2, edgecolors="none", alpha=0.95)
    ax.set_xlim(-0.5, len(groups) - 0.5); ax.set_ylim(len(modules_ordered) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels([GROUP_DISPLAY[g] for g in groups], fontsize=9)
    ax.set_yticks(np.arange(len(modules_ordered))); ax.set_yticklabels(modules_ordered, fontsize=8.4, fontweight="bold")
    ax.set_title("Systemic reprogramming of metabolic modules", fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="x", linestyle=(0, (2, 3)), linewidth=0.5, alpha=0.35)
    for side in ["top","right"]:
        ax.spines[side].set_visible(False)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.05)
    cbar.set_label("Average expression", fontsize=8); cbar.ax.tick_params(labelsize=7)

    legend_sizes = [25, 50, 75]
    handles = [ax.scatter([], [], s=size_min + (s / 100.0) * (size_max - size_min), c="#7A7A7A") for s in legend_sizes]
    ax.legend(handles, [str(s) for s in legend_sizes], title="Percent expressed", frameon=False, bbox_to_anchor=(1.35, 0.22), loc="center left", fontsize=7, title_fontsize=8)

    for ext in ["png","pdf","svg"]:
        fig.savefig(out_dir / f"Fig6_dotplot_only.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

def plot_combined_figure(module_heat, meta_heat, mean_z, pct_wide, groups, row_order, out_dir):
    X_sorted = module_heat.copy()
    X_sorted["__group__"] = meta_heat["group"].values
    X_sorted = X_sorted.sort_values("__group__", key=lambda s: s.map({g: i for i, g in enumerate(groups)}))
    meta_sorted = meta_heat.loc[X_sorted.index].reset_index(drop=True)
    X_sorted = X_sorted.drop(columns="__group__").reset_index(drop=True)

    heat_mat = row_zscore(X_sorted.T.values)[row_order, :]
    modules_ordered = X_sorted.columns[row_order].tolist()
    group_codes = meta_sorted["group"].map({g: i for i, g in enumerate(groups)}).values
    group_boundaries = get_group_boundaries(meta_sorted, groups)
    mean_plot = mean_z.loc[modules_ordered, groups]
    pct_plot = pct_wide.loc[modules_ordered, groups]
    Z, _ = compute_row_order(row_zscore(X_sorted.T.values))

    fig = plt.figure(figsize=(14.6, 10.8))
    gs = fig.add_gridspec(3, 6, width_ratios=[1.8, 8.6, 1.15, 4.6, 0.45, 0.75], height_ratios=[0.34, 5.0, 4.7], wspace=0.05, hspace=0.12)
    ax_top = fig.add_subplot(gs[0, 1]); ax_den = fig.add_subplot(gs[1, 0]); ax_heat = fig.add_subplot(gs[1, 1]); ax_pad = fig.add_subplot(gs[:, 2]); ax_pad.axis("off")
    ax_dot = fig.add_subplot(gs[2, 1:4]); cax_heat = fig.add_subplot(gs[1, 4]); cax_dot = fig.add_subplot(gs[2, 4]); ax_leg = fig.add_subplot(gs[2, 5]); ax_leg.axis("off")

    cmap_top = ListedColormap([GROUP_COLORS[g] for g in groups])
    ax_top.imshow(group_codes[np.newaxis, :], aspect="auto", cmap=cmap_top, vmin=0, vmax=len(groups)-1)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for g, start, end in group_boundaries:
        ax_top.text((start + end) / 2, -0.75, GROUP_DISPLAY[g], ha="center", va="bottom", fontsize=9.0, fontweight="bold")
        ax_top.axvline(start - 0.5, color="#666666", linewidth=0.7)
    ax_top.axvline(len(group_codes) - 0.5, color="#666666", linewidth=0.7)
    for side in ["top","right","left","bottom"]:
        ax_top.spines[side].set_visible(False)

    dendrogram(Z, orientation="left", no_labels=True, color_threshold=0, above_threshold_color="#909090", ax=ax_den)
    ax_den.invert_yaxis()
    ax_den.set_xticks([]); ax_den.set_yticks([]); ax_den.margins(x=0.10)
    for side in ["top","right","bottom","left"]:
        ax_den.spines[side].set_visible(False)

    im = ax_heat.imshow(heat_mat, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2, interpolation="nearest", origin="upper")
    ax_heat.set_xticks([])
    ax_heat.set_yticks(np.arange(len(modules_ordered)))
    ax_heat.set_yticklabels(modules_ordered, fontsize=8.4, fontweight="bold")
    ax_heat.yaxis.tick_right(); ax_heat.tick_params(axis="y", length=0, pad=2)
    for _, start, end in group_boundaries:
        ax_heat.axvline(start - 0.5, color="#808080", linewidth=0.9)
    ax_heat.axvline(len(group_codes) - 0.5, color="#808080", linewidth=0.9)

    xs, ys, sizes, colors = [], [], [], []
    size_min, size_max = 17, 165
    for yi, module in enumerate(modules_ordered):
        for xi, g in enumerate(groups):
            xs.append(xi); ys.append(yi)
            pct = float(pct_plot.loc[module, g])
            sizes.append(size_min + (pct / 100.0) * (size_max - size_min))
            colors.append(float(mean_plot.loc[module, g]))

    sc = ax_dot.scatter(xs, ys, s=sizes, c=colors, cmap="coolwarm", vmin=-2, vmax=2, edgecolors="none", alpha=0.95)
    ax_dot.set_xlim(-0.5, len(groups) - 0.5); ax_dot.set_ylim(len(modules_ordered) - 0.5, -0.5)
    ax_dot.set_xticks(np.arange(len(groups))); ax_dot.set_xticklabels([GROUP_DISPLAY[g] for g in groups], fontsize=9.4)
    ax_dot.set_yticks(np.arange(len(modules_ordered))); ax_dot.set_yticklabels(modules_ordered, fontsize=8.3, fontweight="bold")
    ax_dot.set_title("Systemic reprogramming of metabolic modules", fontsize=12.1, fontweight="bold", pad=8)
    ax_dot.tick_params(axis="both", length=0)
    ax_dot.grid(axis="x", linestyle=(0, (2, 3)), linewidth=0.5, alpha=0.35)
    for side in ["top","right"]:
        ax_dot.spines[side].set_visible(False)

    cb1 = fig.colorbar(im, cax=cax_heat); cb1.set_label("Z-score", fontsize=8); cb1.ax.tick_params(labelsize=7)
    cb2 = fig.colorbar(sc, cax=cax_dot); cb2.set_label("Average\nexpression", fontsize=8); cb2.ax.tick_params(labelsize=7)

    legend_sizes = [25, 50, 75]
    handles = [ax_dot.scatter([], [], s=size_min + (s / 100.0) * (size_max - size_min), c="#7A7A7A") for s in legend_sizes]
    ax_leg.legend(handles, [str(s) for s in legend_sizes], title="Percent\nexpressed", frameon=False, loc="center", fontsize=7, title_fontsize=8)

    fig.suptitle("Metabolic pathway enrichment dynamics across stages", fontsize=14, fontweight="bold", y=0.992)
    for ext in ["png","pdf","svg"]:
        fig.savefig(out_dir / f"Fig6_module_enrichment_dynamics.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def filter_empty_or_flat_modules(module_all, mapping_df, count_df, min_module_markers=1, min_module_std=1e-6):
    r"""
    Remove module rows that look blank because the module has no mapped marker
    or the module score is nearly constant.
    """
    keep_modules = []
    summary = []

    for module in module_all.columns:
        if module in count_df["module"].values:
            n_markers = int(count_df.loc[count_df["module"] == module, "n_markers"].iloc[0])
        else:
            n_markers = 0

        sd = float(np.nanstd(module_all[module].values))
        mean_val = float(np.nanmean(module_all[module].values))
        keep = (n_markers >= min_module_markers) and (sd >= min_module_std)

        summary.append({
            "module": module,
            "n_markers": n_markers,
            "mean_score": mean_val,
            "std_score": sd,
            "kept_for_plot": keep,
            "reason_if_removed": "" if keep else (
                "no_or_too_few_mapped_markers" if n_markers < min_module_markers else "near_constant_blank_row"
            )
        })

        if keep:
            keep_modules.append(module)

    if len(keep_modules) < 3:
        var_rank = module_all.std(axis=0).sort_values(ascending=False)
        keep_modules = var_rank.head(min(8, len(var_rank))).index.tolist()
        for item in summary:
            if item["module"] in keep_modules:
                item["kept_for_plot"] = True
                item["reason_if_removed"] = ""

    return module_all.loc[:, keep_modules].copy(), pd.DataFrame(summary), keep_modules


def main():
    parser = argparse.ArgumentParser(description="Module enrichment dynamics from nested 2D Raman .mat files")
    parser.add_argument("--data-dir", type=str, default="data_2d")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--target-layer", type=int, default=1)
    parser.add_argument("--target-superpoints-per-sample", type=int, default=110)
    parser.add_argument("--heatmap-max-per-group", type=int, default=90)
    parser.add_argument("--min-markers-per-pixel", type=int, default=10)
    parser.add_argument("--min-nonzero-features", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--keep-all-modules", action="store_true", help="Keep all 12 modules even if some rows are blank/flat.")
    parser.add_argument("--min-module-markers", type=int, default=1, help="Minimum mapped markers required for a module to be plotted.")
    parser.add_argument("--min-module-std", type=float, default=1e-6, help="Minimum module-score SD required for a module to be plotted.")
    args = parser.parse_args()

    script_name = Path(__file__).stem
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / script_name
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = ["Normal","High","Middle","Low"]
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    mat_files = find_mat_files(data_dir)
    parsed, skipped_parse = [], []
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

    per_sample, all_markers = assemble_sample_matrix(sample_dict, min_markers_per_pixel=args.min_markers_per_pixel, min_nonzero_features=args.min_nonzero_features)
    pd.DataFrame({"marker": all_markers}).to_csv(out_dir / "used_markers.csv", index=False, encoding="utf-8-sig")
    if len(per_sample) < 2:
        raise SystemExit("Too few valid samples after assembling matrices.")

    X_all, meta_all = prepare_superpoint_matrices(per_sample, all_markers, target_superpoints_per_sample=args.target_superpoints_per_sample, random_state=args.random_state)
    module_all, mapping_df, count_df = compute_module_scores(X_all)
    mapping_df.to_csv(out_dir / "marker_to_module_mapping.csv", index=False, encoding="utf-8-sig")
    count_df.to_csv(out_dir / "module_marker_counts.csv", index=False, encoding="utf-8-sig")

    if args.keep_all_modules:
        module_plot = module_all.copy()
        module_filter_summary = count_df.copy()
        module_filter_summary["kept_for_plot"] = True
        module_filter_summary["reason_if_removed"] = ""
        kept_modules = module_plot.columns.tolist()
    else:
        module_plot, module_filter_summary, kept_modules = filter_empty_or_flat_modules(
            module_all,
            mapping_df,
            count_df,
            min_module_markers=args.min_module_markers,
            min_module_std=args.min_module_std
        )

    module_filter_summary.to_csv(out_dir / "module_filtering_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"kept_module_for_plot": kept_modules}).to_csv(out_dir / "kept_modules_for_plot.csv", index=False, encoding="utf-8-sig")

    module_heat, meta_heat = balanced_column_sample(module_plot, meta_all, groups=groups, max_per_group=args.heatmap_max_per_group, random_state=args.random_state)
    heat_for_order = row_zscore(module_heat.T.values)
    _, row_order = compute_row_order(heat_for_order)
    mean_df, pct_df, mean_wide, pct_wide, mean_z = compute_module_dotplot_tables(module_plot, meta_all, groups)

    meta_heat.to_csv(out_dir / "heatmap_selected_columns_metadata.csv", index=False, encoding="utf-8-sig")
    mean_df.to_csv(out_dir / "module_dotplot_mean_expression_long.csv", index=False, encoding="utf-8-sig")
    pct_df.to_csv(out_dir / "module_dotplot_percent_expressed_long.csv", index=False, encoding="utf-8-sig")
    mean_wide.to_csv(out_dir / "module_dotplot_mean_expression_wide.csv", encoding="utf-8-sig")
    pct_wide.to_csv(out_dir / "module_dotplot_percent_expressed_wide.csv", encoding="utf-8-sig")
    mean_z.to_csv(out_dir / "module_dotplot_mean_expression_zscore_wide.csv", encoding="utf-8-sig")
    module_all.to_csv(out_dir / "module_score_matrix_all_12_modules.csv", index=False, encoding="utf-8-sig")
    module_plot.to_csv(out_dir / "module_score_matrix_plotted_modules.csv", index=False, encoding="utf-8-sig")

    plot_heatmap_only(module_heat, meta_heat, groups, row_order, out_dir)
    plot_dotplot_only(mean_z, pct_wide, row_order, groups, out_dir)
    plot_combined_figure(module_heat, meta_heat, mean_z, pct_wide, groups, row_order, out_dir)

    params = vars(args)
    params.update({"script_name": script_name, "groups_present": groups, "n_mat_files_found": len(mat_files), "n_valid_samples": len(per_sample), "n_total_superpoints": len(X_all), "n_heatmap_columns": len(module_heat), "n_features": len(all_markers), "n_modules": len(MODULE_ORDER)})
    with open(out_dir / "run_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    readme = [
        "Fig06 v2 clean-layout version.",
        f"Script: {script_name}.py",
        "",
        "Main improvements:",
        "  1) heatmap labels and colorbar separated",
        "  2) extra spacer inserted on the right",
        "  3) dendrogram area widened on the left",
        "  4) blank/flat modules are removed by default",
        "  5) use --keep-all-modules if you want to keep all 12 modules",
        "  6) check module_filtering_summary.csv to see which rows were removed"
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
