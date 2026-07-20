from pathlib import Path
import re
import json
import shutil
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

GROUP_ALIASES = {
    "Normal": "Normal", "NAT": "Normal", "Healthy": "Normal", "Control": "Normal",
    "High": "High", "WD": "High", "Well": "High", "WellDifferentiated": "High", "Well_Differentiated": "High",
    "Middle": "Middle", "Moderate": "Middle", "MD": "Middle", "Mid": "Middle", "ModeratelyDifferentiated": "Middle", "Moderately_Differentiated": "Middle",
    "Low": "Low", "Poor": "Low", "PD": "Low", "PoorlyDifferentiated": "Low", "Poorly_Differentiated": "Low",
}

GROUP_ORDER = ["Normal", "High", "Middle", "Low"]
GROUP_DISPLAY = {
    "Normal": "Normal",
    "High": "High differentiated",
    "Middle": "Moderately differentiated",
    "Low": "Poorly differentiated",
}

GROUP_COLORS = {
    "Normal": "#66c2a5",
    "High": "#8da0cb",
    "Middle": "#fc8d62",
    "Low": "#e64b35",
}

def parse_filename(path: Path):
    stem = path.stem
    group_terms = sorted(GROUP_ALIASES.keys(), key=len, reverse=True)
    group_regex = "|".join([re.escape(g) for g in group_terms])
    m = re.match(rf"^(.+)-({group_regex})-(.+?)_layer(\d+)$", stem)
    if m is None:
        return None
    return {
        "marker": m.group(1),
        "raw_group": m.group(2),
        "group": GROUP_ALIASES.get(m.group(2), m.group(2)),
        "patient": m.group(3),
        "layer": int(m.group(4)),
        "file": str(path),
    }

def find_mat_files(data_dir: Path):
    return sorted([p for p in data_dir.rglob("*.mat") if p.is_file()])

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
                        a = np.array(f[k])
                        if np.issubdtype(a.dtype, np.number):
                            candidates.append((k, a))
                    except Exception:
                        pass
                if not candidates:
                    raise ValueError("No numeric array found in v7.3 mat")
                _, arr = max(candidates, key=lambda kv: kv[1].size)
        arr = np.asarray(arr).squeeze().astype(float).flatten()
        arr[~np.isfinite(arr)] = np.nan
        return arr

def robust_clip(x, lower_q=0.005, upper_q=0.995):
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

def build_specimen_marker_arrays(meta_df, target_layer=1):
    specimen_map = defaultdict(dict)
    specimen_info = {}
    skipped = []
    for _, row in meta_df.iterrows():
        if row["group"] not in GROUP_ORDER:
            skipped.append({**row.to_dict(), "reason": "group_filtered"})
            continue
        if int(row["layer"]) != int(target_layer):
            skipped.append({**row.to_dict(), "reason": "layer_filtered"})
            continue
        key = (row["group"], row["patient"], int(row["layer"]))
        try:
            arr = load_mat_numeric_array(Path(row["file"]))
            arr = robust_clip(arr)
            arr = arr[np.isfinite(arr)]
        except Exception as e:
            skipped.append({**row.to_dict(), "reason": f"load_failed:{e}"})
            continue
        if arr.size == 0:
            skipped.append({**row.to_dict(), "reason": "empty_after_cleaning"})
            continue
        specimen_map[key][row["marker"]] = arr
        specimen_info[key] = {"group": row["group"], "patient": row["patient"], "layer": int(row["layer"])}
    return specimen_map, specimen_info, pd.DataFrame(skipped)

def common_markers(specimen_map, min_presence_ratio=0.7):
    counts = defaultdict(int)
    total = len(specimen_map)
    for key in specimen_map:
        for m in specimen_map[key]:
            counts[m] += 1
    selected = [m for m, c in counts.items() if c >= max(2, int(np.ceil(total * min_presence_ratio)))]
    return sorted(selected)

def build_metaspot_matrix(specimen_map, specimen_info, markers, aggregation_size=40, max_spots_per_specimen=120):
    rows = []
    skipped = []
    for key, m2arr in specimen_map.items():
        use_markers = [m for m in markers if m in m2arr]
        if len(use_markers) < max(10, int(len(markers) * 0.5)):
            skipped.append({"specimen": str(key), "reason": "too_few_markers"})
            continue
        min_len = min(len(m2arr[m]) for m in use_markers)
        if min_len < aggregation_size:
            skipped.append({"specimen": str(key), "reason": "too_few_points"})
            continue
        n_spots = min(max_spots_per_specimen, max(20, min_len // aggregation_size))
        edges = np.linspace(0, min_len, n_spots + 1).astype(int)
        if np.any(np.diff(edges) <= 0):
            skipped.append({"specimen": str(key), "reason": "invalid_edges"})
            continue

        for i in range(n_spots):
            s, e = edges[i], edges[i + 1]
            if e <= s:
                continue
            rec = {
                "group": specimen_info[key]["group"],
                "patient": specimen_info[key]["patient"],
                "layer": specimen_info[key]["layer"],
                "metaspot_id": f"{specimen_info[key]['group']}_{specimen_info[key]['patient']}_spot{i+1}",
            }
            for m in markers:
                if m in m2arr:
                    vec = m2arr[m][:min_len]
                    chunk = vec[s:e]
                    if chunk.size == 0:
                        rec[m] = np.nan
                    else:
                        rec[m] = float(np.nanmean(np.log1p(np.clip(chunk, 0, None))))
                else:
                    rec[m] = np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    return df, pd.DataFrame(skipped)

def fill_missing_by_group(df, marker_cols):
    out = df.copy()
    for grp, idx in out.groupby("group").groups.items():
        sub = out.loc[idx, marker_cols]
        med = sub.median(axis=0, skipna=True)
        med = med.fillna(out[marker_cols].median(axis=0, skipna=True))
        med = med.fillna(0.0)
        out.loc[idx, marker_cols] = sub.fillna(med)
    out[marker_cols] = out[marker_cols].fillna(out[marker_cols].median(axis=0)).fillna(0.0)
    return out

def compute_embedding(X, n_pca=30, umap_neighbors=30, umap_min_dist=0.15, random_state=42):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_comp = min(n_pca, Xs.shape[1], Xs.shape[0] - 1) if Xs.shape[0] > 1 else 2
    n_comp = max(2, n_comp)
    pca = PCA(n_components=n_comp, random_state=random_state)
    Xp = pca.fit_transform(Xs)

    method = "PCA"
    axis_labels = ("PCA 1", "PCA 2")
    emb = Xp[:, :2].copy()

    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(umap_neighbors, max(5, X.shape[0] - 1)),
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=random_state,
        )
        emb = reducer.fit_transform(Xp)
        method = "UMAP"
        axis_labels = ("UMAP 1", "UMAP 2")
    except Exception:
        pass

    return emb, Xp, method, axis_labels

def compute_knn_pseudotime(embedding, group_labels, n_neighbors=14):
    n = embedding.shape[0]
    k = min(n_neighbors, max(3, n - 1))
    graph = kneighbors_graph(embedding, n_neighbors=k, mode="distance", include_self=False)
    dist_graph = 0.5 * (graph + graph.T)

    # root: point nearest centroid of Normal group if available
    normal_idx = np.where(np.array(group_labels) == "Normal")[0]
    if len(normal_idx) > 0:
        centroid = embedding[normal_idx].mean(axis=0)
        d = ((embedding - centroid) ** 2).sum(axis=1)
        root_idx = int(np.argmin(d))
    else:
        root_idx = 0

    dists = shortest_path(dist_graph, directed=False, indices=root_idx)
    if np.isinf(dists).any():
        finite = np.isfinite(dists)
        fill = np.nanmax(dists[finite]) if finite.any() else 1.0
        dists[~finite] = fill

    dmin, dmax = float(dists.min()), float(dists.max())
    if dmax <= dmin:
        pseudotime = np.zeros_like(dists, dtype=float)
    else:
        pseudotime = (dists - dmin) / (dmax - dmin) * 100.0
    return pseudotime, root_idx, dist_graph

def compute_trajectory_skeleton(embedding, pseudotime, n_centers=34, random_state=42):
    n_centers = min(n_centers, max(8, embedding.shape[0] // 15))
    km = KMeans(n_clusters=n_centers, random_state=random_state, n_init=10)
    labels = km.fit_predict(embedding)
    centers = km.cluster_centers_

    # root center: minimum average pseudotime
    center_time = np.array([np.mean(pseudotime[labels == i]) for i in range(n_centers)])
    root_center = int(np.argmin(center_time))

    # MST between centers
    dmat = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    mst = minimum_spanning_tree(dmat).toarray()
    mst = mst + mst.T
    edges = np.argwhere(np.triu(mst > 0, 1))
    return centers, labels, edges, root_center

def draw_stage_trajectory(ax, df, axis_labels, centers, edges, title):
    groups_present = [g for g in GROUP_ORDER if g in df["group"].unique()]
    for g in groups_present:
        sub = df[df["group"] == g]
        ax.scatter(sub["dim1"], sub["dim2"], s=5, c=GROUP_COLORS[g], label=GROUP_DISPLAY[g], alpha=0.85, linewidths=0)
    for i, j in edges:
        ax.plot([centers[i, 0], centers[j, 0]], [centers[i, 1], centers[j, 1]], color="black", lw=1.2, alpha=0.85, zorder=5)
    ax.set_xlabel(axis_labels[0], fontsize=9)
    ax.set_ylabel(axis_labels[1], fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLORS[g], markersize=7, label=GROUP_DISPLAY[g]) for g in groups_present]
    ax.legend(handles=handles, title="Group", fontsize=8, title_fontsize=8, frameon=False, loc="center right")

def draw_pseudotime_panel(ax, df, axis_labels, centers, edges, title):
    sc = ax.scatter(df["dim1"], df["dim2"], s=5, c=df["pseudotime"], cmap="magma", alpha=0.9, linewidths=0)
    for i, j in edges:
        ax.plot([centers[i, 0], centers[j, 0]], [centers[i, 1], centers[j, 1]], color="black", lw=1.0, alpha=0.65, zorder=5)
    ax.set_xlabel(axis_labels[0], fontsize=9)
    ax.set_ylabel(axis_labels[1], fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    return sc

def marker_scores_for_heatmap(df, marker_cols):
    pseudo = df["pseudotime"].values.astype(float)
    scores = []
    for m in marker_cols:
        vals = df[m].values.astype(float)
        if np.nanstd(vals) == 0:
            rho = 0.0
        else:
            rho = spearmanr(pseudo, vals, nan_policy="omit").statistic
            if not np.isfinite(rho):
                rho = 0.0
        dyn = np.nanstd(vals)
        scores.append((m, abs(rho) + 0.25 * dyn, rho))
    score_df = pd.DataFrame(scores, columns=["marker", "score", "rho"]).sort_values("score", ascending=False)
    return score_df

def build_pseudotime_heatmap(df, marker_cols, n_bins=110, top_n=120, smooth_sigma=2.0):
    score_df = marker_scores_for_heatmap(df, marker_cols)
    top_markers = score_df["marker"].head(top_n).tolist()

    order = np.argsort(df["pseudotime"].values.astype(float))
    pseudo_sorted = df["pseudotime"].values[order].astype(float)
    X_sorted = df[top_markers].values[order].astype(float)

    bins = np.linspace(0, 100, n_bins + 1)
    mat = np.full((len(top_markers), n_bins), np.nan)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for b in range(n_bins):
        sel = (pseudo_sorted >= bins[b]) & (pseudo_sorted < bins[b + 1] if b < n_bins - 1 else pseudo_sorted <= bins[b + 1])
        if sel.sum() == 0:
            continue
        mat[:, b] = np.nanmean(X_sorted[sel, :], axis=0)

    # fill empty bins by nearest existing
    for i in range(mat.shape[0]):
        row = mat[i]
        good = np.where(np.isfinite(row))[0]
        if len(good) == 0:
            row[:] = 0.0
        else:
            missing = np.where(~np.isfinite(row))[0]
            for idx in missing:
                nearest = good[np.argmin(np.abs(good - idx))]
                row[idx] = row[nearest]
        row[:] = gaussian_filter1d(row.astype(float), sigma=smooth_sigma, mode="nearest")
        rmin, rmax = np.min(row), np.max(row)
        if rmax > rmin:
            row[:] = (row - rmin) / (rmax - rmin)
        else:
            row[:] = 0.5
        mat[i] = row

    # order markers by peak position then trend
    peak_pos = np.argmax(mat, axis=1)
    mean_pos = np.sum(mat * np.arange(mat.shape[1])[None, :], axis=1) / (mat.sum(axis=1) + 1e-8)
    order_rows = np.lexsort((mean_pos, peak_pos))
    mat = mat[order_rows]
    ordered_markers = [top_markers[i] for i in order_rows]
    return mat, ordered_markers, score_df

def save_single_panel(fig, out_base):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(str(out_base.with_suffix(f".{ext}")),
                    dpi=600 if ext == ".png" else None,
                    bbox_inches="tight", facecolor="white")

def main():
    parser = argparse.ArgumentParser(description="Figure 10 spatiotemporal trajectory and metabolic cascade atlas")
    parser.add_argument("--data-dir", type=str, default="data_2d")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--target-layer", type=int, default=1)
    parser.add_argument("--aggregation-size", type=int, default=40)
    parser.add_argument("--max-spots-per-specimen", type=int, default=100)
    parser.add_argument("--min-marker-presence", type=float, default=0.70)
    parser.add_argument("--top-heatmap-markers", type=int, default=120)
    parser.add_argument("--heatmap-bins", type=int, default=110)
    parser.add_argument("--groups", nargs="*", default=None, help="Optional group order. Example: --groups Normal High Middle Low")
    args = parser.parse_args()

    script_name = Path(__file__).stem
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / script_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    files = find_mat_files(data_dir)
    parsed, skipped_parse = [], []
    for f in files:
        info = parse_filename(f)
        if info is None:
            skipped_parse.append({"file": str(f), "reason": "filename_not_recognized"})
        else:
            parsed.append(info)
    meta_df = pd.DataFrame(parsed)
    pd.DataFrame(skipped_parse).to_csv(out_dir / "skipped_unrecognized_files.csv", index=False, encoding="utf-8-sig")
    if meta_df.empty:
        raise SystemExit("No recognized files found.")

    specimen_map, specimen_info, skipped_load = build_specimen_marker_arrays(meta_df, target_layer=args.target_layer)
    if not skipped_load.empty:
        skipped_load.to_csv(out_dir / "skipped_after_loading.csv", index=False, encoding="utf-8-sig")
    if len(specimen_map) == 0:
        raise SystemExit("No specimen data loaded.")

    groups_present = sorted({specimen_info[k]["group"] for k in specimen_info}, key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 999)
    if args.groups is not None and len(args.groups) > 0:
        groups_present = [g for g in args.groups if g in groups_present]

    markers = common_markers(specimen_map, min_presence_ratio=args.min_marker_presence)
    if len(markers) < 15:
        raise SystemExit("Too few common markers across specimens.")
    pd.DataFrame({"common_markers": markers}).to_csv(out_dir / "common_markers.csv", index=False, encoding="utf-8-sig")

    metaspot_df, skipped_meta = build_metaspot_matrix(
        specimen_map,
        specimen_info,
        markers,
        aggregation_size=args.aggregation_size,
        max_spots_per_specimen=args.max_spots_per_specimen,
    )
    if not skipped_meta.empty:
        skipped_meta.to_csv(out_dir / "skipped_metaspot_specs.csv", index=False, encoding="utf-8-sig")
    if metaspot_df.empty:
        raise SystemExit("Failed to build metaspot matrix.")
    metaspot_df = metaspot_df[metaspot_df["group"].isin(groups_present)].copy()
    marker_cols = [m for m in markers if m in metaspot_df.columns]
    metaspot_df = fill_missing_by_group(metaspot_df, marker_cols)

    X = metaspot_df[marker_cols].values.astype(float)
    emb, Xp, method, axis_labels = compute_embedding(X)
    metaspot_df["dim1"] = emb[:, 0]
    metaspot_df["dim2"] = emb[:, 1]

    pseudotime, root_idx, graph = compute_knn_pseudotime(emb, metaspot_df["group"].tolist(), n_neighbors=14)
    metaspot_df["pseudotime"] = pseudotime

    centers, cluster_labels, edges, root_center = compute_trajectory_skeleton(emb, pseudotime, n_centers=34)
    metaspot_df["cluster"] = cluster_labels

    # Heatmap
    heatmap_mat, ordered_markers, score_df = build_pseudotime_heatmap(
        metaspot_df, marker_cols, n_bins=args.heatmap_bins, top_n=args.top_heatmap_markers, smooth_sigma=2.0
    )
    score_df.to_csv(out_dir / "heatmap_marker_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"ordered_heatmap_markers": ordered_markers}).to_csv(out_dir / "ordered_heatmap_markers.csv", index=False, encoding="utf-8-sig")
    metaspot_df.to_csv(out_dir / "metaspot_embedding_pseudotime_table.csv", index=False, encoding="utf-8-sig")

    # -------- overall figure --------
    fig = plt.figure(figsize=(14.8, 10.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.22, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    draw_stage_trajectory(ax1, metaspot_df, axis_labels, centers, edges, "Trajectory by Clinical Stage")

    ax2 = fig.add_subplot(gs[0, 1])
    sc = draw_pseudotime_panel(ax2, metaspot_df, axis_labels, centers, edges, "Spatiotemporal Progression")
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.045, pad=0.03)
    cbar.set_label("Pseudotime", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax3 = fig.add_subplot(gs[1, :])
    im = ax3.imshow(heatmap_mat, aspect="auto", interpolation="nearest", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax3.set_title("Pseudotime (0 → 100): Metabolic Cascade Trajectory", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Pseudotime", fontsize=9)
    ax3.set_xticks(np.linspace(0, heatmap_mat.shape[1] - 1, 6))
    ax3.set_xticklabels([f"{int(v)}" for v in np.linspace(0, 100, 6)], fontsize=8)
    ax3.set_yticks([])
    ax3.tick_params(axis="both", labelsize=8)
    cbar2 = fig.colorbar(im, ax=ax3, fraction=0.02, pad=0.015)
    cbar2.set_label("Relative\nabundance", fontsize=9, rotation=0, labelpad=18)
    cbar2.ax.tick_params(labelsize=8)

    fig.suptitle("Spatiotemporal trajectory and metabolic cascade atlas", fontsize=16, fontweight="bold", y=0.985)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"Fig10_spatiotemporal_trajectory_metabolic_cascade.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # -------- single panels --------
    f1, a1 = plt.subplots(figsize=(6.2, 5.5))
    draw_stage_trajectory(a1, metaspot_df, axis_labels, centers, edges, "Trajectory by Clinical Stage")
    save_single_panel(f1, out_dir / "panel_top_left_stage_trajectory")
    plt.close(f1)

    f2, a2 = plt.subplots(figsize=(6.2, 5.5))
    sc2 = draw_pseudotime_panel(a2, metaspot_df, axis_labels, centers, edges, "Spatiotemporal Progression")
    cb = f2.colorbar(sc2, ax=a2, fraction=0.046, pad=0.04)
    cb.set_label("Pseudotime", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    save_single_panel(f2, out_dir / "panel_top_right_pseudotime")
    plt.close(f2)

    f3, a3 = plt.subplots(figsize=(11.2, 5.8))
    im3 = a3.imshow(heatmap_mat, aspect="auto", interpolation="nearest", cmap="RdYlBu_r", vmin=0, vmax=1)
    a3.set_title("Pseudotime (0 → 100): Metabolic Cascade Trajectory", fontsize=11, fontweight="bold")
    a3.set_xlabel("Pseudotime", fontsize=9)
    a3.set_xticks(np.linspace(0, heatmap_mat.shape[1] - 1, 6))
    a3.set_xticklabels([f"{int(v)}" for v in np.linspace(0, 100, 6)], fontsize=8)
    a3.set_yticks([])
    cb3 = f3.colorbar(im3, ax=a3, fraction=0.025, pad=0.02)
    cb3.set_label("Relative abundance", fontsize=9)
    cb3.ax.tick_params(labelsize=8)
    save_single_panel(f3, out_dir / "panel_bottom_pseudotime_heatmap")
    plt.close(f3)

    params = vars(args)
    params["n_files_found"] = len(files)
    params["n_specimens"] = len(specimen_map)
    params["n_common_markers"] = len(markers)
    params["n_metaspots"] = len(metaspot_df)
    params["embedding_method"] = method
    params["axis_labels"] = axis_labels
    params["groups_present"] = groups_present
    params["root_metaspot_index"] = int(root_idx)
    params["root_cluster"] = int(root_center)

    with open(out_dir / "run_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    readme = [
        "Figure 10: spatiotemporal trajectory and metabolic cascade atlas",
        f"Script: {script_name}.py",
        "",
        "Main idea:",
        "  1) align marker arrays within each specimen",
        "  2) aggregate consecutive signal points into metaspots",
        "  3) compute 2D embedding (UMAP if available, otherwise PCA)",
        "  4) infer pseudotime with a kNN graph shortest-path approach",
        "  5) summarize marker dynamics along pseudotime in a smoothed heatmap",
        "",
        "Outputs:",
        "  - Fig10_spatiotemporal_trajectory_metabolic_cascade.png/pdf/svg",
        "  - panel_top_left_stage_trajectory.*",
        "  - panel_top_right_pseudotime.*",
        "  - panel_bottom_pseudotime_heatmap.*",
        "  - metaspot_embedding_pseudotime_table.csv",
        "",
        "Recommended run:",
        "  python scripts/2d/fig10_spatiotemporal_trajectory_metabolic_cascade.py --data-dir data_2d --aggregation-size 40 --max-spots-per-specimen 100 --top-heatmap-markers 120",
        "",
        "Optional package:",
        "  pip install umap-learn",
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
