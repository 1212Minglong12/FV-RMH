from pathlib import Path
import re
import json
import shutil
import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import umap

warnings.filterwarnings("ignore")

# ============================================================
# Fig02: UMAP + KDE panel by differentiation group
#
# Top row:
#   Highlighted UMAP scatter for each group on gray background
#
# Bottom row:
#   2D KDE contour for each group on the same UMAP manifold
#
# Default groups:
#   Normal / High differentiated / Moderately differentiated / Poorly differentiated
#
# Run:
#   cd FV-RMH
#   python scripts\2d\fig02_umap_kde_by_grade_panels.py --data-dir data_2d
#
# Optional:
#   Only High/Middle/Low, no Normal:
#   python scripts\2d\fig02_umap_kde_by_grade_panels.py --data-dir data_2d --exclude-normal
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0

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
    "Normal": "#2C7BB6",   # blue
    "High": "#7BC8A4",     # soft green
    "Middle": "#F0C38E",   # soft orange
    "Low": "#E57F7F"       # soft red
}

LIGHT_GRAY = "#D9D9D9"


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

    marker = m.group(1)
    raw_group = m.group(2)
    patient = m.group(3)
    layer = int(m.group(4))
    group = GROUP_ALIASES.get(raw_group, raw_group)

    return {
        "marker": marker,
        "raw_group": raw_group,
        "group": group,
        "patient": patient,
        "layer": layer,
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


def patient_center_correction(X_df: pd.DataFrame, meta: pd.DataFrame):
    r"""
    Lightweight batch correction:
    subtract patient-specific mean and add global mean.
    """
    Xcorr = X_df.copy()
    grand_mean = Xcorr.mean(axis=0)

    for patient in meta["patient"].unique():
        idx = meta.index[meta["patient"] == patient]
        pmean = Xcorr.loc[idx].mean(axis=0)
        Xcorr.loc[idx] = Xcorr.loc[idx] - pmean + grand_mean

    Xcorr = Xcorr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xcorr


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


def make_superpoints(X, target_n=1000, random_state=42):
    X = fill_nonfinite_matrix(X)
    if X.shape[0] <= target_n:
        return X

    n_clusters = min(target_n, X.shape[0])
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=2048,
        n_init=5
    )
    labels = km.fit_predict(X)

    centers = np.zeros((n_clusters, X.shape[1]), dtype=float)
    for i in range(n_clusters):
        mask = labels == i
        if np.any(mask):
            centers[i] = X[mask].mean(axis=0)

    return fill_nonfinite_matrix(centers)


def prepare_embedding_matrix(per_sample, all_markers,
                             target_superpoints_per_sample=1000,
                             max_total_points=30000,
                             random_state=42):
    rows = []
    meta_rows = []

    for item in per_sample:
        X = fill_nonfinite_matrix(item["matrix"])
        X = np.log1p(np.maximum(X, 0))
        X = fill_nonfinite_matrix(X)
        X = make_superpoints(X, target_n=target_superpoints_per_sample, random_state=random_state)

        for i in range(X.shape[0]):
            rows.append(X[i])
            meta_rows.append({
                "sample_id": item["sample_id"],
                "patient": item["patient"],
                "group": item["group"],
                "layer": item["layer"]
            })

    X_df = pd.DataFrame(rows, columns=all_markers)
    meta = pd.DataFrame(meta_rows)

    if len(X_df) == 0:
        raise ValueError("No points generated.")

    if len(X_df) > max_total_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_df), size=max_total_points, replace=False)
        X_df = X_df.iloc[idx].reset_index(drop=True)
        meta = meta.iloc[idx].reset_index(drop=True)

    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X_df, meta


def run_umap(X_df: pd.DataFrame, meta: pd.DataFrame, out_dir: Path,
             n_pcs=30, random_state=42, batch_correct=True):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = pd.DataFrame(X_scaled, columns=X_df.columns)

    if batch_correct:
        X_scaled = patient_center_correction(X_scaled, meta)

    n_components = min(n_pcs, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    pcs = pca.fit_transform(X_scaled.values)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.18,
        metric="euclidean",
        random_state=random_state
    )
    coords = reducer.fit_transform(pcs)

    emb = meta.copy()
    emb["umap_1"] = coords[:, 0]
    emb["umap_2"] = coords[:, 1]
    emb.to_csv(out_dir / "umap_coordinates_for_kde_panels.csv", index=False, encoding="utf-8-sig")
    return emb


def balanced_plot_sample(emb: pd.DataFrame, by: str,
                         max_points=18000,
                         max_points_per_class=5000,
                         random_state=42):
    if len(emb) <= max_points:
        return emb.copy()

    rng = np.random.default_rng(random_state)
    selected = []

    for cls, sub in emb.groupby(by):
        n = min(len(sub), max_points_per_class)
        idx = rng.choice(sub.index.values, size=n, replace=False)
        selected.extend(idx.tolist())

    if len(selected) > max_points:
        selected = rng.choice(selected, size=max_points, replace=False).tolist()

    return emb.loc[selected].copy()


def style_axis(ax):
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(labelsize=9, length=3)
    ax.set_facecolor("white")


def draw_kde_contours(ax, x, y, color, levels=7, grid_size=220, bw_adjust=1.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 20:
        return

    xy = np.vstack([x, y])

    try:
        kde = gaussian_kde(xy)
        kde.set_bandwidth(kde.factor * bw_adjust)
    except Exception:
        return

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    xpad = (xmax - xmin) * 0.15 + 1e-6
    ypad = (ymax - ymin) * 0.15 + 1e-6

    xx, yy = np.meshgrid(
        np.linspace(xmin - xpad, xmax + xpad, grid_size),
        np.linspace(ymin - ypad, ymax + ypad, grid_size)
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    zmax = np.nanmax(zz)
    zmin = np.nanmin(zz)
    if not np.isfinite(zmax) or zmax <= 0:
        return

    contour_levels = np.linspace(max(zmin, zmax * 0.18), zmax, levels)
    contour_levels = np.unique(contour_levels)
    if len(contour_levels) < 2:
        return

    ax.contour(
        xx, yy, zz,
        levels=contour_levels,
        colors=color,
        linewidths=1.2
    )


def plot_umap_kde_panels(emb: pd.DataFrame, groups, out_dir: Path,
                         random_state=42,
                         max_points_for_plot=18000,
                         max_points_per_group=5000,
                         kde_levels=7):
    plot_df = balanced_plot_sample(
        emb,
        by="group",
        max_points=max_points_for_plot,
        max_points_per_class=max_points_per_group,
        random_state=random_state
    )

    n_groups = len(groups)
    fig, axes = plt.subplots(2, n_groups, figsize=(4.1 * n_groups, 8.0), sharex=True, sharey=True)
    if n_groups == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    xmin, xmax = plot_df["umap_1"].min(), plot_df["umap_1"].max()
    ymin, ymax = plot_df["umap_2"].min(), plot_df["umap_2"].max()
    xpad = (xmax - xmin) * 0.05
    ypad = (ymax - ymin) * 0.05

    for i, g in enumerate(groups):
        color = GROUP_COLORS[g]
        display = GROUP_DISPLAY[g]

        # Top row: highlighted scatter on gray background
        ax_top = axes[0, i]
        ax_top.scatter(
            plot_df["umap_1"], plot_df["umap_2"],
            s=6, c=LIGHT_GRAY, alpha=0.55, edgecolors="none", rasterized=True
        )
        sub = plot_df[plot_df["group"] == g]
        ax_top.scatter(
            sub["umap_1"], sub["umap_2"],
            s=6, c=color, alpha=0.85, edgecolors="none", rasterized=True
        )
        ax_top.set_title(display, fontsize=12, fontweight="bold", pad=6)
        style_axis(ax_top)
        ax_top.set_xlim(xmin - xpad, xmax + xpad)
        ax_top.set_ylim(ymin - ypad, ymax + ypad)

        # Bottom row: KDE contour on gray background
        ax_bot = axes[1, i]
        ax_bot.scatter(
            plot_df["umap_1"], plot_df["umap_2"],
            s=6, c=LIGHT_GRAY, alpha=0.40, edgecolors="none", rasterized=True
        )
        draw_kde_contours(
            ax_bot,
            sub["umap_1"].values,
            sub["umap_2"].values,
            color=color,
            levels=kde_levels,
            grid_size=220,
            bw_adjust=1.0
        )
        style_axis(ax_bot)
        ax_bot.set_xlim(xmin - xpad, xmax + xpad)
        ax_bot.set_ylim(ymin - ypad, ymax + ypad)

    # Axis labels only on leftmost / middle bottom similar to panel style
    axes[0, 0].set_ylabel("UMAP 2", fontsize=11)
    axes[1, 0].set_ylabel("UMAP 2", fontsize=11)

    mid = n_groups // 2
    axes[0, mid].set_xlabel("UMAP 1", fontsize=11)
    axes[1, mid].set_xlabel("UMAP 1", fontsize=11)

    plt.subplots_adjust(wspace=0.03, hspace=0.08)

    for ext in ["png", "pdf", "svg"]:
        plt.savefig(
            out_dir / f"Fig2_UMAP_KDE_by_grade_panels.{ext}",
            dpi=600 if ext == "png" else None,
            bbox_inches="tight"
        )
    plt.close(fig)

    # Also save legend-only figure
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=GROUP_COLORS[g], markeredgecolor="none",
               markersize=10, label=GROUP_DISPLAY[g])
        for g in groups
    ]

    fig, ax = plt.subplots(figsize=(4.6, 0.8 + 0.55 * len(groups)))
    ax.axis("off")
    ax.legend(handles=handles, loc="center left", frameon=False, fontsize=12)
    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(out_dir / f"Fig2_group_legend.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="UMAP + KDE panel by differentiation grade from nested 2D Raman .mat files")

    parser.add_argument("--data-dir", type=str, default="data_2d")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--target-layer", type=int, default=1)

    parser.add_argument("--exclude-normal", action="store_true")
    parser.add_argument("--no-batch-correct", action="store_true")

    parser.add_argument("--target-superpoints-per-sample", type=int, default=900)
    parser.add_argument("--max-total-points", type=int, default=30000)
    parser.add_argument("--min-markers-per-pixel", type=int, default=10)
    parser.add_argument("--min-nonzero-features", type=int, default=3)

    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--max-points-for-plot", type=int, default=18000)
    parser.add_argument("--max-points-per-group", type=int, default=5000)
    parser.add_argument("--kde-levels", type=int, default=7)

    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    script_name = Path(__file__).stem
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / script_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    groups = ["Normal", "High", "Middle", "Low"]
    if args.exclude_normal:
        groups = ["High", "Middle", "Low"]

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
    pd.DataFrame(skipped_parse).to_csv(
        out_dir / "skipped_unrecognized_files.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if meta_df.empty:
        raise SystemExit("No recognized files. Example filename: CD98-High-PatientB_layer1.mat")

    sample_dict, recognized_df, skipped_df = build_sample_marker_dict(
        meta_df,
        target_layer=args.target_layer,
        allowed_groups=groups
    )

    recognized_df.to_csv(out_dir / "recognized_input_files.csv", index=False, encoding="utf-8-sig")
    if not skipped_df.empty:
        skipped_df.to_csv(out_dir / "skipped_after_loading.csv", index=False, encoding="utf-8-sig")

    if recognized_df.empty:
        raise SystemExit("Files were parsed but none were valid after loading.")

    sample_summary = recognized_df.groupby(["group", "patient"])["marker"].nunique().reset_index(name="n_markers")
    sample_summary.to_csv(out_dir / "sample_metadata_summary.csv", index=False, encoding="utf-8-sig")

    per_sample, all_markers = assemble_sample_matrix(
        sample_dict,
        min_markers_per_pixel=args.min_markers_per_pixel,
        min_nonzero_features=args.min_nonzero_features
    )

    pd.DataFrame({"marker": all_markers}).to_csv(
        out_dir / "used_markers.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if len(per_sample) < 2:
        raise SystemExit("Too few valid samples after assembling matrices.")

    X_df, meta = prepare_embedding_matrix(
        per_sample,
        all_markers,
        target_superpoints_per_sample=args.target_superpoints_per_sample,
        max_total_points=args.max_total_points,
        random_state=args.random_state
    )

    meta.to_csv(out_dir / "metadata_for_umap_kde_panels.csv", index=False, encoding="utf-8-sig")

    emb = run_umap(
        X_df,
        meta,
        out_dir,
        n_pcs=args.n_pcs,
        random_state=args.random_state,
        batch_correct=not args.no_batch_correct
    )

    plot_umap_kde_panels(
        emb,
        groups=groups,
        out_dir=out_dir,
        random_state=args.random_state,
        max_points_for_plot=args.max_points_for_plot,
        max_points_per_group=args.max_points_per_group,
        kde_levels=args.kde_levels
    )

    params = vars(args)
    params["script_name"] = script_name
    params["groups_present"] = groups
    params["n_mat_files_found"] = len(mat_files)
    params["n_recognized_files"] = len(recognized_df)
    params["n_valid_samples"] = len(per_sample)
    params["n_embedding_input_points"] = len(X_df)

    with open(out_dir / "run_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    readme = [
        "UMAP + KDE panel output folder.",
        f"Script: {script_name}.py",
        "",
        "Main figure:",
        "  Fig2_UMAP_KDE_by_grade_panels.png / pdf / svg",
        "",
        "Legend figure:",
        "  Fig2_group_legend.png / pdf / svg",
        "",
        "Default groups:",
        "  Normal / High differentiated / Moderately differentiated / Poorly differentiated",
        "",
        "Use --exclude-normal for only High / Middle / Low."
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
