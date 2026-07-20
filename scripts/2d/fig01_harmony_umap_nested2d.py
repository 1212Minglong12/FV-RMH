# -*- coding: utf-8 -*-
r"""
Fig01: 2D Raman multi-sample integration, denoising, Harmony batch correction and UMAP.

Designed for nested 2D Raman mat files, e.g.:

raman_python_test/
├── scripts/
│   └── fig01_harmony_umap_nested2d.py
├── data_2d/
│   ├── Acetyl_CoA/
│   │   ├── Acetyl_CoA-High-PatientA_layer1.mat
│   │   ├── Acetyl_CoA-High-PatientA_layer1.tif
│   │   └── ...
│   ├── CD98/
│   │   ├── CD98-Low-PatientB_layer1.mat
│   │   └── ...
│   └── ...
└── results/
    └── fig01_harmony_umap/

Run from the project root:
python scripts\2d\fig01_harmony_umap_nested2d.py --data-dir data_2d --out-dir results\fig01_harmony_umap

File naming rule:
    Marker-Group-Patient_layer1.mat
Example:
    CD98-High-PatientB_layer1.mat
    Acetyl_CoA-High-PatientA_layer1.mat

"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# -----------------------------
# Optional packages
# -----------------------------
try:
    import harmonypy as hm
    HARMONY_AVAILABLE = True
except Exception:
    HARMONY_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except Exception:
    H5PY_AVAILABLE = False


# -----------------------------
# Plot style
# -----------------------------
def set_plot_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.linewidth"] = 1.1
    plt.rcParams["xtick.major.width"] = 1.0
    plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11


# -----------------------------
# Naming and group handling
# -----------------------------
GROUP_ALIAS = {
    "Normal": "Normal",
    "NAT": "Normal",
    "Control": "Normal",
    "Healthy": "Normal",
    "High": "High",
    "WD": "High",
    "Well": "High",
    "WellDifferentiated": "High",
    "Middle": "Middle",
    "Medium": "Middle",
    "Moderate": "Middle",
    "MD": "Middle",
    "Low": "Low",
    "Poor": "Low",
    "Poorly": "Low",
    "PD": "Low",
}

GROUP_ORDER = ["Normal", "High", "Middle", "Low"]
GROUP_LABEL = {
    "Normal": "Normal",
    "High": "High",
    "Middle": "Middle",
    "Low": "Low",
}
GROUP_COLORS = {
    "Normal": "#2C7FB8",
    "High": "#31A354",
    "Middle": "#FF7F00",
    "Low": "#D62728",
}


def parse_filename(filename: str) -> Optional[Tuple[str, str, str, int]]:
    """Parse Marker-Group-Patient_layerN.mat from the end.

    Marker may contain hyphens. Patient may contain letters/underscores.
    """
    name = Path(filename).name
    if not name.lower().endswith(".mat"):
        return None

    group_pattern = "|".join(sorted(map(re.escape, GROUP_ALIAS.keys()), key=len, reverse=True))
    pattern = rf"^(.+)-({group_pattern})-(.+)_layer(\d+)\.mat$"
    m = re.match(pattern, name, flags=re.IGNORECASE)
    if not m:
        return None

    marker = m.group(1)
    raw_group = m.group(2)
    patient = m.group(3)
    layer = int(m.group(4))

    # Preserve canonical alias irrespective of case
    group = None
    for k, v in GROUP_ALIAS.items():
        if raw_group.lower() == k.lower():
            group = v
            break
    if group is None:
        group = raw_group

    # Clean marker/patient strings for stable IDs
    marker = marker.strip()
    patient = patient.strip()

    return marker, group, patient, layer


# -----------------------------
# MAT reading
# -----------------------------
def _first_numeric_from_h5(file_path: Path) -> np.ndarray:
    if not H5PY_AVAILABLE:
        raise RuntimeError("h5py is not installed; cannot read MATLAB v7.3 HDF5 .mat files.")
    with h5py.File(file_path, "r") as f:
        candidates = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                try:
                    arr = obj[()]
                    if np.issubdtype(arr.dtype, np.number):
                        candidates.append((name, arr))
                except Exception:
                    pass

        f.visititems(visitor)
        if not candidates:
            raise ValueError(f"No numeric dataset found in {file_path.name}")
        # Prefer coeffVector if present
        for name, arr in candidates:
            if "coeffVector" in name:
                return np.asarray(arr)
        return np.asarray(candidates[0][1])


def read_mat_vector(file_path: Path, preferred_key: str = "coeffVector") -> np.ndarray:
    """Read 1D numeric vector from .mat.

    Prefer 'coeffVector'. If missing, use first numeric variable.
    Also supports sparse arrays.
    """
    try:
        mat = sio.loadmat(file_path)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        data = None
        if preferred_key in mat:
            data = mat[preferred_key]
        else:
            for k in keys:
                v = mat[k]
                if sparse.issparse(v):
                    data = v.toarray()
                    break
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                    data = v
                    break
        if data is None:
            raise ValueError(f"No numeric variable found in {file_path.name}")
    except NotImplementedError:
        data = _first_numeric_from_h5(file_path)
    except ValueError as e:
        # scipy may fail on v7.3 files with unknown mat format
        if "Unknown mat file type" in str(e) or "Please use HDF reader" in str(e):
            data = _first_numeric_from_h5(file_path)
        else:
            raise

    if sparse.issparse(data):
        data = data.toarray()
    vec = np.asarray(data).squeeze().astype(float).ravel()
    return vec


# -----------------------------
# Data scanning
# -----------------------------
def scan_mat_files(data_dir: Path, target_layer: Optional[int]) -> pd.DataFrame:
    all_files = sorted(data_dir.rglob("*.mat"))
    rows = []
    skipped = []
    for fp in all_files:
        parsed = parse_filename(fp.name)
        if parsed is None:
            skipped.append(str(fp.relative_to(data_dir)))
            continue
        marker, group, patient, layer = parsed
        if target_layer is not None and layer != target_layer:
            continue
        rows.append({
            "file_path": str(fp),
            "relative_path": str(fp.relative_to(data_dir)),
            "marker": marker,
            "group": group,
            "patient": patient,
            "layer": layer,
            "sample_id": f"{group}-{patient}-layer{layer}",
        })
    df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame({"skipped_relative_path": skipped})
    return df, skipped_df


# -----------------------------
# Preprocessing and aggregation
# -----------------------------
def robust_clip_matrix(X: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    """Clip each feature column by quantiles, ignoring NaNs."""
    X = X.astype(float, copy=True)
    n_features = X.shape[1]
    for j in range(n_features):
        col = X[:, j]
        finite = np.isfinite(col)
        if finite.sum() < 10:
            continue
        lo = np.nanpercentile(col[finite], low_q)
        hi = np.nanpercentile(col[finite], high_q)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            col[finite] = np.clip(col[finite], lo, hi)
            X[:, j] = col
    return X


def build_sample_matrix(sample_files: pd.DataFrame, marker_order: List[str], remove_zero: bool,
                        min_nonzero_features: int, low_clip: float, high_clip: float) -> Optional[np.ndarray]:
    """Build pixels × markers matrix for one patient-layer sample."""
    marker_to_vec: Dict[str, np.ndarray] = {}
    min_len = None

    for _, row in sample_files.iterrows():
        marker = row["marker"]
        fp = Path(row["file_path"])
        try:
            vec = read_mat_vector(fp)
        except Exception as e:
            print(f"[WARN] failed reading {fp.name}: {e}")
            continue
        if vec.size == 0:
            continue
        marker_to_vec[marker] = vec
        min_len = vec.size if min_len is None else min(min_len, vec.size)

    if not marker_to_vec or min_len is None or min_len < 10:
        return None

    X = np.full((min_len, len(marker_order)), np.nan, dtype=np.float32)
    marker_index = {m: i for i, m in enumerate(marker_order)}
    for marker, vec in marker_to_vec.items():
        if marker not in marker_index:
            continue
        X[:, marker_index[marker]] = vec[:min_len]

    # Drop all-NaN pixels
    valid = ~np.all(~np.isfinite(X), axis=1)
    X = X[valid]
    if X.shape[0] < 10:
        return None

    # Fill missing per marker by median within sample
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(X))
    if len(inds[0]) > 0:
        X[inds] = med[inds[1]]

    if remove_zero:
        nonzero_count = np.sum(X > 0, axis=1)
        X = X[nonzero_count >= min_nonzero_features]
        if X.shape[0] < 10:
            return None

    X = robust_clip_matrix(X, low_clip, high_clip)
    return X.astype(np.float32)


def deterministic_subsample(X: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if X.shape[0] <= max_n:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_n, replace=False)
    return X[idx]


def aggregate_superpoints(X: np.ndarray, target_n: int, seed: int, max_pixels_before: int) -> np.ndarray:
    """Aggregate pixels into feature-space superpoints using MiniBatchKMeans.

    This is a practical approximation of micro-superpoint aggregation.
    """
    if X.shape[0] <= target_n:
        return X

    X_fit = deterministic_subsample(X, max_pixels_before, seed)
    n_clusters = min(target_n, max(2, X_fit.shape[0] // 5))
    n_clusters = min(n_clusters, X_fit.shape[0])

    # Pre-standardize within sample for stable kmeans, but return original-scale centers.
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(4096, max(512, n_clusters * 4)),
        n_init=3,
        max_iter=80,
        reassignment_ratio=0.01,
        verbose=0,
    )
    km.fit(X_fit_scaled)

    centers_scaled = km.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    return centers.astype(np.float32)


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(args):
    set_plot_style()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy this script into the result folder for reproducibility.
    try:
        this_file = Path(__file__).resolve()
        shutil.copy2(this_file, out_dir / this_file.name)
    except Exception:
        pass

    with open(out_dir / "run_parameters.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Fig01 2D Raman Harmony + UMAP pipeline")
    print("Data directory:", data_dir)
    print("Output directory:", out_dir)
    print("Harmony available:", HARMONY_AVAILABLE)
    print("UMAP available:", UMAP_AVAILABLE)
    print("=" * 80)

    files_df, skipped_df = scan_mat_files(data_dir, args.target_layer)
    files_df.to_csv(out_dir / "recognized_input_files.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(out_dir / "skipped_unrecognized_files.csv", index=False, encoding="utf-8-sig")

    if files_df.empty:
        raise RuntimeError(
            "No valid .mat files were recognized. Check naming rule: Marker-Group-Patient_layer1.mat\n"
            "Example: CD98-High-PatientB_layer1.mat\n"
            f"Scanned folder: {data_dir}"
        )

    print(f"Recognized .mat files: {len(files_df)}")
    print(f"Unique markers: {files_df['marker'].nunique()}")
    print(f"Unique patients: {files_df['patient'].nunique()}")
    print(f"Unique samples: {files_df['sample_id'].nunique()}")
    print("Groups:", files_df["group"].value_counts().to_dict())

    # Marker filtering: keep markers appearing in at least min_samples_for_marker samples.
    marker_counts = files_df.groupby("marker")["sample_id"].nunique().sort_values(ascending=False)
    marker_counts.to_csv(out_dir / "marker_sample_counts.csv", encoding="utf-8-sig")

    marker_order = marker_counts[marker_counts >= args.min_samples_for_marker].index.tolist()
    if len(marker_order) < 3:
        print("[WARN] Very few markers pass min_samples_for_marker. Using all markers instead.")
        marker_order = marker_counts.index.tolist()
    pd.DataFrame({"marker": marker_order}).to_csv(out_dir / "used_markers.csv", index=False, encoding="utf-8-sig")

    print(f"Markers used for integration: {len(marker_order)}")

    # Build per sample superpoint matrices.
    X_blocks = []
    meta_blocks = []
    sample_summary_rows = []

    grouped = list(files_df.groupby("sample_id"))
    for sample_idx, (sample_id, sdf) in enumerate(grouped):
        group = sdf["group"].iloc[0]
        patient = sdf["patient"].iloc[0]
        layer = int(sdf["layer"].iloc[0])

        X_sample = build_sample_matrix(
            sdf,
            marker_order=marker_order,
            remove_zero=args.remove_zero,
            min_nonzero_features=args.min_nonzero_features,
            low_clip=args.low_clip,
            high_clip=args.high_clip,
        )

        if X_sample is None:
            print(f"[WARN] skip sample {sample_id}: insufficient valid pixels")
            sample_summary_rows.append({
                "sample_id": sample_id,
                "group": group,
                "patient": patient,
                "layer": layer,
                "n_markers_found": sdf["marker"].nunique(),
                "n_pixels_after_filter": 0,
                "n_superpoints": 0,
                "status": "skipped",
            })
            continue

        n_pixels = X_sample.shape[0]
        X_super = aggregate_superpoints(
            X_sample,
            target_n=args.target_superpoints_per_sample,
            seed=args.random_seed + sample_idx,
            max_pixels_before=args.max_pixels_before_aggregation,
        )

        X_blocks.append(X_super)
        n_super = X_super.shape[0]
        meta = pd.DataFrame({
            "sample_id": sample_id,
            "group": group,
            "patient": patient,
            "layer": layer,
            "superpoint_index": np.arange(n_super),
        })
        meta_blocks.append(meta)
        sample_summary_rows.append({
            "sample_id": sample_id,
            "group": group,
            "patient": patient,
            "layer": layer,
            "n_markers_found": sdf["marker"].nunique(),
            "n_pixels_after_filter": int(n_pixels),
            "n_superpoints": int(n_super),
            "status": "used",
        })
        print(f"{sample_idx+1:03d}/{len(grouped)} {sample_id}: pixels={n_pixels}, superpoints={n_super}, markers={sdf['marker'].nunique()}")

    sample_summary = pd.DataFrame(sample_summary_rows)
    sample_summary.to_csv(out_dir / "sample_metadata_summary.csv", index=False, encoding="utf-8-sig")

    if not X_blocks:
        raise RuntimeError("No sample could be converted into superpoints. Check MAT variables and filtering settings.")

    X = np.vstack(X_blocks).astype(np.float32)
    meta_df = pd.concat(meta_blocks, ignore_index=True)

    print("Global superpoint matrix:", X.shape)

    # Limit total points for memory/time.
    if X.shape[0] > args.max_total_points:
        rng = np.random.default_rng(args.random_seed)
        idx = rng.choice(X.shape[0], size=args.max_total_points, replace=False)
        X = X[idx]
        meta_df = meta_df.iloc[idx].reset_index(drop=True)
        print(f"Subsample global superpoints to {args.max_total_points} for plotting/integration.")

    # Log transform and z-score.
    X = np.where(X < 0, 0, X)
    X_log = np.log1p(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # Remove zero-variance markers after scaling if any.
    finite_feature = np.isfinite(X_scaled).all(axis=0)
    std_feature = np.nanstd(X_scaled, axis=0) > 0
    keep = finite_feature & std_feature
    X_scaled = X_scaled[:, keep]
    final_markers = [m for m, k in zip(marker_order, keep) if k]
    pd.DataFrame({"marker": final_markers}).to_csv(out_dir / "final_markers_after_scaling.csv", index=False, encoding="utf-8-sig")

    n_components = min(args.n_pcs, X_scaled.shape[1], X_scaled.shape[0] - 1)
    if n_components < 2:
        raise RuntimeError("Too few features/samples for PCA.")

    pca = PCA(n_components=n_components, random_state=args.random_seed)
    pcs = pca.fit_transform(X_scaled)
    explained = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(n_components)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })
    explained.to_csv(out_dir / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")

    # Harmony batch correction by patient.
    if HARMONY_AVAILABLE and args.use_harmony:
        print("Running Harmony correction by patient...")
        harmony_meta = meta_df[["patient"]].copy()
        try:
            ho = hm.run_harmony(
                pcs,
                harmony_meta,
                vars_use=["patient"],
                max_iter_harmony=args.harmony_max_iter,
                verbose=False,
            )
            Z_corr = ho.Z_corr.T
            harmony_used = True
        except Exception as e:
            print("[WARN] Harmony failed, using PCA coordinates instead:", e)
            Z_corr = pcs
            harmony_used = False
    else:
        if not HARMONY_AVAILABLE:
            print("[WARN] harmonypy not installed. Using PCA coordinates as fallback.")
        Z_corr = pcs
        harmony_used = False

    # UMAP on corrected low-dimensional embedding.
    if not UMAP_AVAILABLE:
        raise RuntimeError("umap-learn is not installed. Install by: python -m pip install umap-learn")

    print("Running UMAP...")
    n_neighbors = min(args.umap_neighbors, max(2, Z_corr.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.random_seed,
        verbose=False,
    )
    umap_xy = reducer.fit_transform(Z_corr[:, :min(args.umap_input_dims, Z_corr.shape[1])])

    coords = meta_df.copy().reset_index(drop=True)
    coords["harmony_1"] = Z_corr[:, 0]
    coords["harmony_2"] = Z_corr[:, 1]
    coords["umap_1"] = umap_xy[:, 0]
    coords["umap_2"] = umap_xy[:, 1]
    coords.to_csv(out_dir / "Fig1_harmony_umap_coordinates.csv", index=False, encoding="utf-8-sig")

    # Plot.
    fig = plt.figure(figsize=(8.2, 10.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    # Patient colors. Use tab20 / hsv fallback.
    patients = sorted(coords["patient"].unique().tolist())
    if len(patients) <= 20:
        patient_cmap = plt.get_cmap("tab20")
        patient_colors = {p: patient_cmap(i % 20) for i, p in enumerate(patients)}
    else:
        patient_cmap = plt.get_cmap("hsv")
        patient_colors = {p: patient_cmap(i / max(1, len(patients))) for i, p in enumerate(patients)}

    # To reduce overplotting, plot in randomized order.
    rng = np.random.default_rng(args.random_seed)
    order = rng.permutation(len(coords))
    coords_plot = coords.iloc[order]

    for p in patients:
        sub = coords_plot[coords_plot["patient"] == p]
        if sub.empty:
            continue
        ax1.scatter(
            sub["harmony_1"], sub["harmony_2"],
            s=args.point_size,
            alpha=args.point_alpha,
            color=patient_colors[p],
            linewidths=0,
            rasterized=True,
            label=p,
        )

    title1 = "Harmony: By Patient (Batch Corrected)" if harmony_used else "PCA: By Patient (Harmony unavailable/fallback)"
    ax1.set_title(title1, fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel("harmony_1" if harmony_used else "PC1")
    ax1.set_ylabel("harmony_2" if harmony_used else "PC2")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    if len(patients) <= args.max_patient_legend:
        ax1.legend(
            frameon=False,
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            fontsize=8,
            markerscale=4,
            title="Patient",
            title_fontsize=9,
        )
    else:
        # Avoid crowded patient legend.
        ax1.text(
            0.99, 0.02,
            f"Patients: n={len(patients)}",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.3",
        )

    # UMAP by pathology group.
    groups_existing = [g for g in GROUP_ORDER if g in coords_plot["group"].unique()]
    groups_existing += [g for g in sorted(coords_plot["group"].unique()) if g not in groups_existing]
    for g in groups_existing:
        sub = coords_plot[coords_plot["group"] == g]
        if sub.empty:
            continue
        ax2.scatter(
            sub["umap_1"], sub["umap_2"],
            s=args.point_size,
            alpha=args.point_alpha,
            color=GROUP_COLORS.get(g, "#777777"),
            linewidths=0,
            rasterized=True,
            label=GROUP_LABEL.get(g, g),
        )

    ax2.set_title("UMAP: By Pathology (Biological Trajectory)", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("umap_1")
    ax2.set_ylabel("umap_2")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=9,
        markerscale=4,
        title="Pathology",
        title_fontsize=9,
    )

    fig.subplots_adjust(right=0.78)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"Fig1_2D_Harmony_UMAP_by_patient_pathology.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Extra: individual panels for easier layout editing.
    for ax_kind in ["patient", "pathology"]:
        fig2, ax = plt.subplots(figsize=(6.4, 5.2))
        if ax_kind == "patient":
            for p in patients:
                sub = coords_plot[coords_plot["patient"] == p]
                ax.scatter(sub["harmony_1"], sub["harmony_2"], s=args.point_size, alpha=args.point_alpha,
                           color=patient_colors[p], linewidths=0, rasterized=True, label=p)
            ax.set_title(title1, fontsize=12, fontweight="bold", pad=8)
            ax.set_xlabel("harmony_1" if harmony_used else "PC1")
            ax.set_ylabel("harmony_2" if harmony_used else "PC2")
            if len(patients) <= args.max_patient_legend:
                ax.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8, markerscale=4)
        else:
            for g in groups_existing:
                sub = coords_plot[coords_plot["group"] == g]
                ax.scatter(sub["umap_1"], sub["umap_2"], s=args.point_size, alpha=args.point_alpha,
                           color=GROUP_COLORS.get(g, "#777777"), linewidths=0, rasterized=True, label=GROUP_LABEL.get(g, g))
            ax.set_title("UMAP: By Pathology (Biological Trajectory)", fontsize=12, fontweight="bold", pad=8)
            ax.set_xlabel("umap_1")
            ax.set_ylabel("umap_2")
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=9, markerscale=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for ext in ["png", "pdf", "svg"]:
            fig2.savefig(out_dir / f"Fig1_{ax_kind}_panel.{ext}", dpi=600, bbox_inches="tight")
        plt.close(fig2)

    print("\nDone. Main outputs:")
    print(" -", out_dir / "Fig1_2D_Harmony_UMAP_by_patient_pathology.png")
    print(" -", out_dir / "Fig1_2D_Harmony_UMAP_by_patient_pathology.pdf")
    print(" -", out_dir / "Fig1_2D_Harmony_UMAP_by_patient_pathology.svg")
    print(" -", out_dir / "recognized_input_files.csv")
    print(" -", out_dir / "sample_metadata_summary.csv")
    print(" -", out_dir / "Fig1_harmony_umap_coordinates.csv")


def parse_args():
    p = argparse.ArgumentParser(description="2D Raman Harmony + UMAP pipeline for nested marker folders")
    p.add_argument("--data-dir", default="data_2d", help="2D data folder; recursively scans *.mat")
    p.add_argument("--out-dir", default="results/fig01_harmony_umap", help="output folder for this figure")
    p.add_argument("--target-layer", type=int, default=1, help="which layer to use; default 1 for 2D data")

    p.add_argument("--low-clip", type=float, default=0.5, help="low percentile clipping per marker")
    p.add_argument("--high-clip", type=float, default=99.5, help="high percentile clipping per marker")
    p.add_argument("--remove-zero", action="store_true", default=True, help="remove background-like pixels")
    p.add_argument("--min-nonzero-features", type=int, default=3, help="minimum number of positive markers per pixel")

    p.add_argument("--min-samples-for-marker", type=int, default=3, help="marker must exist in at least this many samples")
    p.add_argument("--target-superpoints-per-sample", type=int, default=1200, help="superpoints per patient-layer sample")
    p.add_argument("--max-pixels-before-aggregation", type=int, default=60000, help="max pixels per sample for kmeans fitting")
    p.add_argument("--max-total-points", type=int, default=50000, help="max total points for PCA/Harmony/UMAP")

    p.add_argument("--n-pcs", type=int, default=30, help="number of PCA components before Harmony")
    p.add_argument("--use-harmony", action="store_true", default=True, help="use harmonypy if installed")
    p.add_argument("--harmony-max-iter", type=int, default=20, help="max Harmony iterations")
    p.add_argument("--umap-input-dims", type=int, default=20, help="number of corrected PCs for UMAP")
    p.add_argument("--umap-neighbors", type=int, default=30, help="UMAP n_neighbors")
    p.add_argument("--umap-min-dist", type=float, default=0.18, help="UMAP min_dist")

    p.add_argument("--point-size", type=float, default=1.2, help="scatter point size")
    p.add_argument("--point-alpha", type=float, default=0.55, help="scatter alpha")
    p.add_argument("--max-patient-legend", type=int, default=14, help="show patient legend only if patient number <= this")
    p.add_argument("--random-seed", type=int, default=42, help="random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(args)
    except Exception as e:
        print("\n[ERROR]", e)
        print("\nTroubleshooting:")
        print("1) Make sure you run from project root, e.g. FV-RMH")
        print("2) Data files should be under data_2d, possibly nested by marker folders.")
        print("3) File name should be Marker-Group-Patient_layer1.mat, e.g. CD98-High-PatientB_layer1.mat")
        print("4) Install packages: python -m pip install numpy pandas scipy matplotlib scikit-learn umap-learn harmonypy")
        raise
