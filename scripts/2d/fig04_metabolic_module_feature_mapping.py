from pathlib import Path
import re
import json
import shutil
import argparse
import warnings
import math

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import umap

warnings.filterwarnings("ignore")

# ============================================================
# Fig04: Global manifold spatial mapping of dominant metabolic modules
#
# Outputs:
#   1) One overall large figure with 12 modules x group panels
#   2) One single figure for each metabolic module
#
# Default grouping:
#   High / Middle / Low
#
# Optional:
#   add Normal with --include-normal
#
# Run:
#   cd FV-RMH
#   python scripts\2d\fig04_metabolic_module_feature_mapping.py --data-dir data_2d
#
# If needed:
#   python -m pip install umap-learn scipy scikit-learn matplotlib pandas numpy
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
    "High": "High",
    "Middle": "Moderate",
    "Low": "Poor"
}

MODULE_ORDER = [
    "Extracellular Matrix",
    "Amino Acids",
    "Interleukins",
    "Calcification",
    "Infection & Immunity",
    "Hormones",
    "Carotenoids",
    "Other Metabolites",
    "Glycolipid Metabolism",
    "Vitamins",
    "Repair & Cytokines",
    "Oxidative Stress"
]

MODULE_SHORT = {
    "Extracellular Matrix": "ECM",
    "Amino Acids": "Amino acids",
    "Interleukins": "Interleukins",
    "Calcification": "Calcification",
    "Infection & Immunity": "Infection & immunity",
    "Hormones": "Hormones",
    "Carotenoids": "Carotenoids",
    "Other Metabolites": "Other metabolites",
    "Glycolipid Metabolism": "Glycolipid metabolism",
    "Vitamins": "Vitamins",
    "Repair & Cytokines": "Repair & cytokines",
    "Oxidative Stress": "Oxidative stress"
}

MODULE_RULES = {
    "Extracellular Matrix": [
        "acta2", "collagen", "laminin", "elastin", "fibronectin", "syndecan", "versican", "decorin", "biglycan"
    ],
    "Amino Acids": [
        "aspartic", "proline", "methionine", "glutamine", "asparagine", "alanine", "glycine", "serine",
        "tryptophan", "tyrosine", "valine", "leucine", "isoleucine", "lysine", "histidine", "threonine"
    ],
    "Interleukins": [
        "interleukin", "il_", "il-", "cxcl", "ccl", "tnf", "ifn"
    ],
    "Calcification": [
        "calcium", "hydroxyapatite", "phosphate", "mineral", "alkalinephosphatase"
    ],
    "Infection & Immunity": [
        "cd68", "cd31", "cd98", "b7h3", "b7-h3", "b7_h3", "pdl1", "cd4", "cd8", "immune", "lysozyme", "ccl7"
    ],
    "Hormones": [
        "estrogen", "progesterone", "androgen", "cortisol", "thyroid", "hormone"
    ],
    "Carotenoids": [
        "carotene", "betacarotene", "beta-carotene"
    ],
    "Other Metabolites": [
        "formaldehyde", "fumarate", "nicotinamide", "succinate", "citrate", "malate", "bmp4", "aspergillus", "butyricacid"
    ],
    "Glycolipid Metabolism": [
        "acetyl", "fructose", "glucose", "phosphoglycerate", "lipid", "glycolipid", "cholesterol",
        "fatty", "triglyceride", "sphing", "glycerol", "nadh"
    ],
    "Vitamins": [
        "vitamin", "pyridoxine", "albumin"
    ],
    "Repair & Cytokines": [
        "repair", "cytokine", "wound", "healing", "tgfb", "tgf-b", "tgf", "cdkn1a", "cdh1"
    ],
    "Oxidative Stress": [
        "superoxidedismutase", "sod", "oxidative", "ros", "glutathione", "cml"
    ]
}


def normalize_marker_name(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def assign_module_to_marker(marker_name):
    m = normalize_marker_name(marker_name)
    matched = []

    for module, keywords in MODULE_RULES.items():
        for kw in keywords:
            kw_norm = normalize_marker_name(kw)
            if kw_norm and kw_norm in m:
                matched.append(module)
                break

    if len(matched) == 0:
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


def patient_center_correction(X_df: pd.DataFrame, meta: pd.DataFrame):
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


def make_superpoints(X, target_n=900, random_state=42):
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
                             target_superpoints_per_sample=900,
                             max_total_points=24000,
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


def run_umap_embedding(X_df: pd.DataFrame, meta: pd.DataFrame, n_pcs=30, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = pd.DataFrame(X_scaled, columns=X_df.columns)
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
    return emb, X_scaled


def compute_module_scores(X_scaled_df: pd.DataFrame):
    marker_to_module = {m: assign_module_to_marker(m) for m in X_scaled_df.columns}
    mapping_df = pd.DataFrame({
        "marker": list(marker_to_module.keys()),
        "module": list(marker_to_module.values())
    })

    score_df = pd.DataFrame(index=X_scaled_df.index)
    for module in MODULE_ORDER:
        cols = [m for m, mod in marker_to_module.items() if mod == module and m in X_scaled_df.columns]
        if len(cols) == 0:
            score_df[module] = 0.0
        elif len(cols) == 1:
            score_df[module] = X_scaled_df[cols[0]].values
        else:
            score_df[module] = X_scaled_df[cols].mean(axis=1).values

    return score_df, mapping_df


def robust_minmax_per_column(df, lower_q=0.02, upper_q=0.98):
    out = pd.DataFrame(index=df.index)
    params = []

    for col in df.columns:
        x = df[col].astype(float).values
        lo = np.nanquantile(x, lower_q)
        hi = np.nanquantile(x, upper_q)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            norm = np.zeros_like(x) + 0.5
        else:
            norm = (x - lo) / (hi - lo)
            norm = np.clip(norm, 0, 1)
        out[col] = norm
        params.append({"module": col, "q02": lo, "q98": hi})

    return out, pd.DataFrame(params)


def style_small_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.35)
        ax.spines[side].set_color("#9E9E9E")
    ax.set_facecolor("white")


def plot_module_panel(ax, all_df, sub_df, score_col, cmap, xlim, ylim, point_size=3.2):
    ax.scatter(
        all_df["umap_1"], all_df["umap_2"],
        s=point_size, c="#DFE3E8", alpha=0.55, edgecolors="none", rasterized=True
    )
    ax.scatter(
        sub_df["umap_1"], sub_df["umap_2"],
        s=point_size, c=sub_df[score_col], cmap=cmap, vmin=0, vmax=1,
        alpha=0.95, edgecolors="none", rasterized=True
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    style_small_ax(ax)


def plot_overall_grid(plot_df, groups, out_dir: Path):
    n_groups = len(groups)
    n_modules = len(MODULE_ORDER)
    n_block_rows = math.ceil(n_modules / 2)
    n_cols = n_groups * 2

    fig_w = 2.05 * n_cols + 1.2
    fig_h = 1.9 * n_block_rows + 1.0
    fig, axes = plt.subplots(n_block_rows, n_cols, figsize=(fig_w, fig_h))
    if n_block_rows == 1:
        axes = np.array([axes])

    xmin, xmax = plot_df["umap_1"].min(), plot_df["umap_1"].max()
    ymin, ymax = plot_df["umap_2"].min(), plot_df["umap_2"].max()
    xpad = (xmax - xmin) * 0.04
    ypad = (ymax - ymin) * 0.04
    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)

    cmap = plt.cm.YlOrRd
    block_centers = []

    for r in range(n_block_rows):
        mod_idx1 = 2 * r
        mod_idx2 = 2 * r + 1
        mod_pair = [MODULE_ORDER[mod_idx1]]
        if mod_idx2 < n_modules:
            mod_pair.append(MODULE_ORDER[mod_idx2])

        for pair_idx, module in enumerate(mod_pair):
            score_col = f"score__{module}"
            start_c = pair_idx * n_groups
            center_ax = None

            for gi, g in enumerate(groups):
                ax = axes[r, start_c + gi]
                sub = plot_df[plot_df["group"] == g]
                plot_module_panel(ax, plot_df, sub, score_col, cmap, xlim, ylim)
                ax.set_title(GROUP_DISPLAY[g], fontsize=6.8, pad=1.5)
                if gi == n_groups // 2:
                    center_ax = ax

            block_centers.append((module, center_ax))

        # hide unused right block if odd number of modules
        if len(mod_pair) == 1:
            for gi in range(n_groups, n_cols):
                axes[r, gi].axis("off")

    fig.subplots_adjust(left=0.03, right=0.93, top=0.965, bottom=0.035, wspace=0.02, hspace=0.14)

    # module titles above each block
    for module, ax in block_centers:
        if ax is None:
            continue
        bbox = ax.get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2.0,
            bbox.y1 + 0.006,
            MODULE_SHORT[module],
            ha="center", va="bottom",
            fontsize=8.2, fontweight="bold"
        )

    # overall title
    fig.suptitle("Global manifold mapping of 12 metabolic modules", fontsize=12.5, fontweight="bold", y=0.995)

    # shared colorbar
    cax = fig.add_axes([0.945, 0.17, 0.015, 0.62])
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Normalized module score", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7)

    for ext in ["png", "pdf", "svg"]:
        plt.savefig(
            out_dir / f"Fig4_global_module_mapping_grid.{ext}",
            dpi=600 if ext == "png" else None,
            bbox_inches="tight"
        )
    plt.close(fig)


def plot_single_module_figures(plot_df, groups, out_dir: Path):
    xmin, xmax = plot_df["umap_1"].min(), plot_df["umap_1"].max()
    ymin, ymax = plot_df["umap_2"].min(), plot_df["umap_2"].max()
    xpad = (xmax - xmin) * 0.05
    ypad = (ymax - ymin) * 0.05
    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)
    cmap = plt.cm.YlOrRd

    for module in MODULE_ORDER:
        score_col = f"score__{module}"
        n_groups = len(groups)
        fig, axes = plt.subplots(1, n_groups, figsize=(3.1 * n_groups + 0.8, 3.15))
        if n_groups == 1:
            axes = [axes]

        for i, g in enumerate(groups):
            ax = axes[i]
            sub = plot_df[plot_df["group"] == g]
            plot_module_panel(ax, plot_df, sub, score_col, cmap, xlim, ylim, point_size=4.0)
            ax.set_title(GROUP_DISPLAY[g], fontsize=9, pad=3)

        fig.suptitle(MODULE_SHORT[module], fontsize=12, fontweight="bold", y=0.98)
        fig.subplots_adjust(left=0.03, right=0.92, top=0.83, bottom=0.04, wspace=0.03)

        cax = fig.add_axes([0.935, 0.20, 0.018, 0.50])
        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Normalized\nscore", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", MODULE_SHORT[module]).strip("_")
        for ext in ["png", "pdf", "svg"]:
            plt.savefig(
                out_dir / f"Fig4_module_{safe_name}.{ext}",
                dpi=600 if ext == "png" else None,
                bbox_inches="tight"
            )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Global manifold mapping of metabolic modules from nested 2D Raman .mat files")
    parser.add_argument("--data-dir", type=str, default="data_2d")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--target-layer", type=int, default=1)
    parser.add_argument("--include-normal", action="store_true")
    parser.add_argument("--target-superpoints-per-sample", type=int, default=900)
    parser.add_argument("--max-total-points", type=int, default=24000)
    parser.add_argument("--min-markers-per-pixel", type=int, default=10)
    parser.add_argument("--min-nonzero-features", type=int, default=3)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    script_name = Path(__file__).stem
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / script_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    groups = ["High", "Middle", "Low"]
    if args.include_normal:
        groups = ["Normal", "High", "Middle", "Low"]

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
    pd.DataFrame({"marker": all_markers}).to_csv(out_dir / "used_markers.csv", index=False, encoding="utf-8-sig")

    if len(per_sample) < 2:
        raise SystemExit("Too few valid samples after assembling matrices.")

    X_df, meta = prepare_embedding_matrix(
        per_sample,
        all_markers,
        target_superpoints_per_sample=args.target_superpoints_per_sample,
        max_total_points=args.max_total_points,
        random_state=args.random_state
    )

    emb, X_scaled = run_umap_embedding(X_df, meta, n_pcs=args.n_pcs, random_state=args.random_state)
    score_df, mapping_df = compute_module_scores(X_scaled)
    score_norm_df, score_norm_params = robust_minmax_per_column(score_df, lower_q=0.02, upper_q=0.98)

    plot_df = emb.copy()
    for module in MODULE_ORDER:
        plot_df[f"score__{module}"] = score_norm_df[module].values

    plot_df.to_csv(out_dir / "module_mapping_umap_coordinates.csv", index=False, encoding="utf-8-sig")
    score_df.to_csv(out_dir / "module_score_matrix_raw.csv", index=False, encoding="utf-8-sig")
    score_norm_df.to_csv(out_dir / "module_score_matrix_normalized.csv", index=False, encoding="utf-8-sig")
    mapping_df.to_csv(out_dir / "marker_to_module_mapping.csv", index=False, encoding="utf-8-sig")
    score_norm_params.to_csv(out_dir / "module_score_normalization_params.csv", index=False, encoding="utf-8-sig")

    plot_overall_grid(plot_df, groups, out_dir)
    plot_single_module_figures(plot_df, groups, out_dir)

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
        "Global manifold mapping of 12 metabolic modules.",
        f"Script: {script_name}.py",
        "",
        "Main figure:",
        "  Fig4_global_module_mapping_grid.png / pdf / svg",
        "",
        "Single module figures:",
        "  Fig4_module_*.png / pdf / svg",
        "",
        "Notes:",
        "  1) Base manifold is a shared UMAP embedding.",
        "  2) Module score is calculated by averaging scaled markers within each metabolic module.",
        "  3) Each module score is robustly normalized to 0-1 using 2%-98% quantiles.",
        "  4) Default groups are High / Middle / Low. Add --include-normal if needed."
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
