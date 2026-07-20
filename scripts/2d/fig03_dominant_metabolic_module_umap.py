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
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import umap

warnings.filterwarnings("ignore")

# ============================================================
# Fig03: Dominant metabolic module UMAP projection
#
# Overall idea:
#   1. Build marker matrix from nested 2D .mat files
#   2. Do log1p + scaling + lightweight patient correction
#   3. Compute global UMAP
#   4. Aggregate marker scores into metabolic modules
#   5. For each point, assign its dominant metabolic module
#   6. Draw:
#        - one overall multi-panel figure
#        - one single figure for each group
#
# Default groups:
#   Normal / High differentiated / Moderately differentiated / Poorly differentiated
#
# Run:
#   cd FV-RMH
#   python scripts\2d\fig03_dominant_metabolic_module_umap.py --data-dir data_2d
#
# Optional:
#   Only High / Middle / Low:
#   python scripts\2d\fig03_dominant_metabolic_module_umap.py --data-dir data_2d --exclude-normal
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0


# ------------------------------
# group settings
# ------------------------------
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

LIGHT_GRAY = "#DADDE2"


# ------------------------------
# metabolic modules
# ------------------------------
MODULE_ORDER = [
    "ECM",
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

MODULE_COLORS = {
    "ECM": "#F06A50",
    "Amino Acids": "#54B6D7",
    "Interleukins": "#2FB39A",
    "Calcification": "#38588F",
    "Infection & Immunity": "#F2A27D",
    "Hormones": "#8895C4",
    "Carotenoids": "#9BE3D4",
    "Other Metabolites": "#FF3B30",
    "Glycolipid Metabolism": "#A58C72",
    "Vitamins": "#C4B8A2",
    "Repair & Cytokines": "#F4B248",
    "Oxidative Stress": "#9B59B6"
}


# ------------------------------
# marker -> module rule map
# User can modify this section later
# ------------------------------
MODULE_RULES = {
    "ECM": [
        "acta2", "collagen", "laminin", "elastin", "fibronectin", "syndecan", "versican", "decorin", "biglycan"
    ],
    "Amino Acids": [
        "aspartic", "proline", "methionine", "glutamine", "asparagine", "alanine", "glycine", "serine", "tryptophan", "tyrosine", "valine", "leucine"
    ],
    "Interleukins": [
        "interleukin", "il_", "il-", "cxcl", "ccl", "tnf", "tgf", "ifn"
    ],
    "Calcification": [
        "calcium", "hydroxyapatite", "phosphate_baselinesub", "phosphat"
    ],
    "Infection & Immunity": [
        "cd68", "cd31", "cd98", "b7-h3", "b7_h3", "pd-l1", "pdl1", "cd4", "cd8", "lysozyme", "macrophage"
    ],
    "Hormones": [
        "estrogen", "progesterone", "androgen", "cortisol", "thyroid", "hormone"
    ],
    "Carotenoids": [
        "carotene", "beta-carotene", "betacarotene"
    ],
    "Other Metabolites": [
        "formaldehyde", "fumarate", "nicotinamide", "pyruvate", "lactate", "succinate", "citrate", "malate"
    ],
    "Glycolipid Metabolism": [
        "acetyl", "fructose", "glucose", "phosphoglycerate", "lipid", "glycolipid", "cholesterol", "fatty", "triglyceride", "sphing"
    ],
    "Vitamins": [
        "vitamin", "pyridoxine"
    ],
    "Repair & Cytokines": [
        "repair", "cytokine", "wound", "tgfb", "tgf-b", "healing"
    ],
    "Oxidative Stress": [
        "superoxidedismutase", "sod", "ros", "oxidative", "glutathione"
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

    # Prefer more specific biologically interpretable categories
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
                             max_total_points=28000,
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
    return emb, X_scaled


def compute_module_scores(X_scaled_df: pd.DataFrame, marker_names):
    marker_to_module = {m: assign_module_to_marker(m) for m in marker_names}
    mapping_df = pd.DataFrame({
        "marker": list(marker_to_module.keys()),
        "module": list(marker_to_module.values())
    })

    score_df = pd.DataFrame(index=X_scaled_df.index)

    for module in MODULE_ORDER:
        markers = [m for m, mod in marker_to_module.items() if mod == module and m in X_scaled_df.columns]
        if len(markers) == 0:
            score_df[module] = 0.0
        elif len(markers) == 1:
            score_df[module] = X_scaled_df[markers[0]].values
        else:
            score_df[module] = X_scaled_df[markers].mean(axis=1).values

    dominant_module = score_df.idxmax(axis=1)
    dominant_score = score_df.max(axis=1)

    return score_df, dominant_module, dominant_score, mapping_df


def summarize_module_by_group(plot_df: pd.DataFrame):
    tab = plot_df.groupby(["group", "dominant_module"]).size().reset_index(name="n_points")
    tab["fraction_in_group"] = tab.groupby("group")["n_points"].transform(lambda x: x / x.sum())
    return tab.sort_values(["group", "fraction_in_group"], ascending=[True, False])


def balanced_plot_sample(emb: pd.DataFrame, by: str,
                         max_points=18000,
                         max_points_per_class=4500,
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
    ax.set_facecolor("white")
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(labelsize=9, length=3)


def plot_single_group_panel(sub_df, bg_df, group_name, group_title, out_path_base, xlim, ylim):
    fig, ax = plt.subplots(figsize=(6.0, 6.6))

    ax.scatter(
        bg_df["umap_1"], bg_df["umap_2"],
        s=7, c=LIGHT_GRAY, alpha=0.45, edgecolors="none", rasterized=True
    )

    for module in MODULE_ORDER:
        tmp = sub_df[sub_df["dominant_module"] == module]
        if tmp.empty:
            continue
        ax.scatter(
            tmp["umap_1"], tmp["umap_2"],
            s=7, c=MODULE_COLORS[module], alpha=0.86, edgecolors="none", rasterized=True
        )

    ax.set_title(group_title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    style_axis(ax)

    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(
            out_path_base.with_suffix(f".{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight"
        )
    plt.close(fig)


def plot_overall_multigroup(plot_df: pd.DataFrame, groups, out_dir: Path):
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(5.2 * n_groups, 7.2), sharex=True, sharey=True)

    if n_groups == 1:
        axes = [axes]

    xmin, xmax = plot_df["umap_1"].min(), plot_df["umap_1"].max()
    ymin, ymax = plot_df["umap_2"].min(), plot_df["umap_2"].max()
    xpad = (xmax - xmin) * 0.05
    ypad = (ymax - ymin) * 0.05
    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)

    for i, g in enumerate(groups):
        ax = axes[i]
        title = GROUP_DISPLAY[g]

        ax.scatter(
            plot_df["umap_1"], plot_df["umap_2"],
            s=7, c=LIGHT_GRAY, alpha=0.40, edgecolors="none", rasterized=True
        )

        sub = plot_df[plot_df["group"] == g]
        for module in MODULE_ORDER:
            tmp = sub[sub["dominant_module"] == module]
            if tmp.empty:
                continue
            ax.scatter(
                tmp["umap_1"], tmp["umap_2"],
                s=7, c=MODULE_COLORS[module], alpha=0.86, edgecolors="none", rasterized=True
            )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        style_axis(ax)

        if i == 0:
            ax.set_ylabel("UMAP 2", fontsize=11)
        if i == n_groups // 2:
            ax.set_xlabel("UMAP 1", fontsize=11)

        # Save each panel individually as well
        plot_single_group_panel(
            sub_df=sub,
            bg_df=plot_df,
            group_name=g,
            group_title=title,
            out_path_base=out_dir / f"Fig3_{g}_dominant_module_UMAP",
            xlim=xlim,
            ylim=ylim
        )

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MODULE_COLORS[m], markeredgecolor="none",
               markersize=8, label=m)
        for m in MODULE_ORDER
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=10,
        title="Dominant Metabolic Module",
        title_fontsize=11
    )

    plt.subplots_adjust(wspace=0.02, bottom=0.18)

    for ext in ["png", "pdf", "svg"]:
        plt.savefig(
            out_dir / f"Fig3_dominant_metabolic_module_UMAP_all_groups.{ext}",
            dpi=600 if ext == "png" else None,
            bbox_inches="tight"
        )
    plt.close(fig)

    # Save legend only
    fig_leg, ax_leg = plt.subplots(figsize=(9.5, 1.9))
    ax_leg.axis("off")
    ax_leg.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=10,
        title="Dominant Metabolic Module",
        title_fontsize=11
    )
    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(
            out_dir / f"Fig3_metabolic_module_legend.{ext}",
            dpi=600 if ext == "png" else None,
            bbox_inches="tight"
        )
    plt.close(fig_leg)


def main():
    parser = argparse.ArgumentParser(description="Dominant metabolic module UMAP projection from nested 2D Raman .mat files")

    parser.add_argument("--data-dir", type=str, default="data_2d")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--target-layer", type=int, default=1)

    parser.add_argument("--exclude-normal", action="store_true")
    parser.add_argument("--no-batch-correct", action="store_true")

    parser.add_argument("--target-superpoints-per-sample", type=int, default=900)
    parser.add_argument("--max-total-points", type=int, default=28000)
    parser.add_argument("--min-markers-per-pixel", type=int, default=10)
    parser.add_argument("--min-nonzero-features", type=int, default=3)

    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--max-points-for-plot", type=int, default=18000)
    parser.add_argument("--max-points-per-group", type=int, default=4500)

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

    emb, X_scaled = run_umap(
        X_df,
        meta,
        out_dir,
        n_pcs=args.n_pcs,
        random_state=args.random_state,
        batch_correct=not args.no_batch_correct
    )

    score_df, dominant_module, dominant_score, mapping_df = compute_module_scores(X_scaled, X_scaled.columns.tolist())

    emb["dominant_module"] = dominant_module.values
    emb["dominant_module_score"] = dominant_score.values

    emb.to_csv(out_dir / "dominant_module_umap_coordinates.csv", index=False, encoding="utf-8-sig")
    score_df.to_csv(out_dir / "module_score_matrix.csv", index=False, encoding="utf-8-sig")
    mapping_df.to_csv(out_dir / "marker_to_module_mapping.csv", index=False, encoding="utf-8-sig")

    plot_df = emb.copy()
    plot_df = balanced_plot_sample(
        plot_df,
        by="group",
        max_points=args.max_points_for_plot,
        max_points_per_class=args.max_points_per_group,
        random_state=args.random_state
    )

    module_summary = summarize_module_by_group(plot_df)
    module_summary.to_csv(out_dir / "dominant_module_summary_by_group.csv", index=False, encoding="utf-8-sig")

    plot_overall_multigroup(
        plot_df,
        groups=groups,
        out_dir=out_dir
    )

    params = vars(args)
    params["script_name"] = script_name
    params["groups_present"] = groups
    params["n_mat_files_found"] = len(mat_files)
    params["n_recognized_files"] = len(recognized_df)
    params["n_valid_samples"] = len(per_sample)
    params["n_embedding_input_points"] = len(X_df)
    params["n_plot_points"] = len(plot_df)

    with open(out_dir / "run_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    readme = [
        "Dominant metabolic module UMAP output folder.",
        f"Script: {script_name}.py",
        "",
        "Main figure:",
        "  Fig3_dominant_metabolic_module_UMAP_all_groups.png / pdf / svg",
        "",
        "Single group panels:",
        "  Fig3_Normal_dominant_module_UMAP.*",
        "  Fig3_High_dominant_module_UMAP.*",
        "  Fig3_Middle_dominant_module_UMAP.*",
        "  Fig3_Low_dominant_module_UMAP.*",
        "",
        "Legend:",
        "  Fig3_metabolic_module_legend.*",
        "",
        "Important note:",
        "  Marker-to-module assignment is rule-based and editable.",
        "  If you want stricter biological classification, edit MODULE_RULES in the script."
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
