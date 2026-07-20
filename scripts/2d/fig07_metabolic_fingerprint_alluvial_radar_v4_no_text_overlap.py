from pathlib import Path
import re
import json
import shutil
import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore")

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

GROUP_DISPLAY_AXIS = {
    "Normal": "Normal",
    "High": "High\ndifferentiated",
    "Middle": "Moderately\ndifferentiated",
    "Low": "Poorly\ndifferentiated"
}

MODULE_LABELS_WRAPPED = {
    "Extracellular Matrix": "Extracellular\nMatrix",
    "Amino Acids": "Amino Acids",
    "Interleukins": "Interleukins",
    "Calcification": "Calcification",
    "Infection & Immunity": "Infection &\nImmunity",
    "Hormones": "Hormones",
    "Carotenoids": "Carotenoids",
    "Other Metabolites": "Other\nMetabolites",
    "Glycolipid Metabolism": "Glycolipid\nMetabolism",
    "Vitamins": "Vitamins",
    "Repair & Cytokines": "Repair &\nCytokines",
    "Oxidative Stress": "Oxidative\nStress"
}

GROUP_COLORS = {
    "Normal": "#66C2A5",
    "High": "#FC8D62",
    "Middle": "#8DA0CB",
    "Low": "#E78AC3"
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

MODULE_COLORS = {
    "Extracellular Matrix": "#E8A69A",
    "Amino Acids": "#99D7E2",
    "Interleukins": "#7CC6B7",
    "Calcification": "#9AA8C7",
    "Infection & Immunity": "#E6C7BA",
    "Hormones": "#C7CEDF",
    "Carotenoids": "#B8DAD5",
    "Other Metabolites": "#E8898B",
    "Glycolipid Metabolism": "#C9C0B3",
    "Vitamins": "#D8D0C5",
    "Repair & Cytokines": "#EED097",
    "Oxidative Stress": "#BFA3CB"
}

MODULE_RULES = {
    "Extracellular Matrix": ["acta2", "collagen", "laminin", "elastin", "fibronectin", "syndecan", "versican", "decorin", "biglycan"],
    "Amino Acids": ["aspartic", "proline", "methionine", "glutamine", "asparagine", "alanine", "glycine", "serine", "tyrosine", "tryptophan", "valine", "leucine", "isoleucine", "lysine", "histidine", "phenylalanine", "threonine"],
    "Interleukins": ["interleukin", "il_", "il-", "cxcl", "ccl", "tnf", "ifn"],
    "Calcification": ["calcium", "hydroxyapatite", "phosphate", "mineral"],
    "Infection & Immunity": ["cd68", "cd31", "cd98", "b7h3", "b7-h3", "b7_h3", "pdl1", "pdl1", "cd4", "cd8", "immune", "lysozyme"],
    "Hormones": ["estrogen", "progesterone", "androgen", "cortisol", "thyroid", "hormone"],
    "Carotenoids": ["carotene", "betacarotene", "beta-carotene"],
    "Other Metabolites": ["formaldehyde", "fumarate", "succinate", "citrate", "malate", "pyruvate", "lactate", "nicotinamide"],
    "Glycolipid Metabolism": ["acetyl", "fructose", "glucose", "phosphoglycerate", "lipid", "glycolipid", "cholesterol", "fatty", "triglyceride", "sphing", "glycerol"],
    "Vitamins": ["vitamin", "pyridoxine"],
    "Repair & Cytokines": ["repair", "cytokine", "wound", "healing", "tgfb", "tgf-b", "tgf"],
    "Oxidative Stress": ["superoxidedismutase", "sod", "oxidative", "ros", "glutathione"]
}


def normalize_marker_name(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def assign_module_to_marker(marker_name):
    m = normalize_marker_name(marker_name)
    matched = []
    for module, keywords in MODULE_RULES.items():
        for kw in keywords:
            if normalize_marker_name(kw) in m:
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
        per_sample.append({"sample_id": f"{patient}_{group}_layer{layer}", "patient": patient, "group": group, "layer": layer, "matrix": X})
    return per_sample, all_markers


def make_superpoints(X, target_n=900, random_state=42):
    X = fill_nonfinite_matrix(X)
    if X.shape[0] <= target_n:
        return X
    km = MiniBatchKMeans(n_clusters=min(target_n, X.shape[0]), random_state=random_state, batch_size=2048, n_init=5)
    labels = km.fit_predict(X)
    centers = np.zeros((km.n_clusters, X.shape[1]), dtype=float)
    for i in range(km.n_clusters):
        mask = labels == i
        if np.any(mask):
            centers[i] = X[mask].mean(axis=0)
    return fill_nonfinite_matrix(centers)


def prepare_embedding_matrix(per_sample, all_markers, target_superpoints_per_sample=900, max_total_points=26000, random_state=42):
    rows = []
    meta_rows = []
    for item in per_sample:
        X = fill_nonfinite_matrix(item["matrix"])
        X = np.log1p(np.maximum(X, 0))
        X = fill_nonfinite_matrix(X)
        X = make_superpoints(X, target_n=target_superpoints_per_sample, random_state=random_state)
        for i in range(X.shape[0]):
            rows.append(X[i])
            meta_rows.append({"sample_id": item["sample_id"], "patient": item["patient"], "group": item["group"], "layer": item["layer"]})
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


def compute_module_scores(X_df: pd.DataFrame):
    marker_to_module = {m: assign_module_to_marker(m) for m in X_df.columns}
    mapping_df = pd.DataFrame({"marker": list(marker_to_module.keys()), "module": list(marker_to_module.values())})
    score_df = pd.DataFrame(index=X_df.index)
    for module in MODULE_ORDER:
        cols = [m for m, mod in marker_to_module.items() if mod == module and m in X_df.columns]
        if len(cols) == 0:
            score_df[module] = 0.0
        elif len(cols) == 1:
            score_df[module] = X_df[cols[0]].values
        else:
            score_df[module] = X_df[cols].mean(axis=1).values
    dominant_module = score_df.idxmax(axis=1)
    dominant_score = score_df.max(axis=1)
    return score_df, dominant_module, dominant_score, mapping_df


def summarize_dominant_module(meta: pd.DataFrame, dominant_module: pd.Series, groups):
    tmp = meta.copy()
    tmp["dominant_module"] = dominant_module.values
    tab = tmp.groupby(["group", "dominant_module"]).size().reset_index(name="n_points")
    all_rows = []
    for g in groups:
        for m in MODULE_ORDER:
            hit = tab[(tab["group"] == g) & (tab["dominant_module"] == m)]
            n = int(hit["n_points"].iloc[0]) if len(hit) else 0
            all_rows.append({"group": g, "dominant_module": m, "n_points": n})
    out = pd.DataFrame(all_rows)
    out["fraction"] = out.groupby("group")["n_points"].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    out["sqrt_fraction"] = np.sqrt(out["fraction"])
    out["sqrt_fraction_norm"] = out.groupby("group")["sqrt_fraction"].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    return out


def summarize_module_fingerprint(meta: pd.DataFrame, score_df: pd.DataFrame, groups):
    tmp = score_df.copy()
    tmp["group"] = meta["group"].values
    mean_scores = tmp.groupby("group")[MODULE_ORDER].mean().reindex(groups)
    norm_scores = mean_scores.copy()
    for col in MODULE_ORDER:
        vals = mean_scores[col].values.astype(float)
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            norm_scores[col] = (vals - vmin) / (vmax - vmin)
        else:
            norm_scores[col] = 0.5
    return mean_scores, norm_scores


def build_group_stack_table(summary_df: pd.DataFrame, groups, value_col="sqrt_fraction_norm"):
    table = pd.DataFrame(index=MODULE_ORDER, columns=groups, dtype=float)
    for g in groups:
        sub = summary_df[summary_df["group"] == g]
        for m in MODULE_ORDER:
            hit = sub[sub["dominant_module"] == m]
            table.loc[m, g] = float(hit[value_col].iloc[0]) if len(hit) else 0.0
    return table.fillna(0.0)


def add_flow_patch(ax, x0, x1, y0_lo, y0_hi, y1_lo, y1_hi, color, alpha=0.58):
    cx = (x1 - x0) * 0.35
    verts = [
        (x0, y0_hi), (x0 + cx, y0_hi), (x1 - cx, y1_hi), (x1, y1_hi),
        (x1, y1_lo), (x1 - cx, y1_lo), (x0 + cx, y0_lo), (x0, y0_lo), (x0, y0_hi)
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def draw_alluvial(ax, stack_table: pd.DataFrame, groups):
    x_positions = np.arange(len(groups))
    bar_width = 0.34
    bottoms = pd.DataFrame(index=MODULE_ORDER, columns=groups, dtype=float)
    tops = pd.DataFrame(index=MODULE_ORDER, columns=groups, dtype=float)

    for g in groups:
        y = 0.0
        for m in MODULE_ORDER[::-1]:
            h = float(stack_table.loc[m, g])
            bottoms.loc[m, g] = y
            tops.loc[m, g] = y + h
            y += h

    for i in range(len(groups) - 1):
        g0 = groups[i]
        g1 = groups[i + 1]
        x0 = x_positions[i] + bar_width / 2
        x1 = x_positions[i + 1] - bar_width / 2
        for m in MODULE_ORDER:
            add_flow_patch(
                ax, x0, x1,
                bottoms.loc[m, g0], tops.loc[m, g0],
                bottoms.loc[m, g1], tops.loc[m, g1],
                MODULE_COLORS[m], alpha=0.55
            )

    for i, g in enumerate(groups):
        for m in MODULE_ORDER[::-1]:
            y0 = bottoms.loc[m, g]
            y1 = tops.loc[m, g]
            ax.add_patch(
                Rectangle(
                    (x_positions[i] - bar_width / 2, y0),
                    bar_width, y1 - y0,
                    facecolor=MODULE_COLORS[m],
                    edgecolor="black",
                    linewidth=0.9,
                    alpha=0.95
                )
            )

    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([GROUP_DISPLAY_AXIS[g] for g in groups], fontsize=9.5, linespacing=1.05)
    ticks = [0, 0.25, 0.50, 0.75, 1.00]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.2f}" for v in ticks], fontsize=10)
    ax.set_xlabel("Disease Stage", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Relative Density (Sqrt-scaled)", fontsize=12, fontweight="bold", labelpad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", width=1.0, length=4, pad=8)
    ax.tick_params(axis="y", width=1.0, length=4)


def draw_radar(ax, radar_df: pd.DataFrame, groups):
    n = len(MODULE_ORDER)
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed_angles = np.concatenate([base_angles, [2 * np.pi]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.grid(True, linestyle=(0, (4, 4)), linewidth=0.8, alpha=0.65)
    ax.spines["polar"].set_alpha(0.25)

    # Hide default labels and place custom non-overlapping labels outside the circle
    ax.set_xticks(base_angles)
    ax.set_xticklabels([])

    smooth_angles = np.linspace(0, 2 * np.pi, 360)

    for g in groups:
        raw_vals = radar_df.loc[g, MODULE_ORDER].astype(float).tolist()
        closed_vals = np.array(raw_vals + [raw_vals[0]], dtype=float)

        try:
            spline = CubicSpline(closed_angles, closed_vals, bc_type="periodic")
            smooth_vals = spline(smooth_angles)
            smooth_vals = np.clip(smooth_vals, 0, 1.0)
        except Exception:
            smooth_vals = closed_vals
            smooth_angles_local = closed_angles
        else:
            smooth_angles_local = smooth_angles

        ax.plot(
            smooth_angles_local, smooth_vals,
            linewidth=2.4,
            color=GROUP_COLORS[g],
            alpha=0.95
        )
        ax.fill(
            smooth_angles_local, smooth_vals,
            color=GROUP_COLORS[g],
            alpha=0.10
        )

        ax.scatter(
            base_angles, raw_vals,
            s=34,
            c="white",
            edgecolors=GROUP_COLORS[g],
            linewidths=1.6,
            zorder=5
        )

    for ang, module in zip(base_angles, MODULE_ORDER):
        label = MODULE_LABELS_WRAPPED[module]
        deg = np.degrees(ang)

        if 15 < deg < 165:
            ha = "left"
        elif 195 < deg < 345:
            ha = "right"
        else:
            ha = "center"

        ax.text(
            ang, 1.13, label,
            ha=ha, va="center",
            fontsize=8.8,
            fontweight="bold",
            linespacing=0.95,
            clip_on=False
        )

    handles = [
        Line2D(
            [0], [0],
            color=GROUP_COLORS[g],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=GROUP_COLORS[g],
            markeredgewidth=1.5,
            linewidth=2.4,
            label=GROUP_DISPLAY[g]
        )
        for g in groups
    ]
    ax.legend(
        handles=handles,
        title="Group",
        frameon=False,
        bbox_to_anchor=(1.34, 0.35),
        loc="center left",
        fontsize=10,
        title_fontsize=10
    )


def plot_combined_figure(stack_table, radar_df, groups, out_dir: Path):
    fig = plt.figure(figsize=(15.8, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.06], wspace=0.42)
    ax1 = fig.add_subplot(gs[0, 0])
    draw_alluvial(ax1, stack_table, groups)
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    draw_radar(ax2, radar_df, groups)
    ax2.set_title("Metabolic Fingerprint Profiles by Stage", fontsize=16, fontweight="bold", pad=18)
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=MODULE_COLORS[m], markeredgecolor="black", markeredgewidth=0.8, markersize=10, label=m) for m in MODULE_ORDER]
    ax1.legend(handles=handles, title="Dominant Pathway", frameon=False, bbox_to_anchor=(1.01, 0.50), loc="center left", fontsize=9.5, title_fontsize=10)
    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(out_dir / f"Fig7_metabolic_fingerprint_alluvial_radar.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_alluvial_only(stack_table, groups, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    draw_alluvial(ax, stack_table, groups)
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=MODULE_COLORS[m], markeredgecolor="black", markeredgewidth=0.8, markersize=10, label=m) for m in MODULE_ORDER]
    ax.legend(handles=handles, title="Dominant Pathway", frameon=False, bbox_to_anchor=(1.02, 0.50), loc="center left", fontsize=9.5, title_fontsize=10)
    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(out_dir / f"Fig7_alluvial_only.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_radar_only(radar_df, groups, out_dir: Path):
    fig = plt.figure(figsize=(9.0, 8.0))
    ax = fig.add_subplot(111, polar=True)
    draw_radar(ax, radar_df, groups)
    ax.set_title("Metabolic Fingerprint Profiles by Stage", fontsize=16, fontweight="bold", pad=18)
    plt.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(out_dir / f"Fig7_radar_only.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Metabolic fingerprint radar + alluvial figure from nested 2D Raman .mat files")
    parser.add_argument("--data-dir", type=str, default="data_2d")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--target-layer", type=int, default=1)
    parser.add_argument("--exclude-normal", action="store_true")
    parser.add_argument("--no-batch-correct", action="store_true")
    parser.add_argument("--target-superpoints-per-sample", type=int, default=900)
    parser.add_argument("--max-total-points", type=int, default=26000)
    parser.add_argument("--min-markers-per-pixel", type=int, default=10)
    parser.add_argument("--min-nonzero-features", type=int, default=3)
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
        raise SystemExit("Files were parsed but none were valid after loading.")

    sample_summary = recognized_df.groupby(["group", "patient"])["marker"].nunique().reset_index(name="n_markers")
    sample_summary.to_csv(out_dir / "sample_metadata_summary.csv", index=False, encoding="utf-8-sig")

    per_sample, all_markers = assemble_sample_matrix(sample_dict, min_markers_per_pixel=args.min_markers_per_pixel, min_nonzero_features=args.min_nonzero_features)
    pd.DataFrame({"marker": all_markers}).to_csv(out_dir / "used_markers.csv", index=False, encoding="utf-8-sig")
    if len(per_sample) < 2:
        raise SystemExit("Too few valid samples after assembling matrices.")

    X_df, meta = prepare_embedding_matrix(per_sample, all_markers, target_superpoints_per_sample=args.target_superpoints_per_sample, max_total_points=args.max_total_points, random_state=args.random_state)

    X_work = StandardScaler().fit_transform(X_df.values)
    X_work = pd.DataFrame(np.nan_to_num(X_work, nan=0.0, posinf=0.0, neginf=0.0), columns=X_df.columns)
    if not args.no_batch_correct:
        X_work = patient_center_correction(X_work, meta)

    score_df, dominant_module, dominant_score, mapping_df = compute_module_scores(X_work)
    summary_df = summarize_dominant_module(meta, dominant_module, groups)
    mean_fingerprint, radar_df = summarize_module_fingerprint(meta, score_df, groups)
    stack_table = build_group_stack_table(summary_df, groups, value_col="sqrt_fraction_norm")

    mapping_df.to_csv(out_dir / "marker_to_module_mapping.csv", index=False, encoding="utf-8-sig")
    score_df.to_csv(out_dir / "module_score_matrix.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "dominant_module_composition_by_group.csv", index=False, encoding="utf-8-sig")
    mean_fingerprint.to_csv(out_dir / "mean_module_fingerprint_by_group.csv", encoding="utf-8-sig")
    radar_df.to_csv(out_dir / "normalized_module_fingerprint_for_radar.csv", encoding="utf-8-sig")
    stack_table.to_csv(out_dir / "sqrt_scaled_alluvial_table.csv", encoding="utf-8-sig")

    plot_combined_figure(stack_table, radar_df, groups, out_dir)
    plot_alluvial_only(stack_table, groups, out_dir)
    plot_radar_only(radar_df, groups, out_dir)

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
        "Metabolic fingerprint radar + alluvial output folder.",
        f"Script: {script_name}.py",
        "",
        "Main figure: Fig7_metabolic_fingerprint_alluvial_radar.*",
        "Single figures: Fig7_alluvial_only.* , Fig7_radar_only.*",
        "",
        "Marker-to-module assignment is rule-based and can be edited in MODULE_RULES.",
        "Alluvial composition uses square-root-scaled relative density, then renormalizes within each stage."
    ]
    (out_dir / "00_README.txt").write_text("\n".join(readme), encoding="utf-8")
    try:
        shutil.copy2(Path(__file__), out_dir / f"{script_name}.py")
    except Exception:
        pass

    print(out_dir)


if __name__ == "__main__":
    main()
