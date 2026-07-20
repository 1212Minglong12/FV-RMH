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
from scipy import stats

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

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
    "Normal": "#8FC4E8",
    "High": "#7BC69D",
    "Middle": "#F2C38E",
    "Low": "#E57D7D",
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

def fdr_bh(pvals):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    pv = pvals[finite]
    if len(pv) == 0:
        return qvals
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    q_back = np.empty_like(q)
    q_back[order] = q
    qvals[finite] = q_back
    return qvals

def build_patient_level_table(meta_df, target_layer=1):
    rows, skipped = [], []
    for _, row in meta_df.iterrows():
        if row["group"] not in GROUP_ORDER or int(row["layer"]) != int(target_layer):
            continue
        try:
            vec = load_mat_numeric_array(Path(row["file"]))
        except Exception as e:
            skipped.append({**row.to_dict(), "reason": str(e)})
            continue
        vec = robust_clip(vec)
        vec = vec[np.isfinite(vec)]
        if len(vec) == 0:
            continue
        rows.append({
            "marker": row["marker"],
            "group": row["group"],
            "patient": row["patient"],
            "layer": row["layer"],
            "file": row["file"],
            "mean_value": float(np.mean(vec)),
            "median_value": float(np.median(vec)),
            "percent_positive": float(np.mean(vec > 0) * 100.0),
            "n_pixels": int(len(vec)),
        })
    return pd.DataFrame(rows), pd.DataFrame(skipped)

def compare_all_markers(patient_df, value_col="mean_value", alpha=0.05, min_n_per_group=2):
    rows, pair_rows = [], []
    for marker in sorted(patient_df["marker"].unique()):
        sub = patient_df[patient_df["marker"] == marker].copy()
        group_arrays = []
        usable_groups = []
        group_stats = {}
        for g in GROUP_ORDER:
            vals = sub.loc[sub["group"] == g, value_col].dropna().values.astype(float)
            group_stats[g] = {
                "n": len(vals),
                "mean": float(np.mean(vals)) if len(vals) else np.nan,
                "median": float(np.median(vals)) if len(vals) else np.nan,
            }
            if len(vals) >= min_n_per_group:
                group_arrays.append(vals)
                usable_groups.append(g)
        if len(usable_groups) < 2:
            continue

        H, p_global = stats.kruskal(*group_arrays)
        n_total = sum(len(a) for a in group_arrays)
        k = len(group_arrays)
        eps2 = max(0.0, (H - k + 1) / (n_total - k)) if n_total > k else np.nan

        local_pairs = []
        for i in range(len(usable_groups)):
            for j in range(i + 1, len(usable_groups)):
                g1, g2 = usable_groups[i], usable_groups[j]
                x = sub.loc[sub["group"] == g1, value_col].dropna().values.astype(float)
                y = sub.loc[sub["group"] == g2, value_col].dropna().values.astype(float)
                _, p_pair = stats.mannwhitneyu(x, y, alternative="two-sided")
                local_pairs.append({
                    "marker": marker,
                    "group1": g1,
                    "group2": g2,
                    "n1": len(x),
                    "n2": len(y),
                    "mean1": float(np.mean(x)),
                    "mean2": float(np.mean(y)),
                    "median1": float(np.median(x)),
                    "median2": float(np.median(y)),
                    "delta_mean": float(np.mean(x) - np.mean(y)),
                    "delta_median": float(np.median(x) - np.median(y)),
                    "p_value": float(p_pair),
                })

        if local_pairs:
            qvals = fdr_bh([r["p_value"] for r in local_pairs])
            for r, q in zip(local_pairs, qvals):
                r["q_value"] = float(q)
                r["significant"] = bool(q < alpha)
                pair_rows.append(r)

        rows.append({
            "marker": marker,
            "kruskal_H": float(H),
            "p_value": float(p_global),
            "effect_size_eps2": float(eps2),
            "n_significant_pairs": int(sum(1 for r in local_pairs if r.get("q_value", np.nan) < alpha)),
            **{f"n_{g}": group_stats[g]["n"] for g in GROUP_ORDER},
            **{f"mean_{g}": group_stats[g]["mean"] for g in GROUP_ORDER},
            **{f"median_{g}": group_stats[g]["median"] for g in GROUP_ORDER},
        })

    global_df = pd.DataFrame(rows)
    global_df["q_value"] = fdr_bh(global_df["p_value"].values.astype(float))
    global_df["significant"] = global_df["q_value"] < alpha
    global_df["ranking_score"] = (
        -np.log10(global_df["q_value"].replace(0, 1e-300).fillna(1.0))
        + global_df["effect_size_eps2"].fillna(0.0) * 3.0
        + global_df["n_significant_pairs"].fillna(0).astype(float) * 0.5
    )
    global_df = global_df.sort_values(["significant", "ranking_score", "effect_size_eps2"], ascending=[False, False, False]).reset_index(drop=True)

    pair_df = pd.DataFrame(pair_rows)
    if len(pair_df) > 0:
        pair_df = pair_df.sort_values(["marker", "q_value", "p_value"]).reset_index(drop=True)
    return global_df, pair_df

def compute_display_limits(arrays):
    valid_arrays = [a for a in arrays if len(a) > 0]
    flat = np.concatenate(valid_arrays) if valid_arrays else np.array([0.0, 1.0])
    lo = np.nanquantile(flat, 0.02)
    hi = np.nanquantile(flat, 0.98)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        lo, hi = np.nanmin(flat), np.nanmax(flat)
    margin = (hi - lo) * 0.18 if hi > lo else 1.0
    return lo - margin * 0.15, hi + margin

def add_n_labels(ax, arrays):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for i, arr in enumerate(arrays, start=1):
        ax.text(i, ymin + span * 0.035, f"n={len(arr)}", ha="center", va="bottom", fontsize=7, color="#444444")

def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.20, linewidth=0.6)
    ax.set_axisbelow(True)

def annotate_pairwise_brackets(ax, arrays, pair_df_marker, max_pairs=2):
    if pair_df_marker is None or len(pair_df_marker) == 0:
        return
    sig = pair_df_marker[pair_df_marker["significant"]].copy()
    if len(sig) == 0:
        return
    sig = sig.sort_values(["q_value", "p_value"]).head(max_pairs)
    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    base = y_max - span * 0.20
    step = span * 0.08
    group_to_pos = {g: i + 1 for i, g in enumerate(GROUP_ORDER)}

    for idx, (_, row) in enumerate(sig.iterrows()):
        x1 = group_to_pos[row["group1"]]
        x2 = group_to_pos[row["group2"]]
        if x1 > x2:
            x1, x2 = x2, x1
        y = base + idx * step
        h = step * 0.35
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c="#444444")
        q = row["q_value"]
        label = f"q={q:.2g}" if q >= 0.001 else f"q={q:.1e}"
        ax.text((x1 + x2) / 2.0, y + h + step * 0.06, label, ha="center", va="bottom", fontsize=7)

    ax.set_ylim(y_min, base + max_pairs * step + span * 0.09)

def draw_pretty_violin(ax, arrays, title, ylabel, pair_df_marker=None, show_points=True):
    positions = np.arange(1, len(GROUP_ORDER) + 1)
    vp = ax.violinplot(arrays, positions=positions, widths=0.82, showmeans=False, showmedians=False, showextrema=False)
    for body, g in zip(vp["bodies"], GROUP_ORDER):
        body.set_facecolor(GROUP_COLORS[g])
        body.set_edgecolor("#444444")
        body.set_linewidth(0.9)
        body.set_alpha(0.48)

    bp = ax.boxplot(
        arrays, positions=positions, widths=0.22, patch_artist=True, showfliers=False,
        medianprops=dict(color="#222222", linewidth=1.2),
        whiskerprops=dict(color="#444444", linewidth=0.9),
        capprops=dict(color="#444444", linewidth=0.9),
        boxprops=dict(edgecolor="#444444", linewidth=0.9)
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.9)

    for pos, arr in zip(positions, arrays):
        if len(arr) == 0:
            continue
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        ax.hlines([q1, q3], pos - 0.16, pos + 0.16, colors="#444444", linewidth=0.8, zorder=4)
        ax.scatter([pos], [med], s=16, zorder=5, c="#111111", edgecolors="white", linewidths=0.35)

    if show_points:
        rng = np.random.default_rng(42)
        for pos, arr, g in zip(positions, arrays, GROUP_ORDER):
            if len(arr) == 0:
                continue
            n_show = min(len(arr), 120)
            arr_show = arr if len(arr) <= n_show else arr[rng.choice(len(arr), size=n_show, replace=False)]
            jitter = rng.uniform(-0.085, 0.085, size=len(arr_show))
            ax.scatter(np.full(len(arr_show), pos) + jitter, arr_show, s=10, alpha=0.35,
                       c=GROUP_COLORS[g], edgecolors="none", zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([GROUP_DISPLAY[g] for g in GROUP_ORDER], rotation=18, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8)
    y0, y1 = compute_display_limits(arrays)
    ax.set_ylim(y0, y1)
    annotate_pairwise_brackets(ax, arrays, pair_df_marker=pair_df_marker, max_pairs=2)
    add_n_labels(ax, arrays)
    style_axis(ax)

def choose_top_markers(global_df, top_n=10, only_significant=True):
    if len(global_df) == 0:
        return []
    if only_significant:
        sig = global_df[global_df["significant"]].copy()
        if len(sig) > 0:
            return sig["marker"].head(top_n).tolist()
    return global_df["marker"].head(top_n).tolist()

def build_pixel_level_subset(meta_df, top_markers, target_layer=1, per_file_max=800, per_group_marker_max=3000):
    rows = []
    rng = np.random.default_rng(42)
    for marker in top_markers:
        marker_meta = meta_df[(meta_df["marker"] == marker) & (meta_df["layer"] == target_layer) & (meta_df["group"].isin(GROUP_ORDER))]
        for g in GROUP_ORDER:
            vals_collect = []
            for _, row in marker_meta[marker_meta["group"] == g].iterrows():
                try:
                    vec = load_mat_numeric_array(Path(row["file"]))
                    vec = robust_clip(vec)
                    vec = vec[np.isfinite(vec)]
                    if len(vec) == 0:
                        continue
                    if len(vec) > per_file_max:
                        vec = vec[rng.choice(len(vec), size=per_file_max, replace=False)]
                    vals_collect.append(vec)
                except Exception:
                    pass
            if len(vals_collect) == 0:
                continue
            vals = np.concatenate(vals_collect)
            if len(vals) > per_group_marker_max:
                vals = vals[rng.choice(len(vals), size=per_group_marker_max, replace=False)]
            for v in vals:
                rows.append({"marker": marker, "group": g, "value": float(v)})
    return pd.DataFrame(rows)

def plot_violin_samplelevel(patient_df, top_markers, global_df, pair_df, out_dir: Path, value_col="mean_value"):
    n = len(top_markers)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.8, max(4.2, nrows * 3.3)), squeeze=False)
    axes = axes.flatten()

    for ax, marker in zip(axes, top_markers):
        sub = patient_df[patient_df["marker"] == marker].copy()
        arrays = [sub.loc[sub["group"] == g, value_col].dropna().values.astype(float) for g in GROUP_ORDER]
        row = global_df.loc[global_df["marker"] == marker].iloc[0]
        q = row["q_value"]
        title = f"{marker} | global q={q:.2g} | eps²={row['effect_size_eps2']:.2f}"
        pair_marker = pair_df[pair_df["marker"] == marker].copy() if len(pair_df) > 0 else None
        draw_pretty_violin(ax, arrays, title, "Patient-level abundance", pair_df_marker=pair_marker, show_points=True)

    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("topSubset_violin_samplelevel (optimized + p/q annotations)", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"topSubset_violin_samplelevel_optimized_pvalues.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_violin_pixellevel(pixel_df, top_markers, out_dir: Path):
    if len(pixel_df) == 0:
        return
    n = len(top_markers)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.8, max(4.2, nrows * 3.3)), squeeze=False)
    axes = axes.flatten()
    for ax, marker in zip(axes, top_markers):
        sub = pixel_df[pixel_df["marker"] == marker].copy()
        arrays = [sub.loc[sub["group"] == g, "value"].dropna().values.astype(float) for g in GROUP_ORDER]
        draw_pretty_violin(ax, arrays, f"{marker}", "Pixel-level abundance", pair_df_marker=None, show_points=True)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("topSubset_violin_pixellevel_exploratory (optimized)", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"topSubset_violin_pixellevel_exploratory_optimized.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_significant_marker_ranking(global_df, top_markers, out_dir: Path):
    df = global_df[global_df["marker"].isin(top_markers)].copy()
    if len(df) == 0:
        return
    df["order"] = df["marker"].apply(lambda x: top_markers.index(x))
    df = df.sort_values("order").iloc[::-1]
    score = -np.log10(df["q_value"].replace(0, 1e-300))
    fig, ax = plt.subplots(figsize=(8.2, max(4.6, 0.5 * len(df) + 1.7)))
    ax.barh(df["marker"], score)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(score.iloc[i] + 0.04, i, f"eps²={r['effect_size_eps2']:.2f}", va="center", fontsize=8)
    ax.set_xlabel("-log10(FDR q-value)")
    ax.set_title("significant_marker_ranking (Top 10)")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"significant_marker_ranking.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_dotplot(patient_df, top_markers, value_col, out_dir: Path):
    sub = patient_df[patient_df["marker"].isin(top_markers)].copy()
    agg = sub.groupby(["marker", "group"], as_index=False).agg(
        avg_value=(value_col, "mean"),
        pct_pos=("percent_positive", "mean")
    )
    z_rows = []
    for marker in top_markers:
        vals = agg.loc[agg["marker"] == marker, "avg_value"].values.astype(float)
        groups_here = agg.loc[agg["marker"] == marker, "group"].tolist()
        if len(vals) == 0:
            continue
        m, s = np.nanmean(vals), np.nanstd(vals)
        for g, v in zip(groups_here, vals):
            z = 0.0 if s == 0 or not np.isfinite(s) else (v - m) / s
            z_rows.append({"marker": marker, "group": g, "zscore": float(z)})
    zdf = pd.DataFrame(z_rows)
    agg = agg.merge(zdf, on=["marker", "group"], how="left")
    y_labels = top_markers[::-1]
    x_map = {g: i for i, g in enumerate(GROUP_ORDER)}
    y_map = {m: i for i, m in enumerate(y_labels)}

    fig, ax = plt.subplots(figsize=(7.6, max(4.8, 0.45 * len(top_markers) + 1.6)))
    sizes = 24 + (agg["pct_pos"].fillna(0).values / 100.0) * 240
    sc = ax.scatter(
        [x_map[g] for g in agg["group"]],
        [y_map[m] for m in agg["marker"]],
        s=sizes,
        c=agg["zscore"].fillna(0).values,
        cmap="coolwarm", vmin=-2, vmax=2,
        edgecolors="black", linewidths=0.35
    )
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_DISPLAY[g] for g in GROUP_ORDER], rotation=25, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title("dotplot (Top 10 markers)")
    ax.set_xlabel("Group")
    ax.set_ylabel("Marker")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Row Z-score")
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"dotplot.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

def build_wide_table(patient_df, markers, value_col="mean_value"):
    wide = patient_df[patient_df["marker"].isin(markers)].pivot_table(
        index=["patient", "group"], columns="marker", values=value_col, aggfunc="mean"
    ).reset_index()
    return wide

def safe_cv_splits(y, upper=5):
    counts = np.bincount(y)
    positive_counts = counts[counts > 0]
    if len(positive_counts) == 0:
        return 2
    return max(2, min(upper, int(positive_counts.min())))

def plot_binary_single_marker_rocs(patient_df, top_markers, value_col, out_dir: Path,
                                   positive_groups=("High", "Middle", "Low"), negative_groups=("Normal",)):
    roc_folder = out_dir / "opSubset_single_marker_ROC"
    roc_folder.mkdir(parents=True, exist_ok=True)
    sub = patient_df[patient_df["group"].isin(list(positive_groups) + list(negative_groups))].copy()
    sub["label"] = sub["group"].isin(positive_groups).astype(int)

    auc_rows = []
    for marker in top_markers:
        dm = sub[sub["marker"] == marker].copy()
        if len(dm) < 6 or dm["label"].nunique() < 2:
            continue
        X = dm[[value_col]].values.astype(float)
        y = dm["label"].values.astype(int)
        cv = StratifiedKFold(n_splits=safe_cv_splits(y), shuffle=True, random_state=42)
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ])
        prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)
        auc_rows.append({"marker": marker, "binary_auc": float(roc_auc)})

        fig, ax = plt.subplots(figsize=(5.3, 4.6))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC: {marker}\nNon-normal vs Normal")
        ax.legend(frameon=False, loc="lower right")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        safe_marker = re.sub(r"[^A-Za-z0-9._-]+", "_", marker)
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(roc_folder / f"{safe_marker}_ROC.{ext}", dpi=600 if ext == "png" else None,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)

    auc_df = pd.DataFrame(auc_rows).sort_values("binary_auc", ascending=False)
    auc_df.to_csv(out_dir / "opSubset_single_marker_ROC_auc_table.csv", index=False, encoding="utf-8-sig")
    if len(auc_df) > 0:
        show = auc_df.iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.45 * len(show) + 1.2)))
        ax.barh(show["marker"], show["binary_auc"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("AUC")
        ax.set_title("AUC ranking: single-marker ROC")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(out_dir / f"AUC_ranking_single_marker_binary.{ext}", dpi=600 if ext == "png" else None,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)

def plot_binary_combined_roc(patient_df, top_markers, value_col, out_dir: Path,
                             positive_groups=("High", "Middle", "Low"), negative_groups=("Normal",)):
    wide = build_wide_table(patient_df[patient_df["group"].isin(list(positive_groups) + list(negative_groups))], top_markers, value_col=value_col)
    if len(wide) < 6:
        return
    wide["label"] = wide["group"].isin(positive_groups).astype(int)
    y = wide["label"].values.astype(int)
    if len(np.unique(y)) < 2:
        return
    X = wide[top_markers].copy().values.astype(float)

    cv = StratifiedKFold(n_splits=safe_cv_splits(y), shuffle=True, random_state=42)
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000))
    ])
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.plot(fpr, tpr, lw=2.2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("topSubset_combined_model_ROC\nNon-normal vs Normal")
    ax.legend(frameon=False, loc="lower right")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"topSubset_combined_model_ROC.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame({"combined_model_auc": [roc_auc]}).to_csv(out_dir / "topSubset_combined_model_auc.csv", index=False, encoding="utf-8-sig")

def plot_multiclass_ovr_rocs(patient_df, top_markers, value_col, out_dir: Path):
    wide = build_wide_table(patient_df, top_markers, value_col=value_col)
    if len(wide) < 8:
        return
    class_order = GROUP_ORDER[:]
    class_to_idx = {g: i for i, g in enumerate(class_order)}
    y = wide["group"].map(class_to_idx).values.astype(int)
    if len(np.unique(y)) < 3:
        return
    X = wide[top_markers].copy().values.astype(float)

    cv = StratifiedKFold(n_splits=safe_cv_splits(y), shuffle=True, random_state=42)
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", OneVsRestClassifier(LogisticRegression(max_iter=3000)))
    ])

    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    y_bin = label_binarize(y, classes=np.arange(len(class_order)))

    auc_rows = []
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for i, g in enumerate(class_order):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_rows.append({"class": g, "one_vs_rest_auc": float(roc_auc)})
        ax.plot(fpr, tpr, lw=2, label=f"{GROUP_DISPLAY[g]} (AUC={roc_auc:.3f})", color=GROUP_COLORS[g])

    micro_fpr, micro_tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
    micro_auc = auc(micro_fpr, micro_tpr)
    ax.plot(micro_fpr, micro_tpr, linestyle="--", lw=1.5, label=f"Micro-average (AUC={micro_auc:.3f})", color="black")

    ax.plot([0, 1], [0, 1], "--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Multiclass ROC / One-vs-Rest ROC")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"multiclass_one_vs_rest_ROC.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    auc_df = pd.DataFrame(auc_rows)
    auc_df.loc[len(auc_df)] = {"class": "micro_average", "one_vs_rest_auc": float(micro_auc)}
    auc_df.to_csv(out_dir / "multiclass_one_vs_rest_auc_table.csv", index=False, encoding="utf-8-sig")

def compute_marker_macro_ovr_auc(patient_df, top_markers, value_col):
    class_order = GROUP_ORDER[:]
    class_to_idx = {g: i for i, g in enumerate(class_order)}
    rows = []

    for marker in top_markers:
        dm = patient_df[patient_df["marker"] == marker].copy()
        if len(dm) < 8:
            continue
        y = dm["group"].map(class_to_idx).values.astype(int)
        if len(np.unique(y)) < 3:
            continue
        X = dm[[value_col]].values.astype(float)
        cv = StratifiedKFold(n_splits=safe_cv_splits(y), shuffle=True, random_state=42)
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", OneVsRestClassifier(LogisticRegression(max_iter=3000)))
        ])
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
        y_bin = label_binarize(y, classes=np.arange(len(class_order)))

        auc_list = []
        for i, g in enumerate(class_order):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            auc_list.append(auc(fpr, tpr))
        if len(auc_list) == 0:
            continue
        rows.append({"marker": marker, "macro_ovr_auc": float(np.mean(auc_list))})
    return pd.DataFrame(rows).sort_values("macro_ovr_auc", ascending=False)

def plot_auc_ranking_multiclass(patient_df, top_markers, value_col, out_dir: Path):
    auc_df = compute_marker_macro_ovr_auc(patient_df, top_markers, value_col)
    auc_df.to_csv(out_dir / "AUC_ranking_multiclass_marker_table.csv", index=False, encoding="utf-8-sig")
    if len(auc_df) == 0:
        return
    show = auc_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.45 * len(show) + 1.2)))
    ax.barh(show["marker"], show["macro_ovr_auc"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro one-vs-rest AUC")
    ax.set_title("AUC ranking: multiclass one-vs-rest")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"AUC_ranking_multiclass_one_vs_rest.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Top10 marker all-in-one: optimized violin + p/q labels + ROC + AUC ranking")
    parser.add_argument("--data-dir", type=str, default="data_2d")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--target-layer", type=int, default=1)
    parser.add_argument("--value-col", type=str, default="mean_value", choices=["mean_value", "median_value", "percent_positive"])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-n-per-group", type=int, default=2)
    parser.add_argument("--only-significant", action="store_true", default=True)
    parser.add_argument("--pixel-per-file", type=int, default=800)
    parser.add_argument("--pixel-per-group-marker", type=int, default=3000)
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
    if len(meta_df) == 0:
        raise SystemExit("No recognized .mat files found.")

    patient_df, skipped_load = build_patient_level_table(meta_df, target_layer=args.target_layer)
    if len(skipped_load) > 0:
        skipped_load.to_csv(out_dir / "skipped_after_loading.csv", index=False, encoding="utf-8-sig")
    if len(patient_df) == 0:
        raise SystemExit("No valid patient-level rows.")

    global_df, pair_df = compare_all_markers(patient_df, value_col=args.value_col, alpha=args.alpha, min_n_per_group=args.min_n_per_group)
    top_markers = choose_top_markers(global_df, top_n=args.top_n, only_significant=args.only_significant)

    global_df.to_csv(out_dir / "significant_marker_ranking.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(out_dir / "pairwise_marker_statistics.csv", index=False, encoding="utf-8-sig")
    patient_df.to_csv(out_dir / "patient_level_marker_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"top_markers": top_markers}).to_csv(out_dir / "top10_marker_list.csv", index=False, encoding="utf-8-sig")

    plot_significant_marker_ranking(global_df, top_markers, out_dir)
    plot_dotplot(patient_df, top_markers, args.value_col, out_dir)
    plot_violin_samplelevel(patient_df, top_markers, global_df, pair_df, out_dir, value_col=args.value_col)

    pixel_df = build_pixel_level_subset(meta_df, top_markers, target_layer=args.target_layer,
                                        per_file_max=args.pixel_per_file, per_group_marker_max=args.pixel_per_group_marker)
    if len(pixel_df) > 0:
        pixel_df.to_csv(out_dir / "topSubset_violin_pixellevel_exploratory_table.csv", index=False, encoding="utf-8-sig")
        plot_violin_pixellevel(pixel_df, top_markers, out_dir)

    plot_binary_single_marker_rocs(patient_df, top_markers, args.value_col, out_dir)
    plot_binary_combined_roc(patient_df, top_markers, args.value_col, out_dir)
    plot_multiclass_ovr_rocs(patient_df, top_markers, args.value_col, out_dir)
    plot_auc_ranking_multiclass(patient_df, top_markers, args.value_col, out_dir)

    params = vars(args)
    params["n_files_found"] = len(files)
    params["n_patient_rows"] = len(patient_df)
    params["n_markers_tested"] = int(global_df.shape[0])
    params["top_markers"] = top_markers
    with open(out_dir / "run_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    readme = [
        "All-in-one top10 marker analysis suite",
        "",
        "Main outputs:",
        "  1) significant_marker_ranking.*",
        "  2) dotplot.*",
        "  3) topSubset_violin_samplelevel_optimized_pvalues.*",
        "  4) topSubset_violin_pixellevel_exploratory_optimized.*",
        "  5) opSubset_single_marker_ROC/ (each top marker single ROC)",
        "  6) AUC_ranking_single_marker_binary.*",
        "  7) topSubset_combined_model_ROC.*",
        "  8) multiclass_one_vs_rest_ROC.*",
        "  9) AUC_ranking_multiclass_one_vs_rest.*",
        "",
        "Statistics:",
        "  - Global Kruskal-Wallis + FDR q-value",
        "  - Pairwise Mann-Whitney U + FDR q-value",
        "  - Violin plot shows up to 2 significant pairwise q-value brackets",
        "",
        "Recommended run:",
        "  python scripts/2d/fig12_top10_marker_allinone.py --data-dir data_2d --top-n 10 --only-significant",
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
