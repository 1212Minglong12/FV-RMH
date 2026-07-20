#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fig05 v3: true SHAP TreeExplainer marker-driver triptych with cleaner color and non-overlapping dependence labels.

This version uses the installed SHAP package and a nonlinear Random Forest
one-vs-rest pathology classifier. Unlike the earlier logistic-contribution
version, Panel C is based on actual Tree SHAP values and can capture nonlinear
feature effects.

Input folder (from Fig01 v7):
  source_embedding_coordinates.csv
  source_true_pixel_robust_z_features.csv

Outputs:
  Fig05_3D_true_SHAP_marker_driver_triptych_v3_clean.png/pdf/svg
  source_shap_driver_ranking.csv
  source_shap_model_summary.txt
  source_shap_plot_sample_metadata.csv
  source_shap_top_marker_values.csv

Recommended command:
cd "FV-RMH"

python ".\\scripts\\3d\\fig05_3d_true_shap_marker_driver_triptych_v3_clean.py" ^
  --input-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_2000pixels" ^
  --out-dir "3dresults\\fig05_3d_true_shap_marker_driver_triptych_v3_clean_2000pixels"

Notes:
- Pixels are model observations for spatial molecular-state interpretation;
  they are not independent patient-level replicates.
- The script uses stratified sampling for model fitting and SHAP plotting to
  control runtime while retaining equal representation of pathology groups.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "The SHAP package is not installed. In PowerShell run:\n"
        "py -m pip install --upgrade pip\n"
        "py -m pip install shap scikit-learn pandas matplotlib scipy\n"
    ) from exc

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

GROUP_ORDER = [
    "Cancer-adjacent", "Lepidic", "Acinar", "Papillary",
    "Micropapillary", "Complex glands", "Solid"
]
GROUP_SHORT = {
    "Cancer-adjacent": "CA", "Lepidic": "LEP", "Acinar": "ACN",
    "Papillary": "PAP", "Micropapillary": "MP",
    "Complex glands": "CGP", "Solid": "SOL"
}
BAR_COLOR = "#7B644E"

def get_soft_shap_cmap():
    """Magma without the nearly-black lower end; improves SHAP swarm readability."""
    base = plt.get_cmap("magma")
    return plt.matplotlib.colors.ListedColormap(base(np.linspace(0.18, 0.98, 256)))


def clean_marker_label(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).replace("_", " ")).strip()


def load_inputs(input_dir: Path):
    emb_path = input_dir / "source_embedding_coordinates.csv"
    feature_path = input_dir / "source_true_pixel_robust_z_features.csv"

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing {feature_path}")

    coords = pd.read_csv(emb_path)
    features = pd.read_csv(feature_path)

    if len(coords) != len(features):
        raise RuntimeError(
            f"Embedding rows ({len(coords)}) and feature rows ({len(features)}) do not match."
        )

    keep = coords["group"].isin(GROUP_ORDER).values
    coords = coords.loc[keep].reset_index(drop=True)
    features = features.loc[keep].reset_index(drop=True)

    usable = [
        col for col in features.columns
        if np.isfinite(features[col].values.astype(float)).any()
        and np.nanstd(features[col].values.astype(float)) > 1e-10
    ]
    features = features[usable].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)

    if features.shape[1] < 3:
        raise RuntimeError("Too few nonconstant Raman marker features.")
    return coords, features


def stratified_sample_indices(labels: np.ndarray, n_total: int, random_state: int):
    labels = np.asarray(labels)
    rng = np.random.default_rng(random_state)
    n_total = min(int(n_total), len(labels))

    unique, counts = np.unique(labels, return_counts=True)
    base = n_total // len(unique)
    remainder = n_total - base * len(unique)

    chosen = []
    for i, (group, count) in enumerate(zip(unique, counts)):
        idx = np.where(labels == group)[0]
        target = base + (1 if i < remainder else 0)
        target = min(target, len(idx))
        chosen.extend(rng.choice(idx, size=target, replace=False).tolist())

    if len(chosen) < n_total:
        remain = np.setdiff1d(np.arange(len(labels)), np.asarray(chosen, dtype=int))
        extra = rng.choice(remain, size=min(n_total - len(chosen), len(remain)), replace=False)
        chosen.extend(extra.tolist())

    return np.asarray(sorted(chosen), dtype=int)


def normalize_shap_output(raw_values, n_samples: int, n_features: int, n_classes: int):
    """
    SHAP has different return shapes across versions/models. Convert all
    supported layouts to (n_samples, n_features, n_classes).
    """
    if isinstance(raw_values, list):
        arr = np.stack([np.asarray(x) for x in raw_values], axis=-1)
    else:
        arr = np.asarray(raw_values)

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    if arr.ndim != 3:
        raise RuntimeError(f"Unsupported SHAP output shape: {arr.shape}")

    # Current SHAP commonly: [sample, feature, class]
    if arr.shape == (n_samples, n_features, n_classes):
        return arr

    # Legacy multiclass: [class, sample, feature]
    if arr.shape == (n_classes, n_samples, n_features):
        return np.moveaxis(arr, 0, -1)

    # Alternative: [sample, class, feature]
    if arr.shape == (n_samples, n_classes, n_features):
        return np.moveaxis(arr, 1, -1)

    # Binary or a one-class explanation.
    if arr.shape[0] == n_samples and arr.shape[1] == n_features:
        return arr[:, :, :1]

    raise RuntimeError(
        f"Could not normalize SHAP output shape {arr.shape}; "
        f"expected samples={n_samples}, features={n_features}, classes={n_classes}."
    )


def fit_model_and_compute_shap(coords, features, train_rows, shap_rows, n_trees, random_state):
    labels = coords["group"].astype(str).values
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = encoder.classes_

    train_idx = stratified_sample_indices(y, train_rows, random_state)
    X_train = features.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]

    model = RandomForestClassifier(
        n_estimators=int(n_trees),
        max_depth=12,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Internal held-out estimate, kept as a descriptive QC metric only.
    balanced_acc = np.nan
    try:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
        tr, te = next(split.split(X_train, y_train))
        qc_model = RandomForestClassifier(
            n_estimators=max(120, int(n_trees // 2)),
            max_depth=12,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state + 9,
        )
        qc_model.fit(X_train.iloc[tr], y_train[tr])
        balanced_acc = balanced_accuracy_score(y_train[te], qc_model.predict(X_train.iloc[te]))
    except Exception:
        pass

    shap_idx = stratified_sample_indices(y, shap_rows, random_state + 19)
    X_shap = features.iloc[shap_idx].reset_index(drop=True)
    meta_shap = coords.iloc[shap_idx].reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_shap)
    shap_values = normalize_shap_output(
        raw_shap,
        n_samples=len(X_shap),
        n_features=X_shap.shape[1],
        n_classes=len(classes),
    )

    # RandomForest multiclass should include one SHAP output per fitted class.
    if shap_values.shape[2] != len(classes):
        # Fall back to one explanation axis if a SHAP version returns only a
        # binary-style channel. This remains valid for the visualization.
        classes_for_shap = np.array([classes[-1]])
    else:
        classes_for_shap = classes

    return model, classes_for_shap, X_shap, meta_shap, shap_values, balanced_acc


def build_ranking(features, shap_values, classes_for_shap):
    # Mean absolute SHAP, averaged across pixels and output classes.
    mean_abs_per_class = np.nanmean(np.abs(shap_values), axis=0)  # feature x class
    overall = np.nanmean(mean_abs_per_class, axis=1)

    best_class_index = np.argmax(mean_abs_per_class, axis=1)
    rows = []
    for j, marker in enumerate(features.columns):
        cls_idx = min(int(best_class_index[j]), len(classes_for_shap) - 1)
        rows.append({
            "marker": marker,
            "mean_abs_shap": float(overall[j]),
            "top_output_group": str(classes_for_shap[cls_idx]),
            "top_output_group_short": GROUP_SHORT.get(str(classes_for_shap[cls_idx]), str(classes_for_shap[cls_idx])),
            "best_class_index": int(best_class_index[j]),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def feature_value_normalization(X):
    out = pd.DataFrame(index=X.index)
    for col in X.columns:
        values = X[col].values.astype(float)
        lo, hi = np.nanpercentile(values, [1, 99])
        if not np.isfinite(hi - lo) or (hi - lo) <= 1e-12:
            out[col] = 0.5
        else:
            out[col] = np.clip((values - lo) / (hi - lo), 0, 1)
    return out


def beeswarm_offsets(values, base_y, bins=42, max_spread=0.30, random_state=1):
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    offsets = np.zeros_like(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.full_like(values, base_y, dtype=float)

    v = values[finite]
    lo, hi = np.nanpercentile(v, [0.5, 99.5])
    if hi <= lo:
        lo, hi = np.nanmin(v), np.nanmax(v) + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    bin_ids = np.clip(np.digitize(v, edges) - 1, 0, bins - 1)

    finite_offsets = np.zeros(len(v), dtype=float)
    for b in range(bins):
        idx = np.where(bin_ids == b)[0]
        if len(idx) == 0:
            continue
        stack = np.arange(len(idx), dtype=float) - (len(idx) - 1) / 2
        denom = max(1.0, np.max(np.abs(stack)))
        stack = stack / denom
        finite_offsets[idx] = stack * max_spread * min(1.0, len(idx) / 80.0)
        finite_offsets[idx] += rng.normal(0, 0.008, len(idx))

    offsets[finite] = finite_offsets
    return offsets + base_y


def get_marker_class_shap(shap_values, ranking_row, feature_index):
    class_idx = int(ranking_row["best_class_index"])
    class_idx = min(class_idx, shap_values.shape[2] - 1)
    return shap_values[:, feature_index, class_idx], class_idx


def draw_bar_panel(ax, ranking, top_n):
    show = ranking.head(top_n).iloc[::-1]
    y = np.arange(len(show))
    ax.barh(y, show["mean_abs_shap"], color=BAR_COLOR, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels([clean_marker_label(x) for x in show["marker"]], fontsize=7.5, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|", fontsize=8.5)
    ax.set_title("A. Dominant Raman marker drivers", loc="left", fontsize=10.5, pad=6)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_swarm_panel(ax, ranking, X_shap, shap_values, fval_norm, top_n, max_points=7000, random_state=1):
    rng = np.random.default_rng(random_state)
    markers = ranking.head(top_n)["marker"].tolist()
    reverse_markers = markers[::-1]
    cmap = get_soft_shap_cmap()

    for i, marker in enumerate(reverse_markers):
        marker_index = X_shap.columns.get_loc(marker)
        row = ranking.loc[ranking["marker"] == marker].iloc[0]
        contributions, _ = get_marker_class_shap(shap_values, row, marker_index)
        colors = fval_norm[marker].values.astype(float)

        if len(contributions) > max_points:
            idx = rng.choice(len(contributions), size=max_points, replace=False)
            contributions = contributions[idx]
            colors = colors[idx]

        yy = beeswarm_offsets(contributions, i, random_state=random_state + i)
        ax.scatter(
            contributions,
            yy,
            c=colors,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=3.8,
            alpha=0.50,
            linewidths=0,
            rasterized=True,
        )

    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(np.arange(len(reverse_markers)))
    ax.set_yticklabels([clean_marker_label(x) for x in reverse_markers], fontsize=7.0)
    ax.set_xlabel("SHAP value (impact on pathology model output)", fontsize=8.5)
    ax.set_title("B. SHAP mechanism across pixel states", loc="left", fontsize=10.5, pad=6)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Avoid an over-black terminal band by using robust symmetric limits.
    all_x = []
    for marker in markers:
        marker_index = X_shap.columns.get_loc(marker)
        row = ranking.loc[ranking["marker"] == marker].iloc[0]
        values, _ = get_marker_class_shap(shap_values, row, marker_index)
        all_x.append(values)
    all_x = np.concatenate(all_x)
    lim = np.nanpercentile(np.abs(all_x), 99.2)
    if np.isfinite(lim) and lim > 0:
        ax.set_xlim(-lim * 1.08, lim * 1.08)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Feature value", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)


def choose_interaction_marker(top_index, top_shap_2d, X_shap, ranking):
    # SHAP built-in approximate interaction ranking; fall back to correlation.
    try:
        interaction_order = shap.utils.approximate_interactions(
            top_index,
            top_shap_2d,
            X_shap,
        )
        for idx in interaction_order:
            if int(idx) != int(top_index):
                return X_shap.columns[int(idx)]
    except Exception:
        pass

    top_marker = X_shap.columns[top_index]
    candidates = [m for m in ranking.head(min(15, len(ranking)))["marker"].tolist() if m != top_marker]
    x = X_shap[top_marker].values.astype(float)
    best_marker = candidates[0] if candidates else top_marker
    best_value = -1.0
    for marker in candidates:
        r = np.corrcoef(x, X_shap[marker].values.astype(float))[0, 1]
        if np.isfinite(r) and abs(r) > best_value:
            best_marker = marker
            best_value = abs(r)
    return best_marker


def draw_dependence_panel(ax, ranking, X_shap, shap_values, fval_norm, max_points=9000, random_state=1):
    rng = np.random.default_rng(random_state)
    row = ranking.iloc[0]
    top_marker = row["marker"]
    top_idx = X_shap.columns.get_loc(top_marker)
    shap_y, class_idx = get_marker_class_shap(shap_values, row, top_idx)

    # Use the class-specific SHAP matrix to identify an interacting marker.
    class_idx = min(class_idx, shap_values.shape[2] - 1)
    class_shap_2d = shap_values[:, :, class_idx]
    interaction_marker = choose_interaction_marker(top_idx, class_shap_2d, X_shap, ranking)
    x = X_shap[top_marker].values.astype(float)
    c = fval_norm[interaction_marker].values.astype(float)
    y = shap_y.astype(float)

    if len(x) > max_points:
        idx = rng.choice(len(x), size=max_points, replace=False)
        x, y, c = x[idx], y[idx], c[idx]

    ax.scatter(
        x, y,
        c=c,
        cmap=get_soft_shap_cmap(),
        vmin=0,
        vmax=1,
        s=3.8,
        alpha=0.34,
        linewidths=0,
        rasterized=True,
    )

    # Smoothed median SHAP trend. It reflects the model's actual nonlinear
    # dependence; it is not forced to be curved.
    valid = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[valid], y[valid]
    if len(xx) > 120:
        edges = np.linspace(np.nanpercentile(xx, 1), np.nanpercentile(xx, 99), 42)
        bx, by = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (xx >= lo) & (xx < hi)
            if mask.sum() >= 20:
                bx.append((lo + hi) / 2)
                by.append(np.nanmedian(yy[mask]))
        if len(bx) >= 5:
            bx = np.asarray(bx)
            by = np.asarray(by)
            try:
                spl = UnivariateSpline(bx, by, s=len(bx) * max(np.nanvar(by), 1e-8) * 0.50)
                xs = np.linspace(bx.min(), bx.max(), 220)
                ax.plot(xs, spl(xs), color="#E83D5B", lw=1.5)
            except Exception:
                ax.plot(bx, by, color="#E83D5B", lw=1.5)

    output_group = str(row["top_output_group"])
    ax.set_xlabel(clean_marker_label(top_marker), fontsize=8.5)
    ax.set_ylabel(f"SHAP value for {GROUP_SHORT.get(output_group, output_group)} output", fontsize=8.5)
    ax.set_title(f"C. Nonlinear interaction of {clean_marker_label(top_marker)}", loc="left", fontsize=10.5, pad=6)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Put interaction label and color strip outside the axes so it never covers points.
    ax.text(
        1.04, 0.68,
        f"{clean_marker_label(interaction_marker)}\n(feature value)",
        transform=ax.transAxes,
        fontsize=7.5,
        ha="left",
        va="center",
        clip_on=False,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.5),
    )
    grad = np.linspace(0, 1, 128).reshape(1, -1)
    inset = ax.inset_axes([1.04, 0.56, 0.34, 0.030], transform=ax.transAxes)
    inset.imshow(grad, aspect="auto", cmap=get_soft_shap_cmap())
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)

    # Add extra right-side whitespace for the external annotation.
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.18)

    return top_marker, interaction_marker, output_group


def save_multi(fig, out_dir, stem, dpi=600, pad=0.02):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=pad)


def make_main_figure(ranking, X_shap, shap_values, fval_norm, out_dir, top_n):
    fig = plt.figure(figsize=(14.6, 5.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.18, 1.32], wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    draw_bar_panel(ax_a, ranking, top_n)

    ax_b = fig.add_subplot(gs[0, 1])
    draw_swarm_panel(ax_b, ranking, X_shap, shap_values, fval_norm, top_n)

    ax_c = fig.add_subplot(gs[0, 2])
    selected = draw_dependence_panel(ax_c, ranking, X_shap, shap_values, fval_norm)

    save_multi(fig, out_dir, "Fig05_3D_true_SHAP_marker_driver_triptych_v3_clean")
    plt.close(fig)
    return selected


def export_single_panels(ranking, X_shap, shap_values, fval_norm, out_dir, top_n):
    pdir = out_dir / "single_panels"
    pdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 5.0))
    draw_bar_panel(ax, ranking, top_n)
    fig.tight_layout()
    save_multi(fig, pdir, "panel_A_true_shap_driver_bar")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.25, 5.0))
    draw_swarm_panel(ax, ranking, X_shap, shap_values, fval_norm, top_n)
    fig.tight_layout()
    save_multi(fig, pdir, "panel_B_true_shap_swarm")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.9, 5.0))
    draw_dependence_panel(ax, ranking, X_shap, shap_values, fval_norm)
    fig.tight_layout()
    save_multi(fig, pdir, "panel_C_true_shap_dependence")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="True SHAP marker-driver triptych for 3D Raman data.")
    parser.add_argument("--input-dir", required=True, help="Fig01 v7 output folder.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=18)
    parser.add_argument("--train-rows", type=int, default=28000)
    parser.add_argument("--shap-rows", type=int, default=9000)
    parser.add_argument("--n-trees", type=int, default=260)
    parser.add_argument("--random-state", type=int, default=1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords, features = load_inputs(input_dir)

    model, classes_for_shap, X_shap, meta_shap, shap_values, balanced_acc = fit_model_and_compute_shap(
        coords=coords,
        features=features,
        train_rows=args.train_rows,
        shap_rows=args.shap_rows,
        n_trees=args.n_trees,
        random_state=args.random_state,
    )

    ranking = build_ranking(X_shap, shap_values, classes_for_shap)
    fval_norm = feature_value_normalization(X_shap)
    top_n = min(int(args.top_n), len(ranking))

    top_marker, interaction_marker, output_group = make_main_figure(
        ranking, X_shap, shap_values, fval_norm, out_dir, top_n
    )
    export_single_panels(ranking, X_shap, shap_values, fval_norm, out_dir, top_n)

    ranking.to_csv(out_dir / "source_shap_driver_ranking.csv", index=False)
    meta_shap.to_csv(out_dir / "source_shap_plot_sample_metadata.csv", index=False)
    pd.DataFrame({
        "marker": X_shap.columns,
        "feature_importance_random_forest": model.feature_importances_,
    }).sort_values("feature_importance_random_forest", ascending=False).to_csv(
        out_dir / "source_random_forest_feature_importance.csv", index=False
    )

    # Compact source export: actual SHAP values for the plotted top markers and
    # their strongest one-vs-rest output class.
    top_markers = ranking.head(top_n)["marker"].tolist()
    shap_export = pd.DataFrame(index=np.arange(len(X_shap)))
    for marker in top_markers:
        j = X_shap.columns.get_loc(marker)
        row = ranking.loc[ranking["marker"] == marker].iloc[0]
        values, _ = get_marker_class_shap(shap_values, row, j)
        shap_export[f"SHAP__{marker}"] = values
        shap_export[f"Feature__{marker}"] = X_shap[marker].values
    shap_export.to_csv(out_dir / "source_shap_top_marker_values.csv", index=False)

    summary = (
        "Fig05 true SHAP marker-driver summary\n"
        "=====================================\n"
        f"Input folder: {input_dir}\n"
        f"Total input pixel observations: {len(features)}\n"
        f"Model training observations: {min(args.train_rows, len(features))}\n"
        f"SHAP plotting observations: {len(X_shap)}\n"
        f"Marker features: {features.shape[1]}\n"
        f"Random Forest trees: {args.n_trees}\n"
        f"Internal held-out balanced accuracy: {balanced_acc:.4f}\n"
        f"Top SHAP driver used in Panel C: {top_marker}\n"
        f"Interaction marker used in Panel C: {interaction_marker}\n"
        f"Panel C model output class: {output_group}\n\n"
        "SHAP values were computed with shap.TreeExplainer on a nonlinear "
        "RandomForestClassifier. Pixel-level observations were used for spatial "
        "molecular-state interpretation and must not be treated as independent "
        "patient-level biological replicates.\n"
    )
    (out_dir / "source_shap_model_summary.txt").write_text(summary, encoding="utf-8")

    print("Done.")
    print(f"Output folder: {out_dir}")
    print(f"SHAP version: {getattr(shap, '__version__', 'unknown')}")
    print(f"Training rows: {min(args.train_rows, len(features))}")
    print(f"SHAP rows: {len(X_shap)}")
    print(f"Internal balanced accuracy: {balanced_acc:.4f}")
    print("Main figure: Fig05_3D_true_SHAP_marker_driver_triptych_v3_clean.png")


if __name__ == "__main__":
    main()
