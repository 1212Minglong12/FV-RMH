#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Paper model 01: Final 2D Raman H/M/L differentiation classifier

Final manuscript-ready 2D model:
- Groups: High / Middle / Low
- Feature: 10%–90% percentile trimming + log1p + IQR
- Marker panel: Top 30 markers selected inside each CV training fold
- Model: class-balanced multinomial logistic regression
- Validation: patient-level leave-one-patient-out CV when possible

Input filename format:
    Marker-Group-Patient_layer1.mat
Example:
    CD98-High-PatientA_layer1.mat

Default output:
    paper_model_outputs/01_2d_HML_classifier
"""

from pathlib import Path
import re
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.decomposition import PCA
import joblib

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

GROUP_ALIASES = {
    "High": "High", "WD": "High", "Well": "High",
    "WellDifferentiated": "High", "Well_Differentiated": "High",
    "Middle": "Middle", "Moderate": "Middle", "MD": "Middle", "Mid": "Middle",
    "ModeratelyDifferentiated": "Middle", "Moderately_Differentiated": "Middle",
    "Low": "Low", "Poor": "Low", "PD": "Low",
    "PoorlyDifferentiated": "Low", "Poorly_Differentiated": "Low",
    "Normal": "Normal", "NAT": "Normal", "Healthy": "Normal", "Control": "Normal",
}
GROUPS = ["High", "Middle", "Low"]
GROUP_DISPLAY = {
    "High": "High differentiated",
    "Middle": "Moderately differentiated",
    "Low": "Poorly differentiated",
}
GROUP_COLORS = {"High": "#7BC69D", "Middle": "#F2C38E", "Low": "#E57D7D"}


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
                numeric_arrays = []
                for k in f.keys():
                    try:
                        v = np.array(f[k])
                        if np.issubdtype(v.dtype, np.number):
                            numeric_arrays.append((k, v))
                    except Exception:
                        pass
                if not numeric_arrays:
                    raise ValueError("No numeric array found in v7.3 mat")
                _, arr = max(numeric_arrays, key=lambda kv: kv[1].size)
        arr = np.asarray(arr).squeeze().astype(float).flatten()
        arr[~np.isfinite(arr)] = np.nan
        return arr


def preprocess_to_iqr_feature(vec, lower_q=0.10, upper_q=0.90):
    x = np.asarray(vec, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    lo = np.nanquantile(x, lower_q)
    hi = np.nanquantile(x, upper_q)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        x = np.clip(x, lo, hi)
    min_x = np.nanmin(x)
    if min_x < 0:
        x = x - min_x
    x = np.log1p(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, 0.75) - np.quantile(x, 0.25))


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


def build_2d_feature_table(data_dir: Path, target_layer=1):
    files = sorted([p for p in data_dir.rglob("*.mat") if p.is_file()])
    parsed, skipped_parse = [], []
    for f in files:
        info = parse_filename(f)
        if info is None:
            skipped_parse.append({"file": str(f), "reason": "filename_not_recognized"})
        else:
            parsed.append(info)
    meta_df = pd.DataFrame(parsed)
    rows, skipped_loading = [], []
    if meta_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(skipped_parse), pd.DataFrame()
    for _, row in meta_df.iterrows():
        if row["group"] not in GROUPS:
            continue
        if int(row["layer"]) != int(target_layer):
            continue
        try:
            vec = load_mat_numeric_array(Path(row["file"]))
            value = preprocess_to_iqr_feature(vec)
        except Exception as e:
            skipped_loading.append({**row.to_dict(), "reason": f"load_failed:{e}"})
            continue
        if not np.isfinite(value):
            skipped_loading.append({**row.to_dict(), "reason": "empty_after_preprocess"})
            continue
        rows.append({
            "patient": row["patient"], "group": row["group"], "layer": int(row["layer"]),
            "marker": row["marker"], "value": value, "file": row["file"],
        })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df, pd.DataFrame(), pd.DataFrame(skipped_parse), pd.DataFrame(skipped_loading)
    feature_df = long_df.pivot_table(
        index=["patient", "group", "layer"], columns="marker", values="value", aggfunc="mean"
    ).reset_index()
    feature_df.columns.name = None
    return long_df, feature_df, pd.DataFrame(skipped_parse), pd.DataFrame(skipped_loading)


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
    ])


def rank_markers_on_training(X_train, y_train, top_n=30):
    rows = []
    for marker in X_train.columns:
        arrays, group_stats = [], {}
        for c in GROUPS:
            vals = X_train.loc[np.asarray(y_train) == c, marker].dropna().values.astype(float)
            group_stats[c] = {
                "n": len(vals),
                "mean": float(np.mean(vals)) if len(vals) else np.nan,
                "median": float(np.median(vals)) if len(vals) else np.nan,
            }
            if len(vals) >= 1:
                arrays.append(vals)
        if len(arrays) < 2:
            p, effect = 1.0, 0.0
        else:
            try:
                H, p = stats.kruskal(*arrays)
                n_total = sum(len(a) for a in arrays)
                k = len(arrays)
                effect = max(0.0, (H - k + 1) / (n_total - k)) if n_total > k else 0.0
            except Exception:
                p, effect = 1.0, 0.0
        row = {"marker": marker, "p_value": float(p), "effect": float(effect)}
        for c in GROUPS:
            row[f"mean_{c}"] = group_stats[c]["mean"]
            row[f"median_{c}"] = group_stats[c]["median"]
            row[f"n_{c}"] = group_stats[c]["n"]
        rows.append(row)
    ranking = pd.DataFrame(rows)
    ranking["q_value"] = fdr_bh(ranking["p_value"].values.astype(float))
    ranking["score"] = -np.log10(ranking["q_value"].replace(0, 1e-300)) + ranking["effect"].fillna(0) * 3.0
    ranking = ranking.sort_values(["score", "effect"], ascending=[False, False]).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking["marker"].head(min(top_n, len(ranking))).tolist(), ranking


def get_cv_splits(y, patients):
    y = np.asarray(y)
    patients = np.asarray(patients)
    logo = LeaveOneGroupOut()
    splits = []
    ok = True
    for train_idx, test_idx in logo.split(np.zeros(len(y)), y, patients):
        if len(set(y[train_idx])) < len(GROUPS):
            ok = False
            break
        splits.append((train_idx, test_idx))
    if ok and len(splits) >= 2:
        return splits, "leave-one-patient-out CV"
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min()) if len(counts) else 0
    if min_count >= 2:
        n_splits = min(5, min_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(cv.split(np.zeros(len(y)), y)), f"stratified {n_splits}-fold CV"
    return None, "apparent training only - not validation"


def nested_cv_predict(feature_df, top_n=30):
    marker_cols = [c for c in feature_df.columns if c not in ["patient", "group", "layer"]]
    X_all = feature_df[marker_cols].copy()
    y_all = feature_df["group"].astype(str).values
    patients = feature_df["patient"].astype(str).values
    proba = np.full((len(feature_df), len(GROUPS)), np.nan)
    pred = np.array([""] * len(feature_df), dtype=object)
    marker_records, fold_rankings = [], []
    splits, cv_method = get_cv_splits(y_all, patients)
    if splits is None:
        selected, ranking = rank_markers_on_training(X_all, y_all, top_n=top_n)
        clf = make_model()
        clf.fit(X_all[selected], y_all)
        pp = clf.predict_proba(X_all[selected])
        model_classes = list(clf.named_steps["model"].classes_)
        aligned = np.zeros((len(feature_df), len(GROUPS)))
        for j, c in enumerate(model_classes):
            if c in GROUPS:
                aligned[:, GROUPS.index(c)] = pp[:, j]
        proba = aligned
        pred = np.array(GROUPS)[np.argmax(aligned, axis=1)]
        for m in selected:
            marker_records.append({"fold": "apparent", "marker": m})
        ranking["fold"] = "apparent"
        fold_rankings.append(ranking)
        return pred, proba, pd.DataFrame(marker_records), pd.concat(fold_rankings), cv_method
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, y_train = X_all.iloc[train_idx], y_all[train_idx]
        X_test = X_all.iloc[test_idx]
        selected, ranking = rank_markers_on_training(X_train, y_train, top_n=top_n)
        ranking["fold"] = fold
        fold_rankings.append(ranking)
        for m in selected:
            marker_records.append({"fold": fold, "marker": m})
        clf = make_model()
        clf.fit(X_train[selected], y_train)
        pp = clf.predict_proba(X_test[selected])
        model_classes = list(clf.named_steps["model"].classes_)
        aligned = np.zeros((len(test_idx), len(GROUPS)))
        for j, c in enumerate(model_classes):
            if c in GROUPS:
                aligned[:, GROUPS.index(c)] = pp[:, j]
        proba[test_idx] = aligned
        pred[test_idx] = np.array(GROUPS)[np.argmax(aligned, axis=1)]
    return pred, proba, pd.DataFrame(marker_records), pd.concat(fold_rankings, ignore_index=True), cv_method


def fit_final_model(feature_df, top_n=30):
    marker_cols = [c for c in feature_df.columns if c not in ["patient", "group", "layer"]]
    X = feature_df[marker_cols].copy()
    y = feature_df["group"].astype(str).values
    selected, ranking = rank_markers_on_training(X, y, top_n=top_n)
    clf = make_model()
    clf.fit(X[selected], y)
    return clf, selected, ranking


def save_fig(fig, out_dir: Path, name: str):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confusion(y_true, y_pred, out_dir, cv_method):
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    cm = confusion_matrix(y_true, y_pred, labels=GROUPS)
    disp = ConfusionMatrixDisplay(cm, display_labels=[GROUP_DISPLAY[g] for g in GROUPS])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"2D H/M/L classifier\n{cv_method}\nAccuracy={acc:.3f}, balanced accuracy={bal:.3f}")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, "Fig6_2D_HML_confusion_matrix")
    return acc, bal


def plot_roc(y_true, proba, out_dir, cv_method):
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    rows = []
    try:
        y_bin = label_binarize(y_true, classes=GROUPS)
        for i, g in enumerate(GROUPS):
            if y_bin[:, i].sum() == 0 or y_bin[:, i].sum() == len(y_bin):
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            roc_auc = auc(fpr, tpr)
            rows.append({"class": g, "one_vs_rest_auc": float(roc_auc)})
            ax.plot(fpr, tpr, lw=2, color=GROUP_COLORS[g], label=f"{GROUP_DISPLAY[g]} AUC={roc_auc:.3f}")
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        micro_auc = auc(fpr, tpr)
        rows.append({"class": "micro_average", "one_vs_rest_auc": float(micro_auc)})
        ax.plot(fpr, tpr, "--", color="black", lw=1.5, label=f"Micro-average AUC={micro_auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"One-vs-rest ROC\n{cv_method}")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        save_fig(fig, out_dir, "Fig6_2D_HML_one_vs_rest_ROC")
        auc_df = pd.DataFrame(rows)
        auc_df.to_csv(out_dir / "nested_cv_auc_table.csv", index=False, encoding="utf-8-sig")
        return auc_df
    except Exception:
        plt.close(fig)
        return pd.DataFrame()


def plot_probability_heatmap(feature_df, y_pred, proba, out_dir):
    df = feature_df[["patient", "group", "layer"]].copy()
    df["predicted_group"] = y_pred
    for i, g in enumerate(GROUPS):
        df[f"prob_{g}"] = proba[:, i]
    df["max_probability"] = proba.max(axis=1)
    df = df.sort_values(["group", "predicted_group", "max_probability"], ascending=[True, True, False])
    mat = df[[f"prob_{g}" for g in GROUPS]].values
    fig, ax = plt.subplots(figsize=(7.8, max(4.8, 0.32 * len(df) + 1.8)))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(GROUPS)))
    ax.set_xticklabels([GROUP_DISPLAY[g] for g in GROUPS], rotation=25, ha="right")
    labels = [f"{r.patient} | true {r.group} | pred {r.predicted_group}" for _, r in df.iterrows()]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("2D H/M/L prediction probability heatmap")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Probability")
    fig.tight_layout()
    save_fig(fig, out_dir, "Fig6_2D_HML_probability_heatmap")


def plot_marker_selection_frequency(marker_records_df, out_dir):
    if marker_records_df.empty:
        return pd.DataFrame()
    freq = marker_records_df.groupby("marker", as_index=False).size().rename(columns={"size": "selection_count"})
    freq = freq.sort_values("selection_count", ascending=False).reset_index(drop=True)
    freq.to_csv(out_dir / "nested_cv_marker_selection_frequency.csv", index=False, encoding="utf-8-sig")
    show = freq.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, max(4.8, 0.35 * len(show) + 1.6)))
    ax.barh(show["marker"], show["selection_count"])
    ax.set_xlabel("Selection count across CV folds")
    ax.set_title("Stable marker selection frequency")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, "Fig6_2D_HML_marker_selection_frequency")
    return freq


def plot_final_marker_weights(model, selected_markers, out_dir):
    try:
        coefs = model.named_steps["model"].coef_
        max_abs = np.max(np.abs(coefs), axis=0)
        order = np.argsort(max_abs)[::-1]
        labels = [selected_markers[i] for i in order][::-1]
        vals = max_abs[order][::-1]
        fig, ax = plt.subplots(figsize=(8.2, max(4.8, 0.40 * len(labels) + 1.6)))
        ax.barh(labels, vals)
        ax.set_xlabel("Maximum absolute standardized coefficient")
        ax.set_title("Final marker weight ranking")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        save_fig(fig, out_dir, "Fig6_2D_HML_final_marker_weight_ranking")
    except Exception:
        pass


def plot_selected_marker_pca(feature_df, selected_markers, out_dir):
    if len(feature_df) < 3 or len(selected_markers) < 2:
        return
    try:
        X = feature_df[selected_markers].copy()
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        emb = pca.fit_transform(X)
        y = feature_df["group"].astype(str).values
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        for g in GROUPS:
            idx = y == g
            ax.scatter(emb[idx, 0], emb[idx, 1], s=65, alpha=0.9, color=GROUP_COLORS[g],
                       edgecolors="white", linewidths=0.6, label=GROUP_DISPLAY[g])
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.set_title("PCA based on final selected markers")
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        save_fig(fig, out_dir, "Fig6_2D_HML_selected_marker_PCA")
    except Exception:
        pass


def export_formula(model, selected_markers, out_dir):
    scaler = model.named_steps["scaler"]
    imputer = model.named_steps["imputer"]
    lr = model.named_steps["model"]
    rows = []
    for i, marker in enumerate(selected_markers):
        row = {
            "marker": marker,
            "imputed_median": float(imputer.statistics_[i]),
            "standardization_mean": float(scaler.mean_[i]),
            "standardization_sd": float(scaler.scale_[i]),
        }
        for c_idx, c in enumerate(lr.classes_):
            row[f"coef_{c}"] = float(lr.coef_[c_idx, i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "final_model_formula_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"class": lr.classes_, "intercept": [float(x) for x in lr.intercept_]}).to_csv(
        out_dir / "final_model_formula_intercepts.csv", index=False, encoding="utf-8-sig"
    )
    lines = []
    lines.append("Final 2D H/M/L multinomial logistic regression formula")
    lines.append("")
    lines.append("Feature extraction for marker j:")
    lines.append("  M'_j = clip(M_j, Q10, Q90)")
    lines.append("  Z_j = log(1 + M'_j)")
    lines.append("  x_j = IQR(Z_j) = Q75(Z_j) - Q25(Z_j)")
    lines.append("  z_j = (x_j - mean_j) / sd_j")
    lines.append("")
    lines.append("Class score: S_k = beta_0k + sum_j beta_jk * z_j")
    lines.append("Class probability: P(k) = exp(S_k) / sum_c exp(S_c)")
    lines.append("Predicted class = argmax_k P(k), k in {High, Middle, Low}")
    lines.append("")
    lines.append("Numerical coefficients:")
    for c_idx, c in enumerate(lr.classes_):
        terms = [f"({lr.coef_[c_idx, i]:.6g} * z_{m})" for i, m in enumerate(selected_markers)]
        lines.append("")
        lines.append(f"Score_{c} = {lr.intercept_[c_idx]:.6g} + " + " + ".join(terms))
    (out_dir / "final_model_formula.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Final paper-ready 2D H/M/L Raman classifier")
    parser.add_argument("--data-dir", type=str, default="data_2d")
    parser.add_argument("--out-dir", type=str, default="paper_model_outputs/01_2d_HML_classifier")
    parser.add_argument("--target-layer", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    long_df, feature_df, skipped_parse, skipped_loading = build_2d_feature_table(data_dir, args.target_layer)
    long_df.to_csv(out_dir / "2d_long_marker_summary_iqr_features.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(out_dir / "2d_patient_feature_table_iqr_features.csv", index=False, encoding="utf-8-sig")
    skipped_parse.to_csv(out_dir / "skipped_unrecognized_files.csv", index=False, encoding="utf-8-sig")
    skipped_loading.to_csv(out_dir / "skipped_after_loading.csv", index=False, encoding="utf-8-sig")
    if feature_df.empty:
        raise SystemExit("No valid features generated. Please check filename format and data folder.")
    class_counts = feature_df["group"].value_counts().reindex(GROUPS).fillna(0).astype(int)
    if (class_counts == 0).any():
        raise SystemExit(f"Missing one or more groups. Class counts: {class_counts.to_dict()}")
    y_pred, proba, marker_records, fold_rankings, cv_method = nested_cv_predict(feature_df, top_n=args.top_n)
    y_true = feature_df["group"].astype(str).values
    pred_df = feature_df[["patient", "group", "layer"]].copy()
    pred_df["predicted_group"] = y_pred
    for i, g in enumerate(GROUPS):
        pred_df[f"prob_{g}"] = proba[:, i]
    pred_df.to_csv(out_dir / "nested_cv_predictions.csv", index=False, encoding="utf-8-sig")
    marker_records.to_csv(out_dir / "nested_cv_marker_records.csv", index=False, encoding="utf-8-sig")
    fold_rankings.to_csv(out_dir / "nested_cv_fold_marker_rankings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(classification_report(y_true, y_pred, labels=GROUPS, output_dict=True, zero_division=0)).T.to_csv(
        out_dir / "nested_cv_classification_report.csv", encoding="utf-8-sig"
    )
    acc, bal = plot_confusion(y_true, y_pred, out_dir, cv_method)
    auc_df = plot_roc(y_true, proba, out_dir, cv_method)
    plot_probability_heatmap(feature_df, y_pred, proba, out_dir)
    plot_marker_selection_frequency(marker_records, out_dir)
    final_model, final_selected, final_ranking = fit_final_model(feature_df, top_n=args.top_n)
    joblib.dump(final_model, out_dir / "final_2d_HML_model.joblib")
    pd.DataFrame({"marker": final_selected}).to_csv(out_dir / "final_selected_markers.csv", index=False, encoding="utf-8-sig")
    final_ranking.to_csv(out_dir / "final_model_marker_ranking.csv", index=False, encoding="utf-8-sig")
    plot_final_marker_weights(final_model, final_selected, out_dir)
    plot_selected_marker_pca(feature_df, final_selected, out_dir)
    export_formula(final_model, final_selected, out_dir)
    metadata = {
        "analysis_name": "paper_model_01_2d_HML_classifier",
        "data_dir": str(data_dir),
        "groups": GROUPS,
        "target_layer": args.target_layer,
        "feature_extraction": "10%-90% percentile trimming + log1p + IQR",
        "top_n": args.top_n,
        "marker_selection": "within each CV training fold",
        "model": "class-balanced multinomial logistic regression",
        "cv_method": cv_method,
        "n_samples": int(len(feature_df)),
        "class_counts": class_counts.to_dict(),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "final_selected_markers": final_selected,
    }
    with open(out_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    lines = []
    lines.append("Paper model 01: Final 2D H/M/L differentiation classifier")
    lines.append("")
    lines.append(f"Data folder: {data_dir}")
    lines.append(f"Output folder: {out_dir}")
    lines.append("Feature extraction: 10%–90% percentile trimming + log1p + IQR")
    lines.append(f"Top markers selected within each CV training fold: {args.top_n}")
    lines.append("Model: class-balanced multinomial logistic regression")
    lines.append(f"CV method: {cv_method}")
    lines.append("")
    lines.append(f"Class counts: {class_counts.to_dict()}")
    lines.append(f"Accuracy: {acc:.3f}")
    lines.append(f"Balanced accuracy: {bal:.3f}")
    if not auc_df.empty:
        lines.append("One-vs-rest AUC:")
        for _, r in auc_df.iterrows():
            lines.append(f"  {r['class']}: {r['one_vs_rest_auc']:.3f}")
    lines.append("")
    lines.append("Final selected markers:")
    lines.append(", ".join(final_selected))
    lines.append("")
    lines.append("Main figures:")
    lines.append("  Fig6_2D_HML_confusion_matrix.png")
    lines.append("  Fig6_2D_HML_one_vs_rest_ROC.png")
    lines.append("  Fig6_2D_HML_probability_heatmap.png")
    lines.append("  Fig6_2D_HML_marker_selection_frequency.png")
    lines.append("  Fig6_2D_HML_final_marker_weight_ranking.png")
    lines.append("  Fig6_2D_HML_selected_marker_PCA.png")
    lines.append("")
    lines.append("Manuscript interpretation:")
    lines.append("  This is an exploratory internally validated 2D differentiation classifier.")
    lines.append("  Do not describe this as an externally validated clinical diagnostic model.")
    (out_dir / "00_2d_HML_model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Done.")
    print(f"Results saved to: {out_dir}")
    print(f"CV method: {cv_method}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Balanced accuracy: {bal:.3f}")


if __name__ == "__main__":
    main()
