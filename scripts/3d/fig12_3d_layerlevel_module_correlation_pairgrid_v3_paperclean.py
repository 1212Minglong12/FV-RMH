#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fig12 v3: publication-clean layer-/sample-level Raman molecular-program correlation pair-grid.

Purpose
-------
This version replaces pixel-level residual correlations with layer-/sample-level
module scores. It is designed to avoid two issues in the earlier figure:
1) pixels are not independent biological replicates;
2) regressing out global module intensity can artificially exaggerate negative
   correlations among molecular programs.

Workflow
--------
- Assign Raman markers to biological modules.
- Compute module scores per pixel as the mean of assigned marker features.
- Aggregate pixel scores within each layer/sample-like unit using the median
  (or mean if requested).
- Automatically remove modules that have no matched markers or no variance
  across units. This prevents an empty module such as Amino acid/nitrogen from
  appearing as an all-zero line / NA correlation.
- Compute layer-/sample-level Spearman rho and Benjamini-Hochberg FDR q-values.
- Draw a pair-grid modeled after the reference figure:
  left ridgelines, diagonal distributions, lower scatter plots, and upper
  correlation tiles with rho + FDR stars.

Expected input files inside --input-dir
---------------------------------------
source_embedding_coordinates.csv
source_true_pixel_robust_z_features.csv

Recommended run
---------------
cd "FV-RMH"
py ".\3dscripts\fig12_3d_layerlevel_module_correlation_pairgrid_v3_paperclean.py" ^
  --input-dir ".\3dresults\fig01_3d_pca_umap_clean_twopanel_v7_2000pixels" ^
  --out-dir ".\3dresults\fig12_layerlevel_module_correlation" ^
  --analysis-unit auto ^
  --group all

Dependencies
------------
py -m pip install numpy pandas matplotlib scipy
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde, spearmanr

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

GROUP_ORDER = [
    "Cancer-adjacent",
    "Lepidic",
    "Acinar",
    "Papillary",
    "Micropapillary",
    "Complex glands",
    "Solid",
]

GROUP_SHORT = {
    "Cancer-adjacent": "CA",
    "Lepidic": "LEP",
    "Acinar": "ACN",
    "Papillary": "PAP",
    "Micropapillary": "MP",
    "Complex glands": "CGP",
    "Solid": "SOL",
}

MODULE_ORDER = [
    "Central metabolism",
    "Lipid metabolism",
    "ECM/stromal remodeling",
    "Immune/protein",
    "Stress/glycoxidation",
    "Amino acid/nitrogen",
    "Other",
]

MODULE_SHORT = {
    "Central metabolism": "Central",
    "Lipid metabolism": "Lipid",
    "ECM/stromal remodeling": "ECM",
    "Immune/protein": "Immune",
    "Stress/glycoxidation": "Stress",
    "Amino acid/nitrogen": "Amino acid",
    "Other": "Other",
}

MODULE_LABELS = {
    "Central metabolism": "Central\nmetabolism",
    "Lipid metabolism": "Lipid\nmetabolism",
    "ECM/stromal remodeling": "ECM / stromal\nremodeling",
    "Immune/protein": "Immune /\nprotein",
    "Stress/glycoxidation": "Stress /\nglycoxidation",
    "Amino acid/nitrogen": "Amino acid /\nnitrogen",
    "Other": "Other",
}

MODULE_COLORS = {
    "Central metabolism": "#2B6F56",
    "Lipid metabolism": "#6E9684",
    "ECM/stromal remodeling": "#9FB2A2",
    "Immune/protein": "#D2B9AA",
    "Stress/glycoxidation": "#D97975",
    "Amino acid/nitrogen": "#B84B4B",
    "Other": "#777777",
}

SAMPLE_COLUMN_CANDIDATES = [
    "source_file", "file_name", "filename", "file", "image_id", "image",
    "slide_id", "slide", "layer_id", "layer", "z_index", "z",
    "sample_id", "sample", "sample_name", "specimen_id", "specimen",
    "tissue_id", "tissue", "case_id", "case", "patient_id", "patient", "subject_id", "subject",
]

CORR_CMAP = LinearSegmentedColormap.from_list(
    "green_white_red",
    ["#246B50", "#A9C1B2", "#F7F6F2", "#E9B5AE", "#C64642"],
    N=256,
)


def norm_text(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def module_of(marker: str) -> str:
    k = norm_text(marker)
    rules = {
        "Central metabolism": [
            "lactate", "pyruvate", "phosphoenolpyruvic", "3phosphoglycerate",
            "phosphoglycerate", "glucose", "glycol", "citrate", "fumarate",
            "malate", "succinate", "acetylcoa", "coa", "atp", "nad", "fadh",
            "tca", "oxaloacetate", "mitochond", "fructose", "glycerate",
        ],
        "Lipid metabolism": [
            "lipid", "cholesterol", "cholesteryl", "cholesterolester", "fattyacid",
            "palmit", "oleic", "linole", "triglyceride", "phospholipid", "sphingo",
            "ceramide", "monounsaturated", "saturatedlipid", "unsaturated", "sterol",
            "phosphatidyl", "ester",
        ],
        "ECM/stromal remodeling": [
            "collagen", "elastin", "fibronectin", "laminin", "vitronectin", "tenascin",
            "versican", "syndecan", "matrix", "ecm", "acta2", "actin", "vimentin",
            "cathepsin", "mmp", "integrin", "hyaluron", "periostin", "thrombospondin",
            "spp1", "remodel",
        ],
        "Immune/protein": [
            "pd1", "pdl1", "b7h3", "sting", "cd", "hla", "mhc", "immun",
            "cytokine", "interferon", "tnf", "il", "protein", "histone", "albumin",
            "keratin", "cytokeratin", "ck", "gas6", "mertk", "vegf",
        ],
        "Stress/glycoxidation": [
            "glycoxidation", "glycation", "oxid", "ros", "glutathione", "slactoylglutathione",
            "lactoyl", "gsh", "gssg", "carbonyl", "mda", "4hne", "stress", "hypoxia",
            "nrf2", "hif", "pentosidine", "cml", "redox",
        ],
        "Amino acid/nitrogen": [
            "alanine", "arginine", "asparagine", "aspartate", "cysteine", "glutamate",
            "glutamine", "glycine", "histidine", "isoleucine", "leucine", "lysine",
            "methionine", "phenylalanine", "proline", "serine", "threonine", "tryptophan",
            "tyrosine", "valine", "amino", "urea", "nitrogen", "creatine", "taurine",
        ],
    }
    for module, terms in rules.items():
        if any(term in k for term in terms):
            return module
    return "Other"


def load_inputs(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coord_path = input_dir / "source_embedding_coordinates.csv"
    feature_path = input_dir / "source_true_pixel_robust_z_features.csv"
    if not coord_path.exists():
        raise FileNotFoundError(f"Missing {coord_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing {feature_path}")

    coords = pd.read_csv(coord_path)
    features = pd.read_csv(feature_path)
    if len(coords) != len(features):
        raise RuntimeError(f"Embedding rows ({len(coords)}) and feature rows ({len(features)}) do not match.")
    if "group" not in coords.columns:
        raise KeyError("source_embedding_coordinates.csv must contain a 'group' column.")

    keep = coords["group"].astype(str).isin(GROUP_ORDER).values
    coords = coords.loc[keep].reset_index(drop=True)
    features = features.loc[keep].reset_index(drop=True)

    features = features.apply(pd.to_numeric, errors="coerce")
    valid_cols = [
        c for c in features.columns
        if np.isfinite(features[c].values.astype(float)).sum() >= 10
        and np.nanstd(features[c].values.astype(float)) > 1e-10
    ]
    features = features[valid_cols].replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    if features.shape[1] < 5:
        raise RuntimeError("Too few valid Raman marker features were available.")
    return coords, features


def clean_unit_text(series: pd.Series, keep_layer: bool) -> pd.Series:
    values = series.astype(str).fillna("Unknown")
    values = values.str.replace(r"\\", "/", regex=True)
    values = values.str.replace(r".*/", "", regex=True)
    values = values.str.replace(r"\.(csv|mat|txt|tif|tiff|png)$", "", regex=True, case=False)
    if not keep_layer:
        values = values.str.replace(r"([_-]layer[_-]?\d+.*)$", "", regex=True, case=False)
        values = values.str.replace(r"([_-]z[_-]?\d+.*)$", "", regex=True, case=False)
    values = values.replace("", np.nan).fillna("Unknown")
    return values.astype(str)


def infer_units(coords: pd.DataFrame, analysis_unit: str) -> Tuple[pd.Series, str, str]:
    """
    Return unit ids, unit source column, and a human-readable unit description.

    For layer-level use, source_file / file-like fields are preferred because their
    filenames usually encode a single imaging layer. For specimen use, layer suffixes
    are removed so the same tissue specimen can be combined across layers.
    """
    normalized = {norm_text(col): col for col in coords.columns}

    def match_column(names: Sequence[str]) -> Optional[str]:
        for name in names:
            key = norm_text(name)
            if key in normalized:
                return normalized[key]
        return None

    file_like = ["source_file", "file_name", "filename", "file", "image_id", "image", "slide_id", "slide"]
    sample_like = ["sample_id", "sample", "sample_name", "specimen_id", "specimen", "tissue_id", "tissue", "case_id", "case", "patient_id", "patient", "subject_id", "subject"]
    layer_like = ["layer_id", "layer", "z_index", "z"]

    if analysis_unit in {"auto", "layer"}:
        col = match_column(file_like)
        if col is not None:
            values = clean_unit_text(coords[col], keep_layer=True)
            if values.nunique() >= 4:
                return values, col, f"layer-like units inferred from '{col}'"
        col = match_column(layer_like)
        if col is not None:
            # Layer number alone is not unique across pathology groups, so prepend group.
            values = coords["group"].astype(str) + "_layer_" + coords[col].astype(str)
            if values.nunique() >= 4:
                return values.astype(str), col, f"group-specific layers inferred from '{col}'"
        if analysis_unit == "layer":
            raise RuntimeError("--analysis-unit layer was requested but no usable file/layer identifier was found in source_embedding_coordinates.csv.")

    if analysis_unit in {"auto", "specimen"}:
        col = match_column(sample_like)
        if col is not None:
            values = clean_unit_text(coords[col], keep_layer=False)
            if values.nunique() >= 3:
                return values, col, f"specimen-like units inferred from '{col}'"
        col = match_column(file_like)
        if col is not None:
            values = clean_unit_text(coords[col], keep_layer=False)
            if values.nunique() >= 3:
                return values, col, f"specimen-like units inferred from '{col}'"
        if analysis_unit == "specimen":
            raise RuntimeError("--analysis-unit specimen was requested but no usable specimen identifier was found.")

    raise RuntimeError(
        "No layer/sample identifier was detected. This figure should not fall back to pixel-level correlations. "
        "Please inspect source_embedding_coordinates.csv and add a source_file, layer, sample_id, or specimen_id column."
    )


def build_pixel_module_scores(features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assignment = pd.DataFrame({
        "marker": features.columns.astype(str),
        "module": [module_of(marker) for marker in features.columns.astype(str)],
    })

    scores = pd.DataFrame(index=features.index)
    for module in MODULE_ORDER:
        cols = assignment.loc[assignment["module"] == module, "marker"].tolist()
        scores[module] = features[cols].mean(axis=1) if cols else np.nan
    return assignment, scores


def aggregate_unit_scores(
    coords: pd.DataFrame,
    pixel_scores: pd.DataFrame,
    unit_ids: pd.Series,
    aggregation: str,
) -> pd.DataFrame:
    df = pixel_scores.copy()
    df["group"] = coords["group"].astype(str).values
    df["unit_id"] = unit_ids.astype(str).values

    score_cols = MODULE_ORDER
    if aggregation == "mean":
        grouped = df.groupby(["group", "unit_id"], observed=True)[score_cols].mean().reset_index()
    else:
        grouped = df.groupby(["group", "unit_id"], observed=True)[score_cols].median().reset_index()

    counts = df.groupby(["group", "unit_id"], observed=True).size().rename("n_pixels").reset_index()
    grouped = grouped.merge(counts, on=["group", "unit_id"], how="left")
    return grouped


def module_coverage(
    assignment: pd.DataFrame,
    unit_scores: pd.DataFrame,
    include_other: bool,
    min_marker_count: int,
) -> Tuple[List[str], pd.DataFrame]:
    rows = []
    active = []
    for module in MODULE_ORDER:
        marker_count = int((assignment["module"] == module).sum())
        vals = pd.to_numeric(unit_scores[module], errors="coerce").values.astype(float)
        finite = vals[np.isfinite(vals)]
        variance = float(np.nanvar(finite)) if len(finite) else np.nan
        is_active = marker_count >= min_marker_count and len(finite) >= 4 and np.isfinite(variance) and variance > 1e-12
        if module == "Other" and not include_other:
            is_active = False
        rows.append({
            "module": module,
            "marker_count": marker_count,
            "n_finite_unit_scores": int(len(finite)),
            "unit_score_variance": variance,
            "included": bool(is_active),
        })
        if is_active:
            active.append(module)
    coverage = pd.DataFrame(rows)
    if len(active) < 3:
        raise RuntimeError(
            "Fewer than three active molecular modules remained after marker-coverage and variance checks. "
            "Inspect source_marker_module_assignment.csv and adjust module keywords if needed."
        )
    return active, coverage


def robust_zscore_dataframe(data: pd.DataFrame, modules: List[str]) -> pd.DataFrame:
    out = data.copy()
    for module in modules:
        x = pd.to_numeric(out[module], errors="coerce").values.astype(float)
        med = np.nanmedian(x)
        q1, q3 = np.nanpercentile(x, [25, 75])
        iqr = max(q3 - q1, 1e-9)
        out[module] = (x - med) / iqr
    return out


def subset_group(table: pd.DataFrame, group_arg: str) -> Tuple[pd.DataFrame, str]:
    clean = str(group_arg).strip()
    if clean.lower() in {"all", "global", "allgroups"}:
        return table.copy(), "All pathology groups"
    full_map = {g.lower(): g for g in GROUP_ORDER}
    short_map = {short.lower(): full for full, short in GROUP_SHORT.items()}
    if clean.lower() in full_map:
        group = full_map[clean.lower()]
    elif clean.lower() in short_map:
        group = short_map[clean.lower()]
    else:
        raise ValueError("Unknown --group. Use all, CA, LEP, ACN, PAP, MP, CGP, SOL, or a full group name.")
    sub = table.loc[table["group"].astype(str) == group].copy()
    return sub, group


def bh_fdr_matrix(pval: pd.DataFrame, modules: List[str]) -> pd.DataFrame:
    q = pd.DataFrame(np.nan, index=modules, columns=modules, dtype=float)
    pairs = []
    for i, m1 in enumerate(modules):
        for j in range(i + 1, len(modules)):
            m2 = modules[j]
            p = pval.loc[m1, m2]
            if np.isfinite(p):
                pairs.append((m1, m2, float(p)))
    if not pairs:
        return q
    ordered = sorted(pairs, key=lambda x: x[2])
    m = len(ordered)
    adjusted = [min(1.0, p * m / (rank + 1)) for rank, (_, _, p) in enumerate(ordered)]
    # Enforce monotonicity from largest p backward.
    for k in range(m - 2, -1, -1):
        adjusted[k] = min(adjusted[k], adjusted[k + 1])
    for (m1, m2, _), qv in zip(ordered, adjusted):
        q.loc[m1, m2] = qv
        q.loc[m2, m1] = qv
    for module in modules:
        q.loc[module, module] = 0.0
    return q


def calculate_correlations(data: pd.DataFrame, modules: List[str]):
    corr = pd.DataFrame(np.eye(len(modules)), index=modules, columns=modules, dtype=float)
    pval = pd.DataFrame(np.zeros((len(modules), len(modules))), index=modules, columns=modules, dtype=float)
    nobs = pd.DataFrame(np.zeros((len(modules), len(modules))), index=modules, columns=modules, dtype=int)
    for i, m1 in enumerate(modules):
        for j, m2 in enumerate(modules):
            if i == j:
                nobs.loc[m1, m2] = int(np.isfinite(data[m1].values).sum())
                continue
            x = pd.to_numeric(data[m1], errors="coerce").values.astype(float)
            y = pd.to_numeric(data[m2], errors="coerce").values.astype(float)
            finite = np.isfinite(x) & np.isfinite(y)
            n = int(finite.sum())
            nobs.loc[m1, m2] = n
            if n < 5 or np.nanstd(x[finite]) <= 1e-12 or np.nanstd(y[finite]) <= 1e-12:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(x[finite], y[finite])
            corr.loc[m1, m2] = rho
            pval.loc[m1, m2] = p
    qval = bh_fdr_matrix(pval, modules)
    return corr, pval, qval, nobs


def stars_from_q(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.0001:
        return "****"
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def safe_kde(values: np.ndarray, xs: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    if len(values) < 6 or np.nanstd(values) <= 1e-12:
        return np.zeros_like(xs)
    try:
        return gaussian_kde(values)(xs)
    except Exception:
        return np.zeros_like(xs)


def draw_ridgelines(ax, data: pd.DataFrame, modules: List[str], xlim: Tuple[float, float]):
    n = len(modules)
    xs = np.linspace(xlim[0], xlim[1], 400)
    y_levels = np.arange(n)[::-1]
    for idx, module in enumerate(modules):
        vals = pd.to_numeric(data[module], errors="coerce").values.astype(float)
        dens = safe_kde(vals, xs)
        if np.nanmax(dens) > 0:
            dens = dens / np.nanmax(dens) * 0.62
        y0 = y_levels[idx]
        color = MODULE_COLORS[module]
        ax.fill_between(xs, y0, y0 + dens, color=color, alpha=0.25, linewidth=0)
        ax.plot(xs, y0 + dens, color=color, lw=1.10)
        finite = vals[np.isfinite(vals)]
        if len(finite):
            rng = np.random.default_rng(20260623 + idx)
            max_plot = min(350, len(finite))
            if len(finite) > max_plot:
                finite = rng.choice(finite, size=max_plot, replace=False)
            jitter = rng.normal(y0 + 0.025, 0.04, len(finite))
            ax.scatter(finite, jitter, s=8.5, color=color, alpha=0.66, linewidths=0, rasterized=True)
    ax.axvline(0, color="#B7B7B7", lw=0.7, zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.45, n - 0.18)
    ax.set_yticks(y_levels)
    ax.set_yticklabels([MODULE_SHORT[m] for m in modules], fontsize=9)
    ax.set_xlabel("Layer-level robust module score", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def draw_diagonal(ax, values: np.ndarray, xlim: Tuple[float, float], color: str):
    values = values[np.isfinite(values)]
    ax.hist(values, bins=min(18, max(8, len(values) // 2)), density=True, color="#F9F4F2", edgecolor=color, linewidth=0.50)
    xs = np.linspace(xlim[0], xlim[1], 250)
    dens = safe_kde(values, xs)
    if np.nanmax(dens) > 0:
        ax.plot(xs, dens, color=color, lw=1.25)
        ax.fill_between(xs, 0, dens, color=color, alpha=0.12)
    ax.set_xlim(xlim)


def draw_scatter(ax, x: np.ndarray, y: np.ndarray, xlim: Tuple[float, float]):
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    ax.scatter(x, y, s=16, color="#356B52", alpha=0.64, linewidths=0, rasterized=True)
    ax.axhline(0, color="#E2E2E2", lw=0.55, zorder=0)
    ax.axvline(0, color="#E2E2E2", lw=0.55, zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)


def draw_corr_cell(ax, rho: float, q: float):
    """Upper-triangle correlation tile: rho + FDR stars only (no n displayed)."""
    value = 0.0 if not np.isfinite(rho) else float(np.clip(rho, -1, 1))
    ax.set_facecolor(CORR_CMAP((value + 1.0) / 2.0))
    rho_label = "NA" if not np.isfinite(rho) else f"{rho:.3f}"
    ax.text(0.5, 0.60, rho_label, ha="center", va="center", fontsize=9.3, color="#202020")
    star = stars_from_q(q)
    if star:
        ax.text(0.5, 0.35, star, ha="center", va="center", fontsize=8.8, color="#202020", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

def style_matrix_axis(ax):
    ax.tick_params(length=0, labelsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#3C3C3C")
        sp.set_linewidth(0.65)


def make_pairgrid(
    data: pd.DataFrame,
    modules: List[str],
    corr: pd.DataFrame,
    qval: pd.DataFrame,
    nobs: pd.DataFrame,
    group_label: str,
    unit_description: str,
    out_dir: Path,
    dpi: int,
):
    """Publication-clean pair grid.

    Deliberately omits per-cell n values and repeated matrix row labels to prevent
    text from overlapping the ridge distributions. The unit count remains available
    in the exported methods summary and source tables.
    """
    all_values = data[modules].values.astype(float).ravel()
    all_values = all_values[np.isfinite(all_values)]
    xlim = tuple(np.nanpercentile(all_values, [0.5, 99.5]))
    span = max(xlim[1] - xlim[0], 1e-9)
    xlim = (xlim[0] - span * 0.10, xlim[1] + span * 0.10)

    n = len(modules)
    fig = plt.figure(figsize=(2.85 + n * 1.38 + 0.92, n * 1.18 + 1.72), facecolor="white")
    gs = fig.add_gridspec(
        nrows=n,
        ncols=n + 2,
        width_ratios=[2.85] + [1.12] * n + [0.30],
        wspace=0.075,
        hspace=0.075,
        left=0.055,
        right=0.957,
        top=0.885,
        bottom=0.105,
    )

    ridge_ax = fig.add_subplot(gs[:, 0])
    draw_ridgelines(ridge_ax, data, modules, xlim)
    ridge_ax.set_xlabel("Layer-level robust module score", fontsize=9.4, labelpad=5)

    for i, row_module in enumerate(modules):
        for j, col_module in enumerate(modules):
            ax = fig.add_subplot(gs[i, j + 1])
            x = pd.to_numeric(data[col_module], errors="coerce").values.astype(float)
            y = pd.to_numeric(data[row_module], errors="coerce").values.astype(float)
            if i == j:
                draw_diagonal(ax, x, xlim, MODULE_COLORS[row_module])
            elif i > j:
                draw_scatter(ax, x, y, xlim)
            else:
                draw_corr_cell(ax, corr.loc[row_module, col_module], qval.loc[row_module, col_module])
            style_matrix_axis(ax)
            if i == n - 1:
                ax.set_xlabel(MODULE_SHORT[col_module], fontsize=8.4, rotation=42, ha="right", labelpad=5)
            # Matrix-side row labels are intentionally omitted: the ridge panel already
            # provides a non-overlapping module label for every row.

    cax = fig.add_subplot(gs[:, -1])
    sm = plt.cm.ScalarMappable(cmap=CORR_CMAP, norm=Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Spearman rho", fontsize=8.8)
    cb.ax.tick_params(labelsize=7.2)
    cb.set_ticks(np.linspace(-1, 1, 5))

    fig.suptitle(
        "Layer-level Raman molecular-program co-variation",
        fontsize=15.2,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.948,
        f"{group_label} | Spearman correlations of layer-level aggregated module scores",
        ha="center",
        va="center",
        fontsize=8.6,
        color="#5B5B5B",
    )
    fig.text(
        0.055,
        0.036,
        "Upper triangle: Spearman rho with Benjamini–Hochberg FDR significance (* q<0.05, ** q<0.01, *** q<0.001, **** q<0.0001).",
        fontsize=7.0,
        color="#5B5B5B",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "Fig12_layerlevel_Raman_module_correlation_pairgrid_paperclean"
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.035, facecolor="white")
    plt.close(fig)

def make_heatmap(corr: pd.DataFrame, qval: pd.DataFrame, modules: List[str], out_dir: Path, dpi: int):
    n = len(modules)
    fig, ax = plt.subplots(figsize=(1.0 + n * 1.18, 0.95 + n * 1.05), facecolor="white")
    mat = corr.loc[modules, modules].values.astype(float)
    im = ax.imshow(mat, cmap=CORR_CMAP, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                text = "1.00"
            else:
                rho = corr.iloc[i, j]
                star = stars_from_q(qval.iloc[i, j])
                text = "NA" if not np.isfinite(rho) else f"{rho:.2f}{star}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.3, color="#222222")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([MODULE_SHORT[m] for m in modules], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([MODULE_SHORT[m] for m in modules], fontsize=9)
    ax.set_title("Layer-level Spearman correlation summary", fontsize=11, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.set_label("Spearman rho", fontsize=8.5)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    stem = "Fig12_layerlevel_module_correlation_heatmap_paperclean"
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.025, facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025, facecolor="white")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.025, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Build a layer-/sample-level Raman module Spearman correlation pair-grid without pixel residualization."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing source_embedding_coordinates.csv and source_true_pixel_robust_z_features.csv")
    parser.add_argument("--out-dir", required=True, help="Output folder")
    parser.add_argument("--group", default="all", help="all, CA, LEP, ACN, PAP, MP, CGP, SOL, or a full pathology group name")
    parser.add_argument("--analysis-unit", choices=["auto", "layer", "specimen"], default="auto", help="Aggregate pixels to layer-like or specimen-like units before correlation. Default: auto")
    parser.add_argument("--aggregation", choices=["median", "mean"], default="median", help="Aggregation of pixel module scores within each unit. Default: median")
    parser.add_argument("--include-other", action="store_true", help="Include broad Other module. Disabled by default.")
    parser.add_argument("--min-marker-count", type=int, default=1, help="Minimum assigned marker count required to plot a module. Default: 1")
    parser.add_argument("--within-group-center", action="store_true", help="Subtract each pathology group's median module score after aggregation. Use only as an optional sensitivity analysis, not the default figure.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG dpi. Default: 600")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords, features = load_inputs(input_dir)
    assignment, pixel_scores = build_pixel_module_scores(features)
    unit_ids, unit_source, unit_description = infer_units(coords, args.analysis_unit)
    unit_scores_raw = aggregate_unit_scores(coords, pixel_scores, unit_ids, args.aggregation)
    modules, coverage = module_coverage(assignment, unit_scores_raw, args.include_other, args.min_marker_count)

    # Spearman is invariant under scaling, but robust z scores make ridgelines/scatter axes comparable.
    unit_scores = robust_zscore_dataframe(unit_scores_raw, modules)

    if args.within_group_center:
        for module in modules:
            unit_scores[module] = unit_scores[module] - unit_scores.groupby("group", observed=True)[module].transform("median")
        score_label = "within-pathology centered robust module score"
    else:
        score_label = "layer-level robust module score"

    subset, group_label = subset_group(unit_scores, args.group)
    if len(subset) < 5:
        raise RuntimeError(f"Too few layer/sample units ({len(subset)}) after group filtering. Need at least 5.")

    corr, pval, qval, nobs = calculate_correlations(subset, modules)
    make_pairgrid(subset, modules, corr, qval, nobs, group_label, unit_description, out_dir, args.dpi)
    make_heatmap(corr, qval, modules, out_dir, args.dpi)

    assignment.to_csv(out_dir / "source_marker_module_assignment.csv", index=False)
    coverage.to_csv(out_dir / "source_module_marker_coverage_and_inclusion.csv", index=False)
    pixel_scores.to_csv(out_dir / "source_pixel_module_scores_raw.csv", index=False)
    unit_scores_raw.to_csv(out_dir / "source_layer_or_specimen_module_scores_raw.csv", index=False)
    subset.to_csv(out_dir / "source_analysis_unit_module_scores_robust_z.csv", index=False)
    corr.to_csv(out_dir / "source_spearman_correlation_matrix.csv")
    pval.to_csv(out_dir / "source_spearman_raw_pvalue_matrix.csv")
    qval.to_csv(out_dir / "source_spearman_fdr_qvalue_matrix.csv")
    nobs.to_csv(out_dir / "source_spearman_n_observations_matrix.csv")

    summary = [
        "Fig12 v3 publication-clean layer-/sample-level Raman molecular-program correlation summary",
        "====================================================================",
        f"Input folder: {input_dir}",
        f"Group subset: {group_label}",
        f"Analysis unit: {unit_description}",
        f"Unit ID source: {unit_source}",
        f"Unit aggregation: {args.aggregation}",
        f"Units used for correlation: {len(subset)}",
        f"Modules plotted: {', '.join(modules)}",
        f"Score scale used in pair-grid: {score_label}",
        "",
        "Method notes:",
        "- Correlations were calculated at the layer-/sample-like unit level, not pixel level.",
        "- Module scores were aggregated from pixel-level marker-module scores using the specified summary statistic.",
        "- No global module-intensity residualization was performed.",
        "- Spearman p values were adjusted using Benjamini-Hochberg FDR across unique module pairs.",
        "- Modules with no matched markers or near-zero variation were removed automatically.",
    ]
    (out_dir / "methods_summary_v2_layerlevel.txt").write_text("\n".join(summary), encoding="utf-8")

    print("Done.")
    print(f"Output folder: {out_dir}")
    print(f"Analysis unit: {unit_description}")
    print(f"Units used: {len(subset)}")
    print(f"Modules plotted: {', '.join(modules)}")
    print("Main figure: Fig12_layerlevel_Raman_module_correlation_pairgrid_paperclean.png")


if __name__ == "__main__":
    main()
