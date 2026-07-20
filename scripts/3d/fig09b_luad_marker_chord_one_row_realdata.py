#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REAL-DATA ONLY
Fig09B: one-row LUAD marker-level chord networks.

Purpose
-------
Build one horizontal row of chord diagrams for the 7 LUAD pathology states
using real 3D Raman pixel-level marker features.

Required input files (searched recursively under --input-dir)
--------------------------------------------------------------
1) source_embedding_coordinates.csv
2) source_true_pixel_robust_z_features.csv

Method
------
- Load pixel-level marker matrix and pathology labels.
- Classify markers into six biological modules using keyword rules.
- Data-driven marker selection: rank markers by pathology dynamic range and pick
  the top N markers per module (default 2 per module) for readability.
- Within each pathology state, compute pairwise Spearman correlations between the
  selected markers.
- Keep edges passing |rho| >= threshold and BH-FDR <= alpha.
- Optionally keep only the strongest fraction of significant edges.
- Draw one-row chord map (7 panels) with abbreviation-based labels + shared legend.

Outputs
-------
- Main figure PNG/PDF/SVG
- source_selected_markers.csv
- source_marker_correlations_by_group.csv
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path as FilePath
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import PathPatch, Wedge, Rectangle
from matplotlib.path import Path as MplPath
from scipy.stats import spearmanr

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

GROUP_ORDER = ["CA", "LEP", "ACN", "PAP", "MP", "CGP", "SOL"]
GROUP_FULL = {
    "CA": "Cancer-adjacent",
    "LEP": "Lepidic",
    "ACN": "Acinar",
    "PAP": "Papillary",
    "MP": "Micropapillary",
    "CGP": "Complex glands",
    "SOL": "Solid",
}
GROUP_ALIASES = {
    "ca": "CA", "cancer-adjacent": "CA", "cancer adjacent": "CA", "adjacent": "CA", "normal": "CA",
    "lep": "LEP", "lepidic": "LEP", "tiebi": "LEP",
    "acn": "ACN", "acinar": "ACN", "xianpao": "ACN",
    "pap": "PAP", "papillary": "PAP",
    "mp": "MP", "micropapillary": "MP", "micropap": "MP",
    "cgp": "CGP", "complex glands": "CGP", "complex_glands": "CGP", "fzxt": "CGP",
    "sol": "SOL", "solid": "SOL", "soild": "SOL",
}

MODULE_ORDER = [
    "Central metabolism",
    "Lipid metabolism",
    "ECM / stromal remodeling",
    "Immune / protein",
    "Stress / glycoxidation",
    "Amino acid / nitrogen",
]
MODULE_COLORS = {
    "Central metabolism": "#E64B35",
    "Lipid metabolism": "#4DBBD5",
    "ECM / stromal remodeling": "#00A087",
    "Immune / protein": "#3C5488",
    "Stress / glycoxidation": "#F39B7F",
    "Amino acid / nitrogen": "#91D1C2",
}
POS_EDGE = "#D96B75"
NEG_EDGE = "#5AA9A7"

MODULE_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Central metabolism", [
        "lactate", "fumarate", "succinate", "citrate", "malate", "pyruvate", "phosphoglycerate",
        "3-pg", "3pg", "phosphoenolpyru", "fructose", "glucose", "pentose", "oxaloacet", "acetyl-coa",
        "acetylcoa", "coenzyme a", "glycolysis", "tca", "ps-sigma"
    ]),
    ("Lipid metabolism", [
        "lipid", "cholesterol", "sterol", "sphingos", "ceramide", "fatty", "phospholipid", "pc", "cholesteryl",
        "cholesterol ester", "monounsaturated"
    ]),
    ("ECM / stromal remodeling", [
        "versican", "vcan", "syndecan", "collagen", "laminin", "vitronectin", "tenascin", "elastin", "ecm",
        "strom", "cathepsin", "acta2", "fibronectin", "remodel"
    ]),
    ("Immune / protein", [
        "pdl1", "pd-l1", "b7-h3", "b7h3", "sting", "cd68", "ttf1", "hyaluronan", "immune", "protein",
        "efna", "cd31", "cd8", "cd4", "mertk", "gas6"
    ]),
    ("Stress / glycoxidation", [
        "oxidative", "stress", "glutathione", "lactoylglutathione", "redox", "disodium fumarate", "hypoxia", "ros",
        "glycox", "oxidation"
    ]),
    ("Amino acid / nitrogen", [
        "amino", "glutamine", "glutamate", "aspartate", "alanine", "glycine", "serine", "tyrosine", "tryptophan",
        "ornithine", "arginine", "nitrogen", "sphingosine"
    ]),
]


def clean_text(x: str) -> str:
    return " ".join(str(x).replace("_", " ").split()).strip()


def canonical_group(x: str) -> str | None:
    return GROUP_ALIASES.get(clean_text(x).lower(), None)


def classify_module(marker: str) -> str | None:
    m = clean_text(marker).lower()
    for module, kws in MODULE_KEYWORDS:
        if any(k in m for k in kws):
            return module
    return None


def abbreviate(text: str, max_len: int = 10) -> str:
    s = clean_text(text)
    specials = {
        "B7 H3": "B7H3",
        "PDL1": "PDL1",
        "PD L1": "PDL1",
        "STING": "STING",
        "CD68": "CD68",
        "ACTA2": "ACTA2",
        "TTF1": "TTF1",
    }
    if s.upper() in specials:
        return specials[s.upper()]
    if re.fullmatch(r"[A-Za-z0-9\-]+", s) and len(s) <= max_len:
        return s.upper()
    tokens = re.split(r"[^A-Za-z0-9]+", s)
    tokens = [t for t in tokens if t]
    if len(tokens) >= 2:
        ab = "".join(t[0].upper() for t in tokens[:4])
        if 2 <= len(ab) <= max_len:
            return ab
    return s.replace(" ", "")[:max_len].upper()


def blend(c1, c2="#FFFFFF", t=0.5):
    a = np.array(to_rgb(c1), dtype=float)
    b = np.array(to_rgb(c2), dtype=float)
    return tuple((1 - t) * a + t * b)


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[valid] = tmp
    return out


def find_file(root: FilePath, filename: str) -> FilePath:
    direct = root / filename
    if direct.exists():
        return direct
    hits = list(root.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    hits = sorted(hits, key=lambda p: (len(p.parts), str(p)))
    return hits[0]


def load_sources(input_dir: FilePath) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coord_path = find_file(input_dir, "source_embedding_coordinates.csv")
    feat_path = find_file(input_dir, "source_true_pixel_robust_z_features.csv")
    coords = pd.read_csv(coord_path)
    feats_raw = pd.read_csv(feat_path)

    group_col = None
    for c in coords.columns:
        if c.lower() in {"group", "pathology", "state", "label", "class"}:
            group_col = c
            break
    if group_col is None:
        raise KeyError("source_embedding_coordinates.csv must contain a group/pathology column.")

    groups = coords[group_col].astype(str).map(canonical_group)
    keep = groups.notna().values
    coords = coords.loc[keep].reset_index(drop=True)
    feats_raw = feats_raw.loc[keep].reset_index(drop=True)
    coords["group_short"] = groups.loc[keep].values

    numeric = feats_raw.apply(pd.to_numeric, errors="coerce")
    valid_cols = [
        c for c in numeric.columns
        if np.isfinite(numeric[c].values).sum() >= 20 and np.nanstd(numeric[c].values.astype(float)) > 1e-10
    ]
    numeric = numeric[valid_cols].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    if numeric.shape[1] < 8:
        raise RuntimeError("Too few valid numeric marker columns found in source_true_pixel_robust_z_features.csv")
    return coords, numeric


def rank_and_select_markers(coords: pd.DataFrame, features: pd.DataFrame, top_markers_per_module: int) -> Tuple[List[str], pd.DataFrame]:
    groups = coords["group_short"].astype(str)
    group_means = features.groupby(groups, sort=False).mean().reindex(GROUP_ORDER)
    rows = []
    for col in features.columns:
        module = classify_module(col)
        if module is None:
            continue
        vec = pd.to_numeric(group_means[col], errors="coerce").values.astype(float)
        finite = np.isfinite(vec)
        if finite.sum() < 2:
            continue
        dynamic_range = float(np.nanmax(vec) - np.nanmin(vec))
        sd = float(np.nanstd(vec))
        rows.append({
            "marker": col,
            "module": module,
            "dynamic_range": dynamic_range,
            "group_sd": sd,
            "score": dynamic_range + sd,
            "abbr": abbreviate(col),
        })
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        raise RuntimeError("No markers could be assigned to the target biological modules.")
    ranking = ranking.sort_values(["module", "score", "dynamic_range", "group_sd"], ascending=[True, False, False, False])

    selected = []
    seen_abbr = set()
    for mod in MODULE_ORDER:
        sub = ranking.loc[ranking["module"] == mod].copy()
        n_take = 0
        for _, row in sub.iterrows():
            ab = row["abbr"]
            if ab in seen_abbr:
                continue
            selected.append(row.to_dict())
            seen_abbr.add(ab)
            n_take += 1
            if n_take >= top_markers_per_module:
                break
    selected_df = pd.DataFrame(selected)
    if selected_df.empty:
        raise RuntimeError("Marker selection returned zero markers.")
    # Order by module blocks, then score
    selected_df["module_order"] = selected_df["module"].map({m: i for i, m in enumerate(MODULE_ORDER)})
    selected_df = selected_df.sort_values(["module_order", "score"], ascending=[True, False]).reset_index(drop=True)
    return selected_df["marker"].tolist(), selected_df


def pairwise_corr(df: pd.DataFrame, cols: List[str], min_n: int = 25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(cols)
    rho = np.full((n, n), np.nan, dtype=float)
    pval = np.full((n, n), np.nan, dtype=float)
    nobs = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            a = pd.to_numeric(df[cols[i]], errors="coerce")
            b = pd.to_numeric(df[cols[j]], errors="coerce")
            mask = np.isfinite(a.values) & np.isfinite(b.values)
            n_ij = int(mask.sum())
            nobs[i, j] = nobs[j, i] = n_ij
            if n_ij < min_n:
                continue
            if i == j:
                rho[i, j] = 1.0
                pval[i, j] = 0.0
            else:
                if np.nanstd(a.values[mask]) < 1e-10 or np.nanstd(b.values[mask]) < 1e-10:
                    continue
                r, p = spearmanr(a.values[mask], b.values[mask])
                rho[i, j] = rho[j, i] = r
                pval[i, j] = pval[j, i] = p
    return rho, pval, nobs


def filter_edges(rho: np.ndarray, pval: np.ndarray, abs_rho_min: float, fdr_alpha: float, top_frac: float) -> np.ndarray:
    n = rho.shape[0]
    keep = np.zeros((n, n), dtype=float)
    iu = np.triu_indices(n, k=1)
    rv = rho[iu]
    pv = pval[iu]
    qv = bh_fdr(pv)
    sig = np.isfinite(rv) & np.isfinite(qv) & (np.abs(rv) >= abs_rho_min) & (qv <= fdr_alpha)
    if sig.sum() == 0:
        return keep
    abs_sig = np.abs(rv[sig])
    if top_frac < 1.0:
        k = max(1, int(math.ceil(len(abs_sig) * max(1e-6, float(top_frac)))))
        thr = np.sort(abs_sig)[-k]
        sig = sig & (np.abs(rv) >= thr)
    keep[iu] = np.where(sig, rv, 0.0)
    keep = keep + keep.T
    np.fill_diagonal(keep, 0.0)
    return keep


def polar_xy(theta_deg: float, r: float) -> Tuple[float, float]:
    th = np.deg2rad(theta_deg)
    return r * np.cos(th), r * np.sin(th)


def equal_sector_layout(n: int, start_angle: float = 90.0, pad_deg: float = 4.5) -> Dict[int, Tuple[float, float]]:
    total = 360.0 - n * pad_deg
    span = total / n
    cur = start_angle
    out = {}
    for i in range(n):
        out[i] = (cur, cur - span)
        cur = cur - span - pad_deg
    return out


def edge_subarcs(mat: np.ndarray, layout: Dict[int, Tuple[float, float]]) -> Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]]:
    n = mat.shape[0]
    partners = {}
    for i in range(n):
        neigh = [(j, abs(mat[i, j])) for j in range(n) if j != i and mat[i, j] != 0]
        neigh = sorted(neigh, key=lambda x: (-x[1], x[0]))
        partners[i] = neigh

    subarc = {}
    for i in range(n):
        a0, a1 = layout[i]
        span = abs(a0 - a1)
        total = sum(v for _, v in partners[i])
        cursor = a0
        alloc = {}
        if total <= 0:
            subarc[i] = alloc
            continue
        for j, v in partners[i]:
            width = span * (v / total)
            alloc[j] = (cursor, cursor - width)
            cursor -= width
        subarc[i] = alloc

    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] != 0 and (j in subarc[i]) and (i in subarc[j]):
                out[(i, j)] = (subarc[i][j], subarc[j][i])
    return out


def ribbon_patch(a_pair, b_pair, r=0.77, facecolor=(0.6, 0.6, 0.6), alpha=0.26):
    (a0, a1), (b0, b1) = a_pair, b_pair
    pa0 = np.array(polar_xy(a0, r))
    pa1 = np.array(polar_xy(a1, r))
    pb0 = np.array(polar_xy(b0, r))
    pb1 = np.array(polar_xy(b1, r))
    c = np.array([0.0, 0.0])
    verts = [pa0, c, pb0, pb1, c, pa1, pa0]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.CLOSEPOLY]
    return PathPatch(MplPath(verts, codes), facecolor=facecolor, edgecolor="none", alpha=alpha)


def draw_panel(ax, mat: np.ndarray, labels: List[str], marker_to_module: Dict[str, str], marker_to_abbr: Dict[str, str], show_labels: bool = False):
    n = len(labels)
    layout = equal_sector_layout(n, start_angle=90, pad_deg=4.0)
    sub = edge_subarcs(mat, layout)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)

    vals = np.abs(mat[np.triu_indices(n, k=1)])
    vmax = np.nanmax(vals) if np.any(vals > 0) else 1.0
    for (i, j), (a_pair, b_pair) in sub.items():
        value = mat[i, j]
        color = POS_EDGE if value > 0 else NEG_EDGE
        alpha = 0.12 + 0.28 * (abs(value) / vmax)
        ax.add_patch(ribbon_patch(a_pair, b_pair, r=0.76, facecolor=color, alpha=alpha))

    for i, lab in enumerate(labels):
        mod = marker_to_module[lab]
        a0, a1 = layout[i]
        ax.add_patch(Wedge((0, 0), 1.0, a1, a0, width=0.13, facecolor=MODULE_COLORS[mod], edgecolor="white", linewidth=0.8))
        if show_labels:
            mid = (a0 + a1) / 2.0
            x, y = polar_xy(mid, 1.11)
            ha = "left" if np.cos(np.deg2rad(mid)) >= 0 else "right"
            ax.text(x, y, marker_to_abbr[lab], ha=ha, va="center", fontsize=7.0, fontweight="bold")


def save_sources(selected_df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: FilePath):
    selected_df.to_csv(out_dir / "source_selected_markers.csv", index=False)
    corr_df.to_csv(out_dir / "source_marker_correlations_by_group.csv", index=False)


def build_figure(group_to_mat: Dict[str, np.ndarray], labels: List[str], selected_df: pd.DataFrame, out_dir: FilePath, out_name: str, show_labels: bool):
    marker_to_module = dict(zip(selected_df["marker"], selected_df["module"]))
    marker_to_abbr = dict(zip(selected_df["marker"], selected_df["abbr"]))

    fig = plt.figure(figsize=(21.0, 7.2), facecolor="white")
    fig.text(0.5, 0.975, "LUAD growth-pattern-specific Raman marker association networks", ha="center", va="top", fontsize=20, fontweight="bold")
    fig.text(0.5, 0.94,
             "One-row real-data chord map | Spearman correlations of selected pixel-level Raman markers within each pathology state",
             ha="center", va="top", fontsize=10.8, color="#444444")

    left, right, top, bottom = 0.03, 0.99, 0.82, 0.33
    panel_w = (right - left) / len(GROUP_ORDER)
    for idx, grp in enumerate(GROUP_ORDER):
        ax = fig.add_axes([left + idx * panel_w + 0.005, bottom, panel_w - 0.01, top - bottom])
        draw_panel(ax, group_to_mat.get(grp, np.zeros((len(labels), len(labels)))), labels, marker_to_module, marker_to_abbr, show_labels=show_labels)
        ax.set_title(grp, fontsize=12.5, fontweight="bold", pad=14)
        ax.text(0.5, -0.10, GROUP_FULL[grp], transform=ax.transAxes, ha="center", va="top", fontsize=8.8, color="#4A4A4A")

    # module legend
    leg1 = fig.add_axes([0.03, 0.16, 0.94, 0.12]); leg1.axis("off")
    leg1.text(0.0, 0.95, "Module colors", fontsize=11.2, fontweight="bold", va="top")
    x = 0.00
    for mod in MODULE_ORDER:
        leg1.add_patch(Rectangle((x, 0.42), 0.018, 0.18, facecolor=MODULE_COLORS[mod], edgecolor="none", transform=leg1.transAxes))
        leg1.text(x + 0.024, 0.51, mod, transform=leg1.transAxes, fontsize=8.8, va="center")
        x += 0.16

    # marker abbreviation legend
    leg2 = fig.add_axes([0.03, 0.04, 0.94, 0.10]); leg2.axis("off")
    leg2.text(0.0, 0.98, "Marker abbreviation legend", fontsize=11.2, fontweight="bold", va="top")
    ncols = 4
    rows = int(np.ceil(len(selected_df) / ncols))
    for i, row in selected_df.reset_index(drop=True).iterrows():
        c = i % ncols
        r = i // ncols
        x = 0.00 + c * 0.245
        y = 0.66 - r * 0.34
        mod = row["module"]
        leg2.add_patch(Rectangle((x, y - 0.07), 0.016, 0.12, facecolor=MODULE_COLORS[mod], edgecolor="none", transform=leg2.transAxes))
        leg2.text(x + 0.022, y, f"{row['abbr']} = {row['marker']}", transform=leg2.transAxes, fontsize=8.5, va="center")

    out_png = out_dir / out_name
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="One-row LUAD marker chord map from real 3D Raman data")
    parser.add_argument("--input-dir", type=str, required=True, help="Root directory searched recursively for the required source CSV files")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--out-name", type=str, default="Fig09B_LUAD_marker_chord_one_row.png")
    parser.add_argument("--top-markers-per-module", type=int, default=2, help="Number of data-driven markers selected per module")
    parser.add_argument("--corr-threshold", type=float, default=0.30, help="Absolute Spearman rho threshold")
    parser.add_argument("--fdr-alpha", type=float, default=0.05, help="BH-FDR cutoff")
    parser.add_argument("--top-frac", type=float, default=0.40, help="Keep strongest fraction of significant edges (0-1]")
    parser.add_argument("--min-n", type=int, default=25, help="Minimum number of pixels required per pairwise test")
    parser.add_argument("--show-labels", action="store_true", help="Show abbreviation labels directly around each ring")
    args = parser.parse_args()

    input_dir = FilePath(args.input_dir)
    out_dir = FilePath(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords, features = load_sources(input_dir)
    selected_markers, selected_df = rank_and_select_markers(coords, features, args.top_markers_per_module)
    marker_table = pd.DataFrame({"group": coords["group_short"].values})
    for m in selected_markers:
        marker_table[m] = features[m].values

    corr_rows = []
    group_to_mat: Dict[str, np.ndarray] = {}
    for grp in GROUP_ORDER:
        sub = marker_table.loc[marker_table["group"] == grp, ["group"] + selected_markers].copy()
        rho, pval, nobs = pairwise_corr(sub, selected_markers, min_n=args.min_n)
        mat = filter_edges(rho, pval, abs_rho_min=args.corr_threshold, fdr_alpha=args.fdr_alpha, top_frac=args.top_frac)
        group_to_mat[grp] = mat

        qvals = bh_fdr(pval[np.triu_indices(len(selected_markers), k=1)])
        k = 0
        for i in range(len(selected_markers)):
            for j in range(i + 1, len(selected_markers)):
                corr_rows.append({
                    "group": grp,
                    "marker_i": selected_markers[i],
                    "marker_j": selected_markers[j],
                    "abbr_i": selected_df.loc[selected_df["marker"] == selected_markers[i], "abbr"].iloc[0],
                    "abbr_j": selected_df.loc[selected_df["marker"] == selected_markers[j], "abbr"].iloc[0],
                    "rho": rho[i, j],
                    "p_value": pval[i, j],
                    "fdr_q": qvals[k] if k < len(qvals) else np.nan,
                    "n_pixels": nobs[i, j],
                    "kept_in_plot": mat[i, j] != 0,
                })
                k += 1

    corr_df = pd.DataFrame(corr_rows)
    save_sources(selected_df, corr_df, out_dir)
    build_figure(group_to_mat, selected_markers, selected_df, out_dir, args.out_name, show_labels=args.show_labels)


if __name__ == "__main__":
    main()
