#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REAL-DATA ONLY
FigE. Core Raman program fingerprints: publication-style radar plot.

This script uses the actual Fig01 source CSV files from the LUAD 3D Raman
analysis and produces a clean line-based radar plot, not a synthetic drawing.

Required real-data input files (either directly in --input-dir or in a child folder):
  - source_embedding_coordinates.csv
  - source_true_pixel_robust_z_features.csv

Scientific interpretation
-------------------------
- Each point is a group-wise summary of real pixel-level Raman module scores.
- Default group summary: median across pixels within each pathology state.
- For radar display only, each molecular module is min-max scaled across the
  available pathology states to 0-1. Therefore the radar shows relative
  between-state fingerprint patterns, not absolute concentration differences.
- The raw group/module score table, normalized score table, group counts and
  marker-to-module mapping are exported for traceability.
- This is a descriptive fingerprint visualization; it is not patient-level
  inferential evidence by itself.

Main outputs
------------
FigE_core_Raman_program_fingerprint_radar_realdata.png/pdf/svg
source_marker_module_mapping.csv
source_group_pixel_counts.csv
source_module_scores_raw.csv
source_module_scores_radar_normalized.csv

Recommended run
---------------
py .\scripts\3d\figE_core_raman_program_fingerprint_radar_realdata_v2.py ^
  --input-dir ".\3dresults\fig01_3d_pca_umap_clean_twopanel_v7_2000pixels" ^
  --out-dir ".\3dresults\figE_radar_realdata"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

# ---------- pathology groups ----------
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
    "ca": "CA", "cancer-adjacent": "CA", "cancer adjacent": "CA", "canceradjacent": "CA", "adjacent": "CA", "normal": "CA", "pneumonia": "CA",
    "lep": "LEP", "lepidic": "LEP", "tiebi": "LEP",
    "acn": "ACN", "acinar": "ACN", "xianpao": "ACN",
    "pap": "PAP", "papillary": "PAP",
    "mp": "MP", "micropapillary": "MP", "micropap": "MP",
    "cgp": "CGP", "complex glands": "CGP", "complex_glands": "CGP", "complexglands": "CGP", "fzxt": "CGP",
    "sol": "SOL", "solid": "SOL", "soild": "SOL",
}

# Soft but visible line colors, aligned with the user's cream/mint/pink violin palette.
GROUP_COLORS = {
    "CA": "#8EA6B6",   # blue grey
    "LEP": "#78B7DE",  # light blue
    "ACN": "#66BFA6",  # mint teal
    "PAP": "#86C785",  # light green
    "MP": "#E0BB64",   # butter gold
    "CGP": "#E6A26B",  # peach
    "SOL": "#D97F8A",  # dusty pink
}

# Six curated biological modules are more defensible for a core program figure.
# "Other" can be added only if explicitly requested.
MODULE_ORDER_BASE = [
    "Central metabolism",
    "Lipid metabolism",
    "ECM/stromal remodeling",
    "Immune/protein",
    "Stress/glycoxidation",
    "Amino acid/nitrogen",
]
MODULE_LABELS = {
    "Central metabolism": "Central\nmetabolism",
    "Lipid metabolism": "Lipid\nmetabolism",
    "ECM/stromal remodeling": "ECM / stromal\nremodeling",
    "Immune/protein": "Immune /\nprotein",
    "Stress/glycoxidation": "Stress /\nglycoxidation",
    "Amino acid/nitrogen": "Amino acid /\nnitrogen",
    "Other": "Other",
}


def normalize_text(x: str) -> str:
    return " ".join(str(x).replace("_", " ").replace("-", "-").split()).strip().lower()


def canonical_group(x: str) -> Optional[str]:
    return GROUP_ALIASES.get(normalize_text(x), None)


def infer_module(marker: str) -> str:
    """Rule-based automatic marker assignment. A manual mapping file can override this."""
    key = normalize_text(marker)

    rules: List[Tuple[str, List[str]]] = [
        ("Central metabolism", [
            "lactate", "fumarate", "succinate", "citrate", "malate", "pyruvate", "phosphoglycerate",
            "3-pg", "3pg", "fructose", "glucose", "glycolysis", "tca", "oxaloacet", "acetyl-coa",
            "acetylcoa", "coenzyme a", "pentose", "phosphoenolpyruvate", "atp", "adp", "amp",
        ]),
        ("Lipid metabolism", [
            "cholesterol", "lipid", "sterol", "sphingos", "ceramide", "fatty", "phospholipid",
            "phosphatid", "triglycer", "diglycer", "choline", "oleic", "linoleic", "palmit",
            "stear", "arachidon",
        ]),
        ("ECM/stromal remodeling", [
            "versican", "vcan", "syndecan", "collagen", "laminin", "fibronectin", "tenascin",
            "elastin", "ecm", "integrin", "acta2", "cathepsin", "mmp", "remodel", "stromal",
            "strom", "vitronectin", "hyaluron",
        ]),
        ("Immune/protein", [
            "pdl1", "pd-l1", "b7-h3", "b7h3", "sting", "cd68", "cd8", "cd4", "cd31",
            "interleukin", "ifn", "tnf", "efna", "mertk", "gas6", "ttf1", "ttf-1", "immune",
            "cytokine", "chemokine",
        ]),
        ("Stress/glycoxidation", [
            "oxidative", "redox", "glutathione", "lactoylglutathione", "disodium fumarate", "stress",
            "hypoxia", "ros", "antioxid", "glycox", "glycat", "pentosidine", "4hne", "mda",
        ]),
        ("Amino acid/nitrogen", [
            "amino", "glutamine", "glutamate", "aspartate", "alanine", "glycine", "serine",
            "tyrosine", "tryptophan", "ornithine", "arginine", "nitrogen", "histidine", "leucine",
            "isoleucine", "valine", "methionine", "phenylalanine", "lysine", "proline", "taurine",
            "urea", "creatine",
        ]),
    ]
    for module, tokens in rules:
        if any(token in key for token in tokens):
            return module
    return "Other"


def find_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct
    hits = sorted(root.rglob(filename), key=lambda p: (len(p.parts), str(p)))
    if not hits:
        raise FileNotFoundError(
            f"Could not find {filename} under {root}. Use the Fig01 result folder containing the source CSV files."
        )
    return hits[0]


def detect_group_column(coords: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in coords.columns:
            raise KeyError(f"--group-col '{requested}' not found. Available columns: {list(coords.columns)}")
        return requested
    for col in coords.columns:
        if str(col).lower() in {"group", "pathology", "state", "label", "class"}:
            return str(col)
    raise KeyError("Could not detect a group column. Expected one of: group, pathology, state, label, class.")


def load_real_sources(input_dir: Path, group_col_arg: str) -> Tuple[pd.Series, pd.DataFrame, Path, Path]:
    coord_path = find_file(input_dir, "source_embedding_coordinates.csv")
    feat_path = find_file(input_dir, "source_true_pixel_robust_z_features.csv")

    coords = pd.read_csv(coord_path)
    features_raw = pd.read_csv(feat_path)
    if len(coords) != len(features_raw):
        raise RuntimeError(
            f"Row mismatch: source_embedding_coordinates.csv has {len(coords)} rows but source_true_pixel_robust_z_features.csv has {len(features_raw)} rows."
        )

    group_col = detect_group_column(coords, group_col_arg)
    groups = coords[group_col].astype(str).map(canonical_group)
    keep = groups.notna().values
    if int(keep.sum()) == 0:
        unique = sorted(coords[group_col].dropna().astype(str).unique().tolist())
        raise RuntimeError(
            "No pathology groups could be mapped to CA/LEP/ACN/PAP/MP/CGP/SOL. "
            f"Found: {unique}"
        )

    groups = groups.loc[keep].reset_index(drop=True)
    features_raw = features_raw.loc[keep].reset_index(drop=True)

    numeric = features_raw.apply(pd.to_numeric, errors="coerce")
    valid_cols = [
        c for c in numeric.columns
        if np.isfinite(numeric[c].values).sum() >= 10
        and np.nanstd(numeric[c].values.astype(float)) > 1e-12
    ]
    numeric = numeric[valid_cols].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    if numeric.shape[1] < 5:
        raise RuntimeError("Too few valid numeric Raman-marker columns were detected.")

    return groups, numeric, coord_path, feat_path


def read_manual_mapping(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manual mapping file not found: {p}")
    df = pd.read_csv(p)
    if df.shape[1] < 2:
        raise ValueError("Manual mapping CSV must contain at least two columns: marker,module")
    df = df.iloc[:, :2].copy()
    df.columns = ["marker", "module"]
    return dict(zip(df["marker"].astype(str), df["module"].astype(str)))


def build_marker_module_map(markers: Iterable[str], manual_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for marker in markers:
        marker_str = str(marker)
        source = "automatic"
        module = infer_module(marker_str)
        if marker_str in manual_map:
            module = manual_map[marker_str]
            source = "manual"
        rows.append({"marker": marker_str, "module": module, "assignment_source": source})
    return pd.DataFrame(rows)


def compute_scores(
    groups: pd.Series,
    features: pd.DataFrame,
    marker_map: pd.DataFrame,
    modules: List[str],
    summary_method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Per-pixel module score = mean of the real robust-z marker values within that module.
    pixel_scores = pd.DataFrame(index=features.index)
    marker_counts = []
    for module in modules:
        members = marker_map.loc[marker_map["module"] == module, "marker"].tolist()
        members = [m for m in members if m in features.columns]
        marker_counts.append({"module": module, "n_markers": len(members), "markers": " | ".join(members)})
        if not members:
            pixel_scores[module] = np.nan
        else:
            pixel_scores[module] = features[members].mean(axis=1)

    raw = pd.DataFrame(index=GROUP_ORDER, columns=modules, dtype=float)
    group_counts = []
    for grp in GROUP_ORDER:
        idx = groups.eq(grp).values
        n_pixels = int(idx.sum())
        group_counts.append({"group": grp, "full_name": GROUP_FULL[grp], "n_pixels": n_pixels})
        for module in modules:
            vals = pixel_scores.loc[idx, module].dropna().values.astype(float)
            if len(vals) == 0:
                raw.loc[grp, module] = np.nan
            elif summary_method == "median":
                raw.loc[grp, module] = float(np.nanmedian(vals))
            else:
                raw.loc[grp, module] = float(np.nanmean(vals))

    return raw, pd.DataFrame(marker_counts), pd.DataFrame(group_counts)


def module_minmax_scale(raw_scores: pd.DataFrame) -> pd.DataFrame:
    """Scale each module across pathology groups to [0,1] for radar display."""
    out = raw_scores.copy().astype(float)
    for module in out.columns:
        x = out[module].values.astype(float)
        finite = np.isfinite(x)
        if finite.sum() < 2:
            out[module] = 0.5
            continue
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if hi - lo < 1e-12:
            out[module] = 0.5
        else:
            out[module] = (x - lo) / (hi - lo)
    return out.fillna(0.5).clip(0.0, 1.0)


def plot_radar(
    scaled_scores: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    out_svg: Path,
    title: str,
    subtitle: str,
    panel_letter: str,
) -> None:
    modules = list(scaled_scores.columns)
    labels = [MODULE_LABELS.get(m, m) for m in modules]
    n_axes = len(modules)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    # Landscape aspect ratio makes the legend and labels roomy.
    fig = plt.figure(figsize=(10.6, 9.4), facecolor="white")
    ax = fig.add_subplot(111, polar=True)
    ax.set_position([0.13, 0.205, 0.74, 0.615])
    ax.set_facecolor("white")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=10.5, color="#646464")
    ax.set_rlabel_position(15)
    ax.yaxis.grid(True, linestyle=(0, (3, 3)), color="#C9CED3", linewidth=0.85)
    ax.xaxis.grid(True, linestyle="-", color="#D8DCE0", linewidth=0.85)
    ax.spines["polar"].set_color("#B9C0C7")
    ax.spines["polar"].set_linewidth(1.0)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12.8, color="#202020", fontweight="normal")
    ax.tick_params(axis="x", pad=18)

    handles = []
    for grp in GROUP_ORDER:
        if grp not in scaled_scores.index:
            continue
        values = scaled_scores.loc[grp, modules].astype(float).tolist()
        values_closed = values + values[:1]
        color = GROUP_COLORS[grp]
        line, = ax.plot(
            angles_closed,
            values_closed,
            color=color,
            linewidth=2.05,
            marker="o",
            markersize=7.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.65,
            alpha=0.98,
            label=f"{grp} ({GROUP_FULL[grp]})",
            zorder=4,
        )
        handles.append(line)

    # Separate title block to avoid overlap with the top axis label.
    fig.text(0.035, 0.968, panel_letter, fontsize=30, fontweight="bold", ha="left", va="top")
    fig.text(0.500, 0.968, title, fontsize=22, fontweight="bold", ha="center", va="top", color="#111111")
    fig.text(0.500, 0.926, subtitle, fontsize=11.2, ha="center", va="top", color="#636363")

    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.030),
        ncol=4,
        frameon=False,
        fontsize=10.4,
        handlelength=2.6,
        handletextpad=0.55,
        columnspacing=1.5,
        labelspacing=0.95,
    )

    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-data publication-style LUAD Raman module radar plot.")
    parser.add_argument("--input-dir", required=True, help="Fig01 output folder containing the two source CSV files (or a parent folder searched recursively).")
    parser.add_argument("--out-dir", required=True, help="Output folder")
    parser.add_argument("--group-col", default="auto", help="Group column in coordinate CSV. Default: auto")
    parser.add_argument("--summary", choices=["median", "mean"], default="median", help="Group summary over real pixel-level module scores. Default: median")
    parser.add_argument("--include-other", action="store_true", help="Add the non-biological 'Other' category to the radar. Default: off")
    parser.add_argument("--manual-module-map", default=None, help="Optional CSV with marker,module columns to override automatic assignment")
    parser.add_argument("--title", default="Core Raman program fingerprints")
    parser.add_argument("--subtitle", default="Relative module activity across LUAD growth-pattern states")
    parser.add_argument("--panel-letter", default="E")
    parser.add_argument("--out-name", default="FigE_core_Raman_program_fingerprint_radar_realdata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups, features, coord_path, feature_path = load_real_sources(input_dir, args.group_col)
    manual_map = read_manual_mapping(args.manual_module_map)
    marker_map = build_marker_module_map(features.columns.tolist(), manual_map)

    modules = MODULE_ORDER_BASE.copy()
    if args.include_other:
        modules.append("Other")

    raw_scores, marker_counts, group_counts = compute_scores(
        groups=groups,
        features=features,
        marker_map=marker_map,
        modules=modules,
        summary_method=args.summary,
    )
    scaled_scores = module_minmax_scale(raw_scores)

    # Traceability exports.
    marker_map.to_csv(out_dir / "source_marker_module_mapping.csv", index=False)
    marker_counts.to_csv(out_dir / "source_module_marker_counts.csv", index=False)
    group_counts.to_csv(out_dir / "source_group_pixel_counts.csv", index=False)
    raw_scores.to_csv(out_dir / "source_module_scores_raw.csv", float_format="%.8f")
    scaled_scores.to_csv(out_dir / "source_module_scores_radar_normalized.csv", float_format="%.8f")
    pd.DataFrame({
        "input_coordinate_csv": [str(coord_path)],
        "input_feature_csv": [str(feature_path)],
        "group_summary": [args.summary],
        "radar_normalization": ["Per-module min-max scaling across CA, LEP, ACN, PAP, MP, CGP, SOL"],
        "include_other": [args.include_other],
    }).to_csv(out_dir / "analysis_provenance.csv", index=False)

    out_png = out_dir / f"{args.out_name}.png"
    out_pdf = out_dir / f"{args.out_name}.pdf"
    out_svg = out_dir / f"{args.out_name}.svg"
    plot_radar(scaled_scores, out_png, out_pdf, out_svg, args.title, args.subtitle, args.panel_letter)

    print("Done.")
    print(f"Coordinate source: {coord_path}")
    print(f"Feature source: {feature_path}")
    print(f"Groups retained: {', '.join([g for g in GROUP_ORDER if (groups == g).any()])}")
    print(f"Output PNG: {out_png}")
    print(f"Output PDF: {out_pdf}")
    print(f"Output SVG: {out_svg}")
    print(f"Raw source table: {out_dir / 'source_module_scores_raw.csv'}")
    print(f"Normalized source table: {out_dir / 'source_module_scores_radar_normalized.csv'}")


if __name__ == "__main__":
    main()
