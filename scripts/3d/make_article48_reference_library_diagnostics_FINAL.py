#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
48-reference Raman library diagnostics for the FV-RMH LUAD manuscript.

This script is locked to the 48 components listed in the manuscript trend/supplementary table.
It generates:
  - 48x48 Pearson correlation matrix
  - ranked pairwise correlations + top 25 pairs
  - hierarchical clustering and cluster order
  - singular-value spectrum
  - matrix rank and condition numbers
  - consolidated Excel workbook
  - NPZ design matrix
  - ZIP package of all outputs

Expected folder layout:
    光谱/
    ├─ make_article48_reference_library_diagnostics_FINAL.py
    └─ Reference/
       ├─ xHoriba.mat
       ├─ Acetyl_CoA.mat
       ├─ ...
       └─ VEGF165A.mat

Run:
    py -u make_article48_reference_library_diagnostics_FINAL.py --refdir "Reference"

Outputs:
    48_reference_library_diagnostics/
    48_reference_library_diagnostics.zip
"""

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform

try:
    import xlsxwriter
except ImportError:
    print("[AUTO-DEPS] xlsxwriter not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xlsxwriter"])
    import xlsxwriter


# ---------------------------------------------------------------------
# FINAL 48 COMPONENTS: exact manuscript list -> actual .mat filenames
# ---------------------------------------------------------------------
ARTICLE48 = [
    ("Acetyl-CoA", "Acetyl_CoA.mat", "Metabolism"),
    ("Disodium D-3-phosphoglycerate (3-PG)", "Disodium_D_3_phosphoglycerate.mat", "Metabolism"),
    ("Lactate", "Lactate.mat", "Metabolism"),
    ("Disodium fumarate (fumarate)", "DisodiumFumarate.mat", "Metabolism"),
    ("D-Fructose-6-phosphate (F6P)", "D_Fructose_6_phosphate.mat", "Metabolism"),
    ("Phosphoenolpyruvic acid (PEP)", "PhosphoenolpyruvicAcid.mat", "Metabolism"),
    ("Alpha-ketoglutaric acid (α-KG)", "alpha_KetoglutaricAcid.mat", "Metabolism"),
    ("Oxaloacetic acid (OAA)", "OxaloaceticAcid.mat", "Metabolism"),

    ("S-lactoylglutathione", "S_Lactoylglutathione.mat", "Metabolic stress / redox"),
    ("Nε-(carboxymethyl)lysine (CML)", "CML.mat", "Protein modification"),
    ("Pentosidine", "Pentosidine.mat", "Protein modification"),

    ("Cholesterol", "Cholesterol.mat", "Lipids"),
    ("Cholesteryl ester", "L11CholesterolEster.mat", "Lipids"),
    ("Phosphatidylcholine (PC)", "L35PC.mat", "Lipids"),
    ("Phosphatidylserine (PS)", "L1Phosphatidylserine.mat", "Lipids"),
    ("Saturated lipid", "LipidSat.mat", "Lipids"),
    ("Monounsaturated lipid", "LipidUnsat.mat", "Lipids"),
    ("Sphingosine", "L41Sphinosine.mat", "Lipids"),

    ("Collagen I (COL1A1/Col-I)", "Collagen1.mat", "ECM"),
    ("Collagen III (COL3A1/Col-III)", "Collagen3.mat", "ECM"),
    ("Collagen XI (COL11A1/Col-XI)", "COL11a1.mat", "ECM"),
    ("Hyaluronan", "Hyaluronan.mat", "ECM"),
    ("Laminin", "Laminin.mat", "ECM"),
    ("Elastin", "Elastin.mat", "ECM"),
    ("Versican (VCAN)", "VersicanV0.mat", "ECM"),
    ("Tenascin-C (TNC)", "TenascinC.mat", "ECM"),
    ("Vitronectin (VTN)", "Vitronectin.mat", "ECM"),
    ("ACTA2 (α-SMA)", "ACTa2.mat", "Stroma"),

    ("SPP1 (Osteopontin)", "SPP1.mat", "Immune / stroma"),
    ("TTF1 (NKX2-1)", "TITF1.mat", "Lineage"),
    ("EFNA3", "EFNA3.mat", "Signalling"),
    ("CDKN2A (p16)", "CDN2A.mat", "Cell cycle"),
    ("CDKN1A (p21)", "CDKN1A.mat", "Cell cycle"),
    ("PRMT1", "PRMT.mat", "Epigenetic regulation"),
    ("CD68", "CD68.mat", "Immune"),
    ("CDH1 (E-cadherin)", "CDH1.mat", "Cell adhesion / EMT"),
    ("Vimentin (VIM)", "Vimentin.mat", "EMT / cytoskeleton"),
    ("Syndecan-1 (SDC1/CD138)", "Syndecan.mat", "Cell surface / ECM"),
    ("Cathepsin K (CTSK)", "CathepsinK.mat", "Protease / invasion"),
    ("Carboxyl ester lipase (CEL)", "CEL.mat", "Lipid metabolism"),
    ("CYP1A2", "CYP1A2.mat", "Drug metabolism"),
    ("CYP3A4", "CYP3A4.mat", "Drug metabolism"),
    ("B7-H3 (CD276)", "B7H3.mat", "Immune checkpoint"),
    ("PD-L1 (CD274)", "PDL1.mat", "Immune checkpoint"),
    ("STING (TMEM173)", "STING.mat", "Innate immunity"),
    ("CD31 (PECAM1)", "CD31.mat", "Vasculature"),
    ("CD98 (SLC3A2)", "CD98.mat", "Amino-acid transport"),
    ("VEGF165A (VEGFA isoform)", "VEGF165A.mat", "Angiogenesis"),
]


SHORT_LABELS = [
    "Acetyl-CoA", "3-PG", "Lactate", "Fumarate", "F6P", "PEP", "α-KG", "OAA",
    "S-LG", "CML", "Pentosidine",
    "Cholesterol", "CE", "PC", "PS", "Saturated lipid", "MUFA", "Sphingosine",
    "Collagen I", "Collagen III", "Collagen XI", "Hyaluronan", "Laminin", "Elastin",
    "Versican", "Tenascin-C", "Vitronectin", "ACTA2",
    "SPP1", "TTF1", "EFNA3", "CDKN2A", "CDKN1A", "PRMT1", "CD68", "CDH1",
    "Vimentin", "Syndecan-1", "CTSK", "CEL", "CYP1A2", "CYP3A4", "B7-H3",
    "PD-L1", "STING", "CD31", "CD98", "VEGF165A"
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate diagnostics for the final 48-reference FV-RMH Raman library."
    )
    p.add_argument("--refdir", default="Reference",
                   help='Reference .mat folder (default: "Reference")')
    p.add_argument("--outdir", default="48_reference_library_diagnostics",
                   help='Output folder (default: "48_reference_library_diagnostics")')
    p.add_argument("--axis-file", default="xHoriba.mat",
                   help='Common Raman-shift axis file (default: "xHoriba.mat")')
    p.add_argument("--fit-min", type=float, default=400.0)
    p.add_argument("--fit-max", type=float, default=3100.0)
    p.add_argument("--silent-min", type=float, default=1800.0)
    p.add_argument("--silent-max", type=float, default=2700.0)
    p.add_argument("--topn", type=int, default=25)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def load_numeric_vector(path: Path):
    mat = sio.loadmat(str(path))
    candidates = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            arr = np.asarray(value).squeeze()
            if arr.ndim == 1 and arr.size > 10:
                candidates.append((key, arr.astype(np.float64)))
    if not candidates:
        raise RuntimeError(f"No numeric 1D spectrum found in: {path}")
    # Prefer the largest vector
    candidates.sort(key=lambda kv: kv[1].size, reverse=True)
    return candidates[0][1], candidates[0][0]


def integrate(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def unit_area_normalize(y, x):
    y = np.asarray(y, dtype=np.float64).copy()
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        raise RuntimeError("Spectrum has too few finite points for AUC normalization.")

    area = integrate(y[finite], x[finite])

    # Preserve the measured spectral shape; use signed AUC when valid.
    # Only fall back to absolute area if the signed integral is numerically degenerate.
    if (not np.isfinite(area)) or abs(area) < 1e-15:
        area = integrate(np.abs(y[finite]), x[finite])

    if (not np.isfinite(area)) or abs(area) < 1e-15:
        raise RuntimeError("Cannot obtain a valid spectral area for normalization.")

    return y / area, float(area)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def correlation_plot(matrix, labels, title, outfile, dpi):
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(matrix, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def dendrogram_plot(Z, labels, outfile, dpi):
    fig, ax = plt.subplots(figsize=(14, 7))
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax
    )
    ax.set_title("Hierarchical clustering of the 48-reference Raman library")
    ax.set_ylabel("Spectral distance (1 - Pearson r)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def svd_plot(S, outfile, dpi):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(np.arange(1, len(S) + 1), S, marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title("Singular-value spectrum of the 48-reference design matrix")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_excel(
    outfile,
    corr,
    long_labels,
    short_labels,
    pair_rows,
    top_rows,
    metrics,
    thresholds,
    cluster_order,
    singular_values,
    inventory_rows,
):
    wb = xlsxwriter.Workbook(str(outfile))

    header_fmt = wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    num_fmt = wb.add_format({"num_format": "0.000000"})
    title_fmt = wb.add_format({"bold": True, "font_size": 12})

    # Correlation matrix
    ws = wb.add_worksheet("Correlation_Matrix")
    ws.write(0, 0, "Reference", header_fmt)
    for j, label in enumerate(short_labels, start=1):
        ws.write(0, j, label, header_fmt)
    for i, label in enumerate(short_labels, start=1):
        ws.write(i, 0, label, header_fmt)
        for j in range(len(short_labels)):
            ws.write_number(i, j + 1, float(corr[i - 1, j]), num_fmt)
    ws.freeze_panes(1, 1)
    ws.set_column(0, 0, 18)
    ws.set_column(1, len(short_labels), 10)

    # All pairs
    ws = wb.add_worksheet("All_Pairs")
    headers = ["Reference_1", "Reference_2", "Pearson_r", "Abs_r"]
    for c, h in enumerate(headers):
        ws.write(0, c, h, header_fmt)
    for r, row in enumerate(pair_rows, start=1):
        ws.write(r, 0, row[0])
        ws.write(r, 1, row[1])
        ws.write_number(r, 2, row[2], num_fmt)
        ws.write_number(r, 3, row[3], num_fmt)
    ws.set_column(0, 1, 28)
    ws.set_column(2, 3, 14)

    # Top pairs
    ws = wb.add_worksheet("Top_25_Pairs")
    for c, h in enumerate(headers):
        ws.write(0, c, h, header_fmt)
    for r, row in enumerate(top_rows, start=1):
        ws.write(r, 0, row[0])
        ws.write(r, 1, row[1])
        ws.write_number(r, 2, row[2], num_fmt)
        ws.write_number(r, 3, row[3], num_fmt)
    ws.set_column(0, 1, 28)
    ws.set_column(2, 3, 14)

    # Condition/SVD
    ws = wb.add_worksheet("Condition_SVD")
    ws.write_row(0, 0, ["Metric", "Value"], header_fmt)
    for r, (metric, value) in enumerate(metrics, start=1):
        ws.write(r, 0, metric)
        if isinstance(value, (int, float, np.integer, np.floating)):
            ws.write_number(r, 1, float(value), num_fmt)
        else:
            ws.write(r, 1, str(value))
    ws.set_column(0, 0, 64)
    ws.set_column(1, 1, 24)

    # Correlation thresholds
    ws = wb.add_worksheet("Correlation_Thresholds")
    ws.write_row(0, 0, ["Threshold", "Number_of_pairs"], header_fmt)
    for r, (threshold, count) in enumerate(thresholds, start=1):
        ws.write(r, 0, threshold)
        ws.write_number(r, 1, int(count))
    ws.set_column(0, 0, 18)

    # Cluster order
    ws = wb.add_worksheet("Cluster_Order")
    ws.write_row(0, 0, ["Cluster_order", "Reference"], header_fmt)
    for r, label in enumerate(cluster_order, start=1):
        ws.write_number(r, 0, r)
        ws.write(r, 1, label)
    ws.set_column(1, 1, 30)

    # Singular values
    ws = wb.add_worksheet("Singular_Values")
    ws.write_row(0, 0, ["Index", "Singular_value"], header_fmt)
    for r, s in enumerate(singular_values, start=1):
        ws.write_number(r, 0, r)
        ws.write_number(r, 1, float(s), num_fmt)

    # Inventory
    ws = wb.add_worksheet("Reference_Inventory")
    inv_headers = ["No.", "Article component", "MAT filename", "Category", "MAT variable", "Full_length", "Finite_points", "AUC_before_normalization"]
    ws.write_row(0, 0, inv_headers, header_fmt)
    for r, row in enumerate(inventory_rows, start=1):
        for c, value in enumerate(row):
            if isinstance(value, (int, float, np.integer, np.floating)):
                ws.write_number(r, c, float(value), num_fmt)
            else:
                ws.write(r, c, str(value))
    ws.set_column(0, 0, 6)
    ws.set_column(1, 1, 40)
    ws.set_column(2, 2, 36)
    ws.set_column(3, 3, 24)
    ws.set_column(4, 7, 20)

    wb.close()


def main():
    args = parse_args()
    refdir = Path(args.refdir)
    outdir = Path(args.outdir)

    if not refdir.exists():
        raise FileNotFoundError(f"Reference folder not found: {refdir.resolve()}")

    if len(ARTICLE48) != 48 or len(SHORT_LABELS) != 48:
        raise RuntimeError("Internal component list is not exactly 48 entries.")

    outdir.mkdir(parents=True, exist_ok=True)

    # Load Raman shift axis
    axis_path = refdir / args.axis_file
    if not axis_path.exists():
        raise FileNotFoundError(f"Raman-shift axis file not found: {axis_path.resolve()}")

    x, x_var = load_numeric_vector(axis_path)
    print(f"[AXIS] {axis_path.name} | variable={x_var} | n={len(x)}")
    print(f"[AXIS] range={np.nanmin(x):.6f} to {np.nanmax(x):.6f} cm^-1")

    # Load + full-spectrum unit-area normalize
    spectra = []
    inventory = []
    missing = []

    print("\n" + "=" * 78)
    print("Checking final 48 manuscript references")
    print("=" * 78)

    for idx, (article_name, filename, category) in enumerate(ARTICLE48, start=1):
        path = refdir / filename
        if not path.exists():
            missing.append((article_name, filename))
            print(f"[MISSING] {idx:02d}. {article_name} -> {filename}")
            continue

        y, var_name = load_numeric_vector(path)
        if len(y) != len(x):
            raise RuntimeError(
                f"Length mismatch for {filename}: spectrum n={len(y)}, axis n={len(x)}"
            )

        y_norm, auc = unit_area_normalize(y, x)
        spectra.append(y_norm)

        inventory.append([
            idx,
            article_name,
            filename,
            category,
            var_name,
            len(y),
            int(np.isfinite(y).sum()),
            auc,
        ])

        print(
            f"[OK] {idx:02d}. {article_name:<42} "
            f"{filename:<38} finite={np.isfinite(y).sum()}/{len(y)}"
        )

    if missing:
        print("\nERROR: missing manuscript reference files:")
        for article_name, filename in missing:
            print(f"  - {article_name} -> {filename}")
        raise RuntimeError("One or more of the final 48 reference files are missing.")

    Y_full = np.column_stack(spectra)  # wavenumber x 48

    # Match the established diagnostic logic:
    # 400-3100 cm^-1, excluding the 1800-2700 cm^-1 silent region.
    base_fit_mask = (
        np.isfinite(x)
        & (x >= args.fit_min)
        & (x <= args.fit_max)
        & ~((x >= args.silent_min) & (x <= args.silent_max))
    )

    # Some lipid standards are only finite over ~456-3149 cm^-1.
    # To avoid artificial extrapolation, keep only fitting points that are finite
    # for ALL 48 references.
    common_finite = np.all(np.isfinite(Y_full), axis=1)
    fit_mask = base_fit_mask & common_finite

    x_fit = x[fit_mask]
    X = Y_full[fit_mask, :]  # fitting points x 48

    if X.shape[0] < 100:
        raise RuntimeError(f"Too few common finite fitting points: {X.shape[0]}")

    print("\n" + "=" * 78)
    print("DESIGN MATRIX")
    print("=" * 78)
    print(f"Base fitting points before common-finite restriction: {base_fit_mask.sum()}")
    print(f"Common finite fitting points used: {fit_mask.sum()}")
    print(f"Final fitting Raman range: {x_fit.min():.3f} to {x_fit.max():.3f} cm^-1")
    print(f"Design matrix shape: {X.shape}")

    # Correlation
    corr = np.corrcoef(X.T)
    corr = np.clip(corr, -1.0, 1.0)

    pair_rows = []
    for i in range(48):
        for j in range(i + 1, 48):
            r = float(corr[i, j])
            pair_rows.append((SHORT_LABELS[i], SHORT_LABELS[j], r, abs(r)))

    pair_rows.sort(key=lambda row: row[3], reverse=True)
    top_rows = pair_rows[:args.topn]

    # Hierarchical clustering
    distance = 1.0 - corr
    np.fill_diagonal(distance, 0.0)
    distance = np.maximum(distance, 0.0)
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    cluster_labels = [SHORT_LABELS[i] for i in order]
    corr_clustered = corr[np.ix_(order, order)]

    # SVD / condition
    singular_values = np.linalg.svd(X, compute_uv=False)
    rank = int(np.linalg.matrix_rank(X))
    cond_actual = float(np.linalg.cond(X))

    col_norms = np.linalg.norm(X, axis=0)
    if np.any(col_norms == 0):
        raise RuntimeError("At least one reference column has zero L2 norm.")
    X_l2 = X / col_norms
    cond_l2 = float(np.linalg.cond(X_l2))

    thresholds = [
        ("r ≥ 0.95", sum(row[2] >= 0.95 for row in pair_rows)),
        ("r ≥ 0.90", sum(row[2] >= 0.90 for row in pair_rows)),
        ("r ≥ 0.80", sum(row[2] >= 0.80 for row in pair_rows)),
    ]

    # Export CSVs
    corr_rows = [["Reference"] + SHORT_LABELS]
    for i, label in enumerate(SHORT_LABELS):
        corr_rows.append([label] + [float(v) for v in corr[i, :]])
    write_csv(outdir / "48_reference_Pearson_correlation_matrix.csv", corr_rows)

    all_pairs_csv = [["Reference_1", "Reference_2", "Pearson_r", "Abs_r"]]
    all_pairs_csv += [[a, b, r, ar] for a, b, r, ar in pair_rows]
    write_csv(outdir / "48_reference_pairwise_correlations_ranked.csv", all_pairs_csv)

    top_csv = [["Reference_1", "Reference_2", "Pearson_r", "Abs_r"]]
    top_csv += [[a, b, r, ar] for a, b, r, ar in top_rows]
    write_csv(outdir / f"top_{args.topn}_correlated_reference_pairs.csv", top_csv)

    summary_rows = [
        ["Metric", "Value"],
        ["Number of reference components", 48],
        ["Number of fitting wavenumber points", int(X.shape[0])],
        ["Fitting spectral range", f"{args.fit_min:g}-{args.fit_max:g} cm^-1"],
        ["Excluded silent region", f"{args.silent_min:g}-{args.silent_max:g} cm^-1"],
        ["Common finite Raman range used", f"{x_fit.min():.3f}-{x_fit.max():.3f} cm^-1"],
        ["Matrix rank", rank],
        ["2-norm condition number (unit-area-normalized design matrix)", cond_actual],
        ["2-norm condition number after additional L2 column normalization (diagnostic only)", cond_l2],
        ["Largest singular value", float(singular_values[0])],
        ["Smallest singular value", float(singular_values[-1])],
    ]
    write_csv(outdir / "condition_number_and_SVD_summary.csv", summary_rows)

    write_csv(
        outdir / "singular_values.csv",
        [["Index", "Singular_value"]]
        + [[i + 1, float(v)] for i, v in enumerate(singular_values)]
    )

    write_csv(
        outdir / "hierarchical_cluster_order.csv",
        [["Cluster_order", "Reference"]]
        + [[i + 1, label] for i, label in enumerate(cluster_labels)]
    )

    write_csv(
        outdir / "reference_inventory_48.csv",
        [["No.", "Article component", "MAT filename", "Category", "MAT variable",
          "Full_length", "Finite_points", "AUC_before_normalization"]]
        + inventory
    )

    # Save exact fitted design matrix
    np.savez_compressed(
        outdir / "interpolated_reference_design_matrix.npz",
        x_fit=x_fit,
        X=X,
        labels=np.array(SHORT_LABELS, dtype=object),
        article_labels=np.array([r[0] for r in ARTICLE48], dtype=object),
        filenames=np.array([r[1] for r in ARTICLE48], dtype=object),
    )

    # Figures
    correlation_plot(
        corr,
        SHORT_LABELS,
        "Pearson correlation matrix of the 48-reference Raman library",
        outdir / "correlation_matrix.png",
        args.dpi
    )
    correlation_plot(
        corr_clustered,
        cluster_labels,
        "Cluster-ordered Pearson correlation matrix of the 48-reference Raman library",
        outdir / "correlation_matrix_clustered.png",
        args.dpi
    )
    dendrogram_plot(
        Z,
        SHORT_LABELS,
        outdir / "hierarchical_cluster_dendrogram.png",
        args.dpi
    )
    svd_plot(
        singular_values,
        outdir / "singular_value_spectrum.png",
        args.dpi
    )

    # Diagnostic summary text
    with open(outdir / "diagnostic_summary.txt", "w", encoding="utf-8") as f:
        f.write("FV-RMH 48-reference Raman library diagnostics\n")
        f.write("=" * 64 + "\n")
        f.write("Reference panel: exact 48 components listed in the manuscript table.\n")
        f.write(f"Reference axis: {args.axis_file} ({len(x)} points)\n")
        f.write(f"Reference axis range: {np.nanmin(x):.6f}-{np.nanmax(x):.6f} cm^-1\n")
        f.write("Normalization: each reference was normalized to unit integrated spectral area\n")
        f.write("over its finite full reference axis before fitting-window selection.\n")
        f.write(f"Fitting window: {args.fit_min:g}-{args.fit_max:g} cm^-1\n")
        f.write(f"Excluded silent region: {args.silent_min:g}-{args.silent_max:g} cm^-1\n")
        f.write(f"Base fitting points: {base_fit_mask.sum()}\n")
        f.write(f"Common finite fitting points used: {fit_mask.sum()}\n")
        f.write(f"Design matrix shape: {X.shape[0]} x {X.shape[1]}\n")
        f.write(f"Matrix rank: {rank}/48\n")
        f.write(f"2-norm condition number (unit-area matrix): {cond_actual:.6f}\n")
        f.write(f"2-norm condition number after L2 normalization (diagnostic): {cond_l2:.6f}\n")
        f.write("\nPairwise Pearson-correlation thresholds:\n")
        for threshold, count in thresholds:
            f.write(f"  {threshold}: {count} pairs\n")
        f.write("\nTop correlated pairs:\n")
        for a, b, r, _ in top_rows:
            f.write(f"  {a} vs {b}: r={r:.6f}\n")
        f.write("\nInterpretation note:\n")
        f.write("Full rank excludes exact linear dependence, but high pairwise correlations and\n")
        f.write("a large condition number indicate substantial spectral collinearity among some\n")
        f.write("references. Diagnostics should therefore support transparent, reference-constrained\n")
        f.write("interpretation rather than claims that all components are spectrally orthogonal.\n")

    # Consolidated Excel
    metrics = [(row[0], row[1]) for row in summary_rows[1:]]
    make_excel(
        outdir / "48_reference_library_diagnostics.xlsx",
        corr,
        [r[0] for r in ARTICLE48],
        SHORT_LABELS,
        pair_rows,
        top_rows,
        metrics,
        thresholds,
        cluster_labels,
        singular_values,
        inventory,
    )

    # ZIP package, analogous to the previously supplied 42-reference package
    zip_path = shutil.make_archive(
        base_name=str(outdir.resolve()),
        format="zip",
        root_dir=str(outdir.parent.resolve()),
        base_dir=outdir.name,
    )

    print("\n" + "=" * 78)
    print("FINISHED SUCCESSFULLY")
    print("=" * 78)
    print(f"Matrix rank: {rank}/48")
    print(f"Condition number (unit-area): {cond_actual:.6f}")
    print(f"Condition number (additional L2 diagnostic): {cond_l2:.6f}")
    print(f"Results folder: {outdir.resolve()}")
    print(f"ZIP package: {zip_path}")
    print("\nKey outputs:")
    print("  correlation_matrix_clustered.png")
    print("  hierarchical_cluster_dendrogram.png")
    print("  singular_value_spectrum.png")
    print("  48_reference_library_diagnostics.xlsx")
    print("  diagnostic_summary.txt")


if __name__ == "__main__":
    main()
