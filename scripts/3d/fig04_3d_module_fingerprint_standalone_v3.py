#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fig04 v3 standalone: smooth radar + alluvial module fingerprints directly from Fig01.

This version does NOT require Fig03 output files.
It directly reads the Fig01 v7 result folder:
  source_embedding_coordinates.csv
  source_true_pixel_robust_z_features.csv

Then it:
1) assigns each Raman marker to a broad molecular module;
2) calculates dominant module composition per pathology group;
3) calculates group-level module fingerprints;
4) draws a smooth alluvial + smooth radar figure.

Run:
cd "FV-RMH"

python ".\\scripts\\3d\\fig04_3d_module_fingerprint_standalone_v3.py" ^
  --input-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_2000pixels" ^
  --out-dir "3dresults\\fig04_3d_module_fingerprint_standalone_v3_2000pixels"

For 1500-pixels-per-layer:
python ".\\scripts\\3d\\fig04_3d_module_fingerprint_standalone_v3.py" ^
  --input-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_1500pixels" ^
  --out-dir "3dresults\\fig04_3d_module_fingerprint_standalone_v3_1500pixels"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy.interpolate import make_interp_spline

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
GROUP_COLORS = {
    "Cancer-adjacent": "#7A7A7A", "Lepidic": "#4C78A8",
    "Acinar": "#54A24B", "Papillary": "#F58518",
    "Micropapillary": "#E45756", "Complex glands": "#B279A2",
    "Solid": "#C92D39"
}
MODULE_ORDER = [
    "Central metabolism", "Lipid metabolism", "ECM/stromal remodeling",
    "Immune/protein", "Stress/glycoxidation", "Amino acid/nitrogen", "Other"
]
MODULE_SHORT = {
    "Central metabolism": "Central\nmetabolism",
    "Lipid metabolism": "Lipid\nmetabolism",
    "ECM/stromal remodeling": "ECM / stromal\nremodeling",
    "Immune/protein": "Immune /\nprotein",
    "Stress/glycoxidation": "Stress /\nglycoxidation",
    "Amino acid/nitrogen": "Amino acid /\nnitrogen",
    "Other": "Other",
}
MODULE_COLORS = {
    "Central metabolism": "#F58518",
    "Lipid metabolism": "#4C78A8",
    "ECM/stromal remodeling": "#E45756",
    "Immune/protein": "#72B7B2",
    "Stress/glycoxidation": "#B279A2",
    "Amino acid/nitrogen": "#54A24B",
    "Other": "#9D9487",
}

def normalize_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def infer_marker_module(marker: str) -> str:
    key = normalize_name(marker)
    central = [
        "lactate", "pyruvate", "phosphoenolpyruvic", "3phosphoglycerate",
        "phosphoglycerate", "glucose", "glycol", "citrate", "fumarate",
        "malate", "succinate", "acetylcoa", "coa", "atp", "nad", "fadh",
        "tca", "oxaloacetate", "mitochond"
    ]
    lipid = [
        "lipid", "cholesterol", "cholesteryl", "cholesterolester",
        "fattyacid", "palmit", "oleic", "linole", "triglyceride",
        "phospholipid", "sphingo", "ceramide", "monounsaturated",
        "saturatedlipid", "unsaturated"
    ]
    ecm = [
        "collagen", "elastin", "fibronectin", "laminin", "vitronectin",
        "tenascin", "versican", "syndecan", "matrix", "ecm", "acta2",
        "actin", "vimentin", "cathepsin", "mmp", "integrin", "hyaluron",
        "periostin", "thrombospondin"
    ]
    immune = [
        "pd1", "pdl1", "b7h3", "sting", "cd", "hla", "mhc",
        "immun", "cytokine", "interferon", "tnf", "il", "protein",
        "histone", "albumin", "keratin", "cytokeratin", "ck"
    ]
    stress = [
        "glycoxidation", "glycation", "oxid", "ros", "glutathione",
        "slactoylglutathione", "lactoyl", "gsh", "gssg", "carbonyl",
        "mda", "4hne", "stress", "hypoxia", "nrf2", "hif"
    ]
    amino = [
        "alanine", "arginine", "asparagine", "aspartate", "cysteine",
        "glutamate", "glutamine", "glycine", "histidine", "isoleucine",
        "leucine", "lysine", "methionine", "phenylalanine", "proline",
        "serine", "threonine", "tryptophan", "tyrosine", "valine",
        "amino", "urea", "nitrogen", "creatine", "taurine"
    ]
    if any(k in key for k in central): return "Central metabolism"
    if any(k in key for k in lipid): return "Lipid metabolism"
    if any(k in key for k in ecm): return "ECM/stromal remodeling"
    if any(k in key for k in immune): return "Immune/protein"
    if any(k in key for k in stress): return "Stress/glycoxidation"
    if any(k in key for k in amino): return "Amino acid/nitrogen"
    return "Other"

def build_marker_module_table(feature_cols):
    return pd.DataFrame({
        "marker": list(feature_cols),
        "module": [infer_marker_module(m) for m in feature_cols]
    })

def compute_module_scores(features, marker_module):
    out = pd.DataFrame(index=features.index)
    for module in MODULE_ORDER:
        cols = marker_module.loc[marker_module["module"] == module, "marker"].tolist()
        cols = [c for c in cols if c in features.columns]
        out[module] = features[cols].mean(axis=1) if cols else np.nan
    valid = [m for m in MODULE_ORDER if out[m].notna().any()]
    if not valid:
        raise RuntimeError("No module score could be calculated.")
    out["dominant_module"] = out[valid].idxmax(axis=1)
    out["dominant_score"] = out[valid].max(axis=1)
    return out

def composition_matrix(comp):
    mat = pd.DataFrame(0.0, index=GROUP_ORDER, columns=MODULE_ORDER)
    for _, row in comp.iterrows():
        g, m = str(row["group"]), str(row["dominant_module"])
        if g in mat.index and m in mat.columns:
            mat.loc[g, m] = float(row["fraction"])
    return mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

def radar_matrix(scores, coords):
    module_cols = [m for m in MODULE_ORDER if m in scores.columns]
    temp = pd.concat([coords[["group"]].reset_index(drop=True), scores[module_cols].reset_index(drop=True)], axis=1)
    temp = temp[temp["group"].isin(GROUP_ORDER)].copy()
    means = temp.groupby("group", observed=True)[module_cols].mean().reindex(GROUP_ORDER)
    scaled = pd.DataFrame(0.0, index=GROUP_ORDER, columns=MODULE_ORDER)
    for m in module_cols:
        v = means[m].values.astype(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        scaled[m] = (v - lo) / (hi - lo) if np.isfinite(hi - lo) and (hi-lo)>1e-12 else 0.0
    return scaled.fillna(0.0)

def draw_alluvial(ax, mat, with_text=True):
    x = np.arange(len(GROUP_ORDER), dtype=float)
    width = 0.34
    bottom = pd.DataFrame(index=GROUP_ORDER, columns=MODULE_ORDER, dtype=float)
    top = pd.DataFrame(index=GROUP_ORDER, columns=MODULE_ORDER, dtype=float)
    cumulative = np.zeros(len(GROUP_ORDER), dtype=float)
    for m in MODULE_ORDER:
        vals = mat[m].values.astype(float)
        bottom[m] = cumulative
        top[m] = cumulative + vals
        cumulative += vals

    for m in MODULE_ORDER:
        for i in range(len(GROUP_ORDER)-1):
            x0, x1 = x[i]+width/2, x[i+1]-width/2
            y0b, y0t = float(bottom.iloc[i][m]), float(top.iloc[i][m])
            y1b, y1t = float(bottom.iloc[i+1][m]), float(top.iloc[i+1][m])
            if (y0t-y0b)<=0 and (y1t-y1b)<=0:
                continue
            dx = x1-x0
            verts = [
                (x0,y0b),(x0+0.45*dx,y0b),(x1-0.45*dx,y1b),(x1,y1b),
                (x1,y1t),(x1-0.45*dx,y1t),(x0+0.45*dx,y0t),(x0,y0t),(x0,y0b)
            ]
            codes = [MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
                     MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MplPath(verts,codes), facecolor=MODULE_COLORS[m], edgecolor="none", alpha=0.24, zorder=1))
    for j,g in enumerate(GROUP_ORDER):
        for m in MODULE_ORDER:
            h = float(mat.loc[g,m])
            if h <= 0: continue
            ax.bar(x[j],h,bottom=float(bottom.loc[g,m]),width=width,color=MODULE_COLORS[m],edgecolor="black",linewidth=0.45,zorder=3)
    ax.set_xlim(x[0]-0.58,x[-1]+0.58); ax.set_ylim(0,1)
    if with_text:
        ax.set_xticks(x); ax.set_xticklabels([GROUP_SHORT[g] for g in GROUP_ORDER],fontsize=8)
        ax.set_ylabel("Relative dominant-module fraction",fontsize=9)
        ax.set_title("Dominant module composition along LUAD growth-pattern progression",loc="left",fontsize=11.5,fontweight="bold",pad=8)
        ax.tick_params(axis="y",labelsize=8)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

def smooth_closed_curve(angles, values, n_points=280):
    ang = np.asarray(angles,float)
    val = np.asarray(values,float)
    ang_closed = np.r_[ang, ang[0]+2*np.pi]
    val_closed = np.r_[val, val[0]]
    dense = np.linspace(0,2*np.pi,n_points)
    try:
        spline = make_interp_spline(ang_closed,val_closed,k=3,bc_type="periodic")
        rad = spline(dense)
    except Exception:
        rad = np.interp(dense,ang_closed,val_closed)
    rad = np.clip(rad,0,1.05)
    return np.r_[dense,dense[0]], np.r_[rad,rad[0]]

def draw_radar(ax, mat, with_text=True):
    modules = MODULE_ORDER
    n = len(modules)
    angles = np.linspace(0,2*np.pi,n,endpoint=False)
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_ylim(0,1.05)
    ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["0.25","0.50","0.75","1.00"] if with_text else [],fontsize=6,color="#777")
    ax.grid(True,color="#D5D5D5",linewidth=0.55,linestyle="--",alpha=0.65)
    ax.spines["polar"].set_color("#B8B8B8"); ax.spines["polar"].set_linewidth(0.8)
    if with_text:
        ax.set_xticks(angles); ax.set_xticklabels([MODULE_SHORT[m] for m in modules],fontsize=7)
    else:
        ax.set_xticks([]); ax.set_yticklabels([])
    for g in GROUP_ORDER:
        values = mat.loc[g,modules].values.astype(float)
        th, rr = smooth_closed_curve(angles, values)
        ax.plot(th,rr,color=GROUP_COLORS[g],linewidth=1.8,label=GROUP_SHORT[g],alpha=0.95,solid_capstyle="round",solid_joinstyle="round")
        ax.fill(th,rr,color=GROUP_COLORS[g],alpha=0.065)
        ax.scatter(angles,values,s=18,facecolor="white",edgecolor=GROUP_COLORS[g],linewidth=0.9,zorder=5)
    if with_text:
        ax.set_title("Raman molecular fingerprint profiles",fontsize=11.5,fontweight="bold",pad=18)
        ax.legend(frameon=False,fontsize=7.5,loc="center left",bbox_to_anchor=(1.15,0.5))

def save_multi(fig,out_dir,stem,dpi=600,pad=0.02):
    out_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(out_dir/f"{stem}.png",dpi=dpi,bbox_inches="tight",pad_inches=pad)
    fig.savefig(out_dir/f"{stem}.pdf",bbox_inches="tight",pad_inches=pad)
    fig.savefig(out_dir/f"{stem}.svg",bbox_inches="tight",pad_inches=pad)

def make_fig(mat,radar,out_dir,with_text=True):
    fig = plt.figure(figsize=(13.6,6.0),constrained_layout=True)
    gs = fig.add_gridspec(1,2,width_ratios=[1.28,1.0],wspace=0.22)
    ax1 = fig.add_subplot(gs[0,0]); draw_alluvial(ax1,mat,with_text)
    ax2 = fig.add_subplot(gs[0,1],projection="polar"); draw_radar(ax2,radar,with_text)
    if with_text:
        ax1.text(-0.10,1.04,"a",transform=ax1.transAxes,fontsize=14,fontweight="bold")
        ax2.text(-0.12,1.08,"b",transform=ax2.transAxes,fontsize=14,fontweight="bold")
        fig.suptitle("Raman module fingerprints across cancer-adjacent tissue and LUAD growth patterns",fontsize=14,fontweight="bold",y=1.03)
    stem = "Fig04_3D_module_fingerprint_standalone_v3" + ("" if with_text else "_no_text")
    save_multi(fig,out_dir,stem)
    plt.close(fig)

def export_single(mat,radar,out_dir):
    for mode in ["with_text","no_text"]:
        wt = mode=="with_text"
        d = out_dir/f"single_panels_{mode}"; d.mkdir(parents=True,exist_ok=True)
        fig,ax=plt.subplots(figsize=(7.2,5.0)); draw_alluvial(ax,mat,wt); fig.tight_layout(); save_multi(fig,d,f"panel_a_alluvial_{mode}"); plt.close(fig)
        fig=plt.figure(figsize=(6.0,5.6)); ax=fig.add_subplot(111,projection="polar"); draw_radar(ax,radar,wt); fig.tight_layout(); save_multi(fig,d,f"panel_b_smooth_radar_{mode}"); plt.close(fig)

def main():
    p=argparse.ArgumentParser(description="Standalone alluvial + smooth radar from Fig01 output.")
    p.add_argument("--input-dir",required=True,help="Fig01 v7 output folder.")
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()
    input_dir=Path(a.input_dir); out_dir=Path(a.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    emb_path=input_dir/"source_embedding_coordinates.csv"
    feat_path=input_dir/"source_true_pixel_robust_z_features.csv"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}")
    coords=pd.read_csv(emb_path)
    features=pd.read_csv(feat_path)
    if len(coords)!=len(features):
        raise RuntimeError(f"Embedding rows ({len(coords)}) and feature rows ({len(features)}) do not match.")
    coords=coords[coords["group"].isin(GROUP_ORDER)].reset_index(drop=True)
    features=features.iloc[:len(coords)].reset_index(drop=True)
    marker_module=build_marker_module_table(features.columns)
    scores=compute_module_scores(features,marker_module)
    work=coords.copy()
    work["dominant_module"]=scores["dominant_module"].values
    work["dominant_score"]=scores["dominant_score"].values
    comp=(work.groupby(["group","dominant_module"]).size().reset_index(name="n_pixels"))
    total=work.groupby("group").size().rename("total_pixels").reset_index()
    comp=comp.merge(total,on="group",how="left"); comp["fraction"]=comp["n_pixels"]/comp["total_pixels"]
    mat=composition_matrix(comp)
    radar=radar_matrix(scores,coords)
    marker_module.to_csv(out_dir/"source_marker_module_assignment.csv",index=False)
    scores.to_csv(out_dir/"source_pixel_module_scores.csv",index=False)
    work.to_csv(out_dir/"source_embedding_with_dominant_module.csv",index=False)
    comp.to_csv(out_dir/"source_dominant_module_composition.csv",index=False)
    mat.to_csv(out_dir/"source_alluvial_module_fraction_matrix.csv")
    radar.to_csv(out_dir/"source_radar_module_score_matrix.csv")
    make_fig(mat,radar,out_dir,True); make_fig(mat,radar,out_dir,False); export_single(mat,radar,out_dir)
    print("Done."); print(f"Output folder: {out_dir}"); print("Main figure: Fig04_3D_module_fingerprint_standalone_v3.png")

if __name__=="__main__":
    main()
