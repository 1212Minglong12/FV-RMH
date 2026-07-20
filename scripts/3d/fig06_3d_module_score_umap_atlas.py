#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fig06: global UMAP-aligned molecular module atlas for 3D Raman data.

Rows = broad Raman molecular modules.
Columns = cancer-adjacent, Lepidic, Acinar, Papillary, Micropapillary,
          Complex glands, Solid.
Every panel retains the same UMAP coordinates: global pixels are pale grey and
the selected pathology group is overlaid in yellow-to-red module activity.

Run:
cd "FV-RMH"
python ".\\scripts\\3d\\fig06_3d_module_score_umap_atlas.py" ^
  --input-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_2000pixels" ^
  --out-dir "3dresults\\fig06_3d_module_score_umap_atlas_2000pixels"
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams["font.family"]="Arial"
plt.rcParams["pdf.fonttype"]=42
plt.rcParams["ps.fonttype"]=42
plt.rcParams["svg.fonttype"]="none"

GROUP_ORDER=["Cancer-adjacent","Lepidic","Acinar","Papillary","Micropapillary","Complex glands","Solid"]
GROUP_SHORT={"Cancer-adjacent":"CA","Lepidic":"LEP","Acinar":"ACN","Papillary":"PAP","Micropapillary":"MP","Complex glands":"CGP","Solid":"SOL"}
MODULE_ORDER=["Central metabolism","Lipid metabolism","ECM/stromal remodeling","Immune/protein","Stress/glycoxidation","Amino acid/nitrogen","Other"]
MODULE_LABELS={
"Central metabolism":"Central\nmetabolism","Lipid metabolism":"Lipid\nmetabolism",
"ECM/stromal remodeling":"ECM / stromal\nremodeling","Immune/protein":"Immune /\nprotein",
"Stress/glycoxidation":"Stress /\nglycoxidation","Amino acid/nitrogen":"Amino acid /\nnitrogen","Other":"Other"}

def norm(x): return re.sub(r"[^a-z0-9]+","",str(x).lower())
def module_of(marker):
    k=norm(marker)
    d={
"Central metabolism":["lactate","pyruvate","phosphoenolpyruvic","3phosphoglycerate","phosphoglycerate","glucose","glycol","citrate","fumarate","malate","succinate","acetylcoa","coa","atp","nad","fadh","tca","oxaloacetate","mitochond"],
"Lipid metabolism":["lipid","cholesterol","cholesteryl","cholesterolester","fattyacid","palmit","oleic","linole","triglyceride","phospholipid","sphingo","ceramide","monounsaturated","saturatedlipid","unsaturated"],
"ECM/stromal remodeling":["collagen","elastin","fibronectin","laminin","vitronectin","tenascin","versican","syndecan","matrix","ecm","acta2","actin","vimentin","cathepsin","mmp","integrin","hyaluron","periostin","thrombospondin","spp1"],
"Immune/protein":["pd1","pdl1","b7h3","sting","cd","hla","mhc","immun","cytokine","interferon","tnf","il","protein","histone","albumin","keratin","cytokeratin","ck"],
"Stress/glycoxidation":["glycoxidation","glycation","oxid","ros","glutathione","slactoylglutathione","lactoyl","gsh","gssg","carbonyl","mda","4hne","stress","hypoxia","nrf2","hif"],
"Amino acid/nitrogen":["alanine","arginine","asparagine","aspartate","cysteine","glutamate","glutamine","glycine","histidine","isoleucine","leucine","lysine","methionine","phenylalanine","proline","serine","threonine","tryptophan","tyrosine","valine","amino","urea","nitrogen","creatine","taurine"]}
    for m,words in d.items():
        if any(w in k for w in words): return m
    return "Other"

def load(input_dir):
    ep=input_dir/"source_embedding_coordinates.csv"; fp=input_dir/"source_true_pixel_robust_z_features.csv"
    if not ep.exists(): raise FileNotFoundError(f"Missing {ep}")
    if not fp.exists(): raise FileNotFoundError(f"Missing {fp}")
    c=pd.read_csv(ep); f=pd.read_csv(fp)
    if len(c)!=len(f): raise RuntimeError(f"Embedding rows ({len(c)}) and feature rows ({len(f)}) do not match.")
    keep=c["group"].isin(GROUP_ORDER).values
    c=c.loc[keep].reset_index(drop=True); f=f.loc[keep].reset_index(drop=True)
    cols=[x for x in f.columns if np.nanstd(f[x].astype(float).values)>1e-10]
    f=f[cols].replace([np.inf,-np.inf],np.nan).fillna(f[cols].median()).fillna(0.0)
    return c,f

def scores(features):
    assignment=pd.DataFrame({"marker":features.columns,"module":[module_of(x) for x in features.columns]})
    raw=pd.DataFrame(index=features.index)
    out=pd.DataFrame(index=features.index)
    for m in MODULE_ORDER:
        cols=assignment.loc[assignment.module==m,"marker"].tolist()
        if cols: raw[m]=features[cols].mean(axis=1)
        else: raw[m]=np.nan
        a=raw[m].values.astype(float); finite=a[np.isfinite(a)]
        if len(finite)<10: out[m]=0.0
        else:
            lo,hi=np.nanpercentile(finite,[2,98])
            out[m]=0.5 if not np.isfinite(hi-lo) or hi-lo<=1e-12 else np.clip((a-lo)/(hi-lo),0,1)
    return assignment,raw,out

def limits(c):
    x=c.umap_1.astype(float).values; y=c.umap_2.astype(float).values; good=np.isfinite(x)&np.isfinite(y)
    xmin,xmax=np.nanpercentile(x[good],[.2,99.8]); ymin,ymax=np.nanpercentile(y[good],[.2,99.8])
    return xmin-(xmax-xmin)*.05,xmax+(xmax-xmin)*.05,ymin-(ymax-ymin)*.05,ymax+(ymax-ymin)*.05

def style(ax,lim,frame=True):
    xmin,xmax,ymin,ymax=lim; ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_aspect("equal","box"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(frame); sp.set_color("#B8B8B8"); sp.set_linewidth(.55)

def save(fig,out,stem):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ["png","pdf","svg"]:
        kw={"dpi":600} if ext=="png" else {}
        fig.savefig(out/f"{stem}.{ext}",bbox_inches="tight",pad_inches=.02,**kw)

def atlas(c,score,out,with_text=True):
    lim=limits(c); bg=c.sample(n=min(42000,len(c)),random_state=1)
    fig,axes=plt.subplots(len(MODULE_ORDER),len(GROUP_ORDER),figsize=(14.0,12.1),constrained_layout=True)
    for r,m in enumerate(MODULE_ORDER):
        for q,g in enumerate(GROUP_ORDER):
            ax=axes[r,q]
            ax.scatter(bg.umap_1,bg.umap_2,s=.38,c="#D9DDE1",alpha=.27,linewidths=0,rasterized=True,zorder=1)
            idx=c.group.astype(str).values==g
            ax.scatter(c.loc[idx,"umap_1"],c.loc[idx,"umap_2"],c=score.loc[idx,m].values,cmap="YlOrRd",norm=Normalize(0,1),s=1.0,alpha=.84,linewidths=0,rasterized=True,zorder=2)
            style(ax,lim,with_text)
            if with_text and r==0: ax.set_title(GROUP_SHORT[g],fontsize=7.5,fontweight="bold",pad=2)
            if with_text and q==0: ax.set_ylabel(MODULE_LABELS[m],fontsize=7,rotation=0,ha="right",va="center",labelpad=28,fontweight="bold")
    if with_text:
        fig.suptitle("Global UMAP-aligned Raman molecular module landscape across LUAD growth patterns",fontsize=14,fontweight="bold",y=1.01)
        sm=plt.cm.ScalarMappable(cmap="YlOrRd",norm=Normalize(0,1)); sm.set_array([])
        cb=fig.colorbar(sm,ax=axes.ravel().tolist(),fraction=.012,pad=.012,location="right")
        cb.set_label("Normalized module activity",fontsize=8); cb.ax.tick_params(labelsize=7)
    save(fig,out,"Fig06_3D_global_UMAP_module_atlas"+("" if with_text else "_no_text")); plt.close(fig)

def strips(c,score,out):
    lim=limits(c); bg=c.sample(n=min(42000,len(c)),random_state=1)
    for mode in ["with_text","no_text"]:
        wt=mode=="with_text"; folder=out/f"single_module_strips_{mode}"; folder.mkdir(parents=True,exist_ok=True)
        for m in MODULE_ORDER:
            fig,axs=plt.subplots(1,len(GROUP_ORDER),figsize=(14.3,2.4),constrained_layout=True)
            for j,g in enumerate(GROUP_ORDER):
                ax=axs[j]; ax.scatter(bg.umap_1,bg.umap_2,s=.38,c="#D9DDE1",alpha=.27,linewidths=0,rasterized=True)
                idx=c.group.astype(str).values==g
                ax.scatter(c.loc[idx,"umap_1"],c.loc[idx,"umap_2"],c=score.loc[idx,m].values,cmap="YlOrRd",norm=Normalize(0,1),s=1.0,alpha=.84,linewidths=0,rasterized=True)
                style(ax,lim,wt)
                if wt:
                    ax.set_title(GROUP_SHORT[g],fontsize=8,fontweight="bold")
                    if j==0: ax.set_ylabel(MODULE_LABELS[m],fontsize=8,rotation=0,ha="right",va="center",labelpad=30,fontweight="bold")
            save(fig,folder,m.replace("/","_").replace(" ","_")+"_UMAP_strip"+("" if wt else "_no_text")); plt.close(fig)

def summary(c,raw,out):
    tab=pd.DataFrame(index=GROUP_ORDER,columns=MODULE_ORDER,dtype=float)
    for g in GROUP_ORDER:
        idx=c.group.astype(str).values==g
        for m in MODULE_ORDER: tab.loc[g,m]=float(np.nanmedian(raw.loc[idx,m]))
    tab.to_csv(out/"source_group_module_median_score_matrix.csv")
    tab.reset_index(names="group").melt(id_vars="group",var_name="module",value_name="median_module_score").to_csv(out/"source_group_module_median_score_long.csv",index=False)

def main():
    p=argparse.ArgumentParser(description="Global UMAP-aligned Raman module score atlas.")
    p.add_argument("--input-dir",required=True); p.add_argument("--out-dir",required=True); a=p.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    c,f=load(Path(a.input_dir)); assign,raw,score=scores(f)
    assign.to_csv(out/"source_marker_module_assignment.csv",index=False); raw.to_csv(out/"source_pixel_module_scores_raw.csv",index=False); score.to_csv(out/"source_pixel_module_scores_normalized.csv",index=False)
    summary(c,raw,out); atlas(c,score,out,True); atlas(c,score,out,False); strips(c,score,out)
    print("Done."); print(f"Output folder: {out}"); print(f"Pixel observations: {len(c)}"); print(f"Marker features: {f.shape[1]}"); print("Main figure: Fig06_3D_global_UMAP_module_atlas.png")
if __name__=="__main__": main()
