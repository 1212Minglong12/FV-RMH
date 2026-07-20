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
import matplotlib as mpl
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import pdist

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

GROUPS = ["Normal", "High", "Middle", "Low"]
GROUP_DISPLAY = {
    "Normal": "Normal",
    "High": "High differentiated",
    "Middle": "Moderately differentiated",
    "Low": "Poorly differentiated"
}

MODULE_ORDER = [
    "Oxidative Stress", "Calcification", "Extracellular Matrix", "Vitamins",
    "Interleukins", "Glycolipid Metabolism", "Amino Acids", "Infection & Immunity",
    "Repair & Cytokines", "Carotenoids", "Hormones", "Other Metabolites",
]

MODULE_COLORS = {
    "Oxidative Stress": "#F4A6A1",
    "Calcification": "#85A9D6",
    "Extracellular Matrix": "#9FD2D0",
    "Vitamins": "#F5C3A7",
    "Interleukins": "#C8B7E3",
    "Glycolipid Metabolism": "#B8B0A2",
    "Amino Acids": "#F07D7A",
    "Infection & Immunity": "#E88C8C",
    "Repair & Cytokines": "#D3C8C0",
    "Carotenoids": "#F2CF85",
    "Hormones": "#B792C8",
    "Other Metabolites": "#BDBDBD",
}

MODULE_RULES = {
    "Extracellular Matrix": ["acta2", "collagen", "laminin", "elastin", "fibronectin", "syndecan", "versican", "decorin", "biglycan", "vitronectin"],
    "Amino Acids": ["aspartic", "proline", "methionine", "glutamine", "asparagine", "alanine", "glycine", "serine", "tryptophan", "tyrosine", "valine", "leucine", "isoleucine", "lysine", "histidine", "threonine"],
    "Interleukins": ["interleukin", "il_", "il-", "cxcl", "ccl", "tnf", "ifn"],
    "Calcification": ["calcium", "hydroxyapatite", "phosphate", "mineral", "alkalinephosphatase", "alkalinephosphatasealpl"],
    "Infection & Immunity": ["cd68", "cd31", "cd98", "b7h3", "b7-h3", "b7_h3", "pdl1", "cd4", "cd8", "immune", "lysozyme", "ccl7", "sting"],
    "Hormones": ["estrogen", "progesterone", "androgen", "cortisol", "thyroid", "hormone", "cyp"],
    "Carotenoids": ["carotene", "betacarotene", "beta-carotene"],
    "Other Metabolites": ["formaldehyde", "fumarate", "nicotinamide", "succinate", "citrate", "malate", "bmp4", "aspergillus", "butyricacid", "prmt"],
    "Glycolipid Metabolism": ["acetyl", "fructose", "glucose", "phosphoglycerate", "lipid", "glycolipid", "cholesterol", "fatty", "triglyceride", "sphing", "glycerol", "nadh", "pyruvic", "ketoglutaric", "sphinosine"],
    "Vitamins": ["vitamin", "pyridoxine", "albumin"],
    "Repair & Cytokines": ["repair", "cytokine", "wound", "healing", "tgfb", "tgf-b", "tgf", "cdkn1a", "cdh1", "titf1"],
    "Oxidative Stress": ["superoxidedismutase", "sod", "oxidative", "ros", "glutathione", "cml", "pentosidine"],
}

def normalize_marker_name(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def assign_module(marker_name):
    m = normalize_marker_name(marker_name)
    hits = []
    for module, kws in MODULE_RULES.items():
        for kw in kws:
            if normalize_marker_name(kw) in m:
                hits.append(module)
                break
    if not hits:
        return "Other Metabolites"
    priority = {m: i for i, m in enumerate(MODULE_ORDER)}
    return sorted(hits, key=lambda x: priority.get(x, 999))[0]

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

def build_marker_group_table(meta_df, target_layer=1):
    rows, skipped = [], []
    for _, row in meta_df.iterrows():
        if row["group"] not in GROUPS:
            skipped.append({**row.to_dict(), "reason": "group_filtered"})
            continue
        if int(row["layer"]) != int(target_layer):
            skipped.append({**row.to_dict(), "reason": "layer_filtered"})
            continue
        try:
            vec = load_mat_numeric_array(Path(row["file"]))
        except Exception as e:
            skipped.append({**row.to_dict(), "reason": f"load_failed:{e}"})
            continue
        vec = robust_clip(vec)
        finite = vec[np.isfinite(vec)]
        if finite.size == 0:
            skipped.append({**row.to_dict(), "reason": "all_values_missing"})
            continue
        rows.append({
            "marker": row["marker"],
            "group": row["group"],
            "patient": row["patient"],
            "layer": row["layer"],
            "mean_value": float(np.nanmean(finite)),
            "median_value": float(np.nanmedian(finite)),
            "percent_positive": float(np.mean(finite > 0) * 100.0),
            "n_values": int(finite.size),
        })
    return pd.DataFrame(rows), pd.DataFrame(skipped)

def make_marker_stage_matrix(marker_df, value_col="mean_value"):
    agg = marker_df.groupby(["marker", "group"], as_index=False)[value_col].mean()
    wide = agg.pivot(index="marker", columns="group", values=value_col).reindex(columns=GROUPS)
    wide = wide.fillna(wide.mean(axis=1))
    wide = wide.fillna(0.0)
    return wide

def row_zscore(df):
    arr = df.values.astype(float)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    z = (arr - mu) / sd
    return pd.DataFrame(np.clip(z, -2.0, 2.0), index=df.index, columns=df.columns)

def select_top_dynamic_markers(z_df, top_n=30):
    score = z_df.std(axis=1) + 0.5 * (z_df.max(axis=1) - z_df.min(axis=1))
    top_markers = score.sort_values(ascending=False).head(top_n).index.tolist()
    return z_df.loc[top_markers].copy(), score.sort_values(ascending=False)

def cluster_markers(z_df):
    X = z_df.values
    if X.shape[0] <= 2:
        return z_df.copy(), list(range(X.shape[0])), None
    dist = pdist(X, metric="euclidean")
    Z = linkage(dist, method="average")
    d = dendrogram(Z, no_plot=True)
    leaves = d["leaves"]
    return z_df.iloc[leaves].copy(), leaves, Z

def polar_angle_positions(n, start_deg=90, gap_deg=8):
    total = 2 * np.pi - np.deg2rad(gap_deg)
    step = total / n
    start = np.deg2rad(start_deg) + np.deg2rad(gap_deg) / 2.0
    theta = start - np.arange(n) * step
    return theta, step

def draw_ring(ax, values, theta, step, r0, r1, cmap, vmin, vmax):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    width = step * 1.005
    height = r1 - r0
    for t, val in zip(theta, values):
        c = cmap(norm(val))
        ax.bar(t, height=height, width=width, bottom=r0, color=c, edgecolor=c, linewidth=0.0, align="center")

def draw_module_ring(ax, modules, theta, step, r0, r1):
    width = step * 1.005
    height = r1 - r0
    for t, m in zip(theta, modules):
        c = MODULE_COLORS.get(m, "#BDBDBD")
        ax.bar(t, height=height, width=width, bottom=r0, color=c, edgecolor=c, linewidth=0.0, align="center")

def compute_tree_geometry(Z, n_leaves, theta):
    if Z is None:
        return None, None, None, None
    root, _ = to_tree(Z, rd=True)
    leaf_map = {leaf_id: i for i, leaf_id in enumerate(range(n_leaves))}
    leaf_angle = {i: theta[i] for i in range(n_leaves)}
    dist_map, angle_map = {}, {}

    def walk(node):
        dist_map[node.id] = float(node.dist)
        if node.is_leaf():
            idx = leaf_map[node.id]
            angle_map[node.id] = float(leaf_angle[idx])
            return angle_map[node.id]
        a_left = walk(node.left)
        a_right = walk(node.right)
        angle_map[node.id] = (a_left + a_right) / 2.0
        return angle_map[node.id]

    walk(root)
    max_dist = max(dist_map.values()) if len(dist_map) else 1.0
    return root, dist_map, angle_map, max_dist

def radius_from_dist(d, max_dist, r_inner, r_outer):
    if max_dist <= 0:
        return r_outer
    return r_inner + (1.0 - d / max_dist) * (r_outer - r_inner)

def draw_circular_dendrogram(ax, node, dist_map, angle_map, max_dist, r_inner, r_outer, color="#333333", lw=0.68):
    if node is None or node.is_leaf():
        return
    left, right = node.left, node.right
    r_parent = radius_from_dist(dist_map[node.id], max_dist, r_inner, r_outer)
    r_left = radius_from_dist(dist_map[left.id], max_dist, r_inner, r_outer)
    r_right = radius_from_dist(dist_map[right.id], max_dist, r_inner, r_outer)
    th_left = angle_map[left.id]
    th_right = angle_map[right.id]
    ax.plot([th_left, th_left], [r_left, r_parent], color=color, lw=lw, solid_capstyle="round", zorder=2)
    ax.plot([th_right, th_right], [r_right, r_parent], color=color, lw=lw, solid_capstyle="round", zorder=2)
    t1, t2 = th_left, th_right
    if t1 < t2:
        t1, t2 = t2, t1
    ts = np.linspace(t1, t2, 60)
    ax.plot(ts, np.full_like(ts, r_parent), color=color, lw=lw, solid_capstyle="round", zorder=2)
    draw_circular_dendrogram(ax, left, dist_map, angle_map, max_dist, r_inner, r_outer, color=color, lw=lw)
    draw_circular_dendrogram(ax, right, dist_map, angle_map, max_dist, r_inner, r_outer, color=color, lw=lw)

def plot_circular_cluster_figure(z_df_ordered, modules, Z, out_dir: Path):
    markers = z_df_ordered.index.tolist()
    n = len(markers)
    theta, step = polar_angle_positions(n, start_deg=90, gap_deg=8)

    fig = plt.figure(figsize=(17.8, 10.8))

    # 主图放大，去掉外圈文字后就把空间留给主图
    ax = fig.add_axes([0.22, 0.05, 0.66, 0.90], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.03)
    ax.set_axis_off()

    r_mod0, r_mod1 = 0.19, 0.225
    r_den0, r_den1 = 0.24, 0.50
    ring_gap = 0.008
    ring_h = 0.075
    group_rings = {}
    r0 = 0.525
    for g in ["Low", "Middle", "High", "Normal"]:
        group_rings[g] = (r0, r0 + ring_h)
        r0 += ring_h + ring_gap

    if Z is not None:
        root, dist_map, angle_map, max_dist = compute_tree_geometry(Z, n, theta)
        draw_circular_dendrogram(ax, root, dist_map, angle_map, max_dist, r_den0, r_den1, color="#333333", lw=0.70)

    draw_module_ring(ax, modules, theta, step, r_mod0, r_mod1)

    cmap = mpl.cm.coolwarm
    for g in ["Low", "Middle", "High", "Normal"]:
        draw_ring(ax, z_df_ordered[g].values, theta, step, group_rings[g][0], group_rings[g][1], cmap, -2, 2)

    for rr in [r_mod0, r_mod1, r_den0, r_den1]:
        ax.plot(np.linspace(0, 2*np.pi, 400), np.full(400, rr), color="#666666", lw=0.30, alpha=0.55)
    for g in group_rings:
        rr0, rr1 = group_rings[g]
        ax.plot(np.linspace(0, 2*np.pi, 400), np.full(400, rr0), color="#666666", lw=0.25, alpha=0.45)
        ax.plot(np.linspace(0, 2*np.pi, 400), np.full(400, rr1), color="#666666", lw=0.25, alpha=0.45)

    fig.text(0.03, 0.95, "Circular clustering and module-annotated heatmap of core metabolites",
             fontsize=16, fontweight="bold", ha="left", va="center")
    fig.text(0.03, 0.905, "Outer text labels removed; the main circular map is enlarged.",
             fontsize=9.5, ha="left", color="#444444")

    # 左下角保留说明
    fig.text(0.03, 0.75, "Outer heatmap rings", fontsize=10.5, fontweight="bold", ha="left")
    ring_y = [0.72, 0.688, 0.656, 0.624]
    for y, g in zip(ring_y, ["Normal", "High", "Middle", "Low"]):
        fig.add_artist(Rectangle((0.03, y - 0.008), 0.018, 0.014, transform=fig.transFigure,
                                 facecolor=cmap(mpl.colors.Normalize(-2, 2)(float(z_df_ordered[g].mean()))),
                                 edgecolor="none"))
        fig.text(0.053, y, GROUP_DISPLAY[g], fontsize=9.5, ha="left", va="center")

    cax = fig.add_axes([0.05, 0.28, 0.016, 0.15])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-2, vmax=2), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Z-score", fontsize=10, labelpad=6)
    cb.ax.tick_params(labelsize=9)

    leg_ax = fig.add_axes([0.05, 0.05, 0.24, 0.19])
    leg_ax.axis("off")
    leg_ax.text(0.0, 1.02, "Metabolic module", fontsize=11, fontweight="bold", ha="left", va="bottom")
    y = 0.92
    for m in MODULE_ORDER:
        leg_ax.add_patch(Rectangle((0.0, y - 0.035), 0.040, 0.055, facecolor=MODULE_COLORS[m], edgecolor="none"))
        leg_ax.text(0.055, y - 0.008, m, fontsize=8.6, ha="left", va="center")
        y -= 0.073

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"Fig8_circular_cluster_module_heatmap.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Circular clustering and module heatmap without outer labels")
    parser.add_argument("--data-dir", type=str, default="data_2d")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--target-layer", type=int, default=1)
    parser.add_argument("--top-n-markers", type=int, default=30, help="Recommended 24-36.")
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
    if meta_df.empty:
        raise SystemExit("No recognized files found. Example filename: CD98-High-PatientB_layer1.mat")

    marker_df, skipped_load = build_marker_group_table(meta_df, target_layer=args.target_layer)
    marker_df.to_csv(out_dir / "marker_group_patient_summary.csv", index=False, encoding="utf-8-sig")
    if not skipped_load.empty:
        skipped_load.to_csv(out_dir / "skipped_after_loading.csv", index=False, encoding="utf-8-sig")
    if marker_df.empty:
        raise SystemExit("No valid data rows after loading.")

    stage_mat = make_marker_stage_matrix(marker_df, value_col="mean_value")
    stage_z = row_zscore(stage_mat)
    top_mat, dynamic_score = select_top_dynamic_markers(stage_z, top_n=args.top_n_markers)
    top_ordered, _, Z = cluster_markers(top_mat)

    module_map = pd.DataFrame({
        "marker": top_ordered.index,
        "module": [assign_module(m) for m in top_ordered.index],
        "dynamic_score": [float(dynamic_score.loc[m]) for m in top_ordered.index],
    })

    top_ordered.to_csv(out_dir / "selected_marker_stage_zscore_matrix.csv", encoding="utf-8-sig")
    dynamic_score.to_csv(out_dir / "all_marker_dynamic_scores.csv", encoding="utf-8-sig", header=["dynamic_score"])
    module_map.to_csv(out_dir / "selected_marker_module_mapping.csv", index=False, encoding="utf-8-sig")

    plot_circular_cluster_figure(
        z_df_ordered=top_ordered,
        modules=module_map["module"].tolist(),
        Z=Z,
        out_dir=out_dir
    )

    params = vars(args)
    params["n_files_found"] = len(files)
    params["n_parsed_files"] = len(meta_df)
    params["n_valid_rows"] = len(marker_df)
    params["n_selected_markers"] = int(top_ordered.shape[0])

    with open(out_dir / "run_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    readme = [
        "Fig08 v7 no-outer-label version",
        f"Script: {script_name}.py",
        "",
        "Main changes:",
        "  1) all outer marker labels removed",
        "  2) main circular plot enlarged",
        "  3) only the bottom-left legends are retained",
        "",
        "Recommended settings:",
        "  --top-n-markers 30",
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
