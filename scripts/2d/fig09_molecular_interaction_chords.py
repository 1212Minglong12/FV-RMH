from pathlib import Path
import re
import json
import shutil
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path as MplPath

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

MODULE_ORDER = [
    "Extracellular Matrix",
    "Amino Acids",
    "Interleukins",
    "Calcification",
    "Infection & Immunity",
    "Hormones",
    "Carotenoids",
    "Other Metabolites",
    "Glycolipid Metabolism",
    "Vitamins",
    "Repair & Cytokines",
    "Oxidative Stress",
]

MODULE_COLORS = {
    "Extracellular Matrix": "#EAA39A",
    "Amino Acids": "#8CD1CC",
    "Interleukins": "#B0B8D9",
    "Calcification": "#C5CBE6",
    "Infection & Immunity": "#F3D3BE",
    "Hormones": "#D7C1E5",
    "Carotenoids": "#E7E0D8",
    "Other Metabolites": "#F08C8C",
    "Glycolipid Metabolism": "#CFC9C1",
    "Vitamins": "#D6D0C9",
    "Repair & Cytokines": "#F0D5A0",
    "Oxidative Stress": "#C9A8DA",
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

def build_summary(meta_df, target_layer=1):
    rows, skipped = [], []
    for _, row in meta_df.iterrows():
        if row["group"] not in GROUP_ORDER:
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
            "mean_value": float(np.nanmean(finite)),
            "percent_positive": float(np.mean(finite > 0) * 100.0),
            "n_values": int(finite.size),
            "module": assign_module(row["marker"]),
        })
    return pd.DataFrame(rows), pd.DataFrame(skipped)

def build_group_patient_matrices(summary_df):
    out = {}
    for grp, sub in summary_df.groupby("group"):
        mat = sub.groupby(["patient", "marker"], as_index=False)["mean_value"].mean().pivot(
            index="patient", columns="marker", values="mean_value"
        )
        if mat.shape[0] >= 2:
            mat = mat.loc[:, mat.notna().sum(axis=0) >= max(2, int(np.ceil(mat.shape[0] * 0.5)))]
        mat = mat.copy()
        for col in mat.columns:
            med = np.nanmedian(mat[col].values.astype(float))
            if not np.isfinite(med):
                med = 0.0
            mat[col] = mat[col].fillna(med)
        out[grp] = mat
    return out

def build_group_stage_matrix(summary_df):
    agg = summary_df.groupby(["marker", "group"], as_index=False)["mean_value"].mean()
    wide = agg.pivot(index="marker", columns="group", values="mean_value").reindex(columns=GROUP_ORDER)
    wide = wide.fillna(wide.mean(axis=1))
    wide = wide.fillna(0.0)
    return wide

def marker_dynamic_score(stage_mat):
    return stage_mat.std(axis=1) + (stage_mat.max(axis=1) - stage_mat.min(axis=1)) * 0.5

def pol2cart(r, theta):
    return r * np.cos(theta), r * np.sin(theta)

def make_equal_angles(n, gap_deg=4):
    gap = np.deg2rad(gap_deg)
    total_gap = n * gap
    usable = 2 * np.pi - total_gap
    width = usable / n
    angles = []
    current = np.pi / 2
    for _ in range(n):
        start = current
        end = current - width
        angles.append((start, end))
        current = end - gap
    return angles

def build_edge_list(corr_df, node_order, top_quantile=0.90, max_edges=25):
    vals = []
    for i in range(len(node_order)):
        for j in range(i + 1, len(node_order)):
            a, b = node_order[i], node_order[j]
            if a not in corr_df.index or b not in corr_df.columns:
                continue
            v = float(corr_df.loc[a, b])
            if np.isfinite(v):
                vals.append((a, b, v, abs(v)))
    if not vals:
        return []
    abs_arr = np.array([x[3] for x in vals], dtype=float)
    thresh = np.quantile(abs_arr, top_quantile)
    edges = [x for x in vals if x[3] >= thresh]
    edges = sorted(edges, key=lambda x: x[3], reverse=True)
    return edges[:max_edges]

def draw_outer_labels(ax, angle_map, labels, show_labels, radius=1.02, text_radius=1.28, fontsize=8.0):
    entries = []
    for lab in show_labels:
        th = angle_map[lab]
        x, y = pol2cart(radius, th)
        entries.append({"label": lab, "theta": th, "x": x, "y": y, "side": "right" if x >= 0 else "left"})
    if not entries:
        return

    def place_side(side_entries, side):
        side_entries = sorted(side_entries, key=lambda d: d["y"], reverse=True)
        if not side_entries:
            return
        ys = np.linspace(0.90, -0.90, len(side_entries))
        x_text = text_radius if side == "right" else -text_radius
        x_elbow = 1.09 if side == "right" else -1.09
        for target_y, item in zip(ys, side_entries):
            anchor_x, anchor_y = item["x"], item["y"]
            ax.plot([anchor_x, x_elbow], [anchor_y, target_y], color="#777777", lw=0.6, alpha=0.7, zorder=4)
            ha = "left" if side == "right" else "right"
            ax.text(x_text, target_y, item["label"], fontsize=fontsize, ha=ha, va="center", zorder=5)

    left = [e for e in entries if e["side"] == "left"]
    right = [e for e in entries if e["side"] == "right"]
    place_side(left, "left")
    place_side(right, "right")

def draw_chord_panel(ax, node_order, node_colors, corr_df, title,
                     label_style="horizontal_outside", show_labels=None,
                     edge_quantile=0.90, max_edges=25, fontsize=8.0):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.25, 1.25)

    arcs = make_equal_angles(len(node_order), gap_deg=4)
    angle_map = {node: (s + e) / 2 for node, (s, e) in zip(node_order, arcs)}

    # outer ring
    for node, (start, end) in zip(node_order, arcs):
        theta1 = np.degrees(end)
        theta2 = np.degrees(start)
        wedge = Wedge((0, 0), 1.00, theta1, theta2, width=0.085,
                      facecolor=node_colors[node], edgecolor="white", linewidth=1.0, alpha=0.98)
        ax.add_patch(wedge)

    # chords
    edges = build_edge_list(corr_df, node_order, top_quantile=edge_quantile, max_edges=max_edges)
    if edges:
        max_w = max(x[3] for x in edges)
    else:
        max_w = 1.0
    for a, b, corr, w in edges:
        th1 = angle_map[a]
        th2 = angle_map[b]
        p0 = np.array(pol2cart(0.90, th1))
        p3 = np.array(pol2cart(0.90, th2))
        c1 = p0 * 0.35
        c2 = p3 * 0.35
        verts = [p0, c1, c2, p3]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        color = node_colors[a]
        lw = 0.6 + 7.0 * (w / max_w)
        alpha = 0.10 + 0.35 * (w / max_w)
        patch = PathPatch(MplPath(verts, codes), facecolor="none", edgecolor=color, lw=lw, alpha=alpha, zorder=1)
        ax.add_patch(patch)

    # labels
    if show_labels is None:
        show_labels = node_order
    if label_style == "horizontal_outside":
        draw_outer_labels(ax, angle_map, node_order, show_labels, radius=1.02, text_radius=1.30, fontsize=fontsize)

    ax.text(0, 1.16, title, ha="center", va="bottom", fontsize=11, fontweight="bold")

def module_patient_matrix(group_matrix):
    if group_matrix.empty:
        return pd.DataFrame()
    marker_to_module = {c: assign_module(c) for c in group_matrix.columns}
    parts = []
    for mod in MODULE_ORDER:
        cols = [c for c in group_matrix.columns if marker_to_module[c] == mod]
        if cols:
            parts.append(group_matrix[cols].mean(axis=1).rename(mod))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1)
    return out

def save_panel(fig, out_path):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(str(out_path.with_suffix(f".{ext}")),
                    dpi=600 if ext == ".png" else None,
                    bbox_inches="tight", facecolor="white")

def main():
    parser = argparse.ArgumentParser(description="Figure 9 molecular interaction evolution chord diagram")
    parser.add_argument("--data-dir", type=str, default="data_2d")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--target-layer", type=int, default=1)
    parser.add_argument("--top-markers", type=int, default=28, help="How many global markers to keep for the bottom-row interactome.")
    parser.add_argument("--global-labels", type=int, default=16, help="How many marker labels to show in each bottom panel.")
    parser.add_argument("--module-edge-quantile", type=float, default=0.84)
    parser.add_argument("--global-edge-quantile", type=float, default=0.985)
    parser.add_argument("--module-max-edges", type=int, default=18)
    parser.add_argument("--global-max-edges", type=int, default=60)
    parser.add_argument("--groups", nargs="*", default=None, help="Optional group order, e.g. --groups Normal High Middle Low")
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
        raise SystemExit("No recognized files found.")

    summary_df, skipped_load = build_summary(meta_df, target_layer=args.target_layer)
    summary_df.to_csv(out_dir / "marker_group_patient_summary.csv", index=False, encoding="utf-8-sig")
    if not skipped_load.empty:
        skipped_load.to_csv(out_dir / "skipped_after_loading.csv", index=False, encoding="utf-8-sig")
    if summary_df.empty:
        raise SystemExit("No valid data rows after loading.")

    group_mats = build_group_patient_matrices(summary_df)
    stage_mat = build_group_stage_matrix(summary_df)
    dyn_score = marker_dynamic_score(stage_mat).sort_values(ascending=False)

    groups_available = [g for g in GROUP_ORDER if g in group_mats]
    if args.groups is not None and len(args.groups) > 0:
        groups_available = [g for g in args.groups if g in group_mats]
    if not groups_available:
        raise SystemExit("No usable groups found.")

    # select top markers for global interactome
    candidate_markers = [m for m in dyn_score.index if any(m in group_mats[g].columns for g in groups_available)]
    top_markers = candidate_markers[:args.top_markers]

    # persist top markers table
    pd.DataFrame({
        "marker": top_markers,
        "dynamic_score": [float(dyn_score.loc[m]) for m in top_markers],
        "module": [assign_module(m) for m in top_markers],
    }).to_csv(out_dir / "selected_global_markers.csv", index=False, encoding="utf-8-sig")

    # combined figure
    n = len(groups_available)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 10.2))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    legend_handles = []

    for col, grp in enumerate(groups_available):
        mat = group_mats[grp]

        # ---------- top row: module network ----------
        mod_mat = module_patient_matrix(mat)
        if mod_mat.shape[0] >= 3 and mod_mat.shape[1] >= 3:
            mod_corr = mod_mat.corr(method="spearman")
            module_nodes = [m for m in MODULE_ORDER if m in mod_corr.columns]
        else:
            module_nodes = [m for m in MODULE_ORDER if m in [assign_module(c) for c in mat.columns]]
            mod_corr = pd.DataFrame(np.eye(len(module_nodes)), index=module_nodes, columns=module_nodes)

        module_colors = {m: MODULE_COLORS[m] for m in module_nodes}
        draw_chord_panel(
            axes[0, col],
            node_order=module_nodes,
            node_colors=module_colors,
            corr_df=mod_corr,
            title=f"Metabolic Network: {GROUP_DISPLAY.get(grp, grp)}",
            label_style="horizontal_outside",
            show_labels=module_nodes,
            edge_quantile=args.module_edge_quantile,
            max_edges=args.module_max_edges,
            fontsize=8.0,
        )

        # ---------- bottom row: global interactome ----------
        marker_nodes = [m for m in top_markers if m in mat.columns]
        if len(marker_nodes) >= 4 and mat.shape[0] >= 3:
            gmat = mat[marker_nodes].copy()
            gcorr = gmat.corr(method="spearman")
        else:
            gcorr = pd.DataFrame(np.eye(len(marker_nodes)), index=marker_nodes, columns=marker_nodes)

        marker_strength = gcorr.abs().sum(axis=1).sort_values(ascending=False)
        # order markers by module, then within module by strength
        marker_nodes_sorted = []
        for mod in MODULE_ORDER:
            cols = [m for m in marker_nodes if assign_module(m) == mod]
            cols = sorted(cols, key=lambda x: float(marker_strength.get(x, 0.0)), reverse=True)
            marker_nodes_sorted.extend(cols)

        marker_colors = {m: MODULE_COLORS[assign_module(m)] for m in marker_nodes_sorted}
        label_subset = marker_strength.head(args.global_labels).index.tolist()

        draw_chord_panel(
            axes[1, col],
            node_order=marker_nodes_sorted,
            node_colors=marker_colors,
            corr_df=gcorr,
            title=f"Global Interactome: {GROUP_DISPLAY.get(grp, grp)}",
            label_style="horizontal_outside",
            show_labels=label_subset,
            edge_quantile=args.global_edge_quantile,
            max_edges=args.global_max_edges,
            fontsize=7.0,
        )

        # save per-panel data
        mod_corr.to_csv(out_dir / f"module_correlation_{grp}.csv", encoding="utf-8-sig")
        gcorr.to_csv(out_dir / f"global_correlation_{grp}.csv", encoding="utf-8-sig")

        # individual panels
        f1, ax1 = plt.subplots(figsize=(6.2, 5.8))
        draw_chord_panel(
            ax1, module_nodes, module_colors, mod_corr,
            title=f"Metabolic Network: {GROUP_DISPLAY.get(grp, grp)}",
            label_style="horizontal_outside",
            show_labels=module_nodes,
            edge_quantile=args.module_edge_quantile,
            max_edges=args.module_max_edges,
            fontsize=8.0,
        )
        save_panel(f1, out_dir / f"panel_top_module_network_{grp}")
        plt.close(f1)

        f2, ax2 = plt.subplots(figsize=(6.8, 6.2))
        draw_chord_panel(
            ax2, marker_nodes_sorted, marker_colors, gcorr,
            title=f"Global Interactome: {GROUP_DISPLAY.get(grp, grp)}",
            label_style="horizontal_outside",
            show_labels=label_subset,
            edge_quantile=args.global_edge_quantile,
            max_edges=args.global_max_edges,
            fontsize=7.0,
        )
        save_panel(f2, out_dir / f"panel_bottom_global_interactome_{grp}")
        plt.close(f2)

    fig.suptitle("Molecular interaction evolution chord map", fontsize=16, fontweight="bold", y=0.985)

    # global legend
    handles = []
    labels = []
    for mod in MODULE_ORDER:
        handles.append(plt.Line2D([0], [0], color=MODULE_COLORS[mod], lw=6))
        labels.append(mod)

    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.965])

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"Fig9_molecular_interaction_chord_map.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    params = vars(args)
    params["groups_used"] = groups_available
    params["n_files_found"] = len(files)
    params["n_summary_rows"] = len(summary_df)

    with open(out_dir / "run_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    readme = [
        "Figure 9 molecular interaction evolution chord map",
        f"Script: {script_name}.py",
        "",
        "Outputs:",
        "  - Fig9_molecular_interaction_chord_map.png/pdf/svg",
        "  - Top-row individual module-network panels",
        "  - Bottom-row individual global-interactome panels",
        "  - Correlation matrices for each group",
        "",
        "Design notes:",
        "  1) all labels are horizontal",
        "  2) labels are kept outside the circle",
        "  3) only a subset of marker labels is shown in the bottom row to reduce crowding",
        "",
        "Recommended run:",
        "  python scripts/2d/fig09_molecular_interaction_chords.py --data-dir data_2d --top-markers 28 --global-labels 16",
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
