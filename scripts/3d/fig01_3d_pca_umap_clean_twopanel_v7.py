#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fig01 v7: clean two-panel PCA + UMAP for 3D Raman middle-layer integration.

This version was rewritten after inspecting the previous output tables.

Why v7
------
The previous figure was not suitable for a paper because:
1) Acinar and Papillary were excluded by an overly strict pixel mask:
   a pixel was required to be nonzero in ~80% of markers;
2) zeros were treated as missing values, which removed valid low-signal tissue;
3) the layout was presentation-like rather than a clean journal figure.

v7 uses a quality-controlled central-layer strategy:
- Candidate layers are ranked by proximity to the geometric center of each 3D volume.
- The nearest usable central layers are selected (default: 3 layers).
- Zero values are retained as low Raman signal.
- A consensus tissue mask requires only a small number of positive marker signals
  (default: 2) and removes only global near-empty background.
- Each selected layer contributes an equal maximum number of pixel observations.
- Final figure has only two square, publication-style panels:
  A. PCA colored by selected central-layer position
  B. UMAP colored by pathology group

Important:
- The source files labelled "Pneumonia" are displayed as "Cancer-adjacent".
- Normal is excluded from the 40+ marker embedding because it has too few marker maps.
- No Harmony/patient batch correction claim is made because patient IDs are not available
  in this 3D folder.

Folder layout:
FV-RMH
    data_3d
    3dscripts
    3dresults

Run:
cd "FV-RMH"

python ".\\scripts\\3d\\fig01_3d_pca_umap_clean_twopanel_v7.py" ^
  --data-dir "data_3d" ^
  --out-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7"

If runtime is slow:
python ".\\scripts\\3d\\fig01_3d_pca_umap_clean_twopanel_v7.py" ^
  --data-dir "data_3d" ^
  --out-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7" ^
  --pixels-per-layer 500

For only one quality-controlled central layer:
python ".\\scripts\\3d\\fig01_3d_pca_umap_clean_twopanel_v7.py" ^
  --data-dir "data_3d" ^
  --out-dir "3dresults\\fig01_3d_pca_umap_clean_twopanel_v7_center1" ^
  --middle-layers 1
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


# Final manuscript display order requested by the user.
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
GROUP_COLORS = {
    "Cancer-adjacent": "#7A7A7A",
    "Lepidic": "#4C78A8",
    "Acinar": "#54A24B",
    "Papillary": "#F58518",
    "Micropapillary": "#E45756",
    "Complex glands": "#B279A2",
    "Solid": "#C92D39",
}
MIDDLE_COLORS = {
    "Central-1": "#4C78A8",
    "Central": "#54A24B",
    "Central+1": "#E45756",
}

GROUP_ALIASES = {
    "normal": "Normal",
    "healthy": "Normal",
    "control": "Normal",
    "nat": "Normal",
    # The original source label is deliberately shown as cancer-adjacent.
    "pneumonia": "Cancer-adjacent",
    "canceradjacent": "Cancer-adjacent",
    "cancer_adjacent": "Cancer-adjacent",
    "peritumoral": "Cancer-adjacent",
    "lepidic": "Lepidic",
    "tiebi": "Lepidic",
    "tiebie": "Lepidic",
    "acinar": "Acinar",
    "xianpao": "Acinar",
    "papillary": "Papillary",
    "micropapillary": "Micropapillary",
    "micropap": "Micropapillary",
    "complexglands": "Complex glands",
    "complexgland": "Complex glands",
    "complex_glands": "Complex glands",
    "fzxt": "Complex glands",
    "solid": "Solid",
    "soild": "Solid",
}


# ---------------------------------------------------------------------
# File parsing / .mat loading
# ---------------------------------------------------------------------
def norm_text(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def clean_marker_name(x: str) -> str:
    raw = str(x).strip()
    lookup = {
        "cdn2a": "CDKN2A",
        "cdkn2a": "CDKN2A",
        "p16": "CDKN2A",
        "acetylcoa": "Acetyl-CoA",
        "slactoylglutathione": "S-lactoylglutathione",
        "cholesterolester": "Cholesterol ester",
        "monounsaturatedlipid": "Monounsaturated lipid",
        "saturatedlipid": "Saturated lipid",
        "phosphoenolpyruvicacid": "Phosphoenolpyruvic acid",
        "3phosphoglycerate": "3-phosphoglycerate",
        "d3phosphoglycerate": "3-phosphoglycerate",
        "disodiumd3phosphoglycerate": "3-phosphoglycerate",
    }
    return lookup.get(norm_text(raw), raw.replace("_", "-"))


def parse_layer_from_stem(stem: str):
    match = re.search(r"_?layer[_-]?(\d+)$", stem, flags=re.IGNORECASE)
    if match:
        return stem[:match.start()], int(match.group(1))
    return stem, None


def parse_mat_filename(path: Path):
    stem, layer = parse_layer_from_stem(path.stem)
    if layer is None:
        return None

    parts = [p for p in re.split(r"[-_]+", stem) if p]
    if not parts:
        return None

    group = None
    group_idx = None
    for idx, part in enumerate(parts):
        key = norm_text(part)
        if key in GROUP_ALIASES:
            group = GROUP_ALIASES[key]
            group_idx = idx
            break

    if group is None:
        compact = norm_text(stem)
        for alias, canonical in sorted(GROUP_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if alias in compact:
                group = canonical
                break

    if group is None:
        return None

    marker_parts = parts[:group_idx] if group_idx is not None and group_idx > 0 else [parts[0]]
    source_group_label = "Pneumonia source label" if group == "Cancer-adjacent" else group

    return {
        "file": str(path),
        "filename": path.name,
        "marker": clean_marker_name("-".join(marker_parts)),
        "group": group,
        "source_group_label": source_group_label,
        "layer": int(layer),
    }


def read_metadata(data_dir: Path) -> pd.DataFrame:
    rows = []
    for path in data_dir.rglob("*.mat"):
        parsed = parse_mat_filename(path)
        if parsed is not None:
            rows.append(parsed)

    meta = pd.DataFrame(rows)
    if meta.empty:
        raise RuntimeError(f"No recognized .mat files found under: {data_dir.resolve()}")

    meta = meta[meta["group"].isin(GROUP_ORDER)].copy()
    meta["group"] = pd.Categorical(meta["group"], categories=GROUP_ORDER, ordered=True)
    meta = meta.sort_values(["group", "layer", "marker"]).reset_index(drop=True)

    if meta.empty:
        raise RuntimeError("No usable data files were found for the seven required groups.")
    return meta


def load_mat_vector(path: Path) -> np.ndarray:
    """Read coeffVector or the largest numeric matrix and flatten it."""
    try:
        mat = sio.loadmat(path)
        if "coeffVector" in mat:
            value = mat["coeffVector"]
        else:
            numeric = [
                (name, value)
                for name, value in mat.items()
                if not name.startswith("__")
                and isinstance(value, np.ndarray)
                and np.issubdtype(value.dtype, np.number)
            ]
            if not numeric:
                raise ValueError("No numeric MATLAB variable found")
            _, value = max(numeric, key=lambda item: item[1].size)
        return np.asarray(value, dtype=float).ravel()
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            if "coeffVector" in f:
                value = np.asarray(f["coeffVector"])
            else:
                numeric = []
                for key in f.keys():
                    try:
                        value = np.asarray(f[key])
                        if np.issubdtype(value.dtype, np.number):
                            numeric.append((key, value))
                    except Exception:
                        continue
                if not numeric:
                    raise ValueError("No numeric HDF5 variable found")
                _, value = max(numeric, key=lambda item: item[1].size)
        return np.asarray(value, dtype=float).ravel()


# ---------------------------------------------------------------------
# Central-layer quality control
# ---------------------------------------------------------------------
def common_markers(meta: pd.DataFrame):
    marker_sets = []
    for group in GROUP_ORDER:
        marker_sets.append(set(meta.loc[meta["group"].astype(str) == group, "marker"].astype(str)))
    return sorted(set.intersection(*marker_sets)) if marker_sets else []


def ranked_central_layers(meta: pd.DataFrame, group: str):
    layers = sorted(meta.loc[meta["group"].astype(str) == group, "layer"].astype(int).unique().tolist())
    if not layers:
        return []

    # Do not select the first / last layer when a reasonable number of layers exists.
    candidate_layers = layers[1:-1] if len(layers) >= 5 else layers
    centre = (len(layers) - 1) / 2.0

    indexed = [(layer, layers.index(layer), abs(layers.index(layer) - centre)) for layer in candidate_layers]
    indexed = sorted(indexed, key=lambda x: (x[2], x[0]))
    return [item[0] for item in indexed]


def prepare_layer_matrix(
    meta: pd.DataFrame,
    group: str,
    layer: int,
    markers: list[str],
    lower_percentile: float,
    upper_percentile: float,
    min_positive_markers: int,
    tissue_floor_percentile: float,
):
    """
    Returns a processed tissue-pixel matrix for one group/layer.

    Key methodological change from v6:
    - zeros are retained as biologically meaningful low/no Raman signal;
    - only a consensus tissue mask removes near-empty background;
    - maps are aligned as flattened spatial vectors, so an arbitrary 2D reshape
      is not needed for PCA/UMAP.
    """
    subset = meta[
        (meta["group"].astype(str) == group)
        & (meta["layer"].astype(int) == int(layer))
        & (meta["marker"].isin(markers))
    ].copy()

    vectors = {}
    lengths = []
    for _, item in subset.iterrows():
        try:
            vec = load_mat_vector(Path(item["file"]))
            if vec.size < 30:
                continue
            vectors[str(item["marker"])] = vec
            lengths.append(int(vec.size))
        except Exception:
            continue

    if len(vectors) < min_positive_markers:
        return None, {
            "group": group,
            "layer": layer,
            "status": "too_few_marker_vectors",
            "n_markers": len(vectors),
            "n_candidate_pixels": 0,
            "n_tissue_pixels": 0,
        }

    # Keep vectors at the modal spatial-vector length.
    modal_length = int(pd.Series(lengths).value_counts().index[0])
    vectors = {marker: vec for marker, vec in vectors.items() if vec.size == modal_length}
    usable_markers = [marker for marker in markers if marker in vectors]

    if len(usable_markers) < min_positive_markers:
        return None, {
            "group": group,
            "layer": layer,
            "status": "too_few_modal_length_markers",
            "n_markers": len(usable_markers),
            "n_candidate_pixels": modal_length,
            "n_tissue_pixels": 0,
        }

    raw = np.column_stack([vectors[marker] for marker in usable_markers]).astype(float)
    raw[~np.isfinite(raw)] = 0.0
    raw = np.maximum(raw, 0.0)

    # Per-marker clipping and scaling. Keep zeros untouched.
    transformed = np.zeros_like(raw, dtype=float)
    for j in range(raw.shape[1]):
        values = raw[:, j]
        positive = values[values > 0]
        if positive.size < 10:
            transformed[:, j] = 0.0
            continue

        lo, hi = np.percentile(positive, [lower_percentile, upper_percentile])
        if not np.isfinite(hi) or hi <= 0:
            transformed[:, j] = 0.0
            continue

        clipped = np.clip(values, max(0.0, lo), hi)
        clipped[values <= 0] = 0.0
        transformed[:, j] = np.log1p(clipped / hi)

    positive_count = (raw > 0).sum(axis=1)
    total_signal = transformed.sum(axis=1)

    candidate = positive_count >= int(min_positive_markers)
    if candidate.sum() < 30:
        return None, {
            "group": group,
            "layer": layer,
            "status": "too_few_consensus_pixels",
            "n_markers": len(usable_markers),
            "n_candidate_pixels": int(candidate.sum()),
            "n_tissue_pixels": 0,
        }

    floor = np.percentile(total_signal[candidate], tissue_floor_percentile)
    tissue = candidate & (total_signal >= floor)

    if tissue.sum() < 30:
        return None, {
            "group": group,
            "layer": layer,
            "status": "too_few_tissue_pixels",
            "n_markers": len(usable_markers),
            "n_candidate_pixels": int(candidate.sum()),
            "n_tissue_pixels": int(tissue.sum()),
        }

    # Build a full fixed marker matrix. If one of the common markers is absent in
    # this particular layer, retain a zero column rather than discard the layer.
    output = np.zeros((int(tissue.sum()), len(markers)), dtype=float)
    marker_to_col = {marker: idx for idx, marker in enumerate(markers)}
    for j, marker in enumerate(usable_markers):
        output[:, marker_to_col[marker]] = transformed[tissue, j]

    qc = {
        "group": group,
        "layer": int(layer),
        "status": "usable",
        "n_markers": len(usable_markers),
        "n_candidate_pixels": int(candidate.sum()),
        "n_tissue_pixels": int(tissue.sum()),
        "vector_length": modal_length,
        "quality_score": float(tissue.sum() * (len(usable_markers) / max(1, len(markers)))),
    }
    return output, qc


def choose_quality_controlled_central_layers(
    meta: pd.DataFrame,
    markers: list[str],
    n_layers: int,
    lower_percentile: float,
    upper_percentile: float,
    min_positive_markers: int,
    tissue_floor_percentile: float,
):
    """
    Select nearest valid central layers. If a central layer fails QC, use the next
    nearest usable central candidate rather than silently dropping the pathology group.
    """
    selected_data = []
    qc_rows = []

    for group in GROUP_ORDER:
        candidates = ranked_central_layers(meta, group)
        usable = []

        for central_rank, layer in enumerate(candidates, start=1):
            matrix, qc = prepare_layer_matrix(
                meta=meta,
                group=group,
                layer=layer,
                markers=markers,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                min_positive_markers=min_positive_markers,
                tissue_floor_percentile=tissue_floor_percentile,
            )
            qc["central_rank"] = central_rank

            if matrix is not None:
                usable.append((layer, matrix, qc))
            else:
                qc_rows.append(qc)

            if len(usable) >= n_layers:
                break

        if len(usable) == 0:
            raise RuntimeError(
                f"{group} has no quality-controlled central layer. "
                "Try --min-positive-markers 1 or lower --tissue-floor-percentile."
            )

        # Restore actual layer order for labelling and figure transparency.
        usable = sorted(usable, key=lambda item: item[0])
        if len(usable) == 1:
            labels = ["Central"]
        elif len(usable) == 2:
            labels = ["Central-1", "Central+1"]
        elif len(usable) == 3:
            labels = ["Central-1", "Central", "Central+1"]
        else:
            labels = [f"Central_{i+1}" for i in range(len(usable))]

        for selected_rank, ((layer, matrix, qc), label) in enumerate(zip(usable, labels), start=1):
            qc = dict(qc)
            qc["selected_rank"] = selected_rank
            qc["middle_position"] = label
            qc["status"] = "selected"
            qc_rows.append(qc)
            selected_data.append((group, int(layer), label, matrix, qc))

    qc_df = pd.DataFrame(qc_rows)
    return selected_data, qc_df


def build_balanced_pixel_matrix(selected_data, markers, pixels_per_layer, random_state):
    rng = np.random.default_rng(random_state)
    all_x = []
    all_meta = []
    balance_rows = []

    for group, layer, middle_position, matrix, qc in selected_data:
        n_available = matrix.shape[0]
        n_take = min(int(pixels_per_layer), int(n_available))
        choice = rng.choice(n_available, size=n_take, replace=False) if n_available > n_take else np.arange(n_available)
        sampled = matrix[choice, :]

        all_x.append(sampled)
        all_meta.append(pd.DataFrame({
            "group": group,
            "layer": int(layer),
            "middle_position": middle_position,
            "pixel_within_tissue": choice.astype(int),
            "n_markers_available": int(qc["n_markers"]),
        }))
        balance_rows.append({
            "group": group,
            "layer": int(layer),
            "middle_position": middle_position,
            "n_available_tissue_pixels": int(n_available),
            "n_selected_pixels": int(n_take),
            "n_markers_available": int(qc["n_markers"]),
        })

    X = np.vstack(all_x)
    pixel_meta = pd.concat(all_meta, ignore_index=True)
    balance = pd.DataFrame(balance_rows)

    # Marker-wise robust z-score across the balanced final matrix.
    Xz = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        values = X[:, j]
        median = np.nanmedian(values)
        q1, q3 = np.nanpercentile(values, [25, 75])
        scale = q3 - q1
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.nanstd(values)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        Xz[:, j] = (values - median) / scale

    X_df = pd.DataFrame(X, columns=markers)
    Xz_df = pd.DataFrame(Xz, columns=markers)
    return X_df, Xz_df, pixel_meta, balance


# ---------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------
def compute_embeddings(Xz_df: pd.DataFrame, pixel_meta: pd.DataFrame, n_pcs=20, random_state=1):
    if Xz_df.shape[0] < 10 or Xz_df.shape[1] < 2:
        raise RuntimeError("Insufficient final pixel observations or marker features for PCA/UMAP.")

    scaled = StandardScaler().fit_transform(Xz_df.values.astype(float))
    n_components = min(int(n_pcs), scaled.shape[1], scaled.shape[0] - 1)
    n_components = max(2, n_components)

    pca = PCA(n_components=n_components, random_state=random_state)
    pcs = pca.fit_transform(scaled)
    pc_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])

    method = "UMAP"
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=min(35, max(10, Xz_df.shape[0] // 100)),
            min_dist=0.20,
            n_components=2,
            metric="euclidean",
            random_state=random_state,
        )
        embedding = reducer.fit_transform(pc_df.values)
    except Exception:
        # A deterministic fallback that avoids very slow t-SNE on large pixel matrices.
        method = "PCA fallback (install umap-learn for UMAP)"
        embedding = pc_df[["PC1", "PC2"]].values

    coords = pixel_meta.copy().reset_index(drop=True)
    coords["pca_1"] = pc_df["PC1"].values
    coords["pca_2"] = pc_df["PC2"].values
    coords["umap_1"] = embedding[:, 0]
    coords["umap_2"] = embedding[:, 1]
    coords["embedding_method"] = method

    explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })
    return coords, explained, method


# ---------------------------------------------------------------------
# Clean publication-style plotting
# ---------------------------------------------------------------------
def save_multi(fig, out_dir: Path, stem: str, dpi=600, pad=0.02):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=pad)


def style_axis(ax, show_axis=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3)
    ax.grid(False)
    if not show_axis:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def plot_pca_layer_panel(ax, coords, with_text=True):
    order = ["Central-1", "Central", "Central+1"]
    present = [label for label in order if label in set(coords["middle_position"].astype(str))]

    for label in present:
        sub = coords[coords["middle_position"].astype(str) == label]
        ax.scatter(
            sub["pca_1"],
            sub["pca_2"],
            s=1.0,
            color=MIDDLE_COLORS.get(label, "#777777"),
            alpha=0.40,
            linewidths=0,
            rasterized=True,
            label=label,
        )

    if with_text:
        ax.set_title("PCA: selected central layers", loc="left", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        handles = [
            plt.Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=MIDDLE_COLORS[label],
                       markeredgecolor="none", markersize=7, label=label)
            for label in present
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    style_axis(ax, show_axis=with_text)


def plot_umap_pathology_panel(ax, coords, method, with_text=True):
    present = [group for group in GROUP_ORDER if group in set(coords["group"].astype(str))]

    for group in present:
        sub = coords[coords["group"].astype(str) == group]
        ax.scatter(
            sub["umap_1"],
            sub["umap_2"],
            s=1.0,
            color=GROUP_COLORS[group],
            alpha=0.48,
            linewidths=0,
            rasterized=True,
            label=GROUP_SHORT[group],
        )

    if with_text:
        title_prefix = "UMAP" if method == "UMAP" else "2D embedding"
        ax.set_title(f"{title_prefix}: pathology-associated molecular states", loc="left", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("UMAP1" if method == "UMAP" else "Embedding 1", fontsize=10)
        ax.set_ylabel("UMAP2" if method == "UMAP" else "Embedding 2", fontsize=10)
        handles = [
            plt.Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=GROUP_COLORS[group],
                       markeredgecolor="none", markersize=7, label=GROUP_SHORT[group])
            for group in present
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
    style_axis(ax, show_axis=with_text)


def make_main_figure(out_dir, coords, method):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.15), constrained_layout=True)
    plot_pca_layer_panel(axes[0], coords, with_text=True)
    plot_umap_pathology_panel(axes[1], coords, method=method, with_text=True)

    # Minimal manuscript-style panel tags; no text overlays inside the point clouds.
    axes[0].text(-0.12, 1.06, "a", transform=axes[0].transAxes, fontsize=14, fontweight="bold")
    axes[1].text(-0.12, 1.06, "b", transform=axes[1].transAxes, fontsize=14, fontweight="bold")

    save_multi(fig, out_dir, "Fig01_3D_clean_PCA_UMAP_twopanel_v7")
    plt.close(fig)


def export_single_panels(out_dir, coords, method):
    with_dir = out_dir / "single_panels_with_text"
    no_text_dir = out_dir / "single_panels_no_text"
    with_dir.mkdir(parents=True, exist_ok=True)
    no_text_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.3, 5.0))
    plot_pca_layer_panel(ax, coords, with_text=True)
    save_multi(fig, with_dir, "panel_a_pca_selected_central_layers")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    plot_umap_pathology_panel(ax, coords, method=method, with_text=True)
    save_multi(fig, with_dir, "panel_b_umap_pathology_states")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    plot_pca_layer_panel(ax, coords, with_text=False)
    save_multi(fig, no_text_dir, "panel_a_pca_selected_central_layers_no_text")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    plot_umap_pathology_panel(ax, coords, method=method, with_text=False)
    save_multi(fig, no_text_dir, "panel_b_umap_pathology_states_no_text")
    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Clean two-panel 3D Raman PCA/UMAP figure.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--middle-layers", type=int, default=3)
    parser.add_argument("--pixels-per-layer", type=int, default=700)
    parser.add_argument("--min-positive-markers", type=int, default=2)
    parser.add_argument("--tissue-floor-percentile", type=float, default=5.0)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_metadata(data_dir)
    markers = common_markers(meta)
    if len(markers) < 3:
        raise RuntimeError("Fewer than three shared markers were found across the seven groups.")

    selected_data, qc_summary = choose_quality_controlled_central_layers(
        meta=meta,
        markers=markers,
        n_layers=args.middle_layers,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        min_positive_markers=args.min_positive_markers,
        tissue_floor_percentile=args.tissue_floor_percentile,
    )

    selected_groups = {item[0] for item in selected_data}
    missing_groups = [group for group in GROUP_ORDER if group not in selected_groups]
    if missing_groups:
        raise RuntimeError(
            "These groups have no selected quality-controlled central layer: "
            + ", ".join(missing_groups)
            + ". Try --min-positive-markers 1 or --tissue-floor-percentile 1."
        )

    raw_df, z_df, pixel_meta, balance_summary = build_balanced_pixel_matrix(
        selected_data=selected_data,
        markers=markers,
        pixels_per_layer=args.pixels_per_layer,
        random_state=args.random_state,
    )

    coords, explained, method = compute_embeddings(
        z_df,
        pixel_meta,
        n_pcs=args.n_pcs,
        random_state=args.random_state,
    )

    # Full transparent source data outputs.
    meta.to_csv(out_dir / "source_parsed_file_metadata.csv", index=False)
    pd.DataFrame({"common_marker": markers}).to_csv(out_dir / "source_common_marker_list.csv", index=False)
    qc_summary.to_csv(out_dir / "source_central_layer_quality_control.csv", index=False)
    balance_summary.to_csv(out_dir / "source_selected_layer_pixel_balance.csv", index=False)
    raw_df.to_csv(out_dir / "source_true_pixel_log_scaled_features.csv", index=False)
    z_df.to_csv(out_dir / "source_true_pixel_robust_z_features.csv", index=False)
    pixel_meta.to_csv(out_dir / "source_true_pixel_metadata.csv", index=False)
    coords.to_csv(out_dir / "source_embedding_coordinates.csv", index=False)
    explained.to_csv(out_dir / "source_pca_explained_variance.csv", index=False)

    make_main_figure(out_dir, coords, method)
    export_single_panels(out_dir, coords, method)

    print("Done.")
    print(f"Output folder: {out_dir}")
    print(f"Selected groups: {', '.join(GROUP_ORDER)}")
    print(f"Shared markers: {z_df.shape[1]}")
    print(f"Pixel observations: {z_df.shape[0]}")
    print(f"Embedding method: {method}")
    print("Important QC file: source_central_layer_quality_control.csv")


if __name__ == "__main__":
    main()
