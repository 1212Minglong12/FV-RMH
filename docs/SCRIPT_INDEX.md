# Script index

All commands below are run from the repository root. Use `py <script> --help` for complete options.

## 2D scripts

| Script | Purpose |
|---|---|
| `fig01_harmony_umap_nested2d.py` | 2D multi-sample integration, denoising, optional Harmony batch correction, PCA and UMAP. |
| `fig02_umap_kde_by_grade_panels.py` | 2D group-specific UMAP and KDE density panels by differentiation grade. |
| `fig03_dominant_metabolic_module_umap.py` | 2D dominant molecular-module assignment on the integrated UMAP. |
| `fig04_metabolic_module_feature_mapping.py` | 2D mapping of individual module scores onto the shared manifold. |
| `fig05_data_overview_heatmap_dotplot_v3_balanced_spacing.py` | 2D marker heatmap and dot-plot overview. |
| `fig06_module_enrichment_dynamics_v3_avoid_blank_modules.py` | 2D module enrichment heatmap and dynamics summaries. |
| `fig07_metabolic_fingerprint_alluvial_radar_v4_no_text_overlap.py` | 2D radar and alluvial summaries of module fingerprints. |
| `fig08_circular_cluster_module_heatmap_v7_no_outer_labels_bigplot.py` | 2D circular clustered marker/module heatmap. |
| `fig09_molecular_interaction_chords.py` | 2D molecular/module correlation chord diagrams. |
| `fig10_spatiotemporal_trajectory_metabolic_cascade.py` | 2D computational ordering and feature cascade heatmap. |
| `fig12_top10_marker_allinone.py` | 2D top-marker integrated visualization and statistics. |
| `paper_model_01_2d_HML_classifier.py` | Specimen/patient-level exploratory well–moderate–poor differentiation classifier. |

### Common 2D commands

```powershell
py .\scripts\2d\fig01_harmony_umap_nested2d.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig01_harmony_umap_nested2d"
py .\scripts\2d\fig02_umap_kde_by_grade_panels.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig02_umap_kde_by_grade_panels"
py .\scripts\2d\fig03_dominant_metabolic_module_umap.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig03_dominant_metabolic_module_umap"
py .\scripts\2d\fig04_metabolic_module_feature_mapping.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig04_metabolic_module_feature_mapping"
py .\scripts\2d\fig05_data_overview_heatmap_dotplot_v3_balanced_spacing.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig05_data_overview_heatmap_dotplot_v3_balanced_spacing"
py .\scripts\2d\fig06_module_enrichment_dynamics_v3_avoid_blank_modules.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig06_module_enrichment_dynamics_v3_avoid_blank_modules"
py .\scripts\2d\fig07_metabolic_fingerprint_alluvial_radar_v4_no_text_overlap.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig07_metabolic_fingerprint_alluvial_radar_v4_no_text_overlap"
py .\scripts\2d\fig08_circular_cluster_module_heatmap_v7_no_outer_labels_bigplot.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig08_circular_cluster_module_heatmap_v7_no_outer_labels_bigplot"
py .\scripts\2d\fig09_molecular_interaction_chords.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig09_molecular_interaction_chords"
py .\scripts\2d\fig10_spatiotemporal_trajectory_metabolic_cascade.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig10_spatiotemporal_trajectory_metabolic_cascade"
py .\scripts\2d\fig12_top10_marker_allinone.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig12_top10_marker_allinone"
py .\scripts\2d\paper_model_01_2d_HML_classifier.py --data-dir ".\data\2d" --out-dir ".\results\2d\paper_model_01_2d_HML_classifier"
```

## 3D scripts

| Script | Purpose |
|---|---|
| `fig01_3d_pca_umap_clean_twopanel_v7.py` | Primary 3D central-layer integration; exports PCA/UMAP coordinates and robust marker features. |
| `fig02_3d_umap_pathology_density_drift.py` | 3D pathology-specific UMAP views and density contours. |
| `fig03_3d_dominant_module_umap_flow.py` | 3D dominant-module UMAP/flow visualization. |
| `fig04_3d_module_fingerprint_standalone_v3.py` | 3D module fingerprint summary. |
| `fig05_3d_true_shap_marker_driver_triptych_v3_clean.py` | 3D supervised feature attribution and SHAP driver visualization. |
| `fig06_3d_module_score_umap_atlas.py` | 3D module-score atlas on the shared UMAP. |
| `fig07_3d_marker_heatmap_dotplot.py` | 3D marker-level heatmap and dot plot. |
| `fig08_3d_module_heatmap_dotplot.py` | 3D module-level heatmap and dot plot. |
| `fig09b_luad_marker_chord_one_row_realdata.py` | 3D pathology-specific marker correlation chord diagrams. |
| `fig11_3d_spatiotemporal_trajectory_pseudotime.py` | 3D graph/pathology-ordered molecular trajectory and heatmap. |
| `fig11_3d_spatiotemporal_trajectory_pseudotime_v2_curve_tech.py` | Alternative curved/technical trajectory visualization retained for provenance. |
| `fig12_3d_layerlevel_module_correlation_pairgrid_v3_paperclean.py` | Layer/specimen-aggregated module correlation pair grid. |
| `figE_core_raman_program_fingerprint_radar_realdata_v2.py` | Real-data group-level molecular-program radar plot. |
| `fig_ridgeplot_raman_realdata.py` | Ridge plots for marker or module distributions. |
| `fig_top8_marker_violin_box_swarm_paperstyle_v6.py` | Violin/box/jitter plots with nonparametric marker statistics. |

### Primary 3D integration

```powershell
py .\scripts\3d\fig01_3d_pca_umap_clean_twopanel_v7.py --data-dir ".\data\3d" --out-dir ".\results\3d\fig01_integration" --pixels-per-layer 2000
```

### Downstream 3D commands

Most downstream scripts use `--input-dir .\results\3d\fig01_integration`.

```powershell
py .\scripts\3d\fig02_3d_umap_pathology_density_drift.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig02_3d_umap_pathology_density_drift"
py .\scripts\3d\fig03_3d_dominant_module_umap_flow.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig03_3d_dominant_module_umap_flow"
py .\scripts\3d\fig04_3d_module_fingerprint_standalone_v3.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig04_3d_module_fingerprint_standalone_v3"
py .\scripts\3d\fig05_3d_true_shap_marker_driver_triptych_v3_clean.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig05_3d_true_shap_marker_driver_triptych_v3_clean"
py .\scripts\3d\fig06_3d_module_score_umap_atlas.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig06_3d_module_score_umap_atlas"
py .\scripts\3d\fig07_3d_marker_heatmap_dotplot.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig07_3d_marker_heatmap_dotplot"
py .\scripts\3d\fig08_3d_module_heatmap_dotplot.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig08_3d_module_heatmap_dotplot"
py .\scripts\3d\fig09b_luad_marker_chord_one_row_realdata.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig09b_luad_marker_chord_one_row_realdata"
py .\scripts\3d\fig11_3d_spatiotemporal_trajectory_pseudotime.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig11_3d_spatiotemporal_trajectory_pseudotime"
py .\scripts\3d\fig11_3d_spatiotemporal_trajectory_pseudotime_v2_curve_tech.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig11_3d_spatiotemporal_trajectory_pseudotime_v2_curve_tech"
py .\scripts\3d\fig12_3d_layerlevel_module_correlation_pairgrid_v3_paperclean.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig12_3d_layerlevel_module_correlation_pairgrid_v3_paperclean"
py .\scripts\3d\figE_core_raman_program_fingerprint_radar_realdata_v2.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\figE_core_raman_program_fingerprint_radar_realdata_v2"
py .\scripts\3d\fig_ridgeplot_raman_realdata.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig_ridgeplot_raman_realdata"
py .\scripts\3d\fig_top8_marker_violin_box_swarm_paperstyle_v6.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig_top8_marker_violin_box_swarm_paperstyle_v6"
```

> Some scripts provide additional required or optional arguments. Check `--help` before execution. The alternative Fig. 11 trajectory versions are both retained; select the manuscript-final version before DOI archiving.
