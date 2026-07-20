# FV-RMH

Code associated with the manuscript:

**Volumetric Raman molecular histology enables multiplexed chemical mapping of fresh lung tumours**

## Overview

This repository contains the custom Python scripts used for two-dimensional and three-dimensional Raman data integration, visualization, molecular-module analysis, feature attribution, computational molecular ordering, and exploratory differentiation classification.

The repository has been reorganized for public sharing. The numerical analysis logic in the supplied scripts has been retained. Workstation-specific paths and potentially identifying example patient names were removed from comments and example messages.

## Repository structure

```text
FV-RMH/
├── scripts/
│   ├── 2d/                 # 2D Raman analyses and differentiation classifier
│   └── 3d/                 # 3D integration and downstream analyses
├── data/
│   ├── 2d/                 # place de-identified 2D .mat files here
│   └── 3d/                 # place de-identified 3D .mat files here
├── results/                # generated outputs
├── docs/                   # data format, script index, and reproducibility notes
├── requirements.txt
├── environment.yml
└── CITATION.cff
```

## Installation

Python 3.10 or 3.11 is recommended.

### Using pip

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Using conda

```bash
conda env create -f environment.yml
conda activate fvrmh
```

## Data

Human Raman data are **not included** in this code-only repository. Before public release, only de-identified files should be added. Do not upload patient names, hospital numbers, dates of birth, contact information, access credentials, or other direct identifiers.

See [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for expected file naming and folder conventions.

## Minimal workflow

Run commands from the repository root.

### 2D analyses

The 2D scripts recursively scan `.mat` files in `data/2d`. A typical command is:

```powershell
py .\scripts\2d\fig01_harmony_umap_nested2d.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig01_harmony_umap"
```

Exploratory differentiation classifier:

```powershell
py .\scripts\2d\paper_model_01_2d_HML_classifier.py --data-dir ".\data\2d" --out-dir ".\results\2d\HML_classifier"
```

Use `py <script> --help` to inspect the available options for any script.

### 3D analyses

First generate the integrated PCA/UMAP source tables:

```powershell
py .\scripts\3d\fig01_3d_pca_umap_clean_twopanel_v7.py --data-dir ".\data\3d" --out-dir ".\results\3d\fig01_integration" --pixels-per-layer 2000
```

Most downstream 3D scripts use the source CSV files created by the first script:

- `source_embedding_coordinates.csv`
- `source_true_pixel_robust_z_features.csv`

Example downstream command:

```powershell
py .\scripts\3d\fig05_3d_true_shap_marker_driver_triptych_v3_clean.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig05_shap"
```

Additional commands are listed in [`docs/SCRIPT_INDEX.md`](docs/SCRIPT_INDEX.md).

## Reproducibility status

- All 27 supplied Python scripts pass Python syntax compilation.
- Random seeds are exposed in the principal scripts where stochastic analyses are used.
- Full end-to-end execution was not possible in this code-only package because the Raman input matrices and source CSV files were not included.
- Package versions in `requirements.txt` are compatibility ranges, not the exact historical analysis environment. Before journal release, run the complete workflow once on the authors' workstation and freeze the verified environment.

See [`docs/REPRODUCIBILITY_NOTES.md`](docs/REPRODUCIBILITY_NOTES.md).

## Code availability statement

Suggested manuscript wording is provided in [`docs/CODE_AVAILABILITY.md`](docs/CODE_AVAILABILITY.md).

## License

No open-source license has been selected yet. The authors and relevant institutions/company should approve a license before public release. See [`LICENSE_PENDING.md`](LICENSE_PENDING.md).

## Contact

Please contact the corresponding authors listed in the manuscript for scientific questions regarding the study and code.
