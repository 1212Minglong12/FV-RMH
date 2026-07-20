# Data format

## 2D input

The 2D scripts recursively scan `.mat` files. The expected file-name pattern is:

```text
Marker-Group-PatientID_layerN.mat
```

De-identified example:

```text
CD98-High-P001_layer1.mat
```

Supported group aliases in the supplied scripts include variations of:

- `Normal` / `NAT` / `Healthy` / `Control`
- `High` / `WD` / `Well`
- `Middle` / `MD` / `Moderate`
- `Low` / `PD` / `Poor`

Use pseudonymous study identifiers such as `P001`; do not use names or hospital identifiers.

A `.mat` file should contain `coeffVector` when available. Otherwise, the scripts generally select the largest numeric array.

## 3D input

The primary 3D script recursively scans layer-resolved `.mat` files and recognizes pathology aliases including:

- cancer-adjacent / pneumonia / peritumoral
- lepidic / TieBi
- acinar / XianPao
- papillary
- micropapillary
- complex glands / FZXT
- solid

Layer numbers must be encoded near the end of the file stem, for example `_layer5.mat`.

## Source tables for downstream 3D analyses

The primary 3D integration script exports the two central tables required by most downstream scripts:

```text
source_embedding_coordinates.csv
source_true_pixel_robust_z_features.csv
```

Keep both tables together in the folder supplied through `--input-dir`.

## Data-release recommendation

For journal review, provide the minimum de-identified input dataset needed to execute representative workflows. Large raw hyperspectral volumes may be deposited in a data repository or made available through controlled access where ethically justified. The GitHub repository should contain code, metadata schemas, small example data, and source tables sufficient to verify the computational figures when permitted.
