# Reproducibility notes

## Checks completed during repository preparation

1. The supplied archive contained 27 Python scripts: 12 in the 2D workflow and 15 in the 3D workflow.
2. Every script passed `python -m py_compile` after anonymization and path cleanup.
3. No password, GitHub credential, API key, or access token assignment was detected in the supplied files.
4. Potentially identifying example names were replaced with pseudonymous examples.
5. Workstation-specific `C:\Users\...` paths were removed from comments and usage text.

## Checks still required by the authors

Because input data were not included, these items cannot yet be certified:

- complete end-to-end execution;
- exact historical package versions;
- exact correspondence between every script and final manuscript panel;
- absence of identifiers in future data files and generated metadata tables;
- equivalence of the retained alternative Fig. 11 scripts to the final submitted figure;
- permissions for public release and license selection.

## Recommended pre-submission validation

1. Create a clean Python environment from `environment.yml`.
2. Run the primary 2D and 3D workflows on de-identified data.
3. Run all manuscript-final downstream scripts.
4. Compare exported figures and source-data tables with the submitted figures.
5. Freeze the verified environment:

```powershell
pip freeze > requirements-lock.txt
```

6. Create a GitHub release, for example `v1.0.0`, and archive that release in Zenodo to obtain a DOI.

## Syntax check command

```powershell
py .\tools\validate_repository.py
```
