# FV-RMH 代码仓库说明

对应论文：

**Volumetric Raman molecular histology enables multiplexed chemical mapping of fresh lung tumours**

本仓库已经把你提供的 27 个 Python 脚本按 `2d` 和 `3d` 两部分整理，并完成以下公开前处理：

- 删除注释和报错示例中的电脑绝对路径；
- 将可能包含真实姓名的示例患者名替换为 `PatientA`、`PatientB`；
- 保留原分析逻辑和原脚本文件名；
- 增加安装依赖、数据格式、脚本索引、运行命令和代码可用性声明；
- 全部脚本已通过 Python 语法编译检查。

## 你需要放入的数据

- 二维 `.mat` 数据放入 `data/2d/`；
- 三维 `.mat` 数据放入 `data/3d/`；
- 不要上传患者姓名、住院号、身份证号、电话号码、密码或 Token。

## Windows 安装

在仓库根目录打开 PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2D 示例命令

```powershell
py .\scripts\2d\fig01_harmony_umap_nested2d.py --data-dir ".\data\2d" --out-dir ".\results\2d\fig01_harmony_umap"
```

分类模型：

```powershell
py .\scripts\2d\paper_model_01_2d_HML_classifier.py --data-dir ".\data\2d" --out-dir ".\results\2d\HML_classifier"
```

## 3D 示例命令

先运行主整合脚本：

```powershell
py .\scripts\3d\fig01_3d_pca_umap_clean_twopanel_v7.py --data-dir ".\data\3d" --out-dir ".\results\3d\fig01_integration" --pixels-per-layer 2000
```

再运行下游图形脚本，例如：

```powershell
py .\scripts\3d\fig05_3d_true_shap_marker_driver_triptych_v3_clean.py --input-dir ".\results\3d\fig01_integration" --out-dir ".\results\3d\fig05_shap"
```

更多命令见 `docs/SCRIPT_INDEX.md`。

## 当前限制

你这次只上传了代码，没有上传原始 `.mat` 数据或中间 CSV，因此我可以确认代码语法和仓库结构，但不能在这里完整复现论文中的全部图。正式公开前，建议你在原电脑上按 README 跑一次，并将验证成功的软件版本固定下来。
