# MOSAIS-electrochemical-prediction

Machine-learning code for electrochemical-property prediction and candidate ranking of MOF-polymer solid-state electrolytes within the MOSAIS framework.

## What this repository is for

This repository is used to organize and preserve the **code side** of the MOSAIS electrochemical-prediction workflow.

At the current stage, the repository contains two different layers of code:

1. **A review-ready main workflow** for multi-objective electrochemical prediction and candidate ranking.
2. **An early exploration script** for single-target model screening under small-sample conditions.

This structure is intentional. The main workflow represents the more mature code path, while the exploration script records the earlier model-selection logic and trial-and-error process.

## Current code files

### 1. Main workflow script

Recommended file name:

- `src/MOSAIS_model1_workflow.py`

Current function:

- builds separate models for the three electrochemical targets
- supports candidate prediction and ranking
- supports entropy-weighted TOPSIS and Pareto analysis
- exports summary tables and publication-style figures

Target defaults:

- `Conductivity_25C`
- `Li_Transfer`
- `E_Window`

This script expects:

- `ML_clean.xlsx`
- `ML_input.xlsx`

### 2. Early exploration script

Recommended file name:

- `exploration/01_single_target_model_screening.py`

Current function:

- benchmarks multiple regression models for each single target independently
- compares model suitability under a unified cross-validation protocol
- outputs tables, rankings, and figures for model screening

Current model pool:

- Ridge
- SVR
- KNN
- RandomForest
- ExtraTrees
- GradientBoosting
- XGBoost
- LightGBM
- CatBoost

Current evaluation metrics:

- CV R²
- CV RMSE
- CV MAE
- Train R²
- Overfit gap

This script is mainly used to answer a practical early-stage question:

> For a given electrochemical target under small-sample conditions, which algorithm is the most suitable baseline model for later refinement?

---

## Suggested repository structure

A clean first-version structure is:

```text
MOSAIS-electrochemical-prediction/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ exploration/
│  └─ 01_single_target_model_screening.py
├─ src/
│  └─ MOSAIS_model1_workflow.py
├─ data/
│  ├─ sample/
│  └─ README.md
└─ outputs/
```

If you are not ready to upload the Excel data files yet, that is completely fine. You can keep the code first and add data later, or only provide sample/template inputs.

---

## Input files

### For the main workflow

Expected input files:

- `ML_clean.xlsx`: training dataset with descriptors and target values
- `ML_input.xlsx`: candidate dataset for prediction and ranking

### For the early exploration script

Expected input file:

- `ML_clean.xlsx`

The exploration script can optionally check `ML_input.xlsx` to align shared descriptor columns when present, but it can still run without it.

---

## Main outputs

### Main workflow outputs

Typical outputs include:

- overall model-performance summaries
- candidate ranking tables
- parity plots
- feature-importance plots
- SHAP plots
- embedding / projection figures
- multi-objective ranking results

### Single-target screening outputs

Typical outputs include:

- per-target model ranking tables
- overall benchmark summary tables
- best-model OOF prediction tables
- CV R² ranking figures
- parity plots for the best screened model
- heatmap-style summary figure across targets and algorithms

---

## Installation

Create and activate your Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` is designed to cover both the main workflow and the early model-screening script.

---

## How to run

### Run the early exploration script

```bash
python exploration/01_single_target_model_screening.py
```

### Run the main workflow script

```bash
python src/MOSAIS_model1_workflow.py
```

If your current file names are still the older ones, you can run those names directly first and rename them later after the repository structure is cleaned up.

---

## Notes on dependencies

### Required core packages

These packages are expected for normal use:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `openpyxl`
- `lightgbm`

### Additional packages used in this repository

These packages extend functionality for model families, interpretation, or representation learning:

- `xgboost`
- `catboost`
- `shap`
- `umap-learn`
- `torch`

If some optional packages are not installed, certain models or extra figures may be skipped depending on the script implementation.

---

## Recommended use strategy

For your current stage, a sensible workflow is:

1. use `01_single_target_model_screening.py` to benchmark baseline algorithms for each target;
2. identify the best or most stable model family for each property;
3. refine the shortlisted models;
4. then move to the integrated multi-objective workflow.

This makes the repository more transparent and also preserves the logic of how the final workflow was reached.

---

## Scope of this repository

This repository is intended to store:

- cleaned and interpretable research code
- early exploration scripts with clear methodological value
- documentation needed to rerun the workflow

This repository is **not** intended to store every temporary script, every intermediate failed attempt, or every unpublished internal asset.

---

## Simple Chinese note

这个仓库目前建议理解成两个层次：

- `src/`：较正式、较成熟的 MOSAIS 主流程代码
- `exploration/`：早期探索与试错代码，记录“为什么先选这些算法、怎么比较、最后为什么走向后续多目标流程”

对你现在这个阶段来说，最合理的做法不是把所有历史脚本全都扔进去，而是：

1. 保留真正有方法学价值的早期探索脚本；
2. 保留较成熟的主流程脚本；
3. 用 README 把两者之间的关系讲清楚。

这样这个仓库会更像一个规范的科研代码仓库，而不是一个杂乱的脚本堆放处。
