# MOSAIS-electrochemical-prediction

Machine-learning code for electrochemical-property prediction and candidate ranking of MOF-polymer solid-state electrolytes within the MOSAIS framework.

## Overview

This repository contains the review-ready **Model-1** workflow used for multi-objective discovery of MOF-polymer solid-state electrolytes (MSPEs). The current implementation is provided as a **single Python script** for transparency and reproducibility.

The workflow supports:

- separate supervised regression models for three electrochemical targets
- optional self-supervised denoising autoencoder preprocessing
- 4-fold cross-validation with a fixed 20% held-out test split
- log-transformed modeling for ionic conductivity at 25 °C
- model-family comparison across tree-based and kernel methods
- lightweight Bayesian hyperparameter optimization for shortlisted models
- residual-based prediction intervals and lower confidence bounds
- multi-objective ranking using **entropy-weighted TOPSIS** and **Pareto frontier** analysis
- export of summary tables, candidate rankings, and publication-style figures

## Current repository status

At present, this repository is organized as a **single-file implementation** of the MOSAIS Model-1 workflow. This is intentional: the script is kept compact and readable for reviewer-facing use.

Current main script:

- `MOSAIS_multi_objective_discovery_for editor and reviewers.py`

For cleaner long-term maintenance, it is recommended to rename this file later to something simpler, such as:

- `MOSAIS_model1_workflow.py`

## Predicted targets

The script builds separate models for the following targets:

- `Conductivity_25C`
- `Li_Transfer`
- `E_Window`

## Input files

The workflow expects two Excel files in the working directory (or another location that the script can detect):

- `ML_clean.xlsx` — training dataset with descriptors and target values
- `ML_input.xlsx` — candidate dataset for prediction and ranking

If these files are not present, the script will stop with a file-not-found error.

## Main outputs

By default, the script writes results to:

- `out_MOSAIS_Model1_review_ready/`

Typical outputs include:

- `metrics_summary_all_targets.xlsx`
- candidate ranking tables for `ML_input.xlsx`
- parity plots for train-fit and cross-validation predictions
- feature-importance plots
- SHAP summary plots
- embedding maps and 3D prediction-space visualizations

## Requirements

### Core dependencies

The workflow uses standard scientific Python packages, including:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `openpyxl`
- `lightgbm`

### Optional dependencies

The script can also use the following packages when available:

- `xgboost`
- `shap`
- `umap-learn`
- `torch`

`lightgbm` is required. The other optional packages enable additional model families or visualization/representation-learning functionality.

## Installation

Create and activate your Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` may look like this:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
openpyxl
lightgbm
xgboost
shap
umap-learn
torch
```

## How to run

### Option 1: run the file using its current name

```bash
python "MOSAIS_multi_objective_discovery_for editor and reviewers.py"
```

### Option 2: rename it first, then run

```bash
python MOSAIS_model1_workflow.py
```

## Suggested repository structure

A simple first-version repository structure is:

```text
MOSAIS-electrochemical-prediction/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ MOSAIS_model1_workflow.py
├─ ML_clean.xlsx              # optional: add only if appropriate to share
├─ ML_input.xlsx              # optional: add only if appropriate to share
└─ out_MOSAIS_Model1_review_ready/
```

If you do not want to upload the full Excel datasets yet, that is completely fine. In that case, keep the code in the repository and add the data files later or provide sample/template input files.

## Notes for repository development

This repository is currently private and intended as a clean code base for the MOSAIS electrochemical prediction module. A sensible next step is:

1. keep the single-file implementation as the first stable version
2. add `requirements.txt`
3. improve documentation
4. optionally split the workflow into multiple scripts later, such as data preprocessing, model training, prediction, ranking, and plotting

## Scope

This repository focuses on the **Model-1 electrochemical prediction and candidate-ranking workflow**. It is not intended to contain every intermediate experiment, temporary script, or unpublished internal asset.

## Citation and reuse

If this code supports a manuscript submission, it is recommended to align the public repository contents with the final published version of the workflow, figures, and data-sharing policy.
