# -*- coding: utf-8 -*-
"""
01_single_target_model_screening.py

Early-stage MOSAIS baseline script for small-sample single-target model screening.

What this script does
---------------------
1. Reads the training table (default: ML_clean.xlsx).
2. Automatically selects usable numeric descriptor columns.
3. Benchmarks candidate regression models for each target independently.
4. Reports unified metrics:
   - CV R2
   - CV RMSE
   - CV MAE
   - Train R2
   - Overfit gap (= Train R2 - CV R2)
5. Saves ranking tables, best-model OOF prediction tables, and publication-style figures.

Target defaults are aligned with the current MOSAIS workflow:
- Conductivity_25C
- Li_Transfer
- E_Window

Notes
-----
- This is an early exploration / model-screening script, not the final multi-objective workflow.
- Conductivity_25C is log10-transformed during fitting by default, but all reported metrics
  are returned on the original scale for easier interpretation.
- XGBoost / LightGBM / CatBoost are optional. If a package is not installed, that model is skipped.
"""

from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


# =============================================================================
# Configuration
# =============================================================================
FILE_CLEAN = "ML_clean.xlsx"
FILE_INPUT_OPTIONAL = "ML_input.xlsx"  # used only to align shared descriptor columns when present

TARGETS = ["Conductivity_25C", "Li_Transfer", "E_Window"]
TARGET_TRANSFORMS = {
    "Conductivity_25C": "log10",
    "Li_Transfer": "identity",
    "E_Window": "identity",
}

DROP_ALWAYS = {"Cycle_Life_Score"}
DROP_SALT_DESCRIPTORS = {"Anion_HOMO_eV", "Lattice_Energy_kJ_mol", "Anion_Radius_nm"}

RANDOM_SEED = 42
CV_N_SPLITS = 5
MIN_SAMPLES_FOR_MODELING = 8

OUT_DIR = "out_single_target_model_screening"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
PRED_DIR = os.path.join(OUT_DIR, "predictions")

SAVE_EXCEL = True
FIG_DPI = 600


# =============================================================================
# Small utilities
# =============================================================================
def locate_file(filename: str, required: bool = True) -> Optional[str]:
    """Search for a file in common local locations."""
    candidates = [os.path.join(os.getcwd(), filename)]
    if "__file__" in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates += [
            os.path.join(script_dir, filename),
            os.path.join(os.path.dirname(script_dir), filename),
        ]
    home = os.path.expanduser("~")
    candidates += [
        os.path.join(home, "Desktop", filename),
        os.path.join(home, "桌面", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    if required:
        raise FileNotFoundError(
            f"Could not locate {filename}. Searched:\n  - " + "\n  - ".join(candidates)
        )
    return None


def make_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metric_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return {"R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan")}
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def set_pub_style() -> None:
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.linewidth"] = 1.2
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.major.width"] = 1.0
    mpl.rcParams["ytick.major.width"] = 1.0
    mpl.rcParams["savefig.dpi"] = FIG_DPI


def savefig_close(fig, outpath: str) -> None:
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def safe_sheet_name(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_")[:31]


def clip_predictions(target_name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if target_name == "Conductivity_25C":
        return np.clip(arr, 1e-12, None)
    if target_name == "Li_Transfer":
        return np.clip(arr, 0.0, 1.0)
    if target_name == "E_Window":
        return np.clip(arr, 0.0, None)
    return arr


# =============================================================================
# Target transform
# =============================================================================
class TargetTransformer:
    def __init__(self, kind: str = "identity"):
        self.kind = (kind or "identity").lower()

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.kind == "log10":
            if np.any(y <= 0):
                raise ValueError("log10 transform requested, but non-positive values were found in the target.")
            return np.log10(y)
        return y

    def inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.kind == "log10":
            return np.power(10.0, z)
        return z


# =============================================================================
# Feature selection
# =============================================================================
def pick_feature_columns(df_clean: pd.DataFrame, df_input: Optional[pd.DataFrame] = None) -> List[str]:
    """Pick numeric descriptor columns usable for model screening."""
    df_clean = df_clean.copy()
    df_clean = ensure_numeric(df_clean, df_clean.columns.tolist())

    num_cols_clean = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in num_cols_clean if c not in set(TARGETS)]
    candidates = [c for c in candidates if c not in DROP_ALWAYS]
    candidates = [c for c in candidates if c not in DROP_SALT_DESCRIPTORS]
    candidates = [c for c in candidates if not df_clean[c].isna().all()]

    keep = []
    for col in candidates:
        if df_clean[col].dropna().nunique() <= 1:
            continue
        if ("Anion_" in col) or ("Lattice_Energy" in col):
            continue
        keep.append(col)

    if df_input is not None:
        common = [c for c in keep if c in df_input.columns]
        if len(common) >= 5:
            return common
    return keep


# =============================================================================
# Model registry
# =============================================================================
@dataclass
class ModelSpec:
    name: str
    builder: Callable[[], Any]
    enabled: bool = True
    note: str = ""


def build_model_registry() -> List[ModelSpec]:
    models: List[ModelSpec] = [
        ModelSpec(
            name="Ridge",
            builder=lambda: Ridge(alpha=1.0, random_state=RANDOM_SEED),
            enabled=True,
        ),
        ModelSpec(
            name="SVR",
            builder=lambda: SVR(C=10.0, epsilon=0.05, kernel="rbf"),
            enabled=True,
        ),
        ModelSpec(
            name="KNN",
            builder=lambda: KNeighborsRegressor(n_neighbors=5, weights="distance", p=2),
            enabled=True,
        ),
        ModelSpec(
            name="RandomForest",
            builder=lambda: RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            enabled=True,
        ),
        ModelSpec(
            name="ExtraTrees",
            builder=lambda: ExtraTreesRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            enabled=True,
        ),
        ModelSpec(
            name="GradientBoosting",
            builder=lambda: GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_SEED,
            ),
            enabled=True,
        ),
        ModelSpec(
            name="XGBoost",
            builder=lambda: XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=0,
            ),
            enabled=HAS_XGB,
            note="Skipped if xgboost is not installed.",
        ),
        ModelSpec(
            name="LightGBM",
            builder=lambda: LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=0.0,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1,
            ),
            enabled=HAS_LGBM,
            note="Skipped if lightgbm is not installed.",
        ),
        ModelSpec(
            name="CatBoost",
            builder=lambda: CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function="RMSE",
                eval_metric="RMSE",
                random_seed=RANDOM_SEED,
                verbose=False,
                allow_writing_files=False,
            ),
            enabled=HAS_CATBOOST,
            note="Skipped if catboost is not installed.",
        ),
    ]
    return models


def make_pipeline(model_spec: ModelSpec) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model_spec.builder()),
        ]
    )


# =============================================================================
# CV evaluation
# =============================================================================
def get_cv_splitter(n_samples: int) -> KFold:
    n_splits = min(CV_N_SPLITS, n_samples)
    if n_splits < 3:
        raise ValueError(f"Too few samples ({n_samples}) for reliable CV. Need at least 3 usable rows.")
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)


def evaluate_single_model_cv(
    X_df: pd.DataFrame,
    y_raw: np.ndarray,
    target_name: str,
    model_spec: ModelSpec,
    return_oof: bool = False,
) -> Dict[str, Any]:
    transformer = TargetTransformer(TARGET_TRANSFORMS.get(target_name, "identity"))
    y_model = transformer.transform(y_raw)

    cv = get_cv_splitter(len(X_df))
    pipeline = make_pipeline(model_spec)

    fold_rows: List[Dict[str, Any]] = []
    oof_pred_raw = np.full(len(X_df), np.nan, dtype=float)

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_df), start=1):
        X_tr = X_df.iloc[tr_idx]
        X_va = X_df.iloc[va_idx]
        y_tr_raw = y_raw[tr_idx]
        y_va_raw = y_raw[va_idx]
        y_tr_model = y_model[tr_idx]

        pipe = clone(pipeline)
        pipe.fit(X_tr, y_tr_model)

        pred_va_raw = clip_predictions(target_name, transformer.inverse(pipe.predict(X_va)))
        pred_tr_raw = clip_predictions(target_name, transformer.inverse(pipe.predict(X_tr)))

        oof_pred_raw[va_idx] = pred_va_raw

        rep_cv = metric_report(y_va_raw, pred_va_raw)
        rep_train = metric_report(y_tr_raw, pred_tr_raw)

        fold_rows.append(
            {
                "Fold": fold_id,
                "CV_R2": rep_cv["R2"],
                "CV_RMSE": rep_cv["RMSE"],
                "CV_MAE": rep_cv["MAE"],
                "Train_R2": rep_train["R2"],
                "Train_RMSE": rep_train["RMSE"],
                "Train_MAE": rep_train["MAE"],
                "Overfit_gap": rep_train["R2"] - rep_cv["R2"],
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    overall_oof = metric_report(y_raw, oof_pred_raw)

    result = {
        "Model": model_spec.name,
        "Target": target_name,
        "N_samples": int(len(X_df)),
        "CV_R2": float(overall_oof["R2"]),
        "CV_RMSE": float(overall_oof["RMSE"]),
        "CV_MAE": float(overall_oof["MAE"]),
        "Train_R2": float(fold_df["Train_R2"].mean()),
        "Train_RMSE": float(fold_df["Train_RMSE"].mean()),
        "Train_MAE": float(fold_df["Train_MAE"].mean()),
        "Overfit_gap": float(fold_df["Overfit_gap"].mean()),
        "CV_R2_fold_mean": float(fold_df["CV_R2"].mean()),
        "CV_R2_fold_std": float(fold_df["CV_R2"].std(ddof=0)),
        "CV_RMSE_fold_mean": float(fold_df["CV_RMSE"].mean()),
        "CV_RMSE_fold_std": float(fold_df["CV_RMSE"].std(ddof=0)),
        "CV_MAE_fold_mean": float(fold_df["CV_MAE"].mean()),
        "CV_MAE_fold_std": float(fold_df["CV_MAE"].std(ddof=0)),
        "fold_metrics": fold_df,
    }
    if return_oof:
        result["oof_pred_raw"] = oof_pred_raw
    return result


# =============================================================================
# Plotting
# =============================================================================
def plot_target_ranking(target_name: str, ranking_df: pd.DataFrame, outpath: str) -> None:
    import matplotlib.pyplot as plt

    set_pub_style()
    df_plot = ranking_df.sort_values("CV_R2", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.45 * len(df_plot) + 1.2)))
    ax.barh(df_plot["Model"], df_plot["CV_R2"], edgecolor="black", linewidth=0.7)
    ax.set_xlabel("CV $R^2$", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(f"{target_name}: single-target model screening", fontsize=13)
    ax.tick_params(labelsize=10)
    savefig_close(fig, outpath)



def plot_best_model_parity(
    target_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    cv_r2: float,
    cv_rmse: float,
    cv_mae: float,
    outpath: str,
) -> None:
    import matplotlib.pyplot as plt

    set_pub_style()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    vmin = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    vmax = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    pad = 0.04 * (vmax - vmin) if vmax > vmin else 1.0

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(y_true, y_pred, s=32, edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], "-", linewidth=1.2)
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xlabel("Observed", fontsize=12)
    ax.set_ylabel("OOF predicted", fontsize=12)
    ax.set_title(f"{target_name}: best model parity", fontsize=13)
    ax.tick_params(labelsize=10)

    txt = (
        f"Model: {model_name}\n"
        f"CV $R^2$ = {cv_r2:.3f}\n"
        f"CV RMSE = {cv_rmse:.4g}\n"
        f"CV MAE = {cv_mae:.4g}"
    )
    ax.text(
        0.04,
        0.96,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9),
    )
    savefig_close(fig, outpath)



def plot_summary_heatmap(best_df: pd.DataFrame, outpath: str) -> None:
    import matplotlib.pyplot as plt

    set_pub_style()
    if best_df.empty:
        return

    pivot = best_df.pivot(index="Target", columns="Metric", values="Value")
    row_order = list(best_df["Target"].drop_duplicates())
    col_order = [c for c in ["CV_R2", "CV_RMSE", "CV_MAE", "Train_R2", "Overfit_gap"] if c in pivot.columns]
    pivot = pivot.reindex(index=row_order, columns=col_order)

    data = pivot.values.astype(float)
    fig, ax = plt.subplots(figsize=(1.8 * len(col_order) + 1.8, 0.8 * len(row_order) + 1.8))
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=10)
    ax.set_title("Best-model metric summary", fontsize=13)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    savefig_close(fig, outpath)


# =============================================================================
# Main workflow helpers
# =============================================================================
def benchmark_one_target(
    df_clean: pd.DataFrame,
    feature_cols: List[str],
    target_name: str,
    model_specs: List[ModelSpec],
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray, np.ndarray]:
    if target_name not in df_clean.columns:
        raise KeyError(f"Target column '{target_name}' not found in training table.")

    use_cols = feature_cols + [target_name]
    sub = ensure_numeric(df_clean.copy(), use_cols)[use_cols].copy()
    sub = sub.loc[~sub[target_name].isna()].copy()

    if len(sub) < MIN_SAMPLES_FOR_MODELING:
        raise ValueError(
            f"Target '{target_name}' has only {len(sub)} usable rows after dropping NaN values; "
            f"need at least {MIN_SAMPLES_FOR_MODELING}."
        )

    X_df = sub[feature_cols].copy()
    y_raw = sub[target_name].astype(float).values

    result_rows: List[Dict[str, Any]] = []
    detailed_results: Dict[str, Any] = {}

    for spec in model_specs:
        if not spec.enabled:
            print(f"[SKIP] {spec.name}: {spec.note}")
            continue
        print(f"  - Evaluating {spec.name} ...")
        res = evaluate_single_model_cv(X_df, y_raw, target_name, spec, return_oof=False)
        result_rows.append({k: v for k, v in res.items() if k != "fold_metrics"})
        detailed_results[spec.name] = res

    if not result_rows:
        raise RuntimeError(
            f"No models were successfully evaluated for target '{target_name}'. "
            f"Please check package availability and data quality."
        )

    ranking_df = pd.DataFrame(result_rows).sort_values(
        by=["CV_R2", "CV_RMSE", "CV_MAE", "Overfit_gap"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    ranking_df.insert(0, "Rank", np.arange(1, len(ranking_df) + 1))

    best_model_name = ranking_df.loc[0, "Model"]
    best_spec = next(spec for spec in model_specs if spec.name == best_model_name and spec.enabled)
    best_full = evaluate_single_model_cv(X_df, y_raw, target_name, best_spec, return_oof=True)

    return ranking_df, best_full, X_df.values, y_raw


# =============================================================================
# Main driver
# =============================================================================
def main() -> None:
    print("=" * 92)
    print("MOSAIS | Early-stage single-target model screening")
    print("=" * 92)

    make_dirs(OUT_DIR, TABLE_DIR, FIG_DIR, PRED_DIR)

    clean_path = locate_file(FILE_CLEAN, required=True)
    input_path = locate_file(FILE_INPUT_OPTIONAL, required=False)

    print(f"Training file : {clean_path}")
    print(f"Input file    : {input_path if input_path else '[not found; feature selection will use ML_clean only]'}")
    print(f"Output folder : {os.path.abspath(OUT_DIR)}")

    df_clean = pd.read_excel(clean_path)
    df_input = pd.read_excel(input_path) if input_path else None

    for drop_col in list(DROP_ALWAYS):
        for df_ in [df_clean] + ([df_input] if df_input is not None else []):
            if df_ is not None and drop_col in df_.columns:
                df_.drop(columns=[drop_col], inplace=True)

    feature_cols = pick_feature_columns(df_clean, df_input)
    if len(feature_cols) < 5:
        raise RuntimeError(
            "Too few usable feature columns after preprocessing. Please check the descriptor table."
        )

    model_specs = build_model_registry()

    print("\nSelected feature count:", len(feature_cols))
    print("Targets:", TARGETS)
    print("Available models:", [m.name for m in model_specs if m.enabled])
    if any(not m.enabled for m in model_specs):
        print("Unavailable models:", [m.name for m in model_specs if not m.enabled])

    with open(os.path.join(TABLE_DIR, "feature_columns_used.txt"), "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(str(col) + "\n")

    config_dump = {
        "FILE_CLEAN": FILE_CLEAN,
        "FILE_INPUT_OPTIONAL": FILE_INPUT_OPTIONAL,
        "TARGETS": TARGETS,
        "TARGET_TRANSFORMS": TARGET_TRANSFORMS,
        "DROP_ALWAYS": sorted(list(DROP_ALWAYS)),
        "DROP_SALT_DESCRIPTORS": sorted(list(DROP_SALT_DESCRIPTORS)),
        "RANDOM_SEED": RANDOM_SEED,
        "CV_N_SPLITS": CV_N_SPLITS,
        "MIN_SAMPLES_FOR_MODELING": MIN_SAMPLES_FOR_MODELING,
        "available_models": [m.name for m in model_specs if m.enabled],
        "unavailable_models": [m.name for m in model_specs if not m.enabled],
    }
    with open(os.path.join(TABLE_DIR, "screening_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2, ensure_ascii=False)

    all_rankings: List[pd.DataFrame] = []
    best_rows: List[Dict[str, Any]] = []
    heatmap_rows: List[Dict[str, Any]] = []

    excel_path = os.path.join(TABLE_DIR, "single_target_screening_summary.xlsx")
    writer = pd.ExcelWriter(excel_path, engine="openpyxl") if SAVE_EXCEL else None

    try:
        for target_name in TARGETS:
            print("\n" + "-" * 92)
            print(f"Target: {target_name}")
            ranking_df, best_res, _, y_raw = benchmark_one_target(df_clean, feature_cols, target_name, model_specs)

            ranking_df.insert(1, "Target", target_name)
            all_rankings.append(ranking_df)

            best_model_name = str(best_res["Model"])
            print(f"Best model for {target_name}: {best_model_name}")
            print(
                f"  CV R2 = {best_res['CV_R2']:.4f}, "
                f"CV RMSE = {best_res['CV_RMSE']:.4g}, "
                f"CV MAE = {best_res['CV_MAE']:.4g}, "
                f"Train R2 = {best_res['Train_R2']:.4f}, "
                f"Overfit gap = {best_res['Overfit_gap']:.4f}"
            )

            # save ranking table
            target_tag = target_name.replace("Conductivity_25C", "sigma")
            csv_rank = os.path.join(TABLE_DIR, f"{target_tag}_model_ranking.csv")
            ranking_df.to_csv(csv_rank, index=False, encoding="utf-8-sig")
            if writer is not None:
                ranking_df.to_excel(writer, index=False, sheet_name=safe_sheet_name(f"{target_tag}_ranking"))

            # save fold-level metrics for best model
            fold_df = best_res["fold_metrics"].copy()
            fold_df.insert(0, "Target", target_name)
            fold_df.insert(1, "Model", best_model_name)
            csv_fold = os.path.join(TABLE_DIR, f"{target_tag}_best_model_fold_metrics.csv")
            fold_df.to_csv(csv_fold, index=False, encoding="utf-8-sig")
            if writer is not None:
                fold_df.to_excel(writer, index=False, sheet_name=safe_sheet_name(f"{target_tag}_folds"))

            # save OOF predictions for best model
            oof_df = pd.DataFrame(
                {
                    "Observed": y_raw,
                    "OOF_Predicted": best_res["oof_pred_raw"],
                    "Residual": y_raw - best_res["oof_pred_raw"],
                }
            )
            csv_oof = os.path.join(PRED_DIR, f"{target_tag}_best_model_oof_predictions.csv")
            oof_df.to_csv(csv_oof, index=False, encoding="utf-8-sig")
            if writer is not None:
                oof_df.to_excel(writer, index=False, sheet_name=safe_sheet_name(f"{target_tag}_oof"))

            # save figures
            fig_rank = os.path.join(FIG_DIR, f"{target_tag}_cv_r2_ranking.png")
            fig_parity = os.path.join(FIG_DIR, f"{target_tag}_best_model_parity.png")
            plot_target_ranking(target_name, ranking_df, fig_rank)
            plot_best_model_parity(
                target_name=target_name,
                y_true=y_raw,
                y_pred=best_res["oof_pred_raw"],
                model_name=best_model_name,
                cv_r2=best_res["CV_R2"],
                cv_rmse=best_res["CV_RMSE"],
                cv_mae=best_res["CV_MAE"],
                outpath=fig_parity,
            )

            best_rows.append(
                {
                    "Target": target_name,
                    "Best_Model": best_model_name,
                    "CV_R2": best_res["CV_R2"],
                    "CV_RMSE": best_res["CV_RMSE"],
                    "CV_MAE": best_res["CV_MAE"],
                    "Train_R2": best_res["Train_R2"],
                    "Overfit_gap": best_res["Overfit_gap"],
                    "N_samples": best_res["N_samples"],
                }
            )
            for metric_name in ["CV_R2", "CV_RMSE", "CV_MAE", "Train_R2", "Overfit_gap"]:
                heatmap_rows.append(
                    {
                        "Target": target_name,
                        "Metric": metric_name,
                        "Value": best_res[metric_name],
                    }
                )

        # combined outputs
        all_summary = pd.concat(all_rankings, axis=0, ignore_index=True)
        best_summary = pd.DataFrame(best_rows)
        heatmap_df = pd.DataFrame(heatmap_rows)

        all_csv = os.path.join(TABLE_DIR, "benchmark_summary_all_targets.csv")
        best_csv = os.path.join(TABLE_DIR, "best_model_summary.csv")
        all_summary.to_csv(all_csv, index=False, encoding="utf-8-sig")
        best_summary.to_csv(best_csv, index=False, encoding="utf-8-sig")

        if writer is not None:
            all_summary.to_excel(writer, index=False, sheet_name="all_rankings")
            best_summary.to_excel(writer, index=False, sheet_name="best_models")
            heatmap_df.to_excel(writer, index=False, sheet_name="best_metric_long")

        plot_summary_heatmap(heatmap_df, os.path.join(FIG_DIR, "best_model_metric_summary.png"))

    finally:
        if writer is not None:
            writer.close()

    print("\n" + "=" * 92)
    print("Single-target screening finished.")
    print(f"Tables      : {os.path.abspath(TABLE_DIR)}")
    print(f"Figures     : {os.path.abspath(FIG_DIR)}")
    print(f"Predictions : {os.path.abspath(PRED_DIR)}")
    print("=" * 92)


if __name__ == "__main__":
    main()
