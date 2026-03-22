# -*- coding: utf-8 -*-
"""
MOSAIS_multi_objective_discovery_review_ready.py

Review-ready, single-file implementation of the MOSAIS Model-1 workflow for
MOF–polymer solid-state electrolytes (MSPEs).

This script is designed to align with the manuscript description by providing:
1. Separate supervised regression models for Conductivity_25C, Li_Transfer, and E_Window.
2. Optional self-supervised denoising autoencoder (DAE) preprocessing with unchanged
   descriptor dimensionality.
3. A 4-fold cross-validation workflow with a fixed 20% held-out test split.
4. log10 target transformation for Conductivity_25C during model fitting.
5. Candidate algorithm comparison (ExtraTrees, LightGBM, XGBoost, RandomForest, SVR).
6. Lightweight Bayesian hyperparameter optimization for shortlisted model families.
7. Out-of-fold residual-based prediction intervals and lower confidence bounds (LCBs).
8. Multi-objective prioritization using entropy-weighted TOPSIS and Pareto frontier analysis.
9. Export of model metrics, candidate rankings, and publication-style figures.

Notes
-----
- This file intentionally focuses on the final MOSAIS Model-1 workflow only.
- The code is written for transparency and reviewer readability rather than maximum speed.
- If optional dependencies are unavailable, the script falls back gracefully where possible.
"""

from __future__ import annotations

import hashlib
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# =============================================================================
# Configuration
# =============================================================================
FILE_CLEAN = "ML_clean.xlsx"
FILE_INPUT = "ML_input.xlsx"
TARGETS = ["Conductivity_25C", "Li_Transfer", "E_Window"]
DROP_ALWAYS = {"Cycle_Life_Score"}
DROP_SALT_DESCRIPTORS = {"Anion_HOMO_eV", "Lattice_Energy_kJ_mol", "Anion_Radius_nm"}

RANDOM_SEED = 42
FROZEN_TEST_SIZE = 0.20
N_SPLITS_CV = 4
OUT_DIR = "out_MOSAIS_Model1_review_ready"

CONFORMAL_ALPHA = 0.10  # ~90% interval
CV_STABILITY_PENALTY = 0.05
SHORTLIST_FAMILIES = 2
BO_INIT_POINTS = 4
BO_ITERATIONS = 4
BO_CANDIDATE_POOL = 256

PREFER_UMAP = True
SHAP_MAX_SAMPLES = 600
SHAP_MAX_DISPLAY = 20

USE_SSL_PRETRAIN = True
SSL_METHOD = "dae"  # "dae", "pca", "none"
SSL_LATENT_DIM = 8
SSL_HIDDEN_DIM = 32
SSL_EPOCHS = 250
SSL_BATCH_SIZE = 32
SSL_NOISE_STD = 0.05
SSL_LR = 1e-3
SSL_WEIGHT_DECAY = 1e-5
SSL_VERBOSE = False

TARGET_TRANSFORMS = {
    "Conductivity_25C": "log10",
    "Li_Transfer": "identity",
    "E_Window": "identity",
}


# =============================================================================
# Small utilities
# =============================================================================
def locate_file(filename: str) -> str:
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
    raise FileNotFoundError(
        f"Could not locate {filename}. Searched:\n  - " + "\n  - ".join(candidates)
    )


def make_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def stable_row_id_features_only(df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    """Stable row identifiers built from feature values only."""
    tmp = ensure_numeric(df.copy(), feature_cols)[feature_cols].copy()
    tmp = tmp.fillna(np.nan)

    def _fmt(v: Any) -> str:
        if pd.isna(v):
            return "NA"
        try:
            return f"{float(v):.12g}"
        except Exception:
            return str(v)

    rows = tmp.apply(lambda r: "|".join(_fmt(v) for v in r.values), axis=1).values
    md5 = [hashlib.md5(s.encode("utf-8")).hexdigest() for s in rows]
    return pd.Series(md5, index=df.index, name="row_id")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metric_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {"R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan")}
    return {
        "R2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
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
    mpl.rcParams["xtick.major.width"] = 1.1
    mpl.rcParams["ytick.major.width"] = 1.1
    mpl.rcParams["xtick.minor.width"] = 0.9
    mpl.rcParams["ytick.minor.width"] = 0.9
    mpl.rcParams["savefig.dpi"] = 600


def savefig(fig, outpath: str) -> None:
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    fig.clf()


# =============================================================================
# Target transforms
# =============================================================================
class TargetTransformer:
    """Simple target transform wrapper used for per-target model fitting."""

    def __init__(self, kind: str):
        self.kind = (kind or "identity").lower()

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.kind == "log10":
            if np.any(y <= 0):
                raise ValueError("log10 transform requested, but non-positive target values were found.")
            return np.log10(y)
        return y

    def inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.kind == "log10":
            return np.power(10.0, z)
        return z

    def lower_bound(self, z_pred: np.ndarray, qhat: float) -> np.ndarray:
        z_pred = np.asarray(z_pred, dtype=float)
        if self.kind == "log10":
            return np.power(10.0, z_pred - qhat)
        return z_pred - qhat

    def upper_bound(self, z_pred: np.ndarray, qhat: float) -> np.ndarray:
        z_pred = np.asarray(z_pred, dtype=float)
        if self.kind == "log10":
            return np.power(10.0, z_pred + qhat)
        return z_pred + qhat


# =============================================================================
# Feature selection
# =============================================================================
def pick_feature_columns(df_clean: pd.DataFrame, df_input: pd.DataFrame) -> List[str]:
    """Pick numeric descriptor columns shared by training and candidate tables."""
    num_cols_clean = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in num_cols_clean if c not in set(TARGETS)]
    candidates = [c for c in candidates if c not in DROP_ALWAYS]
    candidates = [c for c in candidates if c not in DROP_SALT_DESCRIPTORS]
    candidates = [c for c in candidates if not df_clean[c].isna().all()]

    keep = []
    for col in candidates:
        if df_clean[col].dropna().nunique() <= 1:
            continue
        keep.append(col)

    final = []
    for col in keep:
        if ("Anion_" in col) or ("Lattice_Energy" in col):
            continue
        final.append(col)

    common = [c for c in final if c in df_input.columns]
    return common


# =============================================================================
# Optional self-supervised DAE/PCA mapper
# =============================================================================
class SSLEncoder(BaseEstimator, TransformerMixin):
    """
    Optional self-supervised feature mapper.

    The transformer reconstructs the descriptor matrix while preserving the same
    output dimensionality as the input. This keeps downstream interpretability
    (feature names, importance, SHAP/permutation plots) aligned with the original
    descriptor set used in the manuscript.
    """

    def __init__(
        self,
        method: str = "dae",
        latent_dim: int = 8,
        hidden_dim: int = 32,
        epochs: int = 250,
        batch_size: int = 32,
        noise_std: float = 0.05,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.method = method
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.noise_std = float(noise_std)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.model_ = None
        self.pca_ = None
        self.mu_ = None
        self.sig_ = None

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        mu = np.nanmean(X, axis=0)
        sig = np.nanstd(X, axis=0)
        sig = np.where(np.isfinite(sig) & (sig > 1e-12), sig, 1.0)
        self.mu_ = mu.astype(np.float32)
        self.sig_ = sig.astype(np.float32)
        Xs = (X - self.mu_) / self.sig_
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return Xs.astype(np.float32)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mu_) / self.sig_
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return Xs.astype(np.float32)

    def _unstandardize(self, Xs: np.ndarray) -> np.ndarray:
        return (Xs * self.sig_ + self.mu_).astype(np.float32)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] < 2:
            self.method = "none"
            return self

        Xs = self._standardize_fit(X)
        method = (self.method or "none").lower()
        if method == "none":
            return self

        if method == "pca" or (method == "dae" and not HAS_TORCH):
            n_comp = int(max(1, min(self.latent_dim, Xs.shape[1] - 1)))
            self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
            self.pca_.fit(Xs)
            return self

        if method == "dae" and HAS_TORCH:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_feat = Xs.shape[1]
            latent_dim = int(max(1, min(self.latent_dim, n_feat - 1)))
            hidden_dim = int(max(latent_dim + 1, self.hidden_dim))

            class _DAE(nn.Module):
                def __init__(self, in_dim: int, h_dim: int, z_dim: int):
                    super().__init__()
                    self.enc = nn.Sequential(
                        nn.Linear(in_dim, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, z_dim),
                    )
                    self.dec = nn.Sequential(
                        nn.Linear(z_dim, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, in_dim),
                    )

                def forward(self, x):
                    z = self.enc(x)
                    return self.dec(z)

            model = _DAE(n_feat, hidden_dim, latent_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            loss_fn = nn.MSELoss()
            X_tensor = torch.from_numpy(Xs).to(device)
            n = int(X_tensor.shape[0])
            bs = int(max(8, min(self.batch_size, n)))
            rng = np.random.RandomState(self.random_state)

            for epoch in range(self.epochs):
                idx = torch.from_numpy(rng.permutation(n)).to(device)
                model.train()
                epoch_loss = 0.0
                for i in range(0, n, bs):
                    batch_idx = idx[i:i + bs]
                    xb = X_tensor[batch_idx]
                    noise = torch.randn_like(xb) * float(self.noise_std)
                    x_noisy = xb + noise
                    optimizer.zero_grad()
                    x_hat = model(x_noisy)
                    loss = loss_fn(x_hat, xb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.item()) * int(xb.shape[0])
                if self.verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):
                    print(f"[SSL-DAE] epoch={epoch:04d} loss={epoch_loss / max(n, 1):.6f}")

            model.eval()
            self.model_ = model.to("cpu")
            return self

        self.method = "none"
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.mu_ is None or self.sig_ is None:
            return X

        method = (self.method or "none").lower()
        if method == "none":
            return X

        Xs = self._standardize(X)
        if self.pca_ is not None:
            z = self.pca_.transform(Xs)
            x_hat_s = self.pca_.inverse_transform(z).astype(np.float32)
            return self._unstandardize(x_hat_s)
        if self.model_ is not None and HAS_TORCH:
            with torch.no_grad():
                xt = torch.from_numpy(Xs).to("cpu")
                x_hat_s = self.model_(xt).numpy().astype(np.float32)
            return self._unstandardize(x_hat_s)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.asarray(list(input_features), dtype=object)


# =============================================================================
# Model registry and Bayesian search
# =============================================================================
@dataclass
class ModelFamilySpec:
    name: str
    estimator_type: str
    default_params: Dict[str, Any]
    search_space: Dict[str, Tuple[str, Any, Any]]


def get_candidate_model_specs() -> List[ModelFamilySpec]:
    specs: List[ModelFamilySpec] = [
        ModelFamilySpec(
            name="extratrees",
            estimator_type="extratrees",
            default_params={
                "n_estimators": 500,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
            },
            search_space={
                "n_estimators": ("int", 200, 1000),
                "max_depth": ("int_or_none", 3, 20),
                "min_samples_split": ("int", 2, 10),
                "min_samples_leaf": ("int", 1, 5),
                "max_features": ("float", 0.4, 1.0),
            },
        ),
        ModelFamilySpec(
            name="randomforest",
            estimator_type="randomforest",
            default_params={
                "n_estimators": 500,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
            },
            search_space={
                "n_estimators": ("int", 200, 1000),
                "max_depth": ("int_or_none", 3, 20),
                "min_samples_split": ("int", 2, 10),
                "min_samples_leaf": ("int", 1, 5),
                "max_features": ("float", 0.4, 1.0),
            },
        ),
        ModelFamilySpec(
            name="svr",
            estimator_type="svr",
            default_params={
                "kernel": "rbf",
                "C": 10.0,
                "epsilon": 0.05,
                "gamma": "scale",
            },
            search_space={
                "C": ("logfloat", 1e-1, 1e3),
                "epsilon": ("logfloat", 1e-3, 2e-1),
                "gamma": ("logfloat", 1e-4, 1e0),
            },
        ),
    ]
    if HAS_LGBM:
        specs.append(
            ModelFamilySpec(
                name="lightgbm",
                estimator_type="lightgbm",
                default_params={
                    "n_estimators": 700,
                    "learning_rate": 0.03,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "num_leaves": 31,
                    "min_child_samples": 8,
                    "reg_alpha": 0.10,
                    "reg_lambda": 0.80,
                    "random_state": RANDOM_SEED,
                    "n_jobs": -1,
                    "force_col_wise": True,
                },
                search_space={
                    "n_estimators": ("int", 200, 1200),
                    "learning_rate": ("logfloat", 1e-2, 1.5e-1),
                    "subsample": ("float", 0.6, 1.0),
                    "colsample_bytree": ("float", 0.6, 1.0),
                    "num_leaves": ("int", 15, 63),
                    "min_child_samples": ("int", 5, 20),
                    "reg_alpha": ("logfloat", 1e-4, 1.0),
                    "reg_lambda": ("logfloat", 1e-4, 5.0),
                },
            )
        )
    if HAS_XGB:
        specs.append(
            ModelFamilySpec(
                name="xgboost",
                estimator_type="xgboost",
                default_params={
                    "n_estimators": 700,
                    "learning_rate": 0.03,
                    "max_depth": 5,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                    "random_state": RANDOM_SEED,
                    "n_jobs": -1,
                    "objective": "reg:squarederror",
                    "verbosity": 0,
                },
                search_space={
                    "n_estimators": ("int", 200, 1200),
                    "learning_rate": ("logfloat", 1e-2, 1.5e-1),
                    "max_depth": ("int", 3, 8),
                    "subsample": ("float", 0.6, 1.0),
                    "colsample_bytree": ("float", 0.6, 1.0),
                    "reg_alpha": ("logfloat", 1e-4, 1.0),
                    "reg_lambda": ("logfloat", 1e-4, 5.0),
                },
            )
        )
    return specs


def instantiate_estimator(spec: ModelFamilySpec, params: Dict[str, Any]) -> BaseEstimator:
    merged = {**spec.default_params, **params}
    if spec.estimator_type == "lightgbm":
        return LGBMRegressor(**merged)
    if spec.estimator_type == "xgboost":
        return XGBRegressor(**merged)
    if spec.estimator_type == "extratrees":
        return ExtraTreesRegressor(**merged)
    if spec.estimator_type == "randomforest":
        return RandomForestRegressor(**merged)
    if spec.estimator_type == "svr":
        return SVR(**merged)
    raise ValueError(f"Unsupported estimator type: {spec.estimator_type}")


def make_pipeline(spec: ModelFamilySpec, params: Dict[str, Any]) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if USE_SSL_PRETRAIN and SSL_METHOD.lower() != "none":
        steps.append((
            "ssl",
            SSLEncoder(
                method=SSL_METHOD,
                latent_dim=SSL_LATENT_DIM,
                hidden_dim=SSL_HIDDEN_DIM,
                epochs=SSL_EPOCHS,
                batch_size=SSL_BATCH_SIZE,
                noise_std=SSL_NOISE_STD,
                lr=SSL_LR,
                weight_decay=SSL_WEIGHT_DECAY,
                random_state=RANDOM_SEED,
                verbose=SSL_VERBOSE,
            ),
        ))
    steps.append(("model", instantiate_estimator(spec, params)))
    return Pipeline(steps=steps)


def sample_param(space_item: Tuple[str, Any, Any], rng: np.random.RandomState) -> Any:
    kind, low, high = space_item
    if kind == "int":
        return int(rng.randint(int(low), int(high) + 1))
    if kind == "float":
        return float(rng.uniform(float(low), float(high)))
    if kind == "logfloat":
        return float(10 ** rng.uniform(np.log10(float(low)), np.log10(float(high))))
    if kind == "int_or_none":
        if rng.rand() < 0.2:
            return None
        return int(rng.randint(int(low), int(high) + 1))
    raise ValueError(f"Unknown search-space kind: {kind}")


def sample_params(space: Dict[str, Tuple[str, Any, Any]], rng: np.random.RandomState) -> Dict[str, Any]:
    return {name: sample_param(spec, rng) for name, spec in space.items()}


def encode_params_to_unit(params: Dict[str, Any], space: Dict[str, Tuple[str, Any, Any]]) -> np.ndarray:
    vec = []
    for name, spec in space.items():
        kind, low, high = spec
        value = params[name]
        if kind == "int":
            vec.append((float(value) - float(low)) / (float(high) - float(low)))
        elif kind == "float":
            vec.append((float(value) - float(low)) / (float(high) - float(low)))
        elif kind == "logfloat":
            v = np.log10(float(value))
            vec.append((v - np.log10(float(low))) / (np.log10(float(high)) - np.log10(float(low))))
        elif kind == "int_or_none":
            if value is None:
                vec.append(0.0)
            else:
                vec.append((float(value) - float(low)) / (float(high) - float(low)))
        else:
            raise ValueError(f"Unknown search-space kind: {kind}")
    return np.asarray(vec, dtype=float)


def propose_next_params(
    observations: List[Dict[str, Any]],
    scores: List[float],
    space: Dict[str, Tuple[str, Any, Any]],
    rng: np.random.RandomState,
    candidate_pool: int = 256,
) -> Dict[str, Any]:
    if len(observations) < 3:
        return sample_params(space, rng)

    X_obs = np.vstack([encode_params_to_unit(p, space) for p in observations])
    y_obs = np.asarray(scores, dtype=float)

    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=RANDOM_SEED)
    gpr.fit(X_obs, y_obs)

    candidates = [sample_params(space, rng) for _ in range(candidate_pool)]
    X_cand = np.vstack([encode_params_to_unit(p, space) for p in candidates])
    mu, sigma = gpr.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    best = float(np.max(y_obs))
    z = (mu - best) / sigma
    ei = (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)
    best_idx = int(np.argmax(ei))
    return candidates[best_idx]


# =============================================================================
# Cross-validation and model selection
# =============================================================================
def compute_cv_objective(r2_values: List[float]) -> float:
    arr = np.asarray(r2_values, dtype=float)
    return float(np.nanmean(arr) - CV_STABILITY_PENALTY * np.nanstd(arr))


def cross_val_score_model(
    X_df: pd.DataFrame,
    y_raw: np.ndarray,
    spec: ModelFamilySpec,
    params: Dict[str, Any],
    target_name: str,
) -> Dict[str, Any]:
    transformer = TargetTransformer(TARGET_TRANSFORMS.get(target_name, "identity"))
    y_model = transformer.transform(y_raw)
    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)
    fold_r2, fold_rmse, fold_mae = [], [], []

    for tr_idx, va_idx in kf.split(X_df):
        pipe = make_pipeline(spec, params)
        pipe.fit(X_df.iloc[tr_idx], y_model[tr_idx])
        pred = pipe.predict(X_df.iloc[va_idx])
        rep = metric_report(y_model[va_idx], pred)
        fold_r2.append(rep["R2"])
        fold_rmse.append(rep["RMSE"])
        fold_mae.append(rep["MAE"])

    return {
        "cv_r2_mean": float(np.nanmean(fold_r2)),
        "cv_r2_std": float(np.nanstd(fold_r2)),
        "cv_rmse_mean": float(np.nanmean(fold_rmse)),
        "cv_mae_mean": float(np.nanmean(fold_mae)),
        "objective": compute_cv_objective(fold_r2),
    }


def select_model_family_and_params(
    X_df: pd.DataFrame,
    y_raw: np.ndarray,
    target_name: str,
) -> Tuple[ModelFamilySpec, Dict[str, Any], pd.DataFrame]:
    specs = get_candidate_model_specs()
    leaderboard_rows: List[Dict[str, Any]] = []

    # Stage 1: baseline comparison for all candidate families.
    for spec in specs:
        rep = cross_val_score_model(X_df, y_raw, spec, spec.default_params, target_name)
        leaderboard_rows.append({
            "stage": "baseline",
            "model_family": spec.name,
            **spec.default_params,
            **rep,
        })

    baseline_df = pd.DataFrame(leaderboard_rows)
    shortlist_names = baseline_df.sort_values("objective", ascending=False)["model_family"].head(SHORTLIST_FAMILIES).tolist()
    shortlist = [s for s in specs if s.name in shortlist_names]

    rng = np.random.RandomState(RANDOM_SEED)
    all_rows = leaderboard_rows.copy()
    best_score = -np.inf
    best_spec = shortlist[0]
    best_params = shortlist[0].default_params

    # Stage 2: lightweight Bayesian optimization on shortlisted families.
    for spec in shortlist:
        observations: List[Dict[str, Any]] = []
        scores: List[float] = []

        seed_points = [spec.default_params.copy()]
        while len(seed_points) < BO_INIT_POINTS + 1:
            seed_points.append(sample_params(spec.search_space, rng))

        for params in seed_points:
            rep = cross_val_score_model(X_df, y_raw, spec, params, target_name)
            observations.append(params)
            scores.append(rep["objective"])
            all_rows.append({
                "stage": "bayes_opt",
                "model_family": spec.name,
                **params,
                **rep,
            })
            if rep["objective"] > best_score:
                best_score = rep["objective"]
                best_spec = spec
                best_params = params

        for _ in range(BO_ITERATIONS):
            params = propose_next_params(observations, scores, spec.search_space, rng, BO_CANDIDATE_POOL)
            rep = cross_val_score_model(X_df, y_raw, spec, params, target_name)
            observations.append(params)
            scores.append(rep["objective"])
            all_rows.append({
                "stage": "bayes_opt",
                "model_family": spec.name,
                **params,
                **rep,
            })
            if rep["objective"] > best_score:
                best_score = rep["objective"]
                best_spec = spec
                best_params = params

    leaderboard = pd.DataFrame(all_rows).sort_values(["objective", "cv_r2_mean"], ascending=False)
    return best_spec, best_params, leaderboard


# =============================================================================
# Explainability helpers
# =============================================================================
def tree_gain_importance(pipe: Pipeline, feature_cols: List[str]) -> Optional[np.ndarray]:
    model = pipe.named_steps["model"]
    if hasattr(model, "booster_"):
        imp = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
        return imp if imp.shape[0] == len(feature_cols) else None
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        return imp if imp.shape[0] == len(feature_cols) else None
    return None


def compute_shap_values(pipe: Pipeline, X_df: pd.DataFrame) -> Optional[np.ndarray]:
    if not HAS_SHAP:
        return None
    try:
        model = pipe.named_steps["model"]
        X_pre = pipe[:-1].transform(X_df)
        n = int(X_pre.shape[0])
        if n > SHAP_MAX_SAMPLES:
            rng = np.random.RandomState(RANDOM_SEED)
            idx = rng.choice(n, size=SHAP_MAX_SAMPLES, replace=False)
            X_s = X_pre[idx]
        else:
            X_s = X_pre

        if hasattr(model, "booster_") or hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_s)
            return np.asarray(shap_vals, dtype=float)
        return None
    except Exception:
        return None


def permutation_importance_values(pipe: Pipeline, X_df: pd.DataFrame, y_model: np.ndarray) -> np.ndarray:
    result = permutation_importance(
        pipe,
        X_df,
        y_model,
        scoring="r2",
        n_repeats=25,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    return np.asarray(result.importances_mean, dtype=float)


def beeswarm_like(ax, shap_vals: np.ndarray, feat_vals: np.ndarray, feat_names: List[str], max_display: int = 20):
    rng = np.random.RandomState(RANDOM_SEED)
    shap_vals = np.asarray(shap_vals, dtype=float)
    feat_vals = np.asarray(feat_vals, dtype=float)
    n, m = shap_vals.shape
    m2 = min(m, max_display)
    order = np.argsort(-np.mean(np.abs(shap_vals), axis=0))[:m2]
    sv = shap_vals[:, order]
    fv = feat_vals[:, order]
    names = [feat_names[i] for i in order]
    ys = np.arange(m2)[::-1]
    for j in range(m2):
        x = sv[:, j]
        y = np.full(n, ys[j], dtype=float) + rng.normal(scale=0.11, size=n)
        v = fv[:, j]
        vmin = np.nanpercentile(v, 5)
        vmax = np.nanpercentile(v, 95)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            c = np.zeros_like(v, dtype=float)
        else:
            c = (v - vmin) / (vmax - vmin)
        ax.scatter(x, y, c=c, s=12, alpha=0.80, linewidths=0.0, cmap="coolwarm")
    ax.axvline(0.0, color="k", linewidth=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)


# =============================================================================
# Plotting
# =============================================================================
def parity_single_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, xlab: str, ylab: str, outpath: str):
    import matplotlib.pyplot as plt

    set_pub_style()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    rep = metric_report(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.scatter(y_true, y_pred, s=24, edgecolor="k", linewidth=0.6, alpha=0.85)
    vmin = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)])) if len(y_true) else 0.0
    vmax = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)])) if len(y_true) else 1.0
    ax.plot([vmin, vmax], [vmin, vmax], "-", linewidth=1.2)
    ax.set_xlabel(xlab, fontsize=13)
    ax.set_ylabel(ylab, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.text(
        0.05,
        0.95,
        f"R² = {rep['R2']:.3f}\nRMSE = {rep['RMSE']:.3g}\nMAE = {rep['MAE']:.3g}",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
    )
    savefig(fig, outpath)
    plt.close(fig)


def plot_importance_single(target_name: str, pipe: Pipeline, X_train_df: pd.DataFrame, y_model: np.ndarray, feature_cols: List[str], outpath: str):
    import matplotlib.pyplot as plt

    set_pub_style()
    imp = tree_gain_importance(pipe, feature_cols)
    xlabel = "Gain importance"
    if imp is None:
        imp = permutation_importance_values(pipe, X_train_df, y_model)
        xlabel = "Permutation importance (mean ΔR²)"

    order = np.argsort(-imp)
    names = [feature_cols[i] for i in order][:SHAP_MAX_DISPLAY]
    vals = imp[order][:SHAP_MAX_DISPLAY]
    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals[::-1], edgecolor="k", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names[::-1], fontsize=12)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_title(f"Model-1 | {target_name} | Feature importance", fontsize=15)
    savefig(fig, outpath)
    plt.close(fig)


def plot_shap_single(target_name: str, pipe: Pipeline, X_train_df: pd.DataFrame, feature_cols: List[str], outpath: str):
    import matplotlib.pyplot as plt

    set_pub_style()
    fig, ax = plt.subplots(figsize=(9.4, 6.8))
    shap_vals = compute_shap_values(pipe, X_train_df)
    X_pre = pipe[:-1].transform(X_train_df)

    if shap_vals is None:
        ax.text(0.08, 0.5, "SHAP unavailable for the selected model family.", transform=ax.transAxes, fontsize=13)
        ax.axis("off")
    else:
        n_all = int(X_pre.shape[0])
        if n_all > SHAP_MAX_SAMPLES:
            rng = np.random.RandomState(RANDOM_SEED)
            idx = rng.choice(n_all, size=SHAP_MAX_SAMPLES, replace=False)
            X_use = X_pre[idx]
        else:
            X_use = X_pre
        beeswarm_like(ax, shap_vals, X_use, feature_cols, max_display=SHAP_MAX_DISPLAY)
        ax.set_title(f"Model-1 | {target_name} | SHAP summary", fontsize=16, pad=8)

    savefig(fig, outpath)
    plt.close(fig)


def _embed_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if PREFER_UMAP and HAS_UMAP:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=18,
            min_dist=0.15,
            random_state=RANDOM_SEED,
            metric="euclidean",
        )
        return reducer.fit_transform(X)
    return PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)


def plot_embedding_1x3(df_train_ref: pd.DataFrame, df_input_pred: pd.DataFrame, feature_cols: List[str], pred_prefix: str, outpath: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    set_pub_style()
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = ensure_numeric(df_train_ref.copy(), feature_cols)[feature_cols].values
    X_input = ensure_numeric(df_input_pred.copy(), feature_cols)[feature_cols].values
    X_train_imp = imp.fit_transform(X_train)
    X_input_imp = imp.transform(X_input)
    X_train_s = scaler.fit_transform(X_train_imp)
    X_input_s = scaler.transform(X_input_imp)
    X_all = np.vstack([X_train_s, X_input_s])
    Z_all = _embed_2d(X_all)
    n_train = X_train_s.shape[0]
    Z_train = Z_all[:n_train]
    Z_input = Z_all[n_train:]

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6))
    for j, target_name in enumerate(TARGETS):
        ax = axes[j]
        col = f"{pred_prefix}{target_name}"
        if col not in df_train_ref.columns or col not in df_input_pred.columns:
            ax.axis("off")
            continue
        v_train = df_train_ref[col].values.astype(float)
        v_input = df_input_pred[col].values.astype(float)
        vmin = np.nanpercentile(v_train, 3)
        vmax = np.nanpercentile(v_train, 97)
        norm_obj = Normalize(vmin=vmin, vmax=vmax, clip=True)
        sc = ax.scatter(Z_train[:, 0], Z_train[:, 1], c=v_train, s=14, alpha=0.50, linewidths=0.0, norm=norm_obj, cmap="viridis")
        ax.scatter(Z_input[:, 0], Z_input[:, 1], c=v_input, s=28, alpha=0.95, edgecolor="k", linewidth=0.6, norm=norm_obj, cmap="viridis")
        ax.set_title(target_name, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=10)

    savefig(fig, outpath)
    plt.close(fig)


def plot_3d_pred_space(df_train_pred: pd.DataFrame, df_input_pred: pd.DataFrame, pred_prefix: str, outpath: str):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    set_pub_style()
    cols = [f"{pred_prefix}{t}" for t in TARGETS]
    if not all(c in df_train_pred.columns for c in cols) or not all(c in df_input_pred.columns for c in cols):
        return

    A = df_train_pred[cols].values.astype(float)
    B = df_input_pred[cols].values.astype(float)
    fig = plt.figure(figsize=(8.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], s=18, alpha=0.35)
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], s=34, alpha=0.95, edgecolor="k", linewidth=0.6)
    ax.set_xlabel(TARGETS[0], fontsize=12)
    ax.set_ylabel(TARGETS[1], fontsize=12)
    ax.set_zlabel(TARGETS[2], fontsize=12)
    savefig(fig, outpath)
    plt.close(fig)


# =============================================================================
# Multi-objective ranking: entropy-weighted TOPSIS + Pareto
# =============================================================================
def orient_and_minmax(X: np.ndarray, larger_is_better: List[bool]) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    out = X.copy()
    for j, higher in enumerate(larger_is_better):
        col = X[:, j].copy()
        if not higher:
            col = -col
        cmin = np.nanmin(col)
        cmax = np.nanmax(col)
        if not np.isfinite(cmin) or not np.isfinite(cmax) or abs(cmax - cmin) < 1e-12:
            out[:, j] = 1.0
        else:
            out[:, j] = (col - cmin) / (cmax - cmin)
    return np.clip(out, 1e-12, None)


def entropy_weights(X_pos: np.ndarray) -> np.ndarray:
    X = np.asarray(X_pos, dtype=float)
    P = X / np.sum(X, axis=0, keepdims=True)
    P = np.clip(P, 1e-12, 1.0)
    n = X.shape[0]
    k = 1.0 / np.log(max(n, 2))
    E = -k * np.sum(P * np.log(P), axis=0)
    d = 1.0 - E
    if np.sum(d) <= 1e-12:
        return np.ones(X.shape[1], dtype=float) / X.shape[1]
    return d / np.sum(d)


def topsis_closeness(X: np.ndarray, weights: np.ndarray, larger_is_better: List[bool]) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X_oriented = X.copy()
    for j, higher in enumerate(larger_is_better):
        if not higher:
            X_oriented[:, j] = -X_oriented[:, j]
    denom = np.sqrt(np.sum(X_oriented ** 2, axis=0, keepdims=True))
    denom = np.where(denom > 1e-12, denom, 1.0)
    R = X_oriented / denom
    V = R * weights.reshape(1, -1)
    ideal_best = np.max(V, axis=0)
    ideal_worst = np.min(V, axis=0)
    d_pos = np.sqrt(np.sum((V - ideal_best) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((V - ideal_worst) ** 2, axis=1))
    return d_neg / np.maximum(d_pos + d_neg, 1e-12)


def pareto_front_mask(vals: np.ndarray, larger_is_better: List[bool]) -> np.ndarray:
    V = np.asarray(vals, dtype=float)
    S = V.copy()
    for j, higher in enumerate(larger_is_better):
        if not higher:
            S[:, j] = -S[:, j]
    n = S.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominated = np.all(S >= S[i], axis=1) & np.any(S > S[i], axis=1)
        if np.any(dominated):
            is_pareto[i] = False
    return is_pareto


def percentile_like_candidate_ranking(rank: np.ndarray) -> np.ndarray:
    r = np.asarray(rank, dtype=float)
    n = float(np.nanmax(r)) if len(r) else 1.0
    if n <= 1:
        return np.full_like(r, 100.0)
    return 100.0 * (1.0 - (r - 1.0) / (n - 1.0))


def build_candidate_ranking(
    df_input: pd.DataFrame,
    point_preds: Dict[str, np.ndarray],
    lcb_preds: Dict[str, np.ndarray],
    ucb_preds: Dict[str, np.ndarray],
    model_tag: str = "M1",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df_input.copy()
    pred_cols = {t: f"{model_tag}_pred_{t}" for t in TARGETS}
    lcb_cols = {t: f"{model_tag}_LCB_{t}" for t in TARGETS}
    ucb_cols = {t: f"{model_tag}_UCB_{t}" for t in TARGETS}

    for target_name in TARGETS:
        out[pred_cols[target_name]] = np.asarray(point_preds[target_name], dtype=float).ravel()
        out[lcb_cols[target_name]] = np.asarray(lcb_preds[target_name], dtype=float).ravel()
        out[ucb_cols[target_name]] = np.asarray(ucb_preds[target_name], dtype=float).ravel()

    point_matrix = np.vstack([out[pred_cols[t]].values.astype(float) for t in TARGETS]).T
    lcb_matrix = np.vstack([out[lcb_cols[t]].values.astype(float) for t in TARGETS]).T

    point_pos = orient_and_minmax(point_matrix, [True, True, True])
    lcb_pos = orient_and_minmax(lcb_matrix, [True, True, True])
    point_weights = entropy_weights(point_pos)
    lcb_weights = entropy_weights(lcb_pos)

    point_score = topsis_closeness(point_matrix, point_weights, [True, True, True])
    robust_score = topsis_closeness(lcb_matrix, lcb_weights, [True, True, True])

    out[f"{model_tag}_point_topsis_score"] = point_score
    out[f"{model_tag}_point_topsis_rank"] = (-point_score).argsort().argsort() + 1
    out[f"{model_tag}_robust_topsis_score"] = robust_score
    out[f"{model_tag}_robust_topsis_rank"] = (-robust_score).argsort().argsort() + 1
    out[f"{model_tag}_rank"] = out[f"{model_tag}_robust_topsis_rank"]
    out[f"{model_tag}_score"] = out[f"{model_tag}_robust_topsis_score"]
    out[f"{model_tag}_score_percentile"] = percentile_like_candidate_ranking(out[f"{model_tag}_rank"].values)
    out[f"{model_tag}_pareto_front"] = pareto_front_mask(point_matrix, [True, True, True])

    weights_df = pd.DataFrame({
        "Target": TARGETS,
        "EntropyWeight_point": point_weights,
        "EntropyWeight_LCB": lcb_weights,
    })
    return out, weights_df


# =============================================================================
# Training / OOF / final fit
# =============================================================================
def clip_predictions(target_name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if target_name == "Conductivity_25C":
        return np.clip(arr, 1e-12, None)
    if target_name == "Li_Transfer":
        return np.clip(arr, 0.0, 1.0)
    if target_name == "E_Window":
        return np.clip(arr, 0.0, None)
    return arr


def fit_target_workflow(df_train: pd.DataFrame, feature_cols: List[str], target_name: str) -> Dict[str, Any]:
    df_train = ensure_numeric(df_train.copy(), feature_cols + [target_name])
    df_train = df_train.loc[~df_train[target_name].isna()].copy()
    X = df_train[feature_cols]
    y_raw = df_train[target_name].astype(float).values
    transformer = TargetTransformer(TARGET_TRANSFORMS.get(target_name, "identity"))
    y_model = transformer.transform(y_raw)

    best_spec, best_params, leaderboard = select_model_family_and_params(X, y_raw, target_name)

    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)
    oof_model = np.zeros(len(X), dtype=float)
    for tr_idx, va_idx in kf.split(X):
        pipe = make_pipeline(best_spec, best_params)
        pipe.fit(X.iloc[tr_idx], y_model[tr_idx])
        oof_model[va_idx] = pipe.predict(X.iloc[va_idx])

    final_pipe = make_pipeline(best_spec, best_params)
    final_pipe.fit(X, y_model)
    train_fit_model = final_pipe.predict(X)

    # qhat is computed on the model scale to respect the transformed target definition.
    abs_res_model = np.abs(y_model - oof_model)
    qhat_model = float(np.quantile(abs_res_model, 1.0 - CONFORMAL_ALPHA)) if len(abs_res_model) else np.nan

    oof_orig = clip_predictions(target_name, transformer.inverse(oof_model))
    train_fit_orig = clip_predictions(target_name, transformer.inverse(train_fit_model))

    metrics_model_scale = metric_report(y_model, train_fit_model)
    metrics_oof_scale = metric_report(y_model, oof_model)

    return {
        "target": target_name,
        "spec": best_spec,
        "best_params": best_params,
        "leaderboard": leaderboard,
        "pipe": final_pipe,
        "transformer": transformer,
        "X_train_df": X,
        "y_train_raw": y_raw,
        "y_train_model": y_model,
        "oof_pred_model": oof_model,
        "oof_pred_raw": oof_orig,
        "train_fit_pred_model": train_fit_model,
        "train_fit_pred_raw": train_fit_orig,
        "qhat_model": qhat_model,
        "trainfit_metrics_model_scale": metrics_model_scale,
        "oof_metrics_model_scale": metrics_oof_scale,
    }


def predict_target(result: Dict[str, Any], df_any: pd.DataFrame, feature_cols: List[str]) -> Dict[str, np.ndarray]:
    X = ensure_numeric(df_any.copy(), feature_cols)[feature_cols]
    pred_model = result["pipe"].predict(X)
    transformer: TargetTransformer = result["transformer"]
    qhat_model = float(result["qhat_model"])
    point = clip_predictions(result["target"], transformer.inverse(pred_model))
    lcb = clip_predictions(result["target"], transformer.lower_bound(pred_model, qhat_model))
    ucb = clip_predictions(result["target"], transformer.upper_bound(pred_model, qhat_model))
    return {
        "pred_model": np.asarray(pred_model, dtype=float),
        "point": point,
        "lcb": lcb,
        "ucb": ucb,
    }


# =============================================================================
# Main driver
# =============================================================================
def main() -> None:
    if not HAS_LGBM:
        raise RuntimeError("LightGBM is required. Please install: pip install lightgbm")

    make_dirs(OUT_DIR)
    clean_path = locate_file(FILE_CLEAN)
    input_path = locate_file(FILE_INPUT)
    df_clean = pd.read_excel(clean_path)
    df_input = pd.read_excel(input_path)

    for drop_col in list(DROP_ALWAYS):
        for df_ in (df_clean, df_input):
            if drop_col in df_.columns:
                df_.drop(columns=[drop_col], inplace=True)

    feature_cols = pick_feature_columns(df_clean, df_input)
    if len(feature_cols) < 5:
        raise RuntimeError("Too few shared feature columns after preprocessing. Please check descriptor column consistency.")

    print("Final feature count (clean ∩ input):", len(feature_cols))
    print("Dropped salt descriptors:", DROP_SALT_DESCRIPTORS)
    print("Torch available:", HAS_TORCH, "| SHAP available:", HAS_SHAP, "| UMAP available:", HAS_UMAP)

    rid_clean = stable_row_id_features_only(df_clean, feature_cols)
    _, rid_test = train_test_split(rid_clean, test_size=FROZEN_TEST_SIZE, random_state=RANDOM_SEED)
    rid_test_set = set(rid_test.tolist())
    df_test = df_clean.loc[rid_clean.isin(rid_test_set)].copy()
    df_train = df_clean.loc[~rid_clean.isin(rid_test_set)].copy()

    out_model = os.path.join(OUT_DIR, "Model1")
    out_rank = os.path.join(OUT_DIR, "CandidateRanking")
    out_vis = os.path.join(OUT_DIR, "Visualization")
    make_dirs(out_model)
    make_dirs(out_rank)
    make_dirs(out_vis)
    make_dirs(os.path.join(out_model, "Importance"))
    make_dirs(os.path.join(out_model, "SHAP"))
    make_dirs(os.path.join(out_vis, "Parity"))

    results: Dict[str, Dict[str, Any]] = {}
    metrics_rows: List[Dict[str, Any]] = []
    selection_tables: Dict[str, pd.DataFrame] = {}

    for target_name in TARGETS:
        if target_name not in df_train.columns:
            print(f"[WARN] Target '{target_name}' not found; skipped.")
            continue

        print("\n" + "=" * 92)
        print(f"Target: {target_name}")
        res = fit_target_workflow(df_train, feature_cols, target_name)
        results[target_name] = res
        selection_tables[target_name] = res["leaderboard"]

        # Frozen held-out test metrics are reported on the model scale.
        if target_name in df_test.columns:
            y_test_raw = pd.to_numeric(df_test[target_name], errors="coerce").values.astype(float)
            mask = ~np.isnan(y_test_raw)
            if np.sum(mask) >= 2:
                X_test = ensure_numeric(df_test.loc[mask].copy(), feature_cols)[feature_cols]
                y_test_model = res["transformer"].transform(y_test_raw[mask])
                pred_test_model = res["pipe"].predict(X_test)
                rep_test_model = metric_report(y_test_model, pred_test_model)
            else:
                rep_test_model = {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}
        else:
            rep_test_model = {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}

        print("Selected model:", res["spec"].name)
        print("Best params:", res["best_params"])
        print("Frozen TEST (model scale):", rep_test_model)
        print("CV-OOF (model scale):", res["oof_metrics_model_scale"])
        print("Train-fit (model scale):", res["trainfit_metrics_model_scale"])
        print("qhat (model scale):", res["qhat_model"])

        metrics_rows.append({
            "Target": target_name,
            "TargetTransform": TARGET_TRANSFORMS.get(target_name, "identity"),
            "SelectedModel": res["spec"].name,
            "Frozen_R2": rep_test_model["R2"],
            "Frozen_RMSE": rep_test_model["RMSE"],
            "Frozen_MAE": rep_test_model["MAE"],
            "OOF_R2": res["oof_metrics_model_scale"]["R2"],
            "OOF_RMSE": res["oof_metrics_model_scale"]["RMSE"],
            "OOF_MAE": res["oof_metrics_model_scale"]["MAE"],
            "Trainfit_R2": res["trainfit_metrics_model_scale"]["R2"],
            "Trainfit_RMSE": res["trainfit_metrics_model_scale"]["RMSE"],
            "Trainfit_MAE": res["trainfit_metrics_model_scale"]["MAE"],
            "qhat_model_scale": res["qhat_model"],
            "SSL_enabled": bool(USE_SSL_PRETRAIN),
            "SSL_method": str(SSL_METHOD),
            "SSL_latent_dim": int(SSL_LATENT_DIM),
            "SSL_noise_std": float(SSL_NOISE_STD),
        })

        # Publication-style parity plots on the model scale.
        ylab_name = f"Predicted {target_name}" if TARGET_TRANSFORMS.get(target_name, 'identity') == 'identity' else f"Predicted log10({target_name})"
        xlab_name = f"True {target_name}" if TARGET_TRANSFORMS.get(target_name, 'identity') == 'identity' else f"True log10({target_name})"
        parity_single_plot(
            res["y_train_model"],
            res["train_fit_pred_model"],
            title=f"Model-1 | {target_name} | Train-fit",
            xlab=xlab_name,
            ylab=ylab_name,
            outpath=os.path.join(out_vis, "Parity", f"Parity_TrainFit_{target_name}.png"),
        )
        parity_single_plot(
            res["y_train_model"],
            res["oof_pred_model"],
            title=f"Model-1 | {target_name} | CV-OOF",
            xlab=xlab_name,
            ylab=ylab_name,
            outpath=os.path.join(out_vis, "Parity", f"Parity_CV_OOF_{target_name}.png"),
        )

        plot_importance_single(
            target_name,
            res["pipe"],
            res["X_train_df"],
            res["y_train_model"],
            feature_cols,
            os.path.join(out_model, "Importance", f"Importance_{target_name}.png"),
        )
        plot_shap_single(
            target_name,
            res["pipe"],
            res["X_train_df"],
            feature_cols,
            os.path.join(out_model, "SHAP", f"SHAP_{target_name}.png"),
        )

    metrics_path = os.path.join(OUT_DIR, "metrics_summary_all_targets.xlsx")
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
        pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="Metrics", index=False)
        for target_name, table in selection_tables.items():
            sheet = target_name.replace("Conductivity_25C", "sigma")[:31]
            table.to_excel(writer, sheet_name=f"Sel_{sheet}"[:31], index=False)
    print(f"\nExported metrics summary: {metrics_path}")

    # Candidate prediction and ranking.
    df_input2 = df_input.copy()
    for col in feature_cols:
        if col not in df_input2.columns:
            df_input2[col] = np.nan
    df_input2 = ensure_numeric(df_input2, feature_cols)
    df_train_ref = ensure_numeric(df_train.copy(), feature_cols)

    point_preds: Dict[str, np.ndarray] = {}
    lcb_preds: Dict[str, np.ndarray] = {}
    ucb_preds: Dict[str, np.ndarray] = {}
    for target_name in TARGETS:
        pred_pack = predict_target(results[target_name], df_input2, feature_cols)
        point_preds[target_name] = pred_pack["point"]
        lcb_preds[target_name] = pred_pack["lcb"]
        ucb_preds[target_name] = pred_pack["ucb"]
        train_pack = predict_target(results[target_name], df_train_ref, feature_cols)
        df_train_ref[f"M1_{target_name}"] = train_pack["point"]

    rank_df, weights_df = build_candidate_ranking(df_input2, point_preds, lcb_preds, ucb_preds, model_tag="M1")

    out_rank_xlsx = os.path.join(out_rank, "candidate_ranking_ML_input_Model1.xlsx")
    with pd.ExcelWriter(out_rank_xlsx, engine="openpyxl") as writer:
        df_input2.to_excel(writer, sheet_name="ML_input", index=False)
        rank_df.to_excel(writer, sheet_name="CandidateRanking_M1", index=False)
        weights_df.to_excel(writer, sheet_name="EntropyWeights", index=False)
        for target_name, table in selection_tables.items():
            sheet = f"ModelSel_{target_name}"[:31]
            table.to_excel(writer, sheet_name=sheet, index=False)
    print(f"\nExported candidate ranking Excel: {out_rank_xlsx}")

    df_input_pred_merge = df_input2.copy()
    for target_name in TARGETS:
        df_input_pred_merge[f"M1_{target_name}"] = point_preds[target_name]

    plot_embedding_1x3(
        df_train_ref=df_train_ref,
        df_input_pred=df_input_pred_merge,
        feature_cols=feature_cols,
        pred_prefix="M1_",
        outpath=os.path.join(out_vis, "Embedding_Map_1x3_Model1.png"),
    )
    plot_3d_pred_space(
        df_train_pred=df_train_ref,
        df_input_pred=df_input_pred_merge,
        pred_prefix="M1_",
        outpath=os.path.join(out_vis, "3D_pred_space_Model1.png"),
    )

    print("\n✅ All done.")
    print("Key outputs:")
    print(f" - {out_rank_xlsx}")
    print(f" - {metrics_path}")
    print(f" - {os.path.join(out_vis, 'Parity')}")
    print(f" - {os.path.join(out_model, 'Importance')}")
    print(f" - {os.path.join(out_model, 'SHAP')}")
    print(f" - {os.path.join(out_vis, 'Embedding_Map_1x3_Model1.png')}")
    print(f" - {os.path.join(out_vis, '3D_pred_space_Model1.png')}")


if __name__ == "__main__":
    main()
