# pipeline/models/gradient_boost.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import date
from typing import Optional

from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, roc_auc_score

from pipeline.calibration.temperature_scaling import TemperatureScaler


MODEL_DIR = Path("data/models")

FEATURE_COLS = [
    # SP
    "sp_fip_diff", "sp_xera_diff", "sp_k_pct_diff", "sp_bb_pct_diff",
    "home_sp_fip", "away_sp_fip",
    "home_sp_days_rest", "away_sp_days_rest",
    # Bullpen
    "bp_fip_diff", "home_bp_fatigue", "away_bp_fatigue",
    # Batting
    "woba_diff", "xwoba_diff",
    # Context
    "park_factor_overall", "park_factor_R", "park_factor_L",
    "home_field_advantage",
    "is_night_game", "is_weekend",
    "season_early", "season_mid", "season_late", "season_post",
    "dow_friday", "dow_saturday", "dow_sunday",
]

TARGET_COL = "home_win"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y


def _lgbm_params() -> dict:
    return {
        "n_estimators":      600,
        "learning_rate":     0.02,
        "max_depth":         4,
        "num_leaves":        15,
        "min_child_samples": 40,       # conservative — ~2000 games/season
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "class_weight":      "balanced",
        "random_state":      42,
        "n_jobs":            -1,
        "verbosity":         -1,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    df: pd.DataFrame,
    save_tag: str = "latest",
) -> tuple[LGBMClassifier, TemperatureScaler, dict]:
    """
    Train on full df (sorted chronologically).
    Calibrate on last TimeSeriesSplit fold — never on training data.
    Returns (model, scaler, metrics).
    """
    df = df.sort_values("game_date").reset_index(drop=True)
    X, y = _prepare(df)

    # ── Time-series CV for eval metrics (no shuffling) ──────────────────────
    tscv   = TimeSeriesSplit(n_splits=5)
    folds  = list(tscv.split(X))

    oof_probs  = np.zeros(len(X))
    oof_labels = np.zeros(len(X))

    for train_idx, val_idx in folds:
        m = LGBMClassifier(**_lgbm_params())
        m.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[],
        )
        oof_probs[val_idx]  = m.predict_proba(X.iloc[val_idx])[:, 1]
        oof_labels[val_idx] = y.iloc[val_idx].values

    # Use only last fold for calibration (closest to production distribution)
    _, last_val_idx = folds[-1]
    scaler = TemperatureScaler().fit(
        oof_probs[last_val_idx],
        oof_labels[last_val_idx],
    )

    cal_probs = scaler.transform(oof_probs[last_val_idx])

    metrics = {
        "oof_brier_raw":  brier_score_loss(oof_labels[last_val_idx], oof_probs[last_val_idx]),
        "oof_brier_cal":  brier_score_loss(oof_labels[last_val_idx], cal_probs),
        "oof_auc":        roc_auc_score(oof_labels[last_val_idx], oof_probs[last_val_idx]),
        "temperature_T":  scaler.T,
        "n_train":        len(df),
        "train_end_date": df["game_date"].max(),
    }

    # ── Final model: retrain on ALL data ────────────────────────────────────
    final_model = LGBMClassifier(**_lgbm_params())
    final_model.fit(X, y)

    # ── Persist ─────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / f"lgbm_{save_tag}.pkl")
    joblib.dump(scaler,      MODEL_DIR / f"scaler_{save_tag}.pkl")

    return final_model, scaler, metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(
    tag: str = "latest",
) -> tuple[LGBMClassifier, TemperatureScaler]:
    model  = joblib.load(MODEL_DIR / f"lgbm_{tag}.pkl")
    scaler = joblib.load(MODEL_DIR / f"scaler_{tag}.pkl")
    return model, scaler


def predict(
    df: pd.DataFrame,
    model: LGBMClassifier,
    scaler: TemperatureScaler,
) -> np.ndarray:
    """
    Returns calibrated P(home_win) for each row in df.
    Rows with any missing required feature get probability np.nan.
    """
    X, _ = _prepare(df)

    missing_mask = X.isnull().any(axis=1)
    probs = np.full(len(X), np.nan)

    if (~missing_mask).any():
        raw = model.predict_proba(X[~missing_mask])[:, 1]
        probs[~missing_mask] = scaler.transform(raw)

    return probs


# ---------------------------------------------------------------------------
# Feature importance (SHAP-free, built-in LightGBM)
# ---------------------------------------------------------------------------

def feature_importance(model: LGBMClassifier) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )