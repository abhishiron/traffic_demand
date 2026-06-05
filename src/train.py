"""
src/train.py
------------
Time-based holdout validation + XGBoost training.

Validation strategy (mirrors train→test gap):
    fit  : day 48  00:00 – 21:45  (minutes 0 – 1305)
    val  : day 48  22:00 – 23:45  +  day 49  00:00 – 02:00

Tries both raw and logit-transformed demand; keeps the winner.
Retrains winner on full train, saves artifacts to models/.

Run from repo root:
    python src/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "demand"
SEED   = 42


# ── transform helpers ─────────────────────────────────────────────────────────

def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _inv_logit(x: np.ndarray) -> np.ndarray:
    return np.clip(1.0 / (1.0 + np.exp(-x)), 0.0, 1.0)


# ── XGBoost config ────────────────────────────────────────────────────────────

def _base_params() -> dict:
    return dict(
        n_estimators      = 3000,
        learning_rate     = 0.02,
        max_depth         = 7,
        min_child_weight  = 5,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        colsample_bylevel = 0.8,
        reg_alpha         = 0.05,
        reg_lambda        = 1.0,
        gamma             = 0.0,
        random_state      = SEED,
        n_jobs            = -1,
        tree_method       = "hist",   # fast histogram-based split finding
        verbosity         = 0,
        early_stopping_rounds = 150,
    )


# ── training helper ───────────────────────────────────────────────────────────

def _train_eval(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val_raw: np.ndarray,
    use_logit: bool,
) -> tuple[xgb.XGBRegressor, float, int]:
    """Train one XGBoost model and return (model, holdout_r2, best_iteration)."""
    y_tr       = _logit(y_fit)     if use_logit else y_fit
    y_val_eval = _logit(y_val_raw) if use_logit else y_val_raw

    model = xgb.XGBRegressor(**_base_params())
    model.fit(
        X_fit, y_tr,
        eval_set=[(X_val, y_val_eval)],
        verbose=500,
    )

    raw_preds = model.predict(X_val)
    preds     = _inv_logit(raw_preds) if use_logit else np.clip(raw_preds, 0.0, 1.0)
    r2        = r2_score(y_val_raw, preds)
    best_iter = int(model.best_iteration) + 1   # XGBoost best_iteration is 0-indexed
    return model, r2, best_iter


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Traffic Demand — XGBoost Training")
    print("=" * 65)

    # ── Load ─────────────────────────────────────────────────────────
    train = pd.read_csv(DATA_DIR / "train.csv")
    print(f"\nLoaded train: {train.shape}")

    # ── Time-based holdout split ──────────────────────────────────────
    ts_parts = train["timestamp"].str.split(":", expand=True).astype(int)
    minutes  = ts_parts[0] * 60 + ts_parts[1]

    fit_mask = (train["day"] == 48) & (minutes <= 1305)
    val_mask = ((train["day"] == 48) & (minutes >= 1320)) | (train["day"] == 49)

    fit_df = train[fit_mask].reset_index(drop=True)
    val_df = train[val_mask].reset_index(drop=True)
    print(f"  fit : {len(fit_df):,} rows  (day 48, 00:00 - 21:45)")
    print(f"  val : {len(val_df):,} rows  (day 48, 22:00 - 23:45  +  day 49)")

    # ── Build features ────────────────────────────────────────────────
    print("\nBuilding features ...")
    X_fit, enc_fit = build_features(fit_df)
    X_val, _       = build_features(val_df, encoders=enc_fit)
    y_fit = fit_df[TARGET].values
    y_val = val_df[TARGET].values

    print(f"  feature cols ({len(X_fit.columns)}): {list(X_fit.columns)}")

    # ── Holdout: raw demand ───────────────────────────────────────────
    print("\n" + "-" * 40)
    print("  [1/2] Training on RAW demand ...")
    m_raw, r2_raw, iter_raw = _train_eval(X_fit, y_fit, X_val, y_val, use_logit=False)
    print(f"  Holdout R2: {r2_raw:.6f}   score ~ {max(0, 100*r2_raw):.2f}   "
          f"best_iter: {iter_raw}")

    # ── Holdout: logit-transformed demand ─────────────────────────────
    print("\n" + "-" * 40)
    print("  [2/2] Training on LOGIT(demand) ...")
    m_logit, r2_logit, iter_logit = _train_eval(X_fit, y_fit, X_val, y_val, use_logit=True)
    print(f"  Holdout R2: {r2_logit:.6f}   score ~ {max(0, 100*r2_logit):.2f}   "
          f"best_iter: {iter_logit}")

    # ── Pick winner ───────────────────────────────────────────────────
    use_logit = r2_logit > r2_raw
    best_r2   = r2_logit if use_logit else r2_raw
    best_iter = iter_logit if use_logit else iter_raw
    print(f"\n  Winner : {'logit' if use_logit else 'raw'}  |  "
          f"Holdout R2 = {best_r2:.6f}  |  "
          f"Estimated score = {max(0, 100 * best_r2):.2f}")

    # ── Retrain on full train ─────────────────────────────────────────
    scale   = len(train) / max(len(fit_df), 1)
    final_n = max(best_iter, int(best_iter * scale))
    print(f"\n  Retraining on full train ({len(train):,} rows) ...")
    print(f"  n_estimators = {final_n}  (best_iter={best_iter} x scale={scale:.2f})")

    X_full, enc_full = build_features(train)
    y_full    = train[TARGET].values
    y_full_tr = _logit(y_full) if use_logit else y_full

    params_full = {**_base_params(), "n_estimators": final_n}
    params_full.pop("early_stopping_rounds")   # no early stopping on full retrain

    final_model = xgb.XGBRegressor(**params_full)
    final_model.fit(X_full, y_full_tr, verbose=500)

    # ── Feature importances ───────────────────────────────────────────
    fi = pd.Series(
        final_model.feature_importances_,
        index=X_full.columns,
        name="importance",
    ).sort_values(ascending=False)
    print("\n  Feature importances (gain-based, sorted):")
    print(fi.to_string())

    # ── Save artifacts ────────────────────────────────────────────────
    artifacts = {
        "model":        final_model,
        "encoders":     enc_full,
        "use_logit":    use_logit,
        "holdout_r2":   best_r2,
        "feature_cols": list(X_full.columns),
    }
    out = MODEL_DIR / "artifacts.pkl"
    joblib.dump(artifacts, out)
    print(f"\n  Saved -> {out}")
    print(f"\n{'='*65}")
    print(f"  Holdout R2 = {best_r2:.6f}  |  "
          f"Estimated competition score = {max(0, 100 * best_r2):.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
