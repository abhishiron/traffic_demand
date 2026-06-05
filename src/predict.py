"""
src/predict.py
--------------
Load saved artifacts, build test features using the SAME build_features()
as train.py, predict demand, clip to [0,1], write submission.

Run from repo root:
    python src/predict.py

Expects:
    models/artifacts.pkl   (produced by src/train.py)
    data/test.csv
    data/sample_submission.csv
Writes:
    submissions/submission.csv  — columns: Index, demand
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features

DATA_DIR        = Path("data")
MODEL_DIR       = Path("models")
SUBMISSIONS_DIR = Path("submissions")
SUBMISSIONS_DIR.mkdir(exist_ok=True)


def _inv_logit(x: np.ndarray) -> np.ndarray:
    return np.clip(1.0 / (1.0 + np.exp(-x)), 0.0, 1.0)


def main() -> None:
    print("=" * 65)
    print("  Traffic Demand — XGBoost Prediction")
    print("=" * 65)

    # ── Load artifacts ────────────────────────────────────────────────
    arts_path = MODEL_DIR / "artifacts.pkl"
    if not arts_path.exists():
        raise FileNotFoundError(
            f"{arts_path} not found.\n"
            "Run  python src/train.py  first."
        )
    arts       = joblib.load(arts_path)
    model      = arts["model"]
    enc        = arts["encoders"]
    use_logit  = arts["use_logit"]
    holdout_r2 = arts.get("holdout_r2", float("nan"))

    print(f"\n  Loaded artifacts from {arts_path}")
    print(f"  use_logit   = {use_logit}")
    print(f"  holdout R2  = {holdout_r2:.6f}  "
          f"(~ score {max(0, 100*holdout_r2):.2f})")
    print(f"  feature cols: {arts.get('feature_cols', '?')}")

    # ── Load test data ────────────────────────────────────────────────
    test   = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    print(f"\n  test             : {test.shape}")
    print(f"  sample_submission: {sample.shape}")

    # ── Build features ────────────────────────────────────────────────
    X_test, _ = build_features(test, encoders=enc)

    # ── Predict ───────────────────────────────────────────────────────
    raw_preds = model.predict(X_test)
    preds     = _inv_logit(raw_preds) if use_logit else np.clip(raw_preds, 0.0, 1.0)

    print(f"\n  Prediction statistics:")
    print(f"    min    = {preds.min():.6f}")
    print(f"    max    = {preds.max():.6f}")
    print(f"    mean   = {preds.mean():.6f}")
    print(f"    median = {np.median(preds):.6f}")
    print(f"    std    = {preds.std():.6f}")

    # ── Build and save submission ──────────────────────────────────────
    # sample_submission is a 5-row example — use test.csv indices for full output
    sub = test[["Index"]].copy().reset_index(drop=True)
    sub["demand"] = preds

    out_path = SUBMISSIONS_DIR / "submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"\n  Submission saved -> {out_path}  ({len(sub):,} rows)")
    print(f"  Columns: {list(sub.columns)}")
    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()
