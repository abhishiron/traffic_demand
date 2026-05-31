"""
src/features.py
---------------
Single source of truth for feature engineering.
Used by both train.py and predict.py so logic never diverges.

Public API
----------
build_features(df, encoders=None) -> (X: DataFrame, encoders: dict)
    If encoders is None  : fit from df  (training mode, df must have 'demand').
    If encoders provided : apply them   (inference mode, safe for test data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET       = "demand"
CATEGORICALS = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUMERICS     = ["Temperature", "NumberofLanes"]
TE_K         = 10          # smoothing strength for geohash target encoding

# Fixed column order — train.py and predict.py must agree on this
_FEATURE_COLS = [
    "minutes_since_midnight",
    "hour",
    "sin_slot",
    "cos_slot",
    "geohash_te",
    "geohash_slot_mean",
    "Temperature",
    "NumberofLanes",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
]


# ── internal helpers ──────────────────────────────────────────────────────────

def _parse_timestamp(ts: pd.Series) -> pd.DataFrame:
    """
    Convert 'HH:MM' strings to numeric time features.

    Returns DataFrame with columns:
        minutes_since_midnight  int   0 … 1425
        hour                    int   0 … 23
        sin_slot                float cyclical encoding of time-of-day
        cos_slot                float cyclical encoding of time-of-day
    """
    parts   = ts.str.split(":", expand=True).astype(int)
    minutes = parts[0] * 60 + parts[1]
    angle   = 2.0 * np.pi * minutes / 1440.0
    return pd.DataFrame(
        {
            "minutes_since_midnight": minutes,
            "hour":     parts[0],
            "sin_slot": np.sin(angle),
            "cos_slot": np.cos(angle),
        },
        index=ts.index,
    )


def _fit_encoders(df: pd.DataFrame) -> dict:
    """
    Derive all encoder state from a training DataFrame that contains TARGET.

    Never call on test data (TARGET is required).
    """
    global_mean = float(df[TARGET].mean())

    # ── Geohash smoothed target encoding (m-estimate) ─────────────────────
    stats    = df.groupby("geohash")[TARGET].agg(["count", "mean"])
    gh_means = {
        str(g): float((n * m + TE_K * global_mean) / (n + TE_K))
        for g, (n, m) in stats.iterrows()
    }

    # ── Geohash × slot historical mean (day 48 only, leak-free for test) ──
    day48 = df[df["day"] == 48].copy()
    if day48.empty:        # defensive fallback if called on partial data
        day48 = df.copy()

    day48_ts              = _parse_timestamp(day48["timestamp"])
    day48["_min"]         = day48_ts["minutes_since_midnight"].values

    gh_day48_means: dict[str, float] = (
        day48.groupby("geohash")[TARGET].mean()
        .rename(str).to_dict()
    )
    gh_slot_series = day48.groupby(["geohash", "_min"])[TARGET].mean()
    gh_slot_means: dict[tuple[str, int], float] = {
        (str(g), int(m)): float(v)
        for (g, m), v in gh_slot_series.items()
    }

    # ── Numeric medians ────────────────────────────────────────────────────
    num_medians = {
        col: float(df[col].median())
        for col in NUMERICS
        if col in df.columns
    }

    # ── Categorical label maps ─────────────────────────────────────────────
    # Always include "Missing" so NaN values in test are handled gracefully.
    cat_maps: dict[str, dict[str, int]] = {}
    for col in CATEGORICALS:
        if col not in df.columns:
            continue
        vals = sorted(
            set(df[col].fillna("Missing").unique().tolist()) | {"Missing"}
        )
        cat_maps[col] = {v: i for i, v in enumerate(vals)}

    return {
        "global_mean":    global_mean,
        "gh_means":       gh_means,
        "gh_day48_means": gh_day48_means,
        "gh_slot_means":  gh_slot_means,
        "num_medians":    num_medians,
        "cat_maps":       cat_maps,
        "feature_cols":   _FEATURE_COLS,
        "cat_feature_cols": [c for c in CATEGORICALS if c in _FEATURE_COLS],
    }


# ── public API ────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build the feature matrix for either train or test data.

    Parameters
    ----------
    df       : raw DataFrame (columns as delivered by train.csv / test.csv).
    encoders : pre-fitted encoder dict returned by a previous call with
               encoders=None.  Pass None only when df is training data that
               contains the 'demand' column.

    Returns
    -------
    X        : DataFrame of shape (len(df), len(_FEATURE_COLS)) with columns
               in the fixed order defined by _FEATURE_COLS.
    encoders : the encoder dict (fitted or passed-through unchanged).
    """
    if encoders is None:
        encoders = _fit_encoders(df)

    df = df.copy()

    # ── Time features ──────────────────────────────────────────────────────
    ts = _parse_timestamp(df["timestamp"])
    df["minutes_since_midnight"] = ts["minutes_since_midnight"].values
    df["hour"]     = ts["hour"].values
    df["sin_slot"] = ts["sin_slot"].values
    df["cos_slot"] = ts["cos_slot"].values

    # ── Geohash target encoding ────────────────────────────────────────────
    global_mean  = encoders["global_mean"]
    df["geohash_te"] = (
        df["geohash"].map(encoders["gh_means"]).fillna(global_mean)
    )

    # ── Geohash × slot historical mean (two-level fallback) ───────────────
    gh_slot  = encoders["gh_slot_means"]
    gh_day48 = encoders["gh_day48_means"]
    df["geohash_slot_mean"] = [
        gh_slot.get((str(g), int(m)), gh_day48.get(str(g), global_mean))
        for g, m in zip(df["geohash"], df["minutes_since_midnight"])
    ]

    # ── Numeric imputation ─────────────────────────────────────────────────
    for col in NUMERICS:
        if col in df.columns:
            df[col] = df[col].fillna(encoders["num_medians"].get(col, 0.0))

    # ── Categorical encoding ───────────────────────────────────────────────
    for col in CATEGORICALS:
        if col not in df.columns:
            continue
        cat_map = encoders["cat_maps"].get(col, {})
        missing_code = cat_map.get("Missing", 0)
        df[col] = (
            df[col]
            .fillna("Missing")
            .map(cat_map)
            .fillna(missing_code)   # unseen category → treat as Missing
            .astype(int)
        )

    feature_cols = [c for c in _FEATURE_COLS if c in df.columns]
    return df[feature_cols].reset_index(drop=True), encoders
