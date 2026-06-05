"""
src/features.py
---------------
Feature engineering for Traffic Demand prediction.
Designed for XGBoost (all features numeric — no native categorical support needed).

EDA-driven additions over the baseline:
  - is_major_road  : NumberofLanes >= 4  (perfect split: mean demand 0.61 vs 0.08)
  - is_highway     : RoadType == Highway (mean demand 0.61 vs others)
  - is_street      : RoadType == Street  (mean demand 0.27)
  - gh4_te / gh5_te: coarser geohash prefix target encodings
  - peak_midday    : minutes in [660, 840]  — 11 am – 2 pm, highest demand slot
  - peak_morning   : minutes in [360, 540]  — 6 – 9 am
  - peak_evening   : minutes in [960, 1140] — 4 – 7 pm
  - day_mod7       : day modulo 7 (weekend pattern: days 48/49 are Sat/Sun)
  - lanes_rank     : ordinal rank of NumberofLanes with major-road clamp

Public API
----------
build_features(df, encoders=None) -> (X: DataFrame, encoders: dict)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET       = "demand"
CATEGORICALS: list[str] = []   # XGBoost: all features already numeric
NUMERICS     = ["Temperature", "NumberofLanes"]
TE_K         = 10   # smoothing factor for target encoding (m-estimate)

_FEATURE_COLS = [
    # ── time ──────────────────────────────────────────────────────────
    "minutes_since_midnight",
    "hour",
    "sin_slot",
    "cos_slot",
    "peak_morning",
    "peak_midday",
    "peak_evening",
    "day_mod7",
    # ── space ─────────────────────────────────────────────────────────
    "geohash_te",
    "gh5_te",
    "gh4_te",
    "geohash_slot_mean",
    # ── road ──────────────────────────────────────────────────────────
    "NumberofLanes",
    "is_major_road",
    "lanes_rank",
    "is_highway",
    "is_street",
    "LargeVehicles",
    "Landmarks",
    # ── environment ───────────────────────────────────────────────────
    "Temperature",
    "Weather",
    "RoadType",
]


# ── internal helpers ──────────────────────────────────────────────────────────

def _parse_timestamp(ts: pd.Series) -> pd.DataFrame:
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


def _smoothed_te(series: pd.Series, target: pd.Series, global_mean: float, k: int) -> dict:
    """Return m-estimate smoothed target encoding dict for a categorical series."""
    stats = pd.concat([series, target], axis=1)
    stats.columns = ["key", "y"]
    agg = stats.groupby("key")["y"].agg(["count", "mean"])
    return {
        str(key): float((n * m + k * global_mean) / (n + k))
        for key, (n, m) in agg.iterrows()
    }


def _fit_encoders(df: pd.DataFrame) -> dict:
    global_mean = float(df[TARGET].mean())

    # ── Geohash encodings at 3 granularities ──────────────────────────
    gh_means = _smoothed_te(df["geohash"],                 df[TARGET], global_mean, TE_K)
    gh5_means = _smoothed_te(df["geohash"].str[:5],        df[TARGET], global_mean, TE_K)
    gh4_means = _smoothed_te(df["geohash"].str[:4],        df[TARGET], global_mean, TE_K)

    # ── Geohash × time-slot mean (from day 48, leak-free for test day) ─
    day48 = df[df["day"] == 48].copy()
    if day48.empty:
        day48 = df.copy()
    day48_ts          = _parse_timestamp(day48["timestamp"])
    day48["_min"]     = day48_ts["minutes_since_midnight"].values

    gh_day48_means: dict[str, float] = (
        day48.groupby("geohash")[TARGET].mean().rename(str).to_dict()
    )
    gh_slot_series = day48.groupby(["geohash", "_min"])[TARGET].mean()
    gh_slot_means: dict[tuple[str, int], float] = {
        (str(g), int(m)): float(v)
        for (g, m), v in gh_slot_series.items()
    }

    # ── Numeric medians for imputation ────────────────────────────────
    num_medians = {
        col: float(df[col].median())
        for col in NUMERICS
        if col in df.columns
    }

    # ── Categorical label maps (RoadType, Weather, LargeVehicles, Landmarks) ─
    cat_cols = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
    cat_maps: dict[str, dict[str, int]] = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        vals = sorted(set(df[col].fillna("Missing").unique().tolist()) | {"Missing"})
        cat_maps[col] = {v: i for i, v in enumerate(vals)}

    return {
        "global_mean":    global_mean,
        "gh_means":       gh_means,
        "gh5_means":      gh5_means,
        "gh4_means":      gh4_means,
        "gh_day48_means": gh_day48_means,
        "gh_slot_means":  gh_slot_means,
        "num_medians":    num_medians,
        "cat_maps":       cat_maps,
        "feature_cols":   _FEATURE_COLS,
        "cat_feature_cols": [],   # XGBoost: no native categoricals
    }


# ── public API ────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    if encoders is None:
        encoders = _fit_encoders(df)

    df = df.copy()

    # ── Time features ──────────────────────────────────────────────────
    ts = _parse_timestamp(df["timestamp"])
    df["minutes_since_midnight"] = ts["minutes_since_midnight"].values
    df["hour"]     = ts["hour"].values
    df["sin_slot"] = ts["sin_slot"].values
    df["cos_slot"] = ts["cos_slot"].values

    mins = df["minutes_since_midnight"]
    df["peak_morning"] = ((mins >= 360) & (mins <= 540)).astype(int)
    df["peak_midday"]  = ((mins >= 660) & (mins <= 840)).astype(int)
    df["peak_evening"] = ((mins >= 960) & (mins <= 1140)).astype(int)
    df["day_mod7"]     = df["day"] % 7

    # ── Geohash features ───────────────────────────────────────────────
    global_mean = encoders["global_mean"]

    df["geohash_te"] = df["geohash"].map(encoders["gh_means"]).fillna(global_mean)
    df["gh5_te"]     = df["geohash"].str[:5].map(encoders["gh5_means"]).fillna(global_mean)
    df["gh4_te"]     = df["geohash"].str[:4].map(encoders["gh4_means"]).fillna(global_mean)

    gh_slot  = encoders["gh_slot_means"]
    gh_day48 = encoders["gh_day48_means"]
    df["geohash_slot_mean"] = [
        gh_slot.get((str(g), int(m)), gh_day48.get(str(g), global_mean))
        for g, m in zip(df["geohash"], df["minutes_since_midnight"])
    ]

    # ── Road features (EDA key insight: lanes 4-5 = major highway) ────
    df["NumberofLanes"] = df["NumberofLanes"].fillna(encoders["num_medians"].get("NumberofLanes", 2.0))
    df["is_major_road"] = (df["NumberofLanes"] >= 4).astype(int)
    # ordinal rank: 1-3 → 0/1/2, 4-5 → 3/4 (non-linear boost for majors)
    df["lanes_rank"]    = df["NumberofLanes"].clip(1, 5) - 1

    rt_map     = encoders["cat_maps"].get("RoadType", {})
    rt_missing = rt_map.get("Missing", 0)
    rt_encoded = df["RoadType"].fillna("Missing").map(rt_map).fillna(rt_missing).astype(int)
    highway_code = rt_map.get("Highway", -1)
    street_code  = rt_map.get("Street", -1)
    df["is_highway"] = (rt_encoded == highway_code).astype(int)
    df["is_street"]  = (rt_encoded == street_code).astype(int)
    df["RoadType"]   = rt_encoded

    # ── Other categoricals ─────────────────────────────────────────────
    for col in ["LargeVehicles", "Landmarks", "Weather"]:
        if col not in df.columns:
            continue
        cat_map = encoders["cat_maps"].get(col, {})
        missing_code = cat_map.get("Missing", 0)
        df[col] = (
            df[col].fillna("Missing").map(cat_map).fillna(missing_code).astype(int)
        )

    # ── Temperature imputation ─────────────────────────────────────────
    df["Temperature"] = df["Temperature"].fillna(encoders["num_medians"].get("Temperature", 16.0))

    feature_cols = [c for c in _FEATURE_COLS if c in df.columns]
    return df[feature_cols].reset_index(drop=True), encoders
