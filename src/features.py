"""
src/features.py
---------------
Feature engineering for Traffic Demand prediction.

EDA-driven design:
  - Test data is day 49; train covers day 48 (69k rows) + partial day 49 (7k rows)
  - 88.9% of test geohash+slots have a real lag1 from day 48 → lag is the key signal
  - RoadType/Lanes still dominant for rows with missing lag history
  - Interaction TEs (geohash x hour) capture fine-grained spatial-temporal patterns

All output features are numeric (LightGBM-compatible; no native categorical needed).

Public API
----------
fit_encoders(df)                     -> encoders dict (call on training data only)
build_features(df, encoders=None)    -> (X: DataFrame, encoders: dict)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET       = "demand"
CATEGORICALS: list[str] = []
NUMERICS     = ["Temperature", "NumberofLanes"]

# Target-encoding smoothing strengths (m-estimate)
# Reverted to k=10 for geohash (prevents Saturday→Sunday overfit in holdout)
_K_GH6     = 10   # 6-char geohash  (1249 unique)
_K_GH5     = 10   # 5-char prefix   (56 unique)
_K_GH4     = 10   # 4-char prefix   (6 unique)
# Interaction TEs — moderate smoothing; computed on day 48, applied to day 49
_K_GH_HOUR = 5    # geohash × hour  (~30k combos)
_K_GH_PEAK = 5    # geohash × peak  (~2500 combos)
_K_GH_DOW  = 10   # geohash × day_mod7 (day 49 has only ~7k rows → heavy smooth)
_K_RT_HOUR = 5    # road_type × hour (72 combos, lots of obs)

_FEATURE_COLS = [
    # ── time ──────────────────────────────────────────────────────────
    "minutes_since_midnight", "hour", "sin_slot", "cos_slot",
    "peak_morning", "peak_midday", "peak_evening", "day_mod7",
    # ── spatial ───────────────────────────────────────────────────────
    "geohash_te", "gh5_te", "gh4_te", "geohash_slot_mean",
    # ── road ──────────────────────────────────────────────────────────
    "NumberofLanes", "is_major_road", "lanes_rank",
    "is_highway", "is_street",
    "LargeVehicles", "Landmarks",
    # ── environment ───────────────────────────────────────────────────
    "Temperature", "Weather", "RoadType",
]


# ── internal helpers ──────────────────────────────────────────────────────────

def _parse_timestamp(ts: pd.Series) -> pd.DataFrame:
    parts   = ts.str.split(":", expand=True).astype(int)
    minutes = parts[0] * 60 + parts[1]
    angle   = 2.0 * np.pi * minutes / 1440.0
    return pd.DataFrame({
        "minutes_since_midnight": minutes,
        "hour":     parts[0],
        "sin_slot": np.sin(angle),
        "cos_slot": np.cos(angle),
    }, index=ts.index)


def _smoothed_te(series: pd.Series, target: pd.Series, global_mean: float, k: int) -> dict:
    """M-estimate smoothed target encoding."""
    stats = pd.concat([series.rename("key"), target.rename("y")], axis=1)
    agg   = stats.groupby("key")["y"].agg(["count", "mean"])
    return {
        str(key): float((n * m + k * global_mean) / (n + k))
        for key, (n, m) in agg.iterrows()
    }


def _time_bucket(minutes: pd.Series) -> pd.Series:
    b = pd.Series(0, index=minutes.index, dtype=np.int8)
    b[minutes >= 300]  = 1   # 05:00-07:45  early AM
    b[minutes >= 480]  = 2   # 08:00-09:45  morning rush
    b[minutes >= 600]  = 3   # 10:00-13:45  midday
    b[minutes >= 840]  = 4   # 14:00-16:45  afternoon
    b[minutes >= 1020] = 5   # 17:00-19:45  evening rush
    b[minutes >= 1200] = 6   # 20:00-23:45  evening
    return b


def _build_lag_lookup(df: pd.DataFrame) -> dict:
    """
    Build {(geohash_str, day_int, minute_int): demand_float} from df.
    Called on fit set only (during holdout validation) or full train (for final model).
    """
    ts  = _parse_timestamp(df["timestamp"])
    tmp = pd.DataFrame({
        "geohash": df["geohash"].astype(str).values,
        "day":     df["day"].astype(int).values,
        "minute":  ts["minutes_since_midnight"].astype(int).values,
        TARGET:    df[TARGET].values,
    })
    grouped = tmp.groupby(["geohash", "day", "minute"], sort=False)[TARGET].mean()
    return {(g, int(d), int(m)): float(v) for (g, d, m), v in grouped.items()}


def _apply_lag_features(
    df: pd.DataFrame,
    lag_lookup: dict,
    global_mean: float,
) -> dict[str, np.ndarray]:
    """
    Look up lag-1/2/7 demand for each row using the pre-built lag_lookup dict.
    Falls back to global_mean when history is unavailable.
    """
    ts      = _parse_timestamp(df["timestamp"])
    minutes = ts["minutes_since_midnight"].astype(int).values
    ghs     = df["geohash"].astype(str).values
    days    = df["day"].astype(int).values
    n       = len(df)
    gm      = float(global_mean)

    lag1 = np.empty(n, dtype=np.float32)
    lag2 = np.empty(n, dtype=np.float32)
    lag3 = np.empty(n, dtype=np.float32)
    lag7 = np.empty(n, dtype=np.float32)

    for i in range(n):
        gh = ghs[i]; d = days[i]; m = minutes[i]
        lag1[i] = lag_lookup.get((gh, d - 1, m), gm)
        lag2[i] = lag_lookup.get((gh, d - 2, m), gm)
        lag3[i] = lag_lookup.get((gh, d - 3, m), gm)
        lag7[i] = lag_lookup.get((gh, d - 7, m), gm)

    rolling_mean_3 = (lag1 + lag2 + lag3) / 3.0
    gh_slot_std    = np.stack([lag1, lag2, lag3, lag7], axis=1).std(axis=1)

    return {
        "lag1_demand":    lag1,
        "lag2_demand":    lag2,
        "lag7_demand":    lag7,
        "rolling_mean_3": rolling_mean_3,
        "gh_slot_std":    gh_slot_std,
    }


# ── public API ────────────────────────────────────────────────────────────────

def fit_encoders(df: pd.DataFrame) -> dict:
    """
    Derive all encoder state from a training DataFrame that contains TARGET.
    Call on fit set for holdout validation; on full train for final model.
    """
    global_mean = float(df[TARGET].mean())

    ts       = _parse_timestamp(df["timestamp"])
    minutes  = ts["minutes_since_midnight"]
    hours    = ts["hour"]
    is_peak  = ((minutes >= 660) & (minutes <= 840)).astype(int)
    day_mod7 = df["day"] % 7

    # ── Geohash TEs at 3 granularities ────────────────────────────────
    gh_means  = _smoothed_te(df["geohash"],         df[TARGET], global_mean, _K_GH6)
    gh5_means = _smoothed_te(df["geohash"].str[:5], df[TARGET], global_mean, _K_GH5)
    gh4_means = _smoothed_te(df["geohash"].str[:4], df[TARGET], global_mean, _K_GH4)

    # ── Geohash × slot mean (day 48 data only — leak-free for test) ───
    day48 = df[df["day"] == 48].copy()
    if day48.empty:
        day48 = df.copy()
    day48["_min"]   = _parse_timestamp(day48["timestamp"])["minutes_since_midnight"].astype(int).values
    gh_day48_means  = day48.groupby("geohash")[TARGET].mean().rename(str).to_dict()
    gh_slot_raw     = day48.groupby(["geohash", "_min"])[TARGET].mean()
    gh_slot_means   = {(str(g), int(m)): float(v) for (g, m), v in gh_slot_raw.items()}

    # ── Interaction TEs ────────────────────────────────────────────────
    gh_hour_key  = df["geohash"].astype(str) + "_h" + hours.astype(str)
    gh_peak_key  = df["geohash"].astype(str) + "_p" + is_peak.astype(str)
    gh_dow_key   = df["geohash"].astype(str) + "_d" + day_mod7.astype(str)
    rt_hour_key  = df["RoadType"].fillna("Missing").astype(str) + "_h" + hours.astype(str)

    gh_hour_means = _smoothed_te(gh_hour_key, df[TARGET], global_mean, _K_GH_HOUR)
    gh_peak_means = _smoothed_te(gh_peak_key, df[TARGET], global_mean, _K_GH_PEAK)
    gh_dow_means  = _smoothed_te(gh_dow_key,  df[TARGET], global_mean, _K_GH_DOW)
    rt_hour_means = _smoothed_te(rt_hour_key, df[TARGET], global_mean, _K_RT_HOUR)

    # ── Numeric medians for imputation ─────────────────────────────────
    num_medians = {col: float(df[col].median()) for col in NUMERICS if col in df.columns}

    # ── Categorical label maps ─────────────────────────────────────────
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
        "gh_hour_means":  gh_hour_means,
        "gh_peak_means":  gh_peak_means,
        "gh_dow_means":   gh_dow_means,
        "rt_hour_means":  rt_hour_means,
        "num_medians":    num_medians,
        "cat_maps":       cat_maps,
        "feature_cols":   _FEATURE_COLS,
        "cat_feature_cols": [],
    }


def build_features(
    df: pd.DataFrame,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    if encoders is None:
        encoders = fit_encoders(df)

    df = df.copy()
    gm = encoders["global_mean"]

    # ── Time ──────────────────────────────────────────────────────────
    ts = _parse_timestamp(df["timestamp"])
    df["minutes_since_midnight"] = ts["minutes_since_midnight"].values
    df["hour"]     = ts["hour"].values
    df["sin_slot"] = ts["sin_slot"].values
    df["cos_slot"] = ts["cos_slot"].values

    mins = df["minutes_since_midnight"]
    df["norm_minute"]  = (mins / 1440.0).values
    df["time_bucket"]  = _time_bucket(mins).values
    df["day_mod7"]     = (df["day"] % 7).values
    df["is_weekend"]   = df["day_mod7"].isin([5, 6]).astype(int).values
    df["is_rush_hour"] = ((mins.between(480, 600)) | (mins.between(1020, 1200))).astype(int).values
    df["peak_morning"] = mins.between(360,  540).astype(int).values
    df["peak_midday"]  = mins.between(660,  840).astype(int).values
    df["peak_evening"] = mins.between(960, 1140).astype(int).values

    # ── Spatial ───────────────────────────────────────────────────────
    df["geohash_te"] = df["geohash"].map(encoders["gh_means"]).fillna(gm).values
    df["gh5_te"]     = df["geohash"].str[:5].map(encoders["gh5_means"]).fillna(gm).values
    df["gh4_te"]     = df["geohash"].str[:4].map(encoders["gh4_means"]).fillna(gm).values

    gh_slot  = encoders["gh_slot_means"]
    gh_day48 = encoders["gh_day48_means"]
    df["geohash_slot_mean"] = [
        gh_slot.get((str(g), int(m)), gh_day48.get(str(g), gm))
        for g, m in zip(df["geohash"], df["minutes_since_midnight"])
    ]

    # ── Interaction TEs ────────────────────────────────────────────────
    gh_hour_key = df["geohash"].astype(str) + "_h" + df["hour"].astype(str)
    df["geohash_hour_te"] = gh_hour_key.map(encoders["gh_hour_means"]).fillna(gm).values

    gh_peak_key = df["geohash"].astype(str) + "_p" + df["peak_midday"].astype(str)
    df["geohash_peak_te"] = gh_peak_key.map(encoders["gh_peak_means"]).fillna(gm).values

    gh_dow_key = df["geohash"].astype(str) + "_d" + df["day_mod7"].astype(str)
    df["geohash_dow_te"] = gh_dow_key.map(encoders["gh_dow_means"]).fillna(gm).values

    rt_hour_key = df["RoadType"].fillna("Missing").astype(str) + "_h" + df["hour"].astype(str)
    df["roadtype_hour_te"] = rt_hour_key.map(encoders["rt_hour_means"]).fillna(gm).values

    # ── Road ──────────────────────────────────────────────────────────
    df["NumberofLanes"] = df["NumberofLanes"].fillna(encoders["num_medians"].get("NumberofLanes", 2.0))
    df["is_major_road"] = (df["NumberofLanes"] >= 4).astype(int)
    df["lanes_rank"]    = df["NumberofLanes"].clip(1, 5) - 1

    rt_map = encoders["cat_maps"].get("RoadType", {})
    rt_enc = (df["RoadType"].fillna("Missing")
              .map(rt_map).fillna(rt_map.get("Missing", 0)).astype(int))
    df["is_highway"] = (rt_enc == rt_map.get("Highway", -1)).astype(int)
    df["is_street"]  = (rt_enc == rt_map.get("Street",  -1)).astype(int)
    df["RoadType"]   = rt_enc

    df["lanes_x_rush"]   = df["NumberofLanes"].values * df["is_rush_hour"].values
    df["highway_x_peak"] = df["is_highway"].values    * df["peak_midday"].values

    # ── Other categoricals ─────────────────────────────────────────────
    for col in ["LargeVehicles", "Landmarks", "Weather"]:
        if col not in df.columns:
            continue
        cm = encoders["cat_maps"].get(col, {})
        df[col] = (df[col].fillna("Missing")
                   .map(cm).fillna(cm.get("Missing", 0)).astype(int))

    # ── Temperature ────────────────────────────────────────────────────
    df["Temperature"] = df["Temperature"].fillna(encoders["num_medians"].get("Temperature", 16.0))

    feat_cols = [c for c in _FEATURE_COLS if c in df.columns]
    return df[feat_cols].reset_index(drop=True), encoders
