"""
Demand prediction pipeline for TGSRTC daily operations.

Extracted from notebooks/demand_prediction.ipynb.
Handles feature engineering, model training, evaluation, and prediction
generation using XGBoost.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import json
import yaml
import joblib
import warnings

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MODEL_VERSION = "xgb_v1"

GOLD_PARQUET = DATA_DIR / "processed" / "ops_daily_gold.parquet"
FEATURES_DIR = DATA_DIR / "features"
FEATURES_PARQUET = FEATURES_DIR / "ops_daily_features.parquet"

MODEL_DIR = PROJECT_ROOT / "model" / MODEL_VERSION
MODEL_CONFIG_PATH = MODEL_DIR / "config.yaml"
FEATURES_JSON_PATH = MODEL_DIR / "features.json"
MODEL_FILE_PATH = MODEL_DIR / "xgb_model.joblib"

OUTPUT_DIR = PROJECT_ROOT / "output"
EVALUATIONS_DIR = OUTPUT_DIR / "evaluations"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PREDICTIONS_FILE = PREDICTIONS_DIR / "daily_predictions.parquet"

DEPOT_MASTER_CSV = DATA_DIR / "master" / "depot_master.csv"
HOLIDAY_CALENDAR_CSV = DATA_DIR / "master" / "holiday_calendar.csv"
WEATHER_CACHE = DATA_DIR / "cache" / "weather_daily_by_depot_2023_2025.parquet"

for _d in [FEATURES_DIR, MODEL_DIR, EVALUATIONS_DIR, PREDICTIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Default model config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "data": {
        "features_parquet": str(FEATURES_PARQUET),
        "target_col": "passenger_kms",
        "date_col": "date",
        "group_col": "depot",
    },
    "feature_selection": {
        "mode": "drop",
        "cols_to_drop": [
            "depot", "date", "passengers_per_day", "actual_kms", "occupancy_ratio",
            "passenger_kms",
            "festival_intensity", "Holiday_Festival", "fes_hol_category",
            "baseline_same_dow", "expected_festival_demand",
            "year", "day_of_month", "week_of_year",
            "marriage_day", "moudyami_day",
            "is_fes_hol", "is_weekend",
            "avg_temp", "temp_range", "is_heavy_rain",
            "pkm_lag2_dow_dev",
            "telugu_paksham_encoded", "telugu_month_encoded", "Holiday_Festival_encoded",
            "fes_hol_category_encoded",
            "dow",
        ],
    },
    "split": {
        "method": "last_n_days",
        "test_days": 90,
    },
    "xgb": {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
        "min_child_weight": 5.0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    },
    "outputs": {
        "model_file": "xgb_model.joblib",
        "features_file": "features.json",
    },
    "forecast": {
        "horizon": 2,
    },
    "feature_engineering": {
        "outlier_iqr_mult": 3.0,
        "epoch_date": "2023-01-01",
        "festival": {
            "distance_cap": 30,
            "rebound_days": 4,
            "proximity_before": [1, 2, 3],
            "proximity_after": [1, 2],
            "post_festival_weekend_days": [1, 2, 3],
            "same_dow_window": 3,
            "same_dow_min_periods": 3,
        },
        "weather": {
            "rainy_threshold_mm": 0,
            "heavy_rain_threshold_mm": 10,
        },
        "lags": {
            "shifts": [2, 7, 14, 21],
            "rolling_windows": [
                {"window": 7, "min_periods": 3},
                {"window": 14, "min_periods": 5},
                {"window": 21, "min_periods": 7},
            ],
            "same_dow_rolling": {"window": 3, "min_periods": 2},
        },
    },
    "prediction_defaults": {
        "bus_capacity": 45,
        "assumed_or": 0.85,
        "km_per_bus": 250,
        "rolling_or_window_days": 30,
        "rolling_or_min_points": 5,
        "rolling_or_fallback": 0.85,
    },
    "scheduling_policy": {
        "target_or": 0.75,
        "tolerance_pct": 0.03,
        "lookback_days": 7,
        "max_trip_change_per_service": 1,
        "min_trips_per_service": 0,
        "underutilized_or": 0.65,
        "overloaded_or": 0.95,
        "stop_or": 0.45,
        "morning_peak_start": "06:00",
        "morning_peak_end": "10:00",
        "evening_peak_start": "16:30",
        "evening_peak_end": "20:30",
        "prefer_peak_when_adding": True,
        "protect_peak_when_cutting": True,
        "max_changes_per_route": 2,
        "route_column": "route",
        "max_services_changed": 25,
        "prefer_short_kmpt_when_adding": True,
        "fallback_pkm": 2000000,
        "avg_km_per_passenger": 25,
    },
}

FORECAST_HORIZON: int = DEFAULT_CONFIG["forecast"]["horizon"]

# ---------------------------------------------------------------------------
# Predictions tracking columns
# ---------------------------------------------------------------------------

PREDICTIONS_COLUMNS = [
    "run_date", "prediction_date", "depot",
    "predicted_passenger_kms", "actual_passenger_kms",
    "assumed_or", "actual_or", "estimated_kms", "actual_kms",
    "bus_capacity", "estimated_buses", "actual_buses",
    "pkm_error", "pkm_error_pct", "km_error", "km_error_pct", "status",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_gold_data() -> pd.DataFrame:
    gold_df = pd.read_parquet(GOLD_PARQUET)
    gold_df["date"] = pd.to_datetime(gold_df["date"])
    gold_df = gold_df.dropna(subset=["date"]).reset_index(drop=True)
    target_col = DEFAULT_CONFIG["data"]["target_col"]
    if target_col in gold_df.columns:
        gold_df = gold_df.dropna(subset=[target_col]).reset_index(drop=True)
    return gold_df


def clip_outliers_iqr(gold_df: pd.DataFrame, col: str = "passenger_kms", iqr_mult: float | None = None, config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = DEFAULT_CONFIG
    if iqr_mult is None:
        iqr_mult = config["feature_engineering"]["outlier_iqr_mult"]
    gold_df = gold_df.copy()
    for depot, grp in gold_df.groupby("depot"):
        q1 = grp[col].quantile(0.25)
        q3 = grp[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_mult * iqr
        upper = q3 + iqr_mult * iqr
        mask = gold_df["depot"] == depot
        gold_df.loc[mask, col] = gold_df.loc[mask, col].clip(lower=lower, upper=upper)
    return gold_df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def add_temporal_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = DEFAULT_CONFIG
    epoch_date = config["feature_engineering"]["epoch_date"]
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    for d in range(7):
        df[f"dow_{d}"] = (df["dow"] == d).astype(int)
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    df["day_counter"] = (df["date"] - pd.Timestamp(epoch_date)).dt.days
    return df


def fetch_weather_daily(lat: float, lon: float, start_date: str, end_date: str, timeout: int = 60) -> pd.DataFrame:
    import requests as _requests
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Kolkata",
    }
    resp = _requests.get(ARCHIVE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    daily = resp.json()["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "max_temp": daily["temperature_2m_max"],
        "min_temp": daily["temperature_2m_min"],
        "daily_rainfall": daily["precipitation_sum"],
    })


def fetch_weather_forecast(
    lat: float, lon: float, target_date: str, timeout: int = 30,
) -> dict:
    """Fetch weather forecast from Open-Meteo Forecast API for a single date.

    Returns dict with keys max_temp, min_temp, daily_rainfall or None on failure.
    """
    import requests as _requests

    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Kolkata",
        "start_date": target_date,
        "end_date": target_date,
    }
    try:
        resp = _requests.get(FORECAST_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        daily = resp.json()["daily"]
        return {
            "max_temp": daily["temperature_2m_max"][0],
            "min_temp": daily["temperature_2m_min"][0],
            "daily_rainfall": daily["precipitation_sum"][0],
        }
    except Exception:
        return None


def build_weather_for_all_depots(
    depot_master_csv: Path = DEPOT_MASTER_CSV,
    start_date: str = "2023-01-01",
    requested_end_date: str = "2026-12-31",
    cache_parquet: Path = WEATHER_CACHE,
    use_cache: bool = True,
) -> pd.DataFrame:
    if cache_parquet and use_cache and cache_parquet.exists():
        weather = pd.read_parquet(cache_parquet)
        weather["date"] = pd.to_datetime(weather["date"])
        return weather
    depot_master = pd.read_csv(depot_master_csv)
    depot_master.columns = depot_master.columns.str.lower().str.strip()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    end_date = min(requested_end_date, yesterday)
    all_weather = []
    for _, row in depot_master.iterrows():
        depot = row["depot"]
        lat, lon = row["lat"], row["lon"]
        try:
            df = fetch_weather_daily(lat, lon, start_date, end_date)
            df["depot"] = depot
            all_weather.append(df)
        except Exception:
            pass
    weather = pd.concat(all_weather, ignore_index=True)
    if cache_parquet:
        cache_parquet.parent.mkdir(parents=True, exist_ok=True)
        weather.to_parquet(cache_parquet, index=False)
    return weather


def merge_weather_features(features_df: pd.DataFrame, weather_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = DEFAULT_CONFIG
    weather_cfg = config["feature_engineering"]["weather"]
    rainy_threshold = weather_cfg["rainy_threshold_mm"]
    heavy_rain_threshold = weather_cfg["heavy_rain_threshold_mm"]
    features_df = features_df.merge(
        weather_df[["depot", "date", "max_temp", "min_temp", "daily_rainfall"]],
        on=["depot", "date"],
        how="left",
    )
    features_df["temp_range"] = features_df["max_temp"] - features_df["min_temp"]
    features_df["avg_temp"] = (features_df["max_temp"] + features_df["min_temp"]) / 2
    features_df["is_rainy"] = (features_df["daily_rainfall"] > rainy_threshold).astype(int)
    features_df["is_heavy_rain"] = (features_df["daily_rainfall"] > heavy_rain_threshold).astype(int)
    return features_df


def build_festival_features(
    df: pd.DataFrame,
    target_col: str = "passenger_kms",
    add_proximity_flags: bool = True,
    add_intensity_features: bool = True,
    same_dow_window: int | None = None,
    same_dow_min_periods: int | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    if config is None:
        config = DEFAULT_CONFIG
    fes_cfg = config["feature_engineering"]["festival"]
    if same_dow_window is None:
        same_dow_window = fes_cfg["same_dow_window"]
    if same_dow_min_periods is None:
        same_dow_min_periods = fes_cfg["same_dow_min_periods"]
    distance_cap = fes_cfg["distance_cap"]
    rebound_days = fes_cfg["rebound_days"]
    proximity_before = fes_cfg["proximity_before"]
    proximity_after = fes_cfg["proximity_after"]
    post_festival_weekend_days = fes_cfg["post_festival_weekend_days"]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["depot", "date"]).reset_index(drop=True)
    if "dow" not in df.columns:
        df["dow"] = df["date"].dt.dayofweek
    if "is_fes_hol" not in df.columns:
        df["is_fes_hol"] = (
            df["fes_hol_code"].notna() & (df["fes_hol_code"] != 0)
        ).astype(int)
    if add_proximity_flags:
        g = df.groupby("depot", group_keys=False)["is_fes_hol"]
        for n in proximity_before:
            df[f"fes_hol_minus_{n}"] = g.shift(-n).fillna(0).astype(int)
        for n in proximity_after:
            df[f"fes_hol_plus_{n}"] = g.shift(n).fillna(0).astype(int)

    # Continuous festival distance features (per depot)
    df["days_to_next_festival"] = distance_cap
    df["days_since_last_festival"] = distance_cap
    df["next_fes_hol_code"] = 0
    df["next_festival_cluster_len"] = 0
    for depot in df["depot"].unique():
        depot_mask = df["depot"] == depot
        depot_data = df.loc[depot_mask].sort_values("date")
        is_fes = depot_data["is_fes_hol"].values
        fes_codes = depot_data["fes_hol_code"].values
        n = len(is_fes)
        # days_to_next_festival + next_fes_hol_code: forward scan
        dtf = np.full(n, distance_cap, dtype=int)
        nfc = np.zeros(n, dtype=int)
        last_seen = distance_cap
        last_code = 0
        for i in range(n - 1, -1, -1):
            if is_fes[i] == 1:
                last_seen = 0
                last_code = int(fes_codes[i]) if not np.isnan(fes_codes[i]) else 0
            dtf[i] = last_seen
            nfc[i] = last_code
            if last_seen < distance_cap:
                last_seen += 1
        # days_since_last_festival: backward scan
        dsf = np.full(n, distance_cap, dtype=int)
        last_seen = distance_cap
        for i in range(n):
            if is_fes[i] == 1:
                last_seen = 0
            dsf[i] = last_seen
            if last_seen < distance_cap:
                last_seen += 1
        # next_festival_cluster_len: cluster length of the upcoming festival
        # Step 1: compute cluster length at each festival day
        cls = np.zeros(n, dtype=int)
        i_cls = 0
        while i_cls < n:
            if is_fes[i_cls] == 1:
                start = i_cls
                while i_cls < n and is_fes[i_cls] == 1:
                    i_cls += 1
                length = i_cls - start
                for j in range(start, i_cls):
                    cls[j] = length
            else:
                i_cls += 1
        # Step 2: forward-fill — for each non-festival day, carry the cluster_len of the next festival
        nfcl = np.zeros(n, dtype=int)
        current_len = 0
        for i in range(n - 1, -1, -1):
            if is_fes[i] == 1:
                current_len = cls[i]
            nfcl[i] = current_len
        df.loc[depot_data.index, "days_to_next_festival"] = dtf
        df.loc[depot_data.index, "days_since_last_festival"] = dsf
        df.loc[depot_data.index, "next_fes_hol_code"] = nfc
        df.loc[depot_data.index, "next_festival_cluster_len"] = nfcl

    # Post-festival days: count days since last festival ended (0=festival, 1-N=rebound, -1=not near)
    df["post_festival_days"] = -1
    for depot in df["depot"].unique():
        depot_mask = df["depot"] == depot
        depot_data = df.loc[depot_mask].sort_values("date")
        is_fes = depot_data["is_fes_hol"].values
        pfd = np.full(len(is_fes), -1)
        last_festival_idx = -1
        for i in range(len(is_fes)):
            if is_fes[i] == 1:
                pfd[i] = 0
                last_festival_idx = i
            elif last_festival_idx >= 0:
                days_since = i - last_festival_idx
                if days_since <= rebound_days:
                    pfd[i] = days_since
        df.loc[depot_data.index, "post_festival_days"] = pfd

    # Festival cluster length: length of consecutive festival block, carried to rebound days
    df["festival_cluster_len"] = 0
    for depot in df["depot"].unique():
        depot_mask = df["depot"] == depot
        depot_data = df.loc[depot_mask].sort_values("date")
        is_fes = depot_data["is_fes_hol"].values
        cluster_len = np.zeros(len(is_fes), dtype=int)
        i = 0
        while i < len(is_fes):
            if is_fes[i] == 1:
                start = i
                while i < len(is_fes) and is_fes[i] == 1:
                    i += 1
                length = i - start
                for j in range(start, i):
                    cluster_len[j] = length
                for j in range(i, min(i + rebound_days, len(is_fes))):
                    if is_fes[j] == 0:
                        cluster_len[j] = length
                    else:
                        break
            else:
                i += 1
        df.loc[depot_data.index, "festival_cluster_len"] = cluster_len

    # Post-festival weekend: rebound day in configured days AND weekend
    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["post_festival_weekend"] = (
        (df["post_festival_days"].isin(post_festival_weekend_days)) & (df["is_weekend"] == 1)
    ).astype(int)

    # Pre-festival weekend: weekend day within 7 days before a festival, not a festival day itself
    df["pre_festival_weekend"] = (
        (df["is_weekend"] == 1)
        & (df["days_to_next_festival"] > 0)
        & (df["days_to_next_festival"] <= 7)
        & (df["is_fes_hol"] == 0)
    ).astype(int)

    # Post-festival return workday: first workday(s) in the post-festival rebound window
    df["post_festival_return_workday"] = (
        (df["is_weekend"] == 0)
        & (df["post_festival_days"] > 0)
        & (df["post_festival_days"] <= rebound_days)
    ).astype(int)

    # Interaction: weekend × upcoming cluster size (e.g. Sankranti=3 vs Republic Day=1)
    df["pre_festival_wknd_x_cluster"] = df["pre_festival_weekend"] * df["next_festival_cluster_len"]
    # Interaction: post-festival weekend × cluster size of the festival that just ended
    df["post_festival_wknd_x_cluster"] = df["post_festival_weekend"] * df["festival_cluster_len"]

    if add_intensity_features and target_col in df.columns:
        df["baseline_same_dow"] = np.nan
        for depot in df["depot"].unique():
            depot_mask = df["depot"] == depot
            depot_data = df.loc[depot_mask].sort_values("date")
            for dow in range(7):
                dow_mask = depot_data["dow"] == dow
                dow_indices = depot_data.loc[dow_mask].index
                baseline_values = (
                    depot_data.loc[dow_mask, target_col]
                    .shift(1)
                    .rolling(window=same_dow_window, min_periods=same_dow_min_periods)
                    .median()
                )
                df.loc[dow_indices, "baseline_same_dow"] = baseline_values.values
        df["festival_intensity"] = np.where(
            (df["is_fes_hol"] == 1) & (df["baseline_same_dow"] > 0),
            df[target_col] / df["baseline_same_dow"],
            np.nan,
        )
        # Per-depot expanding median (leak-free)
        df["expected_intensity"] = np.nan
        for depot in df["depot"].unique():
            depot_mask = df["depot"] == depot
            depot_data = df.loc[depot_mask].sort_values("date")
            for code in depot_data.loc[depot_data["is_fes_hol"] == 1, "fes_hol_code"].unique():
                code_mask = depot_data["fes_hol_code"] == code
                code_indices = depot_data.loc[code_mask].index
                intensities = depot_data.loc[code_mask, "festival_intensity"]
                # expanding median of PAST occurrences only (shift 1 to exclude current)
                expanding_med = intensities.shift(1).expanding(min_periods=1).median()
                df.loc[code_indices, "expected_intensity"] = expanding_med.values
            # Fill NaN (first occurrence of each festival in each depot) with global fallback
            still_nan = depot_mask & df["expected_intensity"].isna()
            if still_nan.any():
                global_profile = df.loc[
                    (df["is_fes_hol"] == 1) & df["festival_intensity"].notna()
                ].groupby("fes_hol_code")["festival_intensity"].median()
                df.loc[still_nan, "expected_intensity"] = (
                    df.loc[still_nan, "fes_hol_code"].map(global_profile).fillna(1.0)
                )
        df["expected_intensity"] = df["expected_intensity"].fillna(1.0)
        df["expected_festival_demand"] = df["baseline_same_dow"] * df["expected_intensity"]

        # Target encoding: mean PKM per (depot, fes_hol_code), leak-free expanding mean
        df["depot_festival_te"] = np.nan
        for depot in df["depot"].unique():
            depot_mask = df["depot"] == depot
            depot_data = df.loc[depot_mask].sort_values("date")
            for code in depot_data.loc[depot_data["is_fes_hol"] == 1, "fes_hol_code"].unique():
                code_mask = depot_data["fes_hol_code"] == code
                code_indices = depot_data.loc[code_mask].index
                expanding_mean = (
                    depot_data.loc[code_mask, target_col]
                    .shift(1)
                    .expanding(min_periods=1)
                    .mean()
                )
                df.loc[code_indices, "depot_festival_te"] = expanding_mean.values
        # Non-festival days get the depot's overall mean
        df["depot_festival_te"] = df["depot_festival_te"].fillna(
            df.groupby("depot")[target_col].transform("mean")
        )

        # Target encoding: mean PKM per (depot, fes_hol_category), leak-free expanding mean
        if "fes_hol_category" in df.columns:
            df["fes_hol_category_te"] = np.nan
            for depot in df["depot"].unique():
                depot_mask = df["depot"] == depot
                depot_data = df.loc[depot_mask].sort_values("date")
                for cat in depot_data["fes_hol_category"].dropna().unique():
                    if cat == "NONE" or cat == "" or pd.isna(cat):
                        continue
                    cat_mask = depot_data["fes_hol_category"] == cat
                    cat_indices = depot_data.loc[cat_mask].index
                    expanding_mean = (
                        depot_data.loc[cat_mask, target_col]
                        .shift(1)
                        .expanding(min_periods=1)
                        .mean()
                    )
                    df.loc[cat_indices, "fes_hol_category_te"] = expanding_mean.values
            df["fes_hol_category_te"] = df["fes_hol_category_te"].fillna(
                df.groupby("depot")[target_col].transform("mean")
            )

        # Post-festival intensity: depot-specific rebound multiplier
        # Ratio of actual / baseline_same_dow on rebound days, expanding median
        df["post_festival_intensity"] = 1.0
        for depot in df["depot"].unique():
            depot_mask = df["depot"] == depot
            depot_data = df.loc[depot_mask].sort_values("date")
            pf_mask = depot_data["post_festival_days"].isin(
                list(range(1, rebound_days + 1))
            )
            pf_indices = depot_data.loc[pf_mask].index
            if len(pf_indices) == 0:
                continue
            baseline_vals = depot_data.loc[pf_mask, "baseline_same_dow"]
            actual_vals = depot_data.loc[pf_mask, target_col]
            pf_ratio = pd.Series(np.nan, index=pf_indices)
            valid = baseline_vals > 0
            pf_ratio.loc[valid[valid].index] = (
                actual_vals.loc[valid[valid].index] / baseline_vals.loc[valid[valid].index]
            )
            expanding_med = pf_ratio.shift(1).expanding(min_periods=1).median()
            df.loc[pf_indices, "post_festival_intensity"] = expanding_med.values

        # Pre-festival intensity: depot-specific pre-festival multiplier
        # Days that are pre-festival (fes_hol_minus_1/2/3 == 1) but not a festival day
        df["pre_festival_intensity"] = 1.0
        pre_flag_cols = [f"fes_hol_minus_{n}" for n in proximity_before]
        existing_pre_cols = [c for c in pre_flag_cols if c in df.columns]
        if existing_pre_cols:
            for depot in df["depot"].unique():
                depot_mask = df["depot"] == depot
                depot_data = df.loc[depot_mask].sort_values("date")
                is_pre = (depot_data[existing_pre_cols].sum(axis=1) > 0) & (depot_data["is_fes_hol"] == 0)
                pre_indices = depot_data.loc[is_pre].index
                if len(pre_indices) == 0:
                    continue
                baseline_vals = depot_data.loc[is_pre, "baseline_same_dow"]
                actual_vals = depot_data.loc[is_pre, target_col]
                pre_ratio = pd.Series(np.nan, index=pre_indices)
                valid = baseline_vals > 0
                pre_ratio.loc[valid[valid].index] = (
                    actual_vals.loc[valid[valid].index] / baseline_vals.loc[valid[valid].index]
                )
                expanding_med = pre_ratio.shift(1).expanding(min_periods=1).median()
                df.loc[pre_indices, "pre_festival_intensity"] = expanding_med.values

    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "passenger_kms", config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = DEFAULT_CONFIG
    forecast_horizon = config["forecast"]["horizon"]
    lags_cfg = config["feature_engineering"]["lags"]
    shifts = lags_cfg["shifts"]
    rolling_windows = lags_cfg["rolling_windows"]
    same_dow_cfg = lags_cfg["same_dow_rolling"]

    df = df.copy()
    df = df.sort_values(["depot", "date"]).reset_index(drop=True)
    g = df.groupby("depot", group_keys=False)[target_col]

    # Dynamic lag columns
    for s in shifts:
        df[f"pkm_lag_{s}"] = g.shift(s)

    # Dynamic rolling windows
    shifted = g.shift(forecast_horizon)
    for i, rw in enumerate(rolling_windows):
        w, mp = rw["window"], rw["min_periods"]
        df[f"pkm_roll{w}_mean"] = shifted.rolling(window=w, min_periods=mp).mean()
        if i == 0:
            df[f"pkm_roll{w}_std"] = shifted.rolling(window=w, min_periods=mp).std()

    # Momentum: first rolling window / last rolling window
    first_w = rolling_windows[0]["window"]
    last_w = rolling_windows[-1]["window"]
    df[f"pkm_momentum_{first_w}_{last_w}"] = df[f"pkm_roll{first_w}_mean"] / df[f"pkm_roll{last_w}_mean"]

    # Same-DOW median
    sdow_window = same_dow_cfg["window"]
    sdow_min = same_dow_cfg["min_periods"]
    df[f"pkm_same_dow_{sdow_window}med"] = np.nan
    for depot in df["depot"].unique():
        depot_mask = df["depot"] == depot
        depot_data = df.loc[depot_mask].sort_values("date")
        for dow in range(7):
            dow_mask = depot_data["dow"] == dow
            dow_indices = depot_data.loc[dow_mask].index
            median_values = (
                depot_data.loc[dow_mask, target_col]
                .shift(1)
                .rolling(window=sdow_window, min_periods=sdow_min)
                .median()
            )
            df.loc[dow_indices, f"pkm_same_dow_{sdow_window}med"] = median_values.values

    # Deviation: first shift vs same-DOW median
    first_shift = shifts[0]
    df[f"pkm_lag{first_shift}_dow_dev"] = df[f"pkm_lag_{first_shift}"] - df[f"pkm_same_dow_{sdow_window}med"]

    # lag_is_festival: flag if the date used for pkm_lag_{horizon} was a festival day
    if "is_fes_hol" in df.columns:
        df[f"lag{forecast_horizon}_is_festival"] = (
            df.groupby("depot", group_keys=False)["is_fes_hol"]
            .shift(forecast_horizon)
            .fillna(0)
            .astype(int)
        )
    return df


def add_target_encoding(df: pd.DataFrame, target_col: str = "passenger_kms", cat_col: str = "depot", config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = DEFAULT_CONFIG
    forecast_horizon = config["forecast"]["horizon"]
    df = df.copy()
    df = df.sort_values([cat_col, "date"]).reset_index(drop=True)
    df["depot_te"] = (
        df.groupby(cat_col, group_keys=False)[target_col]
        .apply(lambda x: x.expanding().mean().shift(forecast_horizon))
    )
    global_mean = df[target_col].mean()
    df["depot_te"] = df["depot_te"].fillna(global_mean)
    return df


def add_depot_dow_te(df: pd.DataFrame, target_col: str = "passenger_kms", config: dict | None = None) -> pd.DataFrame:
    """Depot × day-of-week target encoding (expanding mean, leak-free)."""
    if config is None:
        config = DEFAULT_CONFIG
    forecast_horizon = config["forecast"]["horizon"]
    df = df.copy()
    df = df.sort_values(["depot", "date"]).reset_index(drop=True)
    if "dow" not in df.columns:
        df["dow"] = df["date"].dt.dayofweek
    df["depot_dow_te"] = np.nan
    for depot in df["depot"].unique():
        depot_mask = df["depot"] == depot
        depot_data = df.loc[depot_mask].sort_values("date")
        for dow in range(7):
            dow_mask = depot_data["dow"] == dow
            dow_indices = depot_data.loc[dow_mask].index
            if len(dow_indices) == 0:
                continue
            expanding_mean = (
                depot_data.loc[dow_mask, target_col]
                .shift(forecast_horizon)
                .expanding(min_periods=1)
                .mean()
            )
            df.loc[dow_indices, "depot_dow_te"] = expanding_mean.values
    # Fill NaN with depot_te if available, else global mean
    if "depot_te" in df.columns:
        df["depot_dow_te"] = df["depot_dow_te"].fillna(df["depot_te"])
    else:
        df["depot_dow_te"] = df["depot_dow_te"].fillna(df[target_col].mean())
    return df


def add_depot_month_te(df: pd.DataFrame, target_col: str = "passenger_kms", config: dict | None = None) -> pd.DataFrame:
    """Depot × month target encoding (expanding mean, leak-free)."""
    if config is None:
        config = DEFAULT_CONFIG
    forecast_horizon = config["forecast"]["horizon"]
    df = df.copy()
    df = df.sort_values(["depot", "date"]).reset_index(drop=True)
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    df["depot_month_te"] = np.nan
    for depot in df["depot"].unique():
        depot_mask = df["depot"] == depot
        depot_data = df.loc[depot_mask].sort_values("date")
        for month in range(1, 13):
            month_mask = depot_data["month"] == month
            month_indices = depot_data.loc[month_mask].index
            if len(month_indices) == 0:
                continue
            expanding_mean = (
                depot_data.loc[month_mask, target_col]
                .shift(forecast_horizon)
                .expanding(min_periods=1)
                .mean()
            )
            df.loc[month_indices, "depot_month_te"] = expanding_mean.values
    # Fill NaN with depot_te if available, else global mean
    if "depot_te" in df.columns:
        df["depot_month_te"] = df["depot_month_te"].fillna(df["depot_te"])
    else:
        df["depot_month_te"] = df["depot_month_te"].fillna(df[target_col].mean())
    return df


def build_all_features(gold_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = DEFAULT_CONFIG
    features_df = gold_df.copy()
    features_df = features_df.sort_values(["depot", "date"]).reset_index(drop=True)
    features_df = add_temporal_features(features_df, config=config)
    weather_df = build_weather_for_all_depots()
    features_df = merge_weather_features(features_df, weather_df, config=config)
    features_df = build_festival_features(features_df, config=config)
    features_df = add_lag_features(features_df, config=config)
    features_df = add_target_encoding(features_df, config=config)
    features_df = add_depot_dow_te(features_df, config=config)
    features_df = add_depot_month_te(features_df, config=config)
    features_df = features_df.sort_values(["depot", "date"]).reset_index(drop=True)
    features_df.attrs.clear()
    features_df.to_parquet(FEATURES_PARQUET, index=False)
    return features_df


# ---------------------------------------------------------------------------
# Model config, preparation, training, evaluation
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Missing keys in *override* fall back to *base* automatically.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_model_config(config_path: Path = MODEL_CONFIG_PATH) -> dict:
    if config_path.exists():
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f) or {}
        config = _deep_merge(DEFAULT_CONFIG, file_config)
    else:
        config = DEFAULT_CONFIG.copy()
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config


def prepare_model_data(config: dict) -> tuple[pd.DataFrame, list[str], str]:
    model_df = pd.read_parquet(FEATURES_PARQUET)
    model_df["date"] = pd.to_datetime(model_df["date"])
    categorical_cols = model_df.select_dtypes(include=["string", "object"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ["depot", "date"]]
    for col in categorical_cols:
        le = LabelEncoder()
        model_df[col] = model_df[col].fillna("UNKNOWN")
        model_df[f"{col}_encoded"] = le.fit_transform(model_df[col])
    target = config["data"]["target_col"]
    cols_to_drop = config["feature_selection"]["cols_to_drop"]
    exclude_cols = set(cols_to_drop + categorical_cols + [target])
    feature_cols = [
        col for col in model_df.columns
        if col not in exclude_cols
        and model_df[col].dtype in ["int64", "int32", "float64", "int8", "Int64", "float32"]
    ]
    return model_df, feature_cols, target


def split_train_test(model_df: pd.DataFrame, feature_cols: list[str], target: str, test_days: int = 90):
    model_df_sorted = model_df.sort_values(["date", "depot"]).reset_index(drop=True)
    unique_dates = model_df_sorted["date"].unique()
    cutoff_date = unique_dates[-test_days]
    train_mask = model_df_sorted["date"] < cutoff_date
    test_mask = model_df_sorted["date"] >= cutoff_date
    train_df = model_df_sorted[train_mask].copy()
    test_df = model_df_sorted[test_mask].copy()
    train_median = train_df[feature_cols].median()
    X_train = train_df[feature_cols].fillna(train_median)
    y_train = train_df[target]
    X_test = test_df[feature_cols].fillna(train_median)
    y_test = test_df[target]
    return X_train, y_train, X_test, y_test, train_df, test_df, cutoff_date, train_median


def train_model(X_train, y_train, X_test, y_test, xgb_params: dict):
    params = xgb_params.copy()
    early_stopping_rounds = params.pop("early_stopping_rounds", None)
    if early_stopping_rounds:
        params["early_stopping_rounds"] = early_stopping_rounds
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    return model


def calculate_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape), "R2": float(r2)}


def save_evaluation_results(
    model, feature_cols, train_metrics, test_metrics,
    X_train, X_test, test_df, target, cutoff_date, training_time,
):
    # Save model
    joblib.dump(model, MODEL_FILE_PATH)
    # Save features.json
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    features_metadata = {
        "version": MODEL_VERSION,
        "created_date": datetime.now().isoformat(),
        "target": target,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "feature_importances": importance_df.set_index("feature")["importance"].to_dict(),
    }
    with open(FEATURES_JSON_PATH, "w") as f:
        json.dump(features_metadata, f, indent=2)
    # Overall metrics
    metrics_overall = {
        "model_version": MODEL_VERSION,
        "evaluation_date": datetime.now().isoformat(),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": len(feature_cols),
        "cutoff_date": str(cutoff_date.date()) if hasattr(cutoff_date, "date") else str(cutoff_date),
        "training_time_seconds": training_time,
    }
    with open(EVALUATIONS_DIR / "metrics_overall.json", "w") as f:
        json.dump(metrics_overall, f, indent=2)
    # Per-depot metrics
    depot_metrics = []
    for depot in test_df["depot"].unique():
        depot_data = test_df[test_df["depot"] == depot]
        metrics = calculate_metrics(depot_data[target].values, depot_data["predicted"].values)
        metrics["depot"] = depot
        metrics["n_rows"] = len(depot_data)
        depot_metrics.append(metrics)
    depot_metrics_df = pd.DataFrame(depot_metrics)
    depot_metrics_df = depot_metrics_df[["depot", "n_rows", "RMSE", "MAE", "MAPE", "R2"]]
    depot_metrics_df.to_csv(EVALUATIONS_DIR / "metrics_per_depot.csv", index=False)
    # Test predictions
    test_df[["depot", "date", target, "predicted", "error", "pct_error"]].to_parquet(
        EVALUATIONS_DIR / "test_predictions.parquet", index=False,
    )
    # Evaluation history
    eval_history_path = EVALUATIONS_DIR / "evaluation_history.csv"
    history_record = {
        "eval_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": MODEL_VERSION,
        "train_rmse": train_metrics["RMSE"],
        "train_mape": train_metrics["MAPE"],
        "train_r2": train_metrics["R2"],
        "test_rmse": test_metrics["RMSE"],
        "test_mape": test_metrics["MAPE"],
        "test_r2": test_metrics["R2"],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": len(feature_cols),
        "cutoff_date": str(cutoff_date.date()) if hasattr(cutoff_date, "date") else str(cutoff_date),
    }
    if eval_history_path.exists():
        history_df = pd.read_csv(eval_history_path)
    else:
        history_df = pd.DataFrame()
    history_df = pd.concat([history_df, pd.DataFrame([history_record])], ignore_index=True)
    history_df.to_csv(eval_history_path, index=False)


# ---------------------------------------------------------------------------
# Predictions tracking
# ---------------------------------------------------------------------------


def load_predictions_file(file_path: Path = PREDICTIONS_FILE) -> pd.DataFrame:
    if file_path.exists():
        df = pd.read_parquet(file_path)
        for col in PREDICTIONS_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df
    return pd.DataFrame(columns=PREDICTIONS_COLUMNS)


def save_predictions_file(df: pd.DataFrame, file_path: Path = PREDICTIONS_FILE) -> None:
    for col in ["run_date", "prediction_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    df.to_parquet(file_path, index=False)


def compute_depot_rolling_or(gold_df: pd.DataFrame, config: dict | None = None) -> dict[str, float]:
    """Compute each depot's rolling average occupancy ratio from recent gold data.

    Returns {depot: rolling_or}. Falls back to config default if insufficient data.
    """
    if config is None:
        config = DEFAULT_CONFIG
    pred_cfg = config["prediction_defaults"]
    window_days = pred_cfg["rolling_or_window_days"]
    min_points = pred_cfg["rolling_or_min_points"]
    fallback = pred_cfg["rolling_or_fallback"]

    if "occupancy_ratio" not in gold_df.columns:
        return {}
    result: dict[str, float] = {}
    cutoff = gold_df["date"].max() - timedelta(days=window_days)
    recent = gold_df[gold_df["date"] >= cutoff]
    for depot in gold_df["depot"].unique():
        depot_data = recent[recent["depot"] == depot]["occupancy_ratio"].dropna()
        if len(depot_data) >= min_points:
            result[depot] = round(float(depot_data.mean()), 4)
        else:
            result[depot] = fallback
    return result


def add_predictions(predictions_df, run_date, prediction_date, depot_predictions, assumed_or=None, bus_capacity=None, depot_or_dict=None, config: dict | None = None):
    if config is None:
        config = DEFAULT_CONFIG
    pred_cfg = config["prediction_defaults"]
    if assumed_or is None:
        assumed_or = pred_cfg["assumed_or"]
    if bus_capacity is None:
        bus_capacity = pred_cfg["bus_capacity"]
    km_per_bus = pred_cfg["km_per_bus"]

    run_dt = pd.to_datetime(run_date)
    pred_dt = pd.to_datetime(prediction_date)
    new_records = []
    for depot, predicted_pkm in depot_predictions.items():
        or_value = depot_or_dict.get(depot, assumed_or) if depot_or_dict else assumed_or
        estimated_kms = predicted_pkm / (or_value * bus_capacity)
        estimated_buses = int(np.ceil(estimated_kms / km_per_bus))
        record = {
            "run_date": run_dt, "prediction_date": pred_dt, "depot": depot,
            "predicted_passenger_kms": predicted_pkm, "actual_passenger_kms": None,
            "assumed_or": or_value, "actual_or": None,
            "estimated_kms": estimated_kms, "actual_kms": None,
            "bus_capacity": bus_capacity, "estimated_buses": estimated_buses, "actual_buses": None,
            "pkm_error": None, "pkm_error_pct": None,
            "km_error": None, "km_error_pct": None, "status": "pending",
        }
        new_records.append(record)
    if len(predictions_df) > 0:
        existing = predictions_df[predictions_df["prediction_date"] == pred_dt]["depot"].tolist()
        new_records = [r for r in new_records if r["depot"] not in existing]
    if new_records:
        predictions_df = pd.concat([predictions_df, pd.DataFrame(new_records)], ignore_index=True)
    return predictions_df


def backfill_test_predictions(predictions_df: pd.DataFrame, test_df: pd.DataFrame, target: str, depot_or_dict: dict | None = None, config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = DEFAULT_CONFIG
    pred_cfg = config["prediction_defaults"]
    bus_capacity = pred_cfg["bus_capacity"]
    assumed_or = pred_cfg["assumed_or"]
    km_per_bus = pred_cfg["km_per_bus"]
    forecast_horizon = config["forecast"]["horizon"]

    backfill_records = []
    for _, row in test_df.iterrows():
        pred_date = pd.to_datetime(row["date"])
        run_date = pred_date - timedelta(days=forecast_horizon - 1)
        predicted = row["predicted"]
        actual = row[target]
        actual_kms_val = row.get("actual_kms", None)
        actual_or_val = row.get("occupancy_ratio", None)
        or_value = depot_or_dict.get(row["depot"], assumed_or) if depot_or_dict else assumed_or
        estimated_kms = predicted / (or_value * bus_capacity)
        estimated_buses = int(np.ceil(estimated_kms / km_per_bus))
        pkm_error = predicted - actual
        pkm_error_pct = (pkm_error / actual * 100) if actual and actual > 0 else None
        km_error = None
        km_error_pct = None
        if actual_kms_val and actual_kms_val > 0:
            km_error = estimated_kms - actual_kms_val
            km_error_pct = km_error / actual_kms_val * 100
        backfill_records.append({
            "run_date": run_date, "prediction_date": pred_date, "depot": row["depot"],
            "predicted_passenger_kms": predicted, "actual_passenger_kms": actual,
            "assumed_or": or_value, "actual_or": actual_or_val,
            "estimated_kms": estimated_kms, "actual_kms": actual_kms_val,
            "bus_capacity": bus_capacity, "estimated_buses": estimated_buses, "actual_buses": None,
            "pkm_error": pkm_error, "pkm_error_pct": pkm_error_pct,
            "km_error": km_error, "km_error_pct": km_error_pct, "status": "completed",
        })
    backfill_df = pd.DataFrame(backfill_records)
    existing_completed = predictions_df[predictions_df["status"] == "completed"].copy()
    if len(existing_completed) > 0:
        existing_completed["_key"] = (
            pd.to_datetime(existing_completed["prediction_date"]).dt.strftime("%Y-%m-%d")
            + "_" + existing_completed["depot"].astype(str)
        )
        backfill_df["_key"] = (
            pd.to_datetime(backfill_df["prediction_date"]).dt.strftime("%Y-%m-%d")
            + "_" + backfill_df["depot"].astype(str)
        )
        backfill_df = backfill_df[~backfill_df["_key"].isin(existing_completed["_key"])]
        backfill_df = backfill_df.drop(columns=["_key"])
    if len(backfill_df) > 0:
        predictions_df = pd.concat([predictions_df, backfill_df], ignore_index=True)
    return predictions_df


# ---------------------------------------------------------------------------
# Holiday calendar lookup
# ---------------------------------------------------------------------------


def load_holiday_calendar(csv_path: Path = HOLIDAY_CALENDAR_CSV) -> dict[date, dict]:
    """Return {date: {fes_hol_code, Holiday_Festival, fes_hol_category}} for all years."""
    cal = pd.read_csv(csv_path, encoding="utf-8-sig")
    date_cols = [c for c in cal.columns if c.endswith("_dates")]
    lookup: dict[date, dict] = {}
    for _, row in cal.iterrows():
        for dc in date_cols:
            raw = row[dc]
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            try:
                dt = pd.to_datetime(str(raw).strip(), dayfirst=True).date()
            except Exception:
                continue
            lookup[dt] = {
                "fes_hol_code": int(row["fes_hol_code"]),
                "Holiday_Festival": row["Holiday_Festival"],
                "fes_hol_category": row["fes_hol_category"],
            }
    return lookup


# ---------------------------------------------------------------------------
# Future feature construction for T+2 prediction
# ---------------------------------------------------------------------------


def construct_future_features(
    target_date: date,
    gold_df: pd.DataFrame,
    feature_cols: list[str],
    train_median: pd.Series,
    target_col: str = "passenger_kms",
    festival_profile: dict | None = None,
    depot_festival_profile: dict | None = None,
    global_festival_profile: dict | None = None,
    post_festival_profile: dict | None = None,
    pre_festival_profile: dict | None = None,
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature rows for *target_date* (T+horizon) using only data up to T.

    Parameters
    ----------
    target_date : date
        The date we are predicting for (T + forecast_horizon).
    gold_df : pd.DataFrame
        Historical data with at least [depot, date, target_col].
        Max date in gold_df is T.
    feature_cols : list[str]
        Ordered feature names the model expects.
    train_median : pd.Series
        Median values per feature from training set, used to fill NaN.
    target_col : str
        Name of the target column in gold_df.
    festival_profile : dict | None
        Deprecated global festival profile {fes_hol_code: intensity}.
        Used as fallback if depot_festival_profile is not provided.
    depot_festival_profile : dict | None
        Per-depot festival intensity: {depot: {fes_hol_code: median_intensity}}.
    global_festival_profile : dict | None
        Global fallback: {fes_hol_code: median_intensity}.
    post_festival_profile : dict | None
        Per-depot post-festival rebound multiplier: {depot: median_ratio}.
    pre_festival_profile : dict | None
        Per-depot pre-festival multiplier: {depot: median_ratio}.
    config : dict | None
        Pipeline configuration. Falls back to DEFAULT_CONFIG when None.

    Returns
    -------
    (info_df, X_future) where info_df has depot + date + raw features,
    and X_future is the model-ready array aligned to feature_cols.
    """
    if config is None:
        config = DEFAULT_CONFIG
    forecast_horizon = config["forecast"]["horizon"]
    fe_cfg = config["feature_engineering"]
    epoch_date = fe_cfg["epoch_date"]
    fes_cfg = fe_cfg["festival"]
    distance_cap = fes_cfg["distance_cap"]
    rebound_days = fes_cfg["rebound_days"]
    proximity_before = fes_cfg["proximity_before"]
    proximity_after = fes_cfg["proximity_after"]
    post_festival_weekend_days = fes_cfg["post_festival_weekend_days"]
    weather_cfg = fe_cfg["weather"]
    rainy_threshold = weather_cfg["rainy_threshold_mm"]
    heavy_rain_threshold = weather_cfg["heavy_rain_threshold_mm"]
    lags_cfg = fe_cfg["lags"]
    shifts = lags_cfg["shifts"]
    rolling_windows = lags_cfg["rolling_windows"]
    same_dow_cfg = lags_cfg["same_dow_rolling"]
    sdow_window = same_dow_cfg["window"]
    sdow_min = same_dow_cfg["min_periods"]

    target_dt = pd.to_datetime(target_date)
    depots = gold_df["depot"].unique()

    # Pre-load lookups
    hol_lookup = load_holiday_calendar()
    weather_df = build_weather_for_all_depots()

    # Load depot master for lat/lon (needed for forecast API)
    depot_master = pd.read_csv(DEPOT_MASTER_CSV)
    depot_master.columns = depot_master.columns.str.lower().str.strip()
    depot_coords = depot_master.set_index("depot")[["lat", "lon"]].to_dict("index")

    # Try forecast API once per depot; cache results
    target_date_str = pd.to_datetime(target_date).strftime("%Y-%m-%d")
    forecast_cache: dict[str, dict | None] = {}
    for depot in depots:
        coords = depot_coords.get(depot)
        if coords:
            forecast_cache[depot] = fetch_weather_forecast(
                coords["lat"], coords["lon"], target_date_str,
            )
        else:
            forecast_cache[depot] = None

    rows = []
    for depot in depots:
        row: dict = {"depot": depot, "date": target_dt}
        depot_gold = (
            gold_df[gold_df["depot"] == depot]
            .sort_values("date")
            .set_index("date")
        )

        # --- Temporal features ---
        row["dow"] = target_dt.dayofweek
        for d in range(7):
            row[f"dow_{d}"] = int(target_dt.dayofweek == d)
        row["is_weekend"] = int(target_dt.dayofweek in (5, 6))
        row["month"] = target_dt.month
        row["day_of_month"] = target_dt.day
        row["week_of_year"] = target_dt.isocalendar()[1]
        row["year"] = target_dt.year
        row["day_counter"] = (target_dt - pd.Timestamp(epoch_date)).days

        # --- Festival / holiday features ---
        hol_info = hol_lookup.get(target_dt.date(), None)
        if hol_info:
            row["fes_hol_code"] = hol_info["fes_hol_code"]
            row["is_fes_hol"] = 1
        else:
            row["fes_hol_code"] = 0
            row["is_fes_hol"] = 0

        # Proximity flags: is the surrounding day a holiday?
        for n in proximity_before:
            neighbour = (target_dt + timedelta(days=n)).date()
            row[f"fes_hol_minus_{n}"] = int(neighbour in hol_lookup)
        for n in proximity_after:
            neighbour = (target_dt - timedelta(days=n)).date()
            row[f"fes_hol_plus_{n}"] = int(neighbour in hol_lookup)

        # expected_intensity: per-depot profile with global fallback
        if row["fes_hol_code"] != 0:
            _dp = depot_festival_profile.get(depot, {}) if depot_festival_profile else {}
            _gp = global_festival_profile if global_festival_profile else (festival_profile if festival_profile else {})
            row["expected_intensity"] = _dp.get(
                row["fes_hol_code"],
                _gp.get(row["fes_hol_code"], 1.0),
            )
        else:
            row["expected_intensity"] = 1.0

        # depot_festival_te: historical mean PKM for this depot on this festival
        depot_gold_df = depot_gold.reset_index()
        if row["fes_hol_code"] != 0 and "fes_hol_code" in depot_gold_df.columns:
            depot_fes_data = depot_gold_df[depot_gold_df["fes_hol_code"] == row["fes_hol_code"]]
            if len(depot_fes_data) > 0 and target_col in depot_fes_data.columns:
                row["depot_festival_te"] = depot_fes_data[target_col].mean()

        # fes_hol_category_te: historical mean PKM for this depot + category
        if hol_info and "fes_hol_category" in hol_info:
            cat = hol_info["fes_hol_category"]
            if "fes_hol_category" in depot_gold_df.columns:
                cat_data = depot_gold_df[depot_gold_df["fes_hol_category"] == cat]
                if len(cat_data) > 0 and target_col in cat_data.columns:
                    row["fes_hol_category_te"] = cat_data[target_col].mean()

        # post_festival_days: distance to most recent festival day (0-rebound_days), or -1
        post_fes_days = -1
        for offset in range(0, rebound_days + 1):
            check_date = (target_dt - timedelta(days=offset)).date()
            if check_date in hol_lookup:
                post_fes_days = offset
                break
        row["post_festival_days"] = post_fes_days

        # days_to_next_festival: scan forward up to distance_cap days
        dtf = distance_cap
        next_fes_date = None
        for fwd in range(0, distance_cap + 1):
            check_d = (target_dt + timedelta(days=fwd)).date()
            if check_d in hol_lookup:
                dtf = fwd
                next_fes_date = check_d
                break
        row["days_to_next_festival"] = dtf

        # next_festival_cluster_len: cluster length of the upcoming festival
        if next_fes_date is not None:
            # Count consecutive festival days starting from next_fes_date
            nfcl = 1
            d = 1
            while True:
                check = next_fes_date + timedelta(days=d)
                if check in hol_lookup:
                    nfcl += 1
                    d += 1
                else:
                    break
            # Also count backward from next_fes_date
            d = 1
            while True:
                check = next_fes_date - timedelta(days=d)
                if check in hol_lookup:
                    nfcl += 1
                    d += 1
                else:
                    break
            row["next_festival_cluster_len"] = nfcl
        else:
            row["next_festival_cluster_len"] = 0

        # days_since_last_festival: scan backward up to distance_cap days
        dsf = distance_cap
        for bwd in range(0, distance_cap + 1):
            check_d = (target_dt - timedelta(days=bwd)).date()
            if check_d in hol_lookup:
                dsf = bwd
                break
        row["days_since_last_festival"] = dsf

        # festival_cluster_len: length of the consecutive festival block
        if post_fes_days == 0:
            # On a festival day — count consecutive holidays forward + backward
            cluster_len = 1
            for direction in [1, -1]:
                d = 1
                while True:
                    check = (target_dt + timedelta(days=direction * d)).date()
                    if check in hol_lookup:
                        cluster_len += 1
                        d += 1
                    else:
                        break
            row["festival_cluster_len"] = cluster_len
        elif 1 <= post_fes_days <= rebound_days:
            # In rebound window — find the cluster that just ended
            last_fes = (target_dt - timedelta(days=post_fes_days)).date()
            cluster_len = 1
            d = 1
            while True:
                check = last_fes - timedelta(days=d)
                if check in hol_lookup:
                    cluster_len += 1
                    d += 1
                else:
                    break
            row["festival_cluster_len"] = cluster_len
        else:
            row["festival_cluster_len"] = 0

        # post_festival_weekend: rebound day in configured days AND weekend
        row["post_festival_weekend"] = int(
            post_fes_days in post_festival_weekend_days and row["is_weekend"] == 1
        )

        # pre_festival_weekend: weekend within 7 days before a festival, not a festival day
        row["pre_festival_weekend"] = int(
            row["is_weekend"] == 1
            and row["days_to_next_festival"] > 0
            and row["days_to_next_festival"] <= 7
            and row["is_fes_hol"] == 0
        )

        # post_festival_return_workday: first workday(s) in the post-festival rebound window
        row["post_festival_return_workday"] = int(
            row["is_weekend"] == 0
            and post_fes_days > 0
            and post_fes_days <= rebound_days
        )

        # Interaction: weekend × upcoming cluster size
        row["pre_festival_wknd_x_cluster"] = row["pre_festival_weekend"] * row["next_festival_cluster_len"]
        # Interaction: post-festival weekend × cluster size of the festival that just ended
        row["post_festival_wknd_x_cluster"] = row["post_festival_weekend"] * row["festival_cluster_len"]

        # lag_is_festival: flag if pkm_lag_{horizon} date was a festival day
        lag_check = (target_dt - timedelta(days=forecast_horizon)).date()
        row[f"lag{forecast_horizon}_is_festival"] = int(lag_check in hol_lookup)

        # --- Weather features ---
        # Prefer Open-Meteo Forecast API; fall back to historical cache
        fc = forecast_cache.get(depot)
        if fc is not None:
            row["max_temp"] = fc["max_temp"]
            row["min_temp"] = fc["min_temp"]
            row["daily_rainfall"] = fc["daily_rainfall"]
        else:
            # Fallback: last known weather from historical cache
            depot_weather = weather_df[weather_df["depot"] == depot].copy()
            if len(depot_weather) > 0:
                depot_weather["date"] = pd.to_datetime(depot_weather["date"])
                depot_weather = depot_weather.sort_values("date").set_index("date")
                if target_dt in depot_weather.index:
                    wr = depot_weather.loc[target_dt]
                else:
                    wr = depot_weather.iloc[-1]
                row["max_temp"] = wr.get("max_temp", np.nan)
                row["min_temp"] = wr.get("min_temp", np.nan)
                row["daily_rainfall"] = wr.get("daily_rainfall", np.nan)

        # Derived weather features
        rain = row.get("daily_rainfall", 0)
        row["is_rainy"] = int(rain > rainy_threshold) if not pd.isna(rain) else 0

        # --- Lag features (calendar-day based, matching training) ---
        roll_end = target_dt - timedelta(days=forecast_horizon)

        # Dynamic lag columns
        for s in shifts:
            lag_date = target_dt - timedelta(days=s)
            row[f"pkm_lag_{s}"] = (
                depot_gold.loc[lag_date, target_col]
                if lag_date in depot_gold.index else np.nan
            )

        # Dynamic rolling windows
        for i, rw in enumerate(rolling_windows):
            w, mp = rw["window"], rw["min_periods"]
            roll_start = target_dt - timedelta(days=forecast_horizon + w - 1)
            roll_slice = depot_gold.loc[roll_start:roll_end, target_col]
            if len(roll_slice) >= mp:
                row[f"pkm_roll{w}_mean"] = roll_slice.mean()
                if i == 0:
                    row[f"pkm_roll{w}_std"] = roll_slice.std()

        # Momentum: first rolling window / last rolling window
        first_w = rolling_windows[0]["window"]
        last_w = rolling_windows[-1]["window"]
        r_first = row.get(f"pkm_roll{first_w}_mean", np.nan)
        r_last = row.get(f"pkm_roll{last_w}_mean", np.nan)
        if not (pd.isna(r_first) or pd.isna(r_last)) and r_last > 0:
            row[f"pkm_momentum_{first_w}_{last_w}"] = r_first / r_last

        # Same-DOW median
        target_dow = target_dt.dayofweek
        same_dow = depot_gold[depot_gold.index.dayofweek == target_dow][target_col]
        safe_cutoff = target_dt - timedelta(days=forecast_horizon)
        same_dow = same_dow.loc[:safe_cutoff]
        if len(same_dow) >= sdow_min:
            row[f"pkm_same_dow_{sdow_window}med"] = same_dow.iloc[-sdow_window:].median()

        # --- Target encoding ---
        if len(depot_gold) > 0:
            # expanding mean up to T (shift(horizon) from T+horizon = T)
            row["depot_te"] = depot_gold[target_col].mean()

        # --- Depot × DOW target encoding ---
        target_dow = target_dt.dayofweek
        if len(depot_gold) > 0:
            depot_dow_data = depot_gold[depot_gold.index.dayofweek == target_dow][target_col]
            if len(depot_dow_data) > 0:
                row["depot_dow_te"] = depot_dow_data.mean()

        # --- Depot × Month target encoding ---
        target_month = target_dt.month
        if len(depot_gold) > 0:
            depot_month_data = depot_gold[depot_gold.index.month == target_month][target_col]
            if len(depot_month_data) > 0:
                row["depot_month_te"] = depot_month_data.mean()

        # --- Post-festival intensity ---
        if 1 <= post_fes_days <= rebound_days:
            row["post_festival_intensity"] = (
                post_festival_profile.get(depot, 1.0) if post_festival_profile else 1.0
            )
        else:
            row["post_festival_intensity"] = 1.0

        # --- Pre-festival intensity ---
        is_pre_festival = any(
            row.get(f"fes_hol_minus_{n}", 0) == 1 for n in proximity_before
        ) and row.get("is_fes_hol", 0) == 0
        if is_pre_festival:
            row["pre_festival_intensity"] = (
                pre_festival_profile.get(depot, 1.0) if pre_festival_profile else 1.0
            )
        else:
            row["pre_festival_intensity"] = 1.0

        rows.append(row)

    info_df = pd.DataFrame(rows)

    # Ensure all feature_cols exist, fill missing with train_median
    for col in feature_cols:
        if col not in info_df.columns:
            info_df[col] = np.nan
    X_future = info_df[feature_cols].fillna(train_median)

    return info_df, X_future


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_demand_prediction() -> dict:
    """
    Run the full demand prediction pipeline.

    Returns dict with prediction_date, depot_predictions, metrics, backfill_count.
    """
    # 0. Load config (used throughout the pipeline)
    config = load_model_config()
    forecast_horizon = config["forecast"]["horizon"]

    # 1. Load and clean gold data
    gold_df = load_gold_data()
    gold_df = clip_outliers_iqr(gold_df, config=config)

    # 2. Build features
    features_df = build_all_features(gold_df, config=config)

    # Extract per-depot festival profiles for construct_future_features
    festival_profile = None
    depot_festival_profile = {}
    global_festival_profile = {}
    if "fes_hol_code" in features_df.columns and "festival_intensity" in features_df.columns:
        fes_mask = features_df["is_fes_hol"] == 1
        if fes_mask.any():
            # Global fallback
            global_festival_profile = (
                features_df[fes_mask]
                .groupby("fes_hol_code")["festival_intensity"]
                .median()
                .to_dict()
            )
            festival_profile = global_festival_profile  # backward compat
            # Per-depot profiles
            for depot in features_df["depot"].unique():
                depot_data = features_df[features_df["depot"] == depot]
                depot_fes = depot_data[depot_data["is_fes_hol"] == 1]
                if len(depot_fes) > 0:
                    depot_festival_profile[depot] = (
                        depot_fes.groupby("fes_hol_code")["festival_intensity"]
                        .median()
                        .to_dict()
                    )

    # Extract post-festival and pre-festival profiles for construct_future_features
    post_festival_profile: dict[str, float] = {}
    pre_festival_profile: dict[str, float] = {}
    if "post_festival_intensity" in features_df.columns:
        for depot in features_df["depot"].unique():
            depot_data = features_df[features_df["depot"] == depot]
            pf_vals = depot_data.loc[
                depot_data["post_festival_intensity"] != 1.0, "post_festival_intensity"
            ].dropna()
            if len(pf_vals) > 0:
                post_festival_profile[depot] = float(pf_vals.median())
    if "pre_festival_intensity" in features_df.columns:
        for depot in features_df["depot"].unique():
            depot_data = features_df[features_df["depot"] == depot]
            pre_vals = depot_data.loc[
                depot_data["pre_festival_intensity"] != 1.0, "pre_festival_intensity"
            ].dropna()
            if len(pre_vals) > 0:
                pre_festival_profile[depot] = float(pre_vals.median())

    # Compute depot-specific occupancy ratios for KM conversion
    depot_or_dict = compute_depot_rolling_or(gold_df, config=config)

    # 3. Prepare model data
    model_df, feature_cols, target = prepare_model_data(config)
    test_days = config["split"]["test_days"]

    # 4. Split and train
    X_train, y_train, X_test, y_test, train_df, test_df, cutoff_date, train_median = split_train_test(
        model_df, feature_cols, target, test_days,
    )
    start_time = datetime.now()
    model = train_model(X_train, y_train, X_test, y_test, config["xgb"])
    training_time = (datetime.now() - start_time).total_seconds()

    # 5. Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    test_df["predicted"] = y_test_pred
    test_df["error"] = test_df[target] - test_df["predicted"]
    test_df["pct_error"] = (test_df["error"] / test_df[target]) * 100
    train_metrics = calculate_metrics(y_train.values, y_train_pred)
    test_metrics = calculate_metrics(y_test.values, y_test_pred)

    # 6. Save model, metrics
    save_evaluation_results(
        model, feature_cols, train_metrics, test_metrics,
        X_train, X_test, test_df, target, cutoff_date, training_time,
    )

    # 7. Generate T+horizon predictions
    #    Data goes up to T (max date in gold_df).
    #    Run date is T+1 (today). Prediction target is T+horizon.
    max_data_date = gold_df["date"].max().date()
    run_date = max_data_date + timedelta(days=1)
    prediction_date = max_data_date + timedelta(days=forecast_horizon)

    info_df, X_future = construct_future_features(
        target_date=prediction_date,
        gold_df=gold_df,
        feature_cols=feature_cols,
        train_median=train_median,
        target_col=target,
        festival_profile=festival_profile,
        depot_festival_profile=depot_festival_profile,
        global_festival_profile=global_festival_profile,
        post_festival_profile=post_festival_profile,
        pre_festival_profile=pre_festival_profile,
        config=config,
    )
    predictions_future = model.predict(X_future)
    depot_predictions = dict(zip(info_df["depot"], predictions_future))

    predictions_df = load_predictions_file()
    predictions_df = add_predictions(
        predictions_df,
        run_date.isoformat(),
        prediction_date.isoformat(),
        depot_predictions,
        depot_or_dict=depot_or_dict,
        config=config,
    )

    # 8. Backfill test-set predictions
    backfill_before = len(predictions_df)
    predictions_df = backfill_test_predictions(predictions_df, test_df, target, depot_or_dict=depot_or_dict, config=config)
    backfill_count = len(predictions_df) - backfill_before

    save_predictions_file(predictions_df)

    return {
        "prediction_date": prediction_date.isoformat(),
        "depot_predictions": {k: float(v) for k, v in depot_predictions.items()},
        "metrics": test_metrics,
        "backfill_count": backfill_count,
    }
