"""
Supply scheduling engine for TGSRTC daily operations.

Extracted from notebooks/supply_scheduling.ipynb.
Adjusts bus trips based on predicted demand using policy-based rules.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import time, datetime, date, timedelta
from collections import defaultdict
import json
import yaml
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "dynamic_schedule"

SERVICE_MASTER_PATH = DATA_DIR / "master" / "service_master.csv"
DEPOT_MASTER_PATH = DATA_DIR / "master" / "depot_master.csv"

SERVICE_OPS_GOLD_PATH = DATA_DIR / "processed" / "ops_daily_service_gold.parquet"
SERVICE_OPS_RAW_PATH = DATA_DIR / "raw" / "ops_daily_service_master.csv"

PREDICTIONS_DIR = PROJECT_ROOT / "output" / "predictions"
PREDICTIONS_FILE = PREDICTIONS_DIR / "daily_predictions.parquet"

MODEL_VERSION = "xgb_v1"
MODEL_CONFIG_PATH = PROJECT_ROOT / "model" / MODEL_VERSION / "config.yaml"


def _parse_time(s: str) -> time:
    """Convert a string like ``'06:00'`` to a ``datetime.time`` object."""
    parts = s.split(":")
    return time(int(parts[0]), int(parts[1]))


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------

POLICY = {
    "target_or": 0.75,
    "tolerance_pct": 0.03,
    "lookback_days": 7,
    "max_trip_change_per_service": 1,
    "min_trips_per_service": 0,
    "underutilized_or": 0.65,
    "overloaded_or": 0.95,
    "stop_or": 0.45,
    "morning_peak": (time(6, 0), time(10, 0)),
    "evening_peak": (time(16, 30), time(20, 30)),
    "prefer_peak_when_adding": True,
    "protect_peak_when_cutting": True,
    "max_changes_per_route": 2,
    "route_column": "route",
    "protected_services": set(),
    "max_services_changed": 25,
    "prefer_short_kmpt_when_adding": True,
}


EPK_POLICY = {
    "lookback_days": 15,
    "or_threshold_add": 0.90,
    "epk_premium_add": 1.05,
    "or_threshold_cut": 0.50,
    "epk_discount_cut": 0.90,
    "default_rev_per_pkm": 0.0,
    "default_cpk": 25.0,
    "or_quadrant_boundary": 0.70,
}


def classify_epk_or_quadrant(epk, cpk, or_val, or_boundary=0.70):
    """Classify a service into the EPK-OR 2x2 quadrant.

    Returns one of: UNDERSUPPLY, OVERSUPPLY, SOCIAL_OBLIGATION, INEFFICIENT.
    """
    profitable = epk >= cpk
    high_or = or_val >= or_boundary
    if profitable and high_or:
        return "UNDERSUPPLY"
    elif profitable and not high_or:
        return "OVERSUPPLY"
    elif not profitable and high_or:
        return "SOCIAL_OBLIGATION"
    else:
        return "INEFFICIENT"


def load_scheduling_policy(config_path: Path = MODEL_CONFIG_PATH) -> dict:
    """Load scheduling policy from config.yaml, merging over hardcoded defaults.

    Time strings (e.g. ``"06:00"``) are converted to ``time`` objects for the
    ``morning_peak`` and ``evening_peak`` tuple values.
    """
    defaults = POLICY.copy()
    if config_path.exists():
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f) or {}
        sp = file_config.get("scheduling_policy", {})
        for key, value in sp.items():
            if key in ("morning_peak_start", "morning_peak_end",
                       "evening_peak_start", "evening_peak_end"):
                continue  # handled below
            defaults[key] = value
        # Convert time strings to tuples
        defaults["morning_peak"] = (
            _parse_time(sp.get("morning_peak_start", "06:00")),
            _parse_time(sp.get("morning_peak_end", "10:00")),
        )
        defaults["evening_peak"] = (
            _parse_time(sp.get("evening_peak_start", "16:30")),
            _parse_time(sp.get("evening_peak_end", "20:30")),
        )
    defaults["protected_services"] = set()
    return defaults


def load_epk_policy(config_path: Path = MODEL_CONFIG_PATH) -> dict:
    """Load EPK policy thresholds from config.yaml, merging over defaults."""
    defaults = EPK_POLICY.copy()
    if config_path.exists():
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f) or {}
        ep = file_config.get("epk_policy", {})
        for key, value in ep.items():
            defaults[key] = value
    return defaults


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------


def clean_service_master(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "depot" not in df.columns:
        df["depot"] = "DEFAULT"
    df["depot"] = df["depot"].astype(str).str.strip()
    df["service_number"] = df["service_number"].astype(str).str.strip()
    df["planned_trips"] = pd.to_numeric(df["planned_trips"], errors="coerce").fillna(0).astype(int)
    df["planned_kms"] = pd.to_numeric(df["planned_kms"], errors="coerce").fillna(0)
    df["avg_seats_per_bus"] = pd.to_numeric(df["avg_seats_per_bus"], errors="coerce").fillna(45)
    if "km_per_trip" not in df.columns or df["km_per_trip"].isna().all():
        df["km_per_trip"] = df["planned_kms"] / df["planned_trips"].replace(0, np.nan)
    df["km_per_trip"] = pd.to_numeric(df["km_per_trip"], errors="coerce").fillna(0)
    if "can_be_cancelled" in df.columns:
        df["can_be_cancelled"] = pd.to_numeric(df["can_be_cancelled"], errors="coerce").fillna(0).astype(int)
    else:
        df["can_be_cancelled"] = 1
    return df


def clean_daily_ops(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "depot" not in df.columns:
        df["depot"] = "DEFAULT"
    df["depot"] = df["depot"].astype(str).str.strip()
    df["service_number"] = df["service_number"].astype(str).str.strip()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    num_cols = ["actual_kms", "actual_trips", "seat_kms", "passenger_kms", "occupancy_ratio", "revenue"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "actual_trips" in df.columns:
        df["actual_trips"] = df["actual_trips"].fillna(0)
    return df


def load_depot_target_or(
    depot_master_path: Path = DEPOT_MASTER_PATH,
    fallback_or: float = 0.75,
) -> dict[str, float]:
    """Load per-depot target OR from depot_master.csv.

    Returns {depot: target_or} dict.  Falls back to *fallback_or* when
    the file is missing or a depot has no value.
    """
    if not depot_master_path.exists():
        return {}
    dm = pd.read_csv(depot_master_path, encoding="utf-8-sig")
    dm.columns = dm.columns.str.lower().str.strip()
    if "target_or" not in dm.columns:
        return {}
    result: dict[str, float] = {}
    for _, row in dm.iterrows():
        depot = str(row["depot"]).strip()
        raw = row["target_or"]
        if pd.isna(raw):
            result[depot] = fallback_or
            continue
        raw_str = str(raw).strip().rstrip("%")
        try:
            val = float(raw_str)
            # Values like "80%" become 80.0 — convert to fraction
            if val > 1:
                val = val / 100.0
            result[depot] = val
        except ValueError:
            result[depot] = fallback_or
    return result


# ---------------------------------------------------------------------------
# Predictions loader
# ---------------------------------------------------------------------------


def load_predictions_for_scheduling(
    predictions_file: Path = PREDICTIONS_FILE,
    target_date: date | None = None,
    policy: dict | None = None,
) -> tuple[dict, date]:
    """Load depot predictions for scheduling.

    Returns (depot_predictions, resolved_target_date).
    When *target_date* is ``None`` the latest prediction date in the
    parquet is used instead of a hard-coded "tomorrow" default.
    """
    if policy is None:
        policy = POLICY
    if not predictions_file.exists():
        fallback = target_date or (date.today() + timedelta(days=1))
        return {}, fallback
    df = pd.read_parquet(predictions_file)
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    if target_date is None:
        target_date = df["prediction_date"].max().date()
    date_mask = df["prediction_date"].dt.date == target_date
    target_predictions = df[date_mask]
    if len(target_predictions) == 0:
        return {}, target_date
    depot_predictions = {}
    bus_capacity = policy.get("bus_capacity", 45)
    target_or = policy["target_or"]
    avg_km_per_passenger = policy.get("avg_km_per_passenger", 25)
    for _, row in target_predictions.iterrows():
        depot = row["depot"]
        if "predicted_passenger_kms" in row and pd.notna(row.get("predicted_passenger_kms")):
            depot_predictions[depot] = row["predicted_passenger_kms"]
        elif "estimated_kms" in row and pd.notna(row.get("estimated_kms")):
            depot_predictions[depot] = row["estimated_kms"] * bus_capacity * target_or
        else:
            predicted_passengers = row.get("predicted_passengers", 0)
            depot_predictions[depot] = predicted_passengers * avg_km_per_passenger
    return depot_predictions, target_date


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def parse_time_safe(x) -> time | None:
    if pd.isna(x):
        return None
    if isinstance(x, time):
        return x
    try:
        return pd.to_datetime(str(x)).time()
    except Exception:
        return None


def is_peak_hour(dep_time_val, policy: dict) -> bool:
    t = parse_time_safe(dep_time_val)
    if t is None:
        return False
    m1, m2 = policy["morning_peak"]
    e1, e2 = policy["evening_peak"]
    return (m1 <= t <= m2) or (e1 <= t <= e2)


def compute_recent_or(
    daily_ops: pd.DataFrame,
    depot: str,
    target_date: date,
    lookback_days: int,
) -> pd.Series:
    depot_ops = daily_ops[daily_ops["depot"] == depot].copy()
    if "date" in depot_ops.columns and pd.api.types.is_datetime64_any_dtype(depot_ops["date"]):
        end_date = pd.Timestamp(target_date)
        start_date = end_date - pd.Timedelta(days=lookback_days)
        depot_ops = depot_ops[(depot_ops["date"] >= start_date) & (depot_ops["date"] < end_date)]
    if "occupancy_ratio" in depot_ops.columns:
        return depot_ops.groupby("service_number")["occupancy_ratio"].mean()
    return pd.Series(dtype=float)


def compute_depot_planned_kms(service_master: pd.DataFrame, depot: str) -> float:
    depot_services = service_master[service_master["depot"] == depot]
    return float((depot_services["planned_trips"] * depot_services["km_per_trip"]).sum())


def compute_target_kms(predicted_pkm: float, avg_seats: float, target_or: float) -> float:
    if not np.isfinite(avg_seats) or avg_seats <= 0:
        avg_seats = 45
    target_seat_km = predicted_pkm / max(target_or, 1e-6)
    return float(target_seat_km / avg_seats)


# ---------------------------------------------------------------------------
# Policy engine — core single-depot logic
# ---------------------------------------------------------------------------


def run_policy_engine(
    service_master: pd.DataFrame,
    daily_ops: pd.DataFrame,
    depot: str,
    target_date: date,
    predicted_depot_pkm: float,
    policy: dict,
    depot_target_or: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    base = service_master[service_master["depot"] == depot].copy()
    if len(base) == 0:
        return pd.DataFrame(), {"error": f"No services found for depot {depot}"}

    # Use depot-specific target_or if provided, otherwise fall back to policy default
    effective_target_or = depot_target_or if depot_target_or is not None else policy["target_or"]

    recent_or = compute_recent_or(daily_ops, depot, target_date, policy["lookback_days"])
    or_col = f"avg_or_last_{policy['lookback_days']}d"
    base = base.merge(recent_or.rename(or_col), on="service_number", how="left")
    base[or_col] = base[or_col].fillna(0)

    base["is_peak"] = base["dep_time"].apply(lambda x: is_peak_hour(x, policy))
    base["is_protected"] = base["service_number"].isin(policy["protected_services"])

    route_col = policy["route_column"]
    if route_col not in base.columns:
        base[route_col] = "UNKNOWN"
    base[route_col] = base[route_col].astype(str).fillna("UNKNOWN")

    planned_kms = compute_depot_planned_kms(service_master, depot)
    avg_seats = base["avg_seats_per_bus"].replace(0, np.nan).mean()
    target_kms = compute_target_kms(predicted_depot_pkm, avg_seats, effective_target_or)
    delta_kms = target_kms - planned_kms

    tol_kms = policy["tolerance_pct"] * max(planned_kms, 1.0)
    should_act = abs(delta_kms) > tol_kms

    base["suggested_trips"] = base["planned_trips"].copy()
    base["action"] = "NO_CHANGE"
    base["reason"] = "Within tolerance"

    total_changed = 0
    route_changes: dict[str, int] = defaultdict(int)

    def can_change_route(route_val: str) -> bool:
        return route_changes[route_val] < policy["max_changes_per_route"]

    def register_change(route_val: str):
        nonlocal total_changed
        route_changes[route_val] += 1
        total_changed += 1

    def reached_cap() -> bool:
        return (
            policy["max_services_changed"] is not None
            and total_changed >= policy["max_services_changed"]
        )

    if should_act:
        if delta_kms > 0:
            remaining = delta_kms
            add_df = base[base[or_col] >= policy["underutilized_or"]].copy()
            sort_cols, sort_asc = [], []
            if policy["prefer_peak_when_adding"]:
                sort_cols.append("is_peak")
                sort_asc.append(False)
            sort_cols.append(or_col)
            sort_asc.append(False)
            if policy["prefer_short_kmpt_when_adding"]:
                sort_cols.append("km_per_trip")
                sort_asc.append(True)
            add_df = add_df.sort_values(sort_cols, ascending=sort_asc)
            for idx in add_df.index:
                if remaining <= 0 or reached_cap():
                    break
                route_val = base.loc[idx, route_col]
                if not can_change_route(route_val):
                    continue
                add_n = policy["max_trip_change_per_service"]
                km_gain = float(base.loc[idx, "km_per_trip"]) * add_n
                if km_gain <= 0:
                    continue
                base.loc[idx, "suggested_trips"] += add_n
                base.loc[idx, "action"] = "INCREASE"
                base.loc[idx, "reason"] = f"Add {add_n} trip(s), OR={float(base.loc[idx, or_col]):.2f}"
                remaining -= km_gain
                register_change(route_val)
        else:
            need_remove = -delta_kms
            cut_df = base[~base["is_protected"] & (base["can_be_cancelled"] == 1)].copy()
            if policy["protect_peak_when_cutting"]:
                cut_first = cut_df[~cut_df["is_peak"]].copy()
                cut_later = cut_df[cut_df["is_peak"]].copy()
            else:
                cut_first = cut_df
                cut_later = pd.DataFrame()
            cut_first = cut_first.sort_values([or_col, "km_per_trip"], ascending=[True, False])
            cut_later = cut_later.sort_values([or_col, "km_per_trip"], ascending=[True, False])

            def try_cut(df_part: pd.DataFrame):
                nonlocal need_remove
                for idx in df_part.index:
                    if need_remove <= 0 or reached_cap():
                        break
                    route_val = base.loc[idx, route_col]
                    if not can_change_route(route_val):
                        continue
                    or_i = float(base.loc[idx, or_col])
                    current = int(base.loc[idx, "suggested_trips"])
                    if current <= policy["min_trips_per_service"]:
                        continue
                    if or_i < policy["stop_or"] and policy["min_trips_per_service"] == 0:
                        remove_n = current
                        base.loc[idx, "suggested_trips"] = 0
                        base.loc[idx, "action"] = "STOP"
                        base.loc[idx, "reason"] = f"Stop service, very low OR={or_i:.2f}"
                    else:
                        remove_n = min(
                            policy["max_trip_change_per_service"],
                            current - policy["min_trips_per_service"],
                        )
                        if remove_n <= 0:
                            continue
                        base.loc[idx, "suggested_trips"] = current - remove_n
                        base.loc[idx, "action"] = "DECREASE"
                        base.loc[idx, "reason"] = f"Cut {remove_n} trip(s), OR={or_i:.2f}"
                    km_saved = float(base.loc[idx, "km_per_trip"]) * remove_n
                    need_remove -= km_saved
                    register_change(route_val)

            try_cut(cut_first)
            if need_remove > 0:
                try_cut(cut_later)

    # Compute impacts
    base["planned_kms_day"] = (base["planned_trips"] * base["km_per_trip"]).round(2)
    base["suggested_kms_day"] = (base["suggested_trips"] * base["km_per_trip"]).round(2)
    base["trip_change"] = (base["suggested_trips"] - base["planned_trips"]).astype(int)
    base["kms_change"] = (base["suggested_kms_day"] - base["planned_kms_day"]).round(2)

    out_cols = [
        "service_number", route_col, "product", "dep_time",
        "is_peak", "is_protected",
        "planned_trips", "suggested_trips", "trip_change",
        "km_per_trip", "planned_kms_day", "suggested_kms_day", "kms_change",
        or_col, "action", "reason",
    ]
    out_cols = [c for c in out_cols if c in base.columns]
    out = base[out_cols].copy()

    action_order = pd.Categorical(
        out["action"],
        categories=["STOP", "DECREASE", "INCREASE", "NO_CHANGE"],
        ordered=True,
    )
    out["_action_rank"] = action_order
    out = out.sort_values(["_action_rank", "kms_change"], ascending=[True, False])
    out = out.drop(columns="_action_rank").reset_index(drop=True)

    summary = {
        "depot": depot,
        "target_date": str(target_date),
        "predicted_depot_pkm": float(predicted_depot_pkm),
        "policy_target_or": float(effective_target_or),
        "depot_planned_kms": float(planned_kms),
        "depot_target_kms": float(target_kms),
        "delta_kms": float(delta_kms),
        "tolerance_kms": float(tol_kms),
        "action_taken": bool(should_act),
        "total_kms_change": float(out["kms_change"].sum()),
        "count_increase": int((out["action"] == "INCREASE").sum()),
        "count_decrease": int((out["action"] == "DECREASE").sum()),
        "count_stop": int((out["action"] == "STOP").sum()),
        "count_no_change": int((out["action"] == "NO_CHANGE").sum()),
        "total_services_changed": total_changed,
    }

    return out, summary


# ---------------------------------------------------------------------------
# EPK / OR-based scheduling engine
# ---------------------------------------------------------------------------


def compute_service_weights(
    daily_ops: pd.DataFrame,
    depot: str,
    target_date: date,
    lookback_days: int,
) -> pd.Series:
    """Weight = service_pkm_15d / depot_total_pkm_15d over the lookback window."""
    depot_ops = daily_ops[daily_ops["depot"] == depot].copy()
    if "date" in depot_ops.columns and pd.api.types.is_datetime64_any_dtype(depot_ops["date"]):
        end_date = pd.Timestamp(target_date)
        start_date = end_date - pd.Timedelta(days=lookback_days)
        depot_ops = depot_ops[(depot_ops["date"] >= start_date) & (depot_ops["date"] < end_date)]
    if "passenger_kms" not in depot_ops.columns or len(depot_ops) == 0:
        return pd.Series(dtype=float)
    svc_pkm = depot_ops.groupby("service_number")["passenger_kms"].sum()
    total_pkm = svc_pkm.sum()
    if total_pkm == 0:
        return svc_pkm * 0.0
    return svc_pkm / total_pkm


def compute_rev_per_pkm(
    daily_ops: pd.DataFrame,
    depot: str,
    target_date: date,
    lookback_days: int,
) -> pd.Series:
    """Mean(daily_revenue / daily_passenger_kms) per service over lookback window."""
    depot_ops = daily_ops[daily_ops["depot"] == depot].copy()
    if "date" in depot_ops.columns and pd.api.types.is_datetime64_any_dtype(depot_ops["date"]):
        end_date = pd.Timestamp(target_date)
        start_date = end_date - pd.Timedelta(days=lookback_days)
        depot_ops = depot_ops[(depot_ops["date"] >= start_date) & (depot_ops["date"] < end_date)]
    if "revenue" not in depot_ops.columns or len(depot_ops) == 0:
        return pd.Series(dtype=float)
    depot_ops = depot_ops[depot_ops["passenger_kms"] > 0].copy()
    if len(depot_ops) == 0:
        return pd.Series(dtype=float)
    depot_ops["_rev_per_pkm"] = depot_ops["revenue"] / depot_ops["passenger_kms"]
    return depot_ops.groupby("service_number")["_rev_per_pkm"].mean()


def find_slot_midpoint(
    service_master: pd.DataFrame,
    depot: str,
    route: str,
    triggering_dep_time,
) -> time | None:
    """Find next departure on same route and return midpoint time.

    Returns None when there is no next departure (last or single service).
    """
    route_services = service_master[
        (service_master["depot"] == depot)
        & (service_master["route"].astype(str) == str(route))
    ].copy()
    if len(route_services) <= 1:
        return None
    route_services["_parsed_time"] = route_services["dep_time"].apply(parse_time_safe)
    route_services = route_services.dropna(subset=["_parsed_time"])
    route_services = route_services.sort_values("_parsed_time").reset_index(drop=True)
    trigger_t = parse_time_safe(triggering_dep_time)
    if trigger_t is None:
        return None
    # Find the triggering service's position
    times = route_services["_parsed_time"].tolist()
    trigger_idx = None
    for i, t in enumerate(times):
        if t == trigger_t:
            trigger_idx = i
            break
    if trigger_idx is None:
        return None
    if trigger_idx >= len(times) - 1:
        return None  # last departure
    next_t = times[trigger_idx + 1]
    # Compute midpoint
    trigger_minutes = trigger_t.hour * 60 + trigger_t.minute
    next_minutes = next_t.hour * 60 + next_t.minute
    mid_minutes = (trigger_minutes + next_minutes) // 2
    return time(mid_minutes // 60, mid_minutes % 60)


def run_epk_policy_engine(
    service_master: pd.DataFrame,
    daily_ops: pd.DataFrame,
    depot: str,
    target_date: date,
    predicted_depot_pkm: float,
    epk_policy: dict,
) -> tuple[pd.DataFrame, dict]:
    """Core EPK/OR decision engine for a single depot.

    Steps:
      1. Filter service_master to depot
      2. Compute 15-day passenger_kms weights per service
      3. Allocate depot forecast to services
      4. Compute revenue/pkm from ops history
      5. Derive EPK and OR per service
      6. Decide ADD_SLOT / CUT / NO_CHANGE
    """
    base = service_master[service_master["depot"] == depot].copy()
    if len(base) == 0:
        return pd.DataFrame(), {"error": f"No services found for depot {depot}"}

    lookback = epk_policy["lookback_days"]

    # Step 2: weights
    weights = compute_service_weights(daily_ops, depot, target_date, lookback)
    base = base.merge(
        weights.rename("weight"), on="service_number", how="left",
    )
    base["weight"] = base["weight"].fillna(0.0)

    # Step 3: allocated pkm
    base["allocated_pkm"] = predicted_depot_pkm * base["weight"]

    # Step 4: rev_per_pkm
    rev_per_pkm = compute_rev_per_pkm(daily_ops, depot, target_date, lookback)
    if len(rev_per_pkm) > 0:
        base = base.merge(
            rev_per_pkm.rename("rev_per_pkm"), on="service_number", how="left",
        )
    else:
        base["rev_per_pkm"] = np.nan
    base["rev_per_pkm"] = base["rev_per_pkm"].fillna(epk_policy["default_rev_per_pkm"])

    # Step 5: revenue
    base["revenue"] = base["allocated_pkm"] * base["rev_per_pkm"]

    # Step 6: EPK = revenue / planned_kms (guarded)
    base["epk"] = np.where(
        base["planned_kms"] > 0,
        base["revenue"] / base["planned_kms"],
        0.0,
    )

    # Step 7: CPK (breakeven cost per km)
    if "breakeven_cpk" in base.columns:
        base["cpk"] = pd.to_numeric(base["breakeven_cpk"], errors="coerce").fillna(
            epk_policy["default_cpk"]
        )
    else:
        base["cpk"] = epk_policy["default_cpk"]

    # Step 8: OR = allocated_pkm / (planned_kms * avg_seats_per_bus)
    denom = base["planned_kms"] * base["avg_seats_per_bus"]
    base["or"] = np.where(denom > 0, base["allocated_pkm"] / denom, 0.0)

    # Step 8b: Quadrant classification and contribution
    or_boundary = epk_policy.get("or_quadrant_boundary", 0.70)
    base["quadrant"] = base.apply(
        lambda r: classify_epk_or_quadrant(r["epk"], r["cpk"], r["or"], or_boundary),
        axis=1,
    )
    base["contribution"] = base["revenue"] - (base["cpk"] * base["planned_kms"])

    # Step 9: Decisions
    add_mask = (
        (base["or"] > epk_policy["or_threshold_add"])
        & (base["epk"] > epk_policy["epk_premium_add"] * base["cpk"])
    )
    cut_mask = (
        (base["or"] < epk_policy["or_threshold_cut"])
        & (base["epk"] < epk_policy["epk_discount_cut"] * base["cpk"])
        & (base["can_be_cancelled"] == 1)
    )

    base["action"] = "NO_CHANGE"
    base.loc[add_mask, "action"] = "ADD_SLOT"
    base.loc[cut_mask, "action"] = "CUT"

    base["reason"] = "Within policy bounds"
    base.loc[add_mask, "reason"] = base.loc[add_mask].apply(
        lambda r: f"OR={r['or']:.2f}>{epk_policy['or_threshold_add']}, EPK={r['epk']:.2f}>{epk_policy['epk_premium_add']}*CPK",
        axis=1,
    )
    base.loc[cut_mask, "reason"] = base.loc[cut_mask].apply(
        lambda r: f"OR={r['or']:.2f}<{epk_policy['or_threshold_cut']}, EPK={r['epk']:.2f}<{epk_policy['epk_discount_cut']}*CPK",
        axis=1,
    )

    # Suggested new slot for ADD_SLOT
    base["suggested_new_slot"] = None
    for idx in base[add_mask].index:
        route = str(base.loc[idx, "route"]) if "route" in base.columns else "UNKNOWN"
        dep_t = base.loc[idx, "dep_time"]
        mid = find_slot_midpoint(service_master, depot, route, dep_t)
        base.loc[idx, "suggested_new_slot"] = str(mid) if mid is not None else None

    # Output columns
    route_col = "route" if "route" in base.columns else None
    out_cols = ["service_number"]
    if route_col:
        out_cols.append(route_col)
    out_cols += [
        "product", "dep_time", "allocated_pkm", "planned_kms", "revenue",
        "epk", "or", "cpk", "quadrant", "contribution",
        "action", "suggested_new_slot", "reason",
    ]
    out_cols = [c for c in out_cols if c in base.columns]
    out = base[out_cols].copy()

    action_order = pd.Categorical(
        out["action"],
        categories=["ADD_SLOT", "CUT", "NO_CHANGE"],
        ordered=True,
    )
    out["_action_rank"] = action_order
    out = out.sort_values("_action_rank").drop(columns="_action_rank").reset_index(drop=True)

    # Quadrant breakdown
    quadrant_counts = base["quadrant"].value_counts().to_dict()
    total_svc = len(base)
    quadrant_pcts = {q: round(c / total_svc * 100, 1) if total_svc > 0 else 0.0
                     for q, c in quadrant_counts.items()}

    # Financial aggregates
    total_revenue = float(base["revenue"].sum())
    total_contribution = float(base["contribution"].sum())
    total_planned_kms = float(base["planned_kms"].sum())
    depot_avg_epk = float(base["epk"].mean()) if total_svc > 0 else 0.0
    depot_avg_or = float(base["or"].mean()) if total_svc > 0 else 0.0

    # Revenue and contribution by quadrant
    revenue_by_quadrant = base.groupby("quadrant")["revenue"].sum().to_dict()
    contribution_by_quadrant = base.groupby("quadrant")["contribution"].sum().to_dict()

    summary = {
        "depot": depot,
        "target_date": str(target_date),
        "predicted_depot_pkm": float(predicted_depot_pkm),
        "total_services": total_svc,
        "count_add_slot": int(add_mask.sum()),
        "count_cut": int(cut_mask.sum()),
        "count_no_change": int((~add_mask & ~cut_mask).sum()),
        "total_allocated_pkm": float(base["allocated_pkm"].sum()),
        "quadrant_counts": quadrant_counts,
        "quadrant_pcts": quadrant_pcts,
        "total_revenue": total_revenue,
        "total_contribution": total_contribution,
        "total_planned_kms": total_planned_kms,
        "depot_avg_epk": depot_avg_epk,
        "depot_avg_or": depot_avg_or,
        "revenue_by_quadrant": revenue_by_quadrant,
        "contribution_by_quadrant": contribution_by_quadrant,
    }

    return out, summary


def run_all_depots_epk(
    service_master: pd.DataFrame,
    daily_ops: pd.DataFrame,
    target_date: date,
    depot_predictions: dict,
    epk_policy: dict,
    output_dir: Path,
) -> tuple[dict, pd.DataFrame]:
    """Multi-depot orchestrator for EPK engine (mirrors run_all_depots)."""
    date_str = target_date.strftime("%Y-%m-%d")
    date_output_dir = output_dir / date_str
    date_output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    all_schedules = []

    for depot, predicted_pkm in depot_predictions.items():
        schedule_df, summary = run_epk_policy_engine(
            service_master=service_master,
            daily_ops=daily_ops,
            depot=depot,
            target_date=target_date,
            predicted_depot_pkm=predicted_pkm,
            epk_policy=epk_policy,
        )
        if "error" in summary:
            continue
        schedule_df["depot"] = depot
        all_schedules.append(schedule_df)
        all_summaries[depot] = summary

        depot_safe = depot.replace(" ", "_").replace("/", "_")
        schedule_path = date_output_dir / f"epk_schedule_{depot_safe}_{date_str}.xlsx"
        schedule_df.to_excel(schedule_path, index=False)

        summary_path = date_output_dir / f"epk_summary_{depot_safe}_{date_str}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    if all_schedules:
        consolidated_df = pd.concat(all_schedules, ignore_index=True)
        consolidated_path = date_output_dir / f"epk_consolidated_schedule_{date_str}.xlsx"
        consolidated_df.to_excel(consolidated_path, index=False)
        consolidated_summary_path = date_output_dir / f"epk_consolidated_summary_{date_str}.json"
        with open(consolidated_summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2, default=str)
    else:
        consolidated_df = pd.DataFrame()

    return all_summaries, consolidated_df


# ---------------------------------------------------------------------------
# Multi-depot runner
# ---------------------------------------------------------------------------


def run_all_depots(
    service_master: pd.DataFrame,
    daily_ops: pd.DataFrame,
    target_date: date,
    depot_predictions: dict,
    policy: dict,
    output_dir: Path,
    depot_target_or_map: dict[str, float] | None = None,
) -> tuple[dict, pd.DataFrame]:
    date_str = target_date.strftime("%Y-%m-%d")
    date_output_dir = output_dir / date_str
    date_output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    all_schedules = []

    for depot, predicted_pkm in depot_predictions.items():
        depot_or = depot_target_or_map.get(depot) if depot_target_or_map else None
        schedule_df, summary = run_policy_engine(
            service_master=service_master,
            daily_ops=daily_ops,
            depot=depot,
            target_date=target_date,
            predicted_depot_pkm=predicted_pkm,
            policy=policy,
            depot_target_or=depot_or,
        )
        if "error" in summary:
            continue
        schedule_df["depot"] = depot
        all_schedules.append(schedule_df)
        all_summaries[depot] = summary

        depot_safe = depot.replace(" ", "_").replace("/", "_")
        schedule_path = date_output_dir / f"schedule_{depot_safe}_{date_str}.xlsx"
        schedule_df.to_excel(schedule_path, index=False)

        summary_path = date_output_dir / f"summary_{depot_safe}_{date_str}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    if all_schedules:
        consolidated_df = pd.concat(all_schedules, ignore_index=True)
        consolidated_path = date_output_dir / f"consolidated_schedule_{date_str}.xlsx"
        consolidated_df.to_excel(consolidated_path, index=False)
        consolidated_summary_path = date_output_dir / f"consolidated_summary_{date_str}.json"
        with open(consolidated_summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2, default=str)
    else:
        consolidated_df = pd.DataFrame()

    return all_summaries, consolidated_df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_supply_scheduling(
    target_date: date | None = None,
    engine: str = "epk",
) -> dict:
    """
    Run supply scheduling for all depots.

    Parameters
    ----------
    target_date : date or None
        Scheduling date.  Defaults to latest prediction date.
    engine : str
        ``"epk"`` (default) for the EPK/OR-based decision engine,
        ``"delta_kms"`` for the original trip-change engine.

    Returns dict with target_date, summaries (per-depot), output_dir.
    """
    # Load policy from config (fresh each run)
    policy = load_scheduling_policy()

    # Load service master
    service_master_raw = pd.read_csv(SERVICE_MASTER_PATH)
    service_master = clean_service_master(service_master_raw)

    # Load daily ops
    if SERVICE_OPS_GOLD_PATH.exists():
        daily_ops_raw = pd.read_parquet(SERVICE_OPS_GOLD_PATH)
    elif SERVICE_OPS_RAW_PATH.exists():
        daily_ops_raw = pd.read_csv(SERVICE_OPS_RAW_PATH)
    else:
        raise FileNotFoundError("No service operations data found!")
    daily_ops = clean_daily_ops(daily_ops_raw)

    # Load predictions (resolves target_date to latest prediction if None)
    depot_predictions, target_date = load_predictions_for_scheduling(
        predictions_file=PREDICTIONS_FILE,
        target_date=target_date,
        policy=policy,
    )

    if not depot_predictions:
        # Fallback to historical averages
        all_depots = service_master["depot"].unique().tolist()
        if "passenger_kms" in daily_ops.columns:
            historical_pkm = (
                daily_ops
                .groupby(["depot", "date"])["passenger_kms"]
                .sum()
                .groupby("depot")
                .mean()
            )
            depot_predictions = historical_pkm.to_dict()
        else:
            fallback_pkm = policy.get("fallback_pkm", 2_000_000)
            depot_predictions = {depot: fallback_pkm for depot in all_depots}

    if engine == "epk":
        epk_policy = load_epk_policy()
        all_summaries, consolidated_schedule = run_all_depots_epk(
            service_master=service_master,
            daily_ops=daily_ops,
            target_date=target_date,
            depot_predictions=depot_predictions,
            epk_policy=epk_policy,
            output_dir=OUTPUT_DIR,
        )
    else:
        # Load per-depot target OR from depot_master.csv
        depot_target_or_map = load_depot_target_or(
            fallback_or=policy["target_or"],
        )

        # Run scheduling
        all_summaries, consolidated_schedule = run_all_depots(
            service_master=service_master,
            daily_ops=daily_ops,
            target_date=target_date,
            depot_predictions=depot_predictions,
            policy=policy,
            output_dir=OUTPUT_DIR,
            depot_target_or_map=depot_target_or_map,
        )

    return {
        "target_date": str(target_date),
        "engine": engine,
        "summaries": all_summaries,
        "output_dir": str(OUTPUT_DIR / target_date.strftime("%Y-%m-%d")),
        "depots_processed": len(all_summaries),
    }
