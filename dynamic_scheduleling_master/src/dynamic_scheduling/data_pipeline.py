"""
Data pipeline for TGSRTC daily operations data.

Extracted from notebooks/data_pipeline.ipynb.
Handles inbound file ingestion, RAW master upsert, GOLD layer building,
and predictions update.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import hashlib
import re
from datetime import datetime, date, timedelta
import json
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INBOUND_DIR = DATA_DIR / "inbound_daily"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARCHIVE_DIR = DATA_DIR / "archive" / "inbound_daily"
LOGS_DIR = DATA_DIR / "logs"
PREDICTIONS_DIR = PROJECT_ROOT / "output" / "predictions"

RAW_MASTER_CSV = RAW_DIR / "ops_daily_master.csv"
GOLD_MASTER_PARQ = PROCESSED_DIR / "ops_daily_gold.parquet"

RAW_SERVICE_MASTER_CSV = RAW_DIR / "ops_daily_service_master.csv"
GOLD_SERVICE_PARQ = PROCESSED_DIR / "ops_daily_service_gold.parquet"

INGEST_LOG_CSV = LOGS_DIR / "ingest_log.csv"
ERROR_LOG_CSV = LOGS_DIR / "ingest_errors.csv"

PREDICTIONS_FILE = PREDICTIONS_DIR / "daily_predictions.parquet"

TELUGU_CAL_PATH = DATA_DIR / "master" / "telugu_calendar.csv"
HOLIDAY_CAL_PATH = DATA_DIR / "master" / "holiday_calendar.csv"
DEPOT_MASTER_CSV = DATA_DIR / "master" / "depot_master.csv"

# Ensure directories exist
for _d in [INBOUND_DIR, RAW_DIR, PROCESSED_DIR, ARCHIVE_DIR, LOGS_DIR, PREDICTIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Schema configuration
# ---------------------------------------------------------------------------


def _load_allowed_depots() -> list[str]:
    if DEPOT_MASTER_CSV.exists():
        df = pd.read_csv(DEPOT_MASTER_CSV)
        df.columns = df.columns.str.lower().str.strip()
        return sorted(df["depot"].astype(str).str.strip().tolist())
    raise FileNotFoundError(f"depot_master.csv not found at {DEPOT_MASTER_CSV}")


ALLOWED_DEPOTS = _load_allowed_depots()

INBOUND_COLUMNS = ["depot", "date", "passengers_per_day", "actual_kms", "occupancy_ratio"]
RAW_COLUMNS = INBOUND_COLUMNS.copy()

SERVICE_INBOUND_COLUMNS = [
    "depot", "date", "service_number",
    "actual_kms", "actual_trips", "seat_kms", "passenger_kms", "occupancy_ratio",
]
SERVICE_RAW_COLUMNS = SERVICE_INBOUND_COLUMNS + ["revenue"]

TELUGU_CAL_COLUMNS = [
    "date", "telugu_thithi", "telugu_paksham", "marriage_day", "telugu_month", "moudyami_day",
]
HOLIDAY_WIDE_COLUMNS = [
    "fes_hol_code", "Holiday_Festival", "fes_hol_category",
    "2023_dates", "2024_dates", "2025_dates", "2026_dates",
]

GOLD_COLUMNS = [
    "depot", "date", "passengers_per_day", "actual_kms", "occupancy_ratio", "passenger_kms",
    "telugu_thithi", "telugu_paksham", "marriage_day", "telugu_month", "moudyami_day",
    "fes_hol_code", "Holiday_Festival", "fes_hol_category", "is_fes_hol",
]

SERVICE_GOLD_COLUMNS = [
    "depot", "date", "service_number",
    "actual_kms", "actual_trips", "seat_kms", "passenger_kms", "occupancy_ratio",
    "revenue",
    "telugu_thithi", "telugu_paksham", "marriage_day", "telugu_month", "moudyami_day",
    "fes_hol_code", "Holiday_Festival", "fes_hol_category", "is_fes_hol",
]

BOUNDS = {
    "passengers_per_day": (0, None),
    "actual_kms": (0, None),
    "occupancy_ratio": (0, 2.0),
    "passenger_kms": (0, None),
}

SERVICE_BOUNDS = {
    "actual_kms": (0, None),
    "actual_trips": (0, None),
    "seat_kms": (0, None),
    "passenger_kms": (0, None),
    "occupancy_ratio": (0, 3.0),
    "revenue": (0, None),
}

DEPOT_FILE_PATTERN = re.compile(r"ops_daily_(\d{4}-\d{2}-\d{2})\.csv$")
SERVICE_FILE_PATTERN = re.compile(r"ops_daily_service_(\d{4}-\d{2}-\d{2})\.csv$")

PREDICTIONS_COLUMNS = [
    "run_date", "prediction_date", "depot",
    "predicted_passenger_kms", "actual_passenger_kms",
    "assumed_or", "actual_or", "estimated_kms", "actual_kms",
    "bus_capacity", "estimated_buses", "actual_buses",
    "pkm_error", "pkm_error_pct", "km_error", "km_error_pct", "status",
]

# ---------------------------------------------------------------------------
# Helper functions — File I/O
# ---------------------------------------------------------------------------


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------


def extract_date_from_filename(filename: str, file_type: str = "depot") -> Optional[date]:
    pattern = DEPOT_FILE_PATTERN if file_type == "depot" else SERVICE_FILE_PATTERN
    match = pattern.search(filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def validate_filename_format(filepath: Path) -> tuple[bool, str, Optional[str]]:
    filename = filepath.name
    if DEPOT_FILE_PATTERN.search(filename):
        return True, "depot", None
    elif SERVICE_FILE_PATTERN.search(filename):
        return True, "service", None
    else:
        return False, "unknown", (
            f"Invalid filename format: {filename}. "
            "Expected: ops_daily_YYYY-MM-DD.csv or ops_daily_service_YYYY-MM-DD.csv"
        )


def validate_not_future_date(data_date: date) -> tuple[bool, Optional[str]]:
    today = date.today()
    if data_date > today:
        return False, f"Future date not allowed: {data_date} (today is {today})"
    return True, None


def validate_date_consistency(filepath: Path, data_date: date, file_type: str) -> tuple[bool, Optional[str]]:
    filename_date = extract_date_from_filename(filepath.name, file_type)
    if filename_date is None:
        return False, f"Could not extract date from filename: {filepath.name}"
    if filename_date != data_date:
        return False, (
            f"Date mismatch! Filename date: {filename_date}, "
            f"Data date: {data_date}. File: {filepath.name}"
        )
    return True, None


def check_already_processed(data_date: date, master_path: Path, date_col: str = "date") -> tuple[bool, int]:
    if not master_path.exists():
        return False, 0
    try:
        df = pd.read_csv(master_path)
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        existing = df[df[date_col].dt.date == data_date]
        return len(existing) > 0, len(existing)
    except Exception:
        return False, 0


def detect_date_gaps(data_date: date, master_path: Path, date_col: str = "date", max_gap_days: int = 30) -> list[date]:
    if not master_path.exists():
        return []
    try:
        df = pd.read_csv(master_path)
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        if len(df) == 0:
            return []
        last_date = df[date_col].max().date()
        if data_date <= last_date:
            return []
        existing_dates = set(df[date_col].dt.date.unique())
        missing = []
        current = last_date + timedelta(days=1)
        while current < data_date and len(missing) < max_gap_days:
            if current not in existing_dates:
                missing.append(current)
            current += timedelta(days=1)
        return missing
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Error logging
# ---------------------------------------------------------------------------


def log_error(filepath: Path, error_type: str, error_message: str, step_failed: str = "unknown") -> None:
    error_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "file": str(filepath.name) if filepath else "N/A",
        "file_path": str(filepath) if filepath else "N/A",
        "error_type": error_type,
        "error_message": error_message,
        "step_failed": step_failed,
    }
    error_df = pd.DataFrame([error_record])
    if ERROR_LOG_CSV.exists():
        existing = pd.read_csv(ERROR_LOG_CSV)
        combined = pd.concat([existing, error_df], ignore_index=True)
        atomic_write_csv(combined, ERROR_LOG_CSV)
    else:
        atomic_write_csv(error_df, ERROR_LOG_CSV)


def log_success(record: dict) -> None:
    log_df = pd.DataFrame([record])
    if INGEST_LOG_CSV.exists():
        existing = pd.read_csv(INGEST_LOG_CSV)
        combined = pd.concat([existing, log_df], ignore_index=True)
        atomic_write_csv(combined, INGEST_LOG_CSV)
    else:
        atomic_write_csv(log_df, INGEST_LOG_CSV)


# ---------------------------------------------------------------------------
# Depot-level inbound functions
# ---------------------------------------------------------------------------


def read_inbound_csv(inbound_path: Path) -> pd.DataFrame:
    df = pd.read_csv(inbound_path)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", regex=True)]
    return df


def coerce_depot_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="raise")
    df["depot"] = df["depot"].astype("string").str.strip()
    for col in ["passengers_per_day", "actual_kms", "occupancy_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_depot_inbound(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in INBOUND_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors
    if df["depot"].isna().any():
        errors.append("Null depot found.")
    if df["date"].isna().any():
        errors.append("Null/invalid date found.")
    if df["date"].nunique() != 1:
        errors.append(f"File must have exactly one date; found {df['date'].nunique()}")
    if df.duplicated(subset=["depot", "date"]).any():
        errors.append("Duplicate (depot, date) rows found.")
    unknown = sorted(set(df["depot"].dropna().astype(str)) - set(ALLOWED_DEPOTS))
    if unknown:
        errors.append(f"Unknown depots: {unknown}")
    if len(df) != len(ALLOWED_DEPOTS):
        errors.append(f"Row count {len(df)} but expected {len(ALLOWED_DEPOTS)} (one per depot).")
    for col, (lo, hi) in BOUNDS.items():
        if col not in df.columns:
            continue
        if lo is not None and (df[col] < lo).any():
            errors.append(f"{col} has values < {lo}")
        if hi is not None and (df[col] > hi).any():
            errors.append(f"{col} has values > {hi}")
    return errors


# ---------------------------------------------------------------------------
# Depot-level RAW master
# ---------------------------------------------------------------------------


def load_depot_raw_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", regex=True)]
    df = df.dropna(subset=["depot", "date"], how="any")
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True, errors="coerce")
    df["depot"] = df["depot"].astype("string").str.strip()
    for col in ["passengers_per_day", "actual_kms", "occupancy_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[RAW_COLUMNS].sort_values(["depot", "date"]).reset_index(drop=True)
    return df


def upsert_depot_raw_master(master: pd.DataFrame, inbound: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    master = master.copy()
    inbound = inbound.copy()[RAW_COLUMNS]
    master_keys = set(zip(master["depot"].astype(str), master["date"].dt.date))
    inbound_keys = list(zip(inbound["depot"].astype(str), inbound["date"].dt.date))
    existing = [k in master_keys for k in inbound_keys]
    corrected = int(np.sum(existing))
    inserted = int(len(inbound) - corrected)
    key_df = inbound[["depot", "date"]].copy()
    merged = master.merge(key_df, on=["depot", "date"], how="left", indicator=True)
    master_kept = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    updated = pd.concat([master_kept, inbound], ignore_index=True)
    updated = updated.sort_values(["depot", "date"]).reset_index(drop=True)
    if updated.duplicated(subset=["depot", "date"]).any():
        raise ValueError("RAW master has duplicate (depot, date) after upsert.")
    return updated, inserted, corrected


# ---------------------------------------------------------------------------
# Service-level inbound functions
# ---------------------------------------------------------------------------


def coerce_service_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="raise")
    df["depot"] = df["depot"].astype("string").str.strip()
    df["service_number"] = df["service_number"].astype("string").str.strip()
    for col in ["actual_kms", "actual_trips", "seat_kms", "passenger_kms", "occupancy_ratio", "revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_service_inbound(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in SERVICE_INBOUND_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors
    if df["depot"].isna().any():
        errors.append("Null depot found.")
    if df["date"].isna().any():
        errors.append("Null/invalid date found.")
    if df["service_number"].isna().any():
        errors.append("Null service_number found.")
    if df["date"].nunique() != 1:
        errors.append(f"File must have exactly one date; found {df['date'].nunique()}")
    if df.duplicated(subset=["depot", "date", "service_number"]).any():
        errors.append("Duplicate (depot, date, service_number) rows found.")
    unknown = sorted(set(df["depot"].dropna().astype(str)) - set(ALLOWED_DEPOTS))
    if unknown:
        errors.append(f"Unknown depots: {unknown}")
    for col, (lo, hi) in SERVICE_BOUNDS.items():
        if col not in df.columns:
            continue
        if lo is not None and (df[col] < lo).any():
            errors.append(f"{col} has values < {lo}")
        if hi is not None and (df[col] > hi).any():
            errors.append(f"{col} has values > {hi}")
    return errors


# ---------------------------------------------------------------------------
# Service-level RAW master
# ---------------------------------------------------------------------------


def load_service_raw_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SERVICE_RAW_COLUMNS)
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", regex=True)]
    df = df.dropna(subset=["depot", "date"], how="any")
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True, errors="coerce")
    df["depot"] = df["depot"].astype("string").str.strip()
    df["service_number"] = df["service_number"].astype("string").str.strip()
    for col in ["actual_kms", "actual_trips", "seat_kms", "passenger_kms", "occupancy_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in SERVICE_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[SERVICE_RAW_COLUMNS].sort_values(["depot", "date", "service_number"]).reset_index(drop=True)
    return df


def upsert_service_raw_master(master: pd.DataFrame, inbound: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    master = master.copy()
    inbound = inbound.copy()
    for col in SERVICE_RAW_COLUMNS:
        if col not in inbound.columns:
            inbound[col] = None
    inbound = inbound[SERVICE_RAW_COLUMNS]
    master_keys = set(zip(
        master["depot"].astype(str),
        master["date"].dt.date,
        master["service_number"].astype(str),
    ))
    inbound_keys = list(zip(
        inbound["depot"].astype(str),
        inbound["date"].dt.date,
        inbound["service_number"].astype(str),
    ))
    existing = [k in master_keys for k in inbound_keys]
    corrected = int(np.sum(existing))
    inserted = int(len(inbound) - corrected)
    key_df = inbound[["depot", "date", "service_number"]].copy()
    merged = master.merge(key_df, on=["depot", "date", "service_number"], how="left", indicator=True)
    master_kept = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    updated = pd.concat([master_kept, inbound], ignore_index=True)
    updated = updated.sort_values(["depot", "date", "service_number"]).reset_index(drop=True)
    if updated.duplicated(subset=["depot", "date", "service_number"]).any():
        raise ValueError("Service RAW master has duplicate keys after upsert.")
    return updated, inserted, corrected


# ---------------------------------------------------------------------------
# Calendar loading
# ---------------------------------------------------------------------------


def load_telugu_calendar(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing telugu_calendar.csv at: {path}")
    cal = pd.read_csv(path)
    cal = cal.loc[:, ~cal.columns.astype(str).str.contains(r"^Unnamed", regex=True)]
    missing = [c for c in TELUGU_CAL_COLUMNS if c not in cal.columns]
    if missing:
        raise ValueError(f"Telugu calendar missing columns: {missing}")
    cal["date"] = cal["date"].astype(str).str.strip()
    cal["date"] = cal["date"].str.replace(r"\s+00:00:00$", "", regex=True)
    cal["date"] = pd.to_datetime(cal["date"], format="mixed", dayfirst=True, errors="coerce").dt.normalize()
    if cal["date"].isna().any():
        bad = cal[cal["date"].isna()][["date"]].head(10)
        raise ValueError(f"Unparseable dates in telugu_calendar.csv. Examples:\n{bad}")
    cal["telugu_thithi"] = cal["telugu_thithi"].astype("string").str.strip()
    cal["telugu_paksham"] = cal["telugu_paksham"].astype("string").str.strip()
    cal["telugu_month"] = cal["telugu_month"].astype("string").str.strip()
    cal["marriage_day"] = pd.to_numeric(cal["marriage_day"], errors="coerce").fillna(0.0)
    cal["moudyami_day"] = pd.to_numeric(cal["moudyami_day"], errors="coerce").fillna(0.0)
    cal = cal[TELUGU_CAL_COLUMNS].sort_values("date").drop_duplicates(subset=["date"])
    return cal


def _split_date_cell(cell) -> list:
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    s = s.replace("\n", ",").replace(";", ",").replace("|", ",")
    return [p.strip() for p in s.split(",") if p.strip()]


def load_holiday_calendar_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing holiday_calendar.csv at: {path}")
    hol = pd.read_csv(path)
    hol = hol.loc[:, ~hol.columns.astype(str).str.contains(r"^Unnamed", regex=True)]
    missing = [c for c in HOLIDAY_WIDE_COLUMNS if c not in hol.columns]
    if missing:
        raise ValueError(f"Holiday calendar missing columns: {missing}")
    year_cols = ["2023_dates", "2024_dates", "2025_dates", "2026_dates"]
    long = hol.melt(
        id_vars=["fes_hol_code", "Holiday_Festival", "fes_hol_category"],
        value_vars=year_cols,
        var_name="year_col",
        value_name="date_list",
    )
    long["date_strs"] = long["date_list"].apply(_split_date_cell)
    long = long.explode("date_strs", ignore_index=True)
    long["date"] = pd.to_datetime(long["date_strs"], dayfirst=True, errors="coerce")
    long = long.dropna(subset=["date"])
    long = long[["date", "fes_hol_code", "Holiday_Festival", "fes_hol_category"]].copy()
    long["fes_hol_code"] = pd.to_numeric(long["fes_hol_code"], errors="coerce").astype("Int64")
    long = long.sort_values(["date", "fes_hol_code"]).drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    return long


# ---------------------------------------------------------------------------
# GOLD layer building
# ---------------------------------------------------------------------------


def build_depot_gold(
    raw_df: pd.DataFrame,
    telugu_cal_df: pd.DataFrame,
    holiday_long_df: pd.DataFrame,
    depot_master_csv: Path = DEPOT_MASTER_CSV,
) -> pd.DataFrame:
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["date_key"] = df["date"].dt.normalize()
    df["depot"] = df["depot"].astype("string")
    tel = telugu_cal_df.copy()
    tel["date"] = pd.to_datetime(tel["date"], errors="coerce")
    tel["date_key"] = tel["date"].dt.normalize()
    tel = tel.drop(columns=["date"]).drop_duplicates(subset=["date_key"])
    hol = holiday_long_df.copy()
    hol["date"] = pd.to_datetime(hol["date"], errors="coerce")
    hol["date_key"] = hol["date"].dt.normalize()
    hol = hol.drop(columns=["date"]).drop_duplicates(subset=["date_key"])
    df = df.merge(tel, on="date_key", how="left")
    df = df.merge(hol, on="date_key", how="left")
    df["is_fes_hol"] = df["fes_hol_code"].notna()
    df["fes_hol_code"] = df["fes_hol_code"].fillna(0).astype("Int64")
    df["Holiday_Festival"] = df["Holiday_Festival"].fillna("NONE").astype("string")
    df["fes_hol_category"] = df["fes_hol_category"].fillna("NONE").astype("string")
    df["is_fes_hol"] = df["is_fes_hol"].astype("int8")
    df = df.drop(columns=["date_key"])
    # Compute passenger_kms = occupancy_ratio * actual_kms * avg_seats_per_bus
    if depot_master_csv.exists():
        depot_master = pd.read_csv(depot_master_csv)
        depot_master.columns = depot_master.columns.str.lower().str.strip()
        depot_master["depot"] = depot_master["depot"].astype("string")
        df = df.merge(depot_master[["depot", "avg_seats_per_bus"]], on="depot", how="left")
        df["avg_seats_per_bus"] = df["avg_seats_per_bus"].fillna(45)
    else:
        df["avg_seats_per_bus"] = 45
    df["passenger_kms"] = df["occupancy_ratio"] * df["actual_kms"] * df["avg_seats_per_bus"]
    df = df.drop(columns=["avg_seats_per_bus"])
    df = df[GOLD_COLUMNS].sort_values(["depot", "date"]).reset_index(drop=True)
    if df.duplicated(subset=["depot", "date"]).any():
        raise ValueError("Depot GOLD has duplicate (depot, date).")
    return df


def build_service_gold(
    raw_df: pd.DataFrame,
    telugu_cal_df: pd.DataFrame,
    holiday_long_df: pd.DataFrame,
) -> pd.DataFrame:
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_key"] = df["date"].dt.normalize()
    df["depot"] = df["depot"].astype("string")
    df["service_number"] = df["service_number"].astype("string")
    tel = telugu_cal_df.copy()
    tel["date"] = pd.to_datetime(tel["date"], errors="coerce")
    tel["date_key"] = tel["date"].dt.normalize()
    tel = tel.drop(columns=["date"]).drop_duplicates(subset=["date_key"])
    hol = holiday_long_df.copy()
    hol["date"] = pd.to_datetime(hol["date"], errors="coerce")
    hol["date_key"] = hol["date"].dt.normalize()
    hol = hol.drop(columns=["date"]).drop_duplicates(subset=["date_key"])
    df = df.merge(tel, on="date_key", how="left")
    df = df.merge(hol, on="date_key", how="left")
    df["is_fes_hol"] = df["fes_hol_code"].notna()
    df["fes_hol_code"] = df["fes_hol_code"].fillna(0).astype("Int64")
    df["Holiday_Festival"] = df["Holiday_Festival"].fillna("NONE").astype("string")
    df["fes_hol_category"] = df["fes_hol_category"].fillna("NONE").astype("string")
    df["is_fes_hol"] = df["is_fes_hol"].astype("int8")
    df = df.drop(columns=["date_key"])
    for col in SERVICE_GOLD_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[SERVICE_GOLD_COLUMNS].sort_values(["depot", "date", "service_number"]).reset_index(drop=True)
    if df.duplicated(subset=["depot", "date", "service_number"]).any():
        raise ValueError("Service GOLD has duplicate (depot, date, service_number).")
    return df


# ---------------------------------------------------------------------------
# Predictions tracker
# ---------------------------------------------------------------------------


def load_predictions_file(file_path: Path) -> pd.DataFrame:
    if file_path.exists():
        df = pd.read_parquet(file_path)
        for col in PREDICTIONS_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df
    return pd.DataFrame(columns=PREDICTIONS_COLUMNS)


def save_predictions_file(df: pd.DataFrame, file_path: Path) -> None:
    for col in ["run_date", "prediction_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    df.to_parquet(file_path, index=False)


def update_predictions_with_actuals(
    predictions_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    predictions_df = predictions_df.copy()
    if len(predictions_df) == 0:
        return predictions_df, 0
    predictions_df["prediction_date"] = pd.to_datetime(predictions_df["prediction_date"])
    gold_df = gold_df.copy()
    gold_df["date"] = pd.to_datetime(gold_df["date"])
    agg_cols: dict = {"occupancy_ratio": "mean"}
    if "passenger_kms" in gold_df.columns:
        agg_cols["passenger_kms"] = "sum"
    if "actual_kms" in gold_df.columns:
        agg_cols["actual_kms"] = "sum"
    actuals = gold_df.groupby(["depot", "date"]).agg(agg_cols).reset_index()
    pending_mask = predictions_df["status"] == "pending"
    updates_count = 0
    for pred_date in predictions_df.loc[pending_mask, "prediction_date"].unique():
        date_actuals = actuals[actuals["date"] == pred_date]
        if len(date_actuals) == 0:
            continue
        for depot in predictions_df.loc[
            (predictions_df["prediction_date"] == pred_date) & pending_mask, "depot"
        ].unique():
            depot_actual = date_actuals[date_actuals["depot"] == depot]
            if len(depot_actual) == 0:
                continue
            actual_pkm = depot_actual["passenger_kms"].values[0] if "passenger_kms" in depot_actual.columns else None
            actual_or = depot_actual["occupancy_ratio"].values[0] if "occupancy_ratio" in depot_actual else None
            actual_kms_val = depot_actual["actual_kms"].values[0] if "actual_kms" in depot_actual.columns else None
            pred_mask = (
                (predictions_df["prediction_date"] == pred_date)
                & (predictions_df["depot"] == depot)
                & (predictions_df["status"] == "pending")
            )
            if not pred_mask.any():
                continue
            predicted = predictions_df.loc[pred_mask, "predicted_passenger_kms"].values[0]
            estimated_kms = predictions_df.loc[pred_mask, "estimated_kms"].values[0]
            pkm_error = predicted - actual_pkm if predicted and actual_pkm else None
            pkm_error_pct = (pkm_error / actual_pkm * 100) if actual_pkm and actual_pkm > 0 else None
            km_error = None
            km_error_pct = None
            if actual_kms_val and actual_kms_val > 0 and estimated_kms:
                km_error = estimated_kms - actual_kms_val
                km_error_pct = km_error / actual_kms_val * 100
            predictions_df.loc[pred_mask, "actual_passenger_kms"] = actual_pkm
            predictions_df.loc[pred_mask, "actual_or"] = actual_or
            predictions_df.loc[pred_mask, "actual_kms"] = actual_kms_val
            predictions_df.loc[pred_mask, "pkm_error"] = pkm_error
            predictions_df.loc[pred_mask, "pkm_error_pct"] = pkm_error_pct
            predictions_df.loc[pred_mask, "km_error"] = km_error
            predictions_df.loc[pred_mask, "km_error_pct"] = km_error_pct
            predictions_df.loc[pred_mask, "status"] = "completed"
            updates_count += 1
    return predictions_df, updates_count


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


def archive_inbound_file(inbound_path: Path, target_date: date) -> Path:
    date_str = target_date.strftime("%Y-%m-%d")
    archived_name = f"{inbound_path.stem}__archived__{date_str}{inbound_path.suffix}"
    archived_path = ARCHIVE_DIR / archived_name
    if archived_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_path = ARCHIVE_DIR / f"{inbound_path.stem}__archived__{date_str}__{ts}{inbound_path.suffix}"
    shutil.move(str(inbound_path), str(archived_path))
    return archived_path


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def scan_inbound_files() -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {"depot": [], "service": [], "unknown": []}
    for f in sorted(INBOUND_DIR.glob("*.csv")):
        is_valid, file_type, _ = validate_filename_format(f)
        if is_valid:
            result[file_type].append(f)
        else:
            result["unknown"].append(f)
    return result


def pre_flight_check(filepath: Path, file_type: str) -> dict:
    result: dict = {
        "filepath": filepath,
        "file_type": file_type,
        "checks_passed": True,
        "errors": [],
        "warnings": [],
        "info": {},
    }
    try:
        df = read_inbound_csv(filepath)
        if file_type == "depot":
            df = coerce_depot_types(df)
            validation_errors = validate_depot_inbound(df)
            master_path = RAW_MASTER_CSV
        else:
            df = coerce_service_types(df)
            validation_errors = validate_service_inbound(df)
            master_path = RAW_SERVICE_MASTER_CSV
        data_date = df["date"].iloc[0].date()
        result["info"]["data_date"] = data_date
        result["info"]["row_count"] = len(df)
        result["info"]["depots"] = df["depot"].unique().tolist()
        date_valid, date_err = validate_date_consistency(filepath, data_date, file_type)
        if not date_valid:
            result["errors"].append(date_err)
            result["checks_passed"] = False
        future_valid, future_err = validate_not_future_date(data_date)
        if not future_valid:
            result["errors"].append(future_err)
            result["checks_passed"] = False
        if validation_errors:
            result["errors"].extend(validation_errors)
            result["checks_passed"] = False
        already_exists, existing_count = check_already_processed(data_date, master_path)
        if already_exists:
            result["warnings"].append(
                f"Date {data_date} already exists in master ({existing_count} rows). Will be OVERWRITTEN."
            )
        result["info"]["already_processed"] = already_exists
        gaps = detect_date_gaps(data_date, master_path)
        if gaps:
            result["warnings"].append(
                f"Missing dates detected: {[str(d) for d in gaps[:5]]}"
                + (f" (and {len(gaps)-5} more)" if len(gaps) > 5 else "")
            )
        result["info"]["date_gaps"] = gaps
    except Exception as e:
        result["errors"].append(f"Failed to read/parse file: {str(e)}")
        result["checks_passed"] = False
    return result


# ---------------------------------------------------------------------------
# Process single files
# ---------------------------------------------------------------------------


def process_depot_file(
    filepath: Path,
    archive: bool = True,
) -> dict:
    result: dict = {"success": False, "file": filepath.name}
    try:
        df = read_inbound_csv(filepath)
        df = df.loc[:, [c for c in df.columns if c in INBOUND_COLUMNS]]
        df = coerce_depot_types(df)
        data_date = df["date"].iloc[0].date()
        result["data_date"] = data_date
        master = load_depot_raw_master(RAW_MASTER_CSV)
        updated_master, inserted, corrected = upsert_depot_raw_master(master, df)
        atomic_write_csv(updated_master, RAW_MASTER_CSV)
        result["raw_inserted"] = inserted
        result["raw_corrected"] = corrected
        if archive:
            archived_path = archive_inbound_file(filepath, data_date)
            result["archived_to"] = str(archived_path)
        log_record = {
            "ingest_timestamp": datetime.now().isoformat(timespec="seconds"),
            "file_type": "depot",
            "inbound_file": str(filepath.name),
            "inbound_sha256": file_sha256(filepath) if filepath.exists() else "archived",
            "data_date": str(data_date),
            "rows_inserted": inserted,
            "rows_corrected": corrected,
        }
        log_success(log_record)
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
        log_error(filepath, "PROCESSING_ERROR", str(e), "process_depot_file")
    return result


def process_service_file(
    filepath: Path,
    archive: bool = True,
) -> dict:
    result: dict = {"success": False, "file": filepath.name}
    try:
        df = read_inbound_csv(filepath)
        allowed = set(SERVICE_INBOUND_COLUMNS) | set(SERVICE_RAW_COLUMNS)
        df = df.loc[:, [c for c in df.columns if c in allowed]]
        df = coerce_service_types(df)
        data_date = df["date"].iloc[0].date()
        result["data_date"] = data_date
        master = load_service_raw_master(RAW_SERVICE_MASTER_CSV)
        updated_master, inserted, corrected = upsert_service_raw_master(master, df)
        atomic_write_csv(updated_master, RAW_SERVICE_MASTER_CSV)
        result["raw_inserted"] = inserted
        result["raw_corrected"] = corrected
        if archive:
            archived_path = archive_inbound_file(filepath, data_date)
            result["archived_to"] = str(archived_path)
        log_record = {
            "ingest_timestamp": datetime.now().isoformat(timespec="seconds"),
            "file_type": "service",
            "inbound_file": str(filepath.name),
            "inbound_sha256": file_sha256(filepath) if filepath.exists() else "archived",
            "data_date": str(data_date),
            "rows_inserted": inserted,
            "rows_corrected": corrected,
        }
        log_success(log_record)
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
        log_error(filepath, "PROCESSING_ERROR", str(e), "process_service_file")
    return result


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_daily_pipeline(
    skip_preflight: bool = False,
    archive_files: bool = True,
    update_predictions: bool = True,
) -> dict:
    """
    Run the complete daily data pipeline.

    Returns dict with depot_files_processed, service_files_processed,
    predictions_updated, and errors.
    """
    results: dict = {
        "depot_files_processed": 0,
        "service_files_processed": 0,
        "predictions_updated": 0,
        "errors": [],
        "details": [],
    }

    # Step 1: Load calendars (needed for both inbound processing and gold rebuild)
    try:
        telugu_cal = load_telugu_calendar(TELUGU_CAL_PATH)
        holiday_long = load_holiday_calendar_long(HOLIDAY_CAL_PATH)
    except Exception as e:
        results["errors"].append(f"Calendar loading failed: {e}")
        return results

    # Step 2: Scan inbound
    files = scan_inbound_files()
    depot_files = files["depot"]
    service_files = files["service"]

    # Log unknown (bad filename) files
    for f in files["unknown"]:
        msg = f"Invalid filename format: {f.name}. Expected: ops_daily_YYYY-MM-DD.csv or ops_daily_service_YYYY-MM-DD.csv"
        log_error(f, "INVALID_FILENAME", msg, step_failed="scan_inbound_files")
        results["errors"].append(msg)

    # Step 3: Pre-flight checks
    if depot_files or service_files:
        if not skip_preflight:
            all_checks = []
            for f in depot_files:
                all_checks.append(pre_flight_check(f, "depot"))
            for f in service_files:
                all_checks.append(pre_flight_check(f, "service"))
            failed = [c for c in all_checks if not c["checks_passed"]]
            if failed:
                for c in failed:
                    for err in c["errors"]:
                        log_error(c["filepath"], "PREFLIGHT_FAILED", err, step_failed="pre_flight_check")
                    results["errors"].extend(c["errors"])
                return results

    # Step 4: Process depot files (raw upsert only)
    if depot_files:
        depot_files_sorted = sorted(
            depot_files,
            key=lambda f: extract_date_from_filename(f.name, "depot") or date.max,
        )
        for f in depot_files_sorted:
            result = process_depot_file(f, archive=archive_files)
            results["details"].append(result)
            if result["success"]:
                results["depot_files_processed"] += 1
            else:
                results["errors"].append(f"{f.name}: {result.get('error')}")

    # Step 5: Process service files (raw upsert only)
    if service_files:
        service_files_sorted = sorted(
            service_files,
            key=lambda f: extract_date_from_filename(f.name, "service") or date.max,
        )
        for f in service_files_sorted:
            result = process_service_file(f, archive=archive_files)
            results["details"].append(result)
            if result["success"]:
                results["service_files_processed"] += 1
            else:
                results["errors"].append(f"{f.name}: {result.get('error')}")

    # Step 6: Build depot gold ONCE from full raw master
    try:
        if RAW_MASTER_CSV.exists():
            raw_master = load_depot_raw_master(RAW_MASTER_CSV)
            if len(raw_master) > 0:
                gold = build_depot_gold(raw_master, telugu_cal, holiday_long)
                atomic_write_parquet(gold, GOLD_MASTER_PARQ)
                results["gold_rebuilt"] = True
                results["gold_rows"] = len(gold)
    except Exception as e:
        results["errors"].append(f"Gold rebuild failed: {e}")

    # Step 7: Build service gold ONCE from full raw master
    try:
        if RAW_SERVICE_MASTER_CSV.exists():
            raw_svc = load_service_raw_master(RAW_SERVICE_MASTER_CSV)
            if len(raw_svc) > 0:
                svc_gold = build_service_gold(raw_svc, telugu_cal, holiday_long)
                atomic_write_parquet(svc_gold, GOLD_SERVICE_PARQ)
                results["service_gold_rebuilt"] = True
    except Exception as e:
        results["errors"].append(f"Service gold rebuild failed: {e}")

    # Step 8: Update predictions with actuals ONCE using final depot gold
    if update_predictions and PREDICTIONS_FILE.exists() and GOLD_MASTER_PARQ.exists():
        try:
            gold = pd.read_parquet(GOLD_MASTER_PARQ)
            pred_df = load_predictions_file(PREDICTIONS_FILE)
            pred_df, predictions_updated = update_predictions_with_actuals(pred_df, gold)
            save_predictions_file(pred_df, PREDICTIONS_FILE)
            results["predictions_updated"] = predictions_updated
        except Exception as e:
            results["errors"].append(f"Predictions update failed: {e}")

    return results
