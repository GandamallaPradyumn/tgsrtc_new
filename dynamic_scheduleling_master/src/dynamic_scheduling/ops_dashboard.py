"""
Operations dashboard data and chart builders for TGSRTC.

Extracted from notebooks/ops_dashboard.ipynb.
Uses Plotly only (no matplotlib) for Streamlit-native rendering.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import warnings

import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

PREDICTIONS_FILE = OUTPUT_DIR / "predictions" / "daily_predictions.parquet"
GOLD_MASTER_PARQ = DATA_DIR / "processed" / "ops_daily_gold.parquet"
SCHEDULE_DIR = OUTPUT_DIR / "dynamic_schedule"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dashboard_data(lookback_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load all dashboard data.

    Returns (predictions_df, gold_df, depots_list_as_dict_with_key).
    Actually returns a tuple of (predictions_df, gold_df, info_dict)
    where info_dict has 'depots' key.
    """
    predictions_df = _load_predictions_data(PREDICTIONS_FILE, lookback_days)
    gold_df = _load_gold_data(GOLD_MASTER_PARQ, lookback_days)

    if len(predictions_df) > 0:
        depots = sorted(predictions_df["depot"].unique().tolist())
    elif len(gold_df) > 0:
        depots = sorted(gold_df["depot"].unique().tolist())
    else:
        depots = []

    info = {"depots": depots, "lookback_days": lookback_days}
    return predictions_df, gold_df, info


def _load_predictions_data(file_path: Path, lookback_days: int = 30) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(file_path)
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df["run_date"] = pd.to_datetime(df["run_date"])
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    df = df[df["prediction_date"] >= cutoff_date]
    return df.sort_values(["depot", "prediction_date"])


def _load_gold_data(file_path: Path, lookback_days: int = 30) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    df = df[df["date"] >= cutoff_date]
    return df.sort_values(["depot", "date"])


# ---------------------------------------------------------------------------
# Accuracy data extraction
# ---------------------------------------------------------------------------


def get_demand_accuracy_data(predictions_df: pd.DataFrame, depot: str) -> pd.DataFrame:
    if len(predictions_df) == 0:
        return pd.DataFrame()
    df = predictions_df[
        (predictions_df["depot"] == depot)
        & (predictions_df["status"].isin(["completed", "pending"]))
    ].copy()
    if len(df) == 0:
        return pd.DataFrame()
    # Prefer new passenger-km columns; fall back to legacy passenger columns
    if "predicted_passenger_kms" in df.columns and df["predicted_passenger_kms"].notna().any():
        cols = ["prediction_date", "predicted_passenger_kms", "actual_passenger_kms",
                "pkm_error", "pkm_error_pct", "status"]
    elif "predicted_passengers" in df.columns and df["predicted_passengers"].notna().any():
        cols = ["prediction_date", "predicted_passengers", "actual_passengers",
                "passenger_error", "passenger_error_pct", "status"]
    else:
        return pd.DataFrame()
    df = df[cols].copy()
    df.columns = [
        "Date", "Predicted Passenger-KMs", "Actual Passenger-KMs",
        "Passenger-KM Error", "Passenger-KM Error %", "Status",
    ]
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%d-%m-%Y")
    return df


def get_supply_accuracy_data(predictions_df: pd.DataFrame, depot: str) -> pd.DataFrame:
    if len(predictions_df) == 0:
        return pd.DataFrame()
    df = predictions_df[
        (predictions_df["depot"] == depot)
        & (predictions_df["status"] == "completed")
    ].copy()
    if len(df) == 0:
        return pd.DataFrame()
    cols = ["prediction_date", "estimated_kms", "actual_kms", "km_error", "km_error_pct"]
    cols = [c for c in cols if c in df.columns]
    if "estimated_kms" not in cols or "actual_kms" not in cols:
        return pd.DataFrame()
    df = df[cols].copy()
    if "km_error" not in df.columns:
        df["km_error"] = df["estimated_kms"] - df["actual_kms"]
    if "km_error_pct" not in df.columns:
        df["km_error_pct"] = (df["km_error"] / df["actual_kms"] * 100).where(df["actual_kms"] > 0, 0)
    df.columns = ["Date", "Estimated KMs", "Actual KMs", "KM Error", "KM Error %"]
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Actual KMs"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%d-%m-%Y")
    return df


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------


def calculate_accuracy_metrics(error_series: pd.Series) -> dict:
    errors = error_series.dropna()
    if len(errors) == 0:
        return {}
    return {
        "Records": len(errors),
        "Mean Error %": round(float(errors.mean()), 1),
        "Mean Abs Error %": round(float(errors.abs().mean()), 1),
        "Median Abs Error %": round(float(errors.abs().median()), 1),
        "Within +/-10%": round(float((errors.abs() <= 10).mean() * 100), 1),
        "Within +/-20%": round(float((errors.abs() <= 20).mean() * 100), 1),
    }


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------


def build_demand_accuracy_chart(df: pd.DataFrame, depot: str) -> go.Figure:
    """Predicted vs Actual Passenger-KMs line chart."""
    if len(df) == 0:
        return _empty_figure("No demand accuracy data available")

    fig = go.Figure()

    completed = df[df["Status"] == "completed"] if "Status" in df.columns else df
    pending = df[df["Status"] == "pending"] if "Status" in df.columns else pd.DataFrame()

    fig.add_trace(
        go.Scatter(
            x=completed["Date"], y=completed["Predicted Passenger-KMs"],
            name="Predicted Passenger-KMs",
            mode="lines+markers", line=dict(color="blue", width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=completed["Date"], y=completed["Actual Passenger-KMs"],
            name="Actual Passenger-KMs",
            mode="lines+markers", line=dict(color="green", width=2),
            marker=dict(size=6),
        )
    )

    if len(pending) > 0:
        # Connect pending predictions to the last completed point
        bridge = completed.tail(1) if len(completed) > 0 else pd.DataFrame()
        pending_with_bridge = pd.concat([bridge, pending], ignore_index=True)
        fig.add_trace(
            go.Scatter(
                x=pending_with_bridge["Date"],
                y=pending_with_bridge["Predicted Passenger-KMs"],
                name="Upcoming Predictions",
                mode="lines+markers",
                line=dict(color="blue", width=2, dash="dash"),
                marker=dict(size=8, symbol="diamond"),
            )
        )

    fig.update_layout(
        title=f"Predicted vs Actual Passenger-KMs — {depot}",
        xaxis_title="Date",
        yaxis_title="Passenger-KMs",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_demand_error_chart(df: pd.DataFrame, depot: str) -> go.Figure:
    """Passenger-KM prediction error % bar chart with threshold lines."""
    if len(df) == 0:
        return _empty_figure("No demand error data available")

    df = df.dropna(subset=["Passenger-KM Error %"])
    if len(df) == 0:
        return _empty_figure("No demand error data available")
    colors = ["green" if x <= 0 else "red" for x in df["Passenger-KM Error %"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df["Date"], y=df["Passenger-KM Error %"], name="Passenger-KM Error %",
               marker_color=colors, opacity=0.7)
    )
    fig.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="+10%")
    fig.add_hline(y=-10, line_dash="dash", line_color="orange", annotation_text="-10%")
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"Passenger-KM Prediction Error % — {depot}",
        xaxis_title="Date",
        yaxis_title="Passenger-KM Error %",
        height=350,
        showlegend=False,
    )
    return fig


def build_supply_accuracy_chart(df: pd.DataFrame, depot: str) -> go.Figure:
    """Estimated vs Actual KMs line chart."""
    if len(df) == 0:
        return _empty_figure("No supply accuracy data available")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["Estimated KMs"], name="Estimated KMs",
            mode="lines+markers", line=dict(color="blue", width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["Actual KMs"], name="Actual KMs",
            mode="lines+markers", line=dict(color="green", width=2),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        title=f"Estimated vs Actual KMs — {depot}",
        xaxis_title="Date",
        yaxis_title="Kilometers",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_supply_error_chart(df: pd.DataFrame, depot: str) -> go.Figure:
    """KM error % bar chart with threshold lines."""
    if len(df) == 0:
        return _empty_figure("No supply error data available")

    df = df.dropna(subset=["KM Error %"])
    if len(df) == 0:
        return _empty_figure("No supply error data available")
    colors = ["green" if x <= 0 else "red" for x in df["KM Error %"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df["Date"], y=df["KM Error %"], name="KM Error %",
               marker_color=colors, opacity=0.7)
    )
    fig.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="+10%")
    fig.add_hline(y=-10, line_dash="dash", line_color="orange", annotation_text="-10%")
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"KM Error % — {depot}",
        xaxis_title="Date",
        yaxis_title="KM Error %",
        height=350,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Operations Overview — EPK-OR quadrant analysis
# ---------------------------------------------------------------------------

QUADRANT_COLORS = {
    "UNDERSUPPLY": "#2ecc71",
    "OVERSUPPLY": "#3498db",
    "SOCIAL_OBLIGATION": "#f39c12",
    "INEFFICIENT": "#e74c3c",
}


def get_operations_overview_data(
    schedules: dict[str, pd.DataFrame],
    depot: str,
) -> dict | None:
    """Extract quadrant analysis from the latest EPK schedule for *depot*.

    If the loaded schedule lacks a ``quadrant`` column (pre-existing files),
    computes it on-the-fly from ``epk``, ``cpk``, ``or`` using
    :func:`classify_epk_or_quadrant`.

    Returns dict with ``schedule_df``, ``quadrant_counts``, ``quadrant_pcts``,
    ``financial_summary``, ``action_summary`` — or ``None`` if no EPK schedule
    is available.
    """
    if depot not in schedules:
        return None

    df = schedules[depot].copy()
    is_epk = df.get("_engine", pd.Series(dtype=str)).eq("epk").any()
    if not is_epk:
        return None

    # Compute quadrant on-the-fly if missing
    if "quadrant" not in df.columns:
        from dynamic_scheduling.supply_scheduling import classify_epk_or_quadrant

        or_boundary = 0.70
        if "epk" in df.columns and "cpk" in df.columns and "or" in df.columns:
            df["quadrant"] = df.apply(
                lambda r: classify_epk_or_quadrant(r["epk"], r["cpk"], r["or"], or_boundary),
                axis=1,
            )
        else:
            return None

    # Compute contribution on-the-fly if missing
    if "contribution" not in df.columns:
        if "revenue" in df.columns and "cpk" in df.columns and "planned_kms" in df.columns:
            df["contribution"] = df["revenue"] - (df["cpk"] * df["planned_kms"])
        else:
            df["contribution"] = 0.0

    total = len(df)
    quadrant_counts = df["quadrant"].value_counts().to_dict()
    quadrant_pcts = {q: round(c / total * 100, 1) if total > 0 else 0.0
                     for q, c in quadrant_counts.items()}

    total_revenue = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
    total_contribution = float(df["contribution"].sum())
    depot_avg_epk = float(df["epk"].mean()) if "epk" in df.columns and total > 0 else 0.0
    depot_avg_or = float(df["or"].mean()) if "or" in df.columns and total > 0 else 0.0

    action_counts = df["action"].value_counts().to_dict() if "action" in df.columns else {}

    return {
        "schedule_df": df,
        "quadrant_counts": quadrant_counts,
        "quadrant_pcts": quadrant_pcts,
        "financial_summary": {
            "total_revenue": total_revenue,
            "total_contribution": total_contribution,
            "depot_avg_epk": depot_avg_epk,
            "depot_avg_or": depot_avg_or,
        },
        "action_summary": {
            "total_services": total,
            "add_slot": action_counts.get("ADD_SLOT", 0),
            "cut": action_counts.get("CUT", 0),
            "no_change": action_counts.get("NO_CHANGE", 0),
        },
    }


def build_epk_or_scatter(schedule_df: pd.DataFrame, depot: str) -> go.Figure:
    """Plotly scatter: EPK (y) vs OR (x), colored by quadrant, sized by allocated_pkm."""
    if len(schedule_df) == 0 or "epk" not in schedule_df.columns or "or" not in schedule_df.columns:
        return _empty_figure("No EPK-OR data available")

    df = schedule_df.copy()

    # Determine CPK and OR boundary lines
    cpk_line = float(df["cpk"].mean()) if "cpk" in df.columns else 25.0
    or_boundary = 0.70

    # Size by allocated_pkm (normalize for display)
    if "allocated_pkm" in df.columns:
        max_pkm = df["allocated_pkm"].max()
        df["_size"] = np.where(max_pkm > 0, df["allocated_pkm"] / max_pkm * 30 + 5, 10)
    else:
        df["_size"] = 10

    fig = go.Figure()

    quadrant_col = "quadrant" if "quadrant" in df.columns else None
    if quadrant_col:
        for quad, color in QUADRANT_COLORS.items():
            mask = df[quadrant_col] == quad
            sub = df[mask]
            if len(sub) == 0:
                continue
            hover_text = sub.apply(
                lambda r: (
                    f"Depot: {r.get('depot','-')}<br>"
                    f"Service: {r.get('service_number', '?')}<br>"
                    f"EPK: {r['epk']:.2f}<br>OR: {r['or']:.2f}<br>"
                    f"CPK: {r.get('cpk', 0):.2f}<br>"
                    f"Action: {r.get('action', '-')}"
                ),
                axis=1,
            )
            fig.add_trace(go.Scatter(
                x=sub["or"], y=sub["epk"],
                mode="markers",
                name=quad,
                marker=dict(color=color, size=sub["_size"], opacity=0.7),
                text=hover_text,
                hoverinfo="text",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df["or"], y=df["epk"],
            mode="markers",
            marker=dict(size=df["_size"], opacity=0.7),
        ))

    # CPK horizontal line (profit boundary)
    fig.add_hline(y=cpk_line, line_dash="dash", line_color="red",
                  annotation_text=f"CPK = {cpk_line:.1f}")
    # OR vertical line (boundary)
    fig.add_vline(x=or_boundary, line_dash="dash", line_color="gray",
                  annotation_text=f"OR = {or_boundary:.0%}")

    # Quadrant label annotations
    y_max = float(df["epk"].max()) if len(df) > 0 else cpk_line * 2
    x_max = float(df["or"].max()) if len(df) > 0 else 1.0
    annotations = [
        dict(x=or_boundary + (x_max - or_boundary) / 2, y=y_max * 0.95,
             text="UNDERSUPPLY", showarrow=False, font=dict(color=QUADRANT_COLORS["UNDERSUPPLY"], size=11)),
        dict(x=or_boundary / 2, y=y_max * 0.95,
             text="OVERSUPPLY", showarrow=False, font=dict(color=QUADRANT_COLORS["OVERSUPPLY"], size=11)),
        dict(x=or_boundary + (x_max - or_boundary) / 2, y=cpk_line * 0.5,
             text="SOCIAL OBLIGATION", showarrow=False, font=dict(color=QUADRANT_COLORS["SOCIAL_OBLIGATION"], size=11)),
        dict(x=or_boundary / 2, y=cpk_line * 0.5,
             text="INEFFICIENT", showarrow=False, font=dict(color=QUADRANT_COLORS["INEFFICIENT"], size=11)),
    ]

    fig.update_layout(
        title=f"EPK vs OR Quadrant Analysis — {depot}",
        xaxis_title="Occupancy Ratio (OR)",
        yaxis_title="Earnings Per KM (EPK)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=annotations,
    )
    return fig


def build_quadrant_breakdown_chart(quadrant_counts: dict, depot: str) -> go.Figure:
    """Horizontal bar chart showing service count per quadrant."""
    if not quadrant_counts:
        return _empty_figure("No quadrant data available")

    quadrant_order = ["UNDERSUPPLY", "OVERSUPPLY", "SOCIAL_OBLIGATION", "INEFFICIENT"]
    labels = [q for q in quadrant_order if q in quadrant_counts]
    values = [quadrant_counts[q] for q in labels]
    colors = [QUADRANT_COLORS.get(q, "#999999") for q in labels]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=values,
        textposition="auto",
    ))
    fig.update_layout(
        title=f"Service Count by Quadrant — {depot}",
        xaxis_title="Number of Services",
        yaxis_title="",
        height=300,
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"),
    )
    fig.update_layout(height=300)
    return fig


# ---------------------------------------------------------------------------
# Schedule loading
# ---------------------------------------------------------------------------


def list_schedule_dates(schedule_dir: Path = SCHEDULE_DIR, depot:str |None = None) -> list[str]:
    """Return available schedule date strings (newest first)."""
    if not schedule_dir.exists():
        return []
    # All date folders under output/dynamic_schedule/
    dates = sorted(
        [d.name for d in schedule_dir.iterdir() if d.is_dir()],
        reverse=True,
    )

    # If no depot (admin/corporation case), return all dates
    if not depot:
        return dates

    # === YOUR REAL FILE PATTERN ===
    # epk_schedule_<DEPOT>_<DATE>.xlsx
    filtered = []
    for d in dates:
        folder = schedule_dir / d
        expected_file = folder / f"epk_schedule_{depot.upper()}_{d}.xlsx"
        if expected_file.exists():
            filtered.append(d)

    return filtered


def load_schedule_for_date(
    schedule_date: str, schedule_dir: Path = SCHEDULE_DIR, depot: str | None = None
) -> tuple[dict[str, pd.DataFrame], str | None]:
    """
    Load schedule files for a specific date folder.
    Filters out consolidated_*.xlsx files.
    Returns (dict of {depot_name: DataFrame}, schedule_date_string or None).
    """
    folder = schedule_dir / schedule_date
    schedules: dict[str, pd.DataFrame] = {}
    if not folder.exists():
        return schedules, None
    # -------- USER-SCOPED MODE (NORMAL CASE) --------
    if depot:
        file_path = folder / f"epk_schedule_{depot.upper()}_{schedule_date}.xlsx"

        if not file_path.exists():
            return {}, None

        try:
            df = pd.read_excel(file_path)
            df["_engine"] = "epk"
            return {depot: df}, schedule_date
        except Exception:
            return {}, None
        
    for f in sorted(folder.glob("*.xlsx")):
        if "consolidated_" in f.name:
            continue
        parts = f.stem.split("_")
        is_epk = f.name.startswith("epk_schedule_")
        if is_epk and len(parts) >= 3:
            depot = parts[2]
        elif f.name.startswith("schedule_") and len(parts) >= 2:
            depot = parts[1]
        else:
            continue
        if depot in schedules and not is_epk:
            existing_engine = schedules[depot].get("_engine", pd.Series(dtype=str))
            if existing_engine.eq("epk").any():
                continue
        try:
            df = pd.read_excel(f)
            df["_engine"] = "epk" if is_epk else "delta_kms"
            schedules[depot] = df
        except Exception:
            pass

    return schedules, schedule_date


def load_latest_schedule(schedule_dir: Path = SCHEDULE_DIR , depot: str | None = None
) -> tuple[dict[str, pd.DataFrame], str | None]:
    """
    Load the most recent schedule files for each depot.
    Filters out consolidated_*.xlsx files.
    Returns (dict of {depot_name: DataFrame}, schedule_date_string or None).
    """
    dates = list_schedule_dates(schedule_dir,depot=depot)
    if not dates:
        return {}, None
    return load_schedule_for_date(dates[0], schedule_dir, depot=depot)
