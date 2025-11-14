import pandas as pd
import streamlit as st
import plotly.express as px
import uuid
from sqlalchemy import create_engine
import itertools
import hashlib
import calendar
from urllib.parse import quote_plus
from auth import get_depot_by_userid, get_role_by_userid
import json
from mysql.connector import Error
import pymysql
from datetime import datetime, timedelta, time

# ---------------------- 1. CONFIG & DB SETUP --------------------------
# --- NOTE: config.json must contain 'db' details (host, user, password, database) ---
with open("config.json") as f:
    config = json.load(f)
DB_CONFIG = config.get("db", {})

def get_connection():
    try:
        return pymysql.connect(
            host=DB_CONFIG.get("host", ""),
            user=DB_CONFIG.get("user", ""),
            password=DB_CONFIG.get("password", ""),
            database=DB_CONFIG.get("database", ""),
        )
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None  

def get_depot_color(depot):
    hex_digest = hashlib.md5(depot.encode()).hexdigest()
    return f'#{hex_digest[:6]}'

def get_current_user_depot():
    if "userid" in st.session_state and st.session_state.userid != "admin":
        depot1 = get_depot_by_userid(st.session_state.userid)  # Call imported function
        st.session_state.depot = depot1
        return depot1
    elif st.session_state.userid == "admin":
        return ""  # Admin logic
    else:
        return ""

role = None
if "userid" in st.session_state:
    role = get_role_by_userid(st.session_state.userid)

def get_depot_settings(_conn):
    """
    Fetches depot configuration settings (region and category) from TS_ADMIN.
    Returns a dictionary mapping DEPOT_NAME -> {'region': R, 'category': C}.
    """
    if _conn is None:
        st.error("Database connection not available for fetching depot settings.")
        return {}
    cursor = None
    try:
        cursor = _conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT depot_name, region, category FROM TS_ADMIN")
        rows = cursor.fetchall()
        settings = {}
        for row in rows:
            depot_name = row.get("depot_name")
            if depot_name is not None:
                normalized_depot = str(depot_name).strip().upper()
                settings[normalized_depot] = {
                    'region': str(row.get("region", "")).strip().upper(),
                    'category': str(row.get("category", "")).strip().title()
                }
        return settings
    except Exception as e:
        st.error(f"Error fetching depot settings: {e}")
        return {}
    finally:
        if cursor:
            cursor.close()

# ---------------------- 2. DATA FETCHING & PROCESSING --------------------------
#@st.cache_data(show_spinner=False, ttl=3600) # Cache the result for 1 hour
def preaggregate(selected_depot, depot_settings):
    
    OP_DATE = "operations_date"
    KM = "opd_kms"
    HOURS = "hours"
    EARNINGS = "daily_earnings"
    DEPOT = "depot"
    EMP_ID = "employee_id"
    LSA_DATE = "date"
    LEAVE_TYPE = "leave_type"

    if not selected_depot:
        st.warning("⚠️ No depot selected for current user.")
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df, empty_df

    selected_depot = str(selected_depot).replace("\xa0", " ").strip().upper()

    try:
        db_user = config['db']['user']
        db_password = quote_plus(config['db']['password'])
        db_host = config['db']['host']
        db_name = config['db']['database']
        engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    except Exception as e:
        st.error(f"❌ Error creating DB engine: {e}")
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df, empty_df

    def read_table(query, params=None):
        try:
            with st.spinner("Loading data…"):
                df = pd.read_sql(query, engine, params=params)
            df.columns = [c.strip().lower() for c in df.columns]
            return df
        except Exception as e:
            st.error(f"❌ Database error: {e}")
            return pd.DataFrame()

    # Query only for selected depot — keep existing behavior for per-depot charts
    rtc_query = f"""
        SELECT * FROM daily_operations 
        WHERE TRIM({DEPOT}) LIKE %s COLLATE utf8mb4_general_ci
    """
    rtc = read_table(rtc_query, params=(selected_depot,))

    service_master = read_table("SELECT service_number, dept_time, arr_time, day_night_code, depot FROM service_master")

    lsa_query = f"""
        SELECT * FROM driver_absenteeism 
        WHERE TRIM({DEPOT}) LIKE %s COLLATE utf8mb4_general_ci
    """
    lsa = read_table(lsa_query, params=(selected_depot,))

    ghc_query = f"""
        SELECT * FROM ghc_2024 
        WHERE TRIM({DEPOT}) LIKE %s COLLATE utf8mb4_general_ci
    """
    ghc_2024 = read_table(ghc_query, params=(selected_depot,))

    for df in [rtc, service_master, lsa, ghc_2024]:
        if DEPOT in df.columns:
            df[DEPOT] = df[DEPOT].astype(str).str.strip().str.upper()

    for df in [rtc, service_master]:
        if 'service_number' in df.columns:
            df['service_number'] = df['service_number'].astype(str).str.strip().str.upper()

    def convert_time(x):
        if pd.isnull(x): return None
        if isinstance(x, (pd.Timedelta, datetime)): return (datetime.min + x if isinstance(x, pd.Timedelta) else x).time()
        if isinstance(x, str):
            try: return datetime.strptime(x, "%H:%M:%S").time()
            except: return None
        return None

    def time_cal(dep, arr, day_night_code):
        if not isinstance(dep, time) or not isinstance(arr, time): return None
        dummy_date = datetime.today().date()
        DEP_TIME_dt = datetime.combine(dummy_date, dep)
        ARR_TIME_dt = datetime.combine(dummy_date, arr)
        if day_night_code == 'N' or ARR_TIME_dt < DEP_TIME_dt:
            ARR_TIME_dt += timedelta(days=1)
            duration = ARR_TIME_dt - DEP_TIME_dt
            if day_night_code == 'N' and duration.total_seconds() < 6 * 3600:
                ARR_TIME_dt += timedelta(days=1)
                duration = ARR_TIME_dt - DEP_TIME_dt
        else:
            duration = ARR_TIME_dt - DEP_TIME_dt
        return duration.total_seconds() / 3600

    if not service_master.empty:
        for col in ['dept_time', 'arr_time']:
            if col in service_master.columns:
                service_master[col] = service_master[col].apply(convert_time)

    if not rtc.empty and not service_master.empty:
        with st.spinner("Merging data…"):
            rtc[DEPOT] = rtc[DEPOT].astype(str).str.strip().str.upper()
            service_master[DEPOT] = service_master[DEPOT].astype(str).str.strip().str.upper()
            service_master_unique = (
                service_master.drop_duplicates(subset=['depot', 'service_number'], keep='first')
            )
            merged_df = pd.merge(
                rtc,
                service_master_unique[['depot', 'service_number', 'dept_time', 'arr_time', 'day_night_code']],
                on=['depot', 'service_number'],
                how='left',
                suffixes=('_rtc', '_master')
            )
            merged_df[HOURS] = merged_df.apply(
                lambda row: time_cal(row.get('dept_time'), row.get('arr_time'), row.get('day_night_code')),
                axis=1
            )
            rtc = merged_df
            rtc[DEPOT] = rtc[DEPOT].astype(str).str.strip().str.upper()
    elif not rtc.empty:
        rtc[HOURS] = None

    for col in [OP_DATE, KM, HOURS, EARNINGS]:
        if col in rtc.columns:
            if col == OP_DATE:
                rtc[col] = pd.to_datetime(rtc[col], errors='coerce')
            else:
                rtc[col] = pd.to_numeric(rtc[col], errors='coerce')

    if not rtc.empty and OP_DATE in rtc.columns:
        rtc = rtc.dropna(subset=[OP_DATE])
        rtc["month"] = rtc[OP_DATE].dt.to_period("M").dt.to_timestamp()
        rtc["year"] = rtc[OP_DATE].dt.year
        with st.spinner("Aggregating depot and driver data…"):
            depot_monthly = rtc.groupby([DEPOT, 'year', 'month']).agg(
                total_km=(KM, "sum"),
                total_hours=(HOURS, "sum"),
                total_earnings=(EARNINGS, "sum")
            ).reset_index()

            driver_monthly = rtc.groupby([DEPOT, EMP_ID, 'year', 'month']).agg(
                total_km=(KM, "sum"),
                total_hours=(HOURS, "sum"),
                total_earnings=(EARNINGS, "sum")
            ).reset_index()
    else:
        depot_monthly = pd.DataFrame(columns=[DEPOT, 'year', 'month', 'total_km', 'total_hours', 'total_earnings'])
        driver_monthly = pd.DataFrame(columns=[DEPOT, EMP_ID, 'year', 'month', 'total_km', 'total_hours', 'total_earnings'])

    if not lsa.empty:
        if LSA_DATE in lsa.columns:
            lsa[LSA_DATE] = pd.to_datetime(lsa[LSA_DATE], errors='coerce')
        lsa_valid = lsa.dropna(subset=[LEAVE_TYPE]).copy()
        if LSA_DATE in lsa_valid.columns:
            lsa_valid["month"] = lsa_valid[LSA_DATE].dt.to_period("M").dt.to_timestamp()
            lsa_valid["year"] = lsa_valid[LSA_DATE].dt.year
            if not depot_monthly.empty:
                depot_abs = lsa_valid.groupby([DEPOT, 'year', 'month']).size().rename("absenteeism").reset_index()
                depot_monthly = pd.merge(depot_monthly, depot_abs, on=[DEPOT, 'year', 'month'], how='left').fillna(0)
            if not driver_monthly.empty:
                driver_abs = lsa_valid.groupby([DEPOT, EMP_ID, 'year', 'month']).size().rename("absenteeism").reset_index()
                driver_monthly = pd.merge(driver_monthly, driver_abs, on=[DEPOT, EMP_ID, 'year', 'month'], how='left').fillna(0)
        else:
            lsa_valid = pd.DataFrame()
    if lsa.empty or LSA_DATE not in lsa.columns:
        lsa_valid = pd.DataFrame()
        if not depot_monthly.empty:
            depot_monthly["absenteeism"] = 0
        if not driver_monthly.empty:
            driver_monthly["absenteeism"] = 0

    if depot_monthly.empty:
        depot_monthly["Category"] = ""
    elif "Category" not in depot_monthly.columns:
        depot_monthly["Category"] = depot_monthly[DEPOT].map(lambda d: depot_settings.get(d, {}).get("category", "Unknown"))
    if "region" not in depot_monthly.columns:
        depot_monthly["region"] = depot_monthly[DEPOT].map(lambda d: depot_settings.get(d, {}).get("region", "Unknown"))

    return depot_monthly, driver_monthly, lsa_valid, ghc_2024

def format_indian(num):
    """Format number in Indian numbering system (Lakhs, Crores)."""
    try:
        num = int(round(num))
    except (ValueError, TypeError):
        return str(num)
    s = str(num)
    if len(s) <= 3:
        return s
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)
    return ",".join(parts[::-1]) + "," + last3

# ---------------------- 3. VISUALIZATION & REGION HELPERS --------------------------

def plot_comparison_bar(combined_df, ycol, avg1, avg2, depot1, depot2, title, ytitle, compare_with_region=False):
    depot1 = depot1.strip()
    depot2 = depot2.strip()
    depots = combined_df["Depot"].unique().tolist()
    color_map = {}
    for depot in depots:
        if depot == depot1:
            color_map[depot] = "#1f77b4"
        elif depot == depot2 and not compare_with_region:
            color_map[depot] = "#add8e6"
        else:
            color_map[depot] = "#d3d3d3"
    if compare_with_region:
        if depot2 not in color_map:
            color_map[depot2] = "#B62929"
    chart_df = (
        combined_df.groupby(['Depot', 'year', 'month'])
        .agg(total=(ycol, 'sum'))
        .reset_index()
        .sort_values(['year', 'month'])
    )
    chart_df['month_label'] = chart_df['month'].dt.strftime('%b-%y')
    fig = px.bar(
        chart_df,
        x='month_label',
        y='total',
        color='Depot',
        barmode='group',
        labels={'total': ytitle, 'month_label': 'Month-Year'},
        text_auto=True,
        color_discrete_map=color_map,
    )
    # fix: use depot1 color for depot1 hline
    if depot1 in color_map:
        fig.add_hline(
            y=avg1,
            line_dash="dot",
            line_color=color_map[depot1],
            annotation_text=f"{depot1} Avg: {avg1:,.0f}",
            annotation_position="top left"
        )
    if compare_with_region:
        fig.add_hline(
            y=avg2,
            line_dash="dot",
            line_color=color_map.get(depot2, "#B62929"),
            annotation_text=f"{depot2} Region Avg: {avg2:,.0f}",
            annotation_position="top right"
        )
    elif depot2 in color_map:
        fig.add_hline(
            y=avg2,
            line_dash="dot",
            line_color=color_map[depot2],
            annotation_text=f"{depot2} Avg: {avg2:,.0f}",
            annotation_position="top right"
        )
    fig.update_layout(
        title=title,
        yaxis_tickformat=',',
        xaxis_title="Month-Year",
        yaxis_title=ytitle,
        legend=dict(
            title="Depot",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))

def plot_bar(df, ycol, depot_avg=None, title="", ytitle="", depot=None, region_avg=None):
    if df.empty or ycol not in df.columns:
        st.info(f"ℹ️ Cannot display graph for '{ytitle}': Dataframe is empty or missing column '{ycol}'.")
        return
    df_safe = df.copy()
    df_safe[ycol] = pd.to_numeric(df_safe[ycol], errors='coerce')
    df_filtered = df_safe.dropna(subset=[ycol])
    if df_filtered[ycol].sum() == 0:
        st.info(f"ℹ️ Cannot display graph for '{ytitle}': Total value is zero.")
        return
    chart_df = (
        df_filtered
        .groupby(["year", "month"])
        .agg(total=(ycol, "sum"))
        .reset_index()
        .sort_values(["year", "month"])
    )
    if chart_df.empty:
        st.info(f"ℹ️ Cannot display graph for '{ytitle}': No data points found after aggregation.")
        return
    chart_df["month_label"] = chart_df["month"].dt.strftime("%b-%y")
    fig = px.bar(
        chart_df,
        x="month_label",
        y="total",
        labels={"total": ytitle, "month_label": "Month-Year"},
        color_discrete_sequence=["#1f77b4"],
        text_auto=True
    )
    if depot_avg is not None:
        fig.add_hline(
            y=depot_avg,
            line_dash="dot",
            line_color="#b41f1f",
            annotation_text=f"Depot Avg: {depot_avg:,.0f}",
            annotation_position="top left",
            annotation_font_color="black",
            annotation_font_size=12
        )
    if region_avg is not None:
        fig.add_hline(
            y=region_avg,
            line_dash="dash",
            line_color="#45B629",
            annotation_text=f"Region Avg: {region_avg:,.0f}",
            annotation_position="bottom right",
            annotation_font_color="black",
            annotation_font_size=12
        )
    fig.update_traces(
        texttemplate='%{y:,.0f}', 
        textposition='inside', 
        textfont=dict(color='black', size=14, family='Arial')
    )
    fig.update_layout(
        title=title,
        yaxis_tickformat=',',
        xaxis_title="Month-Year",
        yaxis_title=ytitle,
        bargap=0.3
    )
    st.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))


def compute_region_avgs(depot_monthly, depot_name, depot_settings):
    """Compute region-level averages for the region to which the selected depot belongs."""
    if depot_monthly.empty or not depot_name:
        return {}
    depot_upper = str(depot_name).strip().upper()
    depot_info = depot_settings.get(depot_upper)
    if not depot_info:
        return {}
    region = depot_info.get("region")
    if not region:
        return {}
    region_depots = [d for d, info in depot_settings.items() if info.get("region") == region]
    region_df = depot_monthly[depot_monthly["depot"].str.upper().isin(region_depots)]
    if region_df.empty:
        return {}
    return {
        "total_km": region_df["total_km"].mean(),
        "total_hours": region_df["total_hours"].mean(),
        "total_earnings": region_df["total_earnings"].mean(),
        "absenteeism": region_df["absenteeism"].mean()
    }

def compute_true_region_avg(depot_monthly, depot_name, depot_settings, year_sel="All"):
    """Compute region-wide average for the selected depot's region."""
    if depot_monthly.empty or not depot_name:
        return {}
    depot_upper = str(depot_name).strip().upper()
    depot_info = depot_settings.get(depot_upper)
    if not depot_info:
        return {}
    region = depot_info.get("region")
    if not region:
        return {}
    region_depots = [d for d, info in depot_settings.items() if info.get("region") == region]
    region_df = depot_monthly[depot_monthly["depot"].str.upper().isin(region_depots)].copy()
    if year_sel != "All":
        region_df = region_df[region_df["year"] == int(year_sel)]
    if region_df.empty:
        return {}
    return {
        "total_km": region_df["total_km"].mean(),
        "total_hours": region_df["total_hours"].mean(),
        "total_earnings": region_df["total_earnings"].mean(),
        "absenteeism": region_df["absenteeism"].mean()
    }

def show_main_bar_line_charts(df1, depot, depot_monthly=None, depot_settings=None, mysql_conn=None):
    """
    Display depot-level performance charts with an optional region average overlay.
    When 'Show Region Average' checkbox is enabled, region-wide averages are drawn as lines.
    Prefer DB-driven region averages (per-depot-per-month) from get_region_avgs_from_ts_admin().
    Fall back to computing per-depot-per-month averages from depot_monthly when DB is unavailable.
    """
    show_region_avg = st.checkbox("Show Region Average", value=False, key="show_region_avg_main")

    if df1 is None or df1.empty:
        st.info(f"No data available for depot {depot}.")
        return

    # Depot averages (monthly total units) used for the depot horizontal line
    depot_avg_km = df1["total_km"].mean() if "total_km" in df1.columns else 0
    depot_avg_hours = df1["total_hours"].mean() if "total_hours" in df1.columns else 0
    depot_avg_earn = df1["total_earnings"].mean() if "total_earnings" in df1.columns else 0
    depot_avg_abs = df1["absenteeism"].mean() if "absenteeism" in df1.columns else 0

    # Region averages (we want per-depot-per-month units)
    region_avg_km = region_avg_hours = region_avg_earn = region_avg_abs = None

    if show_region_avg:
        year_sel = st.session_state.get("year_sel", "All")

        # 1) Prefer DB-driven metric: total_km_per_depot_per_month (if present)
        if mysql_conn is not None:
            try:
                region_avgs_data = get_region_avgs_from_ts_admin(mysql_conn, depot, depot_monthly if depot_monthly is not None else pd.DataFrame(), year_sel)
            except Exception as e:
                region_avgs_data = None
                st.warning(f"Could not fetch region averages from DB: {e}")

            if region_avgs_data and isinstance(region_avgs_data, dict):
                ra = region_avgs_data.get("region_avgs", {})
                # Prefer the explicit per-depot-per-month keys (newer function returns these)
                region_avg_km = ra.get("total_km_per_depot_per_month") or ra.get("avg_km_per_depot_per_month") or ra.get("total_km")
                region_avg_hours = ra.get("total_hours_per_depot_per_month") or ra.get("avg_hours_per_depot_per_month") or ra.get("total_hours")
                region_avg_earn = ra.get("total_earnings_per_depot_per_month") or ra.get("avg_earnings_per_depot_per_month") or ra.get("total_earnings")
                region_avg_abs = ra.get("absenteeism_per_depot_per_month") or ra.get("avg_absenteeism_per_depot_per_month") or ra.get("absenteeism")

                # If DB returned values but they are in different units (e.g. total_km = mean of per-depot totals),
                # we still display them; the preferred keys are the per-depot-per-month ones.
                # if region_avg_km is not None:
                #     st.info(f"✅ Showing DB-driven Region Average (per-depot-per-month if available). Depots included: {len(region_avgs_data.get('region_depots', []))}")
                # else:
                #     # DB returned region_avgs but did not include our preferred keys
                #     st.info("✅ Showing DB-driven Region Average (units may be per-depot over period).")
            else:
                st.info("DB-driven region averages not available; attempting fallback to depot_monthly.")

        # 2) Fallback: compute per-depot-per-month average from depot_monthly if possible
        if region_avg_km is None and depot_monthly is not None and depot_settings is not None:
            try:
                depot_upper = str(depot).strip().upper()
                region_name = depot_settings.get(depot_upper, {}).get("region")
                if region_name:
                    region_depots = [d for d, info in depot_settings.items() if info.get("region") == region_name]
                    region_depots_upper = [str(d).strip().upper() for d in region_depots]
                    region_df = depot_monthly.copy()
                    # normalize depot col name
                    depot_col = None
                    for c in region_df.columns:
                        if c.lower() in ["depot", "depot_name"]:
                            depot_col = c
                            break
                    if depot_col is None:
                        st.warning("Cannot compute region fallback: depot column not found in depot_monthly.")
                    else:
                        region_df[depot_col] = region_df[depot_col].astype(str).str.strip().str.upper()
                        region_df = region_df[region_df[depot_col].isin(region_depots_upper)].copy()
                        if year_sel != "All" and "year" in region_df.columns:
                            try:
                                region_df = region_df[region_df["year"] == int(year_sel)]
                            except Exception:
                                pass
                        # Need at least 2 depots' data to compute a meaningful per-depot-per-month average
                        if not region_df.empty and region_df["depot"].nunique() > 0:
                            # depot_monthly is expected to be one row per depot-month; mean of total_km per depot gives avg km/month per depot
                            per_depot_monthly_avg = region_df.groupby(region_df[depot_col])["total_km"].mean()
                            per_depot_monthly_avg_hours = region_df.groupby(region_df[depot_col])["total_hours"].mean() if "total_hours" in region_df.columns else None
                            per_depot_monthly_avg_earn = region_df.groupby(region_df[depot_col])["total_earnings"].mean() if "total_earnings" in region_df.columns else None
                            per_depot_monthly_avg_abs = region_df.groupby(region_df[depot_col])["absenteeism"].mean() if "absenteeism" in region_df.columns else None

                            if not per_depot_monthly_avg.empty:
                                region_avg_km = float(per_depot_monthly_avg.mean())
                            if per_depot_monthly_avg_hours is not None and not per_depot_monthly_avg_hours.empty:
                                region_avg_hours = float(per_depot_monthly_avg_hours.mean())
                            if per_depot_monthly_avg_earn is not None and not per_depot_monthly_avg_earn.empty:
                                region_avg_earn = float(per_depot_monthly_avg_earn.mean())
                            if per_depot_monthly_avg_abs is not None and not per_depot_monthly_avg_abs.empty:
                                region_avg_abs = float(per_depot_monthly_avg_abs.mean())

                            st.info(f"✅ Showing Region Average computed from available depot_monthly ({region_df[depot_col].nunique()} depots considered).")
                        else:
                            st.warning("Fallback: insufficient region data in depot_monthly to compute region average.")
                else:
                    st.warning(f"Region not found for depot {depot}.")
            except Exception as e:
                st.warning(f"Error computing fallback region averages: {e}")

        # If final region averages are still None, show a note
        if region_avg_km is None:
            st.warning("Region average could not be computed (DB and fallback both failed).")

    # ---------------- Display Charts ----------------
    avg_label = "Region Average" if show_region_avg else "Depot Average"

    # Total KMs
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        plot_bar(
            df=df1,
            ycol="total_km",
            depot_avg=depot_avg_km,
            region_avg=region_avg_km if show_region_avg and region_avg_km is not None else None,
            title=f"{depot} – Total KMs ({avg_label})",
            ytitle="KMs",
            depot=depot
        )

    # Total Hours
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        plot_bar(
            df=df1,
            ycol="total_hours",
            depot_avg=depot_avg_hours,
            region_avg=region_avg_hours if show_region_avg and region_avg_hours is not None else None,
            title=f"{depot} – Total Hours ({avg_label})",
            ytitle="Hours",
            depot=depot
        )

    # Total Earnings
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        plot_bar(
            df=df1,
            ycol="total_earnings",
            depot_avg=depot_avg_earn,
            region_avg=region_avg_earn if show_region_avg and region_avg_earn is not None else None,
            title=f"{depot} – Total Earnings ({avg_label})",
            ytitle="Earnings",
            depot=depot
        )

    # Absenteeism
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        plot_bar(
            df=df1,
            ycol="absenteeism",
            depot_avg=depot_avg_abs,
            region_avg=region_avg_abs if show_region_avg and region_avg_abs is not None else None,
            title=f"{depot} – Absenteeism ({avg_label})",
            ytitle="Absenteeism",
            depot=depot
        )



def show_depot_metrics(df, depot, is_region=False):
    if df is not None and not df.empty:
        avg_kms_month = df['total_km'].mean()
        num_years = len(df['year'].unique())
        years = df['year'].unique()
        total_days = sum(366 if calendar.isleap(int(y)) else 365 for y in years)
        total_days = total_days if total_days > 0 else 1
        avg_kms_day = df['total_km'].mean() * (12 / 365) if num_years > 0 else df['total_km'].mean() / total_days
        avg_hours = df['total_hours'].mean()
        avg_hours_day = df['total_hours'].mean() * (12 / 365) if num_years > 0 else df['total_hours'].mean() / total_days
        avg_earnings = df['total_earnings'].mean()
        avg_earnings_day = df['total_earnings'].mean() * (12 / 365) if num_years > 0 else df['total_earnings'].mean() / total_days   
        avg_absenteeism = df['absenteeism'].mean()
        avg_absenteeism_day = df['absenteeism'].mean() * (12 / 365) if num_years > 0 else df['absenteeism'].mean() / total_days
        label_prefix = "Region" if is_region else depot
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{depot} Avg KMs/Month", format_indian(avg_kms_month))
        with col2:
            st.metric(f"{depot} Avg Hours/Month", format_indian(avg_hours))
        with col3:
            st.metric(f"{depot} Avg Earnings/Month", format_indian(avg_earnings))
        with col4:
            st.metric(f"{label_prefix} Avg Absenteeism/Month", format_indian(avg_absenteeism))
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{depot} Avg KMs/Day", format_indian(avg_kms_day))
        with col2:
            st.metric(f"{depot} Avg Hours/Day", format_indian(avg_hours_day))
        with col3:
            st.metric(f"{depot} Avg Earnings/Day", format_indian(avg_earnings_day))
        with col4:
            st.metric(f"{label_prefix} Avg Absenteeism/Day", format_indian(avg_absenteeism_day))    
    else:
        st.info(f"🚫 No valid data for **{depot.title()}**.")

def get_region_means_from_db(mysql_conn, current_depot, year_sel="All"):
    """
    Compute region means (mean of per-depot totals) from DB for the region of current_depot.

    Returns:
      {
        "region": region_name,
        "region_depots": [...],                 # list of uppercase depot names in region
        "per_depot_totals": DataFrame,          # columns: depot, total_km, total_hours, total_earnings, absenteeism
        "region_means": {                       # mean of per-depot totals
            "total_km_mean": float,
            "total_hours_mean": float,
            "total_earnings_mean": float,
            "absenteeism_mean": float
        }
      }
    Notes:
      - Uses parameterized queries to avoid SQL injection.
      - If year_sel == "All", aggregates over all available years.
      - absenteeism is computed from driver_absenteeism (COUNT per depot).
    """
    try:
        if not current_depot:
            raise ValueError("current_depot required")

        current_depot_upper = str(current_depot).strip().upper()

        # 1) get region for current_depot
        q_region = "SELECT region FROM TS_ADMIN WHERE depot_name = %s LIMIT 1;"
        region_df = pd.read_sql(q_region, mysql_conn, params=(current_depot_upper,))
        if region_df.empty:
            return {"error": f"Region not found for depot {current_depot_upper}"}

        region_name = str(region_df.iloc[0]["region"]).strip()
        if not region_name:
            return {"error": f"Empty region for depot {current_depot_upper}"}

        # 2) get depots in the region
        q_depots = "SELECT depot_name FROM TS_ADMIN WHERE region = %s;"
        depots_df = pd.read_sql(q_depots, mysql_conn, params=(region_name,))
        if depots_df.empty:
            return {"error": f"No depots found for region {region_name}"}

        region_depots = depots_df["depot_name"].astype(str).str.strip().str.upper().tolist()
        if not region_depots:
            return {"error": f"No depot names for region {region_name}"}

        # Build placeholders and params for IN(...)
        placeholders = ",".join(["%s"] * len(region_depots))
        params_ops = tuple(region_depots)
        params_lsa = tuple(region_depots)

        # Optional year filter
        year_filter_ops = ""
        year_filter_lsa = ""
        if year_sel != "All":
            year_filter_ops = " AND YEAR(operations_date) = %s "
            year_filter_lsa = " AND YEAR(`date`) = %s "
            params_ops = params_ops + (int(year_sel),)
            params_lsa = params_lsa + (int(year_sel),)

        # 3) per-depot operational aggregates from daily_operations (sum per depot)
        ops_query = f"""
            SELECT TRIM(UPPER(depot)) AS depot,
                   SUM(COALESCE(opd_kms,0)) AS total_km,
                   
                   SUM(COALESCE(daily_earnings,0)) AS total_earnings
            FROM daily_operations
            WHERE TRIM(UPPER(depot)) IN ({placeholders})
            {year_filter_ops}
            GROUP BY TRIM(UPPER(depot))
        """

        ops_df = pd.read_sql(ops_query, mysql_conn, params=params_ops)

        # 4) per-depot absenteeism counts from driver_absenteeism (count per depot)
        lsa_query = f"""
            SELECT TRIM(UPPER(depot)) AS depot,
                   COUNT(*) AS absenteeism
            FROM driver_absenteeism
            WHERE TRIM(UPPER(depot)) IN ({placeholders})
            {year_filter_lsa}
            GROUP BY TRIM(UPPER(depot))
        """
        lsa_df = pd.read_sql(lsa_query, mysql_conn, params=params_lsa)

        # Normalize/merge
        if ops_df.columns.dtype != "O":
            ops_df.columns = [c.lower() for c in ops_df.columns]
        if lsa_df.columns.dtype != "O":
            lsa_df.columns = [c.lower() for c in lsa_df.columns]

        # ensure lowercase column names
        ops_df.columns = [c.lower() for c in ops_df.columns]
        lsa_df.columns = [c.lower() for c in lsa_df.columns]

        merged = pd.merge(
            ops_df,
            lsa_df,
            on="depot",
            how="left"
        )

        # fill missing absenteeism with 0, ensure numeric
        if "absenteeism" in merged.columns:
            merged["absenteeism"] = pd.to_numeric(merged["absenteeism"], errors="coerce").fillna(0)
        else:
            merged["absenteeism"] = 0

        for col in ["total_km", "total_hours", "total_earnings"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
            else:
                merged[col] = 0

        # 5) compute means across depots (mean of per-depot totals)
        #    this matches your description: sum per-depot then .mean()
        region_means = {
            "total_km_mean": float(merged["total_km"].mean()) if not merged.empty else 0.0,
            "total_hours_mean": float(merged["total_hours"].mean()) if not merged.empty else 0.0,
            "total_earnings_mean": float(merged["total_earnings"].mean()) if not merged.empty else 0.0,
            "absenteeism_mean": float(merged["absenteeism"].mean()) if not merged.empty else 0.0
        }

        return {
            "region": region_name,
            "region_depots": region_depots,
            "per_depot_totals": merged[["depot", "total_km", "total_hours", "total_earnings", "absenteeism"]],
            "region_means": region_means
        }

    except Exception as e:
        # you can st.error here if called from Streamlit
        return {"error": str(e)}
    

def get_region_avgs_from_ts_admin(mysql_conn, current_depot, depot_monthly, year_sel):
    """
    DB-driven: compute region totals then return per-depot-per-month averages.
    Avoids 'int' object has no attribute 'fillna' by ensuring pandas Series are used.
    Returns:
        {
            "region_avgs": {
                "total_km_per_depot_per_month": float,
                "total_hours_per_depot_per_month": float,
                "total_earnings_per_depot_per_month": float,
                "absenteeism_per_depot_per_month": float,
                "region_total_km": float,
                "num_depots": int,
                "months_count": int
            },
            "depot_totals": DataFrame (per-depot totals over the period),
            "region_depots": [list of depot names]
        }
    """
    try:
        if not current_depot:
            st.error("Current depot not provided.")
            return None

        current_depot_upper = str(current_depot).strip().upper()

        # 1) Region for current depot
        query_region = "SELECT region FROM TS_ADMIN WHERE depot_name = %s LIMIT 1"
        region_df = pd.read_sql(query_region, mysql_conn, params=(current_depot,))
        if region_df.empty or 'region' not in region_df.columns:
            st.warning(f"No region found for depot {current_depot} in TS_ADMIN.")
            return None
        region_name = region_df.iloc[0]['region']

        # 2) Depots in region
        query_depots = "SELECT depot_name FROM TS_ADMIN WHERE region = %s"
        depots_df = pd.read_sql(query_depots, mysql_conn, params=(region_name,))
        if depots_df.empty or 'depot_name' not in depots_df.columns:
            st.warning(f"No depots found in region {region_name}.")
            return None
        region_depots = depots_df['depot_name'].astype(str).str.strip().str.upper().tolist()
        if not region_depots:
            st.warning(f"No depot names for region {region_name}.")
            return None
        num_depots = len(region_depots)

        placeholders = ",".join(["%s"] * num_depots)
        params_ops = tuple(region_depots)
        params_lsa = tuple(region_depots)

        year_filter_ops = ""
        year_filter_lsa = ""
        if year_sel != "All":
            year_filter_ops = " AND YEAR(operations_date) = %s "
            year_filter_lsa = " AND YEAR(`date`) = %s "
            params_ops = params_ops + (int(year_sel),)
            params_lsa = params_lsa + (int(year_sel),)

        # 3) Region sums
        ops_sum_query = f"""
            SELECT
                SUM(COALESCE(opd_kms,0)) AS region_total_km,
                
                SUM(COALESCE(daily_earnings,0)) AS region_total_earnings
            FROM daily_operations
            WHERE TRIM(UPPER(depot)) IN ({placeholders})
            {year_filter_ops}
        """
        lsa_sum_query = f"""
            SELECT COUNT(*) AS region_total_absenteeism
            FROM driver_absenteeism
            WHERE TRIM(UPPER(depot)) IN ({placeholders})
            {year_filter_lsa}
        """
        months_query = f"""
            SELECT COUNT(DISTINCT DATE_FORMAT(operations_date, '%%Y-%%m')) AS months_present
            FROM daily_operations
            WHERE TRIM(UPPER(depot)) IN ({placeholders})
            {year_filter_ops}
        """

        # Safe reads and scalar extraction
        try:
            ops_sum_df = pd.read_sql(ops_sum_query, mysql_conn, params=params_ops)
        except Exception as e:
            st.error(f"Error fetching region ops sums for {region_name}: {e}")
            ops_sum_df = pd.DataFrame(columns=['region_total_km','region_total_hours','region_total_earnings'])

        try:
            lsa_sum_df = pd.read_sql(lsa_sum_query, mysql_conn, params=params_lsa)
        except Exception as e:
            st.error(f"Error fetching region absenteeism sum for {region_name}: {e}")
            lsa_sum_df = pd.DataFrame(columns=['region_total_absenteeism'])

        try:
            months_df = pd.read_sql(months_query, mysql_conn, params=params_ops)
        except Exception as e:
            st.error(f"Error fetching months count for {region_name}: {e}")
            months_df = pd.DataFrame(columns=['months_present'])

        # Extract scalars safely (check both emptiness and column existence)
        def safe_scalar(df, col, cast=float, default=0):
            if df is None or df.empty or col not in df.columns:
                return default
            try:
                val = df.iloc[0][col]
                if pd.isna(val):
                    return default
                return cast(val)
            except Exception:
                return default

        region_total_km = safe_scalar(ops_sum_df, 'region_total_km', float, 0.0)
        region_total_hours = safe_scalar(ops_sum_df, 'region_total_hours', float, 0.0)
        region_total_earnings = safe_scalar(ops_sum_df, 'region_total_earnings', float, 0.0)
        region_total_abs = safe_scalar(lsa_sum_df, 'region_total_absenteeism', int, 0)
        months_present = safe_scalar(months_df, 'months_present', int, 0)

        # avoid division by zero
        if months_present <= 0:
            # if specific year requested but no months found, fallback to 1 to avoid crash
            months_present = 1

        # 4) per-depot-per-month calculation (as requested by you)
        km_per_depot_per_month = region_total_km / num_depots / months_present
        hours_per_depot_per_month = region_total_hours / num_depots / months_present
        earnings_per_depot_per_month = region_total_earnings / num_depots / months_present
        absenteeism_per_depot_per_month = region_total_abs / num_depots / months_present

        # 5) Also return per-depot totals (DB side)
        ops_depot_query = f"""
            SELECT TRIM(UPPER(depot)) AS depot,
                   SUM(COALESCE(opd_kms,0)) AS total_km,
                   
                   SUM(COALESCE(daily_earnings,0)) AS total_earnings
            FROM daily_operations
            WHERE TRIM(UPPER(depot)) IN ({placeholders})
            {year_filter_ops}
            GROUP BY TRIM(UPPER(depot))
        """
        lsa_depot_query = f"""
            SELECT TRIM(UPPER(depot)) AS depot,
                   COUNT(*) AS absenteeism
            FROM driver_absenteeism
            WHERE TRIM(UPPER(depot)) IN ({placeholders})
            {year_filter_lsa}
            GROUP BY TRIM(UPPER(depot))
        """

        try:
            ops_df = pd.read_sql(ops_depot_query, mysql_conn, params=params_ops)
        except Exception as e:
            st.error(f"Error fetching per-depot ops for {region_name}: {e}")
            ops_df = pd.DataFrame(columns=['depot','total_km','total_hours','total_earnings'])

        try:
            lsa_df = pd.read_sql(lsa_depot_query, mysql_conn, params=params_lsa)
        except Exception as e:
            st.error(f"Error fetching per-depot absenteeism for {region_name}: {e}")
            lsa_df = pd.DataFrame(columns=['depot','absenteeism'])

        # Normalize columns to lowercase
        for df in [ops_df, lsa_df]:
            if not df.empty:
                df.columns = [c.lower() for c in df.columns]

        if ops_df.empty:
            # produce empty depot_totals with proper columns
            depot_totals = pd.DataFrame(columns=['depot','total_km','total_hours','total_earnings','absenteeism'])
        else:
            # merge and ensure Series exist before using fillna
            if 'depot' in ops_df.columns:
                ops_df['depot'] = ops_df['depot'].astype(str).str.strip().str.upper()
            if not lsa_df.empty and 'depot' in lsa_df.columns:
                lsa_df['depot'] = lsa_df['depot'].astype(str).str.strip().str.upper()

            merged = ops_df.merge(lsa_df, on='depot', how='left')

            # Ensure columns exist as Series and are numeric
            if 'absenteeism' not in merged.columns:
                merged['absenteeism'] = 0
            merged['absenteeism'] = pd.to_numeric(merged['absenteeism'], errors='coerce').fillna(0)

            for col in ['total_km','total_hours','total_earnings']:
                if col not in merged.columns:
                    merged[col] = 0
                merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)

            merged['depot'] = merged['depot'].astype(str).str.strip().str.upper()
            depot_totals = merged[['depot','total_km','total_hours','total_earnings','absenteeism']].copy()

        region_avgs = {
            "total_km_per_depot_per_month": float(km_per_depot_per_month),
            "total_hours_per_depot_per_month": float(hours_per_depot_per_month),
            "total_earnings_per_depot_per_month": float(earnings_per_depot_per_month),
            "absenteeism_per_depot_per_month": float(absenteeism_per_depot_per_month),
            "region_total_km": float(region_total_km),
            "num_depots": int(num_depots),
            "months_count": int(months_present)
        }

        st.info(f"Region: {region_name} | Depots considered: {num_depots} ")

        return {
            "region_avgs": region_avgs,
            "depot_totals": depot_totals,
            "region_depots": region_depots
        }

    except Exception as e:
        st.error(f"Error computing region averages: {e}")
        return None




#--helpers-------------------------------------------------
def get_region_depots(mysql_conn, region_name):
    """
    Return list of depot names for a region (uppercase, trimmed).
    """
    query = "SELECT depot_name FROM TS_ADMIN WHERE region = %s"
    try:
        df = pd.read_sql(query, mysql_conn, params=(region_name,))
    except Exception as e:
        st.error(f"Error fetching region depots: {e}")
        return []
    if df.empty:
        return []
    return df['depot_name'].astype(str).str.strip().str.upper().tolist()
#--main-------------------------------------------------
def depot_DM():
    # Column literals
    DEPOT = "depot"
    EMP_ID = "employee_id"

    st.title("TGSRTC – Depot Productivity Dashboard")

    # ------------------- DB Connection -------------------
    mysql_conn = get_connection()
    if not mysql_conn:
        st.stop()

    # ------------------- Get Current User Depot -------------------
    selected_depot = str(get_current_user_depot()).strip().upper()
    depot1 = selected_depot  # Display name

    # Handle Auth/Fetch Errors 
    if not depot1 or "ERROR" in depot1 or "UNKNOWN" in depot1 or "NOT_MAPPED" in depot1:
        st.error(f"🚨 Access Denied/Error: Failed to retrieve or map depot for user ID '{st.session_state.get('userid', 'unknown')}'. Result: {depot1}")
        if mysql_conn:
            mysql_conn.close()
        st.stop()

    # ------------------- Fetch Depot Settings -------------------
    depot_settings = get_depot_settings(mysql_conn)
    current_depot_info = depot_settings.get(depot1, {'region': 'Unknown', 'category': 'Unknown'})
    depot_region = current_depot_info.get('region', 'Unknown')
    depot_category = current_depot_info.get('category', 'Unknown')

    # Display depot and category
    st.markdown(f"<span style='font-size: 1.5em;'><b>{depot1}</b></span> ({depot_category})", unsafe_allow_html=True)

    # ------------------- Fetch Data for selected depot -------------------
    depot_monthly, driver_monthly, lsa_valid, ghc_2024 = preaggregate(selected_depot, depot_settings)

    # ------------------- Normalize Dataframes -------------------
    if depot_monthly is None or depot_monthly.empty:
        depot_monthly = pd.DataFrame(columns=[DEPOT, "year", "month", "total_km", "total_hours", "total_earnings", "absenteeism", "region", "category"])
    else:
        # ensure region & category columns exist (use depot_settings mapping)
        if "region" not in depot_monthly.columns:
            depot_monthly["region"] = depot_monthly[DEPOT].map(lambda d: depot_settings.get(str(d).strip().upper(), {}).get("region", "Unknown"))
        if "category" not in depot_monthly.columns:
            depot_monthly["category"] = depot_monthly[DEPOT].map(lambda d: depot_settings.get(str(d).strip().upper(), {}).get("category", "Unknown"))
        # normalize depot column
        depot_monthly[DEPOT] = depot_monthly[DEPOT].astype(str).str.strip().str.upper()

    if driver_monthly is None or driver_monthly.empty:
        driver_monthly = pd.DataFrame(columns=[DEPOT, EMP_ID, "year", "month", "total_km", "total_hours", "total_earnings", "absenteeism"])
    if lsa_valid is None or lsa_valid.empty:
        lsa_valid = pd.DataFrame()
    if ghc_2024 is None or ghc_2024.empty:
        ghc_2024 = pd.DataFrame()

    # ------------------- Filter by Depot (for per-depot displays) -------------------
    filtered_depot_monthly = depot_monthly[depot_monthly[DEPOT] == depot1].copy()

    # Determine available years for this depot (used by many UI controls)
    if not filtered_depot_monthly.empty:
        filtered_depot_monthly["year"] = filtered_depot_monthly["year"].astype(int)
        years = sorted(filtered_depot_monthly["year"].dropna().unique())
        year_options = ["All"] + [str(y) for y in years]
        default_index = year_options.index(str(max(years))) if years else 0
    else:
        years = []
        year_options = ["All"]
        default_index = 0
        st.warning(f"No data available for depot {depot1}. Depot charts & metrics will be empty.")

    # ------------------- Year Filter (applies to per-depot & region queries) -------------------
    year_sel = st.selectbox("Year", year_options, index=default_index, key="year_sel")
    if year_sel != "All":
        try:
            year_int = int(year_sel)
            if not filtered_depot_monthly.empty:
                filtered_depot_monthly = filtered_depot_monthly[filtered_depot_monthly["year"] == year_int]
            if not driver_monthly.empty and "year" in driver_monthly.columns:
                driver_monthly = driver_monthly[driver_monthly["year"] == year_int]
            if not lsa_valid.empty and "year" in lsa_valid.columns:
                lsa_valid = lsa_valid[lsa_valid["year"] == year_int]
            if not ghc_2024.empty and "year" in ghc_2024.columns:
                ghc_2024 = ghc_2024[ghc_2024["year"] == year_int]
        except Exception:
            st.warning("Year filter couldn't be applied cleanly; continuing without strict year filter.")

 

    # ------------------- Show Depot Metrics (single depot) -------------------
    if not filtered_depot_monthly.empty:
        st.markdown(f"### Depot Selected: **{depot1}** ({depot_category})")
        show_depot_metrics(filtered_depot_monthly, depot1)
    else:
        st.info(f"No monthly data available for depot {depot1} for the selected filters.")

    # ------------------- Per-depot charts -------------------
    # These show depot-only charts with optional region-average lines handled by show_main_bar_line_charts()
    show_main_bar_line_charts(filtered_depot_monthly, depot1, depot_monthly, depot_settings, mysql_conn=mysql_conn)
    # ----------------- Other Visualizations -------------------
    show_absenteeism_pie(depot1, lsa_valid, year_sel, is_region=False, region_depots=None)
    show_health_grade_distribution(depot1, ghc_2024, is_region=False, region_depots=None)
    show_top_bottom_drivers(depot1, driver_monthly, years, DEPOT, EMP_ID, is_region=False, region_depots=None)
    show_mu_sl_reasons(depot1, config['db'], is_region=False, region_depots=None)

    # ------------------- Close Connection -------------------
    if mysql_conn:
        mysql_conn.close()

# ---------------------- 4. MISC VISUALIZATION FUNCTIONS ----------------------

def show_absenteeism_pie(entity_name, lsa_df, year_sel="All", is_region=False, region_depots=None):
    # Column literals for local function use
    DEPOT = "depot"
    
    st.markdown("## 5. Absenteeism Reasons Distribution")
    if lsa_df is None or lsa_df.empty:
        st.info(f"🚫 No leave data found for **{entity_name}**.")
        return

    lsa_df[DEPOT] = lsa_df[DEPOT].astype(str).str.strip().str.upper()
    if is_region and region_depots:
        depots_upper = [d.upper() for d in region_depots]
        lsa_df = lsa_df[lsa_df[DEPOT].isin(depots_upper)]
    else:
        lsa_df = lsa_df[lsa_df[DEPOT] == entity_name.strip().upper()]

    if lsa_df.empty:
        st.info(f"🚫 No records for **{entity_name}** after filtering.")
        return

    lsa_df['date'] = pd.to_datetime(lsa_df['date'], errors="coerce")
    lsa_df['year'] = lsa_df['date'].dt.year
    lsa_valid = lsa_df.dropna(subset=['leave_type']).copy()
    
    if year_sel != "All":
        lsa_valid = lsa_valid[lsa_valid["year"] == int(year_sel)]

    if lsa_valid.empty:
        st.info(f"🚫 No leave data for **{entity_name}** in year {year_sel}.")
        return

    leave_counts = (
        lsa_valid['leave_type']
        .value_counts()
        .reset_index(name="Count")
        .rename(columns={"index": "leave_type"})
    )
    fig = px.pie(
        leave_counts,
        names="leave_type",
        values="Count",
        title=f"<b>Leave Type Distribution – {entity_name}</b>",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def show_health_grade_distribution(entity_name, ghc_2024, is_region=False, region_depots=None):
    st.markdown("## 6. Employee Health Grade Distribution")
    
    # FIX: Convert columns to string first to avoid AttributeError on Linux
    ghc_2024.columns = ghc_2024.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")

    if "final_grading" not in ghc_2024.columns or "depot" not in ghc_2024.columns:
        st.error("❌ Required health check columns ('final_grading' or 'depot') not found.")
        return

    ghc_2024["depot"] = ghc_2024["depot"].astype(str).str.strip().str.upper()

    if is_region and region_depots:
        depots_upper = [d.upper() for d in region_depots]
        ghc3_filtered = ghc_2024[ghc_2024["depot"].isin(depots_upper)].copy()
    else:
        ghc3_filtered = ghc_2024[ghc_2024["depot"] == entity_name.strip().upper()].copy()

    if ghc3_filtered.empty:
        st.warning(f"⚠️ No health grade data found for **{entity_name}**.")
        return

    ghc3_filtered.rename(columns={"final_grading": "grade"}, inplace=True)
    ghc3_filtered = ghc3_filtered[ghc3_filtered["grade"].notna()]
    ghc3_filtered["grade"] = ghc3_filtered["grade"].astype(str).str.strip()

    grade_counts = ghc3_filtered.groupby("grade").size().reset_index(name="num_employees")
    if grade_counts.empty:
        st.info(f"ℹ️ No Health Grade data available for {entity_name}.")
        return

    fig = px.bar(
        grade_counts.sort_values("grade"),
        x="grade",
        y="num_employees",
        labels={"grade": "Health Grade", "num_employees": "Number of Employees"},
        color_discrete_sequence=["#1172a5"],
        text_auto=True,
    )
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    fig.update_layout(
        title={'text': f"<b>Employee Health Grades – {entity_name}</b>", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title="Health Grade",
        yaxis_title="Number of Employees",
        plot_bgcolor="white",
        bargap=0.3, showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, zeroline=True),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.plotly_chart(fig, use_container_width=True)

#TOP & BOTTOM DRIVERS
def show_top_bottom_drivers(entity_name, driver_monthly, years, DEPOT_COL, EMP_COL, is_region=False, region_depots=None):
    """
    Show Top & Bottom 5 drivers for either a depot or all depots in a region.
    """
    st.markdown("## Top & Bottom 5 Drivers")

    # Filter by depot or region
    if is_region and region_depots:
        depot_keys = [d.strip().lower() for d in region_depots]
        df = driver_monthly[driver_monthly[DEPOT_COL].str.strip().str.lower().isin(depot_keys)].copy()
    else:
        df = driver_monthly[driver_monthly[DEPOT_COL].str.strip().str.lower() == entity_name.strip().lower()].copy()

    if df.empty:
        st.info(f"🚫 No driver data for **{entity_name}**.")
        return

    df["month_label"] = df["month"].dt.strftime("%b-%y")

    # ---------------- Filters ----------------
    col1, col2, col3 = st.columns(3)

    # ✅ Year options with latest year selected by default
    year_list = sorted(df["year"].dropna().unique())
    year_options = ["All"] + [str(y) for y in year_list]
    default_year_index = year_options.index(str(max(year_list))) if len(year_list) > 0 else 0

    top_year = col1.selectbox(
        "Year",
        year_options,
        index=default_year_index,
        key="top_driver_year"
    )

    # ✅ Month options with latest month selected by default
    if top_year == "All":
        month_options = sorted(df["month_label"].unique(), key=lambda x: pd.to_datetime(x, format="%b-%y"))
    else:
        month_options = sorted(
            df[df["year"] == int(top_year)]["month_label"].unique(),
            key=lambda x: pd.to_datetime(x, format="%b-%y")
        )

    month_options_display = ["All"] + month_options
    default_month_index = month_options_display.index(month_options[-1]) if month_options else 0

    top_month = col2.selectbox(
        "Month",
        month_options_display,
        index=default_month_index,
        key="top_driver_month"
    )

    # ✅ Add "Earnings" to rank options
    rank_by = col3.selectbox("Rank by", ["KMs", "Hours", "Earnings", "Absenteeism"])

    # ---------------- Apply filters ----------------
    if top_year != "All":
        df = df[df["year"] == int(top_year)]
    if top_month != "All":
        df = df[df["month_label"] == top_month]

    if df.empty:
        st.info("🚫 No driver data for selected filters.")
        return

    # ✅ Map rank column correctly
    rank_col_map = {
        "KMs": "total_km",
        "Hours": "total_hours",
        "Earnings": "total_earnings",
        "Absenteeism": "absenteeism",
    }
    rank_col = rank_col_map[rank_by]

    # ---------------- Summarize ----------------
    summary = df.groupby(EMP_COL).agg(
        total_km=('total_km', 'sum'),
        total_hours=('total_hours', 'sum'),
        total_earnings=('total_earnings', 'sum'),
        absenteeism=('absenteeism', 'sum')
    ).reset_index()

    if summary.empty:
        st.info("🚫 No data after grouping.")
        return

    # ✅ Top and bottom logic (Absenteeism is reversed)
    if rank_by == "Absenteeism":
        top5 = summary.sort_values(rank_col, ascending=True).head(5)
        bottom5 = summary.sort_values(rank_col, ascending=False).head(5)
    else:
        top5 = summary.sort_values(rank_col, ascending=False).head(5)
        bottom5 = summary.sort_values(rank_col, ascending=True).head(5)

    top5.insert(0, "Rank", range(1, len(top5) + 1))
    bottom5.insert(0, "Rank", range(1, len(bottom5) + 1))
    top5 = top5.set_index("Rank")
    bottom5 = bottom5.set_index("Rank")

    # ---------------- Display ----------------
    colA, colB = st.columns(2)
    colA.subheader(f"7. Top 5 by {rank_by} ({'Region' if is_region else 'Depot'})")
    colA.dataframe(top5)
    colB.subheader(f"8. Bottom 5 by {rank_by} ({'Region' if is_region else 'Depot'})")
    colB.dataframe(bottom5)

#MU & SL REASONS
def show_mu_sl_reasons(entity_name, db_config, is_region=False, region_depots=None):
    st.markdown("## 9. Medical Unfit (MU) & Sick Leave (SL) Reasons")
    db_user = config['db']['user']
    db_password = quote_plus(config['db']['password'])
    db_host = config['db']['host']
    db_name = config['db']['database']

    engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    try:
        df = pd.read_sql("SELECT * FROM input_data", con=engine)
    except Exception as e:
        st.error(f"Error fetching input_data for MU/SL reasons: {e}")
        return
        
    df.columns = df.columns.astype(str).str.strip().str.replace(" ", "_").str.lower()
    
    if "depot_name" not in df.columns or "data_date" not in df.columns:
        st.error("❌ Required columns ('depot_name' or 'data_date') missing in input_data.")
        return

    df["depot_name"] = df["depot_name"].astype(str).str.strip().str.lower()

    if is_region and region_depots:
        region_keys = [d.strip().lower() for d in region_depots]
        df = df[df["depot_name"].isin(region_keys)]
    else:
        df = df[df["depot_name"] == entity_name.strip().lower()]

    if df.empty:
        st.info(f"🚫 No MU/SL data found for **{entity_name.title()}**.")
        return

    df["data_date"] = pd.to_datetime(df["data_date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    df = df[df["data_date"].notna() & (df["data_date"] <= today)]
    df["year"] = df["data_date"].dt.year
    df["month_period"] = df["data_date"].dt.to_period("M")

    col1, col2 = st.columns(2)

    years = sorted(df["year"].dropna().unique())
    year_options = ["All"] + [str(y) for y in years]
    default_year_index = year_options.index(str(max(years))) if len(years) > 0 else 0

    selected_year = col1.selectbox("Year", year_options, index=default_year_index, key="year_selectbox")
    filtered_df = df if selected_year == "All" else df[df["year"] == int(selected_year)]

    month_period_unique = filtered_df["month_period"].dropna().unique()
    month_display = [m.strftime("%b-%Y") for m in sorted(month_period_unique)]

    if month_display:
        default_month = month_display[-1]
        month_options = ["All"] + month_display
        default_month_index = month_options.index(default_month)
        selected_month = col2.selectbox("Month", month_options, index=default_month_index, key="month_selectbox")
    else:
        selected_month = "All"

    if selected_month != "All":
        try:
            selected_period = pd.Period(pd.to_datetime(selected_month, format="%b-%Y"), freq="M")
            filtered_df = filtered_df[filtered_df["month_period"] == selected_period]
        except ValueError:
            st.warning("Invalid month selected for filtering.")


    if filtered_df.empty:
        st.info("🚫 No MU/SL data available for the selected filters.")
        return


    mu_cols = [
        'spondilitis', 'spinal_disc', 'vision_color_blindness',
        'neuro_paralysis_medical', 'ortho', 'diff_mu_reasons', 'total_drivers_mu_reasons'
    ]
    sl_cols = [
        'flu_fever', 'bp', 'orthopedic', 'heart', 'weakness', 'eye',
        'accident_injuries', 'neuro_paralysis_sick_leave', 'piles', 'diabetes',
        'thyroid', 'gas', 'dental', 'ear', 'skin_allergy', 'general_surgery',
        'obesity', 'cancer', 'total_drivers_sl_reasons', 'diff_sl_reasons'
    ]
    
    mu_cols_present = [c for c in mu_cols if c in filtered_df.columns]
    sl_cols_present = [c for c in sl_cols if c in filtered_df.columns]

    if not mu_cols_present and not sl_cols_present:
        st.error("❌ No MU/SL reason columns found in the filtered data.")
        return

    mu_totals = filtered_df[mu_cols_present].sum()
    sl_totals = filtered_df[sl_cols_present].sum()

    mu_df = mu_totals.drop(['diff_mu_reasons', 'total_drivers_mu_reasons'], errors="ignore").reset_index()
    mu_df.columns = ['Reason', 'Count']
    mu_df = mu_df[mu_df['Count'] > 0]
    mu_df['Reason'] = mu_df['Reason'].str.replace('_', ' ').str.title()

    sl_df = sl_totals.drop(['diff_sl_reasons', 'total_drivers_sl_reasons'], errors="ignore").reset_index()
    sl_df.columns = ['Reason', 'Count']
    sl_df = sl_df[sl_df['Count'] > 0]
    sl_df['Reason'] = sl_df['Reason'].str.replace('_', ' ').str.title()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total MU Drivers", int(mu_totals.get('total_drivers_mu_reasons', 0)))
    c2.metric("MU Reasons Count", len(mu_df))
    c3.metric("Total SL Drivers", int(sl_totals.get('total_drivers_sl_reasons', 0)))
    c4.metric("SL Reasons Count", len(sl_df))

    col_mu, col_sl = st.columns(2)
    with col_mu:
        if not mu_df.empty:
            fig_mu = px.bar(mu_df.sort_values('Count', ascending=False),
                            x='Reason', y='Count',
                            color_discrete_sequence=['#1172a5'],
                            text='Count',
                            title=f"Medical Unfit (MU) Reasons – {entity_name.title()}")
            fig_mu.update_traces(texttemplate='%{text}', textposition='outside')
            fig_mu.update_layout(xaxis_title='Reason', yaxis_title='Count')
            st.plotly_chart(fig_mu, use_container_width=True)

    with col_sl:
        if not sl_df.empty:
            fig_sl = px.bar(sl_df.sort_values('Count', ascending=False),
                            x='Reason', y='Count',
                            color_discrete_sequence=['#1172a5'],
                            text='Count',
                            title=f"Sick Leave (SL) Reasons – {entity_name.title()}")
            fig_sl.update_traces(texttemplate='%{text}', textposition='outside')
            fig_sl.update_layout(xaxis_title='Reason', yaxis_title='Count')
            st.plotly_chart(fig_sl, use_container_width=True)


if __name__ == "__main__":
    depot_DM()
