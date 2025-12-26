import streamlit as st
import pandas as pd
import altair as alt
import mysql.connector
from mysql.connector import Error
import json
import unicodedata
import numpy as np
import calendar
from datetime import datetime, timedelta
import io
import openpyxl
from openpyxl.drawing.image import Image
import re

# --- Load config.json ---
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    st.error("Configuration file 'config.json' not found.")
    st.stop()
  
DB_CONFIG = config["db"]
MYSQL_TABLE_NAME = "input_data"


# --- DB Connection ---
def get_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        st.error(f"Error connecting to DB: {e}")
        return None


# --- Load Data ---
def load_all_data(_conn):
    try:
        query = f"""
            SELECT d.*, a.category, a.region
            FROM {MYSQL_TABLE_NAME} d
            JOIN TS_ADMIN a ON d.depot_name = a.depot_name
        """
        df = pd.read_sql(query, _conn)
        
        # Normalize column names and types
        df.rename(columns={"data_date": "Date", "depot_name": "Depot"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # --- NEW LOGIC: Driver/Schedule Ratio ---
        if "Total_Drivers" in df.columns and "Planned_Schedules" in df.columns:
            df["Total_Drivers"] = pd.to_numeric(df["Total_Drivers"], errors='coerce').fillna(0)
            df["Planned_Schedules"] = pd.to_numeric(df["Planned_Schedules"], errors='coerce').fillna(0)

            df["Driver_Schedule"] = df.apply(
                lambda x: round(x["Total_Drivers"] / x["Planned_Schedules"], 2) 
                if x["Planned_Schedules"] > 0 else 0, 
                axis=1
            )
        # ----------------------------------------

        def normalize_text(x):
            if pd.isna(x): return ""
            return unicodedata.normalize("NFKD", str(x)).strip()

        if "category" in df.columns:
            df["category"] = df["category"].apply(normalize_text)
        else:
            df["category"] = ""

        df["category_lower"] = df["category"].str.lower()

        def map_category(s):
            if "rural" in s: return "Rural"
            elif "urban" in s: return "Urban"
            elif s in ["r", "u"]: return "Rural" if s == "r" else "Urban"
            return "Unknown"

        df["category"] = df["category_lower"].apply(map_category)

        if "region" in df.columns:
            df["region"] = df["region"].astype(str).str.strip()
        else:
            df["region"] = ""

        df["Depot"] = df["Depot"].astype(str).str.strip()
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


# --- Helper: convert month+year to start/end timestamps ---
def month_year_to_range(from_month, from_year, to_month, to_year):
    start = pd.Timestamp(year=int(from_year), month=int(from_month), day=1)
    last_day = calendar.monthrange(int(to_year), int(to_month))[1]
    end = pd.Timestamp(year=int(to_year), month=int(to_month), day=last_day, hour=23, minute=59, second=59)
    return start, end

# --- Helper: Clean sheet name ---
def clean_sheet_name(name):
    if not name: return "Sheet"
    cleaned_name = re.sub(r'[\[\]\*\/\\:\?\']', '', name)
    return cleaned_name[:31]


# --- Main Dashboard ---
def eight_ratios_dashboard():
    conn = get_connection()
    if not conn: st.stop()

    df = load_all_data(conn)
    if df.empty:
        st.warning("No data found in database.")
        conn.close()
        st.stop()

    # --- Ratio Headings ---
    RATIO_HEADINGS = {
        'Pct_Weekly_Off_National_Off': 'Weekly Off + National Off %',
        'Pct_Others': 'Others + OD %',
        'Pct_Sick_Leave': 'Sick Leave %',
        'Pct_Spot_Absent': 'Spot Absent %',
        'Pct_Off_Cancellation': 'Off Cancellation %',
        'Pct_Special_Off_Night_Out_IC_Online': 'Special Off/Night Out/IC Online %',
        'Pct_Double_Duty': 'Double Duty %',
        'Pct_Leave_Absent': 'Leave Absent %',
        'Driver_Schedule': 'Driver Schedule' 
    }

    # --- Benchmarks ---
    benchmarks = {
        'Urban': {
            'Pct_Weekly_Off_National_Off': 14,
            'Pct_Special_Off_Night_Out_IC_Online': 27.4,
            'Pct_Others': 1,
            'Pct_Leave_Absent': 6,
            'Pct_Sick_Leave': 2,
            'Pct_Spot_Absent': 2,
            'Pct_Double_Duty': 8,
            'Pct_Off_Cancellation': 2,
            'Driver_Schedule': 2.43 
        },
        'Rural': {
            'Pct_Weekly_Off_National_Off': 14,
            'Pct_Special_Off_Night_Out_IC_Online': 25,
            'Pct_Others': 1.7,
            'Pct_Leave_Absent': 2,
            'Pct_Sick_Leave': 2,
            'Pct_Spot_Absent': 1,
            'Pct_Double_Duty': 16,
            'Pct_Off_Cancellation': 2,
            'Driver_Schedule': 2.18 
        }
    }

    # Title
    st.markdown("<h1 style='text-align: center; color: white; font-size: 50px;background-color: #19bc9c; border-radius: 12px;{* padding:0px;margin:0px}'>Productivity Budget Ratios vs Actual 8 Ratios Dashboard</h1>",
                unsafe_allow_html=True)
    st.markdown("---")

    # View selection
    view_option = st.radio(
        "Select View Type:",
        ["All Depots Combined", "Region Wise", "Rural Depots", "Urban Depots", "Individual Depots"],
        horizontal=True
    )

    filtered_df = df.copy()
    title = "All Depots Combined"

    # --- Dynamic Widget Containers ---
    control_cols = st.columns(2)
    
    with control_cols[0]:
        if view_option == "Region Wise":
            all_regions = sorted(df["region"].dropna().unique().tolist())
            selected_region = None
            if all_regions:
                selected_region = st.selectbox("Select Region", all_regions)
                filtered_df = df[df["region"] == selected_region]
                title = f"Region: {selected_region}"
            else:
                st.warning("No region data available.")
        elif view_option == "Rural Depots":
            filtered_df = df[df["category"] == "Rural"]
            title = "Rural Depots"
        elif view_option == "Urban Depots":
            filtered_df = df[df["category"] == "Urban"]
            title = "Urban Depots"
        elif view_option == "Individual Depots":
            all_depots = sorted(df["Depot"].unique().tolist())
            selected_depot = None
            if all_depots:
                selected_depot = st.selectbox("Select Depot", all_depots)
                filtered_df = df[df["Depot"] == selected_depot]
                title = f"Depot: {selected_depot}"
            else:
                st.warning("No depot data available.")
    
    with control_cols[1]:
        freq_option = st.selectbox("Select Frequency", ["Daily", "Monthly", "Yearly"])

    # Determine global min/max
    overall_min = filtered_df["Date"].min()
    overall_max = filtered_df["Date"].max()
    
    if filtered_df.empty or pd.isna(overall_min) or pd.isna(overall_max):
        if view_option in ["Region Wise", "Individual Depots"]:
             st.warning("Selected view has no date data or is empty.")
             conn.close()
             st.stop()

    if filtered_df.empty:
        st.warning("No data available for the selected view.")
        conn.close()
        st.stop()

    # --- Axis Config ---
    time_period = freq_option
    x_axis_format = '%Y-%m-%d'
    label_angle = 0
    x_axis_title = 'Date'
    tick_count = 'day'
    tooltip_date_format = '%Y-%m-%d'

    if time_period == "Daily":
        x_axis_format = '%a %Y-%m-%d'
        label_angle = -90
        tooltip_date_format = '%A %Y-%m-%d'
        tick_count = 'day'
    elif time_period == "Monthly":
        x_axis_format = '%b %Y'
        label_angle = -45
        tooltip_date_format = '%B %Y'
        x_axis_title = 'Month'
        tick_count = 'month'
    elif time_period == "Yearly":
        x_axis_format = '%Y'
        label_angle = 0
        tooltip_date_format = '%Y'
        x_axis_title = 'Year'
        tick_count = 'year'
    
    date_format = x_axis_format
    tooltip_format = tooltip_date_format

    # --- Date Filters ---
    if freq_option == "Daily":
        overall_min_date = overall_min.date()
        overall_max_date = overall_max.date()
        default_end = overall_max.normalize()
        calculated_start = (default_end - pd.Timedelta(days=30)).normalize()
        default_start = max(calculated_start, overall_min.normalize())

        date_cols = st.columns(2)
        with date_cols[0]:
            start_date = st.date_input("From Date", value=default_start.date(),
                                       min_value=overall_min_date, max_value=overall_max_date)
        with date_cols[1]:
            end_date = st.date_input("To Date", value=default_end.date(),
                                     min_value=overall_min_date, max_value=overall_max_date)
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    elif freq_option == "Monthly":
        months = list(calendar.month_name)[1:]
        years = sorted(filtered_df["Date"].dt.year.unique().tolist())
        if not years: years = [datetime.now().year]

        end_default = overall_max
        calculated_start = end_default - pd.DateOffset(months=11)
        start_default = max(calculated_start, overall_min)

        default_from_month = start_default.month
        default_from_year = start_default.year
        default_to_month = end_default.month
        default_to_year = end_default.year

        month_cols = st.columns([1, 1, 1, 1])
        with month_cols[0]:
            from_month = st.selectbox("From Month", options=months, index=default_from_month - 1)
        with month_cols[1]:
            from_year = st.selectbox("From Year", options=years, index=years.index(default_from_year))
        with month_cols[2]:
            to_month = st.selectbox("To Month", options=months, index=default_to_month - 1)
        with month_cols[3]:
            to_year = st.selectbox("To Year", options=years, index=years.index(default_to_year))

        fm = months.index(from_month) + 1
        tm = months.index(to_month) + 1
        start_ts, end_ts = month_year_to_range(fm, int(from_year), tm, int(to_year))

    else:  # Yearly
        years = sorted(filtered_df["Date"].dt.year.unique().tolist())
        if not years: years = [datetime.now().year]
        end_year_default = overall_max.year
        start_year_default = max(overall_min.year, end_year_default - 4)

        year_cols = st.columns([1, 1])
        with year_cols[0]:
            from_year = st.selectbox("From Year", options=years, index=years.index(start_year_default))
        with year_cols[1]:
            to_year = st.selectbox("To Year", options=years, index=years.index(end_year_default))
        start_ts = pd.Timestamp(int(from_year), 1, 1)
        end_ts = pd.Timestamp(int(to_year), 12, 31, 23, 59, 59)

    df_filtered = filtered_df[(filtered_df["Date"] >= start_ts) & (filtered_df["Date"] <= end_ts)].copy()
    if df_filtered.empty:
        st.warning("No data available for the selected range/view.")
        conn.close()
        st.stop()

    if freq_option == "Daily":
        df_filtered["Period"] = df_filtered["Date"]
    elif freq_option == "Monthly":
        df_filtered["Period"] = df_filtered["Date"].dt.to_period("M").dt.to_timestamp()
    else:
        df_filtered["Period"] = df_filtered["Date"].dt.to_period("Y").dt.to_timestamp()

    st.markdown(f"## {title}")
    st.markdown("---")

    # --- BENCHMARK VISIBILITY LOGIC ---
    if view_option in ["All Depots Combined", "Region Wise"]:
        show_benchmark = False
    else:
        show_benchmark = True
    # ----------------------------------

    charts_to_export = []

    # --- Loop ratios ---
    for actual_column, ratio_heading in RATIO_HEADINGS.items():
        if actual_column not in df_filtered.columns:
            continue

        # --- SPECIAL FORMATTING FOR DRIVER SCHEDULE ---
        # If the current column is Driver_Schedule, use 2 decimals and NO percentage sign
        is_ratio = (actual_column == 'Driver_Schedule')
        
        if is_ratio:
            format_str = "{:.2f}"   # Python format for metrics (e.g., 2.18)
            d3_format = ".2f"       # Altair/D3 format for charts
        else:
            format_str = "{:.1f}%"  # Python format for metrics (e.g., 2.2%)
            d3_format = ".1f"       # Altair/D3 format for charts
        # ----------------------------------------------

        # --- BENCHMARK VALUE RETRIEVAL ---
        benchmark_val = None
        if show_benchmark:
            try:
                depot_category = df_filtered["category"].mode().iloc[0]
                if depot_category not in ["Rural", "Urban"]:
                    depot_category = "Urban" 
            except:
                depot_category = "Urban"

            benchmark_val = benchmarks.get(depot_category, benchmarks["Urban"]).get(actual_column, None)
            if benchmark_val is None:
                 benchmark_val = benchmarks["Urban"].get(actual_column, None)
        # ---------------------------------

        st.markdown(f"<h3 style='font-weight:bold; font-size:22px;'>{ratio_heading}</h3>", unsafe_allow_html=True)

        # --- INDIVIDUAL DEPOTS (LINE CHART) ---
        if view_option == "Individual Depots":
            trend_df = df_filtered.groupby("Period")[actual_column].mean().reset_index().sort_values("Period")
            if trend_df.empty:
                st.write("No data to plot.")
                st.markdown("---")
                continue

            if benchmark_val is not None:
                trend_df["Benchmark"] = benchmark_val
                plot_df = trend_df.melt(
                    id_vars=["Period"],
                    value_vars=[actual_column, "Benchmark"],
                    var_name="Category",
                    value_name="Value"
                )
                plot_df["Category"] = plot_df["Category"].map({actual_column: "Actual", "Benchmark": "Benchmark"})
            else:
                plot_df = trend_df

            avg_val = trend_df[actual_column].mean()
            
            # Use format_str for Metrics
            c1, c2 = st.columns(2)
            c1.metric(f"Average {ratio_heading}", format_str.format(avg_val))
            
            if benchmark_val is not None:
                c2.metric("Benchmark", format_str.format(benchmark_val))
            else:
                c2.metric("Benchmark", "")

            tooltip_list = [alt.Tooltip("Period:T", title="Period", format=tooltip_format)]
            if benchmark_val is not None:
                tooltip_list += [
                    alt.Tooltip("Value:Q", format=d3_format, title="Value"),
                    alt.Tooltip("Category:N", title="KPI")
                ]
            else:
                tooltip_list.append(alt.Tooltip(f"{actual_column}:Q", format=d3_format, title="Actual"))

            axis_config = alt.Axis(
                labelAngle=label_angle, 
                format=date_format, 
                labelOverlap="greedy",
                tickCount=tick_count,
                title=x_axis_title,
                tickMinStep=1 
            )

            base = alt.Chart(plot_df).properties(height=420)

            if benchmark_val is not None:
                chart = base.mark_line(point=True).encode(
                    x=alt.X("Period:T", title=x_axis_title, axis=axis_config),
                    y=alt.Y("Value:Q", title=ratio_heading),
                    color=alt.Color(
                        "Category:N",
                        legend=alt.Legend(title="KPI Label"),
                        scale=alt.Scale(domain=["Actual", "Benchmark"], range=["#1f77b4", "red"])
                    ),
                    detail="Category:N",
                    tooltip=tooltip_list
                )
            else:
                chart = base.mark_line(point=True, color="blue").encode(
                    x=alt.X("Period:T", title=x_axis_title, axis=axis_config),
                    y=alt.Y(f"{actual_column}:Q", title=ratio_heading),
                    tooltip=tooltip_list,
                )

            st.altair_chart(chart, use_container_width=True)
            try:
                buffer = io.BytesIO()
                chart.save(buffer, format="png")
                buffer.seek(0)
                charts_to_export.append((ratio_heading, buffer))
            except Exception as e:
                st.warning(f"Could not save chart '{ratio_heading}' for export: {e}.")
            st.markdown("---")

        # --- ALL DEPOTS / REGION WISE (BAR CHART) ---
        else:
            agg_df = df_filtered.groupby("Depot")[actual_column].mean().reset_index()
            if agg_df.empty:
                st.write("No data to plot.")
                st.markdown("---")
                continue

            avg_val = agg_df[actual_column].mean()
            agg_df = agg_df.sort_values(by=actual_column, ascending=False).reset_index(drop=True)

            c1, c2 = st.columns(2)
            c1.metric(f"Average {ratio_heading}", format_str.format(avg_val))
            if benchmark_val is not None:
                c2.metric("Benchmark Target", format_str.format(benchmark_val))
            else:
                c2.metric("Benchmark Target", "")

            bar_tooltip_list = [
                alt.Tooltip("Depot:N"), 
                alt.Tooltip(f"{actual_column}:Q", format=d3_format, title="Actual")
            ]
            
            bar = alt.Chart(agg_df).mark_bar().encode(
                x=alt.X("Depot:N", sort=list(agg_df["Depot"]), title="Depot", axis=alt.Axis(labelAngle=-90)),
                y=alt.Y(f"{actual_column}:Q", title=ratio_heading),
                tooltip=bar_tooltip_list
            ).properties(height=420)

            # Apply d3_format to the text labels on bars
            text = bar.mark_text(align="left", baseline="middle", dy=0, fontSize=12, angle=270).encode(
                text=alt.Text(f"{actual_column}:Q", format=d3_format)
            )

            chart = bar + text

            if benchmark_val is not None:
                rule_df = pd.DataFrame({'Benchmark': [benchmark_val]})
                rule = alt.Chart(rule_df).mark_rule(color="red", strokeDash=[5, 5]).encode(
                    y="Benchmark:Q"
                )
                chart = chart + rule

            st.altair_chart(chart, use_container_width=True)
            try:
                buffer = io.BytesIO()
                chart.save(buffer, format="png")
                buffer.seek(0)
                charts_to_export.append((ratio_heading, buffer))
            except Exception as e:
                st.warning(f"Could not save chart '{ratio_heading}' for export: {e}.")
            st.markdown("---")

   

if __name__ == "__main__":
    eight_ratios_dashboard()
