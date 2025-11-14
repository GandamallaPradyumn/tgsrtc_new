import streamlit as st
import pandas as pd
import altair as alt
import calendar
import json
from datetime import timedelta, date

# ✅ ORM imports
from db_config import get_session
from models import InputData, TSAdmin, User


# Load config
with open("config.json") as f:
    config = json.load(f)

MYSQL_TABLE_NAME = 'input_data'
MYSQL_COLUMNS = config.get("db_columns", [])


# --- ORM Database Functions ---

def get_connection():
    """Deprecated: No longer needed. Kept for compatibility."""
    return None


def get_user_depot(_conn, userid):
    """Fetch depot from 'User' table using SQLAlchemy ORM."""
    if not userid:
        return ""
    try:
        with get_session() as db:
            depot = db.query(User.depot).filter(User.userid == userid).scalar()
            return depot if depot else ""
    except Exception as e:
        st.error(f"Error fetching depot for user {userid}: {e}")
        return ""


def get_depot_settings(_conn=None):
    """Fetch depot_name and category from TS_ADMIN using ORM."""
    try:
        with get_session() as db:
            results = db.query(TSAdmin.depot_name, TSAdmin.category).all()
            return {r[0]: r[1] for r in results}
    except Exception as e:
        st.error(f"Error fetching depot settings: {e}")
        return {}


def load_data(depot_settings, _conn=None):
    """Load input_data from ORM instead of SQL."""
    try:
        with get_session() as db:
            rows = db.query(InputData).all()
            df = pd.DataFrame([{k: v for k, v in row.__dict__.items() if not k.startswith("_")} for row in rows])
    except Exception as e:
        st.error(f"Error fetching data from input_data table: {e}")
        st.stop()

    if df.empty:
        return pd.DataFrame()

    if 'data_date' in df.columns:
        df.rename(columns={'data_date': 'Date'}, inplace=True)
    if 'depot_name' in df.columns:
        df.rename(columns={'depot_name': 'Depot'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.columns = df.columns.str.strip()

    if 'Category' not in df.columns:
        df['Category'] = df['Depot'].map(depot_settings).fillna('Unknown')
    return df


# --- MAIN FUNCTION ---
def eight_ratios_DM():
    # --- ORM replaces MySQL ---
    mysql_conn = get_connection()

    if "userid" in st.session_state and st.session_state.userid != "admin":
        st.session_state.depot = get_user_depot(mysql_conn, st.session_state.userid)
    elif st.session_state.userid == "admin":
        if "depot" not in st.session_state:
            st.session_state.depot = ""
    else:
        st.session_state.depot = ""

    depot_settings = get_depot_settings(mysql_conn)
    df = load_data(depot_settings, mysql_conn)

    # --- KEEPING REST OF YOUR CODE UNCHANGED ---
    if df.empty:
        st.error("No data loaded from the database. Please ensure your input_data table has data and your DB_CONFIG is correct.")
        st.stop()

    # Define benchmarks for Urban and Rural categories
    benchmarks = {
        'Urban': {
            'Pct_Weekly_Off_National_Off': 14,'Pct_Special_Off_Night_Out_IC_Online': 27.4,'Pct_Others': 1,'Pct_Leave_Absent': 6,
            'Pct_Sick_Leave': 2 , 'Pct_Spot_Absent': 2,
            'Pct_Double_Duty': 8, 'Pct_Off_Cancellation': 2
        },
        'Rural': {
            'Pct_Weekly_Off_National_Off': 14, 'Pct_Special_Off_Night_Out_IC_Online': 25,'Pct_Others': 1.7, 'Pct_Leave_Absent': 2,'Pct_Sick_Leave': 2,
            'Pct_Spot_Absent': 1, 'Pct_Double_Duty': 16,'Pct_Off_Cancellation': 2,
        }
    }

    # --- Manual Mapping for Ratio Headings ---
    RATIO_HEADINGS = {
        'Pct_Weekly_Off_National_Off': 'Weekly Off + National Off %',
        'Pct_Others': 'Others + OD %',
        'Pct_Sick_Leave': 'Sick Leave %',
        'Pct_Spot_Absent': 'Spot Absent%',
        'Pct_Off_Cancellation': 'Off Cancellation %',
        'Pct_Special_Off_Night_Out_IC_Online': 'Special Off/Night Out/IC Online %',
        'Pct_Double_Duty': 'Double Duty %',
        'Pct_Leave_Absent': 'Leave Absent %'
    }

    # --- Dashboard Title ---
    st.markdown("<h1 style='text-align: center; color: #1f77b4; font-size: 3em;'>Productivity Budget Ratios vs Actual 8 Ratios Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Filter Options ---
    st.markdown("<h2 style='font-size: 1.8em;'>DEPOT</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        selected_depot = None

        if st.session_state.userid == "admin":
            if not df.empty and 'Depot' in df.columns:
                all_depots_available = sorted(df['Depot'].unique().tolist())
                if st.session_state.depot in all_depots_available:
                    default_index = all_depots_available.index(st.session_state.depot)
                else:
                    default_index = 0
                selected_depot = st.selectbox("Select Depot", all_depots_available, index=default_index)
                st.session_state.depot = selected_depot
            else:
                st.selectbox("Select Depot", ["No depots found in data"], disabled=True)
                st.session_state.depot = ""
        else:
            selected_depot = st.session_state.get("depot", "")
            st.markdown(f" <span style='font-size: 1.5em;'><b>{selected_depot}</b></span>", unsafe_allow_html=True)
            
    with col2:
        time_period = st.selectbox("Select Time Period", ["Daily", "Monthly", "Year"])

    # Determine category for benchmarks
    effective_category_for_benchmarks = 'Urban'
    depot_display_category = "N/A"

    if selected_depot and selected_depot in depot_settings:
        depot_category_from_ts_admin = depot_settings[selected_depot]
        depot_display_category = depot_category_from_ts_admin
        if depot_category_from_ts_admin in benchmarks:
            effective_category_for_benchmarks = depot_category_from_ts_admin
    elif selected_depot:
        if not df.empty and 'Depot' in df.columns and 'Category' in df.columns:
            depot_category_from_df_series = df[df['Depot'] == selected_depot]['Category']
            if not depot_category_from_df_series.empty:
                depot_category_from_df = depot_category_from_df_series.iloc[0]
                depot_display_category = depot_category_from_df
                if depot_category_from_df in benchmarks:
                    effective_category_for_benchmarks = depot_category_from_df
            else:
                depot_display_category = 'Unknown'
        else:
            depot_display_category = 'Unknown'

    # --- Date Range Filters ---
    start_date_filter = None
    end_date_filter = None

    if time_period == "Daily":
        if not df.empty and 'Date' in df.columns:
            max_date_available = df['Date'].max()
            default_daily_end_date = max_date_available.date()
            default_daily_start_date = (max_date_available - timedelta(days=29)).date()

            col_daily_from, col_daily_to = st.columns(2)
            with col_daily_from:
                daily_from_date = st.date_input("From Date", value=default_daily_start_date,
                                                min_value=df['Date'].min().date(),
                                                max_value=default_daily_end_date)
            with col_daily_to:
                daily_to_date = st.date_input("To Date", value=default_daily_end_date,
                                              min_value=daily_from_date,
                                              max_value=default_daily_end_date)
            start_date_filter = pd.to_datetime(daily_from_date)
            end_date_filter = pd.to_datetime(daily_to_date)
        else:
            st.info("Date data not available for Daily filter.")

    elif time_period == "Monthly":
        st.markdown("<h3 style='font-size: 1.4em;'>Month Range (Monthly)</h3>", unsafe_allow_html=True)
        if not df.empty and 'Date' in df.columns:
            min_year = df['Date'].min().year
            max_year = df['Date'].max().year
            all_years = sorted(list(set(range(min_year, max_year + 1)).union({date.today().year})))
            all_months = list(calendar.month_name)[1:]
            month_to_num = {month: i + 1 for i, month in enumerate(all_months)}
            col_from, col_to = st.columns(2)
            with col_from:
                from_month = st.selectbox("From Month", all_months, index=0)
                from_year = st.selectbox("From Year", all_years, index=0)
            with col_to:
                to_month = st.selectbox("To Month", all_months, index=len(all_months) - 1)
                to_year = st.selectbox("To Year", all_years, index=len(all_years) - 1)
            start_date_filter = pd.to_datetime(f"{from_year}-{month_to_num[from_month]}-01")
            end_date_filter = pd.to_datetime(f"{to_year}-{month_to_num[to_month]}-01") + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        else:
            st.info("Date data not available for Monthly filter.")

    elif time_period == "Year":
        if not df.empty and 'Date' in df.columns:
            years = sorted(df['Date'].dt.year.unique())
            col_from, col_to = st.columns(2)
            with col_from:
                from_year = st.selectbox("From Year", years, index=0)
            with col_to:
                to_year = st.selectbox("To Year", years, index=len(years) - 1)
            start_date_filter = pd.to_datetime(f"{from_year}-01-01")
            end_date_filter = pd.to_datetime(f"{to_year}-12-31")
        else:
            st.info("Date data not available for Year filter.")

    st.markdown("---")

    st.markdown(f"### Data for: *{selected_depot if selected_depot else 'N/A'}* Depot ({depot_display_category})")
    st.markdown(f"*Time Period:* {time_period}")

    filtered_df = df.copy()
    if selected_depot and 'Depot' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Depot'] == selected_depot]
    if start_date_filter is not None and end_date_filter is not None and 'Date' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date_filter) & (filtered_df['Date'] <= end_date_filter)]

    if filtered_df.empty:
        st.warning("NO DATA FOUND FOR SELECTED FILTERS.")
    else:
        for selected_ratio in benchmarks[effective_category_for_benchmarks].keys():
            actual_column = selected_ratio
            current_benchmark = benchmarks[effective_category_for_benchmarks][selected_ratio]
            ratio_display_name = RATIO_HEADINGS.get(selected_ratio, selected_ratio)

            st.markdown(f"<h3 style='font-size: 1.5em;'> <b>{ratio_display_name}</b></h3>", unsafe_allow_html=True)

            if actual_column not in filtered_df.columns:
                st.warning(f"Column '{actual_column}' not found in data. Skipping.")
                continue

            aggregated_df = filtered_df.copy()
            if time_period == "Daily":
                group_cols = ['Depot', 'Date']
                aggregated_df = aggregated_df.groupby(group_cols).agg({actual_column: 'mean'}).reset_index()
            elif time_period == "Monthly":
                group_cols = ['Depot', pd.Grouper(key='Date', freq='MS')]
                aggregated_df = aggregated_df.groupby(group_cols).agg({actual_column: 'mean'}).reset_index()
            elif time_period == "Year":
                group_cols = ['Depot', filtered_df['Date'].dt.year.rename('Year')]
                aggregated_df = filtered_df.groupby(group_cols).agg({actual_column: 'mean'}).reset_index()
                aggregated_df['Date'] = pd.to_datetime(aggregated_df['Year'], format='%Y')

            if aggregated_df.empty:
                st.info(f"No data for {ratio_display_name}")
                continue

            current_actual = aggregated_df[actual_column].mean()
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"Average {ratio_display_name}", value=f"{current_actual:.1f}%")
            with col2:
                st.metric(label=f"Benchmark {ratio_display_name}", value=f"{current_benchmark:.1f}%")

            chart_df_actual = aggregated_df.copy()
            chart_df_actual['Type'] = f"% {ratio_display_name}"
            chart_df_actual['Value'] = chart_df_actual[actual_column] / 100
            chart_df_benchmark = aggregated_df.copy()
            chart_df_benchmark['Type'] = f"Benchmark {ratio_display_name}"
            chart_df_benchmark['Value'] = current_benchmark / 100
            combined_chart_df = pd.concat([chart_df_actual, chart_df_benchmark])

            chart = alt.Chart(combined_chart_df).encode(
                x=alt.X('Date', type='temporal', title='Date'),
                y=alt.Y('Value', title=f'{ratio_display_name}', axis=alt.Axis(format='.1%')),
                color=alt.Color('Type', scale=alt.Scale(range=['steelblue', 'red'])),
                strokeDash=alt.StrokeDash('Type', scale=alt.Scale(range=[[0, 0], [5, 5]])),
                tooltip=['Date', 'Value', 'Type']
            ).mark_line(point=True)

            st.altair_chart(chart, use_container_width=True)
            st.markdown("---")


# --- Run ---
if __name__ == "__main__":
    eight_ratios_DM()
