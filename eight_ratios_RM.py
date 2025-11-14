import streamlit as st
import pandas as pd
import altair as alt
import json
import calendar

# ✅ ORM imports
from db_config import get_session
from models import InputData, TSAdmin, User

# --- Load config.json ---
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    st.error("Configuration file 'config.json' not found.")
    st.stop()

DB_CONFIG = config["db"]
MYSQL_TABLE_NAME = "input_data"


# --- ORM Connection Stub (no longer needed, kept for compatibility) ---
def get_connection():
    return None


# --- Get region for user (ORM version) ---
def get_region_for_user(_conn, user_id):
    try:
        with get_session() as db:
            region = (
                db.query(TSAdmin.region)
                .join(User, User.depot_name == TSAdmin.depot_name)
                .filter(User.user_id == user_id)
                .distinct()
                .scalar()
            )
            return region
    except Exception as e:
        st.error(f"Error fetching region for user {user_id}: {e}")
        return None


# --- Load Data for Region (ORM version) ---
def load_data(_conn, user_region):
    df = pd.DataFrame()
    try:
        with get_session() as db:
            rows = (
                db.query(InputData, TSAdmin.category)
                .join(TSAdmin, InputData.depot_name == TSAdmin.depot_name)
                .filter(TSAdmin.region == user_region)
                .all()
            )
            data = []
            for input_row, category in rows:
                row_dict = {k: v for k, v in input_row.__dict__.items() if not k.startswith("_")}
                row_dict["category"] = category
                data.append(row_dict)
            df = pd.DataFrame(data)

        df.rename(columns={"data_date": "Date", "depot_name": "Depot"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    return df


# --- Main App ---
def eight_ratios_RM():
    conn = get_connection()

    # --- Get region directly from session (set in login.py) ---
    selected_region = st.session_state.get("user_region", None)
    if not selected_region:
        st.error("No region found in session. Please log in as RM.")
        st.stop()

    # --- Load region data ---
    df = load_data(conn, selected_region)
    if df.empty:
        st.warning(f"No data found for region {selected_region}.")
        st.stop()

    # --- Benchmarks ---
    benchmarks = config["benchmarks"]

    # --- Frequency Selector ---
    freq_option = st.selectbox("Select Frequency", ["Daily", "Monthly", "Yearly"])

    min_date, max_date = df["Date"].min(), df["Date"].max()

    if freq_option == "Daily":
        col_from, col_to = st.columns(2)
        with col_from:
            start_date = st.date_input("From Date", min_value=min_date, max_value=max_date, value=min_date)
        with col_to:
            end_date = st.date_input("To Date", min_value=min_date, max_value=max_date, value=max_date)

    elif freq_option == "Monthly":
        months = list(calendar.month_name)[1:]
        years = list(range(min_date.year, max_date.year + 1))

        col_from_m, col_from_y, col_to_m, col_to_y = st.columns(4)
        with col_from_m:
            from_month = st.selectbox("From Month", months, index=0)
        with col_from_y:
            from_year = st.selectbox("From Year", years, index=0)
        with col_to_m:
            to_month = st.selectbox("To Month", months, index=len(months) - 1)
        with col_to_y:
            to_year = st.selectbox("To Year", years, index=len(years) - 1)

        start_date = pd.to_datetime(f"{from_year}-{months.index(from_month)+1}-01")
        end_date = pd.to_datetime(f"{to_year}-{months.index(to_month)+1}-01") + pd.offsets.MonthEnd(1)

    elif freq_option == "Yearly":
        years = list(range(min_date.year, max_date.year + 1))
        col_from_y, col_to_y = st.columns(2)
        with col_from_y:
            from_year = st.selectbox("From Year", years, index=0)
        with col_to_y:
            to_year = st.selectbox("To Year", years, index=len(years)-1)

        start_date = pd.to_datetime(f"{from_year}-01-01")
        end_date = pd.to_datetime(f"{to_year}-12-31")

    # --- Filter Data ---
    df_region = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

    st.markdown(f"## Region: {selected_region}")
    st.markdown("---")

    if df_region.empty:
        st.warning("No data in selected range.")
    else:
        # Loop through each KPI ratio
        for selected_ratio_key in benchmarks["Urban"].keys():
            actual_column = config["category_to_column"].get(selected_ratio_key)
            if actual_column not in df_region.columns:
                continue

            # Aggregate depot averages
            agg_df = df_region.groupby("Depot")[actual_column].mean().reset_index()

            # Benchmark (use category of first depot)
            first_depot = agg_df["Depot"].iloc[0]
            depot_category = df_region[df_region["Depot"] == first_depot]["category"].iloc[0]
            benchmark_val = benchmarks.get(depot_category, benchmarks["Urban"]).get(selected_ratio_key, 0)

            # Region Average
            avg_val = agg_df[actual_column].mean()
            agg_df["Benchmark"] = benchmark_val

            # --- KPI Header (bold + bigger) ---
            st.markdown(f"<h3 style='font-weight:bold; font-size:22px;'>{selected_ratio_key}</h3>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"Average {selected_ratio_key}", value=f"{avg_val:.1f}%")
            with col2:
                st.metric(label=f"Benchmark {selected_ratio_key}", value=f"{benchmark_val:.1f}%")

            # --- Chart with bold depot names + values ---
            bar = alt.Chart(agg_df).mark_bar(color="steelblue").encode(
                x=alt.X("Depot:N", sort="-y", title="Depot",
                        axis=alt.Axis(labelFontWeight="bold", labelFontSize=14, titleFontSize=16, titleFontWeight="bold")),
                y=alt.Y(f"{actual_column}:Q", title="Value (%)",
                        axis=alt.Axis(labelFontWeight="bold", labelFontSize=14, titleFontSize=16, titleFontWeight="bold")),
                tooltip=["Depot", actual_column]
            )

            text = bar.mark_text(
                align="center",
                baseline="bottom",
                dy=-2,
                fontWeight="bold",  # bold values
                fontSize=14         # bigger values
            ).encode(
                text=alt.Text(f"{actual_column}:Q", format=".1f")
            )

            benchmark_line = alt.Chart(agg_df).mark_rule(
                color="red", strokeDash=[5, 5], size=2
            ).encode(
                y="Benchmark:Q"
            )

            st.altair_chart(bar + text + benchmark_line, use_container_width=True)
            st.markdown("---")


# --- Run ---
if __name__ == "__main__":
    eight_ratios_RM()
