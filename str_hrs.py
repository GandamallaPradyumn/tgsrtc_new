import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from urllib.parse import quote_plus
from io import BytesIO
import xlsxwriter
import altair as alt
import datetime
from db_config import engine
import json
import pymysql


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

def str_hrs():

    # 🔐 AUTH CHECK (MATCH LOGIN SYSTEM)
    if not st.session_state.get("logged_in"):
        st.error("Unauthorized access. Please login.")
        st.stop()

    user_id = st.session_state.get("userid")
    user_role = st.session_state.get("user_role")
    user_depot = st.session_state.get("user_depot")

    #st.caption(f"👤 User: {user_id} | 🏷 Role: {user_role}")


    st.title("Steering Hours")
    
    # ------------------- Data Loader -------------------
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_data(selected_depot, user_role, user_id):


        # Daily operations (KMs only)
        oprs_query = """
            SELECT
                employee_id,
                operations_date,
                depot,
                service_number,
                opd_kms
            FROM daily_operations
            WHERE depot = %s
        """

        # Steering hours master
        hrs_query = """
            SELECT
                depot,
                service_number,
                steering_hours,
                no_of_drvs
            FROM steering_hours
            WHERE depot = %s
        """

        params = (selected_depot,)

        oprs = pd.read_sql(
            oprs_query,
            con=engine,
            params=params,
            parse_dates=["operations_date"]
        )

        hrs = pd.read_sql(
            hrs_query,
            con=engine,
            params=params
        )

        # Normalize column names
        oprs.columns = oprs.columns.str.lower()
        hrs.columns = hrs.columns.str.lower()

        # -------------------------
        # Merge steering hours
        # -------------------------
        oprs = oprs.merge(
            hrs,
            on=["depot", "service_number"],
            how="left"
        )

        # -------------------------
        # Driver-wise hour logic (CORRECT + TRACEABLE)
        # -------------------------

        # Ensure valid driver count
        oprs["no_of_drvs"] = oprs["no_of_drvs"].fillna(1).replace(0, 1)

        # Divide only where steering_hours exists
        oprs["steering_hours"] = oprs["steering_hours"] / oprs["no_of_drvs"]

        # -------------------------
        # Track source of hours
        # -------------------------
        oprs["hours_source"] = np.where(
            oprs["steering_hours"].isna(),
            "Estimated (Median)",
            "Actual"
        )

        # -------------------------
        # Median fallback (explicit)
        # -------------------------
        median_hours = oprs["steering_hours"].median()

        oprs["steering_hours"] = (
            pd.to_numeric(oprs["steering_hours"], errors="coerce")
            .fillna(median_hours)
        )

        # -------------------------
        # 🔴 CRITICAL FIX: avoid double counting
        # -------------------------
        oprs = (
            oprs
            .groupby(
                ["employee_id", "operations_date", "depot", "service_number"],
                as_index=False
            )
            .agg({
                "steering_hours": "first",
                "opd_kms": "sum",
                "hours_source": "first"
            })
        )

        return oprs

    # ------------------- Main App -------------------
    
    # ------------------- Layout -------------------
    depot_col, period_col = st.columns([1, 1])

    with depot_col:

        if user_role == "admin":
            depots = pd.read_sql(
                "SELECT DISTINCT depot FROM daily_operations",
                con=engine
            )
            depot_list = sorted(depots["depot"].dropna().unique().tolist())

            selected_depot = st.selectbox(
                "Select Depot",
                depot_list,
                index=0
            )
        else:
            selected_depot = user_depot
            st.info(f"🏢 Logged in Depot: **{selected_depot}**")

    with period_col:
        period_type = st.selectbox("Select Period Type", ["Monthly", "Yearly"])

    # ------------------- Date Selection -------------------
    from_col, to_col = st.columns(2)

    if period_type == "Monthly":
        years = list(range(2020, datetime.datetime.now().year + 1))
        months = [datetime.date(1900, m, 1).strftime("%b").upper() for m in range(1, 13)]

        with from_col:
            sel_year = st.selectbox("Select Year", years, index=len(years) - 1)
        with to_col:
            sel_month_name = st.selectbox(
                "Select Month",
                months,
                index=datetime.datetime.now().month - 1
            )

        sel_month = datetime.datetime.strptime(sel_month_name, "%b").month
        from_date = datetime.date(sel_year, sel_month, 1)
        to_date = (
            datetime.date(sel_year + 1, 1, 1) - datetime.timedelta(days=1)
            if sel_month == 12
            else datetime.date(sel_year, sel_month + 1, 1) - datetime.timedelta(days=1)
        )
    else:
        years = list(range(2020, datetime.datetime.now().year + 1))
        with from_col:
            from_year = st.selectbox("From Year", years, index=0)
        with to_col:
            to_year = st.selectbox("To Year", years, index=len(years) - 1)

        from_date = datetime.date(from_year, 1, 1)
        to_date = datetime.date(to_year, 12, 31)

    # ------------------- Load Data -------------------
    merged = load_data(selected_depot, user_role, user_id)


    merged = merged[
        merged["operations_date"].between(
            pd.to_datetime(from_date),
            pd.to_datetime(to_date)
        )
    ]

    # ------------------- Estimated Hours Warning -------------------
    if "Estimated (Median)" in merged["hours_source"].values:
        est_count = (merged["hours_source"] == "Estimated (Median)").sum()
        st.warning(
            f"⚠️ {est_count} records use **median-based estimated steering hours** "
            f"because service-wise hours were not available."
        )


    if merged.empty:
        st.warning("No data found for selected filters.")
        return

    # ------------------- Employee Highlight -------------------
    employee_ids = sorted(merged["employee_id"].dropna().unique().tolist())
    selected_employee = st.selectbox(
        "Select Employee ID to Highlight",
        ["None"] + employee_ids
    )
    selected_employees = [selected_employee] if selected_employee != "None" else []

    # ------------------- Pivot Table -------------------
    merged["DateStr"] = merged["operations_date"].dt.strftime("%d-%b-%Y")

    pivot = (
        merged
        .pivot_table(
            index="employee_id",
            columns="DateStr",
            values="steering_hours",
            aggfunc="sum",
            fill_value=0
        )
        .round(0)
        .astype(int)
    )

    # Pivot to track which cells are estimated
    estimate_pivot = (
        merged
        .assign(is_estimated=merged["hours_source"] == "Estimated (Median)")
        .pivot_table(
            index="employee_id",
            columns="DateStr",
            values="is_estimated",
            aggfunc="max",   # if ANY service that day is estimated → True
            fill_value=False
        )
    )

    pivot["Grand Total"] = pivot.sum(axis=1)
    pivot.loc["Total Hours"] = pivot.sum()

    # ------------------- Styling -------------------
    def highlight_hours(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        for r in df.index:
            for c in df.columns:

                # Skip totals
                if r == "Total Hours" or c == "Grand Total":
                    styles.loc[r, c] = "font-weight:bold"
                    continue

                val = df.loc[r, c]

                # Highlight selected employee
                if r in selected_employees:
                    styles.loc[r, c] = "background-color:#E6F3FF;font-weight:bold"
                    continue

                # 🔶 Median-estimated cells
                if r in estimate_pivot.index and c in estimate_pivot.columns:
                    if estimate_pivot.loc[r, c]:
                        styles.loc[r, c] = (
                            "background-color:#FFF3CD;"
                            "font-style:italic;"
                            "border:1px solid #FFCC00"
                        )
                        continue

                # Normal thresholds
                if val == 0:
                    styles.loc[r, c] = ""
                elif val < 15:
                    styles.loc[r, c] = "background-color:#ffcccc"
                else:
                    styles.loc[r, c] = "background-color:#b30000;color:white"

        return styles


    st.subheader(f"STEERING HOURS – {selected_depot}")
    st.markdown(pivot.style.apply(highlight_hours, axis=None).to_html(), unsafe_allow_html=True)

    # ------------------- Charts -------------------
    steering_sum = merged.groupby("employee_id")["steering_hours"].sum().reset_index()
    kms_sum = merged.groupby("employee_id")["opd_kms"].sum().reset_index()

    combined = steering_sum.merge(kms_sum, on="employee_id", how="outer").fillna(0)

    st.subheader("Steering Hours by Employee")
    st.altair_chart(
        alt.Chart(combined)
        .mark_bar(color='blue')
        .encode(
            x=alt.X(
                'employee_id:N',
                sort=alt.SortField(
                    field='steering_hours',
                    order='descending'
                )
            ),
            y='steering_hours:Q',
            tooltip=['employee_id', 'steering_hours']
        ),
        use_container_width=True
    )


    st.subheader("Operational KMs by Employee")
    st.altair_chart(
        alt.Chart(combined)
        .mark_bar(color='green')
        .encode(
            x=alt.X(
                'employee_id:N',
                sort=alt.SortField(
                    field='opd_kms',
                    order='descending'
                )
            ),
            y='opd_kms:Q',
            tooltip=['employee_id', 'opd_kms']
        ),
        use_container_width=True
    )


    # ------------------- Excel Export -------------------
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pivot.to_excel(writer, sheet_name="Pivot")

    st.download_button(
        "📥 Download Dashboard",
        output.getvalue(),
        file_name=f"Steering_Dashboard_{selected_depot}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
