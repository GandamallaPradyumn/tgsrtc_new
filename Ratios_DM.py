import streamlit as st
import pandas as pd
import base64
from datetime import date, timedelta
import json
from db_config import get_session
from models import InputData, TSAdmin

# --------- Load config ----------
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    st.error("Configuration file 'config.json' not found.")
    st.stop()


class prod_ratios_DM:

    def __init__(self):
        self.ratios()

    def ratios(self):
        # Logo
        file_path = r"LOGO.png"
        try:
            with open(file_path, "rb") as img_file:
                b64_img = base64.b64encode(img_file.read()).decode()
        except FileNotFoundError:
            b64_img = ""  # no logo file

        # top page header (normal Streamlit display)
        st.markdown(f"""
            <div style="text-align: center; background-color: #19bc9c; border-radius: 12px; padding:10px;">
                {"<img src='data:image/png;base64," + b64_img + "' width='110' height='110' style='display:block; margin:0 auto;'>" if b64_img else ""}
                <h1 style="color: white; margin:6px 0 8px 0;">Telangana State Road Transport Corporation</h1>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>🚍 Productivity Budget 8 Ratios (Rural/Urban)</h1>", unsafe_allow_html=True)

        # ✅ ORM Database connection
        try:
            with get_session() as db:
                data_query = db.query(InputData).all()
                df = pd.DataFrame([{k: v for k, v in d.__dict__.items() if not k.startswith("_")} for d in data_query])
        except Exception as err:
            st.error(f"MySQL connection error: {err}")
            st.stop()

        if df.empty:
            st.warning("⚠ No data found in input_data table.")
            st.stop()

        df['data_date'] = pd.to_datetime(df['data_date'], errors='coerce')

        all_depots = sorted(df['depot_name'].dropna().unique())
        time_periods = ['Daily', 'Monthly', 'Quarterly', 'Yearly']

        col_a, col_b = st.columns(2)

        # Depot from session state
        with col_a:
            selected_depot = st.session_state.get("user_depot", None)
            if not selected_depot:
                st.error("🚫 Depot not assigned. Please log in properly.")
                st.stop()
            st.markdown(f"""
            <div style="font-weight:bold; padding-bottom:4px;">Depot:</div>
            <div style="background-color:#f0f2f6; padding:0.5em; border-radius:4px; font-weight:bold;">
                {selected_depot}
            </div>
            """, unsafe_allow_html=True)

        # ✅ Fetch category from TS_ADMIN using ORM
        try:
            with get_session() as db:
                category = db.query(TSAdmin.category).filter(TSAdmin.depot_name == selected_depot).scalar()
                category = category.strip().lower() if category else "unknown"
        except Exception as err:
            st.error(f"Category fetch error: {err}")
            category = "unknown"

        with col_b:
            selected_time_period = st.selectbox("Select Time Period:", time_periods)

        filtered_df = df[df['depot_name'] == selected_depot]
        min_date = filtered_df["data_date"].min()
        max_date = filtered_df["data_date"].max()
        if pd.isna(min_date) or pd.isna(max_date):
            st.warning("⚠ No valid date data found for the selected depot.")
            today = date.today()
            min_date = today - timedelta(days=30)
            max_date = today
        else:
            min_date = min_date.date() if hasattr(min_date, "date") else min_date
            max_date = max_date.date() if hasattr(max_date, "date") else max_date

        col_c, col_d = st.columns(2)

        # Time period filtering
        if selected_time_period == "Daily":
            with col_c:
                date_filter = st.date_input("Select Date", min_value=min_date, max_value=max_date, value=max_date)
            filtered_df = filtered_df[filtered_df['data_date'] == pd.to_datetime(date_filter)]

        elif selected_time_period == "Monthly":
            with col_c:
                year_filter = st.selectbox("Year:", sorted(filtered_df['data_date'].dt.year.dropna().unique(), reverse=True), key="monthly_year")
            with col_d:
                month_filter = st.selectbox(
                    "Month:",
                    options=list(range(1, 13)),
                    format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x - 1],
                    key="monthly_month"
                )
            filtered_df = filtered_df[
                (filtered_df['data_date'].dt.year == year_filter) &
                (filtered_df['data_date'].dt.month == month_filter)
            ]

        elif selected_time_period == "Quarterly":
            with col_c:
                year_filter = st.selectbox("Year:", sorted(filtered_df['data_date'].dt.year.dropna().unique(), reverse=True), key="quarterly_year")
            with col_d:
                quarter_filter = st.selectbox("Quarter:", ["Q1 (Jan–Mar)", "Q2 (Apr–Jun)", "Q3 (Jul–Sep)", "Q4 (Oct–Dec)"], key="quarter")
            quarter_map = {
                "Q1 (Jan–Mar)": (1, 3),
                "Q2 (Apr–Jun)": (4, 6),
                "Q3 (Jul–Sep)": (7, 9),
                "Q4 (Oct–Dec)": (10, 12)
            }
            start_month, end_month = quarter_map[quarter_filter]
            filtered_df = filtered_df[
                (filtered_df['data_date'].dt.year == year_filter) &
                (filtered_df['data_date'].dt.month >= start_month) &
                (filtered_df['data_date'].dt.month <= end_month)
            ]

        elif selected_time_period == "Yearly":
            with col_c:
                year_filter = st.selectbox("Year:", sorted(filtered_df['data_date'].dt.year.dropna().unique(), reverse=True), key="yearly_year")
            filtered_df = filtered_df[filtered_df['data_date'].dt.year == year_filter]

        # If data exists
        if not filtered_df.empty:
            num_days = filtered_df['data_date'].nunique()

            st.markdown(f"<h2 style='text-align: center;'>Productivity Ratios For {selected_depot.capitalize()}</h2>", unsafe_allow_html=True)
            st.markdown(f"<mark> Category: {category.capitalize()}  |  Days Considered: {num_days}<mark>", unsafe_allow_html=True)

            planned_schedules = int(filtered_df['Planned_Schedules'].sum())
            total_drivers = int(filtered_df['Total_Drivers'].sum())
            drivers_per_schedule = (total_drivers / planned_schedules) if planned_schedules != 0 else 0
            drivers_per_schedule = round(drivers_per_schedule, 2)

            col1, col2, col3 = st.columns(3)

            # Thresholds
            rural_thresholds = {
                "Weekly Off": 14.0,
                "Special Off (Night Out/IC, Online)": 25.0,
                "Others": 1.70,
                "Long Leave & Absent": 2.0,
                "Sick Leave": 2.0,
                "Spot Absent": 1.0,
                "Double Duty": 16.0,
                "Off Cancellation": 2.00,
                "Drivers/Schedule": 2.18
            }
            urban_thresholds = {
                "Weekly Off": 14.0,
                "Special Off (Night Out/IC, Online)": 27.4,
                "Others": 1.0,
                "Long Leave & Absent": 6.0,
                "Sick Leave": 2.0,
                "Spot Absent": 2.0,
                "Double Duty": 8.0,
                "Off Cancellation": 2.00,
                "Drivers/Schedule": 2.43
            }

            if category == "rural":
                thresholds = rural_thresholds
            elif category == "urban":
                thresholds = urban_thresholds
            else:
                thresholds = urban_thresholds  # default fallback

            # Top 3 metrics display
            with col1:
                st.markdown(f"""<div style="border-radius: 10px; height: 120px;">
                                    <div style="width: 150px; height: 20px; margin: 0 auto;">
                                        <p style="text-align: center; font-size: 15px;">Planned Schedules</p>
                                    </div>
                                    <div style="width: 150px; height: 20px; margin: 0 auto;">
                                        <P style="text-align: center; font-size: 35px;">{planned_schedules}</p>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                                <div style=" border-radius: 10px; height: 120px;">
                                    <div style="text-align: center; width: 150px; height: 20px; margin: 0 auto;">
                                        <p style=" font-size: 15px;">Total Drivers</p>
                                    </div>
                                    <div style="width: 150px; height: 20px; margin: 0 auto;">
                                        <P style="text-align: center; font-size: 35px;">{total_drivers}</p>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

            delta = thresholds.get("Drivers/Schedule", 0) - drivers_per_schedule

            with col3:
                color = "red" if delta < 0 else "green"
                arr = "↓" if delta < 0 else "↑"
                st.markdown(f"""
                                <div style="border-radius: 10px; height:120px; margin: 0 auto;">
                                    <div style="text-align: center;">
                                        <p style="font-size: 15px; margin: 0;">Drivers/Schedule Ratio</p>
                                        <p style="font-size: 35px; margin: 0;">{drivers_per_schedule:.2f}</p>
                                        <p style="font-size: 20px; margin: 0; color: {color};">
                                            <strong>{arr} {delta:.2f}</strong>
                                        </p>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

            # Table for all metrics
            metric_map = {
                "Weekly Off (%)": "Pct_Weekly_Off_National_Off",
                "Special Off (Night Out/IC, Online) (%)": "Pct_Special_Off_Night_Out_IC_Online",
                "Others (%)": "Pct_Others",
                "Long Leave & Absent (%)": "Pct_Leave_Absent",
                "Sick Leave (%)": "Pct_Sick_Leave",
                "Spot Absent (%)": "Pct_Spot_Absent",
                "Double Duty (%)": "Pct_Double_Duty",
                "Off Cancellation (%)": "Pct_Off_Cancellation",
            }

            rows = ""
            for metric, col in metric_map.items():
                value = round(filtered_df[col].mean(), 2)
                base_label = metric.replace(" (%)", "")
                benchmark = thresholds.get(base_label, None)
                variance = round(value - benchmark, 2) if benchmark is not None else None

                benchmark_cell = f"<td class='yellow-bg'>{benchmark}%</td>" if benchmark is not None else "<td></td>"
                variance_color = 'green-bg' if variance is not None and variance <= 0 else 'red-bg'
                variance_cell = f"<td class='{variance_color}'>{variance:+.1f}%</td>" if variance is not None else "<td></td>"
                rows += f"<tr><td style='text-align: left; width: 250px; '><strong>{metric}</strong></td>{benchmark_cell}<td>{value}%</td>{variance_cell}</tr>"

            html_table = f"""
            <style>
            .custom-table {{margin-right : auto; margin-left : auto; border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; border: 2px solid black;}}
            .custom-table th, .custom-table td {{border: 1px solid black; text-align: center; padding: 8px;}}
            .custom-table th {{background-color: #19bc9c; font-weight: bold; color:white;}}
            .yellow-bg {{background-color: yellow;}}
            .green-bg {{background-color: lightgreen;}}
            .red-bg {{background-color: tomato;}}
            </style>

            <table class="custom-table">
                <thead>
                    <tr><th>Metric</th><th>Benchmark</th><th>Actual Values</th><th>Variance</th></tr>
                </thead>
                <tbody>
                    <tr><td style='text-align: left; width: 350px;'><strong>Planned Schedules</strong></td><td style="font-weight:bold;">--- </td><td>{planned_schedules}</td><td style="font-weight:bold;">---</td></tr>
                    <tr><td style='text-align: left; width: 250px;'><strong>Total Drivers</strong></td><td style="font-weight:bold;">--- </td><td >{total_drivers}</td><td style="font-weight:bold;">---</td></tr>
                    {rows}
                    <tr><td style='text-align: left; width: 250px;'><strong>Drivers/Schedule (Ratio)</strong></td><td class='yellow-bg'>{thresholds['Drivers/Schedule']}</td><td>{drivers_per_schedule}</td><td class='{'green-bg' if (drivers_per_schedule - thresholds['Drivers/Schedule']) <= 0 else 'red-bg'}'>{drivers_per_schedule - thresholds['Drivers/Schedule']:+.2f}</td></tr>
                </tbody>
            </table>
            """

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(html_table, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.info("""
                    Note:

                        • 'Planned Schedules' and 'Total Drivers' are total sums.
                        • Variance has red background if exceeding benchmark, green background if within or better.
                        • Benchmark is highlighted in yellow.
                    """, icon="ℹ")

        else:
            st.warning("⚠ No data available for the selected filters.")


if __name__ == "__main__":
    obj = prod_ratios_DM()
