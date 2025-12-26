import streamlit as st
import pandas as pd
import base64
import json
import mysql.connector
from datetime import date, timedelta
from mysql.connector import Error
import calendar
import html as pyhtml
import streamlit.components.v1 as components

# --------- Load config ----------
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    st.error("Configuration file 'config.json' not found.")
    st.stop()

# --------- Benchmarks ----------
BENCHMARKS = {
    "Rural": {
        "% Weekly Off & National Off": 14,
        "% Special Off (Night Out/IC, Online)": 25,
        "% Others": 1.7,
        "% Leave & Absent": 2,
        "% Sick Leave": 2,
        "% Spot Absent": 1,
        "% Double Duty": 16,
        "% Off Cancellation": 2,
        "Drivers/Schedule": 2.18,
    },
    "Urban": {
        "% Weekly Off & National Off": 14,
        "% Special Off (Night Out/IC, Online)": 27.4,
        "% Others": 1,
        "% Leave & Absent": 6,
        "% Sick Leave": 2,
        "% Spot Absent": 2,
        "% Double Duty": 8,
        "% Off Cancellation": 2,
        "Drivers/Schedule": 2.43,
    },
}


# --------- DB connection helper ----------
def get_connection():
    try:
        return mysql.connector.connect(**config["db"])
    except Error as e:
        st.error(f"Error connecting to database: {e}")
        return None


# --------- Main Class ----------
class ProdRatiosDashboard:
    def __init__(self):
        self.display_table()

    def display_table(self):
        # --- Load logo ---
        try:
            with open("LOGO.png", "rb") as img_file:
                b64_img = base64.b64encode(img_file.read()).decode()
        except FileNotFoundError:
            b64_img = ""

        # --- Header ---
        st.markdown(
            f"""
            <div style="text-align: center; background-color: #19bc9c; border-radius: 12px; padding:10px;">
                {"<img src='data:image/png;base64," + b64_img + "' width='110' height='110' style='display:block; margin:0 auto;'>" if b64_img else ""}
                <h1 style="color: white; margin:6px 0 8px 0;">Telangana State Road Transport Corporation</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<h1 style='text-align: center;'>🚍 Productivity Budget - All Depots Comparison</h1>", unsafe_allow_html=True)

        # --- DB Connection ---
        conn = get_connection()
        if not conn:
            st.stop()

        # Load depot data
        try:
            depot_info = pd.read_sql("SELECT depot_name, region, category FROM TS_ADMIN", conn)
        except Exception as e:
            st.error(f"Error fetching depot info: {e}")
            conn.close()
            st.stop()

        # --- Selection Controls ---
        st.subheader("📊 Select View Mode:")
        view_option = st.radio(
            "Choose Data View:",
            ["All Depots Combined", "Region Wise", "Rural Depots", "Urban Depots", "Individual Depots"],
            horizontal=True,
        )

        selected_region = None
        selected_depot = None

        if view_option == "Region Wise":
            selected_region = st.selectbox("Select Region:", sorted(depot_info["region"].unique()))

        elif view_option == "Individual Depots":
            selected_depot = st.selectbox("Select Depot:", sorted(depot_info["depot_name"].unique()))

        # --- Query Logic ---
        try:
            if view_option == "All Depots Combined":
                query = """
                    SELECT d.*, a.category, a.region
                    FROM input_data d
                    JOIN TS_ADMIN a ON d.depot_name = a.depot_name
                """

            elif view_option == "Region Wise" and selected_region:
                query = f"""
                    SELECT d.*, a.category, a.region
                    FROM input_data d
                    JOIN TS_ADMIN a ON d.depot_name = a.depot_name
                    WHERE a.region = '{selected_region}'
                """

            elif view_option == "Rural Depots":
                query = """
                    SELECT d.*, a.category, a.region
                    FROM input_data d
                    JOIN TS_ADMIN a ON d.depot_name = a.depot_name
                    WHERE a.category = 'Rural'
                """

            elif view_option == "Urban Depots":
                query = """
                    SELECT d.*, a.category, a.region
                    FROM input_data d
                    JOIN TS_ADMIN a ON d.depot_name = a.depot_name
                    WHERE a.category = 'Urban'
                """

            elif view_option == "Individual Depots" and selected_depot:
                query = f"""
                    SELECT d.*, a.category, a.region
                    FROM input_data d
                    JOIN TS_ADMIN a ON d.depot_name = a.depot_name
                    WHERE a.depot_name = '{selected_depot}'
                """

            else:
                st.warning("Please select a valid option.")
                conn.close()
                st.stop()

            df = pd.read_sql(query, conn)
            df["data_date"] = pd.to_datetime(df["data_date"], errors="coerce")

        except Exception as e:
            st.error(f"Database Error: {e}")
            conn.close()
            st.stop()

        if df.empty:
            st.warning("⚠ No data found for the selected filters.")
            conn.close()
            st.stop()

        # --- Time Period Selection ---
        time_periods = ["Daily", "Monthly", "Quarterly", "Yearly"]
        selected_time_period = st.selectbox("Select Time Period:", time_periods)
        col1, col2 = st.columns(2)

        min_date = df["data_date"].min().date()
        max_date = df["data_date"].max().date()
        filtered_df = pd.DataFrame()

        if selected_time_period == "Daily":
            with col1:
                date_filter = st.date_input("Select Date:", max_value=max_date, min_value=min_date, value=max_date)
            filtered_df = df[df["data_date"] == pd.to_datetime(date_filter)]

        elif selected_time_period == "Monthly":
            with col1:
                year_filter = st.selectbox("Year:", sorted(df["data_date"].dt.year.unique(), reverse=True))
            with col2:
                month_filter = st.selectbox("Month:", list(range(1, 13)), format_func=lambda x: calendar.month_name[x])
            filtered_df = df[
                (df["data_date"].dt.year == year_filter) & (df["data_date"].dt.month == month_filter)
            ]

        elif selected_time_period == "Quarterly":
            with col1:
                year_filter = st.selectbox("Year:", sorted(df["data_date"].dt.year.unique(), reverse=True))
            with col2:
                quarter_filter = st.selectbox("Quarter:", ["Q1 (Jan–Mar)", "Q2 (Apr–Jun)", "Q3 (Jul–Sep)", "Q4 (Oct–Dec)"])
            quarter_map = {"Q1 (Jan–Mar)": (1,3),"Q2 (Apr–Jun)": (4,6),"Q3 (Jul–Sep)": (7,9),"Q4 (Oct–Dec)": (10,12)}
            sm, em = quarter_map[quarter_filter]
            filtered_df = df[
                (df["data_date"].dt.year == year_filter)
                & (df["data_date"].dt.month >= sm)
                & (df["data_date"].dt.month <= em)
            ]

        elif selected_time_period == "Yearly":
            with col1:
                year_filter = st.selectbox("Year:", sorted(df["data_date"].dt.year.unique(), reverse=True))
            filtered_df = df[df["data_date"].dt.year == year_filter]

        if filtered_df.empty:
            st.warning("⚠ No data available for selected filters.")
            conn.close()
            st.stop()

        # --- Build KPI Table ---
        depots = sorted(filtered_df["depot_name"].unique())
        metric_map = {
            "Planned Schedules": None,
            "Total Drivers": None,
            "Weekly Off & National Off (%)": "% Weekly Off & National Off",
            "Special Off (Night Out/IC, Online) (%)": "% Special Off (Night Out/IC, Online)",
            "Others (%)": "% Others",
            "Long Leave & Absent (%)": "% Leave & Absent",
            "Sick Leave (%)": "% Sick Leave",
            "Spot Absent (%)": "% Spot Absent",
            "Double Duty (%)": "% Double Duty",
            "Off Cancellation (%)": "% Off Cancellation",
            "Drivers/Schedule (Ratio)": "Drivers/Schedule",
        }

        html_rows = ""
        for metric, base_label in metric_map.items():
            # --- Benchmark Cell Logic (with robust normalization for Individual Depot) ---
            benchmark_cell = "<td>---</td>"
            benchmark_value = None
            if base_label:
                if view_option == "Rural Depots":
                    benchmark_value = BENCHMARKS["Rural"].get(base_label)
                elif view_option == "Urban Depots":
                    benchmark_value = BENCHMARKS["Urban"].get(base_label)
                elif view_option == "Individual Depots":
                    # robust category normalization & fallback
                    try:
                        depot_cat_raw = filtered_df["category"].iloc[0]
                        depot_cat_norm = str(depot_cat_raw).strip().title()  # e.g., 'rural ' -> 'Rural'
                        # direct lookup
                        benchmark_value = BENCHMARKS.get(depot_cat_norm, {}).get(base_label)
                        # fallback map if direct lookup failed (handles weird strings)
                        if benchmark_value is None:
                            lc = str(depot_cat_raw).strip().lower()
                            fallback_map = {"rural": "Rural", "urban": "Urban"}
                            for key_prefix, mapped in fallback_map.items():
                                if lc.startswith(key_prefix):
                                    benchmark_value = BENCHMARKS.get(mapped, {}).get(base_label)
                                    break
                    except Exception:
                        benchmark_value = None

                # show benchmark if found
                if benchmark_value is not None:
                    benchmark_cell = f"<td class='yellow-bg'>{benchmark_value}%</td>"

            # --- Depot Value Logic ---
            depot_cells = ""
            depot_values = []
            for depot in depots:
                depot_df = filtered_df[filtered_df["depot_name"] == depot]
                if depot_df.empty:
                    depot_cells += "<td>---</td>"
                    continue

                try:
                    if metric == "Planned Schedules":
                        value = int(depot_df["Planned_Schedules"].sum())
                    elif metric == "Total Drivers":
                        value = int(depot_df["Total_Drivers"].sum())
                    elif metric == "Drivers/Schedule (Ratio)":
                        ps = depot_df["Planned_Schedules"].sum()
                        td = depot_df["Total_Drivers"].sum()
                        value = round(td / ps, 2) if ps else 0
                    else:
                        col_name = config.get("category_to_column", {}).get(base_label, None)
                        value = round(depot_df[col_name].mean(), 1) if col_name and col_name in depot_df.columns else "---"
                except:
                    value = "---"

                try:
                    depot_values.append(float(value))
                except:
                    pass

                # --- Color Logic ---
                if base_label and isinstance(value, (int, float)) and (benchmark_value is not None):
                    delta = value - benchmark_value
                    color = "green" if delta <= 0 else "red"
                    depot_cells += f"<td style='color:{color}; font-weight:bold'>{value}</td>"
                else:
                    depot_cells += f"<td>{value}</td>"

            avg_val = round(sum(depot_values) / len(depot_values), 1) if depot_values else "---"
            html_rows += f"<tr><td><strong>{metric}</strong></td>{benchmark_cell}{depot_cells}<td style='font-weight:bold; background-color:#d3f8d3'>{avg_val}</td></tr>"

        # --- Dynamic Height ---
        num_rows = len(metric_map) + len(depots)
        iframe_height = min(1800, 400 + num_rows * 35)

        safe_headers = " ".join([f"<th>{pyhtml.escape(d)}</th>" for d in depots])
        table_html = f"""
        <!doctype html>
        <html>
        <head>
        <style>
          body {{ font-family: Arial, sans-serif; }}
          .custom-table {{margin:auto; border-collapse: collapse; width: 100%; border: 2px solid black;}}
          .custom-table th, .custom-table td {{border: 1px solid black; text-align: center; padding: 8px;}}
          .custom-table th {{background-color: #19bc9c; color: white;}}
          .yellow-bg {{background-color: yellow; font-weight: bold;}}
          #download-btn {{ margin-top: 12px; padding: 8px 14px; background-color:#19bc9c; color:white; border:none; border-radius:6px; cursor:pointer; }}
        </style>
        </head>
        <body>
        <div id="capture-area" style="text-align:center;">
            <h2 style="margin:0;">{pyhtml.escape(view_option)}</h2>
            <table class="custom-table">
                <thead><tr><th>Metric</th><th>Benchmark</th>{safe_headers}<th>Average</th></tr></thead>
                <tbody>{html_rows}</tbody>
            </table>
        </div>
        <div style="text-align:center; margin-top:12px;">
            <button id="download-btn">📥 Download KPI Table as PNG</button>
        </div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
document.getElementById("download-btn").onclick = function() {{
    const captureArea = document.getElementById("capture-area");
    const originalStyle = captureArea.style.height;
    captureArea.style.height = "auto";

    setTimeout(() => {{
        html2canvas(captureArea, {{
            scale: 2,
            useCORS: true,
            scrollX: 0,
            scrollY: -window.scrollY,
            windowWidth: document.body.scrollWidth,
            windowHeight: document.body.scrollHeight
        }}).then(canvas => {{
            const link = document.createElement('a');
            link.download = 'KPI_Table_Full.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
            captureArea.style.height = originalStyle;
        }}).catch(err => {{
            console.error('Capture failed:', err);
            alert('⚠️ Failed to capture full table. Check console for details.');
        }});
    }}, 300);
}};
</script>


        </body>
        </html>
        """

        components.html(table_html, height=iframe_height, scrolling=True)
        st.info("Green = meets benchmark or better, Red = above benchmark.", icon="ℹ")
        conn.close()


# --------- Run ----------
if __name__ == "__main__":
    ProdRatiosDashboard()
