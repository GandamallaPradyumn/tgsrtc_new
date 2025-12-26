import streamlit as st
from datetime import date
from sqlalchemy import func, extract
import json
import pandas as pd
from db_config import get_session
from models import ActionPlan, InputData, TSAdmin  # ensure TSAdmin model exists


# ---------------------- Load Benchmarks from config.json ----------------------
with open("config.json") as f:
    CONFIG = json.load(f)

BENCHMARKS = CONFIG.get("benchmarks", {})

import requests

BACKEND_URL = "http://localhost:8000"

def fetch_ai_summary(depot, kpi_name, from_date, to_date):
    payload = {
        "depot": depot,
        "kpi": kpi_name,
        "from_date": from_date,
        "to_date": to_date
    }
    r = requests.post(f"{BACKEND_URL}/ask_kpi", json=payload, timeout=60)
    if r.status_code == 200:
        return r.json().get("summary")
    return "AI summary unavailable."

# ---------------------- ORM Helpers ----------------------
def whitelist_columns(data_dict):
    allowed = {
        "Weekly_Off_National_Off",
        "Special_Off_Night_Out_IC_Online",
        "Other_s",
        "Leave_Absent",
        "Sick_Leave",
        "Spot_Absent",
        "Double_Duty",
        "Off_Cancellation",
        "Driver_Schedule"
    }
    return {k: v for k, v in data_dict.items() if k in allowed}

import streamlit.components.v1 as components

def show_toast(message: str, duration: int = 3000):
    """Display a floating toast message (duration in ms)."""
    escaped = message.replace('"', '\\"')
    html = f"""
    <div id="st_toast" style="
      position: fixed;
      right: 16px;
      top: 16px;
      z-index: 9999;
      font-family: sans-serif;
    ">
      <div style="
        background: rgba(33, 150, 83, 0.95);
        color: white;
        padding: 10px 16px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        font-size: 14px;
      ">{escaped}</div>
    </div>
    <script>
      setTimeout(function() {{
        const el = parent.document.getElementById("st_toast_iframe");
        if (el) {{
          el.remove();
        }} else {{
          const t = document.getElementById("st_toast");
          if (t) t.style.display = "none";
        }}
      }}, {duration});
    </script>
    """
    # render a tiny iframe hosting the toast HTML so it floats above the app
    components.html(html, height=0, width=0, scrolling=False)

def fetch_existing(depot, action_date):
    try:
        with get_session() as db:
            record = (
                db.query(ActionPlan)
                .filter(ActionPlan.depot_name == depot, ActionPlan.data_date == action_date)
                .first()
            )
            if record:
                return {col.name: getattr(record, col.name) for col in ActionPlan.__table__.columns}
            return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def insert_or_update_action(depot, action_date, data_dict):
    data_dict = whitelist_columns(data_dict)
    if not data_dict:
        return False, "No valid fields provided."

    try:
        with get_session() as db:
            record = (
                db.query(ActionPlan)
                .filter(ActionPlan.depot_name == depot, ActionPlan.data_date == action_date)
                .first()
            )
            if record:
                for key, value in data_dict.items():
                    setattr(record, key, value)
                msg = "✅ Record updated successfully!"
            else:
                new_record = ActionPlan(depot_name=depot, data_date=action_date, **data_dict)
                db.add(new_record)
                msg = "✅ New record added successfully!"
            db.commit()
            return True, msg
    except Exception as e:
        return False, f"Database error: {e}"


def fetch_monthly_avg(depot, selected_date):
    """Fetch average % values for the selected month from input_data."""
    month = selected_date.month
    year = selected_date.year

    with get_session() as db:
        result = (
        db.query(
            func.avg(InputData.Pct_Weekly_Off_National_Off),
            func.avg(InputData.Pct_Special_Off_Night_Out_IC_Online),
            func.avg(InputData.Pct_Others),
            func.avg(InputData.Pct_Leave_Absent),
            func.avg(InputData.Pct_Sick_Leave),
            func.avg(InputData.Pct_Spot_Absent),
            func.avg(InputData.Pct_Double_Duty),
            func.avg(InputData.Pct_Off_Cancellation),
            (
                func.sum(InputData.Total_Drivers) /
                func.nullif(func.sum(InputData.Planned_Schedules), 0)
            ).label("driver_schedule_ratio"),
        )
        .filter(
            InputData.depot_name == depot,
            extract("year", InputData.data_date) == year,
            extract("month", InputData.data_date) == month,
        )
        .first()
    )


    if result:
        return {
            "Weekly_Off_National_Off": round(result[0] or 0, 2),
            "Special_Off_Night_Out_IC_Online": round(result[1] or 0, 2),
            "Other_s": round(result[2] or 0, 2),
            "Leave_Absent": round(result[3] or 0, 2),
            "Sick_Leave": round(result[4] or 0, 2),
            "Spot_Absent": round(result[5] or 0, 2),
            "Double_Duty": round(result[6] or 0, 2),
            "Off_Cancellation": round(result[7] or 0, 2),
            "Driver_Schedule": round(result[8] or 0, 1),
        }
    return {}


def fetch_depot_category(depot_name: str):
    """Fetch depot's category (Rural/Urban) from ts_admin table."""
    cache_key = f"depot_category_{depot_name}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        with get_session() as db:
            record = db.query(TSAdmin).filter(TSAdmin.depot_name == depot_name).first()
            if record and record.category:
                cat = record.category.strip().title()  # 'Rural' or 'Urban'
                st.session_state[cache_key] = cat
                return cat
            else:
                return None
    except Exception as e:
        st.warning(f"Could not fetch category for {depot_name}: {e}")
        return None


# ---------------------- Utility: Last filled per quarter ----------------------
quarter_map = {"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}


def get_last_filled_for_quarter(depot: str, selected_year: int, q: str):
    start_m, end_m = quarter_map[q]
    try:
        with get_session() as db:
            record = (
                db.query(ActionPlan)
                .filter(
                    ActionPlan.depot_name == depot,
                    ActionPlan.data_date.between(date(selected_year, start_m, 1), date(selected_year, end_m, 31)),
                )
                .order_by(ActionPlan.data_date.desc())
                .first()
            )
            return record.data_date.strftime("%Y-%m-%d") if record else "—"
    except Exception:
        return "—"


# ---------------------- Streamlit UI ----------------------
def action_dm():
    st.title("📘 KPI's Action Plan")

    selected_depot = st.session_state.get("user_depot")
    if not selected_depot:
        st.error("Depot not found in session. Please log in again.")
        st.stop()

    category = fetch_depot_category(selected_depot) or "Rural"
    st.success(f"Depot: **{selected_depot}**  |  Category: **{category}** 🚏")

    # Year / Quarter / Date
    col5, col6, col7 = st.columns(3)
    with col5:
        current_year = date.today().year
        years = list(range(current_year - 3, current_year + 2))
        selected_year = st.selectbox("Select Year", years, index=years.index(current_year))
    with col6:
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        selected_quarter = st.selectbox("Select Quarter", quarters, index=(date.today().month - 1) // 3)
    with col7:
    # Define quarter ranges
        quarter_ranges = {
            "Q1": (date(selected_year, 1, 1), date(selected_year, 3, 31)),
            "Q2": (date(selected_year, 4, 1), date(selected_year, 6, 30)),
            "Q3": (date(selected_year, 7, 1), date(selected_year, 9, 30)),
            "Q4": (date(selected_year, 10, 1), date(selected_year, 12, 31)),
        }

        q_start, q_end = quarter_ranges[selected_quarter]

        # Ensure selected date is within the quarter
        default_date = date.today()
        if not (q_start <= default_date <= q_end):
            default_date = q_start  # fallback to quarter start if today is outside range

        action_date = st.date_input(
            f"Select Date (within {selected_quarter})",
            value=default_date,
            min_value=q_start,
            max_value=q_end,
        )



    # Last filled info per quarter
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"Last filled in Q1: {get_last_filled_for_quarter(selected_depot, selected_year, 'Q1')}")
    with col2:
        st.info(f"Last filled in Q2: {get_last_filled_for_quarter(selected_depot, selected_year, 'Q2')}")
    with col3:
        st.info(f"Last filled in Q3: {get_last_filled_for_quarter(selected_depot, selected_year, 'Q3')}")
    with col4:
        st.info(f"Last filled in Q4: {get_last_filled_for_quarter(selected_depot, selected_year, 'Q4')}")


    monthly_avg = fetch_monthly_avg(selected_depot, action_date)
    benchmark_zone = BENCHMARKS.get(category, {})
    BACKEND_KPI_NAMES = {
    "Weekly_Off_National_Off": "Weekly Off",
    "Special_Off_Night_Out_IC_Online": "Special Off (Night Out/IC, Online)",
    "Other_s": "Others",
    "Leave_Absent": "Long Leave & Absent",
    "Sick_Leave": "Sick Leave",
    "Spot_Absent": "Spot Absent",
    "Double_Duty": "Double Duty",
    "Off_Cancellation": "Off Cancellation",
    "Driver_Schedule": "Drivers/Schedule",
}

    existing = fetch_existing(selected_depot, action_date)
    def get_val(col): return existing.get(col, "") if existing else ""
    
    kpi_map = {
        "Weekly_Off_National_Off": "% Weekly Off & National Off",
        "Special_Off_Night_Out_IC_Online": "% Special Off (Night Out/IC, Online)",
        "Other_s": "% Others",
        "Leave_Absent": "% Leave & Absent",
        "Sick_Leave": "% Sick Leave",
        "Spot_Absent": "% Spot Absent",
        "Double_Duty": "% Double Duty",
        "Off_Cancellation": "% Off Cancellation",
        "Driver_Schedule": "Drivers/Schedule"
    }
    
    # ---------------------- AI Summary Generator ----------------------


    def kpi_with_ai(db_key, label):
        avg_val = monthly_avg.get(db_key)
        bench_val = benchmark_zone.get(kpi_map.get(db_key))

        avg_str = f"{avg_val:.2f}%" if avg_val is not None else "—"
        bench_str = f"{bench_val:.2f}%" if bench_val is not None else "—"

        col_text, col_btn = st.columns([9, 1])

        with col_text:
            st.markdown(
                f"""
                **{label}**  
                🟢 Avg for selected month: **{avg_str}** | 🎯 Benchmark: **{bench_str}**
                """
            )

        with col_btn:
            if st.button("🤖 AI", key=f"ai_{db_key}"):
                with st.spinner("Generating AI insight..."):
                    try:
                        from_date = action_date.replace(day=1).isoformat()
                        to_date = action_date.isoformat()

                        summary = fetch_ai_summary(
                            depot=selected_depot,
                            kpi_name = BACKEND_KPI_NAMES[db_key],  # ✅ FIX
                            from_date=from_date,
                            to_date=to_date
                        )

                        st.session_state[f"text_{db_key}"] = summary
                        st.rerun()

                    except Exception:
                        st.session_state[f"text_{db_key}"] = (
                            "AI summary could not be generated."
                        )
                        st.rerun()

        return st.text_area(
            label="",
            key=f"text_{db_key}",
            height=90,
            placeholder="Enter action plan or click 🤖 AI for suggestion..."
        )



    # Initialize AI text state (one-time)
    for k in kpi_map.keys():
        st.session_state.setdefault(f"text_{k}", get_val(k))

    st.write("### ✍️ Enter Action Plan Details")

    fields = {
            "Weekly_Off_National_Off": kpi_with_ai(
                "Weekly_Off_National_Off", "Weekly Off & National Off"
            ),
            "Special_Off_Night_Out_IC_Online": kpi_with_ai(
                "Special_Off_Night_Out_IC_Online", "Special Off (Night Out IC & Online)"
            ),
            "Other_s": kpi_with_ai("Other_s", "Others"),
            "Leave_Absent": kpi_with_ai("Leave_Absent", "Leave & Absent"),
            "Sick_Leave": kpi_with_ai("Sick_Leave", "Sick Leave"),
            "Spot_Absent": kpi_with_ai("Spot_Absent", "Spot Absent"),
            "Double_Duty": kpi_with_ai("Double_Duty", "Double Duty"),
            "Off_Cancellation": kpi_with_ai("Off_Cancellation", "Off Cancellation"),
            "Driver_Schedule": kpi_with_ai("Driver_Schedule", "Driver/Schedule"),
        }
    with st.form("entry_form", clear_on_submit=True):
        submitted = st.form_submit_button("💾 Save")
        if submitted:
            clean_data = {k: v.strip() for k, v in fields.items() if v and v.strip()}
            if not clean_data:
                st.warning("⚠ Please enter at least one valid field.")
                return
            success, msg = insert_or_update_action(selected_depot, action_date, clean_data)
            if success:
                show_toast("Record saved ✅", duration=300)
                st.success(msg) 
            else:
                st.error(msg)
        # ---------------------- 📊 Action Plan History ----------------------
    st.markdown("---")
    st.subheader("📅 Action Plan History")

    try:
        import pandas as pd
        with get_session() as db:
            records = (
                db.query(ActionPlan)
                .filter(
                    ActionPlan.depot_name == selected_depot,
                    extract("year", ActionPlan.data_date) == selected_year,
                )
                .order_by(ActionPlan.data_date.desc())
                .all()
            )

            # Convert ORM to plain dicts (safe before session closes)
            data = []
            for r in records:
                month = r.data_date.month
                # Determine quarter based on month
                if 1 <= month <= 3:
                    quarter = "Q1"
                elif 4 <= month <= 6:
                    quarter = "Q2"
                elif 7 <= month <= 9:
                    quarter = "Q3"
                else:
                    quarter = "Q4"

                data.append({
                    "Date": r.data_date.strftime("%Y-%m-%d"),
                    "Quarter": quarter,
                    "Weekly Off & National Off": r.Weekly_Off_National_Off,
                    "Special Off (Night Out IC & Online)": r.Special_Off_Night_Out_IC_Online,
                    "Others": r.Other_s,
                    "Leave & Absent": r.Leave_Absent,
                    "Sick Leave": r.Sick_Leave,
                    "Spot Absent": r.Spot_Absent,
                    "Double Duty": r.Double_Duty,
                    "Off Cancellation": r.Off_Cancellation,
                    "Driver/Schedule": r.Driver_Schedule
                })

        # Convert to DataFrame and display
        if data:
            df = pd.DataFrame(data)
            # Reorder columns so Quarter appears right after Date
            df = df[["Date", "Quarter", "Weekly Off & National Off",
                     "Special Off (Night Out IC & Online)", "Others",
                     "Leave & Absent", "Sick Leave", "Spot Absent",
                     "Double Duty", "Off Cancellation","Driver/Schedule"]]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No previous Action Plan entries found for this year.")

    except Exception as e:
        st.error(f"Error fetching history: {e}")
