import streamlit as st
import pandas as pd
from sqlalchemy import extract
from datetime import date
from db_config import get_session
from models import ActionPlan


# ---------------------- MAIN FUNCTION ----------------------
def action_plan_history_rm():

    st.title("📊 Action Plan History (RM)")

    # ---------------------- Depot from session ----------------------
    selected_depot = "NIRMAL"  # Example depot; replace with session-based depot if available

    # ---------------------- YEAR SELECTION ----------------------
    current_year = date.today().year
    years = list(range(current_year - 5, current_year + 1))  # last 5 years

    selected_year = st.selectbox("Select Year", years, index=len(years) - 1)

    st.markdown("---")
    st.subheader("📅 Action Plan History Table")

    # ---------------------- FETCH HISTORY ----------------------
    try:
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

        # ---------------------- FORMAT HISTORY ----------------------
        data = []
        for r in records:
            month = r.data_date.month

            # determine quarter
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
                "DEPOT": r.depot_name,
                "Weekly Off & National Off": r.Weekly_Off_National_Off,
                "Special Off (Night Out IC & Online)": r.Special_Off_Night_Out_IC_Online,
                "Others": r.Other_s,
                "Leave & Absent": r.Leave_Absent,
                "Sick Leave": r.Sick_Leave,
                "Spot Absent": r.Spot_Absent,
                "Double Duty": r.Double_Duty,
                "Off Cancellation": r.Off_Cancellation,
            })

        # ---------------------- DISPLAY TABLE ----------------------
        if data:
            df = pd.DataFrame(data)
            df = df[[
                "Date", "DEPOT", "Quarter",
                "Weekly Off & National Off",
                "Special Off (Night Out IC & Online)",
                "Others", "Leave & Absent", "Sick Leave",
                "Spot Absent", "Double Duty", "Off Cancellation"
            ]]
            st.dataframe(df, use_container_width=True, hide_index=True)

        else:
            st.info("ℹ️ No Action Plan entries found for this year.")

    except Exception as e:
        st.error(f"Error fetching history: {e}")


# ---------------------- RUN ----------------------
if __name__ == "__main__":
    action_plan_history_rm()
