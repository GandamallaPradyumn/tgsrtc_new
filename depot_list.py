import streamlit as st
import json
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import func

from db_config import get_session
from models import TSAdmin, InputData, User

# -------------------- Load config --------------------
with open("config.json") as f:
    config = json.load(f)


# -------------------- ORM Fetch Logic --------------------
def get_depot_status():
    """
    Optimized ORM query to fetch latest data date per depot,
    and mark them as Updated or Pending.
    """
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)

    with get_session() as db:
        query = (
            db.query(
                TSAdmin.region.label("region"),
                TSAdmin.depot_name.label("depot"),
                func.max(InputData.data_date).label("latest_date"),
            )
            .outerjoin(InputData, TSAdmin.depot_name == InputData.depot_name)
            .group_by(TSAdmin.region, TSAdmin.depot_name)
            .order_by(TSAdmin.region, TSAdmin.depot_name)
        )
        results = query.all()

    df = pd.DataFrame(results, columns=["region", "depot", "latest_date"])
    if df.empty:
        return pd.DataFrame(columns=["region", "depot", "latest_date", "STATUS", "DATE_TEXT"])

    df["latest_date"] = pd.to_datetime(df["latest_date"], errors="coerce").dt.date
    df["STATUS"] = df["latest_date"].apply(
        lambda d: "Updated" if pd.notnull(d) and (d == yesterday or d == today) else "Pending"
    )
    df["DATE_TEXT"] = df["latest_date"].apply(
        lambda d: f"({d.strftime('%d-%m-%Y')})" if pd.notnull(d) else ""
    )

    return df


# -------------------- Streamlit UI --------------------
def depotlist():
    """
    Displays depot status grouped by region — same look as your screenshot.
    """
    st.subheader("📋 Region-wise Depot Status")

    try:
        df = get_depot_status()
        if df.empty:
            st.warning("No data found.")
            return

        regions = df.groupby("region")

        for region, group in regions:
            st.markdown(f"##  {region}")

            # --- Column Headers ---
            col1, _, col2 = st.columns([1, 0.2, 1])
            with col1:
                st.markdown("###  *Updated Depots*")
            with col2:
                st.markdown("###  *Not Updated Depots*")
            st.markdown("---")

            # --- Updated / Pending Lists ---
            updated = group[group["STATUS"] == "Updated"]["depot"].tolist()
            pending = group[group["STATUS"] == "Pending"]

            row_col1, _, row_col2 = st.columns([1, 0.2, 1])

            with row_col1:
                if updated:
                    for depot in updated:
                        st.markdown(
                            f"<p style='color:green;font-weight:bold;font-size:18px;'>{depot}</p>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown("<p style='font-size:18px;'>--</p>", unsafe_allow_html=True)

            with row_col2:
                if not pending.empty:
                    for _, r in pending.iterrows():
                        st.markdown(
                            f"<p style='color:red;font-weight:bold;font-size:18px;'>{r['depot']} {r['DATE_TEXT']}</p>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown("<p style='color:red;font-weight:bold;font-size:18px;'>--</p>", unsafe_allow_html=True)

            st.markdown("---")

    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")


# -------------------- Run the App --------------------
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Depot Status Dashboard 🚌")
    depotlist()
