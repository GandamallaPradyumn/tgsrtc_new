import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import func
import json

from db_config import get_session
from models import TSAdmin, InputData

# ---------- Load Config ----------
with open("config.json") as f:
    config = json.load(f)


# ---------- ORM Query ----------
def get_depot_status():
    """
    Fetches latest data date per depot using SQLAlchemy ORM.
    Calculates pending days and status.
    """
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)

    with get_session() as db:
        results = (
            db.query(
                TSAdmin.zone.label("zone"),
                TSAdmin.region.label("region"),
                TSAdmin.depot_name.label("depot"),
                func.max(InputData.data_date).label("latest_date"),
            )
            .outerjoin(InputData, TSAdmin.depot_name == InputData.depot_name)
            .group_by(TSAdmin.zone, TSAdmin.region, TSAdmin.depot_name)
            .order_by(TSAdmin.zone, TSAdmin.region, TSAdmin.depot_name)
            .all()
        )

    df = pd.DataFrame(results, columns=["zone", "region", "depot", "latest_date"])

    if df.empty:
        return pd.DataFrame(columns=["zone", "region", "depot", "LAST DATE UPDATED", "STATUS", "NO. OF DAYS PENDING"])

    # Format columns
    df["latest_date"] = pd.to_datetime(df["latest_date"], errors="coerce").dt.date

    df["LAST DATE UPDATED"] = df["latest_date"].apply(
        lambda d: d.strftime("%d-%m-%Y") if pd.notnull(d) else "--"
    )

    df["NO. OF DAYS PENDING"] = df["latest_date"].apply(
        lambda d: (yesterday - d).days if pd.notnull(d) else None
    )

    df["STATUS"] = df["NO. OF DAYS PENDING"].apply(
        lambda x: "Updated" if x == 0 else "Pending"
    )

    return df[
        ["zone", "region", "depot", "LAST DATE UPDATED", "STATUS", "NO. OF DAYS PENDING"]
    ]


# ---------- Custom HTML Renderer ----------
def render_merged_table(df):
    html = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            text-align: center;
            font-family: Arial, sans-serif;
            font-size: 15px;
        }
        th {
            background-color: #f2f2f2;
            padding: 8px;
            border: 1px solid #ddd;
        }
        td {
            border: 1px solid #ddd;
            padding: 6px;
        }
        tr:nth-child(even) { background-color: #fafafa; }
    </style>
    <table>
    <tr>
    """

    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>"

    zone_count = df["zone"].value_counts()
    region_count = df.groupby(["zone"])["region"].value_counts()
    zone_seen, region_seen = {}, {}

    for _, row in df.iterrows():
        html += "<tr>"
        if row["zone"] not in zone_seen:
            html += f"<td rowspan='{zone_count[row['zone']]}'><b>{row['zone']}</b></td>"
            zone_seen[row["zone"]] = True
        if (row["zone"], row["region"]) not in region_seen:
            html += f"<td rowspan='{region_count[row['zone']][row['region']]}'><b>{row['region']}</b></td>"
            region_seen[(row["zone"], row["region"])] = True

        html += f"<td>{row['depot']}</td>"
        html += f"<td>{row['LAST DATE UPDATED']}</td>"
        if "Pending" in row["STATUS"]:
            html += f"<td style='color:red; font-weight:bold;'>{row['STATUS']}</td>"
        else:
            html += f"<td style='color:green; font-weight:bold;'>{row['STATUS']}</td>"
        html += f"<td>{row['NO. OF DAYS PENDING'] if row['NO. OF DAYS PENDING'] is not None else '--'}</td>"
        html += "</tr>"

    html += "</table>"
    return html


# ---------- Streamlit UI ----------
def pending_depot():
    st.title("📋 Pending Status by Zone / Region / Depot")

    try:
        df = get_depot_status()
        if df.empty:
            st.warning("No data found.")
            return

        # CSV Download Button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv,
            file_name="pending_depots.csv",
            mime="text/csv",
        )

        # Render table
        st.markdown(render_merged_table(df), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error fetching data: {e}")


# ---------- Run ----------
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Pending Depots Dashboard 🕒")
    pending_depot()
