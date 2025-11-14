import streamlit as st
import pandas as pd
import json
import re
from db_config import get_session
from models import TSAdmin  # ORM model for ts_admin table

# -------------------- Streamlit Admin Function --------------------
def admin():
    # Load config for reference
    with open("config.json") as f:
        config = json.load(f)
    db_config = config.get("db", {})

    # -------------------- Helper: Fetch All Depots --------------------
    def get_all_depots():
        """Fetch all depots using ORM (safe, read-only)."""
        try:
            with get_session() as db:
                depots = db.query(TSAdmin).all()
                data = [{"Depot Name": d.depot_name, "Category": d.category} for d in depots]
                return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error fetching depots: {e}")
            return pd.DataFrame()

    # -------------------- Helper: Add or Update Depot --------------------
    def add_or_update_depot(name, category):
        """Add or update depot entry securely via ORM."""
        if not re.match(r"^[A-Za-z0-9\s\-_]+$", name):
            st.error("❌ Invalid depot name. Use only letters, numbers, spaces, or hyphens.")
            return

        try:
            with get_session() as db:
                depot_obj = db.query(TSAdmin).filter(TSAdmin.depot_name == name).first()
                if depot_obj:
                    depot_obj.category = category
                    msg = f"✅ Updated category for '{name}'"
                else:
                    depot_obj = TSAdmin(
                        zone="Default Zone",
                        region="Default Region",
                        depot_name=name.strip(),
                        category=category.strip(),
                    )
                    db.add(depot_obj)
                    msg = f"✅ Added new depot '{name}'"

                db.commit()
                st.success(msg)
        except Exception as e:
            st.error(f"Database error while saving depot: {e}")

    # -------------------- Streamlit UI --------------------
    st.markdown(
        "<h1 style='text-align: center; color: BLACK;'>✍️ Add or Update Depot Settings</h1>",
        unsafe_allow_html=True
    )

    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            depot_name = st.text_input("🏢 Depot Name")
            category = st.selectbox("🏷️ Depot Type", ["Select Category", "Rural", "Urban"])

        if st.button("💾 Save Depot Settings"):
            if depot_name and category != "Select Category":
                add_or_update_depot(depot_name, category)
            else:
                st.warning("⚠️ Please enter both depot name and valid category.")

        st.markdown("### 📋 All Depots")
        df = get_all_depots()
        st.dataframe(df, use_container_width=True)
