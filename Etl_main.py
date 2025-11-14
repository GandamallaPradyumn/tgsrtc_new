import streamlit as st
import pandas as pd
import importlib
import json
import ast
from utils import get_mysql_engine, insert_to_mysql  # ORM-based helper functions


def run_etl_dashboard():
    # ----------------------------
    # 🔐 Load DB Credentials
    # ----------------------------
    with open("config.json", "r") as f:
        config = json.load(f)
    # ----------------------------
    # 📁 Upload CSV File
    # ----------------------------
    uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

    # ----------------------------
    # 🎛️ Control Bar (All in One Line)
    # ----------------------------
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

    with col1:
        dataset_type = st.selectbox(
            "Select Dataset Type",
            ["Operational Data", "Leave & Absent", "Driver Details", "Service Master"]
        )
    with col2:
        preview_btn = st.button("🔍 Preview CSV")
    with col3:
        transform_btn = st.button("⚙️ Transform Data")
    with col4:
        load_btn = st.button("🚀 Load into Database")
    with col5:
        reset_btn = st.button("♻️ Reset Transform")

    # ----------------------------
    # 🧠 Session State Initialization
    # ----------------------------
    for key in ["original_df", "raw_df", "transformed_df", "target_table", "validation_report"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "validation_report" else {}

    # ----------------------------
    # 📂 File Handling
    # ----------------------------
    if uploaded_file:

        # Load original CSV once
        if st.session_state.original_df is None:
            st.session_state.original_df = pd.read_csv(uploaded_file)
            st.session_state.raw_df = st.session_state.original_df.copy()

        # ----------------------------
        # ♻️ Reset
        # ----------------------------
        if reset_btn:
            st.session_state.transformed_df = None
            st.session_state.raw_df = st.session_state.original_df.copy()
            st.session_state.target_table = None
            st.session_state.validation_report = {}
            st.success("🔄 Transformation has been reset to the original uploaded CSV.")
            st.stop()

        # ----------------------------
        # 🔍 Preview Button
        # ----------------------------
        if preview_btn:
            if st.session_state.transformed_df is not None:
                st.subheader("🧮 Transformed Data Preview (First 10 Rows)")
                st.dataframe(st.session_state.transformed_df.head(10))
            else:
                st.subheader("🧾 Raw Data Preview (First 10 Rows)")
                st.dataframe(st.session_state.raw_df.head(10))

        # ----------------------------
        # ⚙️ Transform Button
        # ----------------------------
        if transform_btn:
            st.subheader("⚙️ Applying Transformations...")

            module_map = {
                "Operational Data": "operational_data",
                "Leave & Absent": "leave_absent",
                "Driver Details": "driver_details",
                "Service Master": "service_master",
            }

            module_name = module_map.get(dataset_type)

            if module_name:
                transformer = importlib.import_module(module_name)

                try:
                    result = transformer.transform(st.session_state.raw_df.copy())

                    # Handle different return structures
                    if isinstance(result, tuple) and len(result) == 3:
                        transformed_df, target_table, validation_report = result
                    else:
                        transformed_df, target_table = result
                        validation_report = {}

                    # Save to session
                    st.session_state.transformed_df = transformed_df
                    st.session_state.target_table = target_table
                    st.session_state.validation_report = validation_report

                    # Handle warnings
                    unmapped = validation_report.get("unmapped_depots", [])
                    missing_data = {
                        k: v for k, v in validation_report.items() if k != "unmapped_depots"
                    }

                    if unmapped:
                        st.warning(
                            f"⚠️ Unmapped depots found ({len(unmapped)}): "
                            + ", ".join(unmapped[:10])
                            + (" ..." if len(unmapped) > 10 else "")
                        )

                    if missing_data:
                        st.error("❌ Missing values detected in transformed data:")
                        for col, count in missing_data.items():
                            st.markdown(f"- **Column:** `{col}` → Missing Rows: {count}")
                    else:
                        st.success(f"✅ Transformation complete for {dataset_type} dataset!")
                        st.dataframe(transformed_df.head(10))

                except ValueError as e:
                    error_msg = str(e)
                    if "Missing values detected" in error_msg:
                        st.error("❌ Missing values detected:")
                        try:
                            missing_info = ast.literal_eval(error_msg.split(":", 1)[1].strip())
                            for col, count in missing_info.items():
                                st.markdown(f"- **Column:** `{col}` → Missing Rows: {count}")
                        except Exception:
                            st.text(error_msg)
                    else:
                        st.error(f"⚠️ Transformation failed: {error_msg}")

                except Exception as e:
                    st.error(f"❌ Unexpected error during transformation: {e}")

            else:
                st.error("⚠️ Invalid dataset type selected.")

        # ----------------------------
        # 🚀 Load Button
        # ----------------------------
        if load_btn:
            if st.session_state.transformed_df is None:
                st.warning("⚠️ Please transform your data before loading.")
            else:
                transformed_df = st.session_state.transformed_df
                target_table = st.session_state.target_table

                if target_table is None:
                    st.error("❌ Target table not defined. Transform again.")
                else:
                    engine = get_mysql_engine(config)
                    if engine:
                        with st.spinner("⏳ Loading data into MySQL..."):
                            insert_to_mysql(engine, transformed_df, target_table)
                        st.success(f"✅ Successfully inserted data into `{target_table}` table!")

    else:
        st.info("📥 Please upload a CSV file to start.")
