import json, os
import streamlit as st
import pandas as pd
from datetime import timedelta
from mysql.connector import Error
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

# FIX: Ensure this import is present to define get_connection()
from auth import get_depot_settings
from db_config import get_session
from models import TSAdmin, InputData  # Add InputData if you have that ORM model


# Load config.json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)

def RM_sheet(user_region, role):
    st.title("TGSRTC PRODUCTIVITY DASHBOARD (REGIONAL MANAGER VIEW)")

    if not user_region:
        st.error("No region assigned. Please log in again.")
        st.stop()

    # --- fetch depots for region from ts_admin ---
    selection = st.radio("Select View Mode", ["INDIVIDUAL DEPOTS", "ALL DEPOT COMBINED DATA"], horizontal=True)
    if selection == "INDIVIDUAL DEPOTS":
        with get_session() as db:
            rm_depots = [r.depot_name for r in db.query(TSAdmin.depot_name)
                        .filter(TSAdmin.region == user_region)
                        .order_by(TSAdmin.depot_name)]


        if not rm_depots:
            st.error(f"No depots found for region {user_region}.")
            st.stop()

        selected_depot = st.selectbox("Select Depot", rm_depots)
        st.success(f"Depot: {selected_depot} (Region: {user_region})")

        # --- find latest available data_date for depot ---
        def get_latest_date(depot_name):
            try:
                with get_session() as db:
                    latest = db.query(InputData.data_date).filter(InputData.depot_name == depot_name).order_by(InputData.data_date.desc()).first()
                    return latest[0] if latest else None

            except:
                return None

        latest_date = get_latest_date(selected_depot)
        if latest_date is None:
            st.warning(f"No data found for depot {selected_depot}.")
            st.stop()

        latest_date = pd.to_datetime(latest_date).date()

        # --- date range filter ---
        st.subheader("📅 Select Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From Date", value=latest_date - timedelta(days=9))
        with col2:
            end_date = st.date_input("To Date", value=latest_date)

        if start_date > end_date:
            st.error("Start date cannot be after end date.")
            st.stop()

        # Build list of dates between start and end
        date_columns = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()

        # Depot Type
        depot_config = get_depot_settings()
        depot_type_from_admin = depot_config.get(selected_depot,{}).get("category","N/A")
        st.markdown(f"**Depot Type:** `{depot_type_from_admin}`")

        # Build DF with empty columns for chosen dates
        df = pd.DataFrame({"Category": config_data["category_rows"]})
        df["Depot Type"] = depot_type_from_admin
        for col_date in date_columns:
            df[col_date] = None  # create empty date columns

        # Fetch data
        def fetch_existing_data_for_dates(depot_name, date_columns):
            try:
                with get_session() as db:
                    data = db.query(InputData).filter(
                        InputData.depot_name == depot_name,
                        InputData.data_date.in_(date_columns)
                    ).all()
                    return pd.DataFrame([{k: v for k, v in d.__dict__.items() if not k.startswith("_")} for d in data])

            except Error as e:
                st.error(f"DB Error: {e}")
                return pd.DataFrame()

        existing = fetch_existing_data_for_dates(selected_depot, date_columns)
        if not existing.empty and "data_date" in existing.columns:
            existing["data_date"] = pd.to_datetime(existing["data_date"]).dt.strftime("%Y-%m-%d")
            reverse_map = {v:k for k,v in config_data["category_to_column"].items()}

            # fill df across selected date columns
            for col_date in date_columns:
                row = existing[existing["data_date"] == col_date]
                if row.empty: 
                    continue
                for db_col, val in row.iloc[0].items():
                    if db_col == "data_date" or pd.isna(val): 
                        continue
                    cat = reverse_map.get(db_col)
                    if cat:
                        idx = df[df["Category"].str.strip().str.lower() == cat.strip().lower()].index
                        if not idx.empty:
                            df.at[idx[0], col_date] = val

        # --- AgGrid Configuration (copied from DM style but view-only) ---
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(resizable=False, sortable=False, wrapText=False, autoHeight=False, editable=False)

        # Apply DM-style formatting for Category column
        gb.configure_column(
            field="Category",
            pinned="left",
            lockPinned=True,
            editable=False,
            width=150,
            resizable=False,
            sortable=False,
            wrapText=True,
            autoHeight=True,
            cellStyle=JsCode("""
                function(params) {
                    const category = params.value;
                    let style = {
                        "background-color": "#f0f0f0",
                        "font-weight": "bold",
                        "border": "1px solid #d3d3d3",
                        "white-space": "normal",
                        "word-break": "break-word"
                    };
                    if (category.startsWith('---')) {
                        style["background-color"] = "#cceeff";
                        style["font-weight"] = "bold";
                        style["text-align"] = "center";
                        style["font-size"] = "1.1em";
                        style["border-top"] = "2px solid #aaddff";
                        style["border-bottom"] = "1px solid #d3d3d3";
                    }
                    return style;
                }
            """),
            cellRenderer=JsCode("""
                class HtmlRenderer {
                    init(params) {
                        this.eGui = document.createElement('div');
                        if (params.value && typeof params.value === 'string' && params.value.includes('<')) {
                            this.eGui.innerHTML = params.value;
                        } else {
                            this.eGui.innerText = params.value;
                        }
                    }
                    getGui() { return this.eGui; }
                    refresh(params) {
                        if (params.value && typeof params.value === 'string' && params.value.includes('<')) {
                            this.eGui.innerHTML = params.value;
                        } else {
                            this.eGui.innerText = params.value;
                        }
                        return true;
                    }
                }
            """)
        )

        
        # Depot Type column styling (with benchmark display like DM, case-insensitive)
        normalized_depot_type = depot_type_from_admin.strip().title()
        current_bench = config_data["benchmarks"].get(normalized_depot_type, {})
        gb.configure_column(
            field="Depot Type",
            header_name="Rural/Urban",
            pinned="left",
            lockPinned=True,
            editable=False,
            width=120,
            resizable=False, 
            sortable=False,
            cellStyle=JsCode("""
                function(params) {
                    let style = {"background-color": "#f0f8ff", "border": "1px solid #d3d3d3"};
                    if (params.data.Category.startsWith('---')) {
                        style["background-color"] = "#cceeff";
                        style["font-weight"] = "bold";
                        style["text-align"] = "center";
                        style["font-size"] = "1.1em";
                    }
                    return style;
                }
            """),
            cellRenderer=JsCode(f"""
                function(params) {{
                    const category = params.data.Category;
                    const currentBenchmarks = {json.dumps(current_bench)};
                    let benchmarkValue = null;
                    const benchmarkCategoryKey = category;
                    if (currentBenchmarks.hasOwnProperty(benchmarkCategoryKey)) {{
                        benchmarkValue = currentBenchmarks[benchmarkCategoryKey];
                    }}
                    if (benchmarkValue !== null && category.includes('%')) {{
                        return 'Benchmark - ' + benchmarkValue + '%';
                    }} else if (category.startsWith('---')) {{
                        return params.value;
                    }} else {{
                        return '';
                    }}
                }}
            """)
        )

        # Apply DM-style cellStyle for date columns
        for col in date_columns:
            gb.configure_column(
                field=col,
                header_name=pd.to_datetime(col).strftime("%d-%b-%y"),
                editable=False,
                type=["numericColumn","rightAligned"],
                # REMOVE THE EXPLICIT WIDTH to allow horizontal scrolling
                # width=65, 
                resizable=False,
                sortable=False,
                cellStyle=JsCode("""
                    function(params) {
                        const category = params.data.Category;
                        const value = params.value;
                        let style = {"border": "1px solid #d3d3d3"};
                        if (category.startsWith('---')) {
                            style["background-color"] = "#cceeff";
                            style["font-weight"] = "bold";
                            style["text-align"] = "center";
                            style["font-size"] = "1.1em";
                        } else if (category === 'Schedules' || category === 'Schedules Services' || category === 'Schedules Kms') {
                            style["background-color"] = "#e6ffe6";
                            style["font-weight"] = "bold";
                        } else if (
                            category === 'Service Variance' ||
                            category === 'KM Variance' ||
                            category === 'Driver shortage' ||
                            category === 'Diff (MU Reasons)' ||
                            category === 'Diff (SL Reasons)' ||
                            category === 'Driver schedule' ||
                            category === 'Drivers on Duty' ||         
                            category === 'Driver for Bus Services' ||         
                            category === 'KM/Driver' ||         
                            category === 'Service/Driver Check'
                        ) {
                            style["background-color"] = "#fffacd";
                            style["font-weight"] = "bold";
                            if (value < 0) { style["color"] = "red"; }
                        } else if (category.includes('%')) {
                            style["background-color"] = "#e0e0f0";
                        }
                        return style;
                    }
                """)
            )

        grid_options = gb.build()

        # --- Final AgGrid (view-only) ---
        AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.NO_UPDATE,
            theme="material",
            height=700,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False,
            # ADD THIS FOR HORIZONTAL SCROLLING
            fit_columns_on_grid_load=False
        )
    if selection == "ALL DEPOT COMBINED DATA":
        # --- fetch all depots for region ---
            with get_session() as db:
                rm_depots = [r.depot_name for r in db.query(TSAdmin.depot_name)
                            .filter(TSAdmin.region == user_region)
                            .order_by(TSAdmin.depot_name)]


            if not rm_depots:
                st.error(f"No depots found for region {user_region}.")
                st.stop()

            # --- detect depot type for this region (all depots are same type) ---
            depot_config = get_depot_settings()
            first_depot_type = depot_config.get(rm_depots[0], {}).get("category", "N/A")
            normalized_region_type = first_depot_type.strip().title()
            current_bench = config_data["benchmarks"].get(normalized_region_type, {})

            st.markdown(f"**Region Type:** `{normalized_region_type}`")

            # --- find latest available date for region (any depot) ---
            def get_latest_region_date(dep_list):
                try:
                    with get_session() as db:
                        latest = db.query(InputData.data_date).filter(InputData.depot_name.in_(dep_list)).order_by(InputData.data_date.desc()).first()
                        return latest[0] if latest else None

                except:
                    return None

            latest_date = get_latest_region_date(rm_depots)
            if latest_date is None:
                st.warning(f"No data found for region {user_region}.")
                st.stop()

            latest_date = pd.to_datetime(latest_date).date()

            # --- date range filter ---
            st.subheader("📅 Select Date Range")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("From Date", value=latest_date - timedelta(days=9))
            with col2:
                end_date = st.date_input("To Date", value=latest_date)

            if start_date > end_date:
                st.error("Start date cannot be after end date.")
                st.stop()

            date_columns = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()

            # --- build empty df ---
            df = pd.DataFrame({"Category": config_data["category_rows"]})
            df["Depot Type"] = "Region Total"
            for col_date in date_columns:
                df[col_date] = None

            # --- fetch & aggregate data ---
            def fetch_region_data(dep_list, date_columns):
                try:
                    with get_session() as db:
                        data = (
                            db.query(InputData)
                            .filter(InputData.depot_name.in_(dep_list))
                            .filter(InputData.data_date.in_(date_columns))
                            .all()
                        )
                        return pd.DataFrame([{k: v for k, v in d.__dict__.items() if not k.startswith("_")} for d in data])
                except Exception as e:
                    st.error(f"DB Error: {e}")
                    return pd.DataFrame()


            existing = fetch_region_data(rm_depots, date_columns)
            if not existing.empty and "data_date" in existing.columns:
                existing["data_date"] = pd.to_datetime(existing["data_date"]).dt.strftime("%Y-%m-%d")
                reverse_map = {v:k for k,v in config_data["category_to_column"].items()}

                # group by date and sum all numeric cols
                grouped = existing.groupby("data_date").sum(numeric_only=True).reset_index()

                # fill df
                for col_date in date_columns:
                    row = grouped[grouped["data_date"] == col_date]
                    if row.empty:
                        continue
                    for db_col, val in row.iloc[0].items():
                        if db_col == "data_date" or pd.isna(val):
                            continue
                        cat = reverse_map.get(db_col)
                        if cat:
                            idx = df[df["Category"].str.strip().str.lower() == cat.strip().lower()].index
                            if not idx.empty:
                                df.at[idx[0], col_date] = val

            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(resizable=False, sortable=False, wrapText=False, autoHeight=False, editable=False)

            # Apply DM-style formatting for Category column
            gb.configure_column(
                field="Category",
                pinned="left",
                lockPinned=True,
                editable=False,
                width=150,
                resizable=False,
                sortable=False,
                wrapText=True,
                autoHeight=True,
                cellStyle=JsCode("""
                    function(params) {
                        const category = params.value;
                        let style = {
                            "background-color": "#f0f0f0",
                            "font-weight": "bold",
                            "border": "1px solid #d3d3d3",
                            "white-space": "normal",
                            "word-break": "break-word"
                        };
                        if (category.startsWith('---')) {
                            style["background-color"] = "#cceeff";
                            style["font-weight"] = "bold";
                            style["text-align"] = "center";
                            style["font-size"] = "1.1em";
                            style["border-top"] = "2px solid #aaddff";
                            style["border-bottom"] = "1px solid #d3d3d3";
                        }
                        return style;
                    }
                """),
                cellRenderer=JsCode("""
                    class HtmlRenderer {
                        init(params) {
                            this.eGui = document.createElement('div');
                            if (params.value && typeof params.value === 'string' && params.value.includes('<')) {
                                this.eGui.innerHTML = params.value;
                            } else {
                                this.eGui.innerText = params.value;
                            }
                        }
                        getGui() { return this.eGui; }
                        refresh(params) {
                            if (params.value && typeof params.value === 'string' && params.value.includes('<')) {
                                this.eGui.innerHTML = params.value;
                            } else {
                                this.eGui.innerText = params.value;
                            }
                            return true;
                        }
                    }
                """)
            )

            # Depot Type column styling (with benchmark display like DM)
            gb.configure_column(
                field="Depot Type",
                header_name="Rural/Urban",
                pinned="left",
                lockPinned=True,
                editable=False,
                width=150,
                resizable=False,
                sortable=False,
                cellStyle=JsCode("""
                    function(params) {
                        let style = {"background-color": "#f0f8ff", "border": "1px solid #d3d3d3"};
                        if (params.data.Category.startsWith('---')) {
                            style["background-color"] = "#cceeff";
                            style["font-weight"] = "bold";
                            style["text-align"] = "center";
                            style["font-size"] = "1.1em";
                        }
                        return style;
                    }
                """),
                cellRenderer=JsCode(f"""
                    function(params) {{
                        const category = params.data.Category;
                        const currentBenchmarks = {json.dumps(current_bench)};
                        let benchmarkValue = null;
                        if (currentBenchmarks.hasOwnProperty(category)) {{
                            benchmarkValue = currentBenchmarks[category];
                        }}
                        if (benchmarkValue !== null && category.includes('%')) {{
                            return 'Benchmark - ' + benchmarkValue + '%';
                        }} else if (category.startsWith('---')) {{
                            return params.value;
                        }} else {{
                            return ' ';
                        }}
                    }}
                """)
            )

            # Apply DM-style cellStyle for date columns
            for col in date_columns:
                gb.configure_column(
                    field=col,
                    header_name=pd.to_datetime(col).strftime("%d-%b-%y"),
                    editable=False,
                    type=["numericColumn","rightAligned"],
                    resizable=False,
                    sortable=False,
                    cellStyle=JsCode("""
                        function(params) {
                            const category = params.data.Category;
                            const value = params.value;
                            let style = {"border": "1px solid #d3d3d3"};
                            if (category.startsWith('---')) {
                                style["background-color"] = "#cceeff";
                                style["font-weight"] = "bold";
                                style["text-align"] = "center";
                                style["font-size"] = "1.1em";
                            } else if (category === 'Schedules' || category === 'Schedules Services' || category === 'Schedules Kms') {
                                style["background-color"] = "#e6ffe6";
                                style["font-weight"] = "bold";
                            } else if (
                                category === 'Service Variance' ||
                                category === 'KM Variance' ||
                                category === 'Driver shortage' ||
                                category === 'Diff (MU Reasons)' ||
                                category === 'Diff (SL Reasons)' ||
                                category === 'Driver schedule' ||
                                category === 'Drivers on Duty' ||         
                                category === 'Driver for Bus Services' ||         
                                category === 'KM/Driver' ||         
                                category === 'Service/Driver Check'
                            ) {
                                style["background-color"] = "#fffacd";
                                style["font-weight"] = "bold";
                                if (value < 0) { style["color"] = "red"; }
                            } else if (category.includes('%')) {
                                style["background-color"] = "#e0e0f0";
                            }
                            return style;
                        }
                    """)
                )

            grid_options = gb.build()

            # --- Final AgGrid (view-only) ---
            AgGrid(
                df,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.NO_UPDATE,
                theme="material",
                height=700,
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False,
                fit_columns_on_grid_load=False
            )
