import json
import os

def user_sheet(user_depot, role):
    import streamlit as st
    import pandas as pd
    from datetime import date, timedelta, datetime
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    from db_config import get_session
    from models import InputData, TSAdmin
    from st_aggrid.shared import JsCode
    from auth import get_depot_settings
    import re

    # --- Page Configuration ---
    st.markdown("""
    <style>
    .ag-theme-material .ag-cell-edit-input,
    .ag-theme-material input[type="text"],
    .ag-theme-material input[type="number"] {
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
    }
    .ag-theme-material .ag-cell {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
    }
    .ag-theme-material .ag-popup-editor {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Load config.json ---
    with open("config.json") as f:
        config = json.load(f)

    st.title("TGSRTC PRODUCTIVITY DASHBOARD")

    category_to_column = config.get("category_to_column", {})

    # --- Save Data ORM ---
    def save_to_db(grid_response, selected_depot, date_to_save):
        df = pd.DataFrame(grid_response["data"])

        # 🧮 Recalculate totals before saving
        for col in df.columns:
            if re.match(r'\d{4}-\d{2}-\d{2}', col):
                for idx, row in df.iterrows():
                    category = row["Category"]

                    if category == "Total Drivers (SL Reasons)":
                        subcategories = [
                            "Flu/Fever", "BP", "Orthopedic", "Heart", "Weakness", "Eye",
                            "Accident/Injuries", "Neuro/Paralysis (Sick Leave)", "Piles", "Diabetes",
                            "Thyroid", "Gas", "Dental", "Ear", "Skin/Allergy", "General Surgery",
                            "Obesity", "Cancer"
                        ]
                        total = 0
                        for sub in subcategories:
                            sub_row = df[df["Category"] == sub]
                            if not sub_row.empty:
                                val = sub_row[col].values[0]
                                total += float(val) if pd.notna(val) and val != '' else 0
                        df.at[idx, col] = total

                    elif category == "Diff (SL Reasons)":
                        total_row = df[df["Category"] == "Total Drivers (SL Reasons)"]
                        sick_leave_row = df[df["Category"] == "Sick Leave"]
                        if not total_row.empty and not sick_leave_row.empty:
                            total_val = float(total_row[col].values[0] or 0)
                            sick_val = float(sick_leave_row[col].values[0] or 0)
                            df.at[idx, col] = total_val - sick_val

        date_columns = [col for col in df.columns if re.match(r'\d{4}-\d{2}-\d{2}', col)]
        rows_to_insert = []
        target_date_str = date_to_save.strftime('%Y-%m-%d')

        if target_date_str not in df.columns:
            st.error(f"❌ Column '{target_date_str}' not found in grid data.")
            return False

        row_data = {"depot_name": selected_depot, "data_date": pd.to_datetime(target_date_str).date()}
        for _, r in df.iterrows():
            cat = r["Category"]
            if not isinstance(cat, str) or not cat.strip():
                continue
            db_col = category_to_column.get(cat.strip())
            if db_col:
                val = r[target_date_str]
                try:
                    val = float(val) if pd.notna(val) and val != '' else None
                    val = int(val) if val and val.is_integer() else val
                except Exception:
                    pass
                row_data[db_col] = val
        # 🧮 Recompute Service/Driver Check before saving to DB
        try:
            if 'Driver for Bus Services' in df.columns and 'Actual Services' in df.columns:
                df['Service/Driver Check'] = (
                    df['Driver for Bus Services'].fillna(0) -
                    df['Actual Services'].fillna(0)
                )
        except Exception as e:
            st.warning(f"⚠️ Could not recompute 'Service/Driver Check' before saving: {e}")

        rows_to_insert.append(row_data)

        try:
            with get_session() as db:
                for row in rows_to_insert:
                    existing = (
                        db.query(InputData)
                        .filter(InputData.depot_name == row["depot_name"])
                        .filter(InputData.data_date == row["data_date"])
                        .first()
                    )
                    if existing:
                        for k, v in row.items():
                            setattr(existing, k, v)
                    else:
                        db.add(InputData(**row))
                db.commit()
            return True
        except Exception as e:
            st.error(f"Error saving data: {e}")
            return False

    # --- Fetch Existing Data ORM ---
    def fetch_existing_data_for_dates(depot_name, date_columns):
        try:
            with get_session() as db:
                data = (
                    db.query(InputData)
                    .filter(InputData.depot_name == depot_name)
                    .filter(InputData.data_date.in_(date_columns))
                    .all()
                )
                return pd.DataFrame([{k: v for k, v in d.__dict__.items() if not k.startswith("_")} for d in data])
        except Exception as e:
            st.error(f"DB Error: {e}")
            return pd.DataFrame()

    # --- Daily Depot Input Sheet Section ---
    st.header("DAILY SCHEDULE AND DRIVER DATA")
    st.markdown("### 📝 Enter New Record")

    col1, col2 = st.columns(2)
    with col1:
        selected_depot = st.session_state.get("user_depot")
        if not selected_depot:
            st.error("Depot not found in session. Please log in again.")
            st.stop()
        else:
            st.success(f"Depot: **{selected_depot}**")

    # ✅ Show recent entries
    with get_session() as db:
        recent_dates = (
            db.query(InputData.data_date)
            .filter(InputData.depot_name == selected_depot)
            .order_by(InputData.data_date.desc())
            .limit(1)
            .all()
        )
        if recent_dates:
            formatted = [pd.to_datetime(r[0]).strftime("%d %b %y").upper() for r in recent_dates]
            st.markdown("🗓 *Recently Entered Dates:* " + ", ".join(formatted))
        else:
            st.info("ℹ No data found in DB for this depot yet.")

    # ✅ Compute next available date
    with get_session() as db:
        result = (
            db.query(InputData.data_date)
            .filter(InputData.depot_name == selected_depot)
            .order_by(InputData.data_date.desc())
            .first()
        )
        next_allowed_date = result[0] + timedelta(days=1) if result else date.today()

    with col2:
        user_selected_date = st.date_input(
            "📅 Select base date for 10-day range",
            value=next_allowed_date,
            min_value=next_allowed_date,
            max_value=next_allowed_date,
        )

    if selected_depot == "Select Depot":
        st.info("Select a depot to proceed with data entry.")
        st.stop()

    # ✅ Prepare date range columns
    date_columns = [(user_selected_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(9, -1, -1)]

    def check_existing_dates(depot_name, date_list):
        try:
            with get_session() as db:
                rows = (
                    db.query(InputData.data_date)
                    .filter(InputData.depot_name == depot_name)
                    .filter(InputData.data_date.in_(date_list))
                    .all()
                )
                return {r[0].strftime("%Y-%m-%d") for r in rows}
        except Exception as e:
            st.error(f"Error checking existing dates: {e}")
            return set()

    existing_dates = check_existing_dates(selected_depot, date_columns)

    # ✅ Depot Category Lookup via ORM
    depot_config = get_depot_settings()
    depot_type_from_admin = depot_config.get(selected_depot, {}).get("category", "N/A")
    st.markdown(f"*Depot Type:* `{depot_type_from_admin}`")

    # ✅ Download Existing Data
    with st.expander("⬇️ Download Data"):
        start_date = st.date_input("From Date", value=date.today() - timedelta(days=7))
        end_date = st.date_input("To Date", value=date.today())

        with get_session() as db:
            data = (
                db.query(InputData)
                .filter(InputData.depot_name == selected_depot)
                .filter(InputData.data_date.between(start_date, end_date))
                .order_by(InputData.data_date)
                .all()
            )
            df_download = pd.DataFrame([{k: v for k, v in d.__dict__.items() if not k.startswith("_")} for d in data])

            if not df_download.empty:
                category_to_column = config["category_to_column"]
                column_to_category = {v: k for k, v in category_to_column.items()}
                df_download = df_download.rename(columns=column_to_category)

                st.download_button(
                    label="📥 Download CSV",
                    data=df_download.to_csv(index=False),
                    file_name=f"{selected_depot}_{start_date}_{end_date}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("⚠️ No data available for this range.")

# --- Benchmarks (loaded from config.json instead of hardcoding) ---
    # --- Benchmarks (case-insensitive lookup) ---
    benchmarks = config.get("benchmarks", {})
    # Normalize depot type (e.g., "RURAL" -> "Rural")
    normalized_depot_type = depot_type_from_admin.strip().title()
    current_bench = benchmarks.get(normalized_depot_type, {})

    st.title(f"🚌 {selected_depot} Data Editor - Last 10 Days")


    category_rows = config.get("category_rows", {})
    date_today_str = date.today().strftime("%Y-%m-%d")


    data = {"Category": category_rows}
    data["Depot Type"] = [depot_type_from_admin] * len(category_rows)


    df = pd.DataFrame(data)
    # FULLY FIXED VERSION OF YOUR DATA INJECTION LOGIC

    # Step 1: Fetch existing data
    existing_data_df = fetch_existing_data_for_dates(selected_depot, date_columns)

    # Step 2: Ensure 'data_date' format matches AgGrid column headers
    # 🔒 Identify DB-fetched cells to lock editing
    fetched_cells = set()
    if not existing_data_df.empty and "data_date" in existing_data_df.columns:
        existing_data_df["data_date"] = pd.to_datetime(existing_data_df["data_date"]).dt.strftime("%Y-%m-%d")
        reverse_map = {v: k for k, v in category_to_column.items()}

        for col_date in date_columns:
            matching_rows = existing_data_df[existing_data_df["data_date"] == col_date]
            if matching_rows.empty:
                continue
            row_data = matching_rows.iloc[0]

            for db_col, value in row_data.items():
                if db_col == "data_date" or pd.isna(value):
                    continue

                category = reverse_map.get(db_col)
                if category:
    # Match the row in DataFrame where Category == this category
                    matched_index = df[df["Category"].str.strip().str.lower() == category.strip().lower()].index
                    if not matched_index.empty:
                        df.at[matched_index[0], col_date] = value
                        fetched_cells.add((category.strip(), col_date))

    formatted_columns = {}
    for col in df.columns:
        if col != "Category":
            try:
                dt = pd.to_datetime(col)
                formatted_columns[col] = dt.strftime("%d-%b-%Y")
            except:
                formatted_columns[col] = col
        else:
            formatted_columns[col] = col

    gb = GridOptionsBuilder.from_dataframe(df)

    # Format column headers for display only — not renaming df
    for col in df.columns:
        if col != "Category":
            try:
                dt = pd.to_datetime(col)
                pretty_label = dt.strftime("%d-%b-%Y")  # e.g., 23-Jun-2025
                gb.configure_column(field=col, header_name=pretty_label)
            except:
                gb.configure_column(field=col)
        else:
            gb.configure_column(field=col, pinned="left", editable=False)

    editable_rows = config.get("editable_rows", {})

    # Categories that are always non-editable and checked directly in JS
    non_editable_fixed_categories = config.get("non_editable_fixed_categories", {})
    benchmark_json = json.dumps(current_bench)
    editable_rows_json = json.dumps(editable_rows)
    non_editable_fixed_categories_json = json.dumps(non_editable_fixed_categories)


    # --- AgGrid Configuration ---
    gb = GridOptionsBuilder.from_dataframe(df)
    # Set resizable to False and sortable to False for default columns
    # Removed enableCellTextSelection from here as it's a gridOptions property
    gb.configure_default_column(resizable=False, sortable=False, wrapText=False, autoHeight=False)
    gb.configure_grid_options(
        domLayout='normal',
        rowHeight=30, # Default row height, will be overridden by autoHeight for wrapped cells
        getRowId=JsCode("function(params) { return params.data.Category; }").js_code,
        suppressMovableColumns=True, # Prevent column reordering
        # Removed enableSorting=False from here as it's not a valid top-level grid option
        onCellValueChanged=JsCode("""
            function(params) {
                const category = params.data.Category;
                const colId = params.column.colId;
                const api = params.api;

                let categoriesToRecalculate = new Set();

                const addCategory = (cat) => {
                    if (!cat.startsWith('---')) {
                        categoriesToRecalculate.add(cat);
                    }
                };
                const editedRowNode = params.node;
                editedRowNode.data[colId] = params.newValue;
                // --- Define explicit dependencies for recalculation ---
                if (category === 'Planned Services' || category === 'Actual Services') {
                    addCategory('Service Variance');
                    addCategory('Service/Driver Check');
                }
                if (category === 'Planned KM' || category === 'Actual KM') {
                    addCategory('KM Variance');
                    addCategory('KM/Driver');
                }
                if (['Actual KM', 'Driver for Bus Services'].includes(category)) {
                    addCategory('KM/Driver');
                    addCategory('Service/Driver Check');
                }


                if (['Total Drivers', 'Medically Unfit', 'Suspended Drivers'].includes(category)) {
                    addCategory('Available Drivers-1');
                    addCategory('% Available Drivers-1');
                    addCategory('Available Drivers-2');
                    addCategory('% Available Drivers-2');
                    addCategory('Attending Drivers');
                    addCategory('% Attending Drivers');
                    addCategory('Driver shortage');
                    addCategory('% Weekly Off & National Off');
                    addCategory('% Special Off (Night Out/IC, Online)');
                    addCategory('% Others');
                    addCategory('% Leave & Absent');
                    addCategory('% Sick Leave');
                    addCategory('Service/Driver Check');
                }

                if (['Weekly Off & National Off', 'Special Off (Night Out/IC, Online)',
                    'Training, PME(medical)', 'Others (SDI, DGT, LO, Parking,<br>Relief Van,Depot Spare,<br> Cargo, Releaving duty)',
                    'Leave & Absent', 'Sick Leave'].includes(category)) {
                    addCategory('Available Drivers-2');
                    addCategory('% Available Drivers-2');
                    addCategory('Attending Drivers');
                    addCategory('% Attending Drivers');
                    addCategory('Driver shortage');
                    addCategory('Service/Driver Check');
                    addCategory('% Weekly Off & National Off');
                    addCategory('% Special Off (Night Out/IC, Online)');
                    addCategory('% Others');
                    addCategory('% Leave & Absent');
                    addCategory('% Sick Leave');
                }

                if (['Available Drivers-2', 'Spot Absent'].includes(category)) {
                    addCategory('Attending Drivers');
                    addCategory('% Spot Absent');
                    addCategory('% Attending Drivers');
                    addCategory('Driver shortage');
                    addCategory('Service/Driver Check');
                }
                if (['Drivers Required', 'Attending Drivers'].includes(category)) {
                    addCategory('Driver shortage');
                    addCategory('Service/Driver Check');
                }
                // NEW DEPENDENCY: Driver schedule calculation
                if (category === 'Drivers Required' || category === 'Planned Schedules') {
                    addCategory('Driver schedule');
                }

                // Recalculate Drivers on Duty if Attending Drivers, Double Duty, or Off Cancellation changes
                if (['Attending Drivers', 'Double Duty', 'Off Cancellation'].includes(category)) {
                    addCategory('Drivers on Duty');
                    addCategory('Driver for Bus Services');
                    addCategory('KM/Driver');
                    addCategory('Service/Driver Check');
                }
                // Recalculate Driver for Bus Services if Drivers on Duty or Drivers as Conductors changes
                if (['Drivers on Duty', 'Drivers as Conductors'].includes(category)) {
                    addCategory('Driver for Bus Services');
                    addCategory('Service/Driver Check');
                    addCategory('KM/Driver');
                }
                // Recalculate KM/Driver if Actual KM or Drivers for Bus Services changes
                if (['Actual KM', 'Driver for Bus Services'].includes(category)) {
                    addCategory('KM/Driver');
                }
                // Recalculate Services/Driver Check if Drivers for Bus Services or Actual Services changes
                if (['Driver for Bus Services', 'Actual Services'].includes(category)) {
                    addCategory('Service/Driver Check');
                }

                if (['Double Duty', 'Off Cancellation', 'Weekly Off & National Off', 'Attending Drivers'].includes(category)) {
                    addCategory('% Double Duty');
                    addCategory('% Off Cancellation');
                    addCategory('Service/Driver Check');
                }

                if (['Spondilitis', 'Spinal Disc', 'Vision/Color Blindness', 'Neuro/Paralysis (Medical)', 'Ortho'].includes(category)) {
                    addCategory('Total Drivers (MU Reasons)');
                    addCategory('Diff (MU Reasons)');
                }
                if (['Total Drivers (MU Reasons)', 'Medically Unfit'].includes(category)) {
                    addCategory('Diff (MU Reasons)');
                }

                if (['Flu/Fever', 'BP', 'Orthopedic', 'Heart', 'Weakness', 'Eye',
                    'Accident/Injuries', 'Neuro/Paralysis (Sick Leave)', 'Piles',
                    'Diabetes', 'Thyroid', 'Gas', 'Dental', 'Ear', 'Skin/Allergy',
                    'General Surgery', 'Obesity', 'Cancer'].includes(category)) {
                    addCategory('Total Drivers (SL Reasons)');
                    addCategory('Diff (SL Reasons)');
                    
                }
                if (['Total Drivers (SL Reasons)', 'Sick Leave'].includes(category)) {
                    addCategory('Diff (SL Reasons)');
                    const currentTotalSLReasons = api.getRowNode('Total Drivers (SL Reasons)').data[colId];
                    const currentSickLeave = api.getRowNode('Sick Leave').data[colId];
                }

                if (categoriesToRecalculate.size > 0) {
                    setTimeout(() => {
                        const rowNodesToRefresh = Array.from(categoriesToRecalculate).map(cat => api.getRowNode(cat));
                        api.refreshCells({
                            rowNodes: rowNodesToRefresh,
                            columns: [colId],
                            force: true
                        });

                        //  Force Service/Driver Check refresh explicitly
                        const driverCheckNode = api.getRowNode("Service/Driver Check");
                        if (driverCheckNode) {
                            api.refreshCells({
                                rowNodes: [driverCheckNode],
                                force: true
                            });
                        }
                    }, 100);
}

            }
        """).js_code,
    )
    fetched_cells_js = json.dumps([[cat.strip(), date] for cat, date in fetched_cells])



    cell_edit_js = JsCode(f"""
    function(params) {{
        const editableRows = JSON.parse('{editable_rows_json}');
        const nonEditableFixedCategories = JSON.parse('{non_editable_fixed_categories_json}');
        const fetchedCells = new Set({fetched_cells_js}.map(([cat, date]) => cat + '||' + date));

        const category = params.data.Category;
        const colId = params.column.colId;

        // 🔒 Disallow editing if cell was fetched from DB
        if (fetchedCells.has(category + '||' + colId)) {{
            return false;
        }}

        // 🔒 Disallow headings, calculated fields, and percentage rows
        if (category.startsWith('---') ||
            category.includes('%') ||
            nonEditableFixedCategories.includes(category)) {{
            return false;
        }}

        // ✅ Allow editing only for whitelisted rows
        return editableRows.includes(category);
    }}
    """)


    renderer_params = {
        "current_benchmarks": current_bench
    }
    
    centered_number_editor = JsCode("""
class CenteredNumberEditor {
    init(params) {
        this.eInput = document.createElement('input');
        this.eInput.type = 'text';
        this.eInput.value = params.value || 0;
    }

    getGui() {
        return this.eInput;
    }

    getValue() {
        return parseFloat(this.eInput.value);
    }

    focusIn() {
        this.eInput.focus();
        this.eInput.select();
    }
}
""")




    for col in date_columns:
        try:
        # Format column label for display only - CHANGED TO SHORTER FORMAT
            pretty_label = pd.to_datetime(col).strftime("%d-%b-%y") # e.g., 18-Jul
        except:
            pretty_label = col  # fallback if parsing fails
        gb.configure_column(
            field=col,
            header_name=pretty_label,
            editable=cell_edit_js,
            type=["numericColumn", "rightAligned"],
            width=65, # ADJUSTED WIDTH FOR SHORTER HEADER
            resizable=False, # Disable resizing for date columns
            sortable=False,  # Explicitly disable sorting for date columns
            cellEditorPopup=True, # Ensure editor appears as a popup
            cellEditor="agTextCellEditor", # Use the built-in number editor for consistency
            cellEditorParams={"min": 0}, # Ensure only non-negative numbers can be entered
            #suppressKeyboardEvent=True, # Prevent default keyboard navigation during editing
            suppressKeyboardEvent=JsCode("""
                    function(params) {
                        const key = params.event.key;
                        // Don't suppress Tab or Enter to allow moving between cells
                        return false;
                    }
                """).js_code,

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
                        style["border-top"] = "2px solid #aaddff";
                        style["border-bottom"] = "1px solid #d3d3d3";
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
                        if (value < 0) {
                            style["color"] = "red";
                        }
                    } else if (category.includes('%')) {
                        style["background-color"] = "#e0e0f0";
                    }
                    return style;
                }
            """).js_code,
            valueGetter=JsCode(f"""
                function(params) {{
                    const category = params.data.Category;
                    const dateCol = params.colDef.field;
                    const api = params.api;

                    // If it's a header row, return an empty string
                    if (category.startsWith('---')) {{
                        return '';
                    }}

                    const getCellValueAsNumber = (cat) => {{
                        const rowNode = api.getRowNode(cat);
                        if (rowNode && rowNode.data[dateCol] !== undefined && rowNode.data[dateCol] !== null && rowNode.data[dateCol] !== "") {{
                            const value = parseFloat(rowNode.data[dateCol]);
                            return isNaN(value) ? 0 : value;
                        }}
                        return 0;
                    }};

                    let result; // Declare result here to avoid 'not defined' errors

                    if (category === "Service Variance") {{
                        const plannedServices = getCellValueAsNumber('Planned Services');
                        const actualServices = getCellValueAsNumber('Actual Services');
                        result = actualServices - plannedServices;
                        params.data[dateCol] = result;
                        return result;
                    }} else if (category === "KM Variance") {{
                        const plannedKM = getCellValueAsNumber('Planned KM');
                        const actualKM = getCellValueAsNumber('Actual KM');
                        result = actualKM - plannedKM;
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "Driver schedule") {{ // NEW CALCULATION
                        const driversRequired = getCellValueAsNumber('Drivers Required');
                        const plannedSchedules = getCellValueAsNumber('Planned Schedules');
                        if (plannedSchedules !== 0) {{
                            result = (driversRequired / plannedSchedules).toFixed(0);
                        }} else {{
                            result = 0;
                        }}
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "Available Drivers-1") {{
                        const totalDrivers = getCellValueAsNumber('Total Drivers');
                        const medicallyUnfit = getCellValueAsNumber('Medically Unfit');
                        const suspendedDrivers = getCellValueAsNumber('Suspended Drivers');
                        result = totalDrivers - medicallyUnfit - suspendedDrivers;
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "% Available Drivers-1") {{
                        const availableDrivers1 = getCellValueAsNumber('Available Drivers-1');
                        const totalDrivers = getCellValueAsNumber('Total Drivers');
                        if (totalDrivers > 0) {{
                            result = ((availableDrivers1 / totalDrivers) * 100).toFixed(0);
                            params.data[dateCol] = result;
                            return result;
                        }}
                        result = '';
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "Available Drivers-2") {{
                        const availableDrivers1 = getCellValueAsNumber('Available Drivers-1');
                        const weeklyOff = getCellValueAsNumber('Weekly Off & National Off');
                        const specialOff = getCellValueAsNumber('Special Off (Night Out/IC, Online)');
                        const trainingPME = getCellValueAsNumber('Training, PME(medical)');
                        const others = getCellValueAsNumber('Others (SDI, DGT, LO, Parking,<br>Relief Van,Depot Spare,<br> Cargo, Releaving duty)');
                        const leaveAbsent = getCellValueAsNumber('Leave & Absent');
                        const sickLeave = getCellValueAsNumber('Sick Leave');

                        const totalDeductions = weeklyOff + specialOff + trainingPME + others + leaveAbsent + sickLeave;
                        result = availableDrivers1 - totalDeductions;
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "% Available Drivers-2") {{
                        const availableDrivers2 = getCellValueAsNumber('Available Drivers-2');
                        const totalDrivers = getCellValueAsNumber('Total Drivers');
                        if (totalDrivers > 0) {{
                            result = ((availableDrivers2 / totalDrivers) * 100).toFixed(0);
                            params.data[dateCol] = result;
                            return result;
                        }}
                        result = '';
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "% Weekly Off & National Off") {{
                        const weeklyOff = getCellValueAsNumber('Weekly Off & National Off');
                        const totalDrivers = getCellValueAsNumber('Total Drivers');
                        if (totalDrivers > 0) {{
                            result = ((weeklyOff / totalDrivers) * 100).toFixed(0);
                            params.data[dateCol] = result;
                            return result;
                        }}
                        result = '';
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "% Special Off (Night Out/IC, Online)") {{
                        const specialOff = getCellValueAsNumber('Special Off (Night Out/IC, Online)');
                        const totalDrivers = getCellValueAsNumber('Total Drivers');
                        if (totalDrivers > 0) {{
                            result = ((specialOff / totalDrivers) * 100).toFixed(0);
                            params.data[dateCol] = result;
                            return result;
                        }}
                        result = '';
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "% Others") {{
                        const trainingPME = getCellValueAsNumber('Training, PME(medical)');
                        const others = getCellValueAsNumber('Others (SDI, DGT, LO, Parking,<br>Relief Van,Depot Spare,<br> Cargo, Releaving duty)');
                        const totalDrivers = getCellValueAsNumber('Total Drivers');

                        if (totalDrivers > 0) {{
                            result = (((trainingPME + others) / totalDrivers) * 100).toFixed(0);
                            params.data[dateCol] = result;
                            return result;
                        }}
                        result = '';
                        params.data[dateCol] = result;
                        return result;
                    }}

                    else if (category === "% Leave & Absent") {{
                        const leaveAbsent = getCellValueAsNumber('Leave & Absent');
                        const totalDrivers = getCellValueAsNumber('Total Drivers');
                        if (totalDrivers > 0) {{
                            result = ((leaveAbsent / totalDrivers) * 100).toFixed(0);
                            params.data[dateCol] = result;
                            return result;
                        }}
                        result = '';
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "% Sick Leave") {{
                        const sickLeave = getCellValueAsNumber('Sick Leave');
                        const totalDrivers = getCellValueAsNumber('Total Drivers');
                        if (totalDrivers > 0) {{
                            result = ((sickLeave / totalDrivers) * 100).toFixed(0);
                            params.data[dateCol] = result;
                            return result;
                        }}
                        result = '';
                        params.data[dateCol] = result;
                        console.log("SL % calc - sickLeave:", sickLeave, ", totalDrivers:", totalDrivers);
                        return result;
                        

                    }}
                    else if (category === "Attending Drivers") {{
                        const availableDrivers2 = getCellValueAsNumber('Available Drivers-2');
                        const spotAbsent = getCellValueAsNumber('Spot Absent');
                        result = availableDrivers2 - spotAbsent;
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "% Attending Drivers") {{
                        const attendingDrivers = getCellValueAsNumber('Attending Drivers');
                        const totalDrivers = getCellValueAsNumber('Total Drivers');
                        if (totalDrivers > 0) {{
                            result = ((attendingDrivers / totalDrivers) * 100).toFixed(0);
                            params.data[dateCol] = result;
                            return result;
                        }}
                        result = '';
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "% Spot Absent") {{
                        const spotAbsent = getCellValueAsNumber('Spot Absent');
                        const totalDrivers = getCellValueAsNumber('Total Drivers');
                        if (totalDrivers > 0) {{
                            result = ((spotAbsent / totalDrivers) * 100).toFixed(0);
                            params.data[dateCol] = result;
                            return result;
                        }}
                        result = '';
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "Driver shortage") {{
                        const driversRequired = getCellValueAsNumber('Drivers Required');
                        const attendingDrivers = getCellValueAsNumber('Attending Drivers');
                        if (driversRequired > attendingDrivers) {{
                            result = driversRequired - attendingDrivers;
                        }} else {{
                            result = 0;
                        }}
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "% Double Duty") {{
                        const doubleDuty = getCellValueAsNumber('Double Duty');
                        const totalDrivers = getCellValueAsNumber('Total Drivers');
                        if (totalDrivers > 0) {{
                            result = ((doubleDuty / totalDrivers) * 100).toFixed(0);
                            params.data[dateCol] = result;
                            return result;
                        }}
                        result = '';
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "% Off Cancellation") {{
                        const offCancellation = getCellValueAsNumber('Off Cancellation');
                        const totalDrivers = getCellValueAsNumber('Total Drivers');
                        if (totalDrivers > 0) {{
                            result = ((offCancellation / totalDrivers) * 100).toFixed(0);
                            params.data[dateCol] = result;
                            return result;
                        }}
                        result = '';
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "Drivers on Duty") {{
                        const attendingDrivers = getCellValueAsNumber('Attending Drivers');
                        const doubleDuty = getCellValueAsNumber('Double Duty');
                        const offCancellation = getCellValueAsNumber('Off Cancellation');
                        result = attendingDrivers + doubleDuty + offCancellation;
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "Driver for Bus Services") {{
                        const driversOnDuty = getCellValueAsNumber('Drivers on Duty');
                        const driversAsConductors = getCellValueAsNumber('Drivers as Conductors');
                        result = driversOnDuty - driversAsConductors;
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "KM/Driver") {{
                        const actualKM = getCellValueAsNumber('Actual KM');
                        const driversForBusServices = getCellValueAsNumber('Driver for Bus Services');
                        console.log(`KM/Driver Calculation - Actual KM: ${{actualKM}} Drivers for Bus Services: ${{driversForBusServices}}`);
                        if (driversForBusServices !== 0) {{
                            result = (actualKM / driversForBusServices).toFixed(0);
                        }} else {{
                            result = '';
                        }}
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "Service/Driver Check") {{
                        const driversForBusServices = getCellValueAsNumber('Driver for Bus Services');
                        const actualServices = getCellValueAsNumber('Actual Services');
                        result = driversForBusServices - actualServices;
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "Total Drivers (MU Reasons)") {{
                        const spondilitis = getCellValueAsNumber('Spondilitis');
                        const spinalDisc = getCellValueAsNumber('Spinal Disc');
                        const vision = getCellValueAsNumber('Vision/Color Blindness');
                        const neuro = getCellValueAsNumber('Neuro/Paralysis (Medical)');
                        const ortho = getCellValueAsNumber('Ortho');
                        result = spondilitis + spinalDisc + vision + neuro + ortho;
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "Diff (MU Reasons)") {{
                        const totalMUReasons = getCellValueAsNumber('Total Drivers (MU Reasons)');
                        const medicallyUnfit = getCellValueAsNumber('Medically Unfit');
                        result = totalMUReasons - medicallyUnfit;
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "Total Drivers (SL Reasons)") {{
                        const fluFever = getCellValueAsNumber('Flu/Fever');
                        const bp = getCellValueAsNumber('BP');
                        const orthopedic = getCellValueAsNumber('Orthopedic');
                        const heart = getCellValueAsNumber('Heart');
                        const weakness = getCellValueAsNumber('Weakness');
                        const eye = getCellValueAsNumber('Eye');
                        const accidentInjuries = getCellValueAsNumber('Accident/Injuries');
                        const neuroSickLeave = getCellValueAsNumber('Neuro/Paralysis (Sick Leave)');
                        const piles = getCellValueAsNumber('Piles');
                        const diabetes = getCellValueAsNumber('Diabetes');
                        const thyroid = getCellValueAsNumber('Thyroid');
                        const gas = getCellValueAsNumber('Gas');
                        const dental = getCellValueAsNumber('Dental');
                        const ear = getCellValueAsNumber('Ear');
                        const skinAllergy = getCellValueAsNumber('Skin/Allergy');
                        const generalSurgery = getCellValueAsNumber('General Surgery');
                        const obesity = getCellValueAsNumber('Obesity');
                        const cancer = getCellValueAsNumber('Cancer');

                        const totalSLReasons = fluFever + bp + orthopedic + heart + weakness + eye + accidentInjuries +
                                neuroSickLeave + piles + diabetes + thyroid + gas + dental + ear +
                                skinAllergy + generalSurgery + obesity + cancer;
                        result = totalSLReasons;
                        params.data[dateCol] = result;
                        return result;
                    }}
                    else if (category === "Diff (SL Reasons)") {{
                            const totalSLReasons = getCellValueAsNumber('Total Drivers (SL Reasons)');
                            const sickLeave = getCellValueAsNumber('Sick Leave');
                            result = totalSLReasons - sickLeave;
                            params.data[dateCol] = result;
                            return result;
                            }}
                    

                    const originalValue = params.data[dateCol];
                    const numOriginalValue = parseFloat(originalValue);
                    return isNaN(numOriginalValue) ? 0 : numOriginalValue;
                }}
            """).js_code,
            cellRenderer=JsCode("""
                class BenchmarkCellRenderer {
                    init(params) {
                        this.eGui = document.createElement('div');
                        this.updateValue(params);
                    }
                    getGui() {
                        return this.eGui;
                    }
                    refresh(params) {
                        this.updateValue(params);
                        return true;
                    }
                    updateValue(params) {
                        const category = params.data.Category || "";
                        const value = parseFloat(params.value);

                        if (isNaN(value) || value === null || value === undefined || value === '') {
                            this.eGui.innerHTML = '';
                            return;
                        }

                        if (category.includes('%')) {
                            const formattedValue = value.toFixed(0) + "%";
                            // Only set the formatted percentage, remove benchmark text
                            this.eGui.innerHTML = formattedValue;
                        } else {
                            this.eGui.innerHTML = params.value;
                        }
                    }
                }
            """).js_code,
            cellRendererParams=renderer_params
        )

    gb.configure_column(
        
        field="Category",
        
        pinned="left",
        lockPinned=True,
        cellClass="locked-col",
        editable=False,
        width=150, # Width set to 150px
        resizable=False,
        sortable=False,
        wrapText=True, # Enable text wrapping
        autoHeight=True, # Enable auto height for rows with wrapped text
        cellStyle=JsCode("""
            function(params) {
                const category = params.value;
                let style = {
                    "background-color": "#f0f0f0",
                    "font-weight": "bold",
                    "border": "1px solid #d3d3d3",
                    "white-space": "normal", // Ensure normal wrapping behavior
                    "word-break": "break-word" // Break long words if necessary
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
        """).js_code,
        # NEW: Using a class-based cellRenderer to explicitly handle HTML rendering
        cellRenderer=JsCode("""
            class HtmlRenderer {
                init(params) {
                    this.eGui = document.createElement('div');
                    // Check if the value contains HTML tags (like <br>)
                    // If it does, set innerHTML to render HTML, otherwise set plain text
                    if (params.value && typeof params.value === 'string' && params.value.includes('<')) {
                        this.eGui.innerHTML = params.value;
                    } else {
                        this.eGui.innerText = params.value;
                    }
                }
                getGui() {
                    return this.eGui;
                }
                refresh(params) {
                    // Update content if the cell data changes
                    if (params.value && typeof params.value === 'string' && params.value.includes('<')) {
                        this.eGui.innerHTML = params.value;
                    } else {
                        this.eGui.innerText = params.value;
                    }
                    return true; // Indicate that the refresh was handled
                }
            }
        """).js_code
    )

    renderer_params = {
        "current_benchmarks": current_bench
    }

    gb.configure_column(
        field="Depot Type",
        header_name="Rural/Urban",
        pinned="left",
        lockPinned=True,
        cellClass="locked-col",
        editable=False,
        width=40,
        resizable=False, 
        sortable=False,
        cellStyle=JsCode("""
            function(params) {
                let style = {"background-color": "#f0f8ff", "font-weight": "normal", "border": "1px solid #d3d3d3"};
                if (params.data.Category.startsWith('---')) {
                    style["background-color"] = "#cceeff";
                    style["font-weight"] = "bold";
                    style["text-align"] = "center";
                    style["font-size"] = "1.1em";
                    style["border-top"] = "2px solid #aaddff";
                    style["border-bottom"] = "1px solid #d3d3d3";
                }
                return style;
            }
        """).js_code,
        cellRenderer=JsCode("""
            function(params) {
                const category = params.data.Category;
                const currentBenchmarks = params.colDef.cellRendererParams.current_benchmarks;

                let benchmarkValue = null;

                // The key in currentBenchmarks for "Others" is "% Others"
                // No need to change benchmarkCategoryKey for this specific case
                // as the 'category' variable already holds "% Others"
                const benchmarkCategoryKey = category; // Keep this line as is

                if (currentBenchmarks.hasOwnProperty(benchmarkCategoryKey)) {
                    benchmarkValue = currentBenchmarks[benchmarkCategoryKey];
                }

                if (benchmarkValue !== null && category.includes('%')) { // Add category.includes('%') to ensure it's a percentage row
                    return 'Benchmark - ' + benchmarkValue + '%';
                } else if (category.startsWith('---')) {
                    return params.value;
                } else {
                    return '';
                }
            }
        """).js_code,
        cellRendererParams=renderer_params
    )

    grid_options = gb.build()

    # --- Calculate dynamic height here ---
    # Assuming each row is 30px (as per rowHeight) and header is approx 35px
    header_height = 35
    total_rows = len(df)
    row_height = 30
    calculated_height = (total_rows * row_height) + header_height + 10
    with st.form("data_submit_form"):
        grid_response = AgGrid(
                            df,
                            gridOptions=grid_options,
                            update_mode=GridUpdateMode.VALUE_CHANGED,
                            theme="material",
                            height=calculated_height,
                            fit_columns_on_grid_load=False,
                            allow_unsafe_jscode=True,
                            enable_enterprise_modules=False,
                            data_return_mode='AS_INPUT',  # ✅ Return exactly the data shown
                            always_return_data=True , # ✅ Ensure we always get the data back
                            custom_js_components={  # 👈 THIS LINE FIXES THE ERROR
                                    "CenteredNumberEditor": centered_number_editor
                                }

        )
        submitted = st.form_submit_button("💾 Submit")

    if submitted:
        df_input = pd.DataFrame(grid_response["data"])
        # Clean up any whitespace/special characters
        df_input["Category"] = df_input["Category"].astype(str).str.strip()


# Ensure date column variable exists
        selected_date_col = user_selected_date.strftime("%Y-%m-%d")

        # Only convert if column exists
        if selected_date_col in df_input.columns:
            df_input[selected_date_col] = pd.to_numeric(df_input[selected_date_col], errors="coerce")
        else:
            st.error(f"⚠️ Column '{selected_date_col}' not found in df_input.columns.")
            st.stop()


        errors = []
        editable_rows_set = set(editable_rows)
        selected_date_col = user_selected_date.strftime("%Y-%m-%d")

        for idx, row in df_input.iterrows():
            category = row["Category"]
            if category not in editable_rows_set:
                continue  # Skip non-editable rows

            value = row.get(selected_date_col)

            # 1. Check missing value
            if value in ("", None):
                errors.append(f"❌ '{category}' is empty on {selected_date_col}")
                continue

            # 2. Check for integer
            try:
                int_value = int(value)
            except:
                errors.append(f"❌ '{category}' must be an integer on {selected_date_col}")
                continue

            # 3. Validation Rules 
            if category in ["Schedules","Schedules Services","Planned Schedules", "Planned Services", "Actual Services", "Total Drivers"]:
                if not (1<= int_value <= 999):
                    errors.append(f"❌ '{category}' Check the value  on {selected_date_col}")

            elif category in ["Schedules Kms","Planned KM", "Actual KM"]:
                if not (1000 <= int_value <= 99999):
                    errors.append(f"❌ '{category}' Check the value on {selected_date_col}")

            elif category =="Drivers Required":
                if int_value <= 0:
                    errors.append(f"❌ '{category}' must be greater than 0 on {selected_date_col}")
            elif category =="Service/Driver Check":
                if int_value <= 0:
                    errors.append(f"❌ '{category}' must be greater than 0 on {selected_date_col}")

            # Remaining editable fields: must be integer and >= 0
            else:
                if int_value < 0:
                    errors.append(f"❌ '{category}' cannot be negative on {selected_date_col}")

        # ✅ Special validation (outside loop!)
        try:
            mu_total = int(df_input.loc[df_input["Category"] == "Total Drivers (MU Reasons)", selected_date_col].values[0])
            medically_unfit = int(df_input.loc[df_input["Category"] == "Medically Unfit", selected_date_col].values[0])
            if mu_total - medically_unfit != 0:
                errors.append(f"❌ 'Diff (MU Reasons)' must be 0 on {selected_date_col}")
        except:
            errors.append(f"❌ Error calculating 'Diff (MU Reasons)' — please check values.")
                # ✅ Special validation for Diff (SL Reasons)
         # ✅ Validation for 'Diff (SL Reasons)' - REVISED CODE
        try:
            # Get the main "Sick Leave" value
            sick_leave_val = df_input.loc[df_input["Category"] == "Sick Leave", selected_date_col].values[0]
            sick_leave_val = float(sick_leave_val) if pd.notna(sick_leave_val) and sick_leave_val != '' else 0.0

            # Manually calculate Total Drivers (SL Reasons) from its sub-components in Python
            sl_subcategories = [
                "Flu/Fever", "BP", "Orthopedic", "Heart", "Weakness", "Eye",
                "Accident/Injuries", "Neuro/Paralysis (Sick Leave)", "Piles", "Diabetes",
                "Thyroid", "Gas", "Dental", "Ear", "Skin/Allergy", "General Surgery",
                "Obesity", "Cancer"
            ]
            calculated_sl_total = 0.0
            for sub_cat in sl_subcategories:
                # Ensure the subcategory exists in your DataFrame to avoid KeyError
                if sub_cat in df_input["Category"].values:
                    sub_val_raw = df_input.loc[df_input["Category"] == sub_cat, selected_date_col].values[0]
                    calculated_sl_total += float(sub_val_raw) if pd.notna(sub_val_raw) and sub_val_raw != '' else 0.0
            # Use a small tolerance for floating-point comparison
            tolerance = 1e-9 # A very small number close to zero
            if abs(calculated_sl_total - sick_leave_val) > tolerance:
                errors.append(f"❌ 'Diff (SL Reasons)' must be 0 on {selected_date_col}. (Sum of reasons: {calculated_sl_total}, Sick Leave: {sick_leave_val})")
        except Exception as e:
            errors.append(f"❌ Error validating 'Diff (SL Reasons)' components — please check values. Details: {e}")



        # ✅ Final error handling
        if errors:
            for e in errors:
                st.error(e)
            st.warning("⚠️ Please correct the above errors and resubmit.")
        else:
            success = save_to_db(grid_response, selected_depot, user_selected_date)
            if success:
                st.success("✅ All changes saved successfully")
                st.info("🔄 Reloading updated data...")
                st.experimental_rerun()
            else:
                st.error("❌ Failed to save changes. Please check the console for errors.")
