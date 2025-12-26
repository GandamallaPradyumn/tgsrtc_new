import json
import os
import streamlit as st
import pandas as pd
from datetime import timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

# expected project imports - adjust if your module names differ
from auth import get_depot_settings
from db_config import get_session
from models import TSAdmin, InputData

# Load config.json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)


def _normalize_name(s):
    """Normalize names for matching: str, strip, lower, underscores -> spaces, collapse spaces, remove punctuation."""
    s = "" if s is None else str(s)
    s = s.strip().lower().replace("_", " ")
    for ch in [".", ",", "-", "(", ")", "%", "/"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def _build_normalized_maps():
    """Return helpful normalized maps:
       - norm_dbcol -> category (from category_to_column)
       - norm_category_row -> category (for reverse lookup)
    """
    norm_dbcol_to_cat = {}
    for cat, dbcol in config_data.get("category_to_column", {}).items():
        norm = _normalize_name(dbcol)
        norm_dbcol_to_cat[norm] = cat

    norm_category_row = {}
    for cat in config_data.get("category_rows", []):
        norm = _normalize_name(cat)
        norm_category_row[norm] = cat

    return norm_dbcol_to_cat, norm_category_row


def _apply_grid_style(df, depot_type, date_columns):
    """DM-style AgGrid formatting (copied / adapted from your RM_sheet)."""
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=False, sortable=False, wrapText=False, autoHeight=False, editable=False)

    # Category styling
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
                if (category && category.startsWith('---')) {
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

    # Depot Type column with benchmark
    normalized_depot_type = depot_type.strip().title() if depot_type else "N/A"
    current_bench = config_data.get("benchmarks", {}).get(normalized_depot_type, {})
    gb.configure_column(
        field="Depot Type",
        header_name="Rural/Urban",
        pinned="left",
        lockPinned=True,
        editable=False,
        width=140,
        resizable=False,
        sortable=False,
        cellStyle=JsCode("""
            function(params) {
                let style = {"background-color": "#f0f8ff", "border": "1px solid #d3d3d3"};
                if (params.data && params.data.Category && params.data.Category.startsWith('---')) {
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
                const category = params.data ? params.data.Category : null;
                const currentBenchmarks = {json.dumps(current_bench)};
                let benchmarkValue = null;
                if (currentBenchmarks && category && currentBenchmarks.hasOwnProperty(category)) {{
                    benchmarkValue = currentBenchmarks[category];
                }}
                if (benchmarkValue !== null && category && category.includes('%')) {{
                    return 'Benchmark - ' + benchmarkValue + '%';
                }} else if (category && category.startsWith('---')) {{
                    return params.value;
                }} else {{
                    return '';
                }}
            }}
        """)
    )

    # Date columns formatting
    for col in date_columns:
        try:
            header = pd.to_datetime(col).strftime("%d-%b-%y")
        except Exception:
            header = col
        gb.configure_column(
            field=col,
            header_name=header,
            editable=False,
            type=["numericColumn", "rightAligned"],
            resizable=False,
            sortable=False,
            cellStyle=JsCode("""
                function(params) {
                    const category = params.data ? params.data.Category : null;
                    const value = params.value;
                    let style = {"border": "1px solid #d3d3d3"};
                    if (category && category.startsWith('---')) {
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
                        if (typeof value === 'number' && value < 0) { style["color"] = "red"; }
                    } else if (category && category.includes('%')) {
                        style["background-color"] = "#e0e0f0";
                    }
                    return style;
                }
            """)
        )

    return gb.build()


def _fetch_inputdata_by_depots(dep_list, date_columns):
    """Fetch InputData rows for given depots and dates as DataFrame using ORM session"""
    try:
        with get_session() as db:
            rows = (
                db.query(InputData)
                .filter(InputData.depot_name.in_(dep_list))
                .filter(InputData.data_date.in_(date_columns))
                .all()
            )
            if not rows:
                return pd.DataFrame()
            # convert ORM objects to dict rows (exclude private attrs)
            df = pd.DataFrame([{k: v for k, v in r.__dict__.items() if not k.startswith("_")} for r in rows])
            # normalize column names to strings
            df.columns = [str(c).strip() for c in df.columns]
            return df
    except Exception as e:
        st.error(f"DB fetch error: {e}")
        return pd.DataFrame()


def corporation_dashboard():
    st.title("TGSRTC PRODUCTIVITY DASHBOARD (CORPORATION VIEW)")

    selection = st.radio(
        "Select View Mode",
        [
            "All Depots Combined",
            "Region Wise",
            "Rural Depots",
            "Urban Depots",
            "Individual Depots"
        ],
        horizontal=True
    )

    depot_config = get_depot_settings()

    # get all depots + region mapping
    with get_session() as db:
        rows = db.query(TSAdmin.depot_name, TSAdmin.region).order_by(TSAdmin.depot_name).all()
    if not rows:
        st.error("No depots found in TS_ADMIN.")
        return
    depot_rows = [(r.depot_name, r.region) for r in rows]
    depots_all = [d for d, _ in depot_rows]

    # Individual Depot view
    if selection == "Individual Depots":
        conn_list = depots_all  # all for selection upstream, but we will restrict to region in RM version
        # list of depots
        selected_depot = st.selectbox("Select Depot", depots_all)
        if not selected_depot:
            st.warning("Select a depot.")
            return

        # get latest date
        with get_session() as db:
            latest = db.query(InputData.data_date).filter(InputData.depot_name == selected_depot).order_by(InputData.data_date.desc()).first()
        latest_date = latest[0] if latest else None
        if latest_date is None:
            st.warning("No data found for selected depot.")
            return
        latest_date = pd.to_datetime(latest_date).date()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From Date", value=latest_date - timedelta(days=9))
        with col2:
            end_date = st.date_input("To Date", value=latest_date)

        if start_date > end_date:
            st.error("Start date cannot be after end date.")
            return

        date_columns = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()

        # build blank df
        df = pd.DataFrame({"Category": config_data.get("category_rows", [])})
        depot_type = depot_config.get(selected_depot, {}).get("category", "N/A")
        df["Depot Type"] = depot_type
        for d in date_columns:
            df[d] = None

        # fetch existing data for this depot
        existing = _fetch_inputdata_by_depots([selected_depot], date_columns)
        #st.write("Fetched columns (individual):", list(existing.columns))
        if not existing.empty:
            existing["data_date"] = pd.to_datetime(existing["data_date"]).dt.strftime("%Y-%m-%d")
            # st.write("Sample rows (individual):")
            # st.dataframe(existing.head())

        # tolerant matching fill
        norm_dbcol_to_cat, norm_category_row = _build_normalized_maps()
        column_match_map = {}
        for col_date in date_columns:
            row = existing[existing["data_date"] == col_date]
            if row.empty:
                continue
            for db_col, val in row.iloc[0].items():
                if str(db_col).lower() in ("data_date", "depot_name") or pd.isna(val):
                    continue
                norm_db = _normalize_name(db_col)

                # try several strategies
                cat = norm_dbcol_to_cat.get(norm_db) or norm_category_row.get(norm_db)
                if not cat:
                    for norm_row, orig_cat in norm_category_row.items():
                        if norm_db == norm_row or norm_db in norm_row or norm_row in norm_db:
                            cat = orig_cat
                            break
                if not cat:
                    compact = norm_db.replace(" ", "")
                    for norm_row, orig_cat in norm_category_row.items():
                        if compact == norm_row.replace(" ", ""):
                            cat = orig_cat
                            break

                column_match_map[db_col] = cat
                if not cat:
                    continue

                idx = df[df["Category"].str.strip().str.lower() == str(cat).strip().lower()].index
                if not idx.empty:
                    df.at[idx[0], col_date] = val

        if column_match_map:
            dbg = pd.DataFrame(list(column_match_map.items()), columns=["db_column", "matched_category_or_None"])
            # st.write("DB column → matched category (individual):")
            # st.dataframe(dbg)

        grid_options = _apply_grid_style(df, depot_type, date_columns)
        AgGrid(df, gridOptions=grid_options, update_mode=GridUpdateMode.NO_UPDATE,
               theme="material", height=700, allow_unsafe_jscode=True,
               enable_enterprise_modules=False, fit_columns_on_grid_load=False)

        return

    # For the remaining selections, build depot subset
    if selection == "Rural Depots":
        depots = [d for d, _ in depot_rows if depot_config.get(d, {}).get("category", "").lower() == "rural"]
        label = "Rural"
    elif selection == "Urban Depots":
        depots = [d for d, _ in depot_rows if depot_config.get(d, {}).get("category", "").lower() == "urban"]
        label = "Urban"
    elif selection == "Region Wise":
        regions = sorted(set(r for _, r in depot_rows))
        selected_region = st.selectbox("Select Region", regions)
        depots = [d for d, r in depot_rows if r == selected_region]
        label = selected_region
    else:  # All Depots Combined
        depots = depots_all
        label = "All Depots"

    if not depots:
        st.warning(f"No depots found for {label}.")
        return

    # latest date across chosen depots
    with get_session() as db:
        latest = db.query(InputData.data_date).filter(InputData.depot_name.in_(depots)).order_by(InputData.data_date.desc()).first()
    latest_date = latest[0] if latest else None
    if latest_date is None:
        st.warning("No data found for selected depots.")
        return
    latest_date = pd.to_datetime(latest_date).date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From Date", value=latest_date - timedelta(days=9), key=f"{label}_from")
    with col2:
        end_date = st.date_input("To Date", value=latest_date, key=f"{label}_to")

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        return

    date_columns = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()

    # build blank df
    df = pd.DataFrame({"Category": config_data.get("category_rows", [])})
    df["Depot Type"] = label
    for d in date_columns:
        df[d] = None

    # fetch aggregated data across depots
    existing = _fetch_inputdata_by_depots(depots, date_columns)
    #st.write("Fetched columns (aggregated):", list(existing.columns))
    if not existing.empty:
        # normalize date and show a sample
        if "data_date" in existing.columns:
            existing["data_date"] = pd.to_datetime(existing["data_date"]).dt.strftime("%Y-%m-%d")
        #st.write("Sample rows (aggregated):")
        #st.dataframe(existing.head())

    # grouping + tolerant mapping + aggregation logic
    if not existing.empty and "data_date" in existing.columns:
        norm_dbcol_to_cat, norm_category_row = _build_normalized_maps()
        # group numeric columns by date
        grouped = existing.groupby("data_date").sum(numeric_only=True).reset_index()
        depots_per_date = existing.groupby("data_date")["depot_name"].nunique().to_dict()
        percentage_categories = [cat for cat in config_data.get("category_rows", []) if "%" in cat]

        column_match_map = {}
        for col_date in date_columns:
            row = grouped[grouped["data_date"] == col_date]
            if row.empty:
                continue
            for db_col, val in row.iloc[0].items():
                if str(db_col).lower() in ("data_date", "depot_name") or pd.isna(val):
                    continue
                norm_db = _normalize_name(db_col)
                cat = norm_dbcol_to_cat.get(norm_db) or norm_category_row.get(norm_db)
                if not cat:
                    for norm_row, orig_cat in norm_category_row.items():
                        if norm_db == norm_row or norm_db in norm_row or norm_row in norm_db:
                            cat = orig_cat
                            break
                if not cat:
                    compact = norm_db.replace(" ", "")
                    for norm_row, orig_cat in norm_category_row.items():
                        if compact == norm_row.replace(" ", ""):
                            cat = orig_cat
                            break

                column_match_map[db_col] = cat
                if not cat:
                    continue

                idx = df[df["Category"].str.strip().str.lower() == str(cat).strip().lower()].index
                if idx.empty:
                    continue

                if cat in percentage_categories:
                    n_depots = depots_per_date.get(col_date, 1) or 1
                    try:
                        numeric_val = float(val)
                    except Exception:
                        numeric_val = 0
                    df.at[idx[0], col_date] = round(numeric_val / n_depots, 2)
                else:
                    df.at[idx[0], col_date] = val

        if column_match_map:
            dbg = pd.DataFrame(list(column_match_map.items()), columns=["db_column", "matched_category_or_None"])
            # st.write("Aggregated DB column → matched category (None = no match):")
            # st.dataframe(dbg)

    grid_options = _apply_grid_style(df, label, date_columns)
    AgGrid(df, gridOptions=grid_options, update_mode=GridUpdateMode.NO_UPDATE,
           theme="material", height=700, allow_unsafe_jscode=True,
           enable_enterprise_modules=False, fit_columns_on_grid_load=False)


if __name__ == "__main__":
    corporation_dashboard()
