import streamlit as st
import pandas as pd
from datetime import date
import json
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from db_config import get_session
from models import InputData  # ORM model


def edit():
    # --------------------------- CONFIG ----------------------------
    with open("config.json") as f:
        config = json.load(f)

    category_to_column = config.get("category_to_column", {})
    editable_rows = list(category_to_column.keys())

    # --------------------------- ORM DATA LOADERS ----------------------------
    def get_all_depots():
        with get_session() as db:
            depots = db.query(InputData.depot_name).distinct().order_by(InputData.depot_name).all()
            return [d[0] for d in depots]

    def fetch_data(depot, date_val):
        with get_session() as db:
            data = (
                db.query(InputData)
                .filter(InputData.depot_name == depot)
                .filter(InputData.data_date == date_val)
                .first()
            )
            if not data:
                return pd.DataFrame()

            data_dict = {c: getattr(data, c) for c in data.__table__.columns.keys()}
            return pd.DataFrame(
                [{"Category": cat, "Value": data_dict.get(col, 0)} for cat, col in category_to_column.items()]
            )

    # --------------------------- ORM DELETE ----------------------------
    def delete_data(depot, date_val):
        with get_session() as db:
            record = (
                db.query(InputData)
                .filter(InputData.depot_name == depot)
                .filter(InputData.data_date == date_val)
                .first()
            )
            if record:
                db.delete(record)
                db.commit()
                return True
            return False

    # --------------------------- RECALCULATIONS ----------------------------
    def recalculate_fields(df):
        df = df.copy()

        def get_val(category):
            try:
                return float(df[df["Category"] == category]["Value"].values[0])
            except:
                return 0

        def set_val(category, value):
            df.loc[df["Category"] == category, "Value"] = round(value) if isinstance(value, (int, float)) else value

        # --- Calculations ---
        planned_services = get_val("Planned Services")
        actual_services = get_val("Actual Services")
        planned_km = get_val("Planned KM")
        actual_km = get_val("Actual KM")
        drivers_required = get_val("Drivers Required")
        planned_schedules = get_val("Planned Schedules")
        total_drivers = get_val("Total Drivers")
        medically_unfit = get_val("Medically Unfit")
        suspended_drivers = get_val("Suspended Drivers")

        weekly_off = get_val("Weekly Off & National Off")
        special_off = get_val("Special Off (Night Out/IC, Online)")
        training_pme = get_val("Training, PME(medical)")
        others = get_val("Others (SDI, DGT, LO, Parking,<br>Relief Van,Depot Spare,<br> Cargo, Releaving duty)")
        leave_absent = get_val("Leave & Absent")
        sick_leave = get_val("Sick Leave")
        spot_absent = get_val("Spot Absent")
        double_duty = get_val("Double Duty")
        off_cancellation = get_val("Off Cancellation")
        drivers_as_conductors = get_val("Drivers as Conductors")

        available_drivers_1 = total_drivers - medically_unfit - suspended_drivers
        set_val("Available Drivers-1", available_drivers_1)
        set_val("% Available Drivers-1", (available_drivers_1 / total_drivers) * 100 if total_drivers else 0)

        deductions = weekly_off + special_off + training_pme + others + leave_absent + sick_leave
        available_drivers_2 = available_drivers_1 - deductions
        set_val("Available Drivers-2", available_drivers_2)
        set_val("% Available Drivers-2", (available_drivers_2 / total_drivers) * 100 if total_drivers else 0)

        set_val("Service Variance", actual_services - planned_services)
        set_val("KM Variance", actual_km - planned_km)
        set_val("% Weekly Off & National Off", (weekly_off / total_drivers) * 100 if total_drivers else 0)
        set_val("% Special Off (Night Out/IC, Online)", (special_off / total_drivers) * 100 if total_drivers else 0)
        set_val("% Others", ((training_pme + others) / total_drivers) * 100 if total_drivers else 0)
        set_val("% Leave & Absent", (leave_absent / total_drivers) * 100 if total_drivers else 0)
        set_val("% Sick Leave", (sick_leave / total_drivers) * 100 if total_drivers else 0)

        attending_drivers = available_drivers_2 - spot_absent
        set_val("Attending Drivers", attending_drivers)
        set_val("% Attending Drivers", (attending_drivers / total_drivers) * 100 if total_drivers else 0)
        set_val("% Spot Absent", (spot_absent / total_drivers) * 100 if total_drivers else 0)

        shortage = max(0, drivers_required - attending_drivers)
        set_val("Driver shortage", shortage)
        set_val("Driver schedule", (drivers_required / planned_schedules) if planned_schedules else 0)
        set_val("% Double Duty", (double_duty / total_drivers) * 100 if total_drivers else 0)
        set_val("% Off Cancellation", (off_cancellation / total_drivers) * 100 if total_drivers else 0)

        drivers_on_duty = attending_drivers + double_duty + off_cancellation
        set_val("Drivers on Duty", drivers_on_duty)
        driver_for_bus_services = drivers_on_duty - drivers_as_conductors
        set_val("Driver for Bus Services", driver_for_bus_services)
        set_val("KM/Driver", (actual_km / driver_for_bus_services) if driver_for_bus_services else 0)
        set_val("Service/Driver Check", driver_for_bus_services - actual_services)

        mu_fields = ["Spondilitis", "Spinal Disc", "Vision/Color Blindness", "Neuro/Paralysis (Medical)", "Ortho"]
        sl_fields = [
            "Flu/Fever", "BP", "Orthopedic", "Heart", "Weakness", "Eye", "Accident/Injuries",
            "Neuro/Paralysis (Sick Leave)", "Piles", "Diabetes", "Thyroid", "Gas", "Dental",
            "Ear", "Skin/Allergy", "General Surgery", "Obesity", "Cancer"
        ]

        total_mu = sum(get_val(cat) for cat in mu_fields)
        set_val("Total Drivers (MU Reasons)", total_mu)
        set_val("Diff (MU Reasons)", total_mu - medically_unfit)

        total_sl = sum(get_val(cat) for cat in sl_fields)
        set_val("Total Drivers (SL Reasons)", total_sl)
        set_val("Diff (SL Reasons)", total_sl - sick_leave)

        return df

    # --------------------------- VALIDATION ----------------------------
    def validate(df):
        errors = []
        allow_negative = {"Service Variance", "KM Variance", "Service/Driver Check"}
        for _, row in df.iterrows():
            cat, val = row["Category"], row["Value"]
            try:
                val = int(float(val))
            except:
                errors.append(f"Invalid number: {cat}")
                continue
            if val < 0 and cat not in allow_negative:
                errors.append(f"Negative value not allowed: {cat}")
        return errors

    # --------------------------- ORM UPDATE ----------------------------
    def update_data(depot, date_val, df):
        with get_session() as db:
            record = (
                db.query(InputData)
                .filter(InputData.depot_name == depot)
                .filter(InputData.data_date == date_val)
                .first()
            )

            data_dict = {}
            for _, row in df.iterrows():
                cat = row["Category"]
                if cat in category_to_column:
                    col = category_to_column[cat]
                    try:
                        data_dict[col] = int(float(row["Value"]))
                    except:
                        data_dict[col] = 0

            if record:
                for k, v in data_dict.items():
                    setattr(record, k, v)
            else:
                record = InputData(depot_name=depot, data_date=date_val, **data_dict)
                db.add(record)

            db.commit()
        return True

    # --------------------------- UI ----------------------------
    st.title("🛠 Edit Saved Depot Data")

    depots = get_all_depots()
    selected_depot = st.selectbox("Select Depot", depots)
    selected_date = st.date_input("Select Date", value=date.today())

    # ✅ Load data button
    if st.button("🔍 Load Existing Data"):
        df = fetch_data(selected_depot, selected_date)
        if df.empty:
            st.warning("No data found — creating a new entry.")
            df = pd.DataFrame({"Category": editable_rows, "Value": [0] * len(editable_rows)})
        st.session_state.df = df.copy()
        st.session_state.data_loaded = True
        st.session_state.confirm_delete = False

    # ✅ Display data grid when loaded
    if st.session_state.get("data_loaded", False):
        df = st.session_state.df
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_column("Category", editable=False)
        gb.configure_column("Value", editable=True)

        grid = AgGrid(
            df,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.MODEL_CHANGED,
            theme="material",
            fit_columns_on_grid_load=True,
            key=f"grid_{selected_depot}_{selected_date}"
        )

        edited_df = recalculate_fields(grid["data"])

        col1, col2 = st.columns(2)

        # 💾 Save Button
        with col1:
            if st.button("💾 Save Changes"):
                errors = validate(edited_df)
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    success = update_data(selected_depot, selected_date, edited_df)
                    if success:
                        st.success("✅ Data updated successfully!")
                        refreshed_df = fetch_data(selected_depot, selected_date)
                        if not refreshed_df.empty:
                            st.markdown("### ✅ Saved Data Preview")
                            st.dataframe(refreshed_df, use_container_width=True)
                    else:
                        st.error("❌ Failed to update data.")

        # 🗑️ Delete Button with Confirmation
        with col2:
            if not st.session_state.get("confirm_delete", False):
                if st.button("🗑️ Delete Data"):
                    st.session_state.confirm_delete = True
                    st.warning("⚠️ Are you sure you want to delete this record? This action cannot be undone.")
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Yes, Delete It"):
                        if delete_data(selected_depot, selected_date):
                            st.success("🗑️ Record deleted successfully!")
                            st.session_state.data_loaded = False
                            st.session_state.confirm_delete = False
                        else:
                            st.error("❌ No record found to delete.")
                            st.session_state.confirm_delete = False
                with col_b:
                    if st.button("❌ Cancel"):
                        st.session_state.confirm_delete = False
                        st.info("Deletion cancelled.")
