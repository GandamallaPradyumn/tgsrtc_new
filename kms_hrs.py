import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
from db_config import engine
#from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

MODEL_DIR = "/tmp/my_models/"+

# --- Page Config ---
#st.set_page_config(page_title="Driver Performance Dashboard", page_icon="📊", layout="wide")

def kms_hrs():

    if "user_depot" not in st.session_state:
        st.error("Unauthorized access. Please login again.")
        st.stop()

    sel_depot = st.session_state["user_depot"]

    st.title("Driver Performance Dashboard")

    st.markdown(
        f"""
        <div style="
            background-color:#e8f2ff;
            padding:12px 16px;
            border-radius:10px;
            border-left:6px solid #1f77b4;
            font-size:17px;
            font-weight:600;
            margin-bottom:12px;
        ">
            🏢 Logged in Depot:
            <span style="color:#1f77b4; font-size:18px;">
                {sel_depot}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    
    AVG_WORKING_DAYS_PER_MONTH = 22

    # =====================
    # Cached Load Functions
    # =====================
    @st.cache_data
    def load_data():
        """Load operational & health data from MySQL (with steering-hours mapping)"""

        # --- Daily operations (KMs only) ---
        oprs = pd.read_sql("""
            SELECT
                depot,
                employee_id,
                operations_date,
                service_number,
                opd_kms
            FROM daily_operations
        """, engine)

        # --- Steering hours master ---
        hrs = pd.read_sql("""
            SELECT
                depot,
                service_number,
                steering_hours,
                no_of_drvs
            FROM steering_hours
        """, engine)

        # --- Health data ---
        ghc = pd.read_sql("""
            SELECT
                employee_id,
                depot,
                bmi_interpret,
                blood_pressure_interpret,
                triglycerides_interpret,
                ecg_comment,
                smoke,
                alcohol,
                arthritis,
                asthama,
                final_grading
            FROM health
        """, engine)

        # Normalize column names
        oprs.columns = oprs.columns.str.lower()
        hrs.columns = hrs.columns.str.lower()
        ghc.columns = ghc.columns.str.lower()

        # ----------------------
        # Merge steering hours into operations
        # ----------------------
        oprs = pd.merge(
            oprs,
            hrs,
            on=["depot", "service_number"],
            how="left"
        )

        # Protect against zero drivers
        oprs["no_of_drvs"] = oprs["no_of_drvs"].replace(0, 1)

        # Apply driver-based hour logic
        oprs["hours"] = np.where(
            oprs["no_of_drvs"].fillna(1) >= 2,
            oprs["steering_hours"] / oprs["no_of_drvs"].fillna(1),
            oprs["steering_hours"]
        )

        # Cleanup
        oprs.drop(columns=["steering_hours", "no_of_drvs"], inplace=True)
        oprs["hours"] = oprs["hours"].fillna(oprs["hours"].median())

        # ----------------------
        # Date features
        # ----------------------
        oprs["operations_date"] = pd.to_datetime(oprs["operations_date"])
        oprs["year"] = oprs["operations_date"].dt.year
        oprs["month"] = oprs["operations_date"].dt.month
        oprs["day"] = oprs["operations_date"].dt.day

        # ----------------------
        # Merge with health
        # ----------------------
        combined = pd.merge(
            oprs,
            ghc,
            on=["employee_id", "depot"],
            how="inner"
        )

        return combined, ghc


    @st.cache_resource
    def load_models(version="v2_no_empid"):
        model_oprs = joblib.load(os.path.join(MODEL_DIR, "model_oprs.pkl"))
        model_hrs  = joblib.load(os.path.join(MODEL_DIR, "model_hrs.pkl"))
        encoders   = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
        base_year  = joblib.load(os.path.join(MODEL_DIR, "base_year.pkl"))
        return model_oprs, model_hrs, encoders, base_year


    # =====================
    # Load Data & Models
    # =====================

    combined_df, ghc_data = load_data()
    combined_df = combined_df[combined_df["depot"] == sel_depot]
    ghc_data = ghc_data[ghc_data["depot"] == sel_depot]

    model_oprs, model_hrs, encoders, base_year = load_models()

    if combined_df is None:
        st.stop()

    # =====================
    # Sidebar Filters
    # =====================

    st.sidebar.header("Primary Filters")

    emps = sorted(combined_df[combined_df["depot"] == sel_depot]["employee_id"].unique())
    sel_emp = st.sidebar.selectbox("Select Employee ID", emps)

    st.sidebar.markdown("### Year & Month Selection")
    pred_year = st.sidebar.number_input(
        "Prediction Year", min_value=base_year - 5, max_value=base_year + 10, value=datetime.now().year
    )
    pred_month = st.sidebar.selectbox(
        "Prediction Month",
        list(range(1, 13)),
        index=datetime.now().month - 1,
        format_func=lambda x: datetime(1900, x, 1).strftime("%B"),
    )

    # =====================
    # Health Parameters Grid (3x3)
    # =====================

    st.markdown("### Health Parameters")
    health_params = {}
    cols = st.columns(3)

    # Exclude employee_id and depot since they are already in sidebar
    health_cols = [c for c in encoders.keys() if c != "depot"]

    for i, col_name in enumerate(health_cols):
        with cols[i % 3]:
            vals = sorted(ghc_data[col_name].dropna().unique().astype(str))
            try:
                default = ghc_data[ghc_data["employee_id"] == sel_emp][col_name].iloc[-1]
                idx = vals.index(str(default)) if str(default) in vals else 0
            except Exception:
                idx = 0

            health_params[col_name] = st.selectbox(
                col_name.replace("_", " ").title(),
                vals,
                index=idx,
                key=col_name,
            )

    # =====================
    # Helper Functions
    # =====================

    def prepare_features(df, combined, base_year):
        """Prepare features for prediction: add averages and encode"""
        df2 = df.copy()

        # Year offset
        df2["year_offset"] = df2.get("year_offset", df2.get("year", 0) - base_year)

        # Employee-level averages
        emp_stats = combined.groupby("employee_id")[["opd_kms", "hours"]].mean().rename(
            columns={"opd_kms": "emp_avg_km", "hours": "emp_avg_hr"}
        )
        df2 = df2.merge(emp_stats, on="employee_id", how="left")

        # Depot-level averages
        depot_stats = combined.groupby("depot")[["opd_kms", "hours"]].mean().rename(
            columns={"opd_kms": "depot_avg_km", "hours": "depot_avg_hr"}
        )
        df2 = df2.merge(depot_stats, on="depot", how="left")

        df2["emp_avg_km"] = df2["emp_avg_km"].fillna(combined["opd_kms"].mean())
        df2["emp_avg_hr"] = df2["emp_avg_hr"].fillna(combined["hours"].mean())


        # Ratios
        # df2["km_ratio_emp"] = df2.get("opd_kms", 1) / df2["emp_avg_km"]
        # df2["hr_ratio_emp"] = df2.get("hours", 1) / df2["emp_avg_hr"]
        # df2["km_ratio_depot"] = df2.get("opd_kms", 1) / df2["depot_avg_km"]
        # df2["hr_ratio_depot"] = df2.get("hours", 1) / df2["depot_avg_hr"]

        # Encode categorical features
        for col, enc in encoders.items():
            if col not in df2.columns:
                df2[col] = "Unknown"
            known = list(enc.classes_)
            df2[col] = df2[col].apply(lambda x: x if x in known else "Unknown")
            if "Unknown" not in enc.classes_:
                enc.classes_ = np.append(enc.classes_, "Unknown")
            df2[col] = enc.transform(df2[col])

        # Add missing features expected by model
        for col in model_oprs.feature_names_in_:
            if col not in df2.columns:
                df2[col] = 0

        return df2[list(model_oprs.feature_names_in_)]

    # =====================
    # Tabs
    # =====================

    tab1, tab2, tab3 = st.tabs(["📅 Single Month", "📈 6-Month Forecast", "📊 Actual vs Predicted"])

    # -------------------------
    # TAB 1: Single Month Prediction
    # -------------------------
    with tab1:
        st.header("Monthly Performance Prediction")

        if st.button("Predict Performance", use_container_width=True):
            df = pd.DataFrame({
                "employee_id": [sel_emp],
                "depot": [sel_depot],
                "year_offset": [pred_year - base_year],
                "month": [pred_month],
                "day": [15],
            })

            for k, v in health_params.items():
                df[k] = [v]

            enc_df = prepare_features(df, combined_df, base_year)
            km = model_oprs.predict(enc_df)[0] * AVG_WORKING_DAYS_PER_MONTH
            hr = model_hrs.predict(enc_df)[0] * AVG_WORKING_DAYS_PER_MONTH

            st.success(f"Predicted for {datetime(pred_year, pred_month, 1).strftime('%B %Y')}")
            c1, c2 = st.columns(2)
            c1.metric("Monthly Kilometers", f"{km:,.0f} km")
            c2.metric("Monthly Hours", f"{hr:,.0f} hrs")

    # -------------------------
    # TAB 2: 6-Month Forecast
    # -------------------------
    with tab2:
        st.header("Next 6 Months Forecast")

        months, years = [], []
        for i in range(6):
            m = (pred_month + i - 1) % 12 + 1
            y = pred_year + ((pred_month + i - 1) // 12)
            months.append(m)
            years.append(y)

        fdata = []
        for y, m in zip(years, months):
            row = {"employee_id": sel_emp, "depot": sel_depot, "year_offset": y - base_year, "month": m, "day": 15}
            for k, v in health_params.items():
                row[k] = v
            fdata.append(row)

        fdf = pd.DataFrame(fdata)
        enc_fdf = prepare_features(fdf, combined_df, base_year)

        fdf["Pred_KMs"] = model_oprs.predict(enc_fdf) * AVG_WORKING_DAYS_PER_MONTH
        fdf["Pred_HRs"] = model_hrs.predict(enc_fdf) * AVG_WORKING_DAYS_PER_MONTH
        fdf["Month"] = [datetime(y, m, 1).strftime("%B %Y") for y, m in zip(years, months)]

        st.dataframe(fdf[["Month", "Pred_KMs", "Pred_HRs"]], use_container_width=True)

        fig1 = px.bar(fdf, x="Month", y="Pred_KMs", title="Forecasted Kilometers (Next 6 Months)")
        fig2 = px.bar(fdf, x="Month", y="Pred_HRs", title="Forecasted Hours (Next 6 Months)")
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

    # -------------------------
    # TAB 3: Actual vs Predicted (Historical)
    # -------------------------
    # with tab3:
    #     st.header("Actual vs Predicted - Historical Data")

    #     hist_df = combined_df[combined_df["employee_id"] == sel_emp].copy()
    #     hist_df["year_offset"] = hist_df["year"] - base_year

    #     encoded_hist = prepare_features(hist_df, combined_df, base_year)
    #     hist_df["Predicted_KMs"] = model_oprs.predict(encoded_hist)
    #     hist_df["Predicted_Hours"] = model_hrs.predict(encoded_hist)

    #     monthly_actual = (
    #         hist_df
    #         .groupby(["year", "month"])[["opd_kms", "hours"]]
    #         .mean()
    #         .reset_index()
    #     )

    #     monthly_pred = hist_df.groupby("month")[["Predicted_KMs", "Predicted_Hours"]].mean().reset_index()
    #     merged = pd.merge(monthly_actual, monthly_pred, on="month")
    #     merged["Month"] = merged["month"].apply(lambda x: datetime(1900, x, 1).strftime("%B"))

    #     st.dataframe(
    #         merged[["Month", "opd_kms", "Predicted_KMs", "hours", "Predicted_Hours"]],
    #         use_container_width=True
    #     )

    #     fig_km = px.line(merged, x="Month", y=["opd_kms", "Predicted_KMs"],
    #                      markers=True, title="Actual vs Predicted Kilometers")
    #     fig_hr = px.line(merged, x="Month", y=["hours", "Predicted_Hours"],
    #                      markers=True, title="Actual vs Predicted Hours")

    #     st.plotly_chart(fig_km, use_container_width=True)
    #     st.plotly_chart(fig_hr, use_container_width=True)

    with tab3:
        st.header("Actual vs Predicted – Last 12 Months")

        # -----------------------------
        # Filter employee data
        # -----------------------------
        hist_df = combined_df[combined_df["employee_id"] == sel_emp].copy()
        hist_df["year_offset"] = hist_df["year"] - base_year

        # Create proper date column
        hist_df["date"] = pd.to_datetime(
            dict(year=hist_df["year"], month=hist_df["month"], day=1)
        )

        # Last 12 months filter
        last_date = hist_df["date"].max()
        start_date = last_date - pd.DateOffset(months=11)

        hist_df = hist_df[hist_df["date"].between(start_date, last_date)]

        if hist_df.empty:
            st.warning("No data available for the last 12 months.")
            st.stop()

        # -----------------------------
        # Predictions
        # -----------------------------
        encoded_hist = prepare_features(hist_df, combined_df, base_year)
        hist_df["Predicted_KMs"] = model_oprs.predict(encoded_hist)
        hist_df["Predicted_Hours"] = model_hrs.predict(encoded_hist)

        # -----------------------------
        # Monthly aggregation
        # -----------------------------
        monthly = (
            hist_df
            .groupby("date")[["opd_kms", "hours", "Predicted_KMs", "Predicted_Hours"]]
            .mean()
            .reset_index()
            .sort_values("date")
        )

        monthly["Month"] = monthly["date"].dt.strftime("%b %Y")

        # -----------------------------
        # Table
        # -----------------------------
        st.dataframe(
            monthly[["Month", "opd_kms", "Predicted_KMs", "hours", "Predicted_Hours"]],
            use_container_width=True
        )

        # -----------------------------
        # Charts
        # -----------------------------
        fig_km = px.line(
            monthly,
            x="Month",
            y=["opd_kms", "Predicted_KMs"],
            markers=True,
            title="Actual vs Predicted Kilometers (Last 12 Months)"
        )

        fig_hr = px.line(
            monthly,
            x="Month",
            y=["hours", "Predicted_Hours"],
            markers=True,
            title="Actual vs Predicted Hours (Last 12 Months)"
        )

        st.plotly_chart(fig_km, use_container_width=True)
        st.plotly_chart(fig_hr, use_container_width=True)


        # Optional admin accuracy view
        # st.markdown("---")
        # if st.checkbox("Show Accuracy Metrics (Admin Only)"):
        #     mae_km = mean_absolute_error(merged["opd_kms"], merged["Predicted_KMs"])
        #     rmse_km = np.sqrt(mean_squared_error(merged["opd_kms"], merged["Predicted_KMs"]))
        #     r2_km = r2_score(merged["opd_kms"], merged["Predicted_KMs"])

        #     mae_hr = mean_absolute_error(merged["hours"], merged["Predicted_Hours"])
        #     rmse_hr = np.sqrt(mean_squared_error(merged["hours"], merged["Predicted_Hours"]))
        #     r2_hr = r2_score(merged["hours"], merged["Predicted_Hours"])

        #     st.json({
        #         "MAE_KMs": mae_km, "RMSE_KMs": rmse_km, "R2_KMs": r2_km,
        #         "MAE_Hours": mae_hr, "RMSE_Hours": rmse_hr, "R2_Hours": r2_hr
        #     })
