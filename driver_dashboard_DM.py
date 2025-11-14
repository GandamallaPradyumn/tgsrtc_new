# ---------------------------------------------------------------
# driver_dashboard_DM.py  (FINAL ORM VERSION - PART 1 OF 4)
# ---------------------------------------------------------------

import base64
import pandas as pd
import altair as alt
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect
import json

# local imports
from db_config import get_session
from models import (
    DriverDetails,
    DailyOperations,
    ServiceMaster,
    DriverAbsenteeism,
    GHC2024
)

# ---------------------------------------------------------------
# ORM Connection Wrapper
# ---------------------------------------------------------------
class ORMConnection:
    def __init__(self):
        try:
            self.session = get_session()
        except Exception as e:
            st.error(f"❌ ORM connection failed: {e}")
            st.stop()

    def load(self, model, depot_col=None, depot_value=None):
        try:
            q = self.session.query(model)

            if depot_col and depot_value:
                # ORM attribute is lowercase (dept_time, arr_time, etc.)
                if hasattr(model, depot_col):
                    q = q.filter(getattr(model, depot_col) == depot_value)
                else:
                    # case-insensitive fallback
                    cols = [c.key for c in inspect(model).mapper.column_attrs]
                    match = next((c for c in cols if c.lower() == depot_col.lower()), None)
                    if match:
                        q = q.filter(getattr(model, match) == depot_value)

            rows = q.all()
            if not rows:
                return pd.DataFrame()

            # serialize rows
            data = [{col.key: getattr(r, col.key)
                     for col in inspect(model).mapper.column_attrs} for r in rows]

            df = pd.DataFrame(data)
            df.columns = df.columns.str.upper()  # dashboard requires uppercase

            return df

        except SQLAlchemyError as e:
            st.error(f"❌ ORM Query Error: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Unexpected ORM error: {e}")
            return pd.DataFrame()


# ---------------------------------------------------------------
# DASHBOARD MAIN CLASS
# ---------------------------------------------------------------
class driver_depot_dashboard_ui_DM:
    def __init__(self, user_depot, role, ops_df=None):
        self.user_depot = user_depot
        self.role = role
        self.ops_df = ops_df
        self.month_year = []

        # --------------------------
        # Header Logo
        # --------------------------
        try:
            with open("driver_dashboard_logo.png", "rb") as f:
                b64_img = base64.b64encode(f.read()).decode()
        except:
            b64_img = ""

        st.markdown(
            f"""
            <div style="text-align:center;">
                {"<img src='data:image/png;base64," + b64_img + "' width='2000' height='260'>" if b64_img else ""}
                <h1>Telangana State Road Transport Corporation</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Load ORM data
        if self.ops_df is None:
            self.load_data()

        # financial years
        self.financial_years = {
            "01-04-2023 to 31-03-2024": (pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
            "01-04-2024 to 31-03-2025": (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
        }

    # ---------------------------------------------------------------
    # LOAD DATA USING ORM
    # ---------------------------------------------------------------
    def load_data(self):
        orm = ORMConnection()

        # DRIVER DETAILS
        self.driver_df = orm.load(DriverDetails, depot_col="unit", depot_value=self.user_depot)
        if self.driver_df.empty:
            self.driver_df = orm.load(DriverDetails, depot_col="depot", depot_value=self.user_depot)

        # OPERATIONS
        self.ops_df = orm.load(DailyOperations, depot_col="depot", depot_value=self.user_depot)
        if self.ops_df.empty:
            self.ops_df = orm.load(DailyOperations, depot_col="unit", depot_value=self.user_depot)

        # SERVICE MASTER
        self.ser_df = orm.load(ServiceMaster, depot_col="depot", depot_value=self.user_depot)
        if self.ser_df.empty:
            self.ser_df = orm.load(ServiceMaster)

        # ABSENTEEISM
        self.abs_df = orm.load(DriverAbsenteeism, depot_col="depot", depot_value=self.user_depot)
        if self.abs_df.empty:
            self.abs_df = orm.load(DriverAbsenteeism)

        # HEALTH
        self.ghc1_df = orm.load(GHC2024, depot_col="depot", depot_value=self.user_depot)
        if self.ghc1_df.empty:
            self.ghc1_df = orm.load(GHC2024)

        # ----------------------------------------------------------
        # DATE CONVERSION
        # ----------------------------------------------------------
        if "OPERATIONS_DATE" in self.ops_df:
            self.ops_df["OPERATIONS_DATE"] = pd.to_datetime(self.ops_df["OPERATIONS_DATE"], errors="coerce")

        if "DATE" in self.abs_df:
            self.abs_df["DATE"] = pd.to_datetime(self.abs_df["DATE"], errors="coerce")

        # ----------------------------------------------------------
        # MONTH_YEAR creation
        # ----------------------------------------------------------
        self.ops_df = self.parse_and_format(self.ops_df, "OPERATIONS_DATE")
        self.abs_df = self.parse_and_format(self.abs_df, "DATE")

        # ----------------------------------------------------
        # FINAL FIX: TGSRTC Correct Hours Calculation
        # ----------------------------------------------------
        if not self.ser_df.empty:

            # Ensure strings
            self.ser_df["DEPT_TIME"] = self.ser_df["DEPT_TIME"].astype(str).str.strip()
            self.ser_df["ARR_TIME"]  = self.ser_df["ARR_TIME"].astype(str).str.strip()

            # Convert to time-of-day only (NOT datetime)
            self.ser_df["DEPT_TD"] = pd.to_timedelta(self.ser_df["DEPT_TIME"])
            self.ser_df["ARR_TD"]  = pd.to_timedelta(self.ser_df["ARR_TIME"])

            # If arrival occurs earlier → it is NEXT DAY
            mask = self.ser_df["ARR_TD"] < self.ser_df["DEPT_TD"]
            self.ser_df.loc[mask, "ARR_TD"] += pd.Timedelta(days=1)

            # Calculate hours
            diff = self.ser_df["ARR_TD"] - self.ser_df["DEPT_TD"]
            self.ser_df["HOURS"] = (diff.dt.total_seconds() / 3600).round(2)

            # If still negative or invalid, force to 0
            self.ser_df["HOURS"] = self.ser_df["HOURS"].clip(lower=0).fillna(0)

        else:
            self.ser_df["HOURS"] = 0




        # ----------------------------------------------------------
        # SERVICE NUMBER NORMALIZATION
        # ----------------------------------------------------------
        if "SERVICE_NUMBER" in self.ops_df:
            self.ops_df["SERVICE_NUMBER"] = (
                self.ops_df["SERVICE_NUMBER"].astype(str).str.strip().str.upper()
            )

        if "SERVICE_NUMBER" in self.ser_df:
            self.ser_df["SERVICE_NUMBER"] = (
                self.ser_df["SERVICE_NUMBER"].astype(str).str.strip().str.upper()
            )

        # ----------------------------------------------------------
        # MERGE HOURS INTO OPERATIONS (PRODUCTIVITY HOURS)
        # ----------------------------------------------------------
        if "SERVICE_NUMBER" in self.ops_df.columns:
            self.drv_hours = pd.merge(
                self.ops_df,
                self.ser_df[["SERVICE_NUMBER", "HOURS"]],
                on="SERVICE_NUMBER",
                how="left"
            )
            self.drv_hours["HOURS"] = self.drv_hours["HOURS"].fillna(0)
        else:
            self.drv_hours = self.ops_df.copy()
            self.drv_hours["HOURS"] = 0

        # ----------------------------------------------------------
        # FINANCIAL YEAR
        # ----------------------------------------------------------
        self.ops_df["FINANCIAL_YEAR"] = self.ops_df["OPERATIONS_DATE"].apply(
            lambda d: self.get_financial_year(d) if pd.notna(d) else None
        )
        self.max_date = self.ops_df["OPERATIONS_DATE"].max()

# ---------------------------------------------------------------
# driver_dashboard_DM.py  (FINAL ORM VERSION - PART 2 OF 4)
# ---------------------------------------------------------------

    # ---------------- Helpers continued ----------------
    def parse_and_format(self, df, date_col):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
            df = df.dropna(subset=[date_col])
            df["MONTH_YEAR"] = df[date_col].dt.to_period("M").dt.strftime("%Y-%m")
        else:
            # create empty MONTH_YEAR column so merges don't fail later
            df["MONTH_YEAR"] = pd.Series(dtype=str)
        return df

    def get_financial_year(self, date):
        if pd.isna(date):
            return None
        year = date.year
        if date.month < 4:
            return f"{year-1}-{year}"
        else:
            return f"{year}-{year+1}"

    @staticmethod
    def get_user_depot(session, userid):
        # optional helper: requires a User model in models.py
        try:
            from models import User
            u = session.query(User).filter(User.userid == userid).first()
            return u.depot if u else ""
        except Exception:
            return ""

    # ---------------- Parameters UI ----------------
    def parameters(self):
        with st.container():
            col1, col2, col3 = st.columns(3)

            # Depot selection (from session_state)
            with col1:
                st.markdown("### Depot")
                selected_depot = st.session_state.get("user_depot", None)
                if not selected_depot:
                    st.error("🚫 Depot not assigned. Please log in properly.")
                    st.stop()
                self.selected_depot = selected_depot
                st.markdown(f"<div style='background-color:#f0f2f6; padding:0.5em; border-radius:4px; font-weight:bold;'>{self.selected_depot}</div>", unsafe_allow_html=True)

            # Financial year selection
            with col2:
                st.markdown("### Financial Year")
                self.selected_fy = st.selectbox("", list(self.financial_years.keys()))
                self.fy_start, self.fy_end = self.financial_years[self.selected_fy]

            # Driver selection
            with col3:
                if "EMPLOYEE_ID" not in self.ops_df.columns:
                    st.warning("⚠ No drivers found for this depot.")
                    st.stop()
                drivers_in_depot = self.ops_df["EMPLOYEE_ID"].unique().tolist()
                if not drivers_in_depot:
                    st.warning("⚠ No drivers found for this depot.")
                    st.stop()
                st.markdown("### Driver ID")
                self.selected_driver = st.selectbox("", drivers_in_depot)

    # ---------------- Driver UI (main) ----------------
    def driver_ui(self):
        # prepare variables to avoid UnboundLocalError
        depot_kms_avg = 0
        depot_earnings_avg = 0
        depot_hours_avg = 0
        month_year_df = pd.DataFrame({"MONTH_YEAR": []})
        try:
            # build month list for the selected financial year (only months <= max_date)
            for i in pd.date_range(start=self.fy_start, end=self.fy_end, freq="MS"):
                if pd.notna(self.max_date) and self.max_date >= i:
                    self.month_year.append(i.strftime("%Y-%m"))
            month_year_df = pd.DataFrame({"MONTH_YEAR": self.month_year})

            # Filter driver operations and leaves for selected driver and FY
            drv_ops = self.ops_df[
                (self.ops_df["EMPLOYEE_ID"] == self.selected_driver)
                & (self.ops_df["OPERATIONS_DATE"] >= self.fy_start)
                & (self.ops_df["OPERATIONS_DATE"] <= self.fy_end)
            ]

            drv_leaves = self.abs_df[
                (self.abs_df["EMPLOYEE_ID"] == self.selected_driver)
                & (self.abs_df["DATE"] >= self.fy_start)
                & (self.abs_df["DATE"] <= self.fy_end)
            ]

            # Merge operations with service hours (productivity)
            if "SERVICE_NUMBER" in self.ops_df.columns and "SERVICE_NUMBER" in self.ser_df.columns:
                drv_hours = pd.merge(self.ops_df, self.ser_df[["SERVICE_NUMBER", "HOURS"]], on="SERVICE_NUMBER", how="left")
                drv_hours["HOURS"] = drv_hours["HOURS"].fillna(0)
            else:
                drv_hours = self.ops_df.copy()
                drv_hours["HOURS"] = 0

            drv_hours2 = drv_hours[
                (drv_hours["OPERATIONS_DATE"] >= self.fy_start) & (drv_hours["OPERATIONS_DATE"] <= self.fy_end)
            ]
            drv_hours = drv_hours[
                (drv_hours["EMPLOYEE_ID"] == self.selected_driver)
                & (drv_hours["OPERATIONS_DATE"] >= self.fy_start)
                & (drv_hours["OPERATIONS_DATE"] <= self.fy_end)
            ]

            # health merge
            drv_health = pd.DataFrame()
            if not self.ghc1_df.empty and "EMPLOYEE_ID" in self.ghc1_df.columns:
                drv_health = self.ghc1_df[self.ghc1_df["EMPLOYEE_ID"] == self.selected_driver]
                drv_ghcgrade = pd.merge(self.ghc1_df[["EMPLOYEE_ID", "FINAL_GRADING"]], drv_hours2[["EMPLOYEE_ID", "HOURS"]], on="EMPLOYEE_ID", how="right")
                drv_ghcgrade["HOURS"] = drv_ghcgrade["HOURS"].fillna(0)
            else:
                drv_ghcgrade = pd.DataFrame()

            # leaves summary across FY
            self.drv_leaves2 = self.abs_df[(self.abs_df["DATE"] >= self.fy_start) & (self.abs_df["DATE"] <= self.fy_end)]
            if not self.drv_leaves2.empty and "EMPLOYEE_ID" in self.drv_leaves2.columns:
                self.drv_leaves2 = self.drv_leaves2.groupby("EMPLOYEE_ID")["LEAVE_TYPE"].count().reset_index(name="LEAVE_COUNT")
            else:
                self.drv_leaves2 = pd.DataFrame(columns=["EMPLOYEE_ID", "LEAVE_COUNT"])

            drv_lsa_ghc = pd.merge(self.drv_leaves2, self.ghc1_df[["EMPLOYEE_ID", "FINAL_GRADING"]], on="EMPLOYEE_ID", how="inner") if not self.ghc1_df.empty else pd.DataFrame()

            # Depot-wide averages for the selected FY
            depot_ops_time = self.ops_df[
                (self.ops_df["OPERATIONS_DATE"] >= self.fy_start) & (self.ops_df["OPERATIONS_DATE"] <= self.fy_end)
            ]
            depot_kms_avg = depot_ops_time["OPD_KMS"].mean() if not depot_ops_time.empty and "OPD_KMS" in depot_ops_time.columns else 0
            depot_earnings_avg = depot_ops_time["DAILY_EARNINGS"].mean() if not depot_ops_time.empty and "DAILY_EARNINGS" in depot_ops_time.columns else 0
            depot_hours_avg = drv_hours2["HOURS"].mean() if not drv_hours2.empty else 0

        except Exception as e:
            st.error(f"Data not Available! {e}")

        # ---------------- Driver Details & Summary ----------------
        try:
            driver_info = self.driver_df[self.driver_df["EMPLOYEE_ID"] == str(self.selected_driver)]
            col_det, col_sum = st.columns(2)

            with col_det:
                st.markdown("## Driver Details")
                if not driver_info.empty:
                    info_row = driver_info.iloc[0]
                    st.write(f"**Name:** {info_row.get('FULL_NAME', 'N/A')}")
                    st.write(f"**Age:** {info_row.get('AGE', 'N/A')}")
                    st.write(f"**Birth Date:** {info_row.get('BIRTH_DATE', 'N/A')}")
                    st.write(f"**Joining Date:** {info_row.get('JOINING_DATE', 'N/A')}")
                    st.write(f"**Gender:** {info_row.get('GENDER', 'N/A')}")
                    st.write(f"**Marital Status:** {info_row.get('MARITAL_STATUS', 'N/A')}")
                else:
                    st.info("No driver details found.")
        except Exception as e:
            st.error(f"No data available for the employee! {e}")

        with col_sum:
            st.markdown("## Performance Summary")
            total_kms = drv_ops["OPD_KMS"].mean() if "OPD_KMS" in drv_ops.columns else 0
            total_earnings = drv_ops["DAILY_EARNINGS"].mean() if "DAILY_EARNINGS" in drv_ops.columns else 0
            total_hours = drv_hours["HOURS"].mean() if "HOURS" in drv_hours.columns else 0
            lsa_leaves = f"{(drv_leaves['LEAVE_TYPE'] == 'L').sum()} + {(drv_leaves['LEAVE_TYPE'] == 'S').sum()} + {(drv_leaves['LEAVE_TYPE'] == 'A').sum()}"

            depot_kms_avg = depot_kms_avg if depot_kms_avg else 0
            depot_earnings_avg = depot_earnings_avg if depot_earnings_avg else 0
            depot_hours_avg = depot_hours_avg if depot_hours_avg else 0

            # safe division
            earnings_per_km = (total_earnings / total_kms) if total_kms else 0
            depot_earnings_per_km = (depot_earnings_avg / depot_kms_avg) if depot_kms_avg else 0

            st.markdown(
                f"""
                <table style="width:100%;font-size:18px;">
                    <tr>
                        <td><b>Driver Avg KMs per day</b></td>
                        <td style="color:#1957a6; text-align:right;"><b>{total_kms:,.2f}</b></td>
                    </tr>
                    <tr>
                        <td style="font-size:14px; color:#888;">Depot Avg KMs per day</td>
                        <td style="font-size:14px; color:#888; text-align:right;">{depot_kms_avg:,.2f}</td>
                    </tr>
                    <tr style="height:8px;"><td colspan="2"></td></tr>
                    <tr>
                        <td><b>Driver Avg Earnings per Km</b></td>
                        <td style="color:#1957a6; text-align:right;"><b>₹{earnings_per_km:,.2f}</b></td>
                    </tr>
                    <tr>
                        <td style="font-size:14px; color:#888;">Depot Avg Earnings per Km</td>
                        <td style="font-size:14px; color:#888; text-align:right;">₹{depot_earnings_per_km:,.2f}</td>
                    </tr>
                    <tr style="height:8px;"><td colspan="2"></td></tr>
                    <tr>
                        <td><b>Driver Avg Hours per day</b></td>
                        <td style="color:#1957a6; text-align:right;"><b>{total_hours:,.2f}</b></td>
                    </tr>
                    <tr>
                        <td style="font-size:14px; color:#888;">Avg Depot Hours per day</td>
                        <td style="font-size:14px; color:#888; text-align:right;">{depot_hours_avg:,.2f}</td>
                    </tr>
                    <tr>
                        <td><b>Leave Days Taken (L+S+A)</b></td>
                        <td style="text-align:right;"><b>{lsa_leaves}</b></td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True,
            )

        # ---------------- chart legend helper ----------------
        def chart_legend(label, bar_color, label2=None, bar_color2=None, avg_label="Average Line: Red"):
            s = "<div style='display:flex;align-items:center;gap:24px;margin:10px 0 20px 0;'>"
            s += f"<span style='display:inline-block;width:35px;height:14px;background:{bar_color};margin-right:8px;border-radius:2px;'></span>"
            s += f"<span style='font-size:15px;'>{label}</span>"
            if label2 and bar_color2:
                s += f"<span style='display:inline-block;width:35px;height:14px;background:{bar_color2};margin-left:15px;margin-right:8px;border-radius:2px;'></span>"
                s += f"<span style='font-size:15px;'>{label2}</span>"
            s += "<span style='display:inline-block;width:36px;height:0;border-top:5px dashed red;margin-left:15px;'></span>"
            s += f"<span style='font-size:15px;color:red;'>{avg_label}</span></div>"
            st.markdown(s, unsafe_allow_html=True)
# ---------------------------------------------------------------
# driver_dashboard_DM.py  (FINAL ORM VERSION - PART 3 OF 4)
# ---------------------------------------------------------------

        # ---------------- MONTHLY KMS ----------------
        st.markdown("### Monthly Kilometers Driven")

        if "OPD_KMS" in drv_ops.columns:
            monthly_kms = drv_ops.groupby("MONTH_YEAR")["OPD_KMS"].sum().reset_index()
        else:
            monthly_kms = pd.DataFrame({"MONTH_YEAR": [], "OPD_KMS": []})

        # outer merge with month list → ensures missing months appear
        monthly_kms = pd.merge(monthly_kms, month_year_df, on="MONTH_YEAR", how="outer").fillna(0)
        total_kms_period = monthly_kms["OPD_KMS"].sum()
        avg_kms = monthly_kms["OPD_KMS"].mean() if not monthly_kms.empty else 0

        st.markdown(
            f"<div style='font-size:20px;color:#1957a6;'><b>Total Kilometers:</b> {total_kms_period:,.2f} KMs</div>",
            unsafe_allow_html=True,
        )

        # Chart: Monthly KMs
        bars = alt.Chart(monthly_kms).mark_bar(color="#1f77b4").encode(
            x=alt.X("MONTH_YEAR:N", sort=month_year_df["MONTH_YEAR"].tolist(), title="Month-Year"),
            y=alt.Y("OPD_KMS:Q", title="Kilometers"),
            tooltip=["MONTH_YEAR", "OPD_KMS"],
        )

        kms_text = alt.Chart(monthly_kms).mark_text(
            align="center", baseline="bottom", dy=-5, color="black", fontSize=14, fontWeight="bold"
        ).encode(
            x=alt.X("MONTH_YEAR:N", sort=month_year_df["MONTH_YEAR"].tolist()),
            y=alt.Y("OPD_KMS:Q"),
            text=alt.Text("OPD_KMS:Q", format=".0f")
        )

        avg_line = alt.Chart(pd.DataFrame({"OPD_KMS": [avg_kms]})).mark_rule(
            color="red", strokeDash=[5, 5]
        ).encode(y="OPD_KMS:Q")

        avg_text = alt.Chart(
            pd.DataFrame({"OPD_KMS": [avg_kms], "label": [f"Avg: {avg_kms:.1f}"]})
        ).mark_text(align="left", dx=5, dy=-7, color="red").encode(
            y="OPD_KMS:Q", text="label:N"
        )

        st.altair_chart((bars + kms_text + avg_line + avg_text).properties(width=900), use_container_width=True)
        chart_legend("Bar: Blue", "#1f77b4")

        # ---------------- MONTHLY EARNINGS ----------------
        if "DAILY_EARNINGS" in drv_ops.columns:
            st.markdown("### Monthly Earnings")

            monthly_earnings = drv_ops.groupby("MONTH_YEAR")["DAILY_EARNINGS"].sum().reset_index()
            monthly_earnings = pd.merge(monthly_earnings, month_year_df, on="MONTH_YEAR", how="outer").fillna(0)

            total_earnings_period = monthly_earnings["DAILY_EARNINGS"].sum()
            avg_earn = monthly_earnings["DAILY_EARNINGS"].mean()

            st.markdown(
                f"<div style='font-size:20px;color:#1957a6;'><b>Total Earnings:</b> ₹{total_earnings_period:,.2f}</div>",
                unsafe_allow_html=True,
            )

            bars2 = alt.Chart(monthly_earnings).mark_bar(color="#1f77b4").encode(
                x="MONTH_YEAR:N",
                y=alt.Y("DAILY_EARNINGS:Q", title="Earnings"),
                tooltip=["MONTH_YEAR", "DAILY_EARNINGS"],
            )

            earnings_text = alt.Chart(monthly_earnings).mark_text(
                align="center", baseline="bottom", dy=-5, color="black",
                fontSize=14, fontWeight="bold"
            ).encode(
                x=alt.X("MONTH_YEAR:N", sort=month_year_df["MONTH_YEAR"].tolist()),
                y="DAILY_EARNINGS:Q",
                text=alt.Text("DAILY_EARNINGS:Q", format=".0f")
            )

            avg_line2 = alt.Chart(pd.DataFrame({"DAILY_EARNINGS": [avg_earn]})).mark_rule(
                color="red", strokeDash=[5, 5]
            ).encode(y="DAILY_EARNINGS:Q")

            avg_text2 = alt.Chart(
                pd.DataFrame({"DAILY_EARNINGS": [avg_earn], "label": [f"Avg: {avg_earn:.1f}"]})
            ).mark_text(
                align="left", dx=5, dy=-7, color="red"
            ).encode(y="DAILY_EARNINGS:Q", text="label:N")

            st.altair_chart((bars2 + earnings_text + avg_line2 + avg_text2).properties(width=900), use_container_width=True)
            chart_legend("Bar: Blue", "#1f77b4")

        # ---------------- DAY VS NIGHT DUTIES ----------------
        if "DAY_NIGHT" in drv_ops.columns:
            st.markdown("### Day vs Night Duties")

            dn_summary = drv_ops.groupby(["DAY_NIGHT", "MONTH_YEAR"]).size().reset_index(name="Count")

            # Ensure both D & N appear for all months
            all_combinations = pd.MultiIndex.from_product(
                [["D", "N"], month_year_df["MONTH_YEAR"]],
                names=["DAY_NIGHT", "MONTH_YEAR"]
            ).to_frame(index=False)

            dn_summary = pd.merge(all_combinations, dn_summary, on=["DAY_NIGHT", "MONTH_YEAR"], how="left").fillna(0)

            total_day = int(dn_summary[dn_summary["DAY_NIGHT"] == "D"]["Count"].sum())
            total_night = int(dn_summary[dn_summary["DAY_NIGHT"] == "N"]["Count"].sum())

            st.markdown(
                f"<span style='font-size:20px;color:#1957a6;'><b>Day Total:</b> {total_day}  &nbsp;&nbsp;&nbsp;  <b>Night Total:</b> {total_night}</span>",
                unsafe_allow_html=True,
            )

            bars_dn = alt.Chart(dn_summary).mark_bar().encode(
                x=alt.X("MONTH_YEAR:N", title="Month-Year", sort=month_year_df["MONTH_YEAR"].tolist()),
                y=alt.Y("Count:Q", title="Duties"),
                color=alt.Color(
                    "DAY_NIGHT:N",
                    scale=alt.Scale(
                        domain=["D", "N"],
                        range=["#1f77b4", "#5A00FF"]
                    ),
                    legend=None,
                ),
                tooltip=["DAY_NIGHT", "MONTH_YEAR", "Count"],
            )

            text_labels = alt.Chart(dn_summary[dn_summary["Count"] > 0]).mark_text(
                align="center", baseline="bottom", dy=-5, color="black", fontSize=13, fontWeight="bold"
            ).encode(
                x=alt.X("MONTH_YEAR:N", sort=month_year_df["MONTH_YEAR"].tolist()),
                y="Count:Q",
                text=alt.Text("Count:Q", format=".0f"),
            )

            day_avg = dn_summary[dn_summary["DAY_NIGHT"] == "D"]["Count"].mean()
            night_avg = dn_summary[dn_summary["DAY_NIGHT"] == "N"]["Count"].mean()
            avg_df = pd.DataFrame([
                {"Count": day_avg, "DAY_NIGHT": "D", "label": f"Day Avg: {day_avg:.1f}"},
                {"Count": night_avg, "DAY_NIGHT": "N", "label": f"Night Avg: {night_avg:.1f}"},
            ])

            avg_lines = alt.Chart(avg_df).mark_rule(strokeDash=[4, 2], color="red").encode(y="Count:Q")
            avg_texts = alt.Chart(avg_df).mark_text(
                align="left", dx=5, dy=-7, color="red"
            ).encode(y="Count:Q", text="label:N")

            st.altair_chart((bars_dn + text_labels + avg_lines + avg_texts).properties(width=900), use_container_width=True)
            chart_legend("Day: Blue", "#1f77b4", "Night: Purple", "#5A00FF")

        # ---------------- PRODUCTIVITY HOURS ----------------
        if "HOURS" in drv_hours.columns and not drv_hours.empty:
            st.markdown("### Productivity Hours")

            hours_monthly = drv_hours.groupby("MONTH_YEAR")["HOURS"].sum().reset_index()
            hours_monthly = pd.merge(hours_monthly, month_year_df, on="MONTH_YEAR", how="outer").fillna(0)

            total_hours = hours_monthly["HOURS"].sum()
            avg_hours = hours_monthly["HOURS"].mean()

            st.markdown(
                f"<div style='font-size:20px;color:#1957a6;'><b>Total Hours:</b> {total_hours} hrs</div>",
                unsafe_allow_html=True,
            )

            hours_bars = alt.Chart(hours_monthly).mark_bar(color="#1f77b4").encode(
                x="MONTH_YEAR:N",
                y="HOURS:Q",
                tooltip=["MONTH_YEAR", "HOURS"],
            )

            hours_text = alt.Chart(hours_monthly).mark_text(
                align="center", baseline="bottom", dy=-5, color="black", fontSize=14, fontWeight="bold"
            ).encode(
                x=alt.X("MONTH_YEAR:N", sort=month_year_df["MONTH_YEAR"].tolist()),
                y="HOURS:Q",
                text=alt.Text("HOURS:Q", format=".0f"),
            )

            hours_avg_line = alt.Chart(pd.DataFrame({"HOURS": [avg_hours]})).mark_rule(
                color="red", strokeDash=[4, 2]
            ).encode(y="HOURS:Q")

            hours_avg_text = alt.Chart(
                pd.DataFrame({"HOURS": [avg_hours], "label": [f"Avg: {avg_hours:.1f}"]})
            ).mark_text(
                align="left", dx=5, dy=-7, color="red"
            ).encode(y="HOURS:Q", text="label:N")

            st.altair_chart((hours_bars + hours_text + hours_avg_line + hours_avg_text).properties(width=900), use_container_width=True)
            chart_legend("Bar: Blue", "#1f77b4")

        else:
            st.info("No hours data available for selected filters.")

# ---------------------------------------------------------------
# driver_dashboard_DM.py  (FINAL ORM VERSION - PART 4 OF 4)
# ---------------------------------------------------------------

        # ---------------- PRODUCTIVITY BY HEALTH GRADE ----------------
        st.markdown("---")
        st.header("**Productivity (Hours) vs Health Grade (GHC2)**")

        if not self.ghc1_df.empty and not drv_hours2.empty:
            required = ["EMPLOYEE_ID", "FINAL_GRADING"]
            if not all(col in self.ghc1_df.columns for col in required):
                st.warning("GHC dataset missing required columns.")
            else:
                ghc_merge = pd.merge(
                    self.ghc1_df[["EMPLOYEE_ID", "FINAL_GRADING"]],
                    drv_hours2[["EMPLOYEE_ID", "HOURS"]],
                    on="EMPLOYEE_ID",
                    how="right"
                )
                ghc_merge["HOURS"] = ghc_merge["HOURS"].fillna(0)

                if not ghc_merge.empty:
                    grp = ghc_merge.groupby(["EMPLOYEE_ID", "FINAL_GRADING"], as_index=False)["HOURS"].sum()

                    box = alt.Chart(grp).mark_boxplot(size=20).encode(
                        x=alt.X("FINAL_GRADING:N", title="Health Grade"),
                        y=alt.Y("HOURS:Q", title="Annual Hours"),
                    ).properties(
                        title=alt.TitleParams(
                            text=f"Productivity (Hours/Yr) by GHC Grade – {self.selected_depot}",
                            anchor="middle"
                        )
                    )

                    points = alt.Chart(grp).mark_point(color="red", size=30).encode(
                        x=alt.X("FINAL_GRADING:N"),
                        y="HOURS:Q",
                        tooltip=["EMPLOYEE_ID", "HOURS"],
                    ).transform_calculate(
                        jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
                    )

                    highlight = alt.Chart(grp[grp["EMPLOYEE_ID"] == self.selected_driver]).mark_point(
                        color="yellow", size=200, filled=True
                    ).encode(
                        x="FINAL_GRADING:N",
                        y="HOURS:Q"
                    )

                    final = box + points + highlight
                    st.altair_chart(final, use_container_width=True)
                else:
                    st.info("No productivity data available.")
        else:
            st.info("Health or hours data not available.")

    #    # ---------------------------------------------------------------
    #     # ABSENTEEISM (DAYS) + HEALTH GRADE (GHC2)  **FIXED**
    #     # ---------------------------------------------------------------
    #     st.markdown("---")
    #     st.header("**Absenteeism (Days) + Health Grade (GHC2)**")

    #     # STEP 1: Filter only L,S,A leaves
    #     abs_filtered = self.abs_df[
    #         (self.abs_df["DATE"] >= self.fy_start) &
    #         (self.abs_df["DATE"] <= self.fy_end) &
    #         (self.abs_df["LEAVE_TYPE"].isin(["L","S","A"]))
    #     ]

    #     if abs_filtered.empty:
    #         st.warning("No absenteeism records (L+S+A) found.")
    #     else:
    #         # STEP 2: Count L+S+A per employee
    #         leave_summary = abs_filtered.groupby("EMPLOYEE_ID").size().reset_index(name="LEAVE_COUNT")

    #         # STEP 3: Merge with Health grades
    #         if "EMPLOYEE_ID" not in self.ghc1_df.columns:
    #             st.error("Health dataset missing EMPLOYEE_ID column!")
    #         elif "FINAL_GRADING" not in self.ghc1_df.columns:
    #             st.error("Health dataset missing FINAL_GRADING column!")
    #         else:
    #             abs_grade = leave_summary.merge(
    #                 self.ghc1_df[["EMPLOYEE_ID", "FINAL_GRADING"]],
    #                 on="EMPLOYEE_ID",
    #                 how="left"
    #             )

    #             # remove employees with no health grade
    #             abs_grade = abs_grade.dropna(subset=["FINAL_GRADING"])

    #             if abs_grade.empty:
    #                 st.warning("No matching health grades for absenteeism records.")
    #             else:
    #                 # STEP 4: Box Plot
    #                 box_plot2 = (
    #                     alt.Chart(abs_grade)
    #                     .mark_boxplot(size=25)
    #                     .encode(
    #                         x=alt.X("FINAL_GRADING:N", title="Health Grade", sort=["A","B","C","D"]),
    #                         y=alt.Y("LEAVE_COUNT:Q", title="Leave Days (L+S+A)")
    #                     )
    #                     .properties(
    #                         title=f"Absenteeism by Health Grade (Days/Yr): {self.selected_depot}"
    #                     )
    #                 )

    #                 # STEP 5: Swarm Points
    #                 swarm_plot2 = (
    #                     alt.Chart(abs_grade)
    #                     .mark_circle(size=40, opacity=0.9, color="red")
    #                     .encode(
    #                         x=alt.X("FINAL_GRADING:N"),
    #                         y=alt.Y("LEAVE_COUNT:Q"),
    #                         tooltip=["EMPLOYEE_ID", "LEAVE_COUNT"]
    #                     )
    #                     .transform_calculate(jitter="(random() - 0.5) * 0.4")
    #                 )

    #                 # STEP 6: Highlight selected driver
    #                 sel = abs_grade[abs_grade["EMPLOYEE_ID"] == self.selected_driver]
    #                 highlight = (
    #                     alt.Chart(sel)
    #                     .mark_point(size=200, filled=True, color="yellow", stroke="black", strokeWidth=2)
    #                     .encode(
    #                         x="FINAL_GRADING:N",
    #                         y="LEAVE_COUNT:Q"
    #                     )
    #                 )

    #                 st.altair_chart(box_plot2 + swarm_plot2 + highlight, use_container_width=True)



    # =====================================================================
    # ==================== DRIVER DEPOT UI (ALL DRIVERS) ==================
    # =====================================================================
    def driver_depot_ui(self):

        # DEPOT-WIDE FILTER
        drv_ops_depot = self.ops_df[
            (self.ops_df["OPERATIONS_DATE"] >= self.fy_start) &
            (self.ops_df["OPERATIONS_DATE"] <= self.fy_end)
        ]

        # MERGE HOURS
        if "SERVICE_NUMBER" in self.ops_df.columns and "SERVICE_NUMBER" in self.ser_df.columns:
            depot_hours = pd.merge(
                self.ops_df,
                self.ser_df[["SERVICE_NUMBER", "HOURS"]],
                on="SERVICE_NUMBER",
                how="left"
            )
            depot_hours["HOURS"] = depot_hours["HOURS"].fillna(0)
            depot_hours = depot_hours[
                (depot_hours["OPERATIONS_DATE"] >= self.fy_start) &
                (depot_hours["OPERATIONS_DATE"] <= self.fy_end)
            ]
        else:
            depot_hours = drv_ops_depot.copy()
            depot_hours["HOURS"] = 0

        # ---------------- GLOBAL AVERAGES ----------------
        kms_avg = drv_ops_depot["OPD_KMS"].mean() if "OPD_KMS" in drv_ops_depot.columns else 0
        earn_avg = drv_ops_depot["DAILY_EARNINGS"].mean() if "DAILY_EARNINGS" in drv_ops_depot.columns else 0
        hours_avg = depot_hours["HOURS"].mean() if "HOURS" in depot_hours.columns else 0

        # ---------------- DRIVER INFORMATION ----------------
        st.markdown("## Driver Information Inside Depot")

        driver_info = self.driver_df[self.driver_df["EMPLOYEE_ID"] == str(self.selected_driver)]
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Driver Details")
            if not driver_info.empty:
                r = driver_info.iloc[0]
                st.write(f"**Name:** {r.get('FULL_NAME','N/A')}")
                st.write(f"**Age:** {r.get('AGE','N/A')}")
                st.write(f"**Birth Date:** {r.get('BIRTH_DATE','N/A')}")
                st.write(f"**Joining Date:** {r.get('JOINING_DATE','N/A')}")
                st.write(f"**Gender:** {r.get('GENDER','N/A')}")
                st.write(f"**Marital Status:** {r.get('MARITAL_STATUS','N/A')}")
            else:
                st.info("Driver info missing.")

        with col2:
            st.markdown("### Depot Avg Performance")

            drv_ops = drv_ops_depot[drv_ops_depot["EMPLOYEE_ID"] == self.selected_driver]
            total_kms = drv_ops["OPD_KMS"].mean() if "OPD_KMS" in drv_ops.columns else 0
            total_earn = drv_ops["DAILY_EARNINGS"].mean() if "DAILY_EARNINGS" in drv_ops.columns else 0
            total_hours = depot_hours[depot_hours["EMPLOYEE_ID"] == self.selected_driver]["HOURS"].mean()

            st.write(f"**Driver Avg KMs:** {total_kms:.2f}   (Depot Avg: {kms_avg:.2f})")
            st.write(f"**Driver Avg Earnings:** ₹{total_earn:.2f}   (Depot Avg: ₹{earn_avg:.2f})")
            st.write(f"**Driver Avg Hours:** {total_hours:.2f}   (Depot Avg: {hours_avg:.2f})")

        # ---------------- DEPOT MONTHLY KMS ----------------
        if "OPD_KMS" in drv_ops_depot.columns:
            st.markdown("### Depot – Monthly Kilometers")

            depot_km = drv_ops_depot.groupby("MONTH_YEAR")["OPD_KMS"].sum().reset_index()
            depot_km = pd.merge(depot_km, pd.DataFrame({"MONTH_YEAR": self.month_year}), on="MONTH_YEAR", how="outer").fillna(0)

            bar = alt.Chart(depot_km).mark_bar(color="#1f77b4").encode(
                x="MONTH_YEAR:N",
                y="OPD_KMS:Q",
                tooltip=["MONTH_YEAR", "OPD_KMS"]
            )

            st.altair_chart(bar.properties(width=900), use_container_width=True)

        # ---------------- DEPOT DAY/NIGHT ----------------
        if "DAY_NIGHT" in drv_ops_depot.columns:
            st.markdown("### Depot – Day vs Night Duties")

            dn = drv_ops_depot.groupby(["DAY_NIGHT", "MONTH_YEAR"]).size().reset_index(name="Count")

            all_combo = pd.MultiIndex.from_product(
                [["D", "N"], self.month_year],
                names=["DAY_NIGHT", "MONTH_YEAR"]
            ).to_frame(index=False)

            dn = pd.merge(all_combo, dn, on=["DAY_NIGHT", "MONTH_YEAR"], how="left").fillna(0)

            dn_chart = alt.Chart(dn).mark_bar().encode(
                x="MONTH_YEAR:N",
                y="Count:Q",
                color=alt.Color("DAY_NIGHT:N", scale=alt.Scale(domain=["D", "N"], range=["#1f77b4", "#5A00FF"])),
                tooltip=["DAY_NIGHT", "MONTH_YEAR", "Count"]
            )

            st.altair_chart(dn_chart.properties(width=900), use_container_width=True)

# ---------------------------------------------------------------
# ENTRY POINT FOR STREAMLIT
# ---------------------------------------------------------------
if __name__ == "__main__":
    user_depot = st.session_state.get("user_depot", None)
    role = st.session_state.get("role", "DM")

    if not user_depot:
        st.error("❌ User depot missing in session. Login again.")
        st.stop()

    obj = driver_depot_dashboard_ui_DM(user_depot, role)

    # UI
    obj.parameters()

    tab1, tab2 = st.tabs(["Driver Performance", "Depot-Wide Performance"])

    with tab1:
        obj.driver_ui()

    with tab2:
        obj.driver_depot_ui()
