import base64
import pandas as pd
import altair as alt
import mysql.connector
import streamlit as st
from mysql.connector import Error
import json


class Sql_connection:

    def __init__(self, config_path="config.json"):
        try:
            with open(config_path) as f:
                config = json.load(f)

            # Only pick DB keys
            allowed_keys = {"host", "user", "password", "database", "port"}
            self.DB_CONFIG = {k: v for k, v in config["db"].items() if k in allowed_keys}

        except FileNotFoundError:
            st.error(f"⚠ Config file not found: {config_path}")
            st.stop()
        except Exception as e:
            st.error(f"⚠ Error reading config: {e}")
            st.stop()

        self._conn = self.connect()

    def connect(self):
        try:
            conn = mysql.connector.connect(**self.DB_CONFIG)
            if conn.is_connected():
                return conn
        except Error as err:
            st.error(f"❌ Error connecting to MySQL: {err}")
            st.error("Dashboard cannot load data. Please check your MySQL connection details and server status.")
            st.stop()
        return None

    def load_sql_data(self, table_name, depot_col='depot', depot_name=None):
        try:
            query = f"SELECT * FROM {table_name} where {depot_col} = '{depot_name}'"
            cursor = self._conn.cursor()
            cursor.execute(query)
            columns = [desc[0].upper() for desc in cursor.description]  # Column names in uppercase
            result = cursor.fetchall()
            df = pd.DataFrame(result, columns=columns)
            return df if not df.empty else pd.DataFrame()
        except Error as e:
            st.error(f"⚠ SQL error: {e}")
            return pd.DataFrame()
        finally:
            cursor.close()


class driver_depot_dashboard_ui_DM:
    def __init__(self, user_depot, role, ops_df=None):
        self.user_depot = user_depot
        self.role = role
        self.ops_df = ops_df
        self.month_year = []
        file_path = r"driver_dashboard_logo.png"
        try:
            with open(file_path, "rb") as img_file:
                b64_img = base64.b64encode(img_file.read()).decode()
        except FileNotFoundError:
            b64_img = ""  # no logo file

        st.markdown(f"""
                        <div style="text-align: center; border-radius: 12px; border-width: 2px; border-color: black; padding: 0px;">
                            {"<img src='data:image/png;base64," + b64_img + "' width='2000' height='270' style='display:block; margin:0 auto;'>" if b64_img else ""}
                            <h1 style="color: black; margin:6px 0 8px 0;">Telangana State Road Transport Corporation</h1>
                        </div>
                    """, unsafe_allow_html=True)

        # Automatically load data if ops_df not provided
        if self.ops_df is None:
            self.load_data()

    def parameters(self):
        with st.container():

            # ------------------------------------------------------
            # DEPOT SELECTION (FIRST)
            # ------------------------------------------------------
            st.markdown("###  Depot")

            selected_depot = st.session_state.get("user_depot", None)

            if not selected_depot:
                st.error("🚫 Depot not assigned. Please log in again.")
                st.stop()

            self.selected_depot = selected_depot

            st.markdown(
                f"<div style='background-color:#eef2f8; padding:5px; border-radius:6px; "
                f"font-weight:bold; font-size:18px;'>{self.selected_depot}</div>",
                unsafe_allow_html=True
            )

            st.markdown("---")

            # ------------------------------------------------------
            # MONTH & YEAR SELECTION
            # ------------------------------------------------------
            st.markdown("###  Select Month & Year:")
            col1,col2=st.columns(2)
            with col1:

                years = [2025, 2024, 2023, 2022]
                self.selected_year = st.selectbox(
                    "Select Year",
                    years,
                    index=0,
                    key="dm_year_select",
                )
            with col2:
                months = {
                    "All": 0,
                    "January": 1, "February": 2, "March": 3, "April": 4,
                    "May": 5, "June": 6, "July": 7, "August": 8,
                    "September": 9, "October": 10, "November": 11, "December": 12
                }

                month_list = list(months.keys())

                self.selected_month = st.selectbox(
                    "Select Month",
                    month_list,
                    index=0,                # default = All
                    key="dm_month_select"
                )

                m = months[self.selected_month]

            # date calculation
            if self.selected_month == "All":
                self.start_date = pd.Timestamp(year=self.selected_year, month=1, day=1)
                self.end_date = pd.Timestamp(year=self.selected_year, month=12, day=31)
            else:
                self.start_date = pd.Timestamp(year=self.selected_year, month=m, day=1)
                self.end_date = self.start_date + pd.offsets.MonthEnd(1)

            st.info(f"📆 Selected period: **{self.start_date.date()}** → **{self.end_date.date()}**")

            st.markdown("---")

            # ------------------------------------------------------
            # DRIVER SELECTION
            # ------------------------------------------------------
            st.markdown("### Select Employee Id:")

            if "EMPLOYEE_ID" not in self.ops_df.columns:
                st.warning("No drivers found in this depot.")
                st.stop()

            drivers = sorted(self.ops_df["EMPLOYEE_ID"].unique().tolist())

            self.selected_driver = st.selectbox(
                "Employee Id",
                drivers,
                key="dm_driver_select"
            )


    # ---------------- Data Loading ----------------
    def load_data(self):
        sql = Sql_connection()

        # Load each dataset safely
        self.driver_df = sql.load_sql_data('driver_details', 'unit', self.user_depot)
        self.ops_df = sql.load_sql_data('daily_operations', 'depot', self.user_depot)
        self.ser_df = sql.load_sql_data('service_master', 'depot', self.user_depot)
        self.abs_df = sql.load_sql_data('driver_absenteeism', 'depot', self.user_depot)
        self.ghc1_df = sql.load_sql_data('ghc_2024', 'depot', self.user_depot)

        # --- Helper for safe datetime parsing ---
        def safe_date_format(df, col):
            if df.empty:
                return df
            if col not in df.columns:
                st.warning(f"⚠ Column '{col}' missing in dataset. Skipping date conversion.")
                return df
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df.dropna(subset=[col], inplace=True)
            df['MONTH_YEAR'] = df[col].dt.to_period('M').astype(str)
            return df

        # --- Operations Data ---
        if self.ops_df.empty:
            st.error("❌ 'operations' data unavailable for this depot.")
            self.ops_df = pd.DataFrame(columns=['EMPLOYEE_ID', 'OPERATIONS_DATE', 'OPD_KMS', 'DAILY_EARNINGS', 'DAY_NIGHT'])
        else:
            # Handle missing date column
            if 'OPERATIONS_DATE' not in self.ops_df.columns:
                st.warning("⚠ 'OPERATIONS_DATE' column missing in daily_operations table.")
                self.ops_df['OPERATIONS_DATE'] = pd.NaT
            self.ops_df = safe_date_format(self.ops_df, 'OPERATIONS_DATE')
            self.ops_df['FINANCIAL_YEAR'] = self.ops_df['OPERATIONS_DATE'].apply(self.get_financial_year)
            self.max_date = self.ops_df['OPERATIONS_DATE'].max()

        # --- Absenteeism Data ---
        if not self.abs_df.empty:
            self.abs_df = safe_date_format(self.abs_df, 'DATE')
        else:
            st.info("No absenteeism records found for this depot.")

        # --- Service Master ---
        if not self.ser_df.empty:
            if {'DEPT_TIME', 'ARR_TIME'}.issubset(self.ser_df.columns):
                self.ser_df['HOURS'] = (self.ser_df['DEPT_TIME'] - self.ser_df['ARR_TIME']).dt.total_seconds().abs() / 3600
            else:
                st.warning("⚠ Missing 'DEPT_TIME' or 'ARR_TIME' columns in service_master.")
                self.ser_df['HOURS'] = 0
        else:
            st.info("No service master data available.")

        # --- GHC / Health Data ---
        if self.ghc1_df.empty:
            st.info("No GHC health data found for this depot.")

        # --- Driver Master ---
        if self.driver_df.empty:
            st.warning("⚠ No driver details available for this depot.")

            # Convert dates
            self.ops_df['OPERATIONS_DATE'] = pd.to_datetime(self.ops_df['OPERATIONS_DATE'], errors='coerce')
            self.abs_df['DATE'] = pd.to_datetime(self.abs_df['DATE'], errors='coerce')

            # Format and add MONTH_YEAR
            self.ops_df = self.parse_and_format(self.ops_df, 'OPERATIONS_DATE')
            self.abs_df = self.parse_and_format(self.abs_df, 'DATE')
            self.ser_df['HOURS'] = self.ser_df['DEPT_TIME'] - self.ser_df['ARR_TIME']
            self.ser_df['HOURS'] = abs(self.ser_df['HOURS'].dt.total_seconds() / 3600).round(2)
            # Add financial year
            self.ops_df['FINANCIAL_YEAR'] = self.ops_df['OPERATIONS_DATE'].apply(self.get_financial_year)
            self.max_date = self.ops_df['OPERATIONS_DATE'].max()
        

    # ---------------- Helpers ----------------
    def parse_and_format(self, df, date_col):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            df.dropna(subset=[date_col], inplace=True)
            df['MONTH_YEAR'] = df[date_col].dt.to_period('M').dt.strftime('%Y-%m')
        return df

    def get_financial_year(self, date):
        year = date.year
        if date.month < 4:
            return f"{year-1}-{year}"
        else:
            return f"{year}-{year+1}"

    @staticmethod
    def get_user_depot(_conn, userid):
        if not userid:
            return ""
        try:
            cursor = _conn.cursor()
            cursor.execute("SELECT depot FROM users WHERE userid = %s", (userid,))
            result = cursor.fetchone()
            return result[0] if result else ""
        except Error as e:
            st.error(f"Error fetching depot for user {userid} from 'users' table: {e}")
            return ""
        finally:
            cursor.close()

   

    
    def driver_ui(self):
    # --- Populate month_year for the selected financial year ---
        try:
            for i in pd.date_range(start=self.start_date, end=self.end_date, freq='MS'):
                if self.max_date >= i:
                    self.month_year.append(i.strftime('%Y-%m'))
            month_year_df = pd.DataFrame({'MONTH_YEAR':self.month_year})        
            # -- Filter data for selection --
            drv_ops = self.ops_df[
                (self.ops_df['EMPLOYEE_ID'] == self.selected_driver) &
                (self.ops_df['OPERATIONS_DATE'] >= self.start_date) & (self.ops_df['OPERATIONS_DATE'] <= self.end_date)
            ]
            drv_leaves = self.abs_df[
                (self.abs_df['EMPLOYEE_ID'] == self.selected_driver) &
                (self.abs_df['DATE'] >= self.start_date) & (self.abs_df['DATE'] <= self.end_date)
            ]
            drv_hours = pd.merge(self.ops_df, self.ser_df[['SERVICE_NUMBER','HOURS']],on='SERVICE_NUMBER',how='left')
            drv_hours['HOURS'] = drv_hours['HOURS'].fillna(0)
            drv_hours2 = drv_hours[(drv_hours['OPERATIONS_DATE'] >= self.start_date) & (drv_hours['OPERATIONS_DATE'] <= self.end_date)]
            drv_hours = drv_hours[(drv_hours['EMPLOYEE_ID'] == self.selected_driver) & (drv_hours['OPERATIONS_DATE'] >= self.start_date) & (drv_hours['OPERATIONS_DATE'] <= self.end_date)]
            drv_health = self.ghc1_df[self.ghc1_df['EMPLOYEE_ID'] == self.selected_driver] if 'EMPLOYEE_ID' in self.ghc1_df.columns else pd.DataFrame()
            drv_ghcgrade = pd.merge(self.ghc1_df[['EMPLOYEE_ID','AGE', 'FINAL_GRADING']], drv_hours2[['EMPLOYEE_ID','HOURS']], on='EMPLOYEE_ID', how='right')
            drv_ghcgrade['HOURS'] = drv_ghcgrade['HOURS'].fillna(0)
            self.drv_leaves2 = self.abs_df[
                (self.abs_df['DATE'] >= self.start_date) & (self.abs_df['DATE'] <= self.end_date)
            ]
            self.drv_leaves2 = self.drv_leaves2.groupby('EMPLOYEE_ID')['LEAVE_TYPE'].count()
            self.drv_leaves2 = self.drv_leaves2.reset_index()
            drv_lsa_ghc = pd.merge(self.drv_leaves2,self.ghc1_df[['EMPLOYEE_ID','FINAL_GRADING']], on='EMPLOYEE_ID',how='inner') 

            # Depot-wide averages for context
            depot_ops_time = self.ops_df[
                (self.ops_df['OPERATIONS_DATE'] >= self.start_date) & (self.ops_df['OPERATIONS_DATE'] <= self.end_date)
            ]
            depot_kms_avg = depot_ops_time['OPD_KMS'].mean() if not depot_ops_time.empty and 'OPD_KMS' in depot_ops_time.columns else 0
            depot_earnings_avg = depot_ops_time['DAILY_EARNINGS'].mean() if not depot_ops_time.empty and 'DAILY_EARNINGS' in depot_ops_time.columns else 0
            depot_hours_avg = drv_hours2['HOURS'].mean()
        except Exception as e:
            st.error(f'Data not Avaliavle!{e}!')

       
        try:
            driver_info = self.driver_df[self.driver_df['EMPLOYEE_ID'] == str(self.selected_driver)]
            col_det, col_sum = st.columns(2)
            with col_det:
                st.markdown("## Driver Details")
                if not driver_info.empty:
                    info_row = driver_info.iloc[0]
                    st.write(f"*Name:* {info_row.get('FULL_NAME', 'N/A')}")
                    #st.write(f"*Unit:* {info_row.get('UNIT', 'N/A')}")
                    st.write(f"*Age:* {info_row.get('AGE', 'N/A')}")
                    st.write(f"*Birth Date:* {info_row.get('BIRTH_DATE', 'N/A')}")
                    st.write(f"*Joining Date:* {info_row.get('JOINING_DATE', 'N/A')}")
                    st.write(f"*Gender:* {info_row.get('GENDER', 'N/A')}")
                    st.write(f"*Marital Status:* {info_row.get('MARITAL_STATUS', 'N/A')}")
                else:
                    st.info("No driver details found.")
        except Exception as e:
            st.error(f"No data available for the employee! {e}")
        

        with col_sum:
            st.markdown("## Performance Summary")

            # Totals
            total_kms = drv_ops['OPD_KMS'].mean() if 'OPD_KMS' in drv_ops.columns else 0
            total_earnings = drv_ops['DAILY_EARNINGS'].mean() if 'DAILY_EARNINGS' in drv_ops.columns else 0
            total_hours = drv_hours['HOURS'].mean() if 'HOURS' in drv_hours.columns else 0
            lsa_leaves = f"{(drv_leaves['LEAVE_TYPE'] == 'L').sum()} + {(drv_leaves['LEAVE_TYPE'] == 'S').sum()} + {(drv_leaves['LEAVE_TYPE'] == 'A').sum()}"

            # Depot-wide averages
            depot_ops_time = self.ops_df[
                (self.ops_df['OPERATIONS_DATE'].between(self.start_date, self.end_date))
            ]
            depot_kms_avg = depot_ops_time['OPD_KMS'].mean() if not depot_ops_time.empty and 'OPD_KMS' in depot_ops_time.columns else 0
            depot_earnings_avg = depot_ops_time['DAILY_EARNINGS'].mean() if not depot_ops_time.empty and 'DAILY_EARNINGS' in depot_ops_time.columns else 0
            depot_hours_avg = drv_hours2['HOURS'].mean() if not drv_hours2.empty else 0

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
                        <td style="color:#1957a6; text-align:right;"><b>₹{total_earnings/total_kms:,.2f}</b></td>
                    </tr>
                    <tr>
                        <td style="font-size:14px; color:#888;">Depot Avg Earnings per Km</td>
                        <td style="font-size:14px; color:#888; text-align:right;">₹{depot_earnings_avg/depot_kms_avg:,.2f}</td>
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
                unsafe_allow_html=True
            )


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

        # --------- Visualizations ---------
        # --- Monthly Kilometers ---
        st.markdown("### Monthly Kilometers Driven")
        monthly_kms = drv_ops.groupby('MONTH_YEAR')['OPD_KMS'].sum().reset_index()
        monthly_kms = pd.merge(monthly_kms, month_year_df, on='MONTH_YEAR',how='outer')
        monthly_kms = monthly_kms.fillna(0)
        total_kms_period = monthly_kms['OPD_KMS'].sum()
        st.markdown(f"<div style='font-size:20px;color:#1957a6;margin-bottom:0;'><b>Total Kilometers:</b> {total_kms_period:,.2f} KMs</div>", unsafe_allow_html=True)
        avg_kms = monthly_kms['OPD_KMS'].mean() if not monthly_kms.empty else 0
        bars = alt.Chart(monthly_kms).mark_bar(color='#1f77b4').encode(
            x=alt.X('MONTH_YEAR:N', sort=monthly_kms['MONTH_YEAR'].tolist(), title='Month-Year'),
            y=alt.Y('OPD_KMS:Q', title='Kilometers'),
            tooltip=['MONTH_YEAR', 'OPD_KMS']
        )
        kms_text = alt.Chart(monthly_kms).mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            color='black',
            fontSize=14,
            fontWeight='bold'
        ).encode(
            x=alt.X('MONTH_YEAR:N', sort=self.month_year),
            y=alt.Y('OPD_KMS:Q'),
            text=alt.Text('OPD_KMS:Q', format='.0f')
        )
        avg_line = alt.Chart(pd.DataFrame({'OPD_KMS':[avg_kms]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='OPD_KMS:Q')
        avg_text = alt.Chart(pd.DataFrame({'OPD_KMS':[avg_kms], 'label':[f'Avg: {avg_kms:.1f}']})).mark_text(
            align='left', dx=5, dy=-7, color='red'
        ).encode(
            y='OPD_KMS:Q',
            text='label:N'
        )
        st.altair_chart((bars + kms_text + avg_line + avg_text).properties(width=900), use_container_width=True)
        chart_legend("Bar: Blue", "#1f77b4", None, None, "Average Line: Red")

        # --- Monthly Earnings ---
        if 'DAILY_EARNINGS' in drv_ops.columns:
            st.markdown("### Monthly Earnings")
            monthly_earnings = drv_ops.groupby('MONTH_YEAR')['DAILY_EARNINGS'].sum().reset_index()
            monthly_earnings = pd.merge(monthly_earnings, month_year_df, on='MONTH_YEAR',how='outer')
            monthly_earnings = monthly_earnings.fillna(0)
            total_earnings_period = monthly_earnings['DAILY_EARNINGS'].sum()
            st.markdown(f"<div style='font-size:20px;color:#1957a6;margin-bottom:0;'><b>Total Earnings:</b> ₹{total_earnings_period:,.2f}</div>", unsafe_allow_html=True)
            avg_earn = monthly_earnings['DAILY_EARNINGS'].mean() if not monthly_earnings.empty else 0
            bars2 = alt.Chart(monthly_earnings).mark_bar(color="#1f77b4").encode(
                x='MONTH_YEAR:N',
                y=alt.Y('DAILY_EARNINGS:Q', title='Earnings'),
                tooltip=['MONTH_YEAR', 'DAILY_EARNINGS']
            )
            earnings_text = alt.Chart(monthly_earnings).mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black',
                fontSize=14,
                fontWeight='bold'
            ).encode(
                x=alt.X('MONTH_YEAR:N', sort=self.month_year),
                y=alt.Y('DAILY_EARNINGS:Q'),
                text=alt.Text('DAILY_EARNINGS:Q', format='.0f')
            )
            avg_line2 = alt.Chart(pd.DataFrame({'DAILY_EARNINGS':[avg_earn]})).mark_rule(color='red', strokeDash=[5,5]).encode(
                y='DAILY_EARNINGS:Q'
            )
            avg_text2 = alt.Chart(pd.DataFrame({'DAILY_EARNINGS':[avg_earn], 'label':[f'Avg: {avg_earn:.1f}']})).mark_text(
                align='left', dx=5, dy=-7, color='red'
            ).encode(
                y='DAILY_EARNINGS:Q',
                text='label:N'
            )
            st.altair_chart((bars2+ earnings_text + avg_line2 + avg_text2).properties(width=900), use_container_width=True)
            chart_legend("Bar: Blue", "#1f77b4", None, None, "Average Line: Red")

        # --- Day vs Night Duties ---
        if 'DAY_NIGHT' in drv_ops.columns:
            st.markdown("### Day vs Night Duties")

            # Group duties
            dn_summary = drv_ops.groupby(['DAY_NIGHT', 'MONTH_YEAR']).size().reset_index(name='Count')

            # Ensure every month has both D and N entries
            duty_types = ['D', 'N']
            all_combinations = pd.MultiIndex.from_product(
                [duty_types, month_year_df['MONTH_YEAR']],
                names=['DAY_NIGHT', 'MONTH_YEAR']
            ).to_frame(index=False)

            dn_summary = pd.merge(all_combinations, dn_summary, on=['DAY_NIGHT', 'MONTH_YEAR'], how='left').fillna(0)

            # Totals
            total_day = int(dn_summary[dn_summary['DAY_NIGHT'] == "D"]['Count'].sum())
            total_night = int(dn_summary[dn_summary['DAY_NIGHT'] == "N"]['Count'].sum())
            st.markdown(
                f"<span style='font-size:20px;color:#1957a6;'><b>Day Out Total:</b> {total_day} &nbsp;&nbsp;&nbsp;<b>Night Out Total:</b> {total_night}</span>",
                unsafe_allow_html=True
            )

            # Bar chart
            bars_dn = alt.Chart(dn_summary).mark_bar().encode(
                x=alt.X('MONTH_YEAR:N', title='Month-Year', sort=month_year_df['MONTH_YEAR'].tolist()),
                y=alt.Y('Count:Q', title='Duties'),
                color=alt.Color('DAY_NIGHT:N',
                                scale=alt.Scale(domain=["D", "N"], range=["#1f77b4", "#5A00FF"]),
                                legend=None),
                tooltip=['DAY_NIGHT', 'MONTH_YEAR', 'Count']
            )

            # Text labels (only for non-zero counts)
            text_labels = alt.Chart(dn_summary[dn_summary['Count'] > 0]).mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black',
                fontSize=13,
                fontWeight='bold'
            ).encode(
                x=alt.X('MONTH_YEAR:N', sort=month_year_df['MONTH_YEAR'].tolist()),
                y=alt.Y('Count:Q'),
                text=alt.Text('Count:Q', format='.0f'),
                
            )

            # Average lines
            day_avg = dn_summary[dn_summary['DAY_NIGHT'] == 'D']['Count'].mean()
            night_avg = dn_summary[dn_summary['DAY_NIGHT'] == 'N']['Count'].mean()

            avg_df = pd.DataFrame([
                {'Count': day_avg, 'DAY_NIGHT': 'D', 'label': f'Day Avg: {day_avg:.1f}'},
                {'Count': night_avg, 'DAY_NIGHT': 'N', 'label': f'Night Avg: {night_avg:.1f}'}
            ])

            avg_lines = alt.Chart(avg_df).mark_rule(strokeDash=[4, 2], color='red').encode(
                y='Count:Q'
            )

            avg_texts = alt.Chart(avg_df).mark_text(
                align='left',
                dx=5,
                dy=-7,
                color='red'
            ).encode(
                y='Count:Q',
                text='label:N'
            )

            # Combine and show chart
            st.altair_chart((bars_dn + text_labels + avg_lines + avg_texts).properties(width=900), use_container_width=True)

            chart_legend("Day Out: Blue", "#1f77b4", "Night Out: Purple", "#5A00FF", "Average Line: Red")


        
        # --- PRODUCTIVITY HOURS ---
        if not drv_hours.empty:
            st.markdown("### PRODUCTIVITY HOURS")
            hours_monthly = drv_hours.groupby('MONTH_YEAR')['HOURS'].sum()
            hours_monthly = pd.merge(hours_monthly, month_year_df, on='MONTH_YEAR',how='outer')
            hours_monthly = hours_monthly.fillna(0)
            total_hours = hours_monthly['HOURS'].sum()
            st.markdown(f"<div style='font-size:20px;color:#1957a6;margin-bottom:0;'><b>Total Hours:</b> {total_hours} hrs</div>", unsafe_allow_html=True)
            avg_hours = hours_monthly['HOURS'].mean() if not hours_monthly.empty else 0
            hours_bars = alt.Chart(hours_monthly).mark_bar(color="#1f77b4").encode(
                x='MONTH_YEAR:N', y='HOURS:Q', tooltip=['MONTH_YEAR', 'HOURS']
            )
            hours_text = alt.Chart(hours_monthly).mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black',
                fontSize=14,
                fontWeight='bold'
            ).encode(
                x=alt.X('MONTH_YEAR:N', sort=self.month_year),
                y=alt.Y('HOURS:Q'),
                text=alt.Text('HOURS:Q', format='.0f')
            )
            hours_avg_line = alt.Chart(pd.DataFrame({'HOURS': [avg_hours]})).mark_rule(color='red', strokeDash=[4,2]).encode(
                y='HOURS:Q'
            )
            hours_avg_text = alt.Chart(pd.DataFrame({'HOURS': [avg_hours], 'label': [f'Avg: {avg_hours:.1f}']})).mark_text(
                align='left', dx=5, dy=-7, color='red'
            ).encode(
                y='HOURS:Q',
                text='label:N'
            )
            st.altair_chart((hours_bars + hours_text + hours_avg_line + hours_avg_text).properties(width=900), use_container_width=True)
            chart_legend("Bar: Blue", "#1f77b4", None, None, "Average Line: Red")
        else:
            st.info("No hours data for selected filters.")
        
        # --- Absenteeism/Leave Monthly ---
        if not drv_leaves.empty:
            st.markdown("### Absenteeism / Leave Summary")
            leave_monthly = drv_leaves.groupby('MONTH_YEAR').size().reset_index(name='Leave_Days')
            leave_monthly = pd.merge(leave_monthly, month_year_df, on='MONTH_YEAR',how='outer')
            leave_monthly = leave_monthly.fillna(0)
            total_leaves_period = leave_monthly['Leave_Days'].sum()
            st.markdown(f"<div style='font-size:20px;color:#1957a6;margin-bottom:0;'><b>Total for Period:</b> {total_leaves_period} Days</div>", unsafe_allow_html=True)
            avg_leave = leave_monthly['Leave_Days'].mean() if not leave_monthly.empty else 0
            leave_bars = alt.Chart(leave_monthly).mark_bar(color="#1f77b4").encode(
                x='MONTH_YEAR:N', y='Leave_Days:Q', tooltip=['MONTH_YEAR', 'Leave_Days']
            )
            leaves_text = alt.Chart(leave_monthly).mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                color='black',
                fontSize=14,
                fontWeight='bold'
            ).encode(
                x=alt.X('MONTH_YEAR:N', sort=self.month_year),
                y=alt.Y('Leave_Days:Q'),
                text=alt.Text('Leave_Days:Q', format='.0f')
            )
            leave_avg_line = alt.Chart(pd.DataFrame({'Leave_Days': [avg_leave]})).mark_rule(color='red', strokeDash=[4,2]).encode(
                y='Leave_Days:Q'
            )
            leave_avg_text = alt.Chart(pd.DataFrame({'Leave_Days': [avg_leave], 'label': [f'Avg: {avg_leave:.1f}']})).mark_text(
                align='left', dx=5, dy=-7, color='red'
            ).encode(
                y='Leave_Days:Q',
                text='label:N'
            )
            st.altair_chart((leave_bars + leaves_text + leave_avg_line + leave_avg_text).properties(width=900), use_container_width=True)
            chart_legend("Bar: Blue", "#1f77b4", None, None, "Average Line: Red")
        else:
            st.info("No leave/absenteeism data for selected filters.")

        # PRODUCTIVITY BY HEALTH GRADE
        # Divider for clear sectioning
        st.markdown("---")

        #Absenteeism Baseline
        st.header("*Productivity (Hours) + Health Grade (GHC2)*") 
        
        
        
        if self.ghc1_df is not None:
            # Ensure the necessary columns exist
            required_cols = ['DEPOT', 'EMPLOYEE_ID', 'FINAL_GRADING']
            missing_cols = [col for col in required_cols if col not in self.ghc1_df.columns]
            if missing_cols:
                st.error(f"Missing required columns in dataset: {', '.join(missing_cols)}")
            else:
                # Filter and aggregate data
                drv_ghcgrade = pd.merge(
                    self.ghc1_df[['EMPLOYEE_ID', 'FINAL_GRADING']],
                    drv_hours2[['EMPLOYEE_ID', 'HOURS']],
                    on='EMPLOYEE_ID', how='right'
                )
                drv_ghcgrade['HOURS'] = drv_ghcgrade['HOURS'].fillna(0)

                if drv_ghcgrade.empty:
                    st.warning("No data available for the selected depot.")
                else:
                    # Aggregate hours per employee and grade
                    sorted_data3 = drv_ghcgrade.dropna(subset=['FINAL_GRADING']).groupby(
                        ['EMPLOYEE_ID', 'FINAL_GRADING'], as_index=False
                    )['HOURS'].sum()

                    # Box plot of hours by health grade
                    box_plot = alt.Chart(sorted_data3).mark_boxplot(size=20).encode(
                        x=alt.X('FINAL_GRADING:N', title='Health Grade'),
                        y=alt.Y('HOURS:Q', title='Annual Hours')
                    ).properties(
                        title=alt.TitleParams(
                            text=f'Productivity by Health Grade (Hours/Yr): {self.selected_depot}',
                            anchor='middle'
                        )
                    )

                    # Swarm plot (jittered points)
                    swarm_plot = alt.Chart(sorted_data3).mark_point(
                        color='red',
                        size=30
                    ).encode(
                        x=alt.X('FINAL_GRADING:N', axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('HOURS:Q'),
                        tooltip=[
                            alt.Tooltip('EMPLOYEE_ID', title='Employee ID'),
                            alt.Tooltip('HOURS', title='Annual Hours')
                        ]
                    ).transform_calculate(
                        jitter='sqrt(-2*log(random()))*cos(2*PI*random())'  # Jitter simulation
                    ).encode(
                        x=alt.X('FINAL_GRADING:N', title='Health Grade'),
                        y=alt.Y('HOURS:Q')
                    )

                    # Highlight the selected employee
                    highlighted_employee = alt.Chart(
                        sorted_data3[sorted_data3['EMPLOYEE_ID'] == self.selected_driver]
                    ).mark_point(
                        color='yellow',
                        size=200,
                        filled=True
                    ).encode(
                        x='FINAL_GRADING:N',
                        y='HOURS:Q'
                    )

                    # Combine charts
                    final_chart = box_plot + swarm_plot + highlighted_employee

                    # Display in Streamlit
                    st.altair_chart(final_chart, use_container_width=True)

        else:
            st.error("Failed to load GHC data.")


        # ABSENTEEISM(DAYS) BY HEALTH GRADE
        # Divider for clear sectioning
        st.markdown("---")

        #Absenteeism Baseline
        st.header("*ABSENTEEISM(DAYS) + Health Grade (GHC2)*")
        
        if self.ghc1_df is not None:
            
        
            # Ensure the depot column exists
            if 'DEPOT' not in self.ghc1_df.columns:
                st.error("Depot column not found in dataset")
            else:
                
                # Sort data by tot_opd_kms in ascending order
                drv_lsa_ghc = drv_lsa_ghc.sort_values(by='FINAL_GRADING', ascending=True)
                drv_lsa_ghc = drv_lsa_ghc.reset_index(drop=True)
                
                if drv_lsa_ghc.empty:
                    st.warning("No data available for the selected driver!")
                else:
                    # Create the bar graph with matplotlib
                                    
                    # Create a box plot
                    box_plot = alt.Chart(drv_lsa_ghc).mark_boxplot(size=20).encode(
                        x=alt.X('FINAL_GRADING:N', title='Health Grade'),
                        y=alt.Y('LEAVE_TYPE:Q', title=None)
                    ).properties(
                        title=alt.TitleParams(
                                        text=f'Productivity by Health Grade (Hours/Yr): {self.selected_depot}',  # Correctly display selected depot in the title
                                        anchor='middle'  # Center the title
                                    ),
                    )
                    
                    # Create a swarm plot (jittered points)
                    swarm_plot = alt.Chart(drv_lsa_ghc).mark_point(
                        color='red',
                        size=30
                    ).encode(
                        x=alt.X('FINAL_GRADING:N', title='Health Grade'),
                        y=alt.Y('LEAVE_TYPE:Q', title=None),
                        tooltip=[
                                    alt.Tooltip('EMPLOYEE_ID', title='Employee ID'),  # Custom label for employee ID
                                    alt.Tooltip('LEAVE_TYPE', title='Annual Leaves')  # Custom label for total operational kilometers
                                ],
                    ).transform_calculate(
                        jitter='sqrt(-2*log(random()))*cos(2*PI*random())'  # Simulate jittering
                    ).encode(
                        x=alt.X('FINAL_GRADING:N', title='Health Grade', axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('LEAVE_TYPE:Q', title=None)
                    )
                    
                    # Highlight the selected employee with a larger yellow dot
                    highlighted_employee = alt.Chart(drv_lsa_ghc[drv_lsa_ghc['EMPLOYEE_ID'] == self.selected_driver]).mark_point(
                        color='yellow',
                        size=200, filled=True
                    ).encode(
                        x=alt.X('FINAL_GRADING:N', title='Health Grade'),
                        y=alt.Y('LEAVE_TYPE:Q'),
                        #tooltip=['employee_id', 'tot_opd_kms']
                    )

                    # Combine the box plot, swarm plot, highlighted employee point, and the custom legend
                    final_chart = (box_plot + swarm_plot + highlighted_employee)
                    
                    # Display the chart in Streamlit
                    st.altair_chart(final_chart, use_container_width=True)
        else:
                st.error("Failed to load data.")

        # Filter health data for selected driver
        drv_health = pd.DataFrame()
        if hasattr(self, 'ghc1_df') and self.ghc1_df is not None:
            if 'EMPLOYEE_ID' in self.ghc1_df.columns:
                drv_health = self.ghc1_df[self.ghc1_df['EMPLOYEE_ID'] == self.selected_driver]

        # --- Health Profile ---
        # --- Health Profile ---
        st.markdown("### Health Profile")
        if not drv_health.empty:
            hr = drv_health.iloc[0]

            # Define a helper to safely display values
            def display_metric(label, value, interpret=None):
                if value is None or pd.isna(value):
                    value = "NA"
                if interpret:
                    st.write(f"{label}:** {value} ({interpret})")
                else:
                    st.write(f"{label}:** {value}")

            st.write(f"**BMI:** {hr.get('BMI','NA')} ({hr.get('BMI_INTERPRET','')})")

            st.write(
                f"**Blood Pressure:** {hr.get('BLOOD_PRESSURE_SYSTOLIC','NA')}/"
                f"{hr.get('BLOOD_PRESSURE_DIASTOLIC','NA')} "
                f"({hr.get('BLOOD_PRESSURE_INTERPRET','')})"
            )

            st.write(
                f"**Hemoglobin:** {hr.get('HEMOGLOBIN_VALUE','NA')} "
                f"({hr.get('HEMOGLOBIN_INTERPRET','')})"
            )

            st.write(
                f"**Glucose (Random):** {hr.get('GLUCOSE_RANDOM_VALUE','NA')} "
                f"({hr.get('GLUCOSE_INTERPRET','')})"
            )

            st.write(
                f"**Cholesterol:** {hr.get('TOTAL_CHOLESTROL','NA')} "
                f"({hr.get('CHOLESTEROL_INTERPRET','')})"
            )

            st.write(
                f"**Creatinine:** {hr.get('CREATININE_VALUE','NA')} "
                f"({hr.get('CREATININE_INTERPRET','')})"
            )

            st.write(
                f"**Bilirubin:** {hr.get('BILIRUBIN_TOTAL_VALUE','NA')} "
                f"({hr.get('BILIRUBIN_INTERPRET','')})"
            )

            st.write(
                f"**ECG Result:** {hr.get('ECG_INTERPRET','NA')} "
                f"({hr.get('ECG_COMMENT','')})"
            )

            st.write(
                f"**LEFT EYE (Day/Night):** {hr.get('D_LEFT_EYE','NA')} / "
                f"{hr.get('N_LEFT_EYE','NA')}"
            )

            st.write(
                f"**RIGHT EYE (Day/Night):** {hr.get('D_RIGHT_EYE','NA')} / "
                f"{hr.get('N_RIGHT_EYE','NA')}"
            )

            st.write(f"**Final Health Grade:** {hr.get('FINAL_GRADING','NA')}")

        else:
            st.info("No health data available for this driver.")




    def driver_depot_ui(self):
        # --- Filter driver-specific data ---
        drv_ops = self.ops_df[
            (self.ops_df['EMPLOYEE_ID'] == self.selected_driver) &
            (self.ops_df['OPERATIONS_DATE'] >= self.start_date) &
            (self.ops_df['OPERATIONS_DATE'] <= self.end_date)
        ]
        
        drv_leaves = self.abs_df[
            (self.abs_df['EMPLOYEE_ID'] == self.selected_driver) &
            (self.abs_df['DATE'] >= self.start_date) &
            (self.abs_df['DATE'] <= self.end_date)
        ]
        
        drv_hours = pd.merge(
            self.ops_df, self.ser_df[['SERVICE_NUMBER','HOURS']], on='SERVICE_NUMBER', how='left'
        )
        drv_hours['HOURS'] = drv_hours['HOURS'].fillna(0)
        drv_hours2 = drv_hours[
            (drv_hours['OPERATIONS_DATE'] >= self.start_date) &
            (drv_hours['OPERATIONS_DATE'] <= self.end_date)
        ]
        drv_hours = drv_hours[
            (drv_hours['EMPLOYEE_ID'] == self.selected_driver) &
            (drv_hours['OPERATIONS_DATE'] >= self.start_date) &
            (drv_hours['OPERATIONS_DATE'] <= self.end_date)
        ]
        
        # --- Driver health data ---
        drv_health = pd.DataFrame()
        if hasattr(self, 'ghc1_df') and self.ghc1_df is not None:
            if 'EMPLOYEE_ID' in self.ghc1_df.columns:
                drv_health = self.ghc1_df[self.ghc1_df['EMPLOYEE_ID'] == self.selected_driver]
        
        # --- Merge health grade with hours ---
        drv_ghcgrade = pd.DataFrame()
        if not drv_hours2.empty and hasattr(self, 'ghc1_df') and self.ghc1_df is not None:
            drv_ghcgrade = pd.merge(
                self.ghc1_df[['EMPLOYEE_ID','AGE','FINAL_GRADING']],
                drv_hours2[['EMPLOYEE_ID','HOURS']],
                on='EMPLOYEE_ID',
                how='right'
            )
            drv_ghcgrade['HOURS'] = drv_ghcgrade['HOURS'].fillna(0)
        
        # --- Absenteeism L+S+A ---
        drv_lsa_ghc = pd.DataFrame()  # Initialize as empty DataFrame

        if not self.abs_df.empty and hasattr(self, 'ghc1_df') and self.ghc1_df is not None:
            drv_leaves_filtered = self.abs_df[
                (self.abs_df['DATE'] >= self.start_date) &
                (self.abs_df['DATE'] <= self.end_date) &
                (self.abs_df['LEAVE_TYPE'].isin(['L', 'S', 'A']))
            ]
            
            if not drv_leaves_filtered.empty:
                self.drv_leaves2 = drv_leaves_filtered.groupby('EMPLOYEE_ID').size().reset_index(name='LEAVE_COUNT')
                drv_lsa_ghc = pd.merge(
                    self.drv_leaves2,
                    self.ghc1_df[['EMPLOYEE_ID', 'FINAL_GRADING']],
                    on='EMPLOYEE_ID',
                    how='outer'
                )
                

        # Always check before sorting/using
        if not drv_lsa_ghc.empty:
            drv_lsa_ghc = drv_lsa_ghc.sort_values(by='FINAL_GRADING', ascending=True).reset_index(drop=True)
            # downstream logic (plot, table, etc.) goes here
        else:
            st.warning("No absenteeism data available for the selected depot and financial year.")

        
        # --- Global averages ---
        global_ops_time = self.ops_df[
            (self.ops_df['OPERATIONS_DATE'] >= self.start_date) &
            (self.ops_df['OPERATIONS_DATE'] <= self.end_date)
        ]
        global_hours = pd.merge(
            self.ops_df, self.ser_df[['SERVICE_NUMBER','HOURS']], on='SERVICE_NUMBER', how='left'
        )
        global_hours = global_hours[
            (global_hours['OPERATIONS_DATE'] >= self.start_date) &
            (global_hours['OPERATIONS_DATE'] <= self.end_date)
        ]
        global_leaves = self.abs_df[
            (self.abs_df['DATE'] >= self.start_date) &
            (self.abs_df['DATE'] <= self.end_date) &
            (self.abs_df['LEAVE_TYPE'].isin(['L','S','A']))
        ]
        
        # GLOBAL AVERAGES (MATCHING TOTALS IN BAR CHARTS)

        # KM average = average of total kms per employee
        global_kms_avg = (
            global_ops_time.groupby('EMPLOYEE_ID')['OPD_KMS'].sum().mean()
            if not global_ops_time.empty else 0
        )

        # Earnings average = average of total earnings per employee
        global_earnings_avg = (
            global_ops_time.groupby('EMPLOYEE_ID')['DAILY_EARNINGS'].sum().mean()
            if not global_ops_time.empty else 0
        )

        # Hours average = average of total hours per employee
        global_hours_avg = (
            global_hours.groupby('EMPLOYEE_ID')['HOURS'].sum().mean()
            if not global_hours.empty else 0
        )

        # Leave average = average leave count per employee
        global_leaves_avg = (
            global_leaves.groupby('EMPLOYEE_ID').size().mean()
            if not global_leaves.empty else 0
        )

        
        # --- Driver Info ---
        driver_info = self.driver_df[self.driver_df['EMPLOYEE_ID'] == str(self.selected_driver)]
        col_det, col_sum = st.columns(2)
        
        with col_det:
            st.markdown("## Driver Details")
            if not driver_info.empty:
                info_row = driver_info.iloc[0]
                st.write(f"*Name:* {info_row.get('FULL_NAME','N/A')}")
                #st.write(f"*Unit:* {info_row.get('UNIT','N/A')}")
                st.write(f"*Age:* {info_row.get('AGE','N/A')}")
                st.write(f"*Birth Date:* {info_row.get('BIRTH_DATE','N/A')}")
                st.write(f"*Joining Date:* {info_row.get('JOINING_DATE','N/A')}")
                st.write(f"*Gender:* {info_row.get('GENDER','N/A')}")
                st.write(f"*Marital Status:* {info_row.get('MARITAL_STATUS','N/A')}")
            else:
                st.info("No driver details found.")
        
        with col_sum:
            st.markdown("## Performance Summary")
            total_kms = drv_ops['OPD_KMS'].mean() if 'OPD_KMS' in drv_ops.columns else 0
            total_earnings = drv_ops['DAILY_EARNINGS'].mean() if 'DAILY_EARNINGS' in drv_ops.columns else 0
            total_hours = drv_hours['HOURS'].mean() if 'HOURS' in drv_hours.columns else 0
            lsa_leaves = f"{(drv_leaves['LEAVE_TYPE']=='L').sum()} + {(drv_leaves['LEAVE_TYPE']=='S').sum()} + {(drv_leaves['LEAVE_TYPE']=='A').sum()}"
            
            st.markdown(f"""
            <table style="width:100%;font-size:18px;">
                <tr>
                    <td><b>Total Kilometers Driven</b></td>
                    <td style="color:#1957a6; text-align:right;"><b>{total_kms:,.2f}</b></td>
                </tr>
                <tr>
                    <td style="font-size:14px; color:#888;">Global Avg KMs</td>
                    <td style="font-size:14px; color:#888; text-align:right;">{global_kms_avg:,.2f}</td>
                </tr>
                <tr style="height:8px;"><td colspan="2"></td></tr>
                <tr>
                    <td><b>Total Earnings</b></td>
                    <td style="color:#1957a6; text-align:right;"><b>₹{total_earnings:,.2f}</b></td>
                </tr>
                <tr>
                    <td style="font-size:14px; color:#888;">Global Avg Earnings</td>
                    <td style="font-size:14px; color:#888; text-align:right;">₹{global_earnings_avg:,.2f}</td>
                </tr>
                <tr style="height:8px;"><td colspan="2"></td></tr>
                <tr>
                    <td><b>Total Hours Driven</b></td>
                    <td style="color:#1957a6; text-align:right;"><b>{total_hours:,.2f}</b></td>
                </tr>
                <tr>
                    <td style="font-size:14px; color:#888;">Global Avg Hours</td>
                    <td style="font-size:14px; color:#888; text-align:right;">{global_hours_avg:,.2f}</td>
                </tr>
                <tr>
                    <td><b>Leave Days Taken (L+S+A)</b></td>
                    <td style="text-align:right;"><b>{lsa_leaves}</b></td>
                </tr>
            </table>
            """, unsafe_allow_html=True)


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
            
       # ================================================================
        # --- Total Kilometers ---
        # ================================================================
        st.markdown("### Total Kilometers Driven by All Employees")

        depot_ops_time = self.ops_df[
            (self.ops_df['OPERATIONS_DATE'] >= self.start_date) &
            (self.ops_df['OPERATIONS_DATE'] <= self.end_date)
        ]

        all_emp_kms = depot_ops_time.groupby('EMPLOYEE_ID')['OPD_KMS'].sum().reset_index()
        all_emp_kms['is_selected'] = all_emp_kms['EMPLOYEE_ID'] == self.selected_driver

        bars_kms = (
            alt.Chart(all_emp_kms)
            .mark_bar()
            .encode(
                x=alt.X('EMPLOYEE_ID:N', title='Employee ID', sort='-y'),
                y=alt.Y('OPD_KMS:Q', title='Total Kilometers'),
                tooltip=['EMPLOYEE_ID', 'OPD_KMS'],
                color=alt.condition(
                    alt.datum.is_selected,
                    alt.value('red'),
                    alt.value('#1f77b4')
                )
            )
        )

        # FIXED FULL-WIDTH AVG LINE
        avg_line_kms = (
            alt.Chart(pd.DataFrame({'avg_value': [global_kms_avg]}))
            .mark_rule(color='red', strokeDash=[4, 2], size=2)
            .encode(y='avg_value:Q')
        )

        final_kms_chart = (bars_kms + avg_line_kms).properties(width=900).configure_view(clip=False)
        st.altair_chart(final_kms_chart, use_container_width=True)
        chart_legend("Depot Employees: Blue", "#1f77b4", "Selected Employee: Red", "red", "Global Average: Red Dashed")


        # ================================================================
        # --- Total Earnings ---
        # ================================================================
        if 'DAILY_EARNINGS' in depot_ops_time.columns:
            st.markdown("### Total Earnings of All Employees")

            all_emp_earnings = depot_ops_time.groupby('EMPLOYEE_ID')['DAILY_EARNINGS'].sum().reset_index()
            all_emp_earnings['is_selected'] = all_emp_earnings['EMPLOYEE_ID'] == self.selected_driver

            bars_earnings = (
                alt.Chart(all_emp_earnings)
                .mark_bar()
                .encode(
                    x=alt.X('EMPLOYEE_ID:N', title='Employee ID', sort='-y'),
                    y=alt.Y('DAILY_EARNINGS:Q', title='Total Earnings'),
                    tooltip=['EMPLOYEE_ID', 'DAILY_EARNINGS'],
                    color=alt.condition(
                        alt.datum.is_selected,
                        alt.value('red'),
                        alt.value('#1f77b4')
                    )
                )
            )

            # FIXED FULL-WIDTH AVG LINE
            avg_line_earnings = (
                alt.Chart(pd.DataFrame({'avg_value': [global_earnings_avg]}))
                .mark_rule(color='red', strokeDash=[4, 2], size=2)
                .encode(y='avg_value:Q')
            )

            final_earn_chart = (bars_earnings + avg_line_earnings).properties(width=900).configure_view(clip=False)
            st.altair_chart(final_earn_chart, use_container_width=True)
            chart_legend("Depot Employees: Blue", "#1f77b4", "Selected Employee: Red", "red", "Global Average: Red Dashed")


        # ================================================================
        # --- Productivity Hours ---
        # ================================================================
        if not drv_hours2.empty:
            st.markdown("### Total Productivity Hours of All Employees")

            all_emp_hours = drv_hours2.groupby('EMPLOYEE_ID')['HOURS'].sum().reset_index()
            all_emp_hours['is_selected'] = all_emp_hours['EMPLOYEE_ID'] == self.selected_driver

            bars_hours = (
                alt.Chart(all_emp_hours)
                .mark_bar()
                .encode(
                    x=alt.X('EMPLOYEE_ID:N', title='Employee ID', sort='-y'),
                    y=alt.Y('HOURS:Q', title='Total Hours'),
                    tooltip=['EMPLOYEE_ID', 'HOURS'],
                    color=alt.condition(
                        alt.datum.is_selected,
                        alt.value('red'),
                        alt.value('#1f77b4')
                    )
                )
            )

            # FIXED FULL-WIDTH AVG LINE
            avg_line_hours = (
                alt.Chart(pd.DataFrame({'avg_value': [global_hours_avg]}))
                .mark_rule(color='red', strokeDash=[4, 2], size=2)
                .encode(y='avg_value:Q')
            )

            final_hours_chart = (bars_hours + avg_line_hours).properties(width=900).configure_view(clip=False)
            st.altair_chart(final_hours_chart, use_container_width=True)
            chart_legend("Depot Employees: Blue", "#1f77b4", "Selected Employee: Red", "red", "Global Average: Red Dashed")
        else:
            st.info("No hours data for selected filters.")


        # ================================================================
        # --- Absenteeism / Leave Days ---
        # ================================================================
        if not self.drv_leaves2.empty:
            st.markdown("### Total Leave Days (L+S+A) of All Employees")

            all_emp_leaves = self.drv_leaves2.groupby('EMPLOYEE_ID')['LEAVE_COUNT'].sum().reset_index(name='Leave_Days')
            all_emp_leaves['is_selected'] = all_emp_leaves['EMPLOYEE_ID'] == self.selected_driver

            bars_leaves = (
                alt.Chart(all_emp_leaves)
                .mark_bar()
                .encode(
                    x=alt.X('EMPLOYEE_ID:N', title='Employee ID', sort='-y'),
                    y=alt.Y('Leave_Days:Q', title='Total Leave Days'),
                    tooltip=['EMPLOYEE_ID', 'Leave_Days'],
                    color=alt.condition(
                        alt.datum.is_selected,
                        alt.value('red'),
                        alt.value('#1f77b4')
                    )
                )
            )

            # FIXED FULL-WIDTH AVG LINE
            avg_line_leaves = (
                alt.Chart(pd.DataFrame({'avg_value': [global_leaves_avg]}))
                .mark_rule(color='red', strokeDash=[4, 2], size=2)
                .encode(y='avg_value:Q')
            )

            final_leaves_chart = (bars_leaves + avg_line_leaves).properties(width=900).configure_view(clip=False)
            st.altair_chart(final_leaves_chart, use_container_width=True)
            chart_legend("Depot Employees: Blue", "#1f77b4", "Selected Employee: Red", "red", "Global Average: Red Dashed")
        else:
            st.info("No leave/absenteeism data for selected filters.")



        # --- PRODUCTIVITY BY HEALTH GRADE ---
        st.markdown("---")
        st.header("*Productivity (Hours) + Health Grade (GHC2)*")
        if self.ghc1_df is not None and 'DEPOT' in self.ghc1_df.columns:
            sorted_data3 = drv_ghcgrade.dropna(subset=['FINAL_GRADING']).sort_values(by='FINAL_GRADING', ascending=True)
            sorted_data3 = sorted_data3.groupby(['EMPLOYEE_ID', 'FINAL_GRADING'])['HOURS'].sum().reset_index()

            if sorted_data3.empty:
                st.warning("No data available for the selected depot.")
            else:
                box_plot = alt.Chart(sorted_data3).mark_boxplot(size=20).encode(
                    x=alt.X('FINAL_GRADING:N', title='Health Grade'),
                    y=alt.Y('HOURS:Q', title=None)
                ).properties(
                    title=alt.TitleParams(text=f'Productivity by Health Grade (Hours/Yr): {self.selected_depot}', anchor='middle')
                )
                
                swarm_plot = alt.Chart(sorted_data3).mark_point(
                    color='red', size=30
                ).encode(
                    x=alt.X('FINAL_GRADING:N', title='Health Grade', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('HOURS:Q', title=None),
                    tooltip=[alt.Tooltip('EMPLOYEE_ID', title='Employee ID'), alt.Tooltip('HOURS', title='Annual Hours')],
                )
                
                # Highlight selected employee in yellow
                highlighted_employee = alt.Chart(sorted_data3[sorted_data3['EMPLOYEE_ID'] == self.selected_driver]).mark_point(
                    color='yellow', size=200, filled=True, stroke='black', strokeWidth=2
                ).encode(
                    x=alt.X('FINAL_GRADING:N', title='Health Grade'),
                    y=alt.Y('HOURS:Q')
                )

                final_chart = (box_plot + swarm_plot + highlighted_employee)
                st.altair_chart(final_chart, use_container_width=True)
        else:
            st.error("Failed to load data.")

        # --- ABSENTEEISM(DAYS) BY HEALTH GRADE ---
        st.markdown("---")
        st.header("*ABSENTEEISM(DAYS) + Health Grade (GHC2)*")
        if self.ghc1_df is not None and 'DEPOT' in self.ghc1_df.columns and not drv_lsa_ghc.empty:
            drv_lsa_ghc = drv_lsa_ghc.sort_values(by='FINAL_GRADING', ascending=True)
            drv_lsa_ghc = drv_lsa_ghc.reset_index(drop=True)

            if drv_lsa_ghc.empty:
                st.warning("No data available for the selected depot.")
            else:
                box_plot = alt.Chart(drv_lsa_ghc).mark_boxplot(size=20).encode(
                    x=alt.X('FINAL_GRADING:N', title='Health Grade'),
                    y=alt.Y('LEAVE_COUNT:Q', title=None)
                ).properties(
                    title=alt.TitleParams(text=f'Absenteeism by Health Grade: {self.selected_depot}', anchor='middle')
                )
                
                swarm_plot = alt.Chart(drv_lsa_ghc).mark_point(
                    color='red', size=30
                ).encode(
                    x=alt.X('FINAL_GRADING:N', title='Health Grade', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('LEAVE_COUNT:Q', title=None),
                    tooltip=[alt.Tooltip('EMPLOYEE_ID', title='Employee ID'), alt.Tooltip('LEAVE_COUNT', title='Annual Leaves')],
                )
                
                # Highlight selected employee in yellow
                highlighted_employee = alt.Chart(drv_lsa_ghc[drv_lsa_ghc['EMPLOYEE_ID'] == self.selected_driver]).mark_point(
                    color='yellow', size=200, filled=True, stroke='black', strokeWidth=2
                ).encode(
                    x=alt.X('FINAL_GRADING:N', title='Health Grade'),
                    y=alt.Y('LEAVE_COUNT:Q')
                )
                
                final_chart = (box_plot + swarm_plot + highlighted_employee)
                st.altair_chart(final_chart, use_container_width=True)
        else:
            st.warning("No Data Available!")

        # --- Health Profile ---
        st.markdown("### Health Profile")
        if not drv_health.empty:
            hr = drv_health.iloc[0]
            st.write(f"**BMI:** {hr.get('BMI','NA')} ({hr.get('BMI_INTERPRET','')})")

            st.write(
                f"**Blood Pressure:** {hr.get('BLOOD_PRESSURE_SYSTOLIC','NA')}/"
                f"{hr.get('BLOOD_PRESSURE_DIASTOLIC','NA')} "
                f"({hr.get('BLOOD_PRESSURE_INTERPRET','')})"
            )

            st.write(
                f"**Hemoglobin:** {hr.get('HEMOGLOBIN_VALUE','NA')} "
                f"({hr.get('HEMOGLOBIN_INTERPRET','')})"
            )

            st.write(
                f"**Glucose (Random):** {hr.get('GLUCOSE_RANDOM_VALUE','NA')} "
                f"({hr.get('GLUCOSE_INTERPRET','')})"
            )

            st.write(
                f"**Cholesterol:** {hr.get('TOTAL_CHOLESTROL','NA')} "
                f"({hr.get('CHOLESTEROL_INTERPRET','')})"
            )

            st.write(
                f"**Creatinine:** {hr.get('CREATININE_VALUE','NA')} "
                f"({hr.get('CREATININE_INTERPRET','')})"
            )

            st.write(
                f"**Bilirubin:** {hr.get('BILIRUBIN_TOTAL_VALUE','NA')} "
                f"({hr.get('BILIRUBIN_INTERPRET','')})"
            )

            st.write(
                f"**ECG Result:** {hr.get('ECG_INTERPRET','NA')} "
                f"({hr.get('ECG_COMMENT','')})"
            )

            st.write(
                f"**LEFT EYE (Day/Night):** {hr.get('D_LEFT_EYE','NA')} / "
                f"{hr.get('N_LEFT_EYE','NA')}"
            )

            st.write(
                f"**RIGHT EYE (Day/Night):** {hr.get('D_RIGHT_EYE','NA')} / "
                f"{hr.get('N_RIGHT_EYE','NA')}"
            )

            st.write(f"**Final Health Grade:** {hr.get('FINAL_GRADING','NA')}")

        else:
            st.info("No health data for this driver.")


if __name__ == '__main__':
    user_depot = st.session_state.user_depot
    role = st.session_state.role

    obj = driver_depot_dashboard_ui_DM(user_depot, role)
    obj.parameters()

        # ---- TAB STYLE ----
    st.markdown("""
    <style>

    /* Default tab style */
    .stTabs [role="tab1"] {
        padding: 16px 28px !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #333 !important;
        background: #f0f0f0 !important;
        border-radius: 8px 8px 0 0 !important;
        margin-right: 6px !important;
        border: 2px solid #ddd !important;
    }

    /* Selected Tab - SUPER VISIBLE */
    st.tab2 [role="tab2"][aria-selected="true"] {
        background: #ffcccc !important;      /* light red fill */
        color: #b30000 !important;           /* dark red text */
        border-bottom: 6px solid #ff0000 !important;  /* thick underline */
        border-top: 3px solid #ff0000 !important;      /* top border highlight */
        border-radius: 10px 10px 0 0 !important;
        font-size: 100px !important;
        font-weight: 900 !important;
    }

    /* Hover effect for better visibility */
    .stTabs [role="tab"]:hover {
        background: #ffe6e6 !important;
        cursor: pointer;
    }

    /* Unselected Tab */
    .stTabs [role="tab"][aria-selected="false"] {
        opacity: 0.7;
        color: #555 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["#Driver Performance", "Driver Performance in Depot"])

    with tab1:
        try:
            obj.driver_ui()
        except Exception as e:
            st.error(f"{e}")

    with tab2:
        try:
            obj.driver_depot_ui()
        except Exception as e:
            st.error(f"{e}")
