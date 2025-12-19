# driver_depot_dashboard_ui.py
# Full single-file dashboard (split into 5 parts). Paste parts in order into one file.

import pandas as pd
import base64
import altair as alt
import mysql.connector
import streamlit as st
from mysql.connector import Error
import json
import datetime
import gc


# ----------------- Small UI helper moved to module scope so all methods can use it -----------------
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
# ----------------- MySQL Connection Helper -----------------
class Sql_connection:
    def __init__(self, config_path="config.json"):
        try:
            with open(config_path) as f:
                config = json.load(f)
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

    def load_sql_data(self, table_name, depots=None, depot_col='depot', start_date=None, end_date=None, date_col=None):
        """
        Generic loader with optional depot + date filtering.
        start_date/end_date can be pandas.Timestamp or datetime/date.
        Returns DataFrame (empty on error).
        """
        try:
            cursor = self._conn.cursor()
            query = f"SELECT * FROM {table_name}"
            where_clauses = []
            params = []

            # Depot filter
            if depots:
                dep_list = list(depots) if isinstance(depots, (list, tuple, set)) else [depots]
                placeholders = ",".join(["%s"] * len(dep_list))
                where_clauses.append(f"{depot_col} IN ({placeholders})")
                params.extend(dep_list)

            # Date filter
            if date_col and start_date is not None and end_date is not None:
                def to_sql_date(x):
                    try:
                        if isinstance(x, pd.Timestamp):
                            return x.strftime("%Y-%m-%d")
                    except Exception:
                        pass
                    try:
                        import datetime as _dt
                        if isinstance(x, (_dt.datetime, _dt.date)):
                            return x.strftime("%Y-%m-%d")
                    except Exception:
                        pass
                    return str(x)
                s = to_sql_date(start_date)
                e = to_sql_date(end_date)
                where_clauses.append(f"{date_col} BETWEEN %s AND %s")
                params.extend([s, e])

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=columns) if rows else (pd.DataFrame(columns=columns) if columns else pd.DataFrame())
            return df
        except Error as e:
            try:
                st.error(f"⚠ SQL error while loading {table_name}: {e}")
                st.error(f"Query: {query}")
                st.error(f"Params: {params}")
            except Exception:
                pass
            return pd.DataFrame()
        finally:
            try:
                cursor.close()
            except Exception:
                pass
# Part 2/5 — dashboard class init, load_data, helpers

class driver_depot_dashboard_ui:
    def __init__(self, user_region=None, role=None, ops_df=None):
        self.user_region = user_region
        self.role = role
        self.ops_df = ops_df
        self.month_year = []

        # Header logo
        file_path = r"driver_dashboard_logo.png"
        try:
            with open(file_path, "rb") as img_file:
                b64_img = base64.b64encode(img_file.read()).decode()
        except FileNotFoundError:
            b64_img = ""

        st.markdown(
            f"""
            <div style="text-align: center; border-radius: 12px; border-width: 2px; border-color: black; padding: 0px;">
                {"<img src='data:image/png;base64," + b64_img + "' width='2000' height='270' style='display:block; margin:0 auto;'>" if b64_img else ""}
                <h1 style="color: black; margin:6px 0 8px 0;">Telangana State Road Transport Corporation</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        # SQL helper and placeholders
        try:
            self.sql = Sql_connection()
            self._conn = self.sql._conn
        except Exception:
            self.sql = None
            self._conn = None

        self.driver_df = pd.DataFrame()
        self.ops_df = pd.DataFrame()
        self.ser_df = pd.DataFrame()
        self.abs_df = pd.DataFrame()
        self.ghc1_df = pd.DataFrame()
        self.max_date = pd.Timestamp.today()

    def load_data(self):
        """
        Load filtered data based on self.selected_depot and date range.
        Must be called after parameters() sets selected_depot, start_date, end_date.
        """
        if not self.sql:
            st.error("SQL helper not initialized.")
            return
        if not hasattr(self, 'selected_depot') or not self.selected_depot:
            st.error("Selected depot not set before calling load_data().")
            return

        sd = self.start_date
        ed = self.end_date

        # Load tables filtered to depot + date where appropriate
        self.driver_df = self.sql.load_sql_data('driver_details', depots=[self.selected_depot], depot_col='unit')
        self.ops_df = self.sql.load_sql_data('daily_operations', depots=[self.selected_depot], depot_col='depot', start_date=sd, end_date=ed, date_col='OPERATIONS_DATE')
        self.ser_df = self.sql.load_sql_data('service_master', depots=[self.selected_depot], depot_col='depot')
        self.abs_df = self.sql.load_sql_data('driver_absenteeism', depots=[self.selected_depot], depot_col='depot', start_date=sd, end_date=ed, date_col='DATE')
        self.ghc1_df = self.sql.load_sql_data('ghc_2024', depots=[self.selected_depot], depot_col='depot')

        # Normalize columns to uppercase and trim strings
        for attr in ('driver_df','ops_df','ser_df','abs_df','ghc1_df'):
            df = getattr(self, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.columns = [str(c).strip().upper() for c in df.columns]
                if 'DEPOT' in df.columns:
                    df['DEPOT'] = df['DEPOT'].astype(str).str.strip().str.upper()
                if 'EMPLOYEE_ID' in df.columns:
                    df['EMPLOYEE_ID'] = df['EMPLOYEE_ID'].astype(str).str.strip()
                setattr(self, attr, df)

        # Convert date columns
        if not self.ops_df.empty and 'OPERATIONS_DATE' in self.ops_df.columns:
            self.ops_df['OPERATIONS_DATE'] = pd.to_datetime(self.ops_df['OPERATIONS_DATE'], errors='coerce')
        if not self.abs_df.empty and 'DATE' in self.abs_df.columns:
            self.abs_df['DATE'] = pd.to_datetime(self.abs_df['DATE'], errors='coerce')

        # Add MONTH_YEAR
        self.ops_df = self.parse_and_format(self.ops_df, 'OPERATIONS_DATE')
        self.abs_df = self.parse_and_format(self.abs_df, 'DATE')

        # Calculate service HOURS if DEPT_TIME & ARR_TIME present
        if not self.ser_df.empty and 'DEPT_TIME' in self.ser_df.columns and 'ARR_TIME' in self.ser_df.columns:
            self.ser_df['DEPT_TIME'] = pd.to_datetime(self.ser_df['DEPT_TIME'], errors='coerce')
            self.ser_df['ARR_TIME'] = pd.to_datetime(self.ser_df['ARR_TIME'], errors='coerce')
            self.ser_df['HOURS'] = (self.ser_df['DEPT_TIME'] - self.ser_df['ARR_TIME']).dt.total_seconds() / 3600
            self.ser_df['HOURS'] = self.ser_df['HOURS'].abs().round(2)
        else:
            if isinstance(self.ser_df, pd.DataFrame):
                self.ser_df['HOURS'] = 0

        # Ensure numeric columns to prevent Altair errors
        numeric_cols = ['HOURS','OPD_KMS','DAILY_EARNINGS','LEAVE_COUNT']
        for attr in ('ops_df','ser_df','abs_df','driver_df','ghc1_df'):
            df = getattr(self, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                for c in numeric_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                setattr(self, attr, df)

        # Financial year and max date
        if not self.ops_df.empty and 'OPERATIONS_DATE' in self.ops_df.columns:
            self.ops_df['FINANCIAL_YEAR'] = self.ops_df['OPERATIONS_DATE'].apply(self.get_financial_year)
            self.max_date = self.ops_df['OPERATIONS_DATE'].max()
        else:
            self.max_date = pd.Timestamp.today()

    def parse_and_format(self, df, date_col):
        if isinstance(df, pd.DataFrame) and not df.empty and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            df = df.dropna(subset=[date_col]).copy()
            df['MONTH_YEAR'] = df[date_col].dt.to_period('M').dt.strftime('%Y-%m')
        return df

    def get_financial_year(self, date):
        if pd.isna(date):
            return None
        year = date.year
        return f"{year-1}-{year}" if date.month < 4 else f"{year}-{year+1}"

    def get_regions(self):
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT DISTINCT REGION FROM TS_ADMIN ORDER BY REGION;")
            result = cursor.fetchall()
            regions = [r[0] for r in result]
            return regions
        except Error as e:
            st.error(f"Error fetching regions: {e}")
            return []
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def get_depots_for_region(self, region):
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT DEPOT_NAME FROM TS_ADMIN WHERE REGION = %s ORDER BY DEPOT_NAME;", (region,))
            result = cursor.fetchall()
            depots = [d[0] for d in result]
            return depots
        except Error as e:
            st.error(f"Error fetching depots for region {region}: {e}")
            return []
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def _ensure_numeric_columns(self, df, cols):
        if not isinstance(df, pd.DataFrame):
            return df
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:
                df[c] = 0
        return df
# Part 3/5 — parameters() and driver_ui (start)

    #---------------- Parameters UI ----------------
    def parameters(self):
        regions = self.get_regions()
        if not regions:
            st.error("No regions found in TS_ADMIN.")
            st.stop()

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Select Region")
                default_index = regions.index(self.user_region) if self.user_region in regions else 0
                self.selected_region = st.selectbox("", regions, index=default_index)
            with col2:
                st.markdown("### Select Depot")
                depots = self.get_depots_for_region(self.selected_region)
                if not depots:
                    st.warning(f"No depots found for region {self.selected_region}.")
                    st.stop()
                depots_display = [str(d).strip() for d in depots]
                self.selected_depot = st.selectbox("", depots_display)

        # Normalize selected depot
        self.selected_depot = str(self.selected_depot).strip().upper()

        # Month & Year
        st.markdown("###  Select Month & Year:")
        col_y1, col_y2, col_y3 = st.columns(3)
        with col_y1:
            years = [2025, 2024, 2023, 2022]
            self.selected_year = st.selectbox("Select Year", years, index=0)
        with col_y2:
            months = {"All":0,"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,"July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
            month_list = list(months.keys())
            self.selected_month = st.selectbox("Select Month", month_list, index=0)

        m = months[self.selected_month]
        if self.selected_month == "All":
            self.start_date = pd.Timestamp(self.selected_year, 1, 1)
            self.end_date = pd.Timestamp(self.selected_year, 12, 31)
        else:
            self.start_date = pd.Timestamp(self.selected_year, m, 1)
            self.end_date = self.start_date + pd.offsets.MonthEnd(1)

        # Load filtered data
        self.load_data()
        gc.collect()

        # Driver selection based on ops_df
        with col_y3:
            if self.ops_df.empty or 'EMPLOYEE_ID' not in self.ops_df.columns:
                st.warning("⚠ No operations data available for this depot & period.")
                st.stop()
            drivers_in_depot = self.ops_df['EMPLOYEE_ID'].astype(str).unique().tolist()
            if not drivers_in_depot:
                st.warning("⚠ No drivers found for this depot in selected period.")
                st.stop()
            self.selected_driver = st.selectbox("Employee ID", drivers_in_depot)

    # ---------------- Driver Performance (Tab 1) ----------------
    def driver_ui(self):
        # build month_year list
        self.month_year = []
        for i in pd.date_range(start=self.start_date, end=self.end_date, freq='MS'):
            if self.max_date >= i:
                self.month_year.append(i.strftime('%Y-%m'))
        month_year_df = pd.DataFrame({'MONTH_YEAR': self.month_year})

        # Filter driver operations and leaves
        drv_ops = self.ops_df[
            (self.ops_df['EMPLOYEE_ID'] == str(self.selected_driver)) &
            (self.ops_df['DEPOT'] == self.selected_depot) &
            (self.ops_df['OPERATIONS_DATE'] >= self.start_date) &
            (self.ops_df['OPERATIONS_DATE'] <= self.end_date)
        ] if not self.ops_df.empty else pd.DataFrame()

        drv_leaves = self.abs_df[
            (self.abs_df['EMPLOYEE_ID'] == str(self.selected_driver)) &
            (self.abs_df['DEPOT'] == self.selected_depot) &
            (self.abs_df['DATE'] >= self.start_date) &
            (self.abs_df['DATE'] <= self.end_date)
        ] if not self.abs_df.empty else pd.DataFrame()

        # Merge ops with service hours
        ops_for_hours = self.ops_df if not self.ops_df.empty else pd.DataFrame()
        ser_small = self.ser_df[['SERVICE_NUMBER','HOURS']].drop_duplicates(subset=['SERVICE_NUMBER']) if (isinstance(self.ser_df,pd.DataFrame) and 'SERVICE_NUMBER' in self.ser_df.columns) else pd.DataFrame(columns=['SERVICE_NUMBER','HOURS'])
        drv_hours = pd.merge(ops_for_hours, ser_small, on='SERVICE_NUMBER', how='left') if (not ops_for_hours.empty and not ser_small.empty) else (ops_for_hours.copy() if not ops_for_hours.empty else pd.DataFrame())
        if not drv_hours.empty:
            if 'HOURS' not in drv_hours.columns:
                drv_hours['HOURS'] = 0
            drv_hours['HOURS'] = pd.to_numeric(drv_hours['HOURS'], errors='coerce').fillna(0)
# Part 4/5 — driver_ui (charts + health) and start of depot tab

        drv_hours2 = drv_hours[
            (drv_hours['DEPOT'] == self.selected_depot) &
            (drv_hours['OPERATIONS_DATE'] >= self.start_date) &
            (drv_hours['OPERATIONS_DATE'] <= self.end_date)
        ] if not drv_hours.empty else pd.DataFrame()

        drv_hours = drv_hours[
            (drv_hours['DEPOT'] == self.selected_depot) &
            (drv_hours['EMPLOYEE_ID'] == str(self.selected_driver)) &
            (drv_hours['OPERATIONS_DATE'] >= self.start_date) &
            (drv_hours['OPERATIONS_DATE'] <= self.end_date)
        ] if not drv_hours.empty else pd.DataFrame()

        drv_health = self.ghc1_df[self.ghc1_df['EMPLOYEE_ID'] == str(self.selected_driver)] if (isinstance(self.ghc1_df,pd.DataFrame) and 'EMPLOYEE_ID' in self.ghc1_df.columns) else pd.DataFrame()

        # Merge GHC with hours for productivity by health grade
        drv_ghcgrade = pd.DataFrame()
        if not drv_hours2.empty and isinstance(self.ghc1_df,pd.DataFrame) and 'EMPLOYEE_ID' in self.ghc1_df.columns:
            drv_ghcgrade = pd.merge(self.ghc1_df[['EMPLOYEE_ID','AGE','FINAL_GRADING']].drop_duplicates(subset=['EMPLOYEE_ID']), drv_hours2[['EMPLOYEE_ID','HOURS']], on='EMPLOYEE_ID', how='right')
            drv_ghcgrade['HOURS'] = pd.to_numeric(drv_ghcgrade['HOURS'], errors='coerce').fillna(0)

        # Build leave counts per employee in depot period for L+S+A
        self.drv_leaves2 = pd.DataFrame()
        if not self.abs_df.empty:
            leaves_filtered = self.abs_df[
                (self.abs_df['DEPOT'] == self.selected_depot) &
                (self.abs_df['DATE'] >= self.start_date) &
                (self.abs_df['DATE'] <= self.end_date)
            ]
            if not leaves_filtered.empty:
                # Count leave events per employee (all types)
                self.drv_leaves2 = leaves_filtered.groupby('EMPLOYEE_ID').size().reset_index(name='LEAVE_COUNT')
                self.drv_leaves2['LEAVE_COUNT'] = pd.to_numeric(self.drv_leaves2['LEAVE_COUNT'], errors='coerce').fillna(0).astype(int)

        # Merge leaves with GHC
        drv_lsa_ghc = pd.DataFrame()
        if not self.drv_leaves2.empty and isinstance(self.ghc1_df,pd.DataFrame) and 'EMPLOYEE_ID' in self.ghc1_df.columns:
            drv_lsa_ghc = pd.merge(self.drv_leaves2, self.ghc1_df[['EMPLOYEE_ID','FINAL_GRADING']].drop_duplicates(subset=['EMPLOYEE_ID']), on='EMPLOYEE_ID', how='inner')
            if 'LEAVE_COUNT' in drv_lsa_ghc.columns:
                drv_lsa_ghc['LEAVE_COUNT'] = pd.to_numeric(drv_lsa_ghc['LEAVE_COUNT'], errors='coerce').fillna(0).astype(int)
            else:
                drv_lsa_ghc['LEAVE_COUNT'] = 0

        # Depot context averages
        depot_ops_time = self.ops_df[
            (self.ops_df['DEPOT'] == self.selected_depot) &
            (self.ops_df['OPERATIONS_DATE'] >= self.start_date) &
            (self.ops_df['OPERATIONS_DATE'] <= self.end_date)
        ] if not self.ops_df.empty else pd.DataFrame()

        depot_kms_avg = depot_ops_time['OPD_KMS'].mean() if not depot_ops_time.empty and 'OPD_KMS' in depot_ops_time.columns else 0
        depot_earnings_avg = depot_ops_time['DAILY_EARNINGS'].mean() if not depot_ops_time.empty and 'DAILY_EARNINGS' in depot_ops_time.columns else 0
        depot_hours_avg = drv_hours2['HOURS'].mean() if not drv_hours2.empty and 'HOURS' in drv_hours2.columns else 0

        # Driver details & summary
        driver_info = self.driver_df[self.driver_df['EMPLOYEE_ID'] == str(self.selected_driver)] if (not self.driver_df.empty and 'EMPLOYEE_ID' in self.driver_df.columns) else pd.DataFrame()
        col_det, col_sum = st.columns(2)
        with col_det:
            st.markdown("## Driver Details")
            if not driver_info.empty:
                info_row = driver_info.iloc[0]
                st.write(f"**Name:** {info_row.get('FULL_NAME','N/A')}")
                st.write(f"**Age:** {info_row.get('AGE','N/A')}")
                st.write(f"**Birth Date:** {info_row.get('BIRTH_DATE','N/A')}")
                st.write(f"**Joining Date:** {info_row.get('JOINING_DATE','N/A')}")
                st.write(f"**Gender:** {info_row.get('GENDER','N/A')}")
                st.write(f"**Marital Status:** {info_row.get('MARITAL_STATUS','N/A')}")
            else:
                st.info("No driver details found.")

        with col_sum:
            st.markdown("## Performance Summary")
            total_kms = drv_ops['OPD_KMS'].mean() if (not drv_ops.empty and 'OPD_KMS' in drv_ops.columns) else 0
            total_earnings = drv_ops['DAILY_EARNINGS'].mean() if (not drv_ops.empty and 'DAILY_EARNINGS' in drv_ops.columns) else 0
            total_hours = drv_hours['HOURS'].mean() if (not drv_hours.empty and 'HOURS' in drv_hours.columns) else 0
            lsa_leaves = f"{(drv_leaves['LEAVE_TYPE']=='L').sum() if not drv_leaves.empty else 0} + {(drv_leaves['LEAVE_TYPE']=='S').sum() if not drv_leaves.empty else 0} + {(drv_leaves['LEAVE_TYPE']=='A').sum() if not drv_leaves.empty else 0}"

            st.markdown(f"""
                <table style="width:100%;font-size:18px;">
                    <tr><td><b>Driver Avg KMs per day</b></td><td style="color:#1957a6; text-align:right;"><b>{total_kms:,.2f}</b></td></tr>
                    <tr><td style="font-size:14px; color:#888;">Depot Avg KMs per day</td><td style="font-size:14px; color:#888; text-align:right;">{depot_kms_avg:,.2f}</td></tr>
                    <tr style="height:8px;"><td colspan="2"></td></tr>
                    <tr><td><b>Driver Avg Earnings per km</b></td><td style="color:#1957a6; text-align:right;"><b>₹{(total_earnings/total_kms) if total_kms else 0:,.2f}</b></td></tr>
                    <tr><td style="font-size:14px; color:#888;">Depot Avg Earnings per km</td><td style="font-size:14px; color:#888; text-align:right;">₹{(depot_earnings_avg/depot_kms_avg) if depot_kms_avg else 0:,.2f}</td></tr>
                    <tr style="height:8px;"><td colspan="2"></td></tr>
                    <tr><td><b>Driver Avg Hours per day</b></td><td style="color:#1957a6; text-align:right;"><b>{total_hours:,.2f}</b></td></tr>
                    <tr><td style="font-size:14px; color:#888;">Avg Depot Hours per day</td><td style="font-size:14px; color:#888; text-align:right;">{depot_hours_avg:,.2f}</td></tr>
                    <tr><td><b>Leave Days Taken (L+S+A)</b></td><td style="text-align:right;"><b>{lsa_leaves}</b></td></tr>
                </table>
            """, unsafe_allow_html=True)

        # Small legend helper
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
# Part 5/5 — remaining charts for driver and depot tab + main entry

        # --- Monthly Kilometers ---
        st.markdown("### Monthly Kilometers Driven")
        monthly_kms = drv_ops.groupby('MONTH_YEAR')['OPD_KMS'].sum().reset_index() if (not drv_ops.empty and 'MONTH_YEAR' in drv_ops.columns and 'OPD_KMS' in drv_ops.columns) else pd.DataFrame({'MONTH_YEAR':self.month_year,'OPD_KMS':[0]*len(self.month_year)})
        monthly_kms = pd.merge(pd.DataFrame({'MONTH_YEAR': self.month_year}), monthly_kms, on='MONTH_YEAR', how='left').fillna(0)
        total_kms_period = monthly_kms['OPD_KMS'].sum() if not monthly_kms.empty else 0
        st.markdown(f"<div style='font-size:20px;color:#1957a6;margin-bottom:0;'><b>Total Kilometers:</b> {total_kms_period:,.2f} KMs</div>", unsafe_allow_html=True)
        avg_kms = monthly_kms['OPD_KMS'].mean() if not monthly_kms.empty else 0

        bars = alt.Chart(monthly_kms).mark_bar().encode(
            x=alt.X('MONTH_YEAR:N', sort=monthly_kms['MONTH_YEAR'].tolist(), title='Month-Year'),
            y=alt.Y('OPD_KMS:Q', title='Kilometers'),
            tooltip=['MONTH_YEAR', 'OPD_KMS']
        )
        kms_text = alt.Chart(monthly_kms).mark_text(align='center', baseline='bottom', dy=-5).encode(
            x=alt.X('MONTH_YEAR:N', sort=self.month_year),
            y=alt.Y('OPD_KMS:Q'),
            text=alt.Text('OPD_KMS:Q', format='.0f')
        )
        avg_line = alt.Chart(pd.DataFrame({'OPD_KMS':[avg_kms]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='OPD_KMS:Q')
        avg_text = alt.Chart(pd.DataFrame({'OPD_KMS':[avg_kms],'label':[f'Avg: {avg_kms:.1f}']})).mark_text(align='left', dx=5, dy=-7, color='red').encode(y='OPD_KMS:Q', text='label:N')
        st.altair_chart((bars + kms_text + avg_line + avg_text).properties(width=900), use_container_width=True)
        chart_legend("Bar: Blue", "#1f77b4", None, None, "Average Line: Red")

        # --- Monthly Earnings ---
        if not drv_ops.empty and 'DAILY_EARNINGS' in drv_ops.columns:
            st.markdown("### Monthly Earnings")
            monthly_earnings = drv_ops.groupby('MONTH_YEAR')['DAILY_EARNINGS'].sum().reset_index()
            monthly_earnings = pd.merge(pd.DataFrame({'MONTH_YEAR': self.month_year}), monthly_earnings, on='MONTH_YEAR', how='left').fillna(0)
            total_earnings_period = monthly_earnings['DAILY_EARNINGS'].sum()
            st.markdown(f"<div style='font-size:20px;color:#1957a6;margin-bottom:0;'><b>Total Earnings:</b> ₹{total_earnings_period:,.2f}</div>", unsafe_allow_html=True)
            avg_earn = monthly_earnings['DAILY_EARNINGS'].mean() if not monthly_earnings.empty else 0

            bars2 = alt.Chart(monthly_earnings).mark_bar().encode(
                x='MONTH_YEAR:N',
                y=alt.Y('DAILY_EARNINGS:Q', title='Earnings'),
                tooltip=['MONTH_YEAR', 'DAILY_EARNINGS']
            )
            earnings_text = alt.Chart(monthly_earnings).mark_text(align='center', baseline='bottom', dy=-5).encode(
                x=alt.X('MONTH_YEAR:N', sort=self.month_year),
                y=alt.Y('DAILY_EARNINGS:Q'),
                text=alt.Text('DAILY_EARNINGS:Q', format='.0f')
            )
            avg_line2 = alt.Chart(pd.DataFrame({'DAILY_EARNINGS':[avg_earn]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='DAILY_EARNINGS:Q')
            avg_text2 = alt.Chart(pd.DataFrame({'DAILY_EARNINGS':[avg_earn],'label':[f'Avg: {avg_earn:.1f}']})).mark_text(align='left', dx=5, dy=-7, color='red').encode(y='DAILY_EARNINGS:Q', text='label:N')
            st.altair_chart((bars2 + earnings_text + avg_line2 + avg_text2).properties(width=900), use_container_width=True)
            chart_legend("Bar: Blue", "#1f77b4", None, None, "Average Line: Red")

        # --- Productivity Hours ---
        if not drv_hours.empty and 'MONTH_YEAR' in drv_hours.columns and 'HOURS' in drv_hours.columns:
            st.markdown("### Productivity Hours")
            hours_monthly = drv_hours.groupby('MONTH_YEAR')['HOURS'].sum().reset_index()
            hours_monthly = pd.merge(pd.DataFrame({'MONTH_YEAR': self.month_year}), hours_monthly, on='MONTH_YEAR', how='left').fillna(0)
            total_hours_sum = hours_monthly['HOURS'].sum()
            st.markdown(f"<div style='font-size:20px;color:#1957a6;margin-bottom:0;'><b>Total Hours:</b> {total_hours_sum:.2f} hrs</div>", unsafe_allow_html=True)
            avg_hours = hours_monthly['HOURS'].mean() if not hours_monthly.empty else 0
            hours_bars = alt.Chart(hours_monthly).mark_bar().encode(x='MONTH_YEAR:N', y='HOURS:Q', tooltip=['MONTH_YEAR','HOURS'])
            hours_text = alt.Chart(hours_monthly).mark_text(align='center', baseline='bottom', dy=-5).encode(x=alt.X('MONTH_YEAR:N', sort=self.month_year), y=alt.Y('HOURS:Q'), text=alt.Text('HOURS:Q', format='.0f'))
            hours_avg_line = alt.Chart(pd.DataFrame({'HOURS':[avg_hours]})).mark_rule(color='red', strokeDash=[4,2]).encode(y='HOURS:Q')
            hours_avg_text = alt.Chart(pd.DataFrame({'HOURS':[avg_hours],'label':[f'Avg: {avg_hours:.1f}']})).mark_text(align='left', dx=5, dy=-7, color='red').encode(y='HOURS:Q', text='label:N')
            st.altair_chart((hours_bars + hours_text + hours_avg_line + hours_avg_text).properties(width=900), use_container_width=True)
            chart_legend("Bar: Blue", "#1f77b4", None, None, "Average Line: Red")
        else:
            st.info("No hours data for selected filters.")

        # --- Absenteeism / Leave Monthly ---
        if not drv_leaves.empty and 'MONTH_YEAR' in drv_leaves.columns:
            st.markdown("### Absenteeism / Leave Summary")
            leave_monthly = drv_leaves.groupby('MONTH_YEAR').size().reset_index(name='Leave_Days')
            leave_monthly = pd.merge(pd.DataFrame({'MONTH_YEAR': self.month_year}), leave_monthly, on='MONTH_YEAR', how='left').fillna(0)
            total_leaves_period = leave_monthly['Leave_Days'].sum()
            st.markdown(f"<div style='font-size:20px;color:#1957a6;margin-bottom:0;'><b>Total for Period:</b> {total_leaves_period} Days</div>", unsafe_allow_html=True)
            avg_leave = leave_monthly['Leave_Days'].mean() if not leave_monthly.empty else 0
            leave_bars = alt.Chart(leave_monthly).mark_bar().encode(x='MONTH_YEAR:N', y='Leave_Days:Q', tooltip=['MONTH_YEAR','Leave_Days'])
            leaves_text = alt.Chart(leave_monthly).mark_text(align='center', baseline='bottom', dy=-5).encode(x=alt.X('MONTH_YEAR:N', sort=self.month_year), y=alt.Y('Leave_Days:Q'), text=alt.Text('Leave_Days:Q', format='.0f'))
            leave_avg_line = alt.Chart(pd.DataFrame({'Leave_Days':[avg_leave]})).mark_rule(color='red', strokeDash=[4,2]).encode(y='Leave_Days:Q')
            leave_avg_text = alt.Chart(pd.DataFrame({'Leave_Days':[avg_leave],'label':[f'Avg: {avg_leave:.1f}']})).mark_text(align='left', dx=5, dy=-7, color='red').encode(y='Leave_Days:Q', text='label:N')
            st.altair_chart((leave_bars + leaves_text + leave_avg_line + leave_avg_text).properties(width=900), use_container_width=True)
            chart_legend("Bar: Blue", "#1f77b4", None, None, "Average Line: Red")
        else:
            st.info("No leave/absenteeism data for selected filters.")

        # --- Productivity by Health Grade (HOURS) ---
        st.markdown("---")
        st.header("**Productivity (Hours) by Health Grade (GHC)**")
        if not drv_ghcgrade.empty:
            sorted_data3 = drv_ghcgrade.dropna(subset=['FINAL_GRADING']).groupby(['EMPLOYEE_ID','FINAL_GRADING'], as_index=False)['HOURS'].sum()
            if sorted_data3.empty:
                st.warning("No productivity-by-health data available.")
            else:
                box_plot = alt.Chart(sorted_data3).mark_boxplot(size=20).encode(x=alt.X('FINAL_GRADING:N', title='Health Grade'), y=alt.Y('HOURS:Q', title='Hours'))
                swarm_plot = alt.Chart(sorted_data3).mark_point(color='red', size=30).encode(x=alt.X('FINAL_GRADING:N', axis=alt.Axis(labelAngle=0)), y=alt.Y('HOURS:Q'), tooltip=[alt.Tooltip('EMPLOYEE_ID', title='Employee ID'), alt.Tooltip('HOURS', title='Hours')])
                highlighted_employee = alt.Chart(sorted_data3[sorted_data3['EMPLOYEE_ID']==str(self.selected_driver)]).mark_point(color='yellow', size=200, filled=True).encode(x='FINAL_GRADING:N', y='HOURS:Q')
                final_chart = box_plot + swarm_plot + highlighted_employee
                st.altair_chart(final_chart, use_container_width=True)
        else:
            st.info("No health-productivity merge data to display.")

        # --- Absenteeism by Health Grade (LEAVE_COUNT) ---
        st.markdown("---")
        st.header("**Absenteeism (Days) by Health Grade (GHC)**")
        if not drv_lsa_ghc.empty:
            drv_lsa_ghc = drv_lsa_ghc.sort_values(by='FINAL_GRADING', ascending=True).reset_index(drop=True)
            box_plot = alt.Chart(drv_lsa_ghc).mark_boxplot(size=20).encode(x=alt.X('FINAL_GRADING:N', title='Health Grade'), y=alt.Y('LEAVE_COUNT:Q', title='Leave Days'))
            swarm_plot = alt.Chart(drv_lsa_ghc).mark_point(color='red', size=30).encode(x=alt.X('FINAL_GRADING:N', axis=alt.Axis(labelAngle=0)), y=alt.Y('LEAVE_COUNT:Q'), tooltip=[alt.Tooltip('EMPLOYEE_ID', title='Employee ID'), alt.Tooltip('LEAVE_COUNT', title='Leave Days')])
            highlighted_employee = alt.Chart(drv_lsa_ghc[drv_lsa_ghc['EMPLOYEE_ID']==str(self.selected_driver)]).mark_point(color='yellow', size=200, filled=True).encode(x='FINAL_GRADING:N', y='LEAVE_COUNT:Q')
            final_chart = box_plot + swarm_plot + highlighted_employee
            st.altair_chart(final_chart, use_container_width=True)
        else:
            st.info("No absenteeism-by-health data to display.")

        # --- Health Profile details ---
        st.markdown("### Health Profile")
        if not drv_health.empty:
            hr = drv_health.iloc[0]
            st.write(f"**BMI:** {hr.get('BMI','NA')} ({hr.get('BMI_INTERPRET','')})")
            st.write(f"**Blood Pressure:** {hr.get('BLOOD_PRESSURE_SYSTOLIC','NA')}/{hr.get('BLOOD_PRESSURE_DIASTOLIC','NA')} ({hr.get('BLOOD_PRESSURE_INTERPRET','')})")
            st.write(f"**Hemoglobin:** {hr.get('HEMOGLOBIN_VALUE','NA')} ({hr.get('HEMOGLOBIN_INTERPRET','')})")
            st.write(f"**Glucose (Random):** {hr.get('GLUCOSE_RANDOM_VALUE','NA')} ({hr.get('GLUCOSE_INTERPRET','')})")
            st.write(f"**Cholesterol:** {hr.get('TOTAL_CHOLESTROL','NA')} ({hr.get('CHOLESTEROL_INTERPRET','')})")
            st.write(f"**Creatinine:** {hr.get('CREATININE_VALUE','NA')} ({hr.get('CREATININE_INTERPRET','')})")
            st.write(f"**ECG Result:** {hr.get('ECG_INTERPRET','NA')} ({hr.get('ECG_COMMENT','')})")
            st.write(f"**LEFT EYE (Day/Night):** {hr.get('D_LEFT_EYE','NA')} / {hr.get('N_LEFT_EYE','NA')}")
            st.write(f"**RIGHT EYE (Day/Night):** {hr.get('D_RIGHT_EYE','NA')} / {hr.get('N_RIGHT_EYE','NA')}")
            st.write(f"**Final Health Grade:** {hr.get('FINAL_GRADING','NA')}")
        else:
            st.info("No health data available for this driver.")


    # ==========================================================
    # TOP & BOTTOM 5 DRIVER PERFORMANCE
    # ==========================================================
    def compute_top_bottom_5(self):
        # Base filtered data
        ops = self.ops_df[
            (self.ops_df['OPERATIONS_DATE'] >= self.start_date) &
            (self.ops_df['OPERATIONS_DATE'] <= self.end_date)
        ]

        hours_df = pd.merge(
            ops,
            self.ser_df[['SERVICE_NUMBER', 'HOURS']],
            on='SERVICE_NUMBER',
            how='left'
        )
        hours_df['HOURS'] = hours_df['HOURS'].fillna(0)

        leaves = self.abs_df[
            (self.abs_df['DATE'] >= self.start_date) &
            (self.abs_df['DATE'] <= self.end_date) &
            (self.abs_df['LEAVE_TYPE'].isin(['L', 'S', 'A']))
        ]

        # Aggregate metrics
        perf = ops.groupby('EMPLOYEE_ID').agg(
            TOTAL_KMS=('OPD_KMS', 'sum'),
            TOTAL_EARNINGS=('DAILY_EARNINGS', 'sum')
        ).reset_index()

        hrs = hours_df.groupby('EMPLOYEE_ID')['HOURS'].sum().reset_index()
        lv = leaves.groupby('EMPLOYEE_ID').size().reset_index(name='LEAVE_COUNT')

        perf = perf.merge(hrs, on='EMPLOYEE_ID', how='left')
        perf = perf.merge(lv, on='EMPLOYEE_ID', how='left')

        perf[['HOURS', 'LEAVE_COUNT']] = perf[['HOURS', 'LEAVE_COUNT']].fillna(0)

        # Normalization helper
        def norm(s):
            return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 0

        perf['SCORE'] = (
            norm(perf['TOTAL_KMS']) +
            norm(perf['TOTAL_EARNINGS']) +
            norm(perf['HOURS']) -
            norm(perf['LEAVE_COUNT'])
        )

        top5 = perf.sort_values('SCORE', ascending=False).head(5)
        bottom5 = perf.sort_values('SCORE').head(5)

        return top5, bottom5



    # ---------------- Depot tab (full) ----------------
    def driver_depot_ui(self):
    # Defensive guards
        if not hasattr(self, 'selected_depot') or not self.selected_depot:
            st.error("Selected depot not set. Return to parameters.")
            return
        if not hasattr(self, 'selected_driver') or not self.selected_driver:
            st.error("Selected driver not set. Return to parameters.")
            return

        # --- small helpers inside method ---
        def has_cols(df, cols):
            return isinstance(df, pd.DataFrame) and set(cols).issubset(set(df.columns))

        def safe_filter(df, conds):
            # conds: list of boolean series or (col, op, value) tuples
            if not isinstance(df, pd.DataFrame) or df.empty:
                return pd.DataFrame()
            mask = pd.Series(True, index=df.index)
            for c in conds:
                if isinstance(c, tuple) and len(c) == 3:
                    col, op, val = c
                    if col not in df.columns:
                        # if column missing, return empty
                        return pd.DataFrame()
                    if op == '==':
                        mask &= (df[col] == val)
                    elif op == '>=':
                        mask &= (df[col] >= val)
                    elif op == '<=':
                        mask &= (df[col] <= val)
                    elif op == 'in':
                        mask &= df[col].isin(val)
                    else:
                        # unsupported op; skip
                        pass
                elif isinstance(c, pd.Series):
                    mask &= c
            return df.loc[mask].copy()

        # ---------------- Filter driver-specific data ----------------
        # Only filter if required columns exist, else get empty DF
        drv_ops = pd.DataFrame()
        if has_cols(self.ops_df, ['EMPLOYEE_ID', 'DEPOT', 'OPERATIONS_DATE']):
            drv_ops = self.ops_df[
                (self.ops_df['EMPLOYEE_ID'].astype(str) == str(self.selected_driver)) &
                (self.ops_df['DEPOT'] == self.selected_depot) &
                (self.ops_df['OPERATIONS_DATE'] >= self.start_date) &
                (self.ops_df['OPERATIONS_DATE'] <= self.end_date)
            ].copy()

        drv_leaves = pd.DataFrame()
        if has_cols(self.abs_df, ['EMPLOYEE_ID', 'DEPOT', 'DATE']):
            drv_leaves = self.abs_df[
                (self.abs_df['EMPLOYEE_ID'].astype(str) == str(self.selected_driver)) &
                (self.abs_df['DEPOT'] == self.selected_depot) &
                (self.abs_df['DATE'] >= self.start_date) &
                (self.abs_df['DATE'] <= self.end_date)
            ].copy()

        # ---------------- Build drv_hours safely (merge ops_df with ser_df on SERVICE_NUMBER) ----------------
        # Prepare safe ser_small (SERVICE_NUMBER,HOURS) or empty fallback
        if has_cols(self.ser_df, ['SERVICE_NUMBER', 'HOURS']):
            ser_small = self.ser_df[['SERVICE_NUMBER', 'HOURS']].drop_duplicates(subset=['SERVICE_NUMBER']).copy()
            ser_small['HOURS'] = pd.to_numeric(ser_small['HOURS'], errors='coerce').fillna(0)
        else:
            ser_small = pd.DataFrame(columns=['SERVICE_NUMBER', 'HOURS'])
            # Optional user-visible warning:
            if isinstance(self.ser_df, pd.DataFrame):
                st.write("")
            else:
                pass
                st.warning("Warning: service_master not loaded — service HOURS defaulting to 0.")

        # Merge operations with service hours (left join so ops rows are preserved)
        if isinstance(self.ops_df, pd.DataFrame) and not self.ops_df.empty:
            # If ops_df doesn't have SERVICE_NUMBER column, fallback to ops copy + HOURS=0
            if 'SERVICE_NUMBER' in self.ops_df.columns:
                try:
                    drv_hours = pd.merge(self.ops_df, ser_small, on='SERVICE_NUMBER', how='left')
                except Exception:
                    # Unexpected merge failure: fallback
                    drv_hours = self.ops_df.copy()
                    drv_hours['HOURS'] = 0
            else:
                drv_hours = self.ops_df.copy()
                drv_hours['HOURS'] = 0
        else:
            drv_hours = pd.DataFrame()

        # Ensure HOURS column exists and numeric
        if isinstance(drv_hours, pd.DataFrame):
            if 'HOURS' not in drv_hours.columns:
                drv_hours['HOURS'] = 0
            drv_hours['HOURS'] = pd.to_numeric(drv_hours['HOURS'], errors='coerce').fillna(0)

        # Filter drv_hours for depot + date range (all employees)
        if not drv_hours.empty and has_cols(drv_hours, ['DEPOT', 'OPERATIONS_DATE']):
            drv_hours2 = drv_hours[
                (drv_hours['DEPOT'] == self.selected_depot) &
                (drv_hours['OPERATIONS_DATE'] >= self.start_date) &
                (drv_hours['OPERATIONS_DATE'] <= self.end_date)
            ].copy()
        else:
            drv_hours2 = pd.DataFrame()

        # Filter drv_hours for selected employee
        if not drv_hours.empty and has_cols(drv_hours, ['DEPOT', 'EMPLOYEE_ID', 'OPERATIONS_DATE']):
            drv_hours = drv_hours[
                (drv_hours['DEPOT'] == self.selected_depot) &
                (drv_hours['EMPLOYEE_ID'].astype(str) == str(self.selected_driver)) &
                (drv_hours['OPERATIONS_DATE'] >= self.start_date) &
                (drv_hours['OPERATIONS_DATE'] <= self.end_date)
            ].copy()
        else:
            drv_hours = pd.DataFrame()

        # ---------------- Driver health data ----------------
        drv_health = pd.DataFrame()
        if has_cols(self.ghc1_df, ['EMPLOYEE_ID']):
            drv_health = self.ghc1_df[self.ghc1_df['EMPLOYEE_ID'].astype(str) == str(self.selected_driver)].copy()

        # ---------------- Merge GHC with hours for productivity by health grade ----------------
        drv_ghcgrade = pd.DataFrame()
        if not drv_hours2.empty and has_cols(self.ghc1_df, ['EMPLOYEE_ID', 'AGE', 'FINAL_GRADING']):
            # Merge by EMPLOYEE_ID; keep RIGHT to retain employees in drv_hours2
            drv_ghcgrade = pd.merge(
                self.ghc1_df[['EMPLOYEE_ID', 'AGE', 'FINAL_GRADING']].drop_duplicates(subset=['EMPLOYEE_ID']),
                drv_hours2[['EMPLOYEE_ID', 'HOURS']],
                on='EMPLOYEE_ID',
                how='right'
            )
            drv_ghcgrade['HOURS'] = pd.to_numeric(drv_ghcgrade['HOURS'], errors='coerce').fillna(0)

        # ---------------- Absenteeism L+S+A (depot aggregated) ----------------
        drv_lsa_ghc = pd.DataFrame()
        if has_cols(self.abs_df, ['DEPOT', 'DATE', 'LEAVE_TYPE', 'EMPLOYEE_ID']) and has_cols(self.ghc1_df, ['EMPLOYEE_ID', 'FINAL_GRADING']):
            drv_leaves_filtered = self.abs_df[
                (self.abs_df['DEPOT'] == self.selected_depot) &
                (self.abs_df['DATE'] >= self.start_date) &
                (self.abs_df['DATE'] <= self.end_date) &
                (self.abs_df['LEAVE_TYPE'].isin(['L', 'S', 'A']))
            ].copy()
            if not drv_leaves_filtered.empty:
                self.drv_leaves2 = drv_leaves_filtered.groupby('EMPLOYEE_ID').size().reset_index(name='LEAVE_COUNT')
                self.drv_leaves2['LEAVE_COUNT'] = pd.to_numeric(self.drv_leaves2['LEAVE_COUNT'], errors='coerce').fillna(0).astype(int)
                drv_lsa_ghc = pd.merge(
                    self.drv_leaves2,
                    self.ghc1_df[['EMPLOYEE_ID', 'FINAL_GRADING']].drop_duplicates(subset=['EMPLOYEE_ID']),
                    on='EMPLOYEE_ID',
                    how='inner'
                )
                if 'LEAVE_COUNT' in drv_lsa_ghc.columns:
                    drv_lsa_ghc['LEAVE_COUNT'] = pd.to_numeric(drv_lsa_ghc['LEAVE_COUNT'], errors='coerce').fillna(0).astype(int)
                else:
                    drv_lsa_ghc['LEAVE_COUNT'] = 0

        if drv_lsa_ghc.empty:
            st.warning("No absenteeism data available for the selected depot and period.")

        # ---------------- Global Averages (safe) ----------------
        global_ops_time = pd.DataFrame()
        if has_cols(self.ops_df, ['OPERATIONS_DATE', 'OPD_KMS', 'DAILY_EARNINGS']):
            global_ops_time = self.ops_df[
                (self.ops_df['OPERATIONS_DATE'] >= self.start_date) &
                (self.ops_df['OPERATIONS_DATE'] <= self.end_date)
            ].copy()

        # global_hours (safe merge)
        if isinstance(self.ops_df, pd.DataFrame) and not self.ops_df.empty:
            if 'SERVICE_NUMBER' in self.ops_df.columns:
                try:
                    global_hours = pd.merge(self.ops_df, ser_small, on='SERVICE_NUMBER', how='left')
                except Exception:
                    global_hours = self.ops_df.copy()
                    global_hours['HOURS'] = 0
            else:
                global_hours = self.ops_df.copy()
                global_hours['HOURS'] = 0
        else:
            global_hours = pd.DataFrame()

        if isinstance(global_hours, pd.DataFrame):
            if 'HOURS' in global_hours.columns:
                global_hours['HOURS'] = pd.to_numeric(global_hours['HOURS'], errors='coerce').fillna(0)
            else:
                global_hours['HOURS'] = 0

        global_leaves = pd.DataFrame()
        if has_cols(self.abs_df, ['DATE', 'LEAVE_TYPE', 'EMPLOYEE_ID']):
            global_leaves = self.abs_df[
                (self.abs_df['DATE'] >= self.start_date) &
                (self.abs_df['DATE'] <= self.end_date) &
                (self.abs_df['LEAVE_TYPE'].isin(['L', 'S', 'A']))
            ].copy()

        global_kms_avg = global_ops_time['OPD_KMS'].mean() if (isinstance(global_ops_time, pd.DataFrame) and 'OPD_KMS' in global_ops_time.columns and not global_ops_time.empty) else 0
        global_earnings_avg = global_ops_time['DAILY_EARNINGS'].mean() if (isinstance(global_ops_time, pd.DataFrame) and 'DAILY_EARNINGS' in global_ops_time.columns and not global_ops_time.empty) else 0
        global_hours_avg = global_hours['HOURS'].mean() if (isinstance(global_hours, pd.DataFrame) and not global_hours.empty) else 0
        if isinstance(global_leaves, pd.DataFrame) and not global_leaves.empty:
            global_leaves_avg = global_leaves.groupby('EMPLOYEE_ID').size().reset_index(name='LEAVE_COUNT')['LEAVE_COUNT'].mean()
        else:
            global_leaves_avg = 0

        # ---------------- Driver Info & Summary UI ----------------
        driver_info = pd.DataFrame()
        if has_cols(self.driver_df, ['EMPLOYEE_ID']):
            driver_info = self.driver_df[self.driver_df['EMPLOYEE_ID'].astype(str) == str(self.selected_driver)].copy()

        col_det, col_sum = st.columns(2)
        with col_det:
            st.markdown("## Driver Details")
            if not driver_info.empty:
                info_row = driver_info.iloc[0]
                st.write(f"**Name:** {info_row.get('FULL_NAME','N/A')}")
                st.write(f"**Age:** {info_row.get('AGE','N/A')}")
                st.write(f"**Birth Date:** {info_row.get('BIRTH_DATE','N/A')}")
                st.write(f"**Joining Date:** {info_row.get('JOINING_DATE','N/A')}")
                st.write(f"**Gender:** {info_row.get('GENDER','N/A')}")
                st.write(f"**Marital Status:** {info_row.get('MARITAL_STATUS','N/A')}")
            else:
                st.info("No driver details found.")

        with col_sum:
            st.markdown("## Performance Summary")
            total_kms = drv_ops['OPD_KMS'].mean() if (isinstance(drv_ops, pd.DataFrame) and 'OPD_KMS' in drv_ops.columns and not drv_ops.empty) else 0
            total_earnings = drv_ops['DAILY_EARNINGS'].mean() if (isinstance(drv_ops, pd.DataFrame) and 'DAILY_EARNINGS' in drv_ops.columns and not drv_ops.empty) else 0
            total_hours = drv_hours['HOURS'].mean() if (isinstance(drv_hours, pd.DataFrame) and 'HOURS' in drv_hours.columns and not drv_hours.empty) else 0

            # safe L+S+A counts
            if isinstance(drv_leaves, pd.DataFrame) and 'LEAVE_TYPE' in drv_leaves.columns and not drv_leaves.empty:
                l_count = (drv_leaves['LEAVE_TYPE'] == 'L').sum()
                s_count = (drv_leaves['LEAVE_TYPE'] == 'S').sum()
                a_count = (drv_leaves['LEAVE_TYPE'] == 'A').sum()
            else:
                l_count = s_count = a_count = 0
            lsa_leaves = f"{l_count} + {s_count} + {a_count}"

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
                    <td style="font-size:14px; color:#888;">Depot Avg Earnings</td>
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

        # small legend helper (reuse existing or inline)
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

        # ---------------- Depot aggregates and charts ----------------
        # depot_ops_time filtered safely
        depot_ops_time = pd.DataFrame()
        if has_cols(self.ops_df, ['DEPOT', 'OPERATIONS_DATE']):
            depot_ops_time = self.ops_df[
                (self.ops_df['DEPOT'] == self.selected_depot) &
                (self.ops_df['OPERATIONS_DATE'] >= self.start_date) &
                (self.ops_df['OPERATIONS_DATE'] <= self.end_date)
            ].copy()

        depot_kms_avg = depot_ops_time['OPD_KMS'].mean() if ('OPD_KMS' in depot_ops_time.columns and not depot_ops_time.empty) else 0
        depot_earnings_avg = depot_ops_time['DAILY_EARNINGS'].mean() if ('DAILY_EARNINGS' in depot_ops_time.columns and not depot_ops_time.empty) else 0
        depot_hours_avg = drv_hours2['HOURS'].mean() if ('HOURS' in drv_hours2.columns and not drv_hours2.empty) else 0

        # Total Kilometers by employee
        st.markdown("### Total Kilometers Driven by All Employees")
        if not depot_ops_time.empty and 'EMPLOYEE_ID' in depot_ops_time.columns and 'OPD_KMS' in depot_ops_time.columns:
            all_emp_kms = depot_ops_time.groupby('EMPLOYEE_ID')['OPD_KMS'].sum().reset_index()
            all_emp_kms = all_emp_kms.sort_values('OPD_KMS', ascending=False).reset_index(drop=True)
            all_emp_kms['is_selected'] = all_emp_kms['EMPLOYEE_ID'].astype(str) == str(self.selected_driver)
            avg_emp_kms = all_emp_kms['OPD_KMS'].mean() if not all_emp_kms.empty else 0

            bars_kms = alt.Chart(all_emp_kms).mark_bar().encode(
                x=alt.X('EMPLOYEE_ID:N', title='Employee ID', sort=all_emp_kms['EMPLOYEE_ID'].tolist()),
                y=alt.Y('OPD_KMS:Q', title='Total Kilometers'),
                tooltip=['EMPLOYEE_ID','OPD_KMS'],
                color=alt.condition(alt.datum.is_selected, alt.value('red'), alt.value('#1f77b4'))
            )
            avg_df = pd.DataFrame({'avg_kms':[avg_emp_kms]})
            avg_line = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[4,2], size=2).encode(y='avg_kms:Q')
            avg_text = alt.Chart(avg_df).mark_text(color='red', dx=5, dy=-10).encode(y='avg_kms:Q', text=alt.value(f"Avg: {avg_emp_kms:,.0f}"))
            st.altair_chart((bars_kms + avg_line + avg_text).properties(width=900), use_container_width=True)
            chart_legend("Depot Employees: Blue", "#1f77b4", "Selected Employee: Red", "red", "Average Line: Red Dashed")
        else:
            st.info("No kilometers data to display for this depot/period.")

        # Total earnings
        if not depot_ops_time.empty and 'DAILY_EARNINGS' in depot_ops_time.columns:
            st.markdown("### Total Earnings of All Employees")
            all_emp_earnings = depot_ops_time.groupby('EMPLOYEE_ID')['DAILY_EARNINGS'].sum().reset_index()
            all_emp_earnings = all_emp_earnings.sort_values('DAILY_EARNINGS', ascending=False).reset_index(drop=True)
            all_emp_earnings['is_selected'] = all_emp_earnings['EMPLOYEE_ID'].astype(str) == str(self.selected_driver)
            avg_emp_earnings = all_emp_earnings['DAILY_EARNINGS'].mean() if not all_emp_earnings.empty else 0

            bars_earnings = alt.Chart(all_emp_earnings).mark_bar().encode(
                x=alt.X('EMPLOYEE_ID:N', title='Employee ID', sort=all_emp_earnings['EMPLOYEE_ID'].tolist()),
                y=alt.Y('DAILY_EARNINGS:Q', title='Total Earnings'),
                tooltip=['EMPLOYEE_ID','DAILY_EARNINGS'],
                color=alt.condition(alt.datum.is_selected, alt.value('red'), alt.value('#1f77b4'))
            )
            avg_df = pd.DataFrame({'avg_earn':[avg_emp_earnings]})
            avg_line = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[4,2], size=2).encode(y='avg_earn:Q')
            avg_text = alt.Chart(avg_df).mark_text(color='red', dx=5, dy=-10).encode(y='avg_earn:Q', text=alt.value(f"Avg: ₹{avg_emp_earnings:,.0f}"))
            st.altair_chart((bars_earnings + avg_line + avg_text).properties(width=900), use_container_width=True)
            chart_legend("Depot Employees: Blue", "#1f77b4", "Selected Employee: Red", "red", "Average Line: Red Dashed")

        # Total productivity hours
        if not drv_hours2.empty and 'EMPLOYEE_ID' in drv_hours2.columns and 'HOURS' in drv_hours2.columns:
            st.markdown("### Total Productivity Hours of All Employees")
            all_emp_hours = drv_hours2.groupby('EMPLOYEE_ID')['HOURS'].sum().reset_index()
            all_emp_hours = all_emp_hours.sort_values('HOURS', ascending=False).reset_index(drop=True)
            all_emp_hours['is_selected'] = all_emp_hours['EMPLOYEE_ID'].astype(str) == str(self.selected_driver)
            avg_emp_hours = all_emp_hours['HOURS'].mean() if not all_emp_hours.empty else 0

            bars_hours = alt.Chart(all_emp_hours).mark_bar().encode(
                x=alt.X('EMPLOYEE_ID:N', title='Employee ID', sort=all_emp_hours['EMPLOYEE_ID'].tolist()),
                y=alt.Y('HOURS:Q', title='Total Hours'),
                tooltip=['EMPLOYEE_ID','HOURS'],
                color=alt.condition(alt.datum.is_selected, alt.value('red'), alt.value('#1f77b4'))
            )
            avg_df = pd.DataFrame({'avg_hours':[avg_emp_hours]})
            avg_line = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[4,2], size=2).encode(y='avg_hours:Q')
            avg_text = alt.Chart(avg_df).mark_text(color='red', dx=5, dy=-10).encode(y='avg_hours:Q', text=alt.value(f"Avg: {avg_emp_hours:,.0f}"))
            st.altair_chart((bars_hours + avg_line + avg_text).properties(width=900), use_container_width=True)
            chart_legend("Depot Employees: Blue", "#1f77b4", "Selected Employee: Red", "red", "Average Line: Red Dashed")
        else:
            st.info("No hours data to display for this depot/period.")

        # Leave days
        if hasattr(self, 'drv_leaves2') and isinstance(self.drv_leaves2, pd.DataFrame) and not self.drv_leaves2.empty:
            st.markdown("### Total Leave Days (L+S+A) of All Employees")
            all_emp_leaves = self.drv_leaves2.groupby('EMPLOYEE_ID')['LEAVE_COUNT'].sum().reset_index(name='Leave_Days')
            all_emp_leaves = all_emp_leaves.sort_values('Leave_Days', ascending=False).reset_index(drop=True)
            all_emp_leaves['is_selected'] = all_emp_leaves['EMPLOYEE_ID'].astype(str) == str(self.selected_driver)
            avg_leave_days = all_emp_leaves['Leave_Days'].mean() if not all_emp_leaves.empty else 0

            bars_leaves = alt.Chart(all_emp_leaves).mark_bar().encode(
                x=alt.X('EMPLOYEE_ID:N', title='Employee ID', sort=all_emp_leaves['EMPLOYEE_ID'].tolist()),
                y=alt.Y('Leave_Days:Q', title='Total Leave Days'),
                tooltip=['EMPLOYEE_ID','Leave_Days'],
                color=alt.condition(alt.datum.is_selected, alt.value('red'), alt.value('#1f77b4'))
            )
            avg_df = pd.DataFrame({'avg_leave':[avg_leave_days]})
            avg_line = alt.Chart(avg_df).mark_rule(color='red', strokeDash=[4,2], size=2).encode(y='avg_leave:Q')
            avg_text = alt.Chart(avg_df).mark_text(color='red', dx=5, dy=-10).encode(y='avg_leave:Q', text=alt.value(f"Avg: {avg_leave_days:,.0f}"))
            st.altair_chart((bars_leaves + avg_line + avg_text).properties(width=900), use_container_width=True)
            chart_legend("Depot Employees: Blue", "#1f77b4", "Selected Employee: Red", "red", "Average Line: Red Dashed")
        else:
            st.info("No leave data available for depot.")

        # ---------------- Productivity by Health Grade ----------------
        st.markdown("---")
        st.header("**Productivity (Hours) + Health Grade (GHC2)**")
        if isinstance(self.ghc1_df, pd.DataFrame) and not drv_ghcgrade.empty:
            sorted_data3 = drv_ghcgrade.dropna(subset=['FINAL_GRADING']).groupby(['EMPLOYEE_ID','FINAL_GRADING'], as_index=False)['HOURS'].sum()
            if sorted_data3.empty:
                st.warning("No data available for the selected depot.")
            else:
                box_plot = alt.Chart(sorted_data3).mark_boxplot(size=20).encode(
                    x=alt.X('FINAL_GRADING:N', title='Health Grade'),
                    y=alt.Y('HOURS:Q', title='Hours')
                )
                swarm_plot = alt.Chart(sorted_data3).mark_point(color='red', size=30).encode(
                    x=alt.X('FINAL_GRADING:N', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('HOURS:Q'),
                    tooltip=[alt.Tooltip('EMPLOYEE_ID', title='Employee ID'), alt.Tooltip('HOURS', title='Hours')]
                )
                highlighted_employee = alt.Chart(sorted_data3[sorted_data3['EMPLOYEE_ID'].astype(str) == str(self.selected_driver)]).mark_point(
                    color='yellow', size=200, filled=True).encode(x='FINAL_GRADING:N', y='HOURS:Q')
                final_chart = box_plot + swarm_plot + highlighted_employee
                st.altair_chart(final_chart, use_container_width=True)
        else:
            st.info("No productivity-by-health data for depot.")

        # ---------------- Absenteeism by Health Grade ----------------
        st.markdown("---")
        st.header("**ABSENTEEISM(DAYS) + Health Grade (GHC2)**")
        if not drv_lsa_ghc.empty:
            drv_lsa_ghc = drv_lsa_ghc.sort_values(by='FINAL_GRADING', ascending=True).reset_index(drop=True)
            if drv_lsa_ghc.empty:
                st.warning("No data available for the selected depot.")
            else:
                box_plot = alt.Chart(drv_lsa_ghc).mark_boxplot(size=20).encode(
                    x=alt.X('FINAL_GRADING:N', title='Health Grade'),
                    y=alt.Y('LEAVE_COUNT:Q', title='Leave Days')
                )
                swarm_plot = alt.Chart(drv_lsa_ghc).mark_point(color='red', size=30).encode(
                    x=alt.X('FINAL_GRADING:N', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('LEAVE_COUNT:Q'),
                    tooltip=[alt.Tooltip('EMPLOYEE_ID', title='Employee ID'), alt.Tooltip('LEAVE_COUNT', title='Leaves')]
                )
                highlighted_employee = alt.Chart(drv_lsa_ghc[drv_lsa_ghc['EMPLOYEE_ID'].astype(str) == str(self.selected_driver)]).mark_point(
                    color='yellow', size=200, filled=True).encode(x='FINAL_GRADING:N', y='LEAVE_COUNT:Q')
                final_chart = box_plot + swarm_plot + highlighted_employee
                st.altair_chart(final_chart, use_container_width=True)
        else:
            st.info("No absenteeism-by-health data for depot.")

        # ---------------- Health profile details ----------------
        st.markdown("### Health Profile")
        if not drv_health.empty:
            hr = drv_health.iloc[0]
            st.write(f"**BMI:** {hr.get('BMI','NA')} ({hr.get('BMI_INTERPRET','')})")
            st.write(f"**Blood Pressure:** {hr.get('BLOOD_PRESSURE_SYSTOLIC','NA')}/{hr.get('BLOOD_PRESSURE_DIASTOLIC','NA')} ({hr.get('BLOOD_PRESSURE_INTERPRET','')})")
            st.write(f"**Hemoglobin:** {hr.get('HEMOGLOBIN_VALUE','NA')} ({hr.get('HEMOGLOBIN_INTERPRET','')})")
            st.write(f"**Glucose (Random):** {hr.get('GLUCOSE_RANDOM_VALUE','NA')} ({hr.get('GLUCOSE_INTERPRET','')})")
            st.write(f"**Cholesterol:** {hr.get('TOTAL_CHOLESTROL','NA')} ({hr.get('CHOLESTEROL_INTERPRET','')})")
            st.write(f"**Creatinine:** {hr.get('CREATININE_VALUE','NA')} ({hr.get('CREATININE_INTERPRET','')})")
            st.write(f"**ECG Result:** {hr.get('ECG_INTERPRET','NA')} ({hr.get('ECG_COMMENT','')})")
            st.write(f"**LEFT EYE (Day/Night):** {hr.get('D_LEFT_EYE','NA')} / {hr.get('N_LEFT_EYE','NA')}")
            st.write(f"**RIGHT EYE (Day/Night):** {hr.get('D_RIGHT_EYE','NA')} / {hr.get('N_RIGHT_EYE','NA')}")
            st.write(f"**Final Health Grade:** {hr.get('FINAL_GRADING','NA')}")
        else:
            st.info("No health data available for this driver.")


        # ================================================================
        # TOP & BOTTOM 5 DRIVERS (PERFORMANCE)
        # ================================================================
        st.markdown("---")
        st.header("🏆 Top & Bottom 5 Drivers (Overall Performance)")

        top5, bottom5 = self.compute_top_bottom_5()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Top 5 Drivers")
            st.dataframe(
                top5[['EMPLOYEE_ID', 'TOTAL_KMS', 'TOTAL_EARNINGS', 'HOURS', 'LEAVE_COUNT', 'SCORE']]
                .style
                .background_gradient(cmap='Greens')
                .format({'SCORE': '{:.2f}'}),
                use_container_width=True
            )

        with col2:
            st.subheader("❌ Bottom 5 Drivers")
            st.dataframe(
                bottom5[['EMPLOYEE_ID', 'TOTAL_KMS', 'TOTAL_EARNINGS', 'HOURS', 'LEAVE_COUNT', 'SCORE']]
                .style
                .background_gradient(cmap='Reds')
                .format({'SCORE': '{:.2f}'}),
                use_container_width=True
            )

# ----------------- Main Entry -----------------
if __name__ == '__main__':
    user_region = st.session_state.get("user_region")
    role = st.session_state.get("role")

    if role is None:
        st.error("❌ 'role' is not set in st.session_state.")
        st.stop()

    obj = driver_depot_dashboard_ui(user_region=user_region, role=role)
    obj.parameters()

    tab1, tab2 = st.tabs(["Driver Performance", "Driver Performance in Depot"])
    with tab1:
        obj.driver_ui()
    with tab2:
        obj.driver_depot_ui()
