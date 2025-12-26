import streamlit as st
import mysql.connector
from datetime import datetime
import json


# --- Load config.json ---
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    st.error("Configuration file 'config.json' not found.")
    st.stop()
  
DB_CONFIG = config["db"]

def get_connection():
    conn = mysql.connector.connect(**DB_CONFIG)
    return conn

def insert():
    # -----------------------------------------
    # Fetch depots
    # -----------------------------------------
    def get_depots():
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT depot FROM predictive_planner_test ORDER BY depot;")
        depots = [row[0] for row in cur.fetchall()]
        conn.close()
        return depots

    depots = get_depots()
    depot = st.selectbox("Select Depot", depots)

    date_selected = st.date_input("Select Date to UPDATE (Actual)")
    actual = st.number_input("Enter Actual Passengers", min_value=0)

    submit = st.button("Update Actual & Insert Next Date Row")

    if submit:

        conn = get_connection()
        cur = conn.cursor()

        # Convert update date to dd-mm-yyyy
        date_str = date_selected.strftime("%d-%m-%Y")

        # -----------------------------------------
        # 1️⃣ UPDATE selected date actual
        # -----------------------------------------
        update_sql = """
            UPDATE predictive_planner_test
            SET passengers_per_day = %s
            WHERE depot = %s
            AND date = %s
            AND (passengers_per_day IS NULL OR passengers_per_day = '');
        """

        cur.execute(update_sql, (actual, depot, date_selected))

        conn.commit()

        st.success(f"✔ Updated actual passengers for {depot} on {date_str}")

        # -----------------------------------------
        conn = get_connection()
        cur = conn.cursor()

        # Convert update date to dd-mm-yyyy
        date_str = date_selected.strftime("%d-%m-%Y")

        # -----------------------------------------
        # 1️⃣ UPDATE selected date actual
        # -----------------------------------------
        update_sql = """
            UPDATE passenger_forecast_store
            SET actual_passengers = %s
            WHERE depot = %s
            AND date = %s
            AND (actual_passengers IS NULL OR actual_passengers = '');
        """

        cur.execute(update_sql, (actual, depot, date_selected))

        conn.commit()

        st.success(f"✔ Updated actual passengers for {depot} on {date_str}")

        # -----------------------------------------
        # 2️⃣ FETCH next future feature date
        # Skip dates that already exist in predictive_planner_test
        # -----------------------------------------
        next_date_sql = """
            SELECT p.date 
            FROM predictive_planner p
            WHERE p.depot = %s
            AND p.date > %s
            AND p.date NOT IN (
                    SELECT date 
                    FROM predictive_planner_test 
                    WHERE depot = %s
            )
            ORDER BY p.date
            LIMIT 1;
        """

        cur.execute(next_date_sql, (depot, date_selected, depot))
        next_date_row = cur.fetchone()

        if not next_date_row:
            st.error("❌ No future feature date available in predictive_planner.")
            conn.close()
            return

        next_date_str = next_date_row[0]

        # -----------------------------------------
        # 3️⃣ Fetch full feature row for next date
        # -----------------------------------------
        feature_sql = """
            SELECT depot, date, day_of_month, month_number,
                telugu_thithi, telugu_paksham, marriage_day,
                telugu_month, moudyami_day, festival_day,
                week_day, festival_effect
            FROM predictive_planner
            WHERE depot = %s AND date = %s
            LIMIT 1;
        """

        cur.execute(feature_sql, (depot, next_date_str))
        feature = cur.fetchone()

        if feature:
            # -----------------------------------------
            # 4️⃣ Insert into predictive_planner_test
            # -----------------------------------------
            insert_sql = """
                INSERT INTO predictive_planner_test (
                    depot, date, day_of_month, month_number,
                    passengers_per_day, telugu_thithi, telugu_paksham,
                    marriage_day, telugu_month, moudyami_day, festival_day,
                    week_day, festival_effect
                ) VALUES (%s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s, %s, %s);
            """

            cur.execute(insert_sql, feature)
            conn.commit()

            st.success(f"✔ Inserted next available date ({next_date_str}) for depot {depot}")

        else:
            st.error(f"❌ Feature row not found for {next_date_str}")

        conn.close()
