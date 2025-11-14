import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
from models import (
    DailyOperations,
    DriverAbsenteeism,
    DriverDetails,
    ServiceMaster,
)
from db_config import Base  # Ensure same Base used for model definitions


# ------------------------------------------------------------
# 1️⃣ Create SQLAlchemy Engine (Safe for Special Characters)
# ------------------------------------------------------------
def get_mysql_engine(config):
    """Create SQLAlchemy engine safely from config.json."""
    try:
        db_conf = config["db"]

        # ✅ Encode password safely (important if it contains '@', '#', '$', etc.)
        encoded_password = quote_plus(db_conf["password"])

        conn_url = (
            f"mysql+mysqlconnector://{db_conf['user']}:{encoded_password}"
            f"@{db_conf['host']}/{db_conf['database']}"
        )

        engine = create_engine(conn_url, pool_pre_ping=True)
        st.success("✅ Database connection established successfully.")
        return engine

    except Exception as e:
        st.error(f"❌ Failed to create SQLAlchemy engine: {e}")
        return None


# ------------------------------------------------------------
# 2️⃣ Generic ORM Insert (Strict Mode: Block NaN)
# ------------------------------------------------------------
def insert_to_mysql(engine, df: pd.DataFrame, table_name: str):
    """
    Inserts a pandas DataFrame into a MySQL table using ORM mappings.
    🚨 Strict Mode: If any NaN/null value exists in the transformed data,
    insertion is blocked, and detailed missing info is shown.
    """

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    # Map table_name → ORM model class
    table_map = {
        "daily_operations": DailyOperations,
        "driver_absenteeism": DriverAbsenteeism,
        "driver_details": DriverDetails,
        "service_master": ServiceMaster,
    }

    orm_class = table_map.get(table_name)
    if orm_class is None:
        st.error(f"⚠️ No ORM model found for table: {table_name}")
        session.close()
        return

    # ------------------------------------------------------------
    # 🔍 Step 1: Validate for NaN / Null values
    # ------------------------------------------------------------
    nan_mask = df.isna()

    if nan_mask.any().any():
        st.error("❌ Missing values found in transformed data:")

        # Display detailed report: column → row indices
        for col in df.columns:
            missing_rows = nan_mask.index[nan_mask[col]].tolist()
            if missing_rows:
                st.markdown(f"- **Column:** `{col}` → Rows: {missing_rows}")

        st.warning("🚫 Data load aborted. Please fix missing values and re-transform the CSV.")
        st.stop()  # Stop Streamlit execution immediately (no partial load)

    # ------------------------------------------------------------
    # ✅ Step 2: Proceed only if DataFrame is clean
    # ------------------------------------------------------------
    try:
        # Convert DataFrame to list of dictionaries for ORM bulk insert
        records = df.to_dict(orient="records")

        if not records:
            st.warning("⚠️ No records found to insert.")
            return

        session.bulk_insert_mappings(orm_class, records)
        session.commit()

        st.success(f"✅ Successfully inserted {len(records)} rows into `{table_name}` table.")

    except SQLAlchemyError as e:
        session.rollback()
        st.error(f"❌ ORM insert failed: {str(e)}")

    finally:
        session.close()
