# =========================================================
# BACKEND: SQL + FEATURE ENGINEERING + ML + PREDICTION
# =========================================================
import json
from auth import db_config, get_role_by_userid, get_depot_by_userid, is_authenticated  # :contentReference[oaicite:1]{index=1}

# import ORM model classes used inside absenteeism_dm (User, TSAdmin)
from models import User, TSAdmin  
from db_config import get_session
from mysql.connector import connect
import os
import pickle
import numpy as np
import pandas as pd
import mysql.connector
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import datetime
from sqlalchemy import create_engine
from urllib.parse import quote_plus

password = quote_plus(db_config["password"])

ENGINE = create_engine(
    f"mysql+mysqlconnector://{db_config['user']}:{password}@{db_config['host']}/{db_config['database']}"
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
MODEL_DIR = "/home/git/model_real"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGETS = ["Leave_Absent", "Sick_Leave", "Spot_Absent"]

# ---------------------------------------------------------
# SQL CONNECTION
# ---------------------------------------------------------
def sql():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="tgsrtc_new"
    )

# ---------------------------------------------------------
# ENSURE PREDICTION TABLE EXISTS
# ---------------------------------------------------------
def ensure_prediction_table():
    conn = sql()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predicted_absenteeism (
            id INT AUTO_INCREMENT PRIMARY KEY,
            depot_name VARCHAR(100),
            data_date DATE,
            Leave_Absent INT,
            Sick_Leave INT,
            Spot_Absent INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_depot_date (depot_name, data_date)
        )
    """)
    conn.commit()
    conn.close()

ensure_prediction_table()

# ---------------------------------------------------------
# LOAD + ENGINEER DATA
# ---------------------------------------------------------
def load_data_engineered():
    conn = sql()
    df = pd.read_sql("SELECT * FROM input_data", ENGINE)
    cal = pd.read_sql("SELECT * FROM calender", ENGINE)
    conn.close()

    df["data_date"] = pd.to_datetime(df["data_date"])

    date_col = next((c for c in cal.columns if "date" in c.lower()), cal.columns[0])
    cal[date_col] = pd.to_datetime(cal[date_col])

    df = df.merge(cal, left_on="data_date", right_on=date_col, how="left")

    df["dow"] = df["data_date"].dt.dayofweek
    df["month"] = df["data_date"].dt.month
    df["weekend"] = (df["dow"] >= 5).astype(int)

    df = df.rename(columns={
        "FESTIVAL_DAY": "festival_day",
        "FESTIVAL_EFFECT": "festival_effect",
        "TYPE_OF_DAY_CODE": "day_type_code"
    })

    for c in ["festival_day", "festival_effect", "day_type_code"]:
        if c not in df.columns:
            df[c] = 0

    return df.sort_values("data_date")

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
def create_features(df, target):
    df = df.sort_values("data_date")

    for lag in [1, 2, 3, 7]:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)

    return df



# ---------------------------------------------------------
# TRAIN MODELS (PER DEPOT, PER TARGET)
# ---------------------------------------------------------
def train_models():
    df = load_data_engineered()

    for depot in df["depot_name"].unique():
        depot_df = df[df["depot_name"] == depot]

        for target in TARGETS:
            data = create_features(depot_df.copy(), target).dropna()
            if len(data) < 60:
                continue

            FEATURES = [
                "dow", "month", "weekend",
                "festival_day", "festival_effect", "day_type_code",
                f"{target}_lag1",
                f"{target}_lag2",
                f"{target}_lag3",
                f"{target}_lag7"
            ]

            X = data[FEATURES]
            y = data[target]   # 🔥 NO SCALING

            model = XGBRegressor(
                n_estimators=700,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=1,
                gamma=0.0,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )

            model.fit(X, y)

            with open(f"{MODEL_DIR}/{depot}_{target}.pkl", "wb") as f:
                pickle.dump(model, f)


# ---------------------------------------------------------
# LOAD SAVED MODELS
# ---------------------------------------------------------
def load_saved_models():
    models = {}
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            with open(os.path.join(MODEL_DIR, f), "rb") as fp:
                models[f.replace(".pkl", "")] = pickle.load(fp)
    return models


# ---------------------------------------------------------
# MODEL PREDICTION ONLY
# ---------------------------------------------------------
def predict_one_date(depot, pred_date, models, history_df):
    hist = history_df.copy().sort_values("data_date")
    results = {}

    for target in TARGETS:
        key = f"{depot}_{target}"

        if key not in models:
            return None

        feat = create_features(hist.copy(), target).dropna()
        if feat.empty:
            return None

        last = feat.iloc[-1:].copy()

        last["dow"] = pred_date.weekday()
        last["month"] = pred_date.month
        last["weekend"] = int(pred_date.weekday() >= 5)

        FEATURES = [
            "dow", "month", "weekend",
            "festival_day", "festival_effect", "day_type_code",
            f"{target}_lag1",
            f"{target}_lag2",
            f"{target}_lag3",
            f"{target}_lag7"
        ]

        pred = models[key].predict(last[FEATURES])[0]
        results[target] = int(round(max(0, pred)))

    return results


# ---------------------------------------------------------
# SQL — FETCH PREDICTION
# ---------------------------------------------------------
def get_prediction_from_sql(depot, date):
    conn = sql()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT Leave_Absent, Sick_Leave, Spot_Absent
        FROM predicted_absenteeism
        WHERE depot_name=%s AND data_date=%s
    """, (depot, date))

    row = cur.fetchone()
    conn.close()
    return row

# ---------------------------------------------------------
# SQL — SAVE PREDICTION (GUARANTEED)
# ---------------------------------------------------------
def save_prediction_to_sql(depot, date, pred):
    conn = sql()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO predicted_absenteeism
        (depot_name, data_date, Leave_Absent, Sick_Leave, Spot_Absent)
        VALUES (%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            Leave_Absent=VALUES(Leave_Absent),
            Sick_Leave=VALUES(Sick_Leave),
            Spot_Absent=VALUES(Spot_Absent)
    """, (
        depot, date,
        int(pred["Leave_Absent"]),
        int(pred["Sick_Leave"]),
        int(pred["Spot_Absent"])
    ))

    conn.commit()
    conn.close()

# ---------------------------------------------------------
# SMART SQL-AWARE PREDICTION
# ---------------------------------------------------------
def get_or_predict(depot, date, models, history_df):

    # 1️⃣ CHECK SQL FIRST
    sql_pred = get_prediction_from_sql(depot, date)
    if sql_pred:
        return {
            "Leave_Absent": int(sql_pred["Leave_Absent"]),
            "Sick_Leave": int(sql_pred["Sick_Leave"]),
            "Spot_Absent": int(sql_pred["Spot_Absent"])
        }, True

    # 2️⃣ MODEL PREDICTION ✅
    pred = predict_one_date(depot, date, models, history_df)

    if pred is None:
        return None, False

    # 3️⃣ SAVE TO SQL ✅
    save_prediction_to_sql(depot, date, pred)

    return pred, False


# ---------------------------------------------------------
# 7-DAY FORECAST (NO SQL SAVE)
# ---------------------------------------------------------
def forecast_days(depot, start_date, days, models):
    df = load_data_engineered()

    # Initialize hist with historical data for the specific depot
    hist = df[df['depot_name'] == depot]  # This filters the historical data for the given depot
    
    out = []

    for i in range(days):
        d = start_date + datetime.timedelta(days=i)
        
        # Get or predict for this date
        pred, _ = get_or_predict(depot, d, models, hist)

        if pred is None:
            continue

        row = {"Date": d}
        row.update(pred)
        out.append(row)

        # Update hist with the new prediction to use for future predictions in the loop
        hist = pd.concat([
            hist,
            pd.DataFrame({
                "data_date": [pd.to_datetime(d)],
                "depot_name": [depot],
                **pred,
                "festival_day": [0],
                "festival_effect": [0],
                "day_type_code": [0]
            })
        ], ignore_index=True)

    return pd.DataFrame(out)
