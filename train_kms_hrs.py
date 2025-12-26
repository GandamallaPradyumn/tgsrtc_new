import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
#from sqlalchemy import create_engine
#from db_config import engine
import warnings
import os

MODEL_DIR = '/tmp/my_models/'

os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    # ----------------------
    # Utility
    # ----------------------
    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    warnings.filterwarnings("ignore")

    # ----------------------
    # DB Connection
    # ----------------------
    print("🚀 Connecting to MySQL using config.json...", flush=True)
    from db_config import engine
    # ----------------------
    # 1️⃣ Load Data
    # ----------------------
    oprs_query = """
    SELECT
        depot,
        employee_id,
        operations_date,
        service_number,
        opd_kms
    FROM daily_operations
    """

    hrs_query = """
    SELECT
        depot,
        service_number,
        steering_hours,
        no_of_drvs
    FROM steering_hours
    """

    ghc_query = """
    SELECT employee_id, depot, bmi_interpret, blood_pressure_interpret,
        triglycerides_interpret, ecg_comment, smoke, alcohol,
        arthritis, asthama, final_grading
    FROM health
    """

    oprs_data = pd.read_sql(oprs_query, engine)
    ghc_data = pd.read_sql(ghc_query, engine)
    hrs_df = pd.read_sql(hrs_query, engine)


    oprs_data.columns = oprs_data.columns.str.lower()
    ghc_data.columns = ghc_data.columns.str.lower()
    hrs_df.columns = hrs_df.columns.str.lower()

    # ----------------------
    # Merge steering hours into daily operations
    # ----------------------
    oprs_data = pd.merge(
        oprs_data,
        hrs_df,
        on=["depot", "service_number"],
        how="left"
    )

    # ----------------------
    # Apply driver-based hour adjustment
    # ----------------------
    oprs_data["hours"] = np.where(
        oprs_data["no_of_drvs"].fillna(1) >= 2,
        oprs_data["steering_hours"] / oprs_data["no_of_drvs"].fillna(1),
        oprs_data["steering_hours"]
    )

    # ----------------------
    # Cleanup
    # ----------------------
    oprs_data.drop(columns=["steering_hours", "no_of_drvs"], inplace=True)

    # Safety fallback
    oprs_data["hours"] = oprs_data["hours"].fillna(oprs_data["hours"].median())


    # ----------------------
    # 2️⃣ Date Features
    # ----------------------
    oprs_data["operations_date"] = pd.to_datetime(oprs_data["operations_date"])
    oprs_data["year"] = oprs_data["operations_date"].dt.year
    oprs_data["month"] = oprs_data["operations_date"].dt.month
    oprs_data["day"] = oprs_data["operations_date"].dt.day

    base_year = oprs_data["year"].min()
    oprs_data["year_offset"] = oprs_data["year"] - base_year

    # ----------------------
    # 3️⃣ Merge
    # ----------------------
    combined = pd.merge(
        oprs_data,
        ghc_data,
        on=["employee_id", "depot"],
        how="inner"
    ).dropna().reset_index(drop=True)

    print(f"✅ Data loaded: {combined.shape[0]} rows", flush=True)

    # ----------------------
    # 4️⃣ Feature Engineering (NO employee_id usage)
    # ----------------------

    # Employee averages (numerical only)
    emp_stats = combined.groupby("employee_id")[["opd_kms", "hours"]].mean().rename(
        columns={"opd_kms": "emp_avg_km", "hours": "emp_avg_hr"}
    )
    combined = combined.merge(emp_stats, on="employee_id", how="left")

    # Depot averages
    depot_stats = combined.groupby("depot")[["opd_kms", "hours"]].mean().rename(
        columns={"opd_kms": "depot_avg_km", "hours": "depot_avg_hr"}
    )
    combined = combined.merge(depot_stats, on="depot", how="left")

    # ----------------------
    # 5️⃣ Feature Selection (employee_id REMOVED)
    # ----------------------
    features = [
        'depot', 'year_offset', 'month', 'day',
        'bmi_interpret', 'blood_pressure_interpret',
        'triglycerides_interpret', 'ecg_comment',
        'smoke', 'alcohol', 'arthritis', 'asthama',
        'final_grading',
        'emp_avg_km', 'emp_avg_hr',
        'depot_avg_km', 'depot_avg_hr'
    ]

    target_km = "opd_kms"
    target_hr = "hours"

    # ----------------------
    # 6️⃣ Encode Categoricals (NO employee_id)
    # ----------------------
    cat_cols = [
        'depot', 'bmi_interpret', 'blood_pressure_interpret',
        'triglycerides_interpret', 'ecg_comment',
        'smoke', 'alcohol', 'arthritis', 'asthama',
        'final_grading'
    ]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
        encoders[col] = le

    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

    # ----------------------
    # 7️⃣ Train / Validation Split
    # ----------------------
    X = combined[features]
    y_km = combined[target_km]
    y_hr = combined[target_hr]

    X_train_km, X_val_km, y_train_km, y_val_km = train_test_split(
        X, y_km, test_size=0.2, random_state=42
    )

    X_train_hr, X_val_hr, y_train_hr, y_val_hr = train_test_split(
        X, y_hr, test_size=0.2, random_state=42
    )

    # ----------------------
    # 8️⃣ Train XGBoost
    # ----------------------
    def train_xgb(X_train, y_train, X_val, y_val, label):
        model = xgb.XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)

        print(f"\n📊 {label} Results", flush=True)
        print(f" MAE  : {mean_absolute_error(y_val, preds):.2f}", flush=True)
        print(f" RMSE : {np.sqrt(mean_squared_error(y_val, preds)):.2f}", flush=True)
        print(f" R²   : {r2_score(y_val, preds):.3f}", flush=True)
        print(f" MAPE : {mape(y_val, preds):.2f}%", flush=True)

        return model

    print("\n🧠 Training KM Model", flush=True)
    model_km = train_xgb(X_train_km, y_train_km, X_val_km, y_val_km, "Kilometers")

    print("\n🧠 Training Hours Model", flush=True)
    model_hr = train_xgb(X_train_hr, y_train_hr, X_val_hr, y_val_hr, "Hours")

    # ----------------------
    # 9️⃣ Save
    # ----------------------
    joblib.dump(model_km, os.path.join(MODEL_DIR, "model_oprs.pkl"))
    joblib.dump(model_hr, os.path.join(MODEL_DIR, "model_hrs.pkl"))
    joblib.dump(base_year, os.path.join(MODEL_DIR, "base_year.pkl"))
    print("\n✅ Training completed successfully", flush=True)

if __name__ == "__main__":
    print("🔥 Training script started", flush=True)
    train()
