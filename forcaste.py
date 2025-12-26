import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px
import json
import pymysql

# ================================
# MySQL IMPORTS + CONNECTION
# ================================
import mysql.connector
from sqlalchemy import create_engine
# ================================
# LOAD DB CONFIG (same as Eight Ratios DM)
# ================================
with open("config.json") as f:
    config = json.load(f)

DB_CONFIG = config.get("db", {})


engine = create_engine(
    f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
)


def get_connection():
    """Reusable MySQL connection"""
    try:
        return pymysql.connect(
            host=DB_CONFIG.get("host"),
            user=DB_CONFIG.get("user"),
            password=DB_CONFIG.get("password"),
            database=DB_CONFIG.get("database"),
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None
    
        
def get_user_depot(conn, userid):
    if not userid:
        return ""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT depot FROM users WHERE userid = %s",
                (userid,)
            )
            row = cur.fetchone()
            return row["depot"] if row else ""
    except Exception as e:
        st.error(f"Error fetching depot for user {userid}: {e}")
        return ""
       
def forecast():
    

    # ================================
    # STEP 5: DB CONNECTION + USER → DEPOT
    # ================================
    conn = get_connection()
    if conn is None:
        st.stop()

    userid = st.session_state.get("userid")

    if userid and userid != "admin":
        selected_depot = get_user_depot(conn, userid)
    else:
        # admin can select depot elsewhere
        selected_depot = st.session_state.get("depot", "")

    if not selected_depot:
        st.error("❌ No depot mapped to this user.")
        conn.close()
        return

    st.subheader(f"🏢 Depot: {selected_depot}")

    # ================================
    # FORECAST SAVE/LOAD FUNCTIONS
    # ================================
    def save_prediction(depot, date, pred):
        conn = get_connection()

        cur = conn.cursor()
        query = """
            INSERT INTO passenger_forecast_store (depot, date, predicted_passengers)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE predicted_passengers = predicted_passengers;
        """
        cur.execute(query, (depot, date, float(pred)))
        conn.commit()
        cur.close()
        conn.close()

    def save_actual(depot, date, actual):
        conn = get_connection()

        cur = conn.cursor()
        query = """
            INSERT INTO passenger_forecast_store (depot, date, actual_passengers)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE actual_passengers = VALUES(actual_passengers);
        """
        cur.execute(query, (depot, date, int(actual)))
        conn.commit()
        cur.close()
        conn.close()

    def get_saved_prediction(depot, date):
        conn = get_connection()

        cur = conn.cursor()

        query = """
            SELECT predicted_passengers
            FROM passenger_forecast_store
            WHERE depot=%s AND date=%s
        """
        cur.execute(query, (depot, date))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row["predicted_passengers"] if row else None

    # ================================
    # Streamlit config
    # ================================
    #st.set_page_config(page_title="Passenger Forecast", layout="wide")
    st.title("🚌 Passenger Forecast Dashboard")

    # ================================
    # Load data
    # ================================
    #@st.cache_data
    def load_data():
        df_train = pd.read_sql("SELECT * FROM predictive_planner_train", con=engine)
        df_pred = pd.read_sql("SELECT * FROM predictive_planner_test", con=engine)

        for df in (df_train, df_pred):
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
            df.dropna(subset=['date'], inplace=True)
            df.sort_values(['depot', 'date'], inplace=True)

        return df_train, df_pred

    df_train, df_pred = load_data()

    # ================================
    # DEPOT FROM LOGIN SESSION
    # ================================
    if not selected_depot:
        st.error("❌ No depot mapped to this user.")
        return

    predict_clicked = st.button("Predict")


    # ================================
    # Feature engineering
    # ================================
    def create_features(df):
        df = df.copy().sort_values('date').reset_index(drop=True)
        tgt = "passengers_per_day"

        df["day"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["dayofweek"] = df["date"].dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["dayofyear"] = df["date"].dt.dayofyear

        for lag in (1, 7, 30):
            df[f"lag_{lag}"] = df[tgt].shift(lag)

        df["roll_mean_3"] = df[tgt].shift(1).rolling(3).mean()
        df["roll_mean_7"] = df[tgt].shift(1).rolling(7).mean()
        df["roll_std_7"] = df[tgt].shift(1).rolling(7).std().fillna(0)
        df["ewm_7"] = df[tgt].shift(1).ewm(span=7).mean()

        df["diff_1"] = df[tgt] - df[tgt].shift(1)
        df["pct_change_7"] = df[tgt].pct_change(7).fillna(0)

        df = df.dropna().reset_index(drop=True)
        return df

    # ================================
    # LABEL ENCODING
    # ================================
    def label_encode(train_df, pred_df):
        train = train_df.copy()
        pred = pred_df.copy()

        encoders = {}
        cat_cols = train.select_dtypes(include="object").columns

        for col in cat_cols:
            le = LabelEncoder()
            all_vals = pd.concat([train[col].astype(str), pred[col].astype(str)])
            le.fit(all_vals)

            encoders[col] = le
            train[col] = le.transform(train[col].astype(str))
            pred[col] = le.transform(pred[col].astype(str))

        return train, pred, encoders

    # ================================
    # FIXED predict_future()
    # ================================
    def predict_future(df_history, model, features, encoders, n_days=2):
        df_work = df_history.copy().sort_values('date').reset_index(drop=True)
        target = "passengers_per_day"
        preds = []
        last_date = df_work['date'].max()

        for _ in range(n_days):

            df_fe = create_features(df_work)
            X_last = df_fe[features].iloc[-1:].copy()

            for col, le in encoders.items():
                if col in X_last.columns:
                    v = str(X_last[col].iloc[0])
                    if v not in le.classes_:
                        if "___UNKNOWN___" not in le.classes_:
                            le.classes_ = np.append(le.classes_, "___UNKNOWN___")
                        X_last[col] = le.transform(["___UNKNOWN___"])[0]
                    else:
                        X_last[col] = le.transform([v])[0]

            pred = float(np.expm1(model.predict(X_last[features])[0]))
            next_date = last_date + timedelta(days=1)

            preds.append((next_date, pred))

            next_row = df_work.iloc[-1:].copy()
            next_row["date"] = next_date
            next_row[target] = pred

            df_work = pd.concat([df_work, next_row], ignore_index=True)
            last_date = next_date

        return preds

    # ================================
    # MAIN PREDICTION
    # ================================
    if predict_clicked and selected_depot:

        df_train_depot = df_train[df_train['depot'] == selected_depot].reset_index(drop=True)
        df_pred_depot = df_pred[df_pred['depot'] == selected_depot].reset_index(drop=True)

        df_train_depot = create_features(df_train_depot)
        df_pred_depot = create_features(df_pred_depot)

        df_train_enc, df_pred_enc, encoders = label_encode(df_train_depot, df_pred_depot)

        target = "passengers_per_day"
        features = [c for c in df_train_enc.columns if c not in ["date", target]]

        X = df_train_enc[features]
        y = np.log1p(df_train_enc[target])

        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        # MODEL PATH
        model_dir = "/home/git/models/forecast"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{selected_depot}.pkl")

        # LOAD OR TRAIN
        if os.path.exists(model_path):
            model, features, metrics, encoders = joblib.load(model_path)
        else:
            base = XGBRegressor(
                n_estimators=600, learning_rate=0.05,
                max_depth=6, subsample=0.85, colsample_bytree=0.85,
                random_state=42, tree_method="hist"
            )
            base.fit(X_train, y_train)

            importances = pd.Series(base.feature_importances_, index=features).sort_values(ascending=False)
            top_features = importances.head(30).index.tolist()

            model = XGBRegressor(
                n_estimators=1000, learning_rate=0.03,
                max_depth=6, subsample=0.9, colsample_bytree=0.9,
                random_state=42, tree_method='hist'
            )
            model.fit(X_train[top_features], y_train)

            val_pred = np.expm1(model.predict(X_val[top_features]))
            y_val_orig = np.expm1(y_val)

            metrics = {
                "MAE": round(mean_absolute_error(y_val_orig, val_pred), 2),
                "RMSE": round(np.sqrt(mean_squared_error(y_val_orig, val_pred)), 2),
                "R2": round(r2_score(y_val_orig, val_pred), 4)
            }

            joblib.dump((model, top_features, metrics, encoders), model_path)

        model, features, metrics, encoders = joblib.load(model_path)

        # ================================
        # PREDICT 2 DAYS
        # ================================
        preds = predict_future(df_pred_depot, model, features, encoders, n_days=2)

        (today_date, today_raw), (tomorrow_date, tomorrow_raw) = preds

        recent_mean = df_pred_depot["passengers_per_day"].tail(7).mean()

        # TODAY
        saved_today = get_saved_prediction(selected_depot, today_date)
        if saved_today is None:
            today_pred = 0.65 * today_raw + 0.35 * recent_mean
            save_prediction(selected_depot, today_date, today_pred)
        else:
            today_pred = saved_today

        # TOMORROW
        saved_tmr = get_saved_prediction(selected_depot, tomorrow_date)
        if saved_tmr is None:
            tomorrow_pred = 0.7 * tomorrow_raw + 0.3 * today_pred
            tomorrow_pred = float(np.clip(tomorrow_pred, today_pred * 0.8, today_pred * 1.2))
            save_prediction(selected_depot, tomorrow_date, tomorrow_pred)
        else:
            tomorrow_pred = saved_tmr

        # ================================
        # SUMMARY TABLE WITH STORED VALUES SUPPORT
        # ================================

        st.subheader(f"📅 Passenger Forecast Summary – {selected_depot}")

        # Get last 6 actual days
        df_recent_fe = create_features(df_pred_depot).tail(6).copy()

        # Encode for XGBoost prediction
        for col, le in encoders.items():
            if col in df_recent_fe.columns:
                df_recent_fe[col] = df_recent_fe[col].astype(str).apply(
                    lambda v: v if v in le.classes_ else "___UNKNOWN___"
                )
                df_recent_fe[col] = le.transform(df_recent_fe[col])

        # --------------------------------------------
        # STEP 1: Predict last 6 days (model prediction)
        # --------------------------------------------
        df_recent_fe["ModelPred"] = np.expm1(model.predict(df_recent_fe[features]))

        # --------------------------------------------
        # STEP 2: Replace Predicted with STORED values (if available)
        # --------------------------------------------
        final_preds = []
        final_dates = list(df_recent_fe["date"]) + [today_date, tomorrow_date]

        for date in final_dates:

            stored = get_saved_prediction(selected_depot, date)

            if stored is not None:
                final_preds.append(stored)
            else:
                # Predict missing future date
                if date == today_date:
                    final_preds.append(today_pred)
                elif date == tomorrow_date:
                    final_preds.append(tomorrow_pred)
                else:
                    # For older dates, use model fallback
                    row = df_recent_fe[df_recent_fe["date"] == date]
                    if not row.empty:
                        final_preds.append(float(row["ModelPred"].iloc[0]))
                    else:
                        final_preds.append(None)

        st.markdown(f"""
        ### 
        - **Today ({today_date.strftime('%d-%b-%Y')})** →  
        **<span style='font-size:24px;color:#2E86C1'><b>{today_pred:,.0f}</b></span> passengers**

        - **Tomorrow ({tomorrow_date.strftime('%d-%b-%Y')})** →  
        **<span style='font-size:24px;color:#D35400'><b>{tomorrow_pred:,.0f}</b></span> passengers**
        """, unsafe_allow_html=True)

        st.markdown("---")

        # --------------------------------------------
        # STEP 3: Construct summary table frame
        # --------------------------------------------
        all_dates = final_dates
        all_actuals = list(df_recent_fe["passengers_per_day"]) + [None, None]

        summary_df = pd.DataFrame({
            "Date": all_dates,
            "Actual": all_actuals,
            "Predicted": final_preds,
        })

        # Compute variance
        summary_df["Variance %"] = (
            (summary_df["Predicted"] - summary_df["Actual"]) / summary_df["Actual"]
        ) * 100

        # Format
        summary_df["Date"] = summary_df["Date"].dt.strftime("%Y-%m-%d")
        summary_df["Actual"] = summary_df["Actual"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
        summary_df["Predicted"] = summary_df["Predicted"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
        summary_df["Variance %"] = summary_df["Variance %"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )

        # Same display format as before
        summary_pivot = summary_df.set_index("Date").T.reset_index()
        summary_pivot = summary_pivot.rename(columns={"index": "Metric"})

        st.dataframe(summary_pivot, use_container_width=True, hide_index=True)



        # ================================
        # VISUALIZATION
        # ================================
        st.header("📈 Forecast Visualization")

        df_plot = df_pred_enc.tail(60).copy()
        df_plot["Predicted"] = np.expm1(model.predict(df_plot[features]))
        df_plot["Variance %"] = (df_plot["Predicted"] - df_plot["passengers_per_day"]) / df_plot["passengers_per_day"] * 100
        df_plot["VarSign"] = np.where(df_plot["Variance %"] >= 0, "Positive", "Negative")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["passengers_per_day"], mode="lines+markers", name="Actual"))
        fig1.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["Predicted"], mode="lines+markers", name="Predicted", line=dict(dash="dash")))
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(df_plot, x="date", y="Variance %", color="VarSign")
        st.plotly_chart(fig2, use_container_width=True)

        # st.markdown("---")
        # st.markdown(f"### 📊 Model Performance Metrics ({selected_depot})")
        # st.write(metrics)

        # st.info("✅ Models are now saved permanently in the 'forecast_models/' folder — they will not be deleted.")
