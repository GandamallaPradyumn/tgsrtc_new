import streamlit as st
import datetime
import pandas as pd
import plotly.express as px
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from auth import is_authenticated
from backend_admin import (
    get_or_predict,
    load_data_engineered,
    train_models,
    load_saved_models,
    predict_one_date,
    forecast_days,
    TARGETS
)



# ---------------------------------------------------------
def absadminmain():
    

    st.title("🚍 Absenteeism Forecast Dashboard")

    if "models" not in st.session_state:
        st.session_state.models = load_saved_models()

    if st.button("🧠 Train / Retrain Models"):
        with st.spinner("Training models..."):
            train_models()
            st.session_state.models = load_saved_models()
        st.success("Models trained successfully")

    models = st.session_state.models

    tab1, tab2, tab3 = st.tabs([
        "🔮 Single Day",
        "📆 7-Day Forecast",
        "📊 Variance"
    ])


    def rolling_disclaimer():
        st.markdown("""
        <div style="width:100%; white-space:nowrap; overflow:hidden;
            color:#0049b7; font-weight:600; font-size:15px;">
            <span style="display:inline-block; padding-left:100%;
                        animation:scroll 12s linear infinite;">
                📌 Predictions are statistical estimates and may vary based on manpower, festivals, holidays, or unexpected operational changes.
            </span>
        </div>

        <style>
        @keyframes scroll {
            0% { transform: translate(0,0); }
            100% { transform: translate(-100%,0); }
        }
        </style>
        """, unsafe_allow_html=True)
    def compute_range(value):
        value = float(value)

        if value >= 10:
            low = max(0, int(value - 3))
            high = int(value + 3)
        else:
            low = max(0, int(value - 2))
            high = int(value + 2)

        return low, high

    # -----------------------------------------------------
    # TAB 1
    # -----------------------------------------------------
    with tab1:
        st.header("🔮 Predict Single Day (Today & Tomorrow)")

        df_eng = load_data_engineered()
        depots = sorted(df_eng["depot_name"].unique())
        depot_sel = st.selectbox("Select Depot", depots, key="tab1depot")

        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)

        colA, colB = st.columns(2)

        # ================= TODAY =================
        with colA:
            if st.button(f"Predict Today ({today})"):
                depot_hist = df_eng[df_eng["depot_name"] == depot_sel]

                pred = predict_one_date(
                    depot_sel,
                    today,
                    models,
                    depot_hist
                )

                if pred is None:
                    st.error("❌ Not enough historical data to predict.")
                else:
                    st.subheader(f"📍 {depot_sel} — Today ({today})")

                    cols = st.columns(3)
                    for i, T in enumerate(TARGETS):
                        val = float(pred.get(T, 0))
                        low, high = compute_range(val)

                        cols[i].markdown(f"""
                        <div class='glass-card'>
                            <h3 style='color:#003d99;'>{T.replace("_"," ")}</h3>
                            <h1 style='color:#0066ff; margin-top:-10px;'>{int(val)}</h1>
                            <p style='font-size:16px; color:#444;'>Range: <b>{low} – {high}</b></p>
                        </div>
                        """, unsafe_allow_html=True)

                    rolling_disclaimer()

        # ================= TOMORROW =================
        with colB:
            if st.button(f"Predict Tomorrow ({tomorrow})"):

                if models is None:
                    st.error("❌ Models not loaded.")
                else:
                    depot_hist = df_eng[df_eng["depot_name"] == depot_sel]

                    pred = predict_one_date(
                        depot_sel,
                        tomorrow,
                        models,
                        depot_hist
                    )

                    if pred is None:
                        st.error("❌ Not enough historical data to predict.")
                    else:
                        st.subheader(f"📍 {depot_sel} — Tomorrow ({tomorrow})")

                        cols = st.columns(3)
                        for i, T in enumerate(TARGETS):
                            val = float(pred.get(T, 0))
                            low, high = compute_range(val)

                            cols[i].markdown(f"""
                            <div class='glass-card'>
                                <h3 style='color:#003d99;'>{T.replace("_"," ")}</h3>
                                <h1 style='color:#0066ff; margin-top:-10px;'>{int(val)}</h1>
                                <p style='font-size:16px; color:#444;'>Range: <b>{low} – {high}</b></p>
                            </div>
                            """, unsafe_allow_html=True)

                        rolling_disclaimer()


    # -----------------------------------------------------
    # TAB 2
    # -----------------------------------------------------
    with tab2:
        df = load_data_engineered()
        depot = st.selectbox("Depot", sorted(df["depot_name"].unique()), key="tab2")

        if st.button("Generate 7-Day Forecast"):
            out = forecast_days(depot, datetime.date.today(), 7, models)

            if out.empty:
                st.error("No predictions generated")
                st.stop()

            available = [c for c in TARGETS if c in out.columns]
            out["Total"] = out[available].sum(axis=1)

            st.dataframe(out, use_container_width=True)
            st.plotly_chart(px.line(out, x="Date", y=available, markers=True))

    # -----------------------------------------------------
    # TAB 3 — METRICS
    # -----------------------------------------------------
# =====================================================
# TAB 3 — ACCURACY + VARIANCE (PAST 15 DAYS)
# =====================================================
    with tab3:
        st.header("📊 Accuracy & Variance (Past 15 Days)")

        df = load_data_engineered()
        depot = st.selectbox(
            "Select Depot",
            sorted(df["depot_name"].unique()),
            key="metrics"
        )

        # -------------------------------------------------
        # PREPARE ACTUAL DATA
        # -------------------------------------------------
        hist = df[df["depot_name"] == depot].copy()
        hist = hist.sort_values("data_date")

        if len(hist) < 20:
            st.warning("Not enough historical data for this depot.")
            st.stop()

        actual = hist.tail(15)

        # -------------------------------------------------
        # GENERATE PREDICTIONS (NO LEAKAGE)
        # -------------------------------------------------
        preds = []
        for d in actual["data_date"]:
            p = predict_one_date(
                    depot,
                    d.date(),
                    models,
                    hist
                )


            if p:
                preds.append([
                    d.date(),
                    p.get("Leave_Absent", 0),
                    p.get("Sick_Leave", 0),
                    p.get("Spot_Absent", 0)
                ])

        if not preds:
            st.error("Predictions could not be generated.")
            st.stop()

        pred_df = pd.DataFrame(
            preds,
            columns=["date", "pred_leave", "pred_sick", "pred_spot"]
        )

        # -------------------------------------------------
        # ACTUAL DATAFRAME
        # -------------------------------------------------
        act_df = actual[[
            "data_date",
            "Leave_Absent",
            "Sick_Leave",
            "Spot_Absent"
        ]].copy()

        act_df["date"] = act_df["data_date"].dt.date

        # -------------------------------------------------
        # MERGE ACTUAL + PREDICTED
        # -------------------------------------------------
        dfm = act_df.merge(pred_df, on="date")

        # -------------------------------------------------
        # TOTALS & VARIANCE
        # -------------------------------------------------
        dfm["actual_total"] = (
            dfm["Leave_Absent"] +
            dfm["Sick_Leave"] +
            dfm["Spot_Absent"]
        )

        dfm["pred_total"] = (
            dfm["pred_leave"] +
            dfm["pred_sick"] +
            dfm["pred_spot"]
        )

        # 🔥 TOTAL VARIANCE
        dfm["total_variance"] = dfm["actual_total"] - dfm["pred_total"]

        # 🔥 REMOVE FIRST DATE (lag warm-up noise)
        dfm = dfm.iloc[1:].reset_index(drop=True)

        # -------------------------------------------------
        # METRICS (BUSINESS SAFE)
        # -------------------------------------------------
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        import numpy as np

        # ------------------------------
        # PER-TARGET R² (CORRECT)
        # ------------------------------
        r2_leave = r2_score(dfm["Leave_Absent"], dfm["pred_leave"])
        r2_sick  = r2_score(dfm["Sick_Leave"],  dfm["pred_sick"])
        r2_spot  = r2_score(dfm["Spot_Absent"], dfm["pred_spot"])

        r2_avg = np.mean([r2_leave, r2_sick, r2_spot])

        # ------------------------------
        # TOTAL-LEVEL ERROR METRICS
        # ------------------------------
        y_true_total = dfm["actual_total"].values
        y_pred_total = dfm["pred_total"].values

        mae = mean_absolute_error(y_true_total, y_pred_total)
        rmse = mean_squared_error(y_true_total, y_pred_total) ** 0.5
        mape = (abs(y_true_total - y_pred_total) / (y_true_total + 1)).mean() * 100


        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg R²", f"{r2_avg:.3f}")
        c2.metric("MAE (Total)", f"{mae:.2f}")
        c3.metric("RMSE (Total)", f"{rmse:.2f}")
        c4.metric("MAPE %", f"{mape:.1f}")

        st.caption(
            f"Leave R²: {r2_leave:.3f} | "
            f"Sick R²: {r2_sick:.3f} | "
            f"Spot R²: {r2_spot:.3f}"
        )


        # -------------------------------------------------
        # VARIANCE TABLE
        # -------------------------------------------------
        st.subheader("📄 Actual vs Predicted with Total Variance")
        # -------------------------------------------------
        # FINAL COLUMN ORDER FIX
        # -------------------------------------------------

        # Drop data_date
        dfm = dfm.drop(columns=["data_date"], errors="ignore")

        # Move date to first column
        cols = ["date"] + [c for c in dfm.columns if c != "date"]
        dfm = dfm[cols]

        #st.dataframe(dfm, use_container_width=True)

     
        # -------------------------------------------------
        # TREND GRAPH
        # -------------------------------------------------
        ACTUAL_COLOR = "#1f77b4"   # Blue
        PRED_COLOR   = "#d62728"   # Red

        st.dataframe(dfm, use_container_width=True)

        st.subheader("📈 Total Absentees — Actual vs Predicted")

        fig_total = px.line(
            dfm,
            x="date",
            y=["actual_total", "pred_total"],
            markers=True,
            color_discrete_map={
                "actual_total": ACTUAL_COLOR,
                "pred_total": PRED_COLOR
            }
        )

        fig_total.update_layout(
            yaxis_title="Total Absentees",
            xaxis_title="Date",
            legend_title="",
        )

        st.plotly_chart(fig_total, use_container_width=True)
        st.subheader("📉 Leave Absent — Actual vs Predicted")

        fig_leave = px.line(
            dfm,
            x="date",
            y=["Leave_Absent", "pred_leave"],
            markers=True,
            color_discrete_map={
                "Leave_Absent": ACTUAL_COLOR,
                "pred_leave": PRED_COLOR
            }
        )

        fig_leave.update_layout(
            yaxis_title="Leave Absent",
            xaxis_title="Date",
            legend_title=""
        )

        st.plotly_chart(fig_leave, use_container_width=True)
        st.subheader("📉 Sick Leave — Actual vs Predicted")

        fig_sick = px.line(
            dfm,
            x="date",
            y=["Sick_Leave", "pred_sick"],
            markers=True,
            color_discrete_map={
                "Sick_Leave": ACTUAL_COLOR,
                "pred_sick": PRED_COLOR
            }
        )

        fig_sick.update_layout(
            yaxis_title="Sick Leave",
            xaxis_title="Date",
            legend_title=""
        )

        st.plotly_chart(fig_sick, use_container_width=True)
        st.subheader("📉 Spot Absent — Actual vs Predicted")

        fig_spot = px.line(
            dfm,
            x="date",
            y=["Spot_Absent", "pred_spot"],
            markers=True,
            color_discrete_map={
                "Spot_Absent": ACTUAL_COLOR,
                "pred_spot": PRED_COLOR
            }
        )

        fig_spot.update_layout(
            yaxis_title="Spot Absent",
            xaxis_title="Date",
            legend_title=""
        )

        st.plotly_chart(fig_spot, use_container_width=True)

if __name__ == "__main__":
    absadminmain()
