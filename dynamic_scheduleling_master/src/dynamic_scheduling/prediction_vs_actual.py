import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import plotly.graph_objects as go
from dynamic_scheduling_master.src.dynamic_scheduling.ops_dashboard import load_schedule_for_date
def render_prediction_vs_actual(
        selected_depot,
        schedule_dates,
        selected_schedule_date,
        schedules,
        SCHEDULE_DIR
):
    

    st.subheader("Prediction vs Actual Monitoring")
    # ── Helper: classify a service number ────────────────────────────────
    def _pva_classify(svc_str, scheduled_services):
        s = str(svc_str).strip()
        if s in scheduled_services:
            return "Scheduled"
        if re.fullmatch(r"[89]\d{3}", s):
            return "Extra Service"
        return "New Service"

    # ── Helper: load actuals from gold parquet ────────────────────────────
    @st.cache_data(ttl=300)
    def _pva_load_actuals(date_str, gold_path):
        if not os.path.exists(gold_path):
            return pd.DataFrame(), f"File not found: {gold_path}"
        try:
            df = pd.read_parquet(gold_path)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            date_col = next((c for c in ["date", "ops_date", "service_date"] if c in df.columns), None)
            if date_col is None:
                return pd.DataFrame(), f"No date column found. Columns: {df.columns.tolist()}"
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
            result = df[df[date_col] == date_str].copy()
            return result, None
        except Exception as e:
            return pd.DataFrame(), str(e)

    # ── Gold parquet path (relative to project root) ──────────────────────
    _sched_dir_str = str(SCHEDULE_DIR).rstrip("/\\")
    _project_root  = os.path.dirname(os.path.dirname(_sched_dir_str))
    GOLD_SERVICE_PATH = os.path.join(_project_root, "data", "processed", "ops_daily_service_gold.parquet")

    # ── Date selector — uses same schedule_dates already loaded at top ────
    if not schedule_dates:
        st.info("No schedule dates found. Run Supply Scheduling from the sidebar first.")
    else:
        # Check which dates also have actuals in gold
        @st.cache_data(ttl=300)
        def _pva_dates_status(sched_dates, gold_path):
            status = {}
            if not os.path.exists(gold_path):
                return {d: False for d in sched_dates}
            try:
                df = pd.read_parquet(gold_path)
                df.columns = [c.strip().lower() for c in df.columns]
                date_col = next((c for c in ["date", "ops_date", "service_date"] if c in df.columns), None)
                available = set(
                    pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique()
                ) if date_col else set()
            except Exception:
                available = set()
            return {d: d in available for d in sched_dates}

        dates_status = _pva_dates_status(tuple(schedule_dates), GOLD_SERVICE_PATH)

        pva_sel_idx = st.selectbox(
            "Select Date",
            range(len(schedule_dates)),
            format_func=lambda i: (
                f"{schedule_dates[i]}  ✅ Both available"
                if dates_status.get(schedule_dates[i])
                else f"{schedule_dates[i]}  ⏳ Predictions only"
            ),
            key="pva_date_select",
        )
        pva_date_str = schedule_dates[pva_sel_idx]
        has_actuals  = dates_status.get(pva_date_str, False)

        # ── Status banner ─────────────────────────────────────────────────
        if not has_actuals:
            st.warning(
                f"⏳ Schedule exists for **{pva_date_str}** but actuals have not been processed yet. "
                "Upload the actuals CSV via the **Files Upload** tab, then run the Data Pipeline."
            )
            st.stop()

        st.success(f"✅ Both predictions and actuals available for **{pva_date_str}**.")

        # ── Load schedule — reuse the already-loaded `schedules` dict if
        #    the selected date matches, otherwise load fresh ────────────────
        if pva_date_str == selected_schedule_date and selected_depot in schedules:
            pva_sched_df = schedules[selected_depot].copy()
        else:
            pva_schedules, _ = load_schedule_for_date(pva_date_str, SCHEDULE_DIR)
            if selected_depot not in pva_schedules:
                st.warning(f"No schedule found for depot **{selected_depot}** on {pva_date_str}.")
                st.stop()
            pva_sched_df = pva_schedules[selected_depot].copy()

        if pva_sched_df.empty:
            st.warning(f"Schedule for {selected_depot} on {pva_date_str} is empty.")
            st.stop()

        # ── Load actuals ──────────────────────────────────────────────────
        actuals_all, err = _pva_load_actuals(pva_date_str, GOLD_SERVICE_PATH)
        if err:
            st.error(f"Could not load actuals: {err}")
            st.stop()
        if actuals_all.empty:
            st.warning(f"Gold parquet loaded but no rows found for date {pva_date_str}.")
            st.stop()

        # Filter actuals to selected depot
        depot_col = next((c for c in ["depot", "depot_name", "depot_code"] if c in actuals_all.columns), None)
        if depot_col:
            act_df = actuals_all[
                actuals_all[depot_col].astype(str).str.upper().str.strip() == selected_depot.upper().strip()
            ].copy()
            if act_df.empty:
                avail = actuals_all[depot_col].unique().tolist()
                st.warning(
                    f"No actuals for depot **{selected_depot}** on {pva_date_str}. "
                    f"Depots found in gold parquet: `{avail}`"
                )
                st.stop()
        else:
            act_df = actuals_all.copy()

        # ── Normalise service_number ──────────────────────────────────────
        pva_sched_df["service_number"] = pva_sched_df["service_number"].astype(str).str.strip()
        act_df["service_number"]       = act_df["service_number"].astype(str).str.strip()

        scheduled_svcs = set(pva_sched_df["service_number"])
        act_df["_category"] = act_df["service_number"].apply(lambda s: _pva_classify(s, scheduled_svcs))

        # ── Detect actual columns ─────────────────────────────────────────
        actual_pkm_col      = next((c for c in ["passenger_kms","passenger_km","pkm","actual_pkm"]   if c in act_df.columns), None)
        actual_kms_col      = next((c for c in ["actual_kms","actual_km","kms"]                      if c in act_df.columns), None)
        actual_trips_col    = next((c for c in ["actual_trips","trips"]                              if c in act_df.columns), None)
        actual_or_col       = next((c for c in ["occupancy_ratio","or"]                              if c in act_df.columns), None)
        actual_revenue_col  = next((c for c in ["revenue","earnings","actual_revenue","actual_earnings"] if c in act_df.columns), None)

        # ── Build slim frames ─────────────────────────────────────────────
        sched_cols = ["service_number", "action"]
        for _c in ["route", "product", "dep_time", "allocated_pkm", "planned_kms", "revenue", "cpk", "contribution"]:
            if _c in pva_sched_df.columns:
                sched_cols.append(_c)
        sched_slim = pva_sched_df[sched_cols].copy()

        # Load planned_trips from service master
        _svc_master_path = os.path.join(_project_root, "data", "master", "service_master.csv")
        if os.path.exists(_svc_master_path):
            try:
                _svc_master = pd.read_csv(_svc_master_path)
                _svc_master.columns = [c.strip().lower().replace(" ", "_") for c in _svc_master.columns]
                _trips_col = next((c for c in ["planned_trips", "trips", "no_of_trips", "num_trips"] if c in _svc_master.columns), None)
                if _trips_col:
                    _svc_slim = _svc_master[["service_number", _trips_col]].copy()
                    _svc_slim["service_number"] = _svc_slim["service_number"].astype(str).str.strip()
                    _svc_slim = _svc_slim.rename(columns={_trips_col: "planned_trips"})
                    sched_slim = sched_slim.merge(_svc_slim, on="service_number", how="left")
            except Exception:
                pass

        act_cols = ["service_number", "_category"]
        for _c in [actual_pkm_col, actual_kms_col, actual_trips_col, actual_or_col, actual_revenue_col]:
            if _c:
                act_cols.append(_c)
        act_slim = act_df[act_cols].copy()

        # ── Merge ─────────────────────────────────────────────────────────
        merged = pd.merge(sched_slim, act_slim, on="service_number", how="outer")
        merged["_category"] = merged["_category"].fillna("Scheduled (No Actuals)")

        if actual_pkm_col and "allocated_pkm" in merged.columns:
            merged["pkm_delta"]     = merged[actual_pkm_col] - merged["allocated_pkm"]
            merged["pkm_delta_pct"] = np.where(
                merged["allocated_pkm"].notna() & (merged["allocated_pkm"] != 0),
                merged["pkm_delta"] / merged["allocated_pkm"] * 100,
                float("nan"),
            )

        # ── Summary metrics ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### {selected_depot} — {pva_date_str}")

        action_counts = pva_sched_df["action"].value_counts().to_dict() if "action" in pva_sched_df.columns else {}
        is_epk_pva    = pva_sched_df.get("_engine", pd.Series(dtype=str)).eq("epk").any() if "_engine" in pva_sched_df.columns else False
        extra_svcs    = act_df[act_df["_category"] == "Extra Service"]
        new_svcs      = act_df[act_df["_category"] == "New Service"]

        mc = st.columns(6)
        mc[0].metric("Scheduled Services", len(pva_sched_df))
        mc[1].metric("Actual Services",    len(act_df))
        mc[2].metric("Add Slot" if is_epk_pva else "Increase",
                    int(action_counts.get("ADD_SLOT", action_counts.get("INCREASE", 0))))
        mc[3].metric("Cut" if is_epk_pva else "Decrease",
                    int(action_counts.get("CUT", action_counts.get("DECREASE", 0))))
        mc[4].metric("Extra Services (8/9xxx)", len(extra_svcs))
        mc[5].metric("New Services",            len(new_svcs))

        if actual_pkm_col and "allocated_pkm" in pva_sched_df.columns:
            total_pred = pva_sched_df["allocated_pkm"].sum()
            total_act  = act_df[actual_pkm_col].sum()
            delta_pkm  = total_act - total_pred
            pct_err    = (delta_pkm / total_pred * 100) if total_pred else 0
            pc = st.columns(3)
            pc[0].metric("Total Allocated PKM (Predicted)", f"{total_pred:,.0f}")
            pc[1].metric("Total Actual PKM",                f"{total_act:,.0f}")
            pc[2].metric("PKM Delta", f"{delta_pkm:+,.0f}", delta=f"{pct_err:+.1f}%",
                        delta_color="inverse" if delta_pkm < 0 else "normal")

        # ── Earnings summary ─────────────────────────────────────────────
        if actual_revenue_col:
            total_pred_rev = pd.to_numeric(pva_sched_df["revenue"], errors="coerce").sum() if "revenue" in pva_sched_df.columns else 0
            total_act_rev  = pd.to_numeric(act_df[actual_revenue_col], errors="coerce").sum()
            delta_rev      = total_act_rev - total_pred_rev
            pct_rev        = (delta_rev / total_pred_rev * 100) if total_pred_rev else 0
            rc = st.columns(3)
            rc[0].metric("Total Predicted Revenue",  f"₹{total_pred_rev/100000:,.2f} L")
            rc[1].metric("Total Actual Earnings",    f"₹{total_act_rev/100000:,.2f} L")
            rc[2].metric("Earnings Delta", f"₹{delta_rev/100000:+,.2f} L", delta=f"{pct_rev:+.1f}%",
                        delta_color="inverse" if delta_rev < 0 else "normal")

        # ── Rename map & helpers ──────────────────────────────────────────
        rename_map = {
            "service_number": "Service No.", "action": "Predicted Action",
            "route": "Route", "product": "Product", "dep_time": "Dep Time",
            "allocated_pkm": "Allocated PKM (Predicted)",
            "planned_kms": "Planned KMs", "planned_trips": "Planned Trips",
            "revenue": "Predicted Revenue", "contribution": "Predicted Contribution",
            "_category": "Category",
            "pkm_delta": "PKM Delta", "pkm_delta_pct": "PKM Delta %",
        }
        if actual_pkm_col:     rename_map[actual_pkm_col]     = "Actual PKM"
        if actual_kms_col:     rename_map[actual_kms_col]     = "Actual KMs"
        if actual_trips_col:   rename_map[actual_trips_col]   = "Actual Trips"
        if actual_or_col:      rename_map[actual_or_col]      = "Actual OR"
        if actual_revenue_col: rename_map[actual_revenue_col] = "Actual Earnings"

        col_order = ["Service No.", "Route", "Product", "Dep Time", "Predicted Action",
                    "Allocated PKM (Predicted)", "Planned KMs", "Planned Trips",
                    "Predicted Revenue", "Predicted Contribution",
                    "Actual PKM", "Actual KMs", "Actual Trips", "Actual OR", "Actual Earnings",
                    "PKM Delta", "PKM Delta %", "Category"]

        def _fmt(df_in):
            d = df_in.rename(columns=rename_map)
            d = d[[c for c in col_order if c in d.columns]]
            for col in ["Allocated PKM (Predicted)", "Planned KMs", "Predicted Revenue", "Predicted Contribution",
                        "Actual PKM", "Actual KMs", "Actual Earnings", "PKM Delta"]:
                if col in d.columns:
                    d[col] = d[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
            if "PKM Delta %" in d.columns:
                d["PKM Delta %"] = d["PKM Delta %"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
            if "Actual OR" in d.columns:
                d["Actual OR"]   = d["Actual OR"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
            return d

        def _colour(row):
            cat = row.get("Category", "")
            if cat == "Extra Service":          return ["background-color: #fff3cd"] * len(row)
            if cat == "New Service":            return ["background-color: #cfe2ff"] * len(row)
            if cat == "Scheduled (No Actuals)": return ["background-color: #f8d7da"] * len(row)
            return [""] * len(row)

        # ── Main table ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Service-Level: Predicted vs Actual")

        # Sort rows: ADD_SLOT → NO_CHANGE → CUT → New Service → Extra Service → Scheduled (No Actuals)
        _cat_sort = {
            "ADD_SLOT": 0, "INCREASE": 0,
            "NO_CHANGE": 1,
            "CUT": 2, "DECREASE": 2, "STOP": 2,
            "New Service": 3,
            "Extra Service": 4,
            "Scheduled (No Actuals)": 5,
        }
        def _row_sort_key(row):
            action = str(row.get("action", "")).strip().upper()
            cat    = str(row.get("_category", "")).strip()
            if cat in ("New Service", "Extra Service", "Scheduled (No Actuals)"):
                return _cat_sort.get(cat, 99)
            return _cat_sort.get(action, 99)

        merged["_sort"] = merged.apply(_row_sort_key, axis=1)
        disp_df  = merged.sort_values(["_sort", "service_number"]).drop(columns=["_sort"])
        disp_fmt = _fmt(disp_df.copy())
        st.dataframe(
            disp_fmt.style.apply(_colour, axis=1),
            use_container_width=True,
            height=520,
            hide_index=True,
            column_config={
                "Service No.": st.column_config.TextColumn("Service No.", pinned=True),
            },
        )
        st.markdown(
            "<small>🟡 <b>Extra Service</b> — 8/9xxx not in schedule &nbsp;|&nbsp; "
            "🔵 <b>New Service</b> — other unscheduled &nbsp;|&nbsp; "
            "🔴 <b>Scheduled (No Actuals)</b> — predicted but no matching actual</small>",
            unsafe_allow_html=True,
        )

        # ── PKM bar chart ─────────────────────────────────────────────────
        sched_matched = merged[merged["_category"] == "Scheduled"].copy()
        if actual_pkm_col and "allocated_pkm" in sched_matched.columns and len(sched_matched) > 0:
            st.markdown("---")
            st.markdown("#### PKM: Predicted vs Actual (Scheduled Services)")

            # Ensure numeric
            sched_matched["allocated_pkm"] = pd.to_numeric(sched_matched["allocated_pkm"], errors="coerce")
            sched_matched[actual_pkm_col]  = pd.to_numeric(sched_matched[actual_pkm_col],  errors="coerce")

            # Hard-cast to plain Python float to avoid Plotly rendering mixed-type bugs
            sched_matched["allocated_pkm"] = sched_matched["allocated_pkm"].apply(
                lambda x: float(x) if pd.notna(x) else None
            )
            sched_matched[actual_pkm_col] = sched_matched[actual_pkm_col].apply(
                lambda x: float(x) if pd.notna(x) else None
            )

            chart_df = (
                sched_matched[sched_matched["service_number"].notna()]
                .sort_values("service_number")
                .reset_index(drop=True)
                .copy()
            )

            # Y-axis ceiling from all data
            _all_vals = pd.concat([
                chart_df["allocated_pkm"].dropna(),
                chart_df[actual_pkm_col].dropna(),
            ])
            _y_max = float(_all_vals.max() * 1.15) if len(_all_vals) > 0 else None

            # Verify values are clean before passing to Plotly
            pred_vals = chart_df["allocated_pkm"].tolist()
            act_vals  = chart_df[actual_pkm_col].tolist()

            fig = go.Figure()
            fig.add_bar(
                x=chart_df["service_number"].tolist(),
                y=pred_vals,
                name="Predicted PKM",
                marker_color="#4c72b0",
            )
            fig.add_bar(
                x=chart_df["service_number"].tolist(),
                y=act_vals,
                name="Actual PKM",
                marker_color="#dd8452",
            )
            fig.update_layout(
                barmode="group",
                xaxis_title="Service Number",
                yaxis=dict(
                    title="Passenger KMs",
                    range=[0, _y_max],
                    tickformat=",",
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450,
                margin=dict(l=60, r=20, t=30, b=80),
            )
            st.plotly_chart(fig, use_container_width=True)

