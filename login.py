import streamlit as st
st.set_page_config(page_title="TGSRTC AI DASHBOARD", layout="wide")

import json
import pandas as pd
import base64
import time
from datetime import datetime

from auth import (
    authenticate_user,
    create_user,
    ensure_admin_exists,
    get_role_by_userid,
    get_depot_by_userid,
    fetch_depot_names,
    is_authenticated,
    logout,
    now_ist
)

# Import your modules
from Input_Data_DM import user_sheet
from Input_Data_RM import RM_sheet
from driver_dashboard_DM import driver_depot_dashboard_ui_DM
from driver_dashboard_RM import driver_depot_dashboard_ui_RM
from admin import admin
from action_plan import action
from action_rm import action_plan_history_rm
from Ratios_DM import prod_ratios_DM
from Ratios_RM import prod_ratios_RM
from eight_ratios_DM import eight_ratios_DM
from eight_ratios_RM import eight_ratios_RM
from depot_dashboard_dm import depot_DM
#from depot_dashboard_rm import depot_RM
from depot_list import depotlist
from pending import pending_depot
from edit_sheet import edit
from Etl_main import run_etl_dashboard

# ORM imports
from db_config import get_session
from models import TSAdmin, User

# ------------------- LOAD CONFIG -------------------
with open("config.json") as f:
    config = json.load(f)
logo_path = config["logo_path"]

# ------------------- ENSURE ADMIN -------------------
ensure_admin_exists()

# ------------------- SESSION INIT -------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.userid = ""
    st.session_state.user_role = None
    st.session_state.user_depot = None
    st.session_state.user_region = None

# ------------------- SESSION FEEDBACK -------------------
if st.session_state.get("session_expired"):
    st.warning("⚠️ Your session has expired. Please log in again.")
    time.sleep(1.5)
    st.session_state["session_expired"] = False
    st.rerun()
elif st.session_state.get("manual_logout"):
    st.info("✅ You have been logged out safely.")
    time.sleep(1.5)
    st.session_state["manual_logout"] = False
    st.rerun()

# ------------------- LOGIN SCREEN -------------------
if not st.session_state.logged_in or not is_authenticated():
    with open(logo_path, "rb") as img_file:
        b64_img = base64.b64encode(img_file.read()).decode()

    st.markdown(f"""
        <div style="text-align: center; background-color: #19bc9c; border-radius: 100px 20px;">
            <br>
            <img src="data:image/png;base64,{b64_img}" width="150" height="150">
            <h1 style="color: white;">Telangana State Road Transport Corporation</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        .stTextInput>div>div>input { background-color: #e4e4e4; color: black; }
        .login-btn button {
            background-color: #F63366 !important;
            color: white !important;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    # Login Form
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            userid = st.text_input("User ID", max_chars=30)
            user_depot_display = get_depot_by_userid(userid) if userid else None
            role = get_role_by_userid(userid) if userid else None
            password = st.text_input("Password", type="password", max_chars=30)

            st.text_input("Role", value=role or "(Role will appear here)", disabled=True)
            st.text_input("Depot/Region", value=user_depot_display or "(Depot/Region will appear here)", disabled=True)

    # Login Button
    login_col = st.columns([1, 2, 1])[1]
    with login_col:
        if st.button("🔐 Login", key="login_button"):
            success, depot, lock_until = authenticate_user(userid, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.userid = userid
                role_from_db = get_role_by_userid(userid)
                st.session_state.user_role = role_from_db

                if role_from_db == "Depot Manager(DMs)":
                    st.session_state.user_depot = depot
                elif role_from_db == "Regional Manager(RMs)":
                    st.session_state.user_region = depot
                else:
                    st.session_state.user_depot = None
                    st.session_state.user_region = None

                st.rerun()
            else:
                if lock_until:
                    remaining = (lock_until - now_ist()).total_seconds() / 60
                    mins_left = max(1, int(remaining))
                    unlock_time = lock_until.strftime("%I:%M %p")
                    st.warning(
                        f"🔒 **Account locked** — Try again in {mins_left} minute(s) "
                        f"(unlocks at **{unlock_time} IST**)."
                    )
                else:
                    st.error("❌ Invalid User ID or Password.")
                time.sleep(1.5)

# ------------------- MAIN APP AFTER LOGIN -------------------
else:
    st.markdown(f"""
    <style>
    @keyframes fadeout {{
        0%   {{ opacity: 1; }}
        80%  {{ opacity: 1; }}
        100% {{ opacity: 0; display: none; }}
    }}
    #welcome {{
        padding: 1rem;
        background-color: #2ecc71;
        color: white;
        text-align: center;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        animation: fadeout 2s forwards;
    }}
    </style>
    <div id="welcome">👋 Welcome, {st.session_state.userid}</div>
    """, unsafe_allow_html=True)

    # Sidebar logout
    with st.sidebar:
        if st.button("🚪 Logout"):
            logout(manual=True)
            st.session_state.logged_in = False
            st.rerun()

    # ------------------- ADMIN -------------------
    if st.session_state.userid == "admin":
        menu = ["Add New User", "Add Depot Category", "INPUT SHEET EDIT","DATA UPLOAD"]
        admin_task = st.sidebar.selectbox("Select screen", menu)
        st.markdown("---")

        if admin_task == "Add New User":
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.subheader("👤 Add New User")
                new_userid = st.text_input("🆕 New User ID")
                new_password = st.text_input("🔑 New Password", type="password")
                roles = [
                    "Select Role",
                    "Depot Manager(DMs)",
                    "Regional Manager(RMs)",
                    "Executive Director(EDs)",
                    "TGSRTC Corporation"
                ]
                role = st.selectbox("🎭 Role", roles, index=0)
                depot_name = region_name = zone_name = None

                # ORM Queries instead of MySQL
                with get_session() as db:
                    if role == "Depot Manager(DMs)":
                        depot_names = [d.depot_name for d in db.query(TSAdmin.depot_name).distinct().order_by(TSAdmin.depot_name)]
                        depot_name = st.selectbox("🏢 Depot Name", ["Select Depot"] + depot_names)
                    elif role == "Regional Manager(RMs)":
                        region_names = [d.region for d in db.query(TSAdmin.region).distinct().order_by(TSAdmin.region)]
                        region_name = st.selectbox("🏙️ Region", ["Select Region"] + region_names)
                    elif role == "Executive Director(EDs)":
                        zone_names = [d.zone for d in db.query(TSAdmin.zone).distinct().order_by(TSAdmin.zone)]
                        zone_name = st.selectbox("🏙️ Zone", ["Select Zone"] + zone_names)

                if st.button("➕ Create New User"):
                    if not new_userid or not new_password:
                        st.warning("⚠️ Please fill both fields.")
                    elif role == "Select Role":
                        st.warning("⚠️ Please select a role.")
                    elif role == "Depot Manager(DMs)" and depot_name not in (None, "Select Depot"):
                        ok = create_user(new_userid, new_password, depot_name, role)
                        st.success(f"✅ User '{new_userid}' created!") if ok else st.error(f"❌ User ID '{new_userid}' exists.")
                    elif role == "Regional Manager(RMs)" and region_name not in (None, "Select Region"):
                        ok = create_user(new_userid, new_password, region_name, role)
                        st.success(f"✅ User '{new_userid}' created!") if ok else st.error(f"❌ User ID '{new_userid}' exists.")
                    elif role == "Executive Director(EDs)" and zone_name not in (None, "Select Zone"):
                        ok = create_user(new_userid, new_password, zone_name, role)
                        st.success(f"✅ User '{new_userid}' created!") if ok else st.error(f"❌ User ID '{new_userid}' exists.")
                    else:
                        st.warning("⚠️ Please select a valid option.")

            st.markdown("---")
            st.subheader("📋 Existing Users")
            try:
                with get_session() as db:
                    users = db.query(User.userid, User.depot, User.role).order_by(User.depot, User.userid).all()
                    df = pd.DataFrame(users, columns=["User ID", "Depot/Region", "Role"])
                    st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Error fetching user data: {e}")

        elif admin_task == "Add Depot Category":
            admin()
        elif admin_task == "INPUT SHEET EDIT":
            edit(); depotlist(); pending_depot()
        elif admin_task == "DATA UPLOAD":
            run_etl_dashboard()

    # ------------------- OTHER ROLES -------------------
    else:
        role = st.session_state.user_role
        if role == "Depot Manager(DMs)":
            menu = [
                "Daily Depot Input Sheet",
                "Productivity Budget 8 Ratios (Rural/Urban)",
                "Productivity Budget vs. Actual 8 Ratios",
                "Action Plan For KPI",
                "Depot Dashboard",
                "Driver Dashboard",
                "AI Depot Tool"
            ]
            selection = st.sidebar.selectbox("Select Screen", menu)
            if selection == "Daily Depot Input Sheet":
                user_sheet(st.session_state.user_depot, role)
            elif selection == "Productivity Budget 8 Ratios (Rural/Urban)":
                prod_ratios_DM()
            elif selection == "Productivity Budget vs. Actual 8 Ratios":
                eight_ratios_DM()
            elif selection == "Depot Dashboard":
                depot_DM()
            elif selection == "Action Plan For KPI":
                action()
            elif selection == "Driver Dashboard":
                obj = driver_depot_dashboard_ui_DM(st.session_state.user_depot, role)
                obj.parameters()
                tab1, tab2 = st.tabs(["Driver Performance", "Driver Performance in Depot"])
                with tab1:
                    obj.driver_ui()
                with tab2:
                    obj.driver_depot_ui()

        elif role == "Regional Manager(RMs)":
            menu = [
                "Daily Depot Input Sheet",
                "Productivity Budget 8 Ratios (Rural/Urban)",
                "Productivity Budget vs. Actual 8 Ratios",
                "Action Plan For KPI",
                "Depot Dashboard",
                "Driver Dashboard"
            ]
            selection = st.sidebar.selectbox("Select Screen", menu)
            if selection == "Daily Depot Input Sheet":
                RM_sheet(st.session_state.user_region, role)
            elif selection == "Productivity Budget 8 Ratios (Rural/Urban)":
                prod_ratios_RM(st.session_state.user_region)
            elif selection == "Productivity Budget vs. Actual 8 Ratios":
                eight_ratios_RM()
            elif selection == "Depot Dashboard":
                depot_RM()
            elif selection == "Driver Dashboard":
                obj = driver_depot_dashboard_ui_RM(st.session_state.user_depot, st.session_state.user_region, role)
                obj.parameters()
                tab1, tab2 = st.tabs(["Driver Performance", "Driver Performance in Depot"])
                with tab1:
                    obj.driver_ui()
                with tab2:
                    obj.driver_depot_ui()
            elif selection == "Action Plan For KPI":
                action_plan_history_rm()