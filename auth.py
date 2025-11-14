# auth.py — Secure version with IST timezone, lockout timer, and anti-session hijacking

import json
import secrets
import bcrypt
import streamlit as st
from datetime import datetime, timedelta
from mysql.connector import connect, Error
import hashlib
import pytz
from db_config import get_session
from models import User  # your ORM model for 'users' table


# -------------------------
# Timezone setup (India Standard Time)
# -------------------------
IST = pytz.timezone("Asia/Kolkata")

def now_ist():
    """Return current IST time (Asia/Kolkata)"""
    return datetime.now(IST)

def to_ist(dt):
    """Convert a naive or UTC datetime to IST safely"""
    if not dt:
        return None
    if dt.tzinfo is None:
        return IST.localize(dt)
    return dt.astimezone(IST)

# -------------------------
# Load config.json
# -------------------------
with open("config.json", "r") as f:
    config = json.load(f)
db_config = config.get("db", {})

# -------------------------
# Security Settings
# -------------------------
LOCKOUT_THRESHOLD = 5        # Failed attempts before lock
LOCKOUT_MINUTES = 15         # Lock duration
SESSION_TTL_MINUTES = 120    # Session validity (2 hours)
ENABLE_SESSION_FINGERPRINT = False  # Optional (True = track device/IP)



# -------------------------
# Utility helpers
# -------------------------
def hash_password(plain_password: str) -> str:
    if isinstance(plain_password, str):
        plain_password = plain_password.encode("utf-8")
    return bcrypt.hashpw(plain_password, bcrypt.gensalt()).decode("utf-8")

def check_password(plain_password: str, hashed_password: str) -> bool:
    try:
        if isinstance(plain_password, str):
            plain_password = plain_password.encode("utf-8")
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode("utf-8")
        return bcrypt.checkpw(plain_password, hashed_password)
    except Exception:
        return False

def _sha256(text: str) -> str:
    """Return SHA-256 hex digest"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def create_session_token():
    """Generate a new secure session token (returns raw token, hashed token)"""
    token = secrets.token_urlsafe(32)
    token_hash = _sha256(token)
    return token, token_hash

def _fingerprint_value():
    """Optional device/IP fingerprint (disabled by default)"""
    if not ENABLE_SESSION_FINGERPRINT:
        return None
    ip = st.session_state.get("client_ip")
    ua = st.session_state.get("client_user_agent")
    if not ip and not ua:
        return None
    return _sha256(f"{ip}|{ua}")

# -------------------------
# Depot / settings
# -------------------------
def fetch_depot_names():
    try:
        with get_session() as db:
            depot_list = [d.depot_name for d in db.query(TSAdmin.depot_name).all()]
            return depot_list

    except Exception as e:
        print("Error fetching depots:", e)
        return []

from models import TSAdmin  # add this at the top if not already

def get_depot_settings():
    try:
        with get_session() as db:
            depots = db.query(TSAdmin).all()
            return {d.depot_name: {
                "zone": d.zone,
                "region": d.region,
                "depot_name": d.depot_name,
                "category": d.category
            } for d in depots}
    except Exception as e:
        print("Error fetching depot settings:", e)
        return {}


# -------------------------
# Admin setup
# -------------------------
def ensure_admin_exists(default_admin_user="admin"):
    """Ensures default admin exists with secure password"""
    try:
        with get_session() as db:
            exists = db.query(User).filter(User.userid == default_admin_user).count()
            if exists == 0:
                temp_pw = secrets.token_urlsafe(12)
                pw_hash = hash_password(temp_pw)
                admin_user = User(userid=default_admin_user, password_hash=pw_hash, role="admin")
                db.add(admin_user)
                db.commit()
                print(f"[ADMIN CREATED] userid='{default_admin_user}' temporary password: {temp_pw}")
                print("Please change this admin password immediately.")
    except Exception as e:
        print("ensure_admin_exists error:", e)


# -------------------------
# Authentication (secure)
# -------------------------
def authenticate_user(userid, password):
    """
    Secure authentication flow with IST timestamping + lockout feedback.
    Returns (success: bool, depot: str | None, lockout_until: datetime | None)
    """
    userid = (userid or "").strip()
    if not userid or not password:
        return False, None, None

    try:
        with get_session() as db:
            user = db.query(User).filter(User.userid == userid).first()

            if not user:
                return False, None, None

            now = now_ist()
            locked_until_ist = to_ist(user.locked_until)

            # 🚫 Locked account
            if locked_until_ist and locked_until_ist > now:
                return False, None, locked_until_ist

            # ✅ bcrypt password (secure)
            if user.password_hash and check_password(password, user.password_hash):
                token, token_hash = create_session_token()
                expires_at = now + timedelta(minutes=SESSION_TTL_MINUTES)
                fingerprint = _fingerprint_value()

                user.failed_attempts = 0
                user.last_failed_at = None
                user.locked_until = None
                user.session_token = token_hash
                user.session_expires_at = expires_at
                user.session_fp = fingerprint
                db.commit()

                st.session_state["user_id"] = user.id
                st.session_state["userid"] = userid
                st.session_state["session_token"] = token
                st.session_state["session_expires_at"] = expires_at.isoformat()
                if fingerprint:
                    st.session_state["session_fp"] = fingerprint

                return True, user.depot, None

            # 🧩 Fallback plaintext → bcrypt migrate
            if user.password and user.password == password:
                new_hash = hash_password(password)
                token, token_hash = create_session_token()
                expires_at = now + timedelta(minutes=SESSION_TTL_MINUTES)
                fingerprint = _fingerprint_value()

                user.password_hash = new_hash
                user.failed_attempts = 0
                user.last_failed_at = None
                user.locked_until = None
                user.session_token = token_hash
                user.session_expires_at = expires_at
                user.session_fp = fingerprint
                db.commit()

                st.session_state["user_id"] = user.id
                st.session_state["userid"] = userid
                st.session_state["session_token"] = token
                st.session_state["session_expires_at"] = expires_at.isoformat()
                if fingerprint:
                    st.session_state["session_fp"] = fingerprint

                return True, user.depot, None

            # ❌ Wrong password
            user.failed_attempts = (user.failed_attempts or 0) + 1
            lock_until = None
            if user.failed_attempts >= LOCKOUT_THRESHOLD:
                lock_until = now + timedelta(minutes=LOCKOUT_MINUTES)

            user.last_failed_at = now
            user.locked_until = lock_until
            db.commit()
            return False, None, lock_until

    except Exception as e:
        print("authenticate_user DB error:", e)
        return False, None, None


# -------------------------
# Session validation
# -------------------------
def is_authenticated():
    uid = st.session_state.get("user_id")
    token = st.session_state.get("session_token")
    expires_iso = st.session_state.get("session_expires_at")

    if not uid or not token or not expires_iso:
        return False

    try:
        expires_dt = datetime.fromisoformat(expires_iso)
        if expires_dt < now_ist():
            logout()
            return False
    except Exception:
        logout()
        return False

    try:
        with get_session() as db:
            user = db.query(User).filter(User.id == uid).first()
            if not user:
                logout()
                return False

            db_hash = user.session_token
            if not db_hash or _sha256(token) != db_hash:
                logout()
                return False

            if ENABLE_SESSION_FINGERPRINT:
                db_fp = user.session_fp
                client_fp = st.session_state.get("session_fp")
                if not db_fp or not client_fp or db_fp != client_fp:
                    logout()
                    return False

            db_expires_ist = to_ist(user.session_expires_at)
            if db_expires_ist and db_expires_ist < now_ist():
                logout()
                return False
            return True
    except Exception as e:
        print("is_authenticated DB error:", e)
        return False


# -------------------------
# Logout
# -------------------------
def logout(manual=False):
    try:
        uid = st.session_state.get("user_id")
        if uid:
            with get_session() as db:
                user = db.query(User).filter(User.id == uid).first()
                if user:
                    user.session_token = None
                    user.session_expires_at = None
                    user.session_fp = None
                    db.commit()
    except Exception as e:
        print("logout error:", e)
    finally:
        if manual:
            st.session_state["manual_logout"] = True
        else:
            st.session_state["session_expired"] = True

        for k in ["user_id", "userid", "session_token", "session_expires_at", "session_fp"]:
            if k in st.session_state:
                del st.session_state[k]


# -------------------------
# Convenience helpers
# -------------------------
def get_depot_by_userid(userid):
    try:
        with get_session() as db:
            user = db.query(User.depot).filter(User.userid == userid).first()
            return user.depot if user else None

    except Exception as e:
        print("Error fetching depot:", e)
        return None

def get_role_by_userid(userid):
    try:
        with get_session() as db:
            user = db.query(User.role).filter(User.userid == userid).first()
            return user.role if user else None

    except Exception as e:
        print("Error fetching role:", e)
        return None

def create_user(userid, password, depot=None, role=None):
    if not userid or not password:
        return False
    try:
            with get_session() as db:
                pw_hash = hash_password(password)
                new_user = User(userid=userid, password_hash=pw_hash, depot=depot, role=role)
                db.add(new_user)
                db.commit()
                return True
    except Exception as e:
            print("create_user DB error:", e)
            return False


# -------------------------
# Self-test (optional)
# -------------------------
if __name__ == "__main__":
    st.title("Auth module self-test")
    if is_authenticated():
        st.success(f"Authenticated as {st.session_state.get('userid')}")
        if st.button("Logout"):
            logout()
            st.experimental_rerun()
    else:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            ok, depot, lock = authenticate_user(u, p)
            if ok:
                st.success("Login successful")
                st.experimental_rerun()
            elif lock:
                mins_left = int((lock - now_ist()).total_seconds() / 60)
                st.error(f"🔒 Account locked. Try again in {mins_left} min(s).")
            else:
                st.error("Invalid username or password / account locked.")
