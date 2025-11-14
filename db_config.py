# db_config.py
import json
import urllib.parse
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from contextlib import contextmanager

# ---------------------------------------------------
# LOAD CONFIG
# ---------------------------------------------------
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    sys.exit("❌ ERROR: config.json not found in project root.")

db = config.get("db", {})
host = db.get("host", "localhost")
user = db.get("user", "")
password = urllib.parse.quote_plus(db.get("password", ""))
database = db.get("database", "")

# ---------------------------------------------------
# DATABASE URL  (mysqlconnector = BEST choice)
# ---------------------------------------------------
DATABASE_URL = f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"

# ---------------------------------------------------
# SQLAlchemy Engine
# ---------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,  # Set True for SQL debugging
)

# ---------------------------------------------------
# Session Factory
# ---------------------------------------------------
SessionFactory = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False
)
SessionLocal = scoped_session(SessionFactory)

# ---------------------------------------------------
# ORM Base
# ---------------------------------------------------
Base = declarative_base()

# ---------------------------------------------------
# NORMAL get_session() → RETURNS REAL SESSION
# ---------------------------------------------------
def get_session():
    """
    Returns a real SQLAlchemy session.
    Used by ORMConnection and dashboard files.
    """
    return SessionLocal()

# ---------------------------------------------------
# CONTEXT MANAGER for safe transactions
# ---------------------------------------------------
@contextmanager
def session_scope():
    """
    Use when inserting/updating/deleting using ORM:
        with session_scope() as db:
            db.add(...)
            db.commit()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# ---------------------------------------------------
# Initialize DB (Create tables if not present)
# ---------------------------------------------------
def init_db():
    """
    Creates tables defined in models.py.
    """
    import models  # Ensure all models are imported before creation
    Base.metadata.create_all(bind=engine)
    print("✅ All tables created/verified successfully.")

# ---------------------------------------------------
# Test Database Connection
# ---------------------------------------------------
def test_connection():
    print("🔍 Testing database connection...")
    try:
        conn = engine.connect()
        print(f"✅ Connected successfully as '{user}' to '{database}'")
        conn.close()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

# ---------------------------------------------------
# Run as standalone
# ---------------------------------------------------
if __name__ == "__main__":
    test_connection()
    init_db()
