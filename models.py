# models.py
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    Text,
)
from db_config import Base

# 🔒 Prevent duplicate Base definitions when Streamlit reruns
if "Base" in globals():
    Base = globals()["Base"]

# ----------------------------------------------------------------------
# 1️⃣ users
# ----------------------------------------------------------------------
class User(Base):  # 👈 Rename to singular 'User' for consistency with auth.py import
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    userid = Column(String(255), unique=True, nullable=False)
    password = Column(String(255))  # 👈 made nullable=True (since bcrypt hash is now preferred)
    password_hash = Column(String(255))

    depot = Column(String(255))
    role = Column(String(50))

    failed_attempts = Column(Integer, default=0)
    last_failed_at = Column(DateTime)
    locked_until = Column(DateTime)

    session_token = Column(String(255))
    session_expires_at = Column(DateTime)
    session_fp = Column(String(255))

    def __repr__(self):
        return f"<User(userid='{self.userid}', role='{self.role}', depot='{self.depot}')>"


# ----------------------------------------------------------------------
# 2️⃣ ts_admin
# ----------------------------------------------------------------------
class TSAdmin(Base):
    __tablename__ = "ts_admin"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone = Column(String(100), nullable=False)
    region = Column(String(100), nullable=False)
    depot_name = Column(String(150), nullable=False)
    category = Column(String(100), nullable=False)

# ----------------------------------------------------------------------
# 3️⃣ input_data
# ----------------------------------------------------------------------
class InputData(Base):
    __tablename__ = "input_data"
    __table_args__ = {'extend_existing': True}

    depot_name = Column(String(255), primary_key=True)
    data_date = Column(Date, primary_key=True)
    Schedules = Column(Integer, default=0)
    Schedules_Services = Column(Integer, default=0)
    Schedules_Kms = Column(Integer)
    Planned_Schedules = Column(Integer, default=0)
    Planned_Services = Column(Integer, default=0)
    Planned_KM = Column(Integer, default=0)
    Actual_Services = Column(Integer, default=0)
    Actual_KM = Column(Integer, default=0)
    Service_Variance = Column(Integer, default=0)
    KM_Variance = Column(Integer, default=0)
    Total_Drivers = Column(Integer, default=0)
    Medically_Unfit = Column(Integer, default=0)
    Suspended_Drivers = Column(Integer, default=0)
    Available_Drivers_1 = Column(Integer, default=0)
    Pct_Available_Drivers_1 = Column(Float, default=0)
    Weekly_Off_National_Off = Column(Integer, default=0)
    Pct_Weekly_Off_National_Off = Column(Float, default=0)
    Special_Off_Night_Out_IC_Online = Column(Integer, default=0)
    Pct_Special_Off_Night_Out_IC_Online = Column(Float, default=0)
    Training_PME_medical = Column(Integer, default=0)
    Others = Column(Integer, default=0)
    Pct_Others = Column(Float, default=0)
    Leave_Absent = Column(Integer, default=0)
    Pct_Leave_Absent = Column(Float, default=0)
    Sick_Leave = Column(Integer, default=0)
    Pct_Sick_Leave = Column(Float, default=0)
    Available_Drivers_2 = Column(Integer, default=0)
    Pct_Available_Drivers_2 = Column(Float, default=0)
    Spot_Absent = Column(Integer, default=0)
    Pct_Spot_Absent = Column(Float, default=0)
    Attending_Drivers = Column(Integer, default=0)
    Pct_Attending_Drivers = Column(Float, default=0)
    Drivers_Required = Column(Integer, default=0)
    Driver_Schedule = Column(Float, default=0)
    Driver_Shortage = Column(Integer, default=0)
    Double_Duty = Column(Integer, default=0)
    Pct_Double_Duty = Column(Float, default=0)
    Off_Cancellation = Column(Integer, default=0)
    Pct_Off_Cancellation = Column(Float, default=0)
    Drivers_as_Conductors = Column(Integer, default=0)
    Drivers_on_Duty = Column(Integer, default=0)
    Driver_for_Bus_Services = Column(Integer, default=0)
    KM_per_Driver = Column(Float, default=0)
    Service_per_Driver_Check = Column(Integer, default=0)
    Spondilitis = Column(Integer, default=0)
    Spinal_Disc = Column(Integer, default=0)
    Vision_Color_Blindness = Column(Integer, default=0)
    Neuro_Paralysis_Medical = Column(Integer, default=0)
    Ortho = Column(Integer, default=0)
    Total_Drivers_MU_Reasons = Column(Integer, default=0)
    Diff_MU_Reasons = Column(Integer, default=0)
    Flu_Fever = Column(Integer, default=0)
    BP = Column(Integer, default=0)
    Orthopedic = Column(Integer, default=0)
    Heart = Column(Integer, default=0)
    Weakness = Column(Integer, default=0)
    Eye = Column(Integer, default=0)
    Accident_Injuries = Column(Integer, default=0)
    Neuro_Paralysis_Sick_Leave = Column(Integer, default=0)
    Piles = Column(Integer, default=0)
    Diabetes = Column(Integer, default=0)
    Thyroid = Column(Integer, default=0)
    Gas = Column(Integer, default=0)
    Dental = Column(Integer, default=0)
    Ear = Column(Integer, default=0)
    Skin_Allergy = Column(Integer, default=0)
    General_Surgery = Column(Integer, default=0)
    Obesity = Column(Integer, default=0)
    Cancer = Column(Integer, default=0)
    Total_Drivers_SL_Reasons = Column(Integer, default=0)
    Diff_SL_Reasons = Column(Integer, default=0)

# ----------------------------------------------------------------------
# 4️⃣ bus_details
# ----------------------------------------------------------------------
class BusDetails(Base):
    __tablename__ = "bus_details"
    __table_args__ = {'extend_existing': True}

    depot = Column(String(50), primary_key=True)
    operations_date = Column(Date, primary_key=True)
    bus_number = Column(String(15), primary_key=True)
    bus_type = Column(String(100))
    comm_date = Column(Text)
    engine_make = Column(String(15))
    eng_kms = Column(Integer)
    progress_km = Column(Integer)

# ----------------------------------------------------------------------
# 5️⃣ daily_operations
# ----------------------------------------------------------------------
class DailyOperations(Base):
    __tablename__ = "daily_operations"
    __table_args__ = {'extend_existing': True}

    depot = Column(String(50), primary_key=True)
    operations_date = Column(Date, primary_key=True)
    employee_id = Column(Integer, primary_key=True)
    vehicle_number = Column(String(15))
    service_type = Column(String(10))
    service_number = Column(String(10))
    opd_kms = Column(Integer)
    daily_earnings = Column(Integer)
    day_night = Column(String(5))
    schedules_count = Column(String(100))
    long_type = Column(String(20))
    route_name = Column(String(100))

# ----------------------------------------------------------------------
# 6️⃣ driver_absenteesim
# ----------------------------------------------------------------------
class DriverAbsenteeism(Base):
    __tablename__ = "driver_absenteeism"
    __table_args__ = {'extend_existing': True}

    employee_id = Column(Integer, primary_key=True)
    depot = Column(Text)
    leave_type = Column(Text)
    date = Column(Text)

# ----------------------------------------------------------------------
# 7️⃣ driver_details
# ----------------------------------------------------------------------
class DriverDetails(Base):
    __tablename__ = "driver_details"
    __table_args__ = {'extend_existing': True}

    employee_id = Column(Integer, primary_key=True)
    full_name = Column(String(50))
    unit = Column(String(50))
    age = Column(Integer)
    birth_date = Column(Date)
    joining_date = Column(Date)
    gender = Column(String(10))
    marital_status = Column(String(20))

# ----------------------------------------------------------------------
# 8️⃣ ghc_2023 and ghc_2024
# ----------------------------------------------------------------------
class GHC2023(Base):
    __tablename__ = "ghc_2023"
    __table_args__ = {'extend_existing': True}

    employee_ID = Column(Integer, primary_key=True)
    depot = Column(String(20))
    final_Grading = Column(String(2))

class GHC2024(Base):
    __tablename__ = "ghc_2024"
    __table_args__ = {'extend_existing': True}

    employee_ID = Column(Integer, primary_key=True)
    depot = Column(String(20))
    final_Grading = Column(String(2))

# ----------------------------------------------------------------------
# 9️⃣ action_plan
# ----------------------------------------------------------------------
class ActionPlan(Base):
    __tablename__ = "action_plan"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    depot_name = Column(String(100), nullable=False)
    data_date = Column(Date, nullable=False)
    Weekly_Off_National_Off = Column(String(100))
    Special_Off_Night_Out_IC_Online = Column(String(100))
    Other_s = Column(String(100))
    Leave_Absent = Column(String(100))
    Sick_Leave = Column(String(100))
    Spot_Absent = Column(String(100))
    Double_Duty = Column(String(100))
    Off_Cancellation = Column(String(100))

# ----------------------------------------------------------------------
# 🔟 service_master
# ----------------------------------------------------------------------
class ServiceMaster(Base):
    __tablename__ = "service_master"
    __table_args__ = {'extend_existing': True}

    depot = Column(String(100), primary_key=True)
    service_number = Column(String(10), primary_key=True)
    route_name = Column(String(20))
    service_type = Column(String(100))
    revenue_kms = Column(Integer)
    day_night_code = Column(String(2))
    oneman_code = Column(String(2))
    private_hire = Column(String(2))
    dept_time = Column(String(10))
    arr_time = Column(String(10))
    Hours = Column(Integer)
