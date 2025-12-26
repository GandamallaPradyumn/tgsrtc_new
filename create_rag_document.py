import pymysql
import json
import datetime
from datetime import datetime as dt

# -------------------------------------------
# 1. CONNECT TO MYSQL
# -------------------------------------------
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="tgsrtc_new",
)

cursor = connection.cursor(pymysql.cursors.DictCursor)

cursor.execute("SELECT * FROM input_data")
rows = cursor.fetchall()


# -------------------------------------------
# 2. SAFE GET FUNCTION
# -------------------------------------------
def get(row, key):
    return float(row.get(key, 0) or 0)


# -------------------------------------------
# 3. PROCESS EACH ROW INTO A RAG DOCUMENT
# -------------------------------------------
rag_documents = []

for row in rows:

    # Convert date to string
    date = row["data_date"].strftime("%Y-%m-%d")
    depot = row["depot_name"]

    # ---------------------------------------
    # CLEAN RAW FIELDS (convert date objects)
    # ---------------------------------------
    raw_fields_clean = {
        k: (v.strftime("%Y-%m-%d") if isinstance(v, (datetime.date, datetime.datetime)) else v)
        for k, v in row.items()
    }

    # ---------------------------------------
    # CALCULATIONS (Your formulas)
    # ---------------------------------------

    # Variances
    service_variance = get(row, "Actual_Services") - get(row, "Planned_Services")
    km_variance = get(row, "Actual_KM") - get(row, "Planned_KM")

    # Driver availability
    available1 = get(row, "Total_Drivers") - get(row, "Medically_Unfit") - get(row, "Suspended_Drivers")
    pct_available1 = (available1 / get(row, "Total_Drivers")) * 100 if get(row, "Total_Drivers") else 0

    deductions = (
        get(row, "Weekly_Off_National_Off") +
        get(row, "Special_Off_Night_Out_IC_Online") +
        get(row, "Training_PME_medical") +
        get(row, "Others") +
        get(row, "Leave_Absent") +
        get(row, "Sick_Leave")
    )

    available2 = available1 - deductions
    pct_available2 = (available2 / get(row, "Total_Drivers")) * 100 if get(row, "Total_Drivers") else 0

    # Percent calculations
    total = get(row, "Total_Drivers")
    pct_weekly_off = (get(row, "Weekly_Off_National_Off") / total) * 100 if total else 0
    pct_special_off = (get(row, "Special_Off_Night_Out_IC_Online") / total) * 100 if total else 0
    pct_others = ((get(row, "Training_PME_medical") + get(row, "Others")) / total) * 100 if total else 0
    pct_leave = (get(row, "Leave_Absent") / total) * 100 if total else 0
    pct_sick = (get(row, "Sick_Leave") / total) * 100 if total else 0

    # Attending drivers
    attending = available2 - get(row, "Spot_Absent")
    pct_attending = (attending / total) * 100 if total else 0
    pct_spot_absent = (get(row, "Spot_Absent") / total) * 100 if total else 0

    # Driver shortage
    drivers_required = get(row, "Drivers_Required")
    driver_shortage = max(drivers_required - attending, 0)

    # Driver schedule
    planned_schedules = get(row, "Planned_Schedules")
    driver_schedule = drivers_required / planned_schedules if planned_schedules else 0

    # Drivers on duty
    drivers_on_duty = attending + get(row, "Double_Duty") + get(row, "Off_Cancellation")

    # Bus service drivers
    drivers_bus = drivers_on_duty - get(row, "Drivers_as_Conductors")

    # Productivity
    km_per_driver = (get(row, "Actual_KM") / drivers_bus) if drivers_bus > 0 else 0
    service_driver_check = drivers_bus - get(row, "Actual_Services")

    # MU totals
    mu_total = (
        get(row, "Spondilitis") +
        get(row, "Spinal_Disc") +
        get(row, "Vision_Color_Blindness") +
        get(row, "Neuro_Paralysis_Medical") +
        get(row, "Ortho")
    )
    mu_diff = mu_total - get(row, "Medically_Unfit")

    # SL totals
    sl_fields = [
        "Flu_Fever", "BP", "Orthopedic", "Heart", "Weakness", "Eye",
        "Accident_Injuries", "Neuro_Paralysis_Sick_Leave", "Piles",
        "Diabetes", "Thyroid", "Gas", "Dental", "Ear", "Skin_Allergy",
        "General_Surgery", "Obesity", "Cancer"
    ]
    sl_total = sum(get(row, field) for field in sl_fields)
    sl_diff = sl_total - get(row, "Sick_Leave")

    # ---------------------------------------
    # BUILD RAG DOCUMENT
    # ---------------------------------------
    rag_doc = {
        "doc_id": f"input_data_{depot}_{date}",
        "depot": depot,
        "date": date,

        "raw_fields": raw_fields_clean,

        "calculated_fields": {
            "Service Variance": service_variance,
            "KM Variance": km_variance,
            "Available Drivers-1": available1,
            "% Available Drivers-1": pct_available1,
            "Available Drivers-2": available2,
            "% Available Drivers-2": pct_available2,
            "% Weekly Off": pct_weekly_off,
            "% Special Off": pct_special_off,
            "% Others": pct_others,
            "% Leave & Absent": pct_leave,
            "% Sick Leave": pct_sick,
            "Attending Drivers": attending,
            "% Attending Drivers": pct_attending,
            "% Spot Absent": pct_spot_absent,
            "Driver shortage": driver_shortage,
            "Driver schedule": driver_schedule,
            "Drivers on Duty": drivers_on_duty,
            "Driver for Bus Services": drivers_bus,
            "KM/Driver": km_per_driver,
            "Service/Driver Check": service_driver_check,
            "Total Drivers (MU Reasons)": mu_total,
            "Diff (MU Reasons)": mu_diff,
            "Total Drivers (SL Reasons)": sl_total,
            "Diff (SL Reasons)": sl_diff
        },

        "text": (
            f"On {date}, depot {depot} had {row['Total_Drivers']} drivers. "
            f"Available Drivers-1 was {available1} and Available Drivers-2 was {available2}. "
            f"Attending Drivers were {attending}. Driver shortage was {driver_shortage}. "
            f"Actual KM was {row['Actual_KM']} giving KM per driver {km_per_driver:.2f}. "
            f"Service variance was {service_variance}."
        )
    }

    rag_documents.append(rag_doc)


# -------------------------------------------
# 4. SAVE DOCUMENTS AS JSONL
# -------------------------------------------
output_path = "/tmp/rag_documents.jsonl"

with open(output_path, "w") as f:
    for doc in rag_documents:
        f.write(json.dumps(doc) + "\n")

print(f"RAG documents created successfully at {output_path}")
