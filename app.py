# app.py — Updated: month parsing uses current year, strict markdown LLM prompt,
# results filtered by date before computing averages/charts.

import os
import json
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# RAG retriever (your module) — must provide: search_faiss(query, top_k, depot, date=None)


# Groq client (or your LLM client)
from groq import Groq

# fuzzy matching
from rapidfuzz import fuzz, process


# ---------------------------
# CONFIG
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY environment variable before running the server.")
client = Groq(api_key=GROQ_API_KEY)

PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant"
]

FUZZY_THRESHOLD = 70
METADATA_FILE = "metadata.json"
app = FastAPI(title="Operational Insights AI (KPI-targeted)")

# ---------------------------
# KPI mapping (authoritative)
# ---------------------------
KPI_KEYS = {
    "Weekly Off": "% Weekly Off",
    "Special Off (Night Out/IC, Online)": "% Special Off",
    "Others": "% Others",
    "Long Leave & Absent": "% Leave & Absent",
    "Sick Leave": "% Sick Leave",
    "Spot Absent": "% Spot Absent",
    "Double Duty": "% Double Duty",
    "Off Cancellation": "% Off Cancellation",
    "Drivers/Schedule": "Driver schedule"
}

KPI_DATA_KEYS = {
    "% Weekly Off": "% Weekly Off",
    "% Special Off": "% Special Off",
    "% Others": "% Others",
    "% Leave & Absent": "% Leave & Absent",
    "% Sick Leave": "% Sick Leave",
    "% Spot Absent": "% Spot Absent",

    # 🔥 CRITICAL FIX
    "% Double Duty": "Pct_Double_Duty",
    "% Off Cancellation": "Pct_Off_Cancellation",

    "Driver schedule": "Driver schedule"
}

# ---------------------------
# BENCHMARKS (as provided)
# ---------------------------
BENCHMARKS = {
    "RURAL": {
        "% Weekly Off": 14,
        "% Special Off": 25,
        "% Others": 1.7,
        "% Leave & Absent": 2,
        "% Sick Leave": 2,
        "% Spot Absent": 2,
        "% Double Duty": 16,
        "% Off Cancellation": 2,
        "Driver schedule": 2.18
    },
    "URBAN": {
        "% Weekly Off": 14,
        "% Special Off": 27.4,
        "% Others": 1,
        "% Leave & Absent": 6,
        "% Sick Leave": 2,
        "% Spot Absent": 2,
        "% Double Duty": 8,
        "% Off Cancellation": 2,
        "Driver schedule": 2.43
    }
}


# ---------------------------
# Load metadata (for depot list)
# ---------------------------
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata_store = json.load(f)["docs"]

DEPOTS = sorted(list({doc["depot"] for doc in metadata_store}))


# ---------------------------
# Request model
# ---------------------------
class AskKPIRequest(BaseModel):
    depot: str
    kpi: str               # Friendly KPI name
    from_date: Optional[str] = None
    to_date: Optional[str] = None


# ---------------------------
# Utilities
# ---------------------------
def fuzzy_find_depot(user_input: Optional[str]) -> Optional[str]:
    if not user_input:
        return None
    ui = user_input.strip().upper()
    for d in DEPOTS:
        if d.upper() == ui:
            return d
    for d in DEPOTS:
        if ui in d.upper() or d.upper() in ui:
            return d
    match = process.extractOne(ui, DEPOTS, scorer=fuzz.token_set_ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        return match[0]
    return None

def filter_results_by_date_range(results: List[dict], from_date_iso: Optional[str], to_date_iso: Optional[str]) -> List[dict]:
    """Return subset of results where date ∈ [from_date, to_date]. If from_date_iso/to_date_iso are None, return original results."""
    if not from_date_iso or not to_date_iso:
        return results
    fd = datetime.fromisoformat(from_date_iso).date()
    td = datetime.fromisoformat(to_date_iso).date()
    filtered: List[dict] = []
    for r in results:
        ds = r.get("date")
        if not ds:
            continue
        try:
            d = datetime.fromisoformat(ds).date()
        except Exception:
            continue
        if fd <= d <= td:
            filtered.append(r)
    return filtered


def resolve_kpi_value(doc: dict, kpi_key: str) -> Optional[float]:
    try:
        if doc.get("calculated_fields") and kpi_key in doc["calculated_fields"]:
            v = doc["calculated_fields"][kpi_key]
            if v is None:
                return None
            return float(v)
    except Exception:
        pass
    try:
        if doc.get("raw_fields") and kpi_key in doc["raw_fields"]:
            v = doc["raw_fields"][kpi_key]
            if v is None:
                return None
            return float(v)
    except Exception:
        pass
    return None

def get_doc_depot(doc: dict) -> Optional[str]:
    """
    Return depot name from a FAISS document, checking top-level, raw_fields, and calculated_fields.
    """
    if "depot" in doc and doc["depot"]:
        return doc["depot"]
    if doc.get("raw_fields") and doc["raw_fields"].get("depot"):
        return doc["raw_fields"]["depot"]
    if doc.get("calculated_fields") and doc["calculated_fields"].get("depot"):
        return doc["calculated_fields"]["depot"]
    return None

def classify_status(avg: Optional[float], benchmark: Optional[float]) -> str:
        if avg is None or benchmark is None:
            return "UNKNOWN"
        if avg < benchmark:
            return "CONTROL"
        if avg > benchmark:
            return "RISK"
        return "WITHIN"

def extract_ts_admin(results: List[dict], depot_name: Optional[str]) -> Optional[dict]:
    # Look for ts_admin doc in results first
    for r in results:
        if r.get("type", "").lower() == "ts_admin" or str(r.get("doc_id", "")).lower().startswith("ts_admin_"):
            return r
    # fallback: search retriever for ts_admin doc
    if depot_name:
        ts_docs = get_docs_from_db(depot_name)
        for c in ts_docs:
            if c.get("type", "").lower() == "ts_admin" or str(c.get("doc_id", "")).lower().startswith("ts_admin_"):
                return c
    return None
# ---------------------------
# ONE-SHOT EXAMPLE (AUTHORITATIVE BEHAVIOR)
# ---------------------------
KPI_INTERPRETATIONS = {
    "Weekly Off": """
Meaning:
Percentage of drivers on weekly or national off.

Interpretation:
- LOWER value → Higher driver availability due to tighter weekly-off control.
- HIGHER value → Reduced driver availability impacting service coverage.
- Sustained LOW → Fatigue accumulation risk if not balanced.
- Sustained HIGH → Planning inefficiency or under-utilization.

Benchmark logic:
- BELOW benchmark = CONTROL
- ABOVE benchmark = AVAILABILITY RISK
""",

    "Special Off (Night Out/IC, Online)": """
Meaning:
Drivers assigned to special night-out, IC, or online duties.

Interpretation:
- LOW value → Stable roster with minimal disruption.
- HIGH value → Dependency on special duties.
- Sustained HIGH → Roster imbalance and night-duty concentration risk.
""",

    "Others": """
Meaning:
Drivers diverted to non-driving duties or training/PME.

Interpretation:
- LOW value → Maximum utilization for core services.
- HIGH value → Reduced service-ready strength.
- Sustained HIGH → Structural manpower planning issue.
""",

    "Leave & Absent": """
Meaning:
Planned and unplanned leave excluding sick leave.

Interpretation:
- LOW value → Good attendance discipline.
- HIGH value → Attendance instability.
- Sustained HIGH → Policy or morale issue.
""",

    "Sick Leave": """
Meaning:
Drivers unavailable due to health reasons.

Interpretation:
- LOW value → Healthy workforce.
- HIGH value → Health stress or fatigue signals.
- Sustained HIGH → Medical risk cluster or workload imbalance.

Important:
- Never interpret as discipline failure.
""",

    "Spot Absent": """
Meaning:
Drivers absent without prior notice.

Interpretation:
- LOW value → Operational stability.
- HIGH value → Daily service disruption risk.
- Sustained HIGH → Weak supervision or morale breakdown.
""",

    "Double Duty": """
Meaning:
Drivers working extra duties to cover shortages.

Interpretation:
- LOW value → Adequate staffing.
- HIGH value → Short-term shortage mitigation.
- Sustained HIGH → Chronic manpower gap and fatigue risk.
""",

    "Off Cancellation": """
Meaning:
Drivers cancel weekly off to meet service demand.

Interpretation:
- LOW value → Balanced planning.
- HIGH value → Reactive service recovery.
- Sustained HIGH → Burnout and poor leave planning.
""",

    "Drivers/Schedule": """
Meaning:
Total Drivers / Planned Schedules.

Interpretation:
- LOW value → Driver shortage pressure.
- OPTIMAL range → Balanced deployment.
- HIGH value → Overstaffing or inefficient scheduling.
"""
}

def build_single_kpi_prompt(
    depot: str,
    kpi: str,
    avg: Optional[float],        # still computed, not shown
    benchmark: Optional[float],  # still computed, not shown
    category: str,
    date_context: str,
    status: str
) -> str:

    interpretation = KPI_INTERPRETATIONS.get(
        kpi,
        "No interpretation available for this KPI."
    )

    return f"""
IMPORTANT DOMAIN RULES (MUST FOLLOW):
- This system analyzes TSRTC public transport depot operations.
- "{kpi}" is strictly a DRIVER OPERATIONS KPI.
- Use ONLY the KPI interpretation provided below.
- NEVER use concepts related to electricity, energy, power, utilities, or consumption.
- Focus ONLY on:
  driver availability,
  roster discipline,
  service continuity,
  fatigue risk,
  depot productivity.
- Recommendations must align with depot category:
  • RURAL → crew pooling, relief availability, route clubbing
  • URBAN → shift balancing, peak-hour control, night-duty rotation

KPI INTERPRETATION (AUTHORITATIVE):
{interpretation}

OPERATIONAL CONTEXT:
- CONTROL → sustain discipline and prevent fatigue buildup
- WITHIN → fine-tuning and preventive balance
- RISK → corrective intervention and service protection

### {kpi} – Operational Insight for {depot}

Category: {category}
{date_context}

Operational Status (System Evaluated): {status}

INSTRUCTIONS:
Produce STRICT Markdown with the following section only:

#### Actionable Recommendations

Formatting rules based on Recommendation Mode:
- SHORT → each point must be ONE concise operational sentence
- DETAILED → each point must include cause + action + operational outcome

Additional Rules:
- Do NOT mention numbers, averages, or benchmarks
- Do NOT restate status differently
- Avoid generic phrases (e.g., “improve efficiency” without mechanism)
- Keep recommendations depot-actionable
"""


def get_depot_category_from_ts(ts_doc: dict) -> Optional[str]:
    if not ts_doc:
        return None
    raw = ts_doc.get("raw_fields", {}) or {}
    for k in ("category", "Category"):
        if k in raw and raw[k]:
            return str(raw[k]).strip().upper()
    calc = ts_doc.get("calculated_fields", {}) or {}
    if "category" in calc and calc["category"]:
        return str(calc["category"]).strip().upper()
    text = ts_doc.get("text") or ts_doc.get("content") or ""
    if isinstance(text, str) and "URBAN" in text.upper():
        return "URBAN"
    if isinstance(text, str) and "RURAL" in text.upper():
        return "RURAL"
    return None


# ---------------------------
# LLM call (Groq) with fallback
# ---------------------------
def ask_llm(prompt: str) -> str:
    last_exc = None
    for model in PREFERRED_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are Operational Insights AI — an expert transport operations analyst. Use only the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_exc = e
            continue
    raise last_exc

# ---------------------------
# DATABASE QUERY FUNCTION
# ---------------------------
def get_docs_from_db(depot: str, from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[dict]:
    """
    Retrieve documents directly from your DB instead of FAISS.
    - depot: filter docs by depot
    - from_date, to_date: optional ISO format strings to filter by date
    Returns: list of dicts
    """
    # Example using JSON file (replace with your real DB query)
    with open("metadata.json", "r", encoding="utf-8") as f:
        all_docs = json.load(f)["docs"]

    # Filter by depot
    docs = [d for d in all_docs if d.get("depot") and depot.upper() in d.get("depot").upper()]

    # Filter by date if provided
    if from_date and to_date:
        docs = filter_results_by_date_range(docs, from_date, to_date)

    return docs
# ---------------------------
# /ask endpoint
# ---------------------------
@app.post("/ask_kpi")
def ask_kpi(req: AskKPIRequest):
    # 1. Normalize depot
    depot = fuzzy_find_depot(req.depot)
    if not depot:
        return {"error": "Invalid depot"}
    # 2. KPI validation
    if req.kpi not in KPI_KEYS:
        return {"error": f"Unsupported KPI: {req.kpi}"}
    
    # NEW: get docs from database instead of FAISS
    docs = get_docs_from_db(depot, req.from_date, req.to_date)

    if req.from_date and req.to_date:
        docs = filter_results_by_date_range(
            docs, req.from_date, req.to_date
        )

    if not docs:
        return {"summary": "No sufficient data available for AI insight."}

    # 5. Extract depot category
    ts_doc = extract_ts_admin(docs, depot)
    category = get_depot_category_from_ts(ts_doc)
    if category is None:
        category = "RURAL"  # safer default for TSRTC


    # 6. Compute KPI average
    friendly = req.kpi
    kpi_label = KPI_KEYS[friendly]          # benchmark label
    data_key = KPI_DATA_KEYS[kpi_label]     # actual stored field

    vals = [
        resolve_kpi_value(d, data_key)
        for d in docs
        if resolve_kpi_value(d, data_key) is not None
    ]


    avg = round(sum(vals) / len(vals), 2) if vals else None

    # 7. Benchmark
    benchmark = BENCHMARKS.get(category, {}).get(kpi_label)

    # 8. Pre-classify KPI status (CONTROL / RISK / WITHIN)
    status = classify_status(avg, benchmark)


    # 8. Date context
    date_context = ""
    if req.from_date and req.to_date:
        fd = req.from_date
        td = req.to_date
        date_context = f"Date Range Used: {fd} → {td}"

    # 9. Build prompt
    prompt = build_single_kpi_prompt(
        depot=depot,
        kpi=friendly,
        avg=avg,
        benchmark=benchmark,
        category=category,
        date_context=date_context,
        status=status
    )

    # 10. LLM call
    try:
        answer = ask_llm(prompt)
    except Exception as e:
        return {"error": "LLM failed", "details": str(e)}
    return {
        "depot": depot,
        "kpi": friendly,
        "average": avg,
        "benchmark": benchmark,
        "category": category,
        "summary": answer
    }
    

    


# ---------------------------
# RUN SERVER
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
