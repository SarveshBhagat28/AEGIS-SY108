from pathlib import Path
from datetime import date
import logging

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detection.baseline_engine import (
    prepare_daily_data,
    build_employee_baseline
)
from detection.login_frequency_detector import evaluate_login_frequency
from detection.login_timing_detector import evaluate_login_timing
from detection.computer_usage_detector import evaluate_computer_usage
from detection.cross_user_correlation import evaluate_cross_user_correlation
from detection.evidence_aggregator import aggregate_employee_results
from detection.file_activity_detector import evaluate_file_activity
from detection.device_activity_detector import evaluate_device_activity
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FEATURES_FILE = DATA_DIR / "unified_activity_features.csv"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AEGIS")

app = FastAPI(
    title="A.E.G.I.S. API",
    description="AI-assisted Insider Threat Detection System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# LOAD DATA ONCE
# --------------------------------------------------

logger.info("Loading employee activity dataset...")

if not FEATURES_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {FEATURES_FILE}")

raw_features = pd.read_csv(
    FEATURES_FILE,
    dtype={"user": "string"},
    low_memory=False
)

daily_features = prepare_daily_data(raw_features)

logger.info(
    "Dataset loaded and prepared: %s rows, %s employees",
    len(daily_features),
    daily_features["user"].nunique()
)
# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class AnalysisRequest(BaseModel):
    employee: str
    target_date: date


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "A.E.G.I.S.",
        "dataset_loaded": True,
        "employee_count": int(daily_features["user"].nunique()),
        "records": len(daily_features)
    }


# --------------------------------------------------
# EMPLOYEE LIST
# --------------------------------------------------

@app.get("/api/employees")
def get_employees(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    employees = sorted(
        daily_features["user"].dropna().unique().tolist()
    )

    return {
        "total": len(employees),
        "limit": limit,
        "offset": offset,
        "employees": employees[offset:offset + limit]
    }


# --------------------------------------------------
# EMPLOYEE ANALYSIS
# --------------------------------------------------

@app.post("/api/analyze")
def analyze_employee(request: AnalysisRequest):

    employee = request.employee.strip()
    target_date = request.target_date.isoformat()

    if not employee:
        raise HTTPException(
            status_code=400,
            detail="Employee ID cannot be empty."
        )

    if employee not in set(daily_features["user"].dropna()):
        raise HTTPException(
            status_code=404,
            detail="Employee not found."
        )

    try:
        # Build one shared historical baseline.
        baseline = build_employee_baseline(
            daily_features,
            employee,
            target_date
        )

        # Run existing detectors.
        frequency_result = evaluate_login_frequency(
            daily_features,
            employee,
            target_date,
            baseline
        )

        timing_result = evaluate_login_timing(
            daily_features,
            employee,
            target_date,
            baseline
        )

        computer_result = evaluate_computer_usage(
            daily_features,
            employee,
            target_date,
            baseline
        )

        correlation_result = evaluate_cross_user_correlation(
            daily_features,
            employee,
            target_date
        )
        file_result = evaluate_file_activity(
            daily_features,
            employee,
            target_date
        )
        
        device_result = evaluate_device_activity(
            daily_features,
            employee,
            target_date
        )
        detector_results = [
            frequency_result,
            timing_result,
            computer_result,
            correlation_result,
            file_result,
            device_result
        ]
        for result in detector_results:
            logger.info(
                "DETECTOR DEBUG | %s | status=%s | severity=%s",
                result.get("detector"),
                result.get("status"),
                result.get("severity")
    )
        # Aggregate evidence.
        assessment = aggregate_employee_results(
            employee,
            target_date,
            detector_results
        )

        return {
            "system": "A.E.G.I.S.",
            "analysis_type": "Historical Behavioural Analysis",
            "assessment": assessment,
            "detectors": detector_results,
            "baseline": {
                "window_days": baseline["window_days"],
                "window_start": baseline["window_start"],
                "window_end_exclusive": baseline["window_end_exclusive"],
                "historical_employee_days": baseline[
                    "historical_employee_days"
                ]
            }
        }

    except ValueError as error:
        logger.exception("Analysis validation error")
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        logger.exception("Unexpected analysis error")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check the backend terminal logs."
        )


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "A.E.G.I.S. Insider Threat Detection API",
        "documentation": "/docs",
        "health": "/api/health"
    }
