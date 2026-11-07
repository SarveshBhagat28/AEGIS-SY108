import pandas as pd
import pytest

from detection.baseline_engine import (
    prepare_daily_data,
    build_employee_baseline
)

from detection.login_frequency_detector import evaluate_login_frequency


EMPLOYEE = "TEST001"
TARGET_DATE = "2026-01-07"


def create_dataset(historical_counts, target_count):
    rows = []
    start_date = pd.Timestamp("2026-01-01")

    for index, count in enumerate(historical_counts):
        rows.append({
            "user": EMPLOYEE,
            "calendar_date": start_date + pd.Timedelta(days=index),
            "logon_count": count,
            "first_login_minutes": 540,
            "computers_used": "PC-01",
            "has_logon_activity": True
        })

    rows.append({
        "user": EMPLOYEE,
        "calendar_date": pd.Timestamp(TARGET_DATE),
        "logon_count": target_count,
        "first_login_minutes": 540,
        "computers_used": "PC-01",
        "has_logon_activity": True
    })

    return prepare_daily_data(pd.DataFrame(rows))


@pytest.mark.parametrize(
    "name,historical_counts,target_count,expected_status,expected_severity",
    [
        ("Normal frequency", [1, 1, 1, 1, 2], 1, "NORMAL", "LOW"),
        ("Moderate anomaly", [1, 1, 1, 1, 2], 2, "ANOMALOUS", "MODERATE"),
        ("High anomaly", [1, 1, 1, 1, 2], 3, "ANOMALOUS", "HIGH"),
        ("Zero variance", [1, 1, 1, 1, 1], 2, "ANOMALOUS", "MODERATE"),
        ("Insufficient history", [1, 1, 1, 1], 3, "INSUFFICIENT_DATA", "UNKNOWN"),
    ]
)
def test_login_frequency(
    name,
    historical_counts,
    target_count,
    expected_status,
    expected_severity
):
    daily_data = create_dataset(historical_counts, target_count)

    baseline = build_employee_baseline(
        daily_data,
        EMPLOYEE,
        TARGET_DATE
    )

    result = evaluate_login_frequency(
        daily_data,
        EMPLOYEE,
        TARGET_DATE,
        baseline=baseline
    )

    assert result["status"] == expected_status, name
    assert result["severity"] == expected_severity, name