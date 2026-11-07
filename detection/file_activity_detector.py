
import pandas as pd


MODERATE_REMOVABLE_THRESHOLD = 1
HIGH_REMOVABLE_THRESHOLD = 3

MODERATE_DELETE_THRESHOLD = 10
HIGH_DELETE_THRESHOLD = 25


def create_result(employee, target_date, status, severity, evidence=None):
    return {
        "employee": employee,
        "date": str(pd.Timestamp(target_date).date()),
        "detector": "FILE_ACTIVITY",
        "status": status,
        "severity": severity,
        "evidence": evidence
    }


def evaluate_file_activity(daily_data, employee, target_date):

    required_columns = {
        "user",
        "calendar_date",
        "file_day_covered",
        "removable_media_events",
        "file_deletes"
    }

    missing = required_columns - set(daily_data.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    target_date = pd.Timestamp(target_date).normalize()

    employee_day = daily_data[
        (daily_data["user"] == employee)
        & (daily_data["calendar_date"] == target_date)
    ]

    if employee_day.empty:
        return create_result(
            employee, target_date,
            "INSUFFICIENT_DATA", "UNKNOWN",
            "No employee-day record exists."
        )

    current = employee_day.iloc[0]

    if pd.isna(current["file_day_covered"]) or not bool(current["file_day_covered"]):
        return create_result(
            employee, target_date,
            "INSUFFICIENT_DATA", "UNKNOWN",
            "File activity telemetry is not covered for this day."
        )

    removable = current["removable_media_events"]
    deletes = current["file_deletes"]

    if pd.isna(removable) or pd.isna(deletes):
        return create_result(
            employee, target_date,
            "INSUFFICIENT_DATA", "UNKNOWN",
            "Required file activity metrics are unavailable."
        )

    removable = int(removable)
    deletes = int(deletes)

    findings = []

    severity = "LOW"

    if removable >= HIGH_REMOVABLE_THRESHOLD:
        severity = "HIGH"
        findings.append(
            f"{removable} removable-media file events detected"
        )
    elif removable >= MODERATE_REMOVABLE_THRESHOLD:
        severity = "MODERATE"
        findings.append(
            f"{removable} removable-media file events detected"
        )

    if deletes >= HIGH_DELETE_THRESHOLD:
        severity = "HIGH"
        findings.append(
            f"{deletes} file deletions detected"
        )
    elif deletes >= MODERATE_DELETE_THRESHOLD:
        if severity != "HIGH":
            severity = "MODERATE"
        findings.append(
            f"{deletes} file deletions detected"
        )

    if findings:
        return create_result(
            employee, target_date,
            "ANOMALOUS", severity,
            "; ".join(findings)
        )

    return create_result(
        employee, target_date,
        "NORMAL", "LOW",
        "No configured file activity thresholds were exceeded."
    )
