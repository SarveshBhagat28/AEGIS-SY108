
import pandas as pd


MODERATE_CONNECT_THRESHOLD = 2
HIGH_CONNECT_THRESHOLD = 5


def create_result(employee, target_date, status, severity, evidence=None):
    return {
        "employee": employee,
        "date": str(pd.Timestamp(target_date).date()),
        "detector": "DEVICE_ACTIVITY",
        "status": status,
        "severity": severity,
        "evidence": evidence
    }


def evaluate_device_activity(daily_data, employee, target_date):

    required_columns = {
        "user",
        "calendar_date",
        "device_day_covered",
        "device_connects",
        "device_disconnects"
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

    if pd.isna(current["device_day_covered"]) or not bool(current["device_day_covered"]):
        return create_result(
            employee, target_date,
            "INSUFFICIENT_DATA", "UNKNOWN",
            "Device activity telemetry is not covered for this day."
        )

    connects = current["device_connects"]
    disconnects = current["device_disconnects"]

    if pd.isna(connects) or pd.isna(disconnects):
        return create_result(
            employee, target_date,
            "INSUFFICIENT_DATA", "UNKNOWN",
            "Required device activity metrics are unavailable."
        )

    connects = int(connects)
    disconnects = int(disconnects)

    if connects >= HIGH_CONNECT_THRESHOLD:
        return create_result(
            employee, target_date,
            "ANOMALOUS", "HIGH",
            f"{connects} device connections and {disconnects} disconnections detected."
        )

    if connects >= MODERATE_CONNECT_THRESHOLD:
        return create_result(
            employee, target_date,
            "ANOMALOUS", "MODERATE",
            f"{connects} device connections and {disconnects} disconnections detected."
        )

    return create_result(
        employee, target_date,
        "NORMAL", "LOW",
        f"{connects} device connections and {disconnects} disconnections recorded."
    )
