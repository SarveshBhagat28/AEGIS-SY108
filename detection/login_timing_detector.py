import pandas as pd
import numpy as np

from detection.baseline_engine import (
    prepare_daily_data,
    build_employee_baseline
)


# Detection thresholds
MODERATE_DEVIATION_MINUTES = 120
HIGH_DEVIATION_MINUTES = 240

# Circular baseline reliability threshold.
# Broad distributions are not automatically suspicious.
MIN_RESULTANT_LENGTH = 0.70


def create_result(
    employee,
    target_date,
    status,
    severity,
    observed_time=None,
    baseline_time=None,
    deviation_minutes=None,
    resultant_length=None,
    baseline_observations=None,
    observed_last_login=None,
    baseline_last_login=None,
    last_login_deviation_minutes=None,
    last_login_resultant_length=None,
    last_login_baseline_observations=None,
    evidence=None
):
    """Create a standardized login timing detection result."""

    return {
        "employee": employee,
        "date": str(pd.Timestamp(target_date).date()),
        "detector": "LOGIN_TIMING",
        "status": status,
        "severity": severity,
        "observed_login_minutes": observed_time,
        "baseline_login_minutes": baseline_time,
        "deviation_minutes": deviation_minutes,
        "resultant_length": resultant_length,
        "baseline_observations": baseline_observations,
        "observed_last_login_minutes": observed_last_login,
        "baseline_last_login_minutes": baseline_last_login,
        "last_login_deviation_minutes": last_login_deviation_minutes,
        "last_login_resultant_length": last_login_resultant_length,
        "last_login_baseline_observations": last_login_baseline_observations,
        "evidence": evidence
    }
def calculate_circular_deviation(observed_minutes, baseline_minutes):
    """
    Calculate the shortest circular distance between two times.

    This correctly handles midnight boundaries.

    Example:
        23:50 and 00:10 -> 20 minutes
    """

    difference = abs(observed_minutes - baseline_minutes)

    return min(difference, 1440 - difference)


def evaluate_login_timing(
    daily_data,
    employee,
    target_date,
    baseline=None
):
    """
    Evaluate an employee's first login time against their
    historical circular login-time baseline.
    """

    required_columns = {
        "user",
        "calendar_date",
        "first_login_minutes",
        "has_logon_activity"
    }

    missing_columns = required_columns - set(daily_data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    target_date = pd.Timestamp(target_date).normalize()

    if baseline is None:
        baseline = build_employee_baseline(
            daily_data,
            employee,
            target_date
        )

    timing_baseline = baseline.get("login_timing", {})
    last_timing_baseline = baseline.get("last_login_timing", {})

    if not timing_baseline.get("baseline_available", False):
        return create_result(
            employee=employee,
            target_date=target_date,
            status="INSUFFICIENT_DATA",
            severity="UNKNOWN",
            evidence="Historical login timing baseline is unavailable."
        )

    baseline_time = timing_baseline.get("circular_mean_minutes")
    resultant_length = timing_baseline.get("resultant_length")
    baseline_observations = timing_baseline.get("observation_count")
    last_baseline_time = last_timing_baseline.get("circular_mean_minutes")
    last_resultant_length = last_timing_baseline.get("resultant_length")
    last_baseline_observations = last_timing_baseline.get("observation_count")
    if baseline_time is None or pd.isna(baseline_time):
        return create_result(
            employee=employee,
            target_date=target_date,
            status="INSUFFICIENT_DATA",
            severity="UNKNOWN",
            evidence="Historical circular mean login time is unavailable."
        )

    if resultant_length is None or pd.isna(resultant_length):
        return create_result(
            employee=employee,
            target_date=target_date,
            status="INSUFFICIENT_DATA",
            severity="UNKNOWN",
            evidence="Historical timing reliability could not be calculated."
        )

    # A weak circular concentration makes the mean potentially misleading.
    if resultant_length < MIN_RESULTANT_LENGTH:
        return create_result(
            employee=employee,
            target_date=target_date,
            status="INSUFFICIENT_DATA",
            severity="UNKNOWN",
            baseline_time=round(float(baseline_time), 2),
            resultant_length=round(float(resultant_length), 4),
            baseline_observations=baseline_observations,
            evidence=(
                "Historical login times have a broad or inconsistent "
                "circular distribution. Timing comparison was skipped."
            )
        )

    employee_day = daily_data[
        (daily_data["user"] == employee)
        & (daily_data["calendar_date"] == target_date)
    ]

    if employee_day.empty:
        return create_result(
            employee=employee,
            target_date=target_date,
            status="INSUFFICIENT_DATA",
            severity="UNKNOWN",
            evidence="No employee-day record exists for the target date."
        )

    current = employee_day.iloc[0]

    if not bool(current["has_logon_activity"]):
        return create_result(
            employee=employee,
            target_date=target_date,
            status="INSUFFICIENT_DATA",
            severity="UNKNOWN",
            evidence="No login activity recorded for the target date."
        )

    observed_time = current["first_login_minutes"]

    if pd.isna(observed_time):
        return create_result(
            employee=employee,
            target_date=target_date,
            status="INSUFFICIENT_DATA",
            severity="UNKNOWN",
            evidence="First login time is missing."
        )

    observed_time = float(observed_time)

    if not 0 <= observed_time < 1440:
        return create_result(
            employee=employee,
            target_date=target_date,
            status="INSUFFICIENT_DATA",
            severity="UNKNOWN",
            observed_time=observed_time,
            evidence="First login time is outside the valid 0-1439 minute range."
        )

    deviation = calculate_circular_deviation(
        observed_time,
        float(baseline_time)
    )

    # Evaluate last login time when its historical baseline is reliable.
    observed_last_login = current.get("last_login_minutes")
    last_deviation = None

    if (
        last_timing_baseline.get("baseline_available", False)
        and last_baseline_time is not None
        and not pd.isna(last_baseline_time)
        and last_resultant_length is not None
        and not pd.isna(last_resultant_length)
        and last_resultant_length >= MIN_RESULTANT_LENGTH
        and observed_last_login is not None
        and not pd.isna(observed_last_login)
    ):
        observed_last_login = float(observed_last_login)

        if 0 <= observed_last_login < 1440:
            last_deviation = calculate_circular_deviation(
                observed_last_login,
                float(last_baseline_time)
            )
        else:
            observed_last_login = None
    else:
        observed_last_login = None

    # Use the highest severity from the available timing comparisons.
    deviations = [deviation]

    if last_deviation is not None:
        deviations.append(last_deviation)

    maximum_deviation = max(deviations)

    if maximum_deviation >= HIGH_DEVIATION_MINUTES:
        status = "ANOMALOUS"
        severity = "HIGH"
    elif maximum_deviation >= MODERATE_DEVIATION_MINUTES:
        status = "ANOMALOUS"
        severity = "MODERATE"
    else:
        status = "NORMAL"
        severity = "LOW"

    if last_deviation is not None and last_deviation == maximum_deviation:
        timing_component = "Last login"
    else:
        timing_component = "First login"

    if severity == "HIGH":
        evidence = (
            f"{timing_component} time differs significantly from the "
            "employee's historical timing pattern."
        )
    elif severity == "MODERATE":
        evidence = (
            f"{timing_component} time moderately deviates from the "
            "employee's historical timing pattern."
        )
    else:
        if last_deviation is not None:
            evidence = (
                "First and last login times are within the configured "
                "deviation thresholds."
            )
        else:
            evidence = (
                "First login time is within the configured deviation threshold. "
                "Last login comparison was unavailable."
            )

    return create_result(
        employee=employee,
        target_date=target_date,
        status=status,
        severity=severity,
        observed_time=round(observed_time, 2),
        baseline_time=round(float(baseline_time), 2),
        deviation_minutes=round(deviation, 2),
        resultant_length=round(float(resultant_length), 4),
        baseline_observations=baseline_observations,
        observed_last_login=(
            round(observed_last_login, 2)
            if observed_last_login is not None else None
        ),
        baseline_last_login=(
            round(float(last_baseline_time), 2)
            if observed_last_login is not None else None
        ),
        last_login_deviation_minutes=(
            round(last_deviation, 2)
            if last_deviation is not None else None
        ),
        last_login_resultant_length=(
            round(float(last_resultant_length), 4)
            if last_deviation is not None else None
        ),
        last_login_baseline_observations=last_baseline_observations,
        evidence=evidence
    )
def detect_employee_login_timing(
    daily_data,
    employee,
    target_date
):
    """Convenience function for standalone detection."""

    return evaluate_login_timing(
        daily_data,
        employee,
        target_date
    )


def main():

    print("=" * 65)
    print("AEGIS - LOGIN TIMING DETECTOR")
    print("=" * 65)

    file_path = "data/daily_activity_features.csv"

    print("\n[1] Loading employee-day dataset...")

    raw_data = pd.read_csv(file_path, low_memory=False)
    daily_data = prepare_daily_data(raw_data)

    print(f"Records loaded: {len(daily_data):,}")
    print(f"Employees: {daily_data['user'].nunique():,}")

    print("\n[2] Selecting employee with valid timing baseline...")

    target_date = daily_data["calendar_date"].max()

    target_day_users = daily_data[
        (daily_data["calendar_date"] == target_date)
        & (daily_data["has_logon_activity"])
    ]["user"].unique()

    selected_employee = None
    selected_baseline = None

    for employee in target_day_users:

        baseline = build_employee_baseline(
            daily_data,
            employee,
            target_date
        )

        timing = baseline.get("login_timing", {})

        if (
            timing.get("baseline_available", False)
            and timing.get("resultant_length") is not None
            and timing["resultant_length"] >= MIN_RESULTANT_LENGTH
        ):
            selected_employee = employee
            selected_baseline = baseline
            break

    if selected_employee is None:
        print("No employee with a reliable timing baseline was found.")
        return

    print(f"Selected employee: {selected_employee}")
    print(f"Target date: {target_date.date()}")

    print("\n[3] Running login timing detector...")

    result = evaluate_login_timing(
        daily_data,
        selected_employee,
        target_date,
        baseline=selected_baseline
    )

    print("\n[4] DETECTION RESULT")
    print("-" * 65)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 65)
    print("LOGIN TIMING DETECTOR EXECUTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()