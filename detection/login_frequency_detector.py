import pandas as pd
from pathlib import Path

from detection.baseline_engine import (
    prepare_daily_data,
    build_employee_baseline
)


# =========================================================
# CONFIGURATION
# =========================================================

MODERATE_Z_THRESHOLD = 2.0
HIGH_Z_THRESHOLD = 3.0


# =========================================================
# RESULT GENERATOR
# =========================================================

def create_result(
    employee,
    target_date,
    status,
    severity,
    evidence,
    observed_count=None,
    baseline_mean=None,
    baseline_std=None,
    deviation_score=None,
    baseline_observations=0
):

    return {
        "employee": employee,
        "date": str(pd.Timestamp(target_date).date()),
        "detector": "LOGIN_FREQUENCY",
        "status": status,
        "severity": severity,
        "observed_count": observed_count,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "deviation_score": deviation_score,
        "baseline_observations": baseline_observations,
        "evidence": evidence
    }


# =========================================================
# LOGIN FREQUENCY DETECTOR
# =========================================================

def evaluate_login_frequency(
    daily_data,
    employee,
    target_date,
    baseline=None
):

    target_date = pd.Timestamp(target_date).normalize()

    required_columns = {
        "user",
        "calendar_date",
        "logon_count",
        "has_logon_activity"
    }

    missing_columns = required_columns - set(daily_data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # -----------------------------------------------------
    # 1. CHECK EMPLOYEE
    # -----------------------------------------------------

    if employee not in daily_data["user"].values:

        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            "Employee does not exist in the dataset."
        )

    # -----------------------------------------------------
    # 2. BUILD BASELINE
    # -----------------------------------------------------

    if baseline is None:

        baseline = build_employee_baseline(
            daily_data,
            employee,
            target_date
        )

    frequency = baseline["login_frequency"]

    if not frequency["baseline_available"]:

        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            "Historical login-frequency baseline unavailable.",
            baseline_observations=frequency.get(
                "observation_count", 0
            )
        )

    baseline_mean = float(frequency["mean"])
    baseline_std = float(frequency["std"])

    observations = frequency["observation_count"]

    # -----------------------------------------------------
    # 3. GET CURRENT EMPLOYEE ACTIVITY
    # -----------------------------------------------------

    employee_day = daily_data[
        (daily_data["user"] == employee)
        &
        (daily_data["calendar_date"] == target_date)
    ]

    if employee_day.empty:

        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            "No employee-day record exists for the target date.",
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            baseline_observations=observations
        )

    record = employee_day.iloc[0]

    if not bool(record["has_logon_activity"]):

        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            "No login activity recorded for this employee-day.",
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            baseline_observations=observations
        )

    observed = record["logon_count"]

    if pd.isna(observed) or observed < 0:

        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            "Invalid or missing login count.",
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            baseline_observations=observations
        )

    observed = float(observed)

    # -----------------------------------------------------
    # 4. ZERO VARIANCE HANDLING
    # -----------------------------------------------------

    if baseline_std == 0:

        if observed == baseline_mean:

            return create_result(
                employee,
                target_date,
                "NORMAL",
                "LOW",
                "Login frequency matches the constant historical baseline.",
                observed,
                baseline_mean,
                baseline_std,
                0.0,
                observations
            )

        if observed < baseline_mean:

            return create_result(
                employee,
                target_date,
                "NORMAL",
                "LOW",
                "Login frequency is below the historical mean.",
                observed,
                baseline_mean,
                baseline_std,
                None,
                observations
            )

        return create_result(
            employee,
            target_date,
            "ANOMALOUS",
            "MODERATE",
            "Login frequency exceeds a historically constant baseline. "
            "A conventional Z-score is undefined because standard "
            "deviation is zero.",
            observed,
            baseline_mean,
            baseline_std,
            None,
            observations
        )

    # -----------------------------------------------------
    # 5. CALCULATE Z-SCORE
    # -----------------------------------------------------

    deviation_score = (
        observed - baseline_mean
    ) / baseline_std

    # Only unusually high login frequency is detected.

    if deviation_score < MODERATE_Z_THRESHOLD:

        return create_result(
            employee,
            target_date,
            "NORMAL",
            "LOW",
            "Login frequency does not exceed the configured threshold.",
            observed,
            baseline_mean,
            baseline_std,
            deviation_score,
            observations
        )

    # -----------------------------------------------------
    # 6. CLASSIFY SEVERITY
    # -----------------------------------------------------

    if deviation_score >= HIGH_Z_THRESHOLD:

        severity = "HIGH"

    else:

        severity = "MODERATE"

    evidence = (
        f"Observed login frequency is {observed:.0f}. "
        f"Historical mean is {baseline_mean:.2f}. "
        f"Deviation is {deviation_score:.2f} standard deviations."
    )

    return create_result(
        employee,
        target_date,
        "ANOMALOUS",
        severity,
        evidence,
        observed,
        baseline_mean,
        baseline_std,
        deviation_score,
        observations
    )


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def detect_employee_login_frequency(
    daily_data,
    employee,
    target_date
):

    return evaluate_login_frequency(
        daily_data,
        employee,
        target_date
    )


# =========================================================
# EXECUTION TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 65)
    print("AEGIS - LOGIN FREQUENCY DETECTOR")
    print("=" * 65)

    # -----------------------------------------------------
    # 1. LOAD DATASET
    # -----------------------------------------------------

    project_root = Path(__file__).resolve().parent.parent

    data_file = (
        project_root
        / "data"
        / "daily_activity_features.csv"
    )

    print("\n[1] Loading employee-day dataset...")

    if not data_file.exists():

        raise FileNotFoundError(
            f"Dataset not found: {data_file}"
        )

    raw_data = pd.read_csv(
        data_file,
        low_memory=False
    )

    daily_data = prepare_daily_data(raw_data)

    print(f"Records loaded: {len(daily_data):,}")
    print(f"Employees: {daily_data['user'].nunique():,}")

    # -----------------------------------------------------
    # 2. SELECT VALID EMPLOYEE
    # -----------------------------------------------------

    print("\n[2] Selecting employee with valid baseline...")

    target_date = daily_data["calendar_date"].max()

    target_day_users = daily_data[
        (daily_data["calendar_date"] == target_date)
        &
        (daily_data["has_logon_activity"] == True)
    ]["user"].unique()

    employee = None
    baseline = None

    for candidate in target_day_users:

        candidate_baseline = build_employee_baseline(
            daily_data,
            candidate,
            target_date
        )

        if candidate_baseline[
            "login_frequency"
        ]["baseline_available"]:

            employee = candidate
            baseline = candidate_baseline
            break

    if employee is None:

        raise RuntimeError(
            "No employee with login activity and an available "
            "historical baseline was found."
        )

    print(f"Selected employee: {employee}")
    print(f"Target date: {target_date.date()}")

    # -----------------------------------------------------
    # 3. RUN DETECTOR
    # -----------------------------------------------------

    print("\n[3] Running login frequency detector...")

    result = evaluate_login_frequency(
        daily_data,
        employee,
        target_date,
        baseline
    )

    # -----------------------------------------------------
    # 4. DISPLAY RESULTS
    # -----------------------------------------------------

    print("\n[4] DETECTION RESULT")
    print("-" * 65)

    for key, value in result.items():

        print(f"{key}: {value}")

    print("\n" + "=" * 65)
    print("LOGIN FREQUENCY DETECTOR EXECUTION COMPLETED")
    print("=" * 65)