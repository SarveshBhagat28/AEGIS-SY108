import pandas as pd

from detection.baseline_engine import (
    prepare_daily_data,
    build_employee_baseline
)


# Initial engineering thresholds
MODERATE_USAGE_THRESHOLD = 1
HIGH_USAGE_THRESHOLD = 2


def create_result(
    employee,
    target_date,
    status,
    severity,
    observed_computers=None,
    unseen_computers=None,
    occasional_computers=None,
    baseline_observations=None,
    evidence=None
):
    """Create a standardized computer usage detection result."""

    return {
        "employee": employee,
        "date": str(pd.Timestamp(target_date).date()),
        "detector": "COMPUTER_USAGE",
        "status": status,
        "severity": severity,
        "observed_computers": observed_computers,
        "unseen_computers": unseen_computers,
        "occasional_computers": occasional_computers,
        "baseline_observations": baseline_observations,
        "evidence": evidence
    }


def parse_computers(value):
    """Convert the computers_used field into a clean set."""

    if pd.isna(value):
        return set()

    if not isinstance(value, str):
        value = str(value)

    return {
        computer.strip()
        for computer in value.split("|")
        if computer.strip()
    }


def evaluate_computer_usage(
    daily_data,
    employee,
    target_date,
    baseline=None
):
    """
    Compare target-day computer usage against historical behaviour.

    Important:
    This detector uses computer identifiers from logon telemetry.
    It does not independently verify physical device activity.
    """

    required_columns = {
        "user",
        "calendar_date",
        "computers_used",
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

    computer_baseline = baseline.get("computer_usage", {})

    if not computer_baseline.get("baseline_available", False):
        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            evidence="Historical computer usage baseline is unavailable."
        )

    baseline_observations = computer_baseline.get(
        "observation_count"
    )

    if baseline_observations is None:
        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            evidence="Computer baseline observation count is unavailable."
        )

    known_computers = set(
        computer_baseline.get("known_computers", [])
    )

    occasional_computers = set(
        computer_baseline.get("occasional_computers", [])
    )

    employee_day = daily_data[
        (daily_data["user"] == employee)
        & (daily_data["calendar_date"] == target_date)
    ]

    if employee_day.empty:
        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            baseline_observations=baseline_observations,
            evidence="No employee-day record exists for the target date."
        )

    current = employee_day.iloc[0]

    if not bool(current["has_logon_activity"]):
        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            baseline_observations=baseline_observations,
            evidence="No login activity recorded for the target date."
        )

    observed_computers = parse_computers(
        current["computers_used"]
    )

    if not observed_computers:
        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            baseline_observations=baseline_observations,
            evidence="Computer usage information is unavailable."
        )

    # Computers never observed in historical data.
    unseen_computers = observed_computers - known_computers

    # Previously observed computers classified as occasional.
    occasional_used = observed_computers & occasional_computers

    # Determine detection result.
    if unseen_computers:

        if len(unseen_computers) >= HIGH_USAGE_THRESHOLD:
            severity = "HIGH"
        else:
            severity = "MODERATE"

        status = "ANOMALOUS"

        evidence = (
            "Previously unseen computer usage detected: "
            + ", ".join(sorted(unseen_computers))
        )

    elif len(occasional_used) >= HIGH_USAGE_THRESHOLD:

        status = "ANOMALOUS"
        severity = "MODERATE"

        evidence = (
            "Multiple historically occasional computers were used: "
            + ", ".join(sorted(occasional_used))
        )

    elif occasional_used:

        status = "ANOMALOUS"
        severity = "MODERATE"

        evidence = (
            "Historically occasional computer usage detected: "
            + ", ".join(sorted(occasional_used))
        )

    else:

        status = "NORMAL"
        severity = "LOW"

        evidence = (
            "All observed computers belong to the employee's "
            "established computer usage baseline."
        )

    return create_result(
        employee=employee,
        target_date=target_date,
        status=status,
        severity=severity,
        observed_computers=sorted(observed_computers),
        unseen_computers=sorted(unseen_computers),
        occasional_computers=sorted(occasional_used),
        baseline_observations=baseline_observations,
        evidence=evidence
    )


def detect_employee_computer_usage(
    daily_data,
    employee,
    target_date
):
    """Convenience function for standalone detection."""

    return evaluate_computer_usage(
        daily_data,
        employee,
        target_date
    )


def main():

    print("=" * 65)
    print("AEGIS - COMPUTER USAGE DETECTOR")
    print("=" * 65)

    file_path = "data/daily_activity_features.csv"

    print("\n[1] Loading employee-day dataset...")

    raw_data = pd.read_csv(file_path, low_memory=False)

    daily_data = prepare_daily_data(raw_data)

    print(f"Records loaded: {len(daily_data):,}")
    print(f"Employees: {daily_data['user'].nunique():,}")

    print("\n[2] Selecting employee with valid computer baseline...")

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

        computer = baseline.get("computer_usage", {})

        if computer.get("baseline_available", False):
            selected_employee = employee
            selected_baseline = baseline
            break

    if selected_employee is None:
        print("No employee with a valid computer baseline was found.")
        return

    print(f"Selected employee: {selected_employee}")
    print(f"Target date: {target_date.date()}")

    print("\n[3] Running computer usage detector...")

    result = evaluate_computer_usage(
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
    print("COMPUTER USAGE DETECTOR EXECUTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
