
import pandas as pd

from detection.baseline_engine import prepare_daily_data


MIN_HISTORY_DAYS = 5
HIGH_SHARED_USERS = 3


def parse_computers(value):
    """Convert a computer list into a set of computer IDs."""

    if pd.isna(value):
        return set()

    return {
        computer.strip()
        for computer in str(value).split("|")
        if computer.strip()
    }


def create_result(
    employee,
    target_date,
    status,
    severity,
    observed_computers=None,
    shared_computers=None,
    new_relationships=None,
    associated_users=None,
    evidence=None
):
    return {
        "employee": employee,
        "date": str(pd.Timestamp(target_date).date()),
        "detector": "CROSS_USER_CORRELATION",
        "status": status,
        "severity": severity,
        "observed_computers": observed_computers or [],
        "shared_computers": shared_computers or [],
        "new_relationships": new_relationships or [],
        "associated_users": associated_users or [],
        "evidence": evidence
    }


def evaluate_cross_user_correlation(
    daily_data,
    employee,
    target_date,
    lookback_days=30
):
    """
    Detect new employee-computer relationships and unusual
    same-day computer sharing.

    Historical relationships exclude the target date and future data.
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

    if lookback_days < 1:
        raise ValueError("lookback_days must be at least 1.")

    employee_data = daily_data[
        daily_data["user"] == employee
    ]

    if employee_data.empty:
        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            evidence="Employee does not exist in the dataset."
        )

    window_start = target_date - pd.Timedelta(days=lookback_days)

    history = daily_data[
        (daily_data["calendar_date"] >= window_start)
        & (daily_data["calendar_date"] < target_date)
        & (daily_data["has_logon_activity"] == True)
    ]

    target_data = daily_data[
        (daily_data["calendar_date"] == target_date)
        & (daily_data["has_logon_activity"] == True)
    ]

    employee_target = target_data[
        target_data["user"] == employee
    ]

    if employee_target.empty:
        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            evidence="No target-day login activity exists."
        )

    observed_computers = set()

    for value in employee_target["computers_used"]:
        observed_computers.update(parse_computers(value))

    if not observed_computers:
        return create_result(
            employee,
            target_date,
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            evidence="Target-day computer telemetry is unavailable."
        )

    # Count historical active days for this employee.
    employee_history = history[
        history["user"] == employee
    ]

    historical_active_days = employee_history[
        "calendar_date"
    ].nunique()

    history_sufficient = (
        historical_active_days >= MIN_HISTORY_DAYS
    )

    # Build historical employee-computer relationships.
    historical_relationships = set()

    for _, row in history.iterrows():

        for computer in parse_computers(row["computers_used"]):
            historical_relationships.add(
                (row["user"], computer)
            )

    # Build target-day computer-user relationships.
    computer_users = {}

    for _, row in target_data.iterrows():

        current_user = row["user"]

        for computer in parse_computers(row["computers_used"]):

            computer_users.setdefault(computer, set()).add(
                current_user
            )

    shared_computers = []
    new_relationships = []
    associated_users = set()

    max_shared_users = 0

    for computer in sorted(observed_computers):

        users_on_computer = computer_users.get(
            computer,
            set()
        )

        other_users = users_on_computer - {employee}

        if not other_users:
            continue

        shared_computers.append(computer)
        associated_users.update(other_users)

        max_shared_users = max(
            max_shared_users,
            len(other_users)
        )

        # Only evaluate relationship novelty with sufficient history.
        if (
            history_sufficient
            and (employee, computer) not in historical_relationships
        ):
            new_relationships.append({
                "computer": computer,
                "other_users": sorted(other_users)
            })

    associated_user_count = len(associated_users)

    # No same-day sharing detected.
    if not shared_computers:

        return create_result(
            employee,
            target_date,
            "NORMAL",
            "LOW",
            observed_computers=sorted(observed_computers),
            evidence="No same-day computer sharing detected."
        )

    # Evaluate sharing per computer.
    unusual_sharing = (
        max_shared_users >= HIGH_SHARED_USERS
    )

    has_new_relationship = bool(new_relationships)

    # High-volume sharing is an independent anomaly signal.
    if unusual_sharing:

        status = "ANOMALOUS"
        severity = "HIGH"

        evidence = (
            f"At least one computer was shared with "
            f"{max_shared_users} or more other employees "
            "on the same day."
        )

        if has_new_relationship:
            evidence += (
                " A new employee-computer relationship was "
                "also detected within the observation window."
            )

    elif has_new_relationship:

        status = "ANOMALOUS"
        severity = "MODERATE"

        evidence = (
            "New employee-computer relationship detected "
            "within the 30-day observation window."
        )

    elif not history_sufficient:

        status = "INSUFFICIENT_DATA"
        severity = "UNKNOWN"

        evidence = (
            f"Same-day computer sharing detected with "
            f"{associated_user_count} other employee(s), but only "
            f"{historical_active_days} historical active day(s) "
            f"are available. At least {MIN_HISTORY_DAYS} are "
            "required to evaluate relationship novelty."
        )

    else:

        status = "NORMAL"
        severity = "LOW"

        evidence = (
            "Known computer sharing detected with "
            f"{associated_user_count} other employee(s). "
            "No new relationship or high-volume sharing signal."
        )

    return create_result(
        employee=employee,
        target_date=target_date,
        status=status,
        severity=severity,
        observed_computers=sorted(observed_computers),
        shared_computers=sorted(shared_computers),
        new_relationships=new_relationships,
        associated_users=sorted(associated_users),
        evidence=evidence
    )


def main():

    print("=" * 65)
    print("AEGIS - CROSS-USER CORRELATION DETECTOR")
    print("=" * 65)

    print("\n[1] Loading employee-day dataset...")

    raw_data = pd.read_csv(
        "data/daily_activity_features.csv",
        low_memory=False
    )

    daily_data = prepare_daily_data(raw_data)

    print(f"Records loaded: {len(daily_data):,}")
    print(f"Employees: {daily_data['user'].nunique():,}")

    target_date = daily_data["calendar_date"].max()

    target_users = daily_data[
        (daily_data["calendar_date"] == target_date)
        & (daily_data["has_logon_activity"] == True)
    ]["user"].unique()

    if len(target_users) == 0:
        print("No employees have target-day login activity.")
        return

    selected_employee = target_users[0]

    print(f"\nSelected employee: {selected_employee}")
    print(f"Target date: {target_date.date()}")

    result = evaluate_cross_user_correlation(
        daily_data,
        selected_employee,
        target_date
    )

    print("\n[2] DETECTION RESULT")
    print("-" * 65)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nCROSS-USER CORRELATION EXECUTION COMPLETED")


if __name__ == "__main__":
    main()
