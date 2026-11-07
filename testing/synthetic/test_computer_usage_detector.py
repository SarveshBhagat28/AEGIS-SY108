import pandas as pd

from detection.baseline_engine import (
    prepare_daily_data,
    build_employee_baseline
)

from detection.computer_usage_detector import (
    evaluate_computer_usage
)


EMPLOYEE = "TEST001"
TARGET_DATE = "2026-01-07"


def create_dataset(historical_computers, target_computers):
    """Create synthetic employee-day computer usage data."""

    rows = []

    start_date = pd.Timestamp("2026-01-01")

    for index, computers in enumerate(historical_computers):

        rows.append({
            "user": EMPLOYEE,
            "calendar_date": start_date + pd.Timedelta(days=index),
            "logon_count": 1,
            "first_login_minutes": 540,
            "computers_used": computers,
            "has_logon_activity": True
        })

    rows.append({
        "user": EMPLOYEE,
        "calendar_date": pd.Timestamp(TARGET_DATE),
        "logon_count": 1 if target_computers is not None else 0,
        "first_login_minutes": 540,
        "computers_used": target_computers,
        "has_logon_activity": True
    })

    return prepare_daily_data(pd.DataFrame(rows))


def run_test(
    name,
    historical_computers,
    target_computers,
    expected_status,
    expected_severity=None
):

    daily_data = create_dataset(
        historical_computers,
        target_computers
    )

    baseline = build_employee_baseline(
        daily_data,
        EMPLOYEE,
        TARGET_DATE
    )

    result = evaluate_computer_usage(
        daily_data,
        EMPLOYEE,
        TARGET_DATE,
        baseline=baseline
    )

    passed = result["status"] == expected_status

    if expected_severity is not None:
        passed = (
            passed
            and result["severity"] == expected_severity
        )

    print(f"\nTest: {name}")
    print(f"Expected status:   {expected_status}")
    print(f"Actual status:     {result['status']}")

    if expected_severity:
        print(f"Expected severity: {expected_severity}")
        print(f"Actual severity:   {result['severity']}")

    print(f"Observed computers: {result['observed_computers']}")
    print(f"Unseen computers:   {result['unseen_computers']}")
    print(f"Occasional:         {result['occasional_computers']}")
    print(f"Baseline observations: {result['baseline_observations']}")
    print(f"Evidence: {result['evidence']}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_missing_employee_day():

    daily_data = create_dataset(
        ["PC-01"] * 5,
        "PC-01"
    )

    # Remove the target-day record.
    daily_data = daily_data[
        daily_data["calendar_date"] != pd.Timestamp(TARGET_DATE)
    ]

    baseline = build_employee_baseline(
        daily_data,
        EMPLOYEE,
        TARGET_DATE
    )

    result = evaluate_computer_usage(
        daily_data,
        EMPLOYEE,
        TARGET_DATE,
        baseline=baseline
    )

    assert result["status"] == "INSUFFICIENT_DATA"

    


def main():

    print("=" * 65)
    print("AEGIS - COMPUTER USAGE DETECTOR VALIDATION")
    print("=" * 65)

    tests = [
        (
            "Normal computer usage",
            ["PC-01"] * 5,
            "PC-01",
            "NORMAL",
            "LOW"
        ),
        (
            "Previously unseen computer",
            ["PC-01"] * 5,
            "PC-02",
            "ANOMALOUS",
            "MODERATE"
        ),
        (
            "Occasional computer usage",
            ["PC-01", "PC-01", "PC-01", "PC-01", "PC-02"],
            "PC-02",
            "ANOMALOUS",
            "MODERATE"
        ),
        (
            "Multiple unseen computers",
            ["PC-01"] * 5,
            "PC-02|PC-03",
            "ANOMALOUS",
            "HIGH"
        ),
        (
            "Insufficient historical data",
            ["PC-01"] * 4,
            "PC-01",
            "INSUFFICIENT_DATA",
            "UNKNOWN"
        ),
        (
            "Missing computer telemetry",
            ["PC-01"] * 5,
            None,
            "INSUFFICIENT_DATA",
            "UNKNOWN"
        )
    ]

    passed = 0
    failed = 0

    # Run the six detector scenarios.
    for test in tests:
        try:
            result = run_test(*test)

            if result:
                passed += 1
            else:
                failed += 1

        except Exception as error:
            failed += 1
            print(f"\nTest: {test[0]}")
            print(f"ERROR: {error}")
            print("Result: FAIL")

    # Run the missing employee-day test separately.
    try:
        test_missing_employee_day()
        passed += 1
        print("Missing employee-day test: PASS")

    except Exception as error:
        failed += 1
        print(f"Missing employee-day test: FAIL")
        print(f"ERROR: {error}")

    # Display summary after all tests.
    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Total tests: {passed + failed}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {failed}")

    if failed == 0:
        print("\nALL TESTS PASSED")
    else:
        print("\nVALIDATION FAILED")


if __name__ == "__main__":
    main()