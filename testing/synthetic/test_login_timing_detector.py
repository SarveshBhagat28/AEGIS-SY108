import pandas as pd

from detection.baseline_engine import (
    prepare_daily_data,
    build_employee_baseline
)

from detection.login_timing_detector import (
    evaluate_login_timing,
    calculate_circular_deviation
)


EMPLOYEE = "TEST001"
TARGET_DATE = "2026-01-07"


def create_dataset(historical_times, target_time):
    """
    Create synthetic employee-day data.

    Historical observations are placed before the target date.
    """

    rows = []

    start_date = pd.Timestamp("2026-01-01")

    for index, login_time in enumerate(historical_times):

        rows.append({
            "user": EMPLOYEE,
            "calendar_date": start_date + pd.Timedelta(days=index),
            "logon_count": 1,
            "first_login_minutes": login_time,
            "computers_used": "PC-01",
            "has_logon_activity": True
        })

    rows.append({
        "user": EMPLOYEE,
        "calendar_date": pd.Timestamp(TARGET_DATE),
        "logon_count": 1 if target_time is not None else 0,
        "first_login_minutes": target_time,
        "computers_used": "PC-01",
        "has_logon_activity": True
    })

    return prepare_daily_data(pd.DataFrame(rows))


def run_test(
    name,
    historical_times,
    target_time,
    expected_status,
    expected_severity=None
):

    daily_data = create_dataset(
        historical_times,
        target_time
    )

    baseline = build_employee_baseline(
        daily_data,
        EMPLOYEE,
        TARGET_DATE
    )

    result = evaluate_login_timing(
        daily_data,
        EMPLOYEE,
        TARGET_DATE,
        baseline=baseline
    )

    passed = result["status"] == expected_status

    if expected_severity is not None:
        passed = passed and result["severity"] == expected_severity

    print(f"\nTest: {name}")
    print(f"Expected status:   {expected_status}")
    print(f"Actual status:     {result['status']}")

    if expected_severity:
        print(f"Expected severity: {expected_severity}")
        print(f"Actual severity:   {result['severity']}")

    print(f"Observed time:     {result['observed_login_minutes']}")
    print(f"Baseline time:     {result['baseline_login_minutes']}")
    print(f"Deviation:         {result['deviation_minutes']}")
    print(f"Resultant length:  {result['resultant_length']}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_midnight_boundary():

    deviation = calculate_circular_deviation(
        1430,  # 23:50
        10     # 00:10
    )

    print("\nTest: Midnight boundary")
    print("23:50 compared with 00:10")
    print("Expected deviation: 20 minutes")
    print(f"Actual deviation:   {deviation} minutes")

    assert deviation == 20

    print("Result: PASS")


def main():

    print("=" * 65)
    print("AEGIS - LOGIN TIMING DETECTOR VALIDATION")
    print("=" * 65)

    tests = [

        (
            "Normal login time",
            [535, 538, 540, 542, 545],
            550,
            "NORMAL",
            "LOW"
        ),

        (
            "Moderate timing deviation",
            [535, 538, 540, 542, 545],
            660,
            "ANOMALOUS",
            "MODERATE"
        ),

        (
            "High timing deviation",
            [535, 538, 540, 542, 545],
            800,
            "ANOMALOUS",
            "HIGH"
        ),

        (
            "Low resultant length",
            [0, 288, 576, 864, 1152],
            600,
            "INSUFFICIENT_DATA",
            "UNKNOWN"
        ),

        (
            "Insufficient historical data",
            [535, 538, 540, 542],
            600,
            "INSUFFICIENT_DATA",
            "UNKNOWN"
        ),

        (
            "Missing login time",
            [535, 538, 540, 542, 545],
            None,
            "INSUFFICIENT_DATA",
            "UNKNOWN"
        ),

        (
            "Invalid login time",
            [535, 538, 540, 542, 545],
            1500,
            "INSUFFICIENT_DATA",
            "UNKNOWN"
        )

    ]

    passed = 0
    failed = 0

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

    # Independent circular-time calculation test
     # Independent circular-time calculation test
    try:
        test_midnight_boundary()
        passed += 1
        print("\nMidnight boundary test: PASS")

    except AssertionError:
        failed += 1
        print("\nMidnight boundary test: FAIL")

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