import pandas as pd

from detection.baseline_engine import build_employee_baseline


# =========================================================
# TEST DATA GENERATOR
# =========================================================

def create_record(
    user,
    date,
    logon_count=1,
    login_minutes=540,
    computers="PC-100",
    has_logon=True
):
    return {
        "user": user,
        "calendar_date": date,
        "logon_count": logon_count,
        "first_login_minutes": (
            login_minutes if has_logon else None
        ),
        "computers_used": computers,
        "has_logon_activity": has_logon
    }


def build_baseline(df, user, target_date, min_observations=5):
    return build_employee_baseline(
        df,
        user=user,
        target_date=target_date,
        min_observations=min_observations
    )


# =========================================================
# TEST 1: CONSISTENT COMPUTER USAGE
# =========================================================

def test_consistent_computer_usage():

    records = [
        create_record("EMP001", f"2026-01-{day:02d}")
        for day in range(1, 11)
    ]

    result = build_baseline(
        pd.DataFrame(records),
        "EMP001",
        "2026-01-11"
    )

    computer = result["computer_usage"]

    assert computer["baseline_available"] is True
    assert computer["known_computers"] == ["PC-100"]
    assert computer["frequent_computers"] == ["PC-100"]

    


# =========================================================
# TEST 2: MULTIPLE COMPUTER USAGE
# =========================================================

def test_multiple_computer_usage():

    records = []

    for day in range(1, 11):

        computers = (
            "PC-100"
            if day < 10
            else "PC-100|PC-200"
        )

        records.append(
            create_record(
                "EMP002",
                f"2026-01-{day:02d}",
                computers=computers
            )
        )

    result = build_baseline(
        pd.DataFrame(records),
        "EMP002",
        "2026-01-11"
    )

    profiles = {
        item["computer_id"]: item
        for item in result["computer_usage"]["computer_profiles"]
    }

    assert profiles["PC-100"]["days_used"] == 10
    assert profiles["PC-200"]["days_used"] == 1

    assert profiles["PC-100"]["classification"] == "REGULAR"
    assert profiles["PC-200"]["classification"] == "OCCASIONAL"



# =========================================================
# TEST 3: INSUFFICIENT OBSERVATIONS
# =========================================================

def test_insufficient_observations():

    records = [
        create_record("EMP003", f"2026-01-{day:02d}")
        for day in range(1, 5)
    ]

    result = build_baseline(
        pd.DataFrame(records),
        "EMP003",
        "2026-01-05",
        min_observations=5
    )

    assert result["login_frequency"]["baseline_available"] is False
    assert result["login_timing"]["baseline_available"] is False
    assert result["computer_usage"]["baseline_available"] is False



# =========================================================
# TEST 4: NO LOGIN HISTORY
# =========================================================

def test_no_login_history():

    records = [
        create_record(
            "EMP004",
            f"2026-01-{day:02d}",
            logon_count=0,
            login_minutes=None,
            computers="PC-300",
            has_logon=False
        )
        for day in range(1, 11)
    ]

    result = build_baseline(
        pd.DataFrame(records),
        "EMP004",
        "2026-01-11"
    )

    assert result["login_frequency"]["baseline_available"] is False
    assert result["login_timing"]["baseline_available"] is False

    assert result["computer_usage"]["baseline_available"] is True
    assert result["computer_usage"]["known_computers"] == ["PC-300"]



# =========================================================
# TEST 5: FUTURE-DATA LEAKAGE
# =========================================================

def test_future_data_exclusion():

    records = [
        create_record("EMP005", "2026-01-01", computers="PC-100"),
        create_record("EMP005", "2026-01-02", computers="PC-100"),

        # Target date must be excluded.
        create_record(
            "EMP005",
            "2026-01-03",
            logon_count=100,
            computers="PC-999"
        ),

        # Future date must be excluded.
        create_record(
            "EMP005",
            "2026-01-04",
            logon_count=200,
            computers="PC-888"
        )
    ]

    result = build_baseline(
        pd.DataFrame(records),
        "EMP005",
        "2026-01-03",
        min_observations=1
    )

    computer = result["computer_usage"]

    assert result["historical_employee_days"] == 2
    assert result["login_frequency"]["mean"] == 1.0

    assert computer["known_computers"] == ["PC-100"]

    assert "PC-999" not in computer["known_computers"]
    assert "PC-888" not in computer["known_computers"]



# =========================================================
# TEST 6: MIDNIGHT-BOUNDARY CIRCULAR MEAN
# =========================================================

def test_midnight_boundary():

    login_times = [1435, 1438, 2, 5, 0]

    records = [
        create_record(
            "EMP006",
            f"2026-01-{day:02d}",
            login_minutes=login_time
        )
        for day, login_time in enumerate(login_times, start=1)
    ]

    result = build_baseline(
        pd.DataFrame(records),
        "EMP006",
        "2026-01-06"
    )

    timing = result["login_timing"]

    assert timing["baseline_available"] is True
    assert timing["observation_count"] == 5

    mean = timing["circular_mean_minutes"]

    assert mean < 30 or mean > 1410, (
        f"Circular mean should be near midnight, got {mean}"
    )

    assert timing["resultant_length"] > 0.95



# =========================================================
# TEST 7: NO COMPUTER HISTORY
# =========================================================

def test_no_computer_history():

    records = [
        create_record(
            "EMP007",
            f"2026-01-{day:02d}",
            computers=""
        )
        for day in range(1, 11)
    ]

    result = build_baseline(
        pd.DataFrame(records),
        "EMP007",
        "2026-01-11"
    )

    computer = result["computer_usage"]

    assert computer["baseline_available"] is False
    assert computer["observation_count"] == 0
    assert computer["known_computers"] == []
    assert computer["known_computer_count"] == 0



# =========================================================
# TEST 8: MIXED LOGIN AND EVENT-ONLY HISTORY
# =========================================================

def test_mixed_login_history():

    records = []

    for day in range(1, 11):

        if day <= 5:
            records.append(
                create_record(
                    "EMP008",
                    f"2026-01-{day:02d}",
                    logon_count=2,
                    login_minutes=540,
                    computers="PC-400",
                    has_logon=True
                )
            )

        else:
            records.append(
                create_record(
                    "EMP008",
                    f"2026-01-{day:02d}",
                    logon_count=0,
                    login_minutes=None,
                    computers="PC-400",
                    has_logon=False
                )
            )

    result = build_baseline(
        pd.DataFrame(records),
        "EMP008",
        "2026-01-11"
    )

    assert result["historical_employee_days"] == 10

    frequency = result["login_frequency"]
    timing = result["login_timing"]
    computer = result["computer_usage"]

    assert frequency["observation_count"] == 5
    assert frequency["mean"] == 2.0

    assert timing["observation_count"] == 5

    assert computer["observation_count"] == 10
    assert computer["known_computers"] == ["PC-400"]



# =========================================================
# TEST 9: INVALID LOGIN-TIME VALUES
# =========================================================

def test_invalid_login_times():

    login_times = [540, -1, 1440, 1500, None]

    records = [
        create_record(
            "EMP009",
            f"2026-01-{day:02d}",
            login_minutes=login_time
        )
        for day, login_time in enumerate(login_times, start=1)
    ]

    result = build_baseline(
        pd.DataFrame(records),
        "EMP009",
        "2026-01-06",
        min_observations=1
    )

    frequency = result["login_frequency"]
    timing = result["login_timing"]

    # All five days have recorded login activity.
    assert frequency["observation_count"] == 5

    # Only 540 is a valid time within 0-1439.
    assert timing["baseline_available"] is True
    assert timing["observation_count"] == 1
    assert timing["circular_mean_minutes"] == 540.0



# =========================================================
# TEST 10: DUPLICATE EMPLOYEE-DAY HANDLING
# =========================================================

def test_duplicate_employee_day():

    records = [
        create_record("EMP010", "2026-01-01"),
        create_record("EMP010", "2026-01-02"),
        create_record("EMP010", "2026-01-02")
    ]

    df = pd.DataFrame(records)

    try:
        build_baseline(
            df,
            "EMP010",
            "2026-01-04",
            min_observations=1
        )

    except ValueError as error:
        assert "duplicate employee-day" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError for duplicate employee-day records."
    )


# =========================================================
# MAIN TEST RUNNER
# =========================================================

def main():

    print("=" * 65)
    print("AEGIS - CONSOLIDATED BASELINE ENGINE VALIDATION")
    print("=" * 65)

    tests = [
        test_consistent_computer_usage,
        test_multiple_computer_usage,
        test_insufficient_observations,
        test_no_login_history,
        test_future_data_exclusion,
        test_midnight_boundary,
        test_no_computer_history,
        test_mixed_login_history,
        test_invalid_login_times,
        test_duplicate_employee_day
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print(f"PASS: {test.__name__}")

        except Exception as error:
            failed += 1
            print(f"FAIL: {test.__name__}")
            print(f"Reason: {type(error).__name__}: {error}")

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\nALL 10 BASELINE TESTS PASSED")
    else:
        print("\nBASELINE VALIDATION FAILED")


if __name__ == "__main__":
    main()