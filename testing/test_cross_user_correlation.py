
import pandas as pd

from detection.cross_user_correlation import (
    evaluate_cross_user_correlation
)


TARGET_DATE = pd.Timestamp("2011-02-10")


def create_data(
    history_days=5,
    target_computers=None,
    sharing=None,
    history_computers=None
):
    rows = []

    history_computers = history_computers or ["PC-A"]

    for day in range(history_days):
        date = TARGET_DATE - pd.Timedelta(days=history_days - day)

        rows.append({
            "user": "EMP001",
            "calendar_date": date,
            "computers_used": "|".join(history_computers),
            "has_logon_activity": True
        })

    target_computers = target_computers or ["PC-A"]
    sharing = sharing or {}

    rows.append({
        "user": "EMP001",
        "calendar_date": TARGET_DATE,
        "computers_used": "|".join(target_computers),
        "has_logon_activity": True
    })

    for computer, users in sharing.items():
        for user in users:
            rows.append({
                "user": user,
                "calendar_date": TARGET_DATE,
                "computers_used": computer,
                "has_logon_activity": True
            })

    return pd.DataFrame(rows)


def run_test(name, data, expected_status, expected_severity):
    result = evaluate_cross_user_correlation(
        data,
        "EMP001",
        TARGET_DATE
    )

    passed = (
        result["status"] == expected_status
        and result["severity"] == expected_severity
    )

    print(f"\n{name}")
    print("Status:", result["status"])
    print("Severity:", result["severity"])
    print("Evidence:", result["evidence"])
    print("RESULT:", "PASSED" if passed else "FAILED")

    assert passed, f"{name} failed: {result}"


def main():

    print("=" * 65)
    print("AEGIS - CROSS-USER CORRELATION TESTS")
    print("=" * 65)

    # 1. No computer sharing.
    data = create_data()

    run_test(
        "1. No computer sharing",
        data,
        "NORMAL",
        "LOW"
    )

    # 2. Known sharing with one other employee.
    data = create_data(
        sharing={"PC-A": ["EMP002"]}
    )

    run_test(
        "2. Known sharing with one employee",
        data,
        "NORMAL",
        "LOW"
    )

    # 3. New employee-computer relationship.
    data = create_data(
        target_computers=["PC-B"],
        sharing={"PC-B": ["EMP002"]}
    )

    run_test(
        "3. New employee-computer relationship",
        data,
        "ANOMALOUS",
        "MODERATE"
    )

    # 4. Three other employees sharing the same computer.
    data = create_data(
        sharing={"PC-A": ["EMP002", "EMP003", "EMP004"]}
    )

    run_test(
        "4. Three employees sharing one computer",
        data,
        "ANOMALOUS",
        "HIGH"
    )

    # 5. Three computers, each shared with one employee.
    data = create_data(
        target_computers=["PC-A", "PC-B", "PC-C"],
        history_computers=["PC-A", "PC-B", "PC-C"],
        sharing={
            "PC-A": ["EMP002"],
            "PC-B": ["EMP003"],
            "PC-C": ["EMP004"]
        }
    )
    
    run_test(
        "5. Three separate computers with one user each",
        data,
        "NORMAL",
        "LOW"
    )

    # 6. Insufficient historical activity.
    data = create_data(
        history_days=2,
        target_computers=["PC-B"],
        sharing={"PC-B": ["EMP002"]}
    )

    run_test(
        "6. Insufficient historical activity",
        data,
        "INSUFFICIENT_DATA",
        "UNKNOWN"
    )

    print("\n" + "=" * 65)
    print("ALL CROSS-USER CORRELATION TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()

