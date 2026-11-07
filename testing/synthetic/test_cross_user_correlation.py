import unittest
import pandas as pd

from detection.baseline_engine import prepare_daily_data
from detection.cross_user_correlation import (
    evaluate_cross_user_correlation
)


TARGET_DATE = "2026-01-31"
HISTORY_DATE = "2026-01-15"


def create_daily_data(rows):
    """
    Build synthetic employee-day records and prepare them
    using the production baseline data preparation function.
    """

    data = pd.DataFrame(
        rows,
        columns=[
            "user",
            "calendar_date",
            "computers_used",
            "has_logon_activity"
        ]
    )

    # Add columns required by prepare_daily_data.
    data["logon_count"] = data["has_logon_activity"].astype(int)
    data["first_login_minutes"] = 540

    return prepare_daily_data(data)


class TestCrossUserCorrelation(unittest.TestCase):

    def test_normal_relationship(self):
        """Employee uses a computer not shared that day."""

        data = create_daily_data([
            ("EMP_A", HISTORY_DATE, "PC-01", True),
            ("EMP_A", TARGET_DATE, "PC-01", True)
        ])

        result = evaluate_cross_user_correlation(
            data, "EMP_A", TARGET_DATE
        )

        self.assertEqual(result["status"], "NORMAL")
        self.assertEqual(result["severity"], "LOW")
        self.assertEqual(result["shared_computers"], [])
    def test_new_shared_relationship(self):
        """Employee uses another employee's computer for the first time."""

        data = create_daily_data([
            ("EMP_B", HISTORY_DATE, "PC-01", True),
            ("EMP_A", TARGET_DATE, "PC-01", True),
            ("EMP_B", TARGET_DATE, "PC-01", True)
        ])

        result = evaluate_cross_user_correlation(
            data, "EMP_A", TARGET_DATE
        )

        self.assertEqual(result["status"], "ANOMALOUS")
        self.assertEqual(result["severity"], "MODERATE")
        self.assertEqual(len(result["new_relationships"]), 1)
        self.assertEqual(
            result["new_relationships"][0]["computer"],
            "PC-01"
        )

    def test_multiple_shared_users(self):
        """Three other employees share the target employee's computer."""

        data = create_daily_data([
            ("EMP_A", HISTORY_DATE, "PC-01", True),
            ("EMP_A", TARGET_DATE, "PC-01", True),
            ("EMP_B", TARGET_DATE, "PC-01", True),
            ("EMP_C", TARGET_DATE, "PC-01", True),
            ("EMP_D", TARGET_DATE, "PC-01", True)
        ])

        result = evaluate_cross_user_correlation(
            data, "EMP_A", TARGET_DATE
        )

        self.assertEqual(result["status"], "ANOMALOUS")
        self.assertEqual(result["severity"], "HIGH")
        self.assertEqual(len(result["associated_users"]), 3)

    def test_existing_shared_relationship(self):
        """Both employees have an established relationship with the PC."""

        data = create_daily_data([
            ("EMP_A", HISTORY_DATE, "PC-01", True),
            ("EMP_B", HISTORY_DATE, "PC-01", True),
            ("EMP_A", TARGET_DATE, "PC-01", True),
            ("EMP_B", TARGET_DATE, "PC-01", True)
        ])

        result = evaluate_cross_user_correlation(
            data, "EMP_A", TARGET_DATE
        )

        self.assertEqual(result["status"], "NORMAL")
        self.assertEqual(result["severity"], "LOW")
        self.assertEqual(result["new_relationships"], [])
    def test_no_target_day_activity(self):
        """Employee exists historically but has no target-day activity."""

        data = create_daily_data([
            ("EMP_A", HISTORY_DATE, "PC-01", True),
            ("EMP_B", TARGET_DATE, "PC-01", True)
        ])

        result = evaluate_cross_user_correlation(
            data, "EMP_A", TARGET_DATE
        )

        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["severity"], "UNKNOWN")

    def test_missing_computer_telemetry(self):
        """Target-day login exists, but computer information is unavailable."""

        data = create_daily_data([
            ("EMP_A", HISTORY_DATE, "PC-01", True),
            ("EMP_A", TARGET_DATE, None, True)
        ])

        result = evaluate_cross_user_correlation(
            data, "EMP_A", TARGET_DATE
        )

        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["severity"], "UNKNOWN")

    def test_relationship_outside_lookback(self):
        """A historical relationship outside the 30-day window is ignored."""

        old_date = "2025-12-01"

        data = create_daily_data([
            ("EMP_A", old_date, "PC-01", True),
            ("EMP_B", old_date, "PC-01", True),
            ("EMP_A", TARGET_DATE, "PC-01", True),
            ("EMP_B", TARGET_DATE, "PC-01", True)
        ])

        result = evaluate_cross_user_correlation(
            data,
            "EMP_A",
            TARGET_DATE,
            lookback_days=30
        )

        self.assertEqual(result["status"], "ANOMALOUS")
        self.assertEqual(result["severity"], "MODERATE")
        self.assertEqual(len(result["new_relationships"]), 1)


if __name__ == "__main__":

    print("=" * 65)
    print("AEGIS - CROSS-USER CORRELATION TEST SUITE")
    print("=" * 65)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestCrossUserCorrelation
    )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 65)
    print(f"Total tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures) + len(result.errors)}")

    if result.wasSuccessful():
        print("ALL CROSS-USER CORRELATION TESTS PASSED")
    else:
        print("CROSS-USER CORRELATION TESTS FAILED")

    print("=" * 65)
