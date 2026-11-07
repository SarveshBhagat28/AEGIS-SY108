import unittest

from detection.evidence_aggregator import aggregate_evidence


EMPLOYEE = "EMP001"
TARGET_DATE = "2026-01-31"


def create_result(
    detector,
    status="NORMAL",
    severity="LOW",
    evidence="Test evidence"
):
    return {
        "employee": EMPLOYEE,
        "date": TARGET_DATE,
        "detector": detector,
        "status": status,
        "severity": severity,
        "evidence": evidence
    }


def create_all_results():
    return [
        create_result("LOGIN_FREQUENCY"),
        create_result("LOGIN_TIMING"),
        create_result("COMPUTER_USAGE"),
        create_result("CROSS_USER_CORRELATION")
    ]


class TestEvidenceAggregator(unittest.TestCase):

    def test_all_detectors_normal(self):
        results = create_all_results()

        assessment = aggregate_evidence(results)

        self.assertEqual(assessment["risk_score"], 0)
        self.assertEqual(assessment["risk_level"], "LOW")
        self.assertEqual(assessment["assessment_status"], "COMPLETE")
        self.assertEqual(assessment["anomalous_detector_count"], 0)

    def test_single_moderate_anomaly(self):
        results = create_all_results()

        results[0] = create_result(
            "LOGIN_FREQUENCY",
            "ANOMALOUS",
            "MODERATE"
        )

        assessment = aggregate_evidence(results)

        self.assertEqual(assessment["risk_score"], 10)
        self.assertEqual(assessment["risk_level"], "LOW")
        self.assertEqual(assessment["anomalous_detector_count"], 1)

    def test_single_high_anomaly(self):
        results = create_all_results()

        results[1] = create_result(
            "LOGIN_TIMING",
            "ANOMALOUS",
            "HIGH"
        )

        assessment = aggregate_evidence(results)

        self.assertEqual(assessment["risk_score"], 25)
        self.assertEqual(assessment["risk_level"], "MODERATE")

    def test_multiple_anomalies(self):
        results = create_all_results()

        results[1] = create_result(
            "LOGIN_TIMING",
            "ANOMALOUS",
            "HIGH"
        )

        results[3] = create_result(
            "CROSS_USER_CORRELATION",
            "ANOMALOUS",
            "MODERATE"
        )

        assessment = aggregate_evidence(results)

        self.assertEqual(assessment["risk_score"], 40)
        self.assertEqual(assessment["risk_level"], "MODERATE")
        self.assertEqual(assessment["anomalous_detector_count"], 2)
        self.assertEqual(len(assessment["findings"]), 2)

    def test_missing_detector(self):
        results = create_all_results()[:3]

        assessment = aggregate_evidence(results)

        self.assertEqual(assessment["assessment_status"], "PARTIAL")
        self.assertEqual(assessment["evidence_coverage_percentage"], 75)
        self.assertEqual(
            assessment["missing_detectors"],
            ["CROSS_USER_CORRELATION"]
        )
        self.assertEqual(assessment["risk_level"], "INSUFFICIENT_DATA")

    def test_insufficient_telemetry(self):
        results = create_all_results()

        results[2] = create_result(
            "COMPUTER_USAGE",
            "INSUFFICIENT_DATA",
            "UNKNOWN",
            "Computer telemetry unavailable"
        )

        assessment = aggregate_evidence(results)

        self.assertEqual(assessment["assessment_status"], "PARTIAL")
        self.assertEqual(assessment["insufficient_detector_count"], 1)
        self.assertEqual(
            assessment["insufficient_detectors"],
            ["COMPUTER_USAGE"]
        )
        self.assertEqual(assessment["risk_level"], "INSUFFICIENT_DATA")

    def test_duplicate_detector_rejected(self):
        results = create_all_results()
        results.append(results[0].copy())

        with self.assertRaises(ValueError):
            aggregate_evidence(results)

    def test_different_employee_rejected(self):
        results = create_all_results()
        results[1]["employee"] = "EMP002"

        with self.assertRaises(ValueError):
            aggregate_evidence(results)

    def test_different_date_rejected(self):
        results = create_all_results()
        results[1]["date"] = "2026-02-01"

        with self.assertRaises(ValueError):
            aggregate_evidence(results)

    def test_invalid_detector_status_rejected(self):
        results = create_all_results()

        results[0]["status"] = "SUSPICIOUS"

        with self.assertRaises(ValueError):
            aggregate_evidence(results)

    def test_invalid_detector_severity_rejected(self):
        results = create_all_results()

        results[0]["severity"] = "EXTREME"

        with self.assertRaises(ValueError):
            aggregate_evidence(results)


if __name__ == "__main__":

    print("=" * 65)
    print("AEGIS - EVIDENCE AGGREGATION TEST SUITE")
    print("=" * 65)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestEvidenceAggregator
    )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    passed = (
        result.testsRun
        - len(result.failures)
        - len(result.errors)
    )

    print("\n" + "=" * 65)
    print(f"Total tests: {result.testsRun}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(result.failures) + len(result.errors)}")

    if result.wasSuccessful():
        print("ALL EVIDENCE AGGREGATION TESTS PASSED")
    else:
        print("EVIDENCE AGGREGATION TESTS FAILED")

    print("=" * 65)
    