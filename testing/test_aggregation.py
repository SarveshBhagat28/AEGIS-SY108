
from detection.evidence_aggregator import (
    aggregate_evidence,
    aggregate_employee_results,
    DETECTOR_WEIGHTS
)


EMPLOYEE = "TEST001"
DATE = "2010-08-18"

DETECTORS = [
    "LOGIN_FREQUENCY",
    "LOGIN_TIMING",
    "COMPUTER_USAGE",
    "CROSS_USER_CORRELATION",
    "FILE_ACTIVITY",
    "DEVICE_ACTIVITY"
]


def create_results(anomalies=None, insufficient=None):
    anomalies = anomalies or {}
    insufficient = insufficient or []

    results = []

    for detector in DETECTORS:

        if detector in insufficient:
            status = "INSUFFICIENT_DATA"
            severity = "UNKNOWN"
            evidence = "Telemetry unavailable."

        elif detector in anomalies:
            status = "ANOMALOUS"
            severity = anomalies[detector]
            evidence = "Test anomaly detected."

        else:
            status = "NORMAL"
            severity = "LOW"
            evidence = "Activity within configured thresholds."

        results.append({
            "employee": EMPLOYEE,
            "date": DATE,
            "detector": detector,
            "status": status,
            "severity": severity,
            "evidence": evidence
        })

    return results


def check_result(name, result, expected_score,
                 expected_level, expected_status,
                 expected_coverage):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Risk score:", result["risk_score"])
    print("Risk level:", result["risk_level"])
    print("Assessment:", result["assessment_status"])
    print("Evidence coverage:", result["evidence_coverage_percentage"])

    assert result["risk_score"] == expected_score
    assert result["risk_level"] == expected_level
    assert result["assessment_status"] == expected_status
    assert result["evidence_coverage_percentage"] == expected_coverage

    print("TEST PASSED")


def main():

    print("AEGIS AGGREGATION ENGINE TESTS")

    print("\n1. Checking detector weights")
    assert sum(DETECTOR_WEIGHTS.values()) == 100
    print("Total detector weights:", sum(DETECTOR_WEIGHTS.values()))
    print("TEST PASSED")

    results = create_results()
    result = aggregate_evidence(results)

    check_result(
        "2. All detectors normal",
        result, 0, "LOW", "COMPLETE", 100
    )

    results = create_results({
        "LOGIN_TIMING": "HIGH"
    })
    result = aggregate_evidence(results)

    check_result(
        "3. One high severity detector",
        result, 20, "LOW", "COMPLETE", 100
    )

    results = create_results({
        "FILE_ACTIVITY": "MODERATE"
    })
    result = aggregate_evidence(results)

    check_result(
        "4. One moderate severity detector",
        result, 12.5, "LOW", "COMPLETE", 100
    )

    results = create_results({
        "LOGIN_TIMING": "HIGH",
        "FILE_ACTIVITY": "MODERATE"
    })
    result = aggregate_evidence(results)

    check_result(
        "5. Moderate overall risk",
        result, 32.5, "MODERATE", "COMPLETE", 100
    )

    results = create_results({
        "LOGIN_TIMING": "HIGH",
        "FILE_ACTIVITY": "HIGH",
        "DEVICE_ACTIVITY": "HIGH"
    })
    result = aggregate_evidence(results)

    check_result(
        "6. High overall risk",
        result, 60, "HIGH", "COMPLETE", 100
    )

    results = create_results({
        detector: "HIGH" for detector in DETECTORS
    })
    result = aggregate_evidence(results)

    check_result(
        "7. Critical overall risk",
        result, 100, "CRITICAL", "COMPLETE", 100
    )

    results = create_results(
        insufficient=["FILE_ACTIVITY"]
    )
    result = aggregate_evidence(results)

    check_result(
        "8. One detector has insufficient data",
        result, 0, "INSUFFICIENT_DATA", "PARTIAL", 83.33
    )

    results = create_results(
        insufficient=["FILE_ACTIVITY", "DEVICE_ACTIVITY"]
    )
    result = aggregate_evidence(results)

    check_result(
        "9. Two detectors have insufficient data",
        result, 0, "INSUFFICIENT_DATA", "PARTIAL", 66.67
    )

    results = create_results({
        "LOGIN_TIMING": "HIGH"
    }, insufficient=["FILE_ACTIVITY"])

    result = aggregate_evidence(results)

    check_result(
        "10. Anomaly with insufficient data",
        result, 20, "LOW", "PARTIAL", 83.33
    )

    print("\n" + "=" * 60)
    print("ALL AGGREGATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
