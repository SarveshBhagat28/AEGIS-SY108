from collections import Counter
from datetime import datetime


# Initial engineering weights.
# These are heuristic values, NOT calibrated probabilities.
DETECTOR_WEIGHTS = {
    "LOGIN_FREQUENCY": 15,
    "LOGIN_TIMING": 20,
    "COMPUTER_USAGE": 10,
    "CROSS_USER_CORRELATION": 15,
    "FILE_ACTIVITY": 25,
    "DEVICE_ACTIVITY": 15
}

SEVERITY_MULTIPLIERS = {
    "LOW": 0.0,
    "MODERATE": 0.5,
    "HIGH": 1.0
}

RISK_THRESHOLDS = {
    "LOW": 0,
    "MODERATE": 25,
    "HIGH": 50,
    "CRITICAL": 75
}

VALID_STATUSES = {
    "NORMAL",
    "ANOMALOUS",
    "INSUFFICIENT_DATA"
}

VALID_SEVERITIES = {
    "LOW",
    "MODERATE",
    "HIGH",
    "UNKNOWN"
}


def validate_detector_result(result):
    """Validate the structure of an individual detector result."""

    required_fields = {
        "employee",
        "date",
        "detector",
        "status",
        "severity"
    }

    if not isinstance(result, dict):
        raise TypeError("Each detector result must be a dictionary.")

    missing = required_fields - set(result.keys())

    if missing:
        raise ValueError(
            f"Detector result missing fields: {sorted(missing)}"
        )

    if result["status"] not in VALID_STATUSES:
        raise ValueError(
            f"Invalid detector status: {result['status']}"
        )

    if result["severity"] not in VALID_SEVERITIES:
        raise ValueError(
            f"Invalid detector severity: {result['severity']}"
        )

    if result["detector"] not in DETECTOR_WEIGHTS:
        raise ValueError(
            f"Unknown detector: {result['detector']}"
        )


def calculate_detector_score(result):
    """
    Calculate the weighted contribution of one detector.

    INSUFFICIENT_DATA contributes no risk points.
    NORMAL contributes no risk points.
    """

    validate_detector_result(result)

    if result["status"] != "ANOMALOUS":
        return 0.0

    severity = result["severity"]

    if severity == "UNKNOWN":
        raise ValueError(
            "An ANOMALOUS result cannot have UNKNOWN severity."
        )

    weight = DETECTOR_WEIGHTS[result["detector"]]
    multiplier = SEVERITY_MULTIPLIERS[severity]

    return round(weight * multiplier, 2)


def determine_risk_level(score):
    """Map an aggregate score to an overall risk category."""

    if score >= RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"

    if score >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"

    if score >= RISK_THRESHOLDS["MODERATE"]:
        return "MODERATE"

    return "LOW"


def aggregate_evidence(detector_results):
    """
    Aggregate detector outputs into an employee-level assessment.

    All results must belong to the same employee and target date.
    Missing or insufficient telemetry is not treated as normal.
    """

    if not isinstance(detector_results, list):
        raise TypeError("detector_results must be a list.")

    if not detector_results:
        raise ValueError("At least one detector result is required.")

    for result in detector_results:
        validate_detector_result(result)

    employees = {
        result["employee"]
        for result in detector_results
    }

    dates = {
        str(result["date"])
        for result in detector_results
    }

    if len(employees) != 1:
        raise ValueError(
            "All detector results must belong to the same employee."
        )

    if len(dates) != 1:
        raise ValueError(
            "All detector results must have the same target date."
        )

    detector_names = [
        result["detector"]
        for result in detector_results
    ]

    if len(detector_names) != len(set(detector_names)):
        raise ValueError(
            "Duplicate detector results are not allowed."
        )

    expected_detectors = set(DETECTOR_WEIGHTS)
    received_detectors = set(detector_names)

    missing_detectors = sorted(
        expected_detectors - received_detectors
    )

    detector_scores = {}
    anomaly_findings = []
    insufficient_detectors = []
    normal_detectors = []

    # Process each detector.
    for result in detector_results:

        detector = result["detector"]
        status = result["status"]

        detector_scores[detector] = calculate_detector_score(result)

        if status == "ANOMALOUS":

            anomaly_findings.append({
                "detector": detector,
                "severity": result["severity"],
                "score": detector_scores[detector],
                "evidence": result.get("evidence")
            })

        elif status == "INSUFFICIENT_DATA":

            insufficient_detectors.append(detector)

        else:

            normal_detectors.append(detector)

    total_score = round(sum(detector_scores.values()), 2)

    risk_level = determine_risk_level(total_score)
    
    anomalous_count = len(anomaly_findings)
    evaluated_count = len(normal_detectors) + anomalous_count
    
    # Confidence here means evidence coverage, not statistical confidence.
    evidence_coverage = round(
        evaluated_count / len(expected_detectors) * 100,
        2
    )

    if missing_detectors or insufficient_detectors:
        assessment_status = "PARTIAL"

    else:
        assessment_status = "COMPLETE"

    # Prevent a fully unevaluated employee from appearing low-risk.
    if anomalous_count == 0 and assessment_status == "PARTIAL":
        risk_level = "INSUFFICIENT_DATA"

    return {
        "employee": next(iter(employees)),
        "date": next(iter(dates)),
        "assessment_status": assessment_status,
        "risk_score": total_score,
        "risk_level": risk_level,
        "evidence_coverage_percentage": evidence_coverage,
        "evaluated_detectors": evaluated_count,
        "expected_detectors": len(expected_detectors),
        "anomalous_detector_count": anomalous_count,
        "normal_detector_count": len(normal_detectors),
        "insufficient_detector_count": len(insufficient_detectors),
        "missing_detectors": missing_detectors,
        "insufficient_detectors": insufficient_detectors,
        "normal_detectors": normal_detectors,
        "detector_scores": detector_scores,
        "findings": anomaly_findings
    }


def aggregate_employee_results(employee, target_date, detector_results):
    """
    Convenience wrapper that verifies the requested employee/date
    before aggregation.
    """

    for result in detector_results:

        if result.get("employee") != employee:
            raise ValueError(
                "A detector result belongs to a different employee."
            )

        if str(result.get("date")) != str(target_date):
            raise ValueError(
                "A detector result belongs to a different date."
            )

    return aggregate_evidence(detector_results)


def main():

    print("=" * 65)
    print("AEGIS - EVIDENCE AGGREGATION ENGINE")
    print("=" * 65)

    # Demonstration using four detector outputs.
    sample_results = [
        {
            "employee": "EMP001",
            "date": "2026-01-31",
            "detector": "LOGIN_FREQUENCY",
            "status": "NORMAL",
            "severity": "LOW",
            "evidence": "Login frequency is within baseline."
        },
        {
            "employee": "EMP001",
            "date": "2026-01-31",
            "detector": "LOGIN_TIMING",
            "status": "ANOMALOUS",
            "severity": "HIGH",
            "evidence": "Login occurred significantly earlier than baseline."
        },
        {
            "employee": "EMP001",
            "date": "2026-01-31",
            "detector": "COMPUTER_USAGE",
            "status": "NORMAL",
            "severity": "LOW",
            "evidence": "Computer usage matches historical behaviour."
        },
        {
            "employee": "EMP001",
            "date": "2026-01-31",
            "detector": "CROSS_USER_CORRELATION",
            "status": "ANOMALOUS",
            "severity": "MODERATE",
            "evidence": "New employee-computer relationship detected."
        }
    ]

    result = aggregate_evidence(sample_results)

    print("\nAGGREGATED EMPLOYEE ASSESSMENT")
    print("-" * 65)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 65)
    print("EVIDENCE AGGREGATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
    