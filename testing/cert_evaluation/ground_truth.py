from datetime import datetime


CERT_SCENARIOS = {
    "1": {
        "scenario": "Scenario 1",
        "insider": "ACM2278",
        "start": "2010-08-18 21:47:42",
        "end": "2010-08-24 03:48:51",
        "known_telemetry": ["logon", "device", "file"],
        "required_telemetry": ["logon", "device", "file"],
    },
    "2": {
        "scenario": "Scenario 2",
        "insider": "CMP2946",
        "start": "2011-02-07 12:28:06",
        "end": "2011-03-04 12:30:25",
        "known_telemetry": ["email"],
        "required_telemetry": ["email"],
    },
    "3": {
        "scenario": "Scenario 3",
        "insider": "PLJ1771",
        "start": "2010-08-12 08:09:00",
        "end": "2010-08-12 16:11:39",
        "known_telemetry": ["email"],
        "required_telemetry": ["email"],
    },
    "4": {
        "scenario": "Scenario 4",
        "insider": "CDE1846",
        "start": "2011-02-21 11:43:39",
        "end": "2011-04-25 17:55:00",
        "known_telemetry": ["logon", "email", "file"],
        "required_telemetry": ["logon", "email", "file"],
    },
    "5": {
        "scenario": "Scenario 5",
        "insider": "MBG3183",
        "start": "2010-10-12 13:21:59",
        "end": "2010-10-12 13:22:56",
        "known_telemetry": ["http"],
        "required_telemetry": ["http"],
    },
}


# Ground-truth classification states
POSITIVE = "POSITIVE"
OUTSIDE_KNOWN_INCIDENT = "OUTSIDE_KNOWN_INCIDENT"
UNLABELED = "UNLABELED"


def load_ground_truth():
    """Return ground-truth records with parsed timestamps."""

    records = []

    for scenario_id, details in CERT_SCENARIOS.items():
        record = details.copy()

        record["scenario_id"] = scenario_id
        record["start"] = datetime.strptime(
            details["start"], "%Y-%m-%d %H:%M:%S"
        )
        record["end"] = datetime.strptime(
            details["end"], "%Y-%m-%d %H:%M:%S"
        )

        if record["start"] > record["end"]:
            raise ValueError(
                f"Invalid incident window for scenario {scenario_id}"
            )

        records.append(record)

    insider_ids = [record["insider"] for record in records]

    if len(insider_ids) != len(set(insider_ids)):
        raise ValueError("Duplicate insider identities found.")

    return records


def get_insider_ids():
    """Return the known insider identities."""

    return {
        record["insider"]
        for record in load_ground_truth()
    }


def get_scenario_for_employee(employee):
    """Return the ground-truth scenario for a known insider."""

    for record in load_ground_truth():
        if record["insider"] == employee:
            return record

    return None


def normalize_timestamp(timestamp):
    """Convert supported timestamp inputs into datetime objects."""

    if isinstance(timestamp, datetime):
        return timestamp

    if isinstance(timestamp, str):
        return datetime.strptime(
            timestamp, "%Y-%m-%d %H:%M:%S"
        )

    raise TypeError(
        "Timestamp must be a datetime object or "
        "a string in YYYY-MM-DD HH:MM:SS format."
    )


def classify_ground_truth(employee, timestamp):
    """
    Classify an employee timestamp against known CERT incidents.

    Returns:
        POSITIVE:
            Known insider and timestamp inside the incident window.

        OUTSIDE_KNOWN_INCIDENT:
            Known insider but timestamp outside the incident window.

        UNLABELED:
            Employee has no configured ground-truth incident.

    Both incident boundaries are inclusive.
    """

    scenario = get_scenario_for_employee(employee)

    if scenario is None:
        return UNLABELED

    timestamp = normalize_timestamp(timestamp)

    if scenario["start"] <= timestamp <= scenario["end"]:
        return POSITIVE

    return OUTSIDE_KNOWN_INCIDENT


def is_inside_incident_window(employee, timestamp):
    """
    Backward-compatible Boolean incident-window check.

    Returns True only for a known positive incident timestamp.
    """

    return classify_ground_truth(employee, timestamp) == POSITIVE


if __name__ == "__main__":

    records = load_ground_truth()

    print("=" * 65)
    print("AEGIS - CERT GROUND-TRUTH VALIDATION")
    print("=" * 65)

    for record in records:
        duration = record["end"] - record["start"]

        print(
            f"{record['scenario']}: "
            f"{record['insider']} | "
            f"{record['start']} -> {record['end']} | "
            f"Duration: {duration}"
        )

        print(
            f"  Required telemetry: "
            f"{', '.join(record['required_telemetry'])}"
        )

    print("-" * 65)
    print(f"Scenarios loaded: {len(records)}")
    print(f"Unique insider identities: {len(get_insider_ids())}")

    # Classification sanity checks
    checks = [
        (
            "Known insider inside incident",
            classify_ground_truth(
                "ACM2278", "2010-08-20 10:00:00"
            ),
            POSITIVE,
        ),
        (
            "Known insider outside incident",
            classify_ground_truth(
                "ACM2278", "2010-08-25 10:00:00"
            ),
            OUTSIDE_KNOWN_INCIDENT,
        ),
        (
            "Unknown employee",
            classify_ground_truth(
                "UNKNOWN_USER", "2010-08-20 10:00:00"
            ),
            UNLABELED,
        ),
        (
            "Scenario 5 exact start boundary",
            classify_ground_truth(
                "MBG3183", "2010-10-12 13:21:59"
            ),
            POSITIVE,
        ),
        (
            "Scenario 5 exact end boundary",
            classify_ground_truth(
                "MBG3183", "2010-10-12 13:22:56"
            ),
            POSITIVE,
        ),
    ]

    print("\nCLASSIFICATION VALIDATION")
    print("-" * 65)

    passed = 0

    for description, actual, expected in checks:
        status = "PASS" if actual == expected else "FAIL"

        if status == "PASS":
            passed += 1

        print(
            f"{status}: {description} | "
            f"Expected={expected}, Actual={actual}"
        )

    print("-" * 65)
    print(f"Classification checks: {passed}/{len(checks)}")

    if passed != len(checks):
        raise SystemExit("Ground-truth classification validation failed.")

    print("Ground-truth configuration loaded successfully.")