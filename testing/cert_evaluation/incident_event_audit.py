import pandas as pd

from pathlib import Path
from datetime import timedelta

from testing.cert_evaluation.ground_truth import load_ground_truth


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

REPORT_DIR = BASE_DIR / "testing" / "reports"

CHUNK_SIZE = 200_000

TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"

AVAILABLE_DATASETS = ["logon", "file", "device"]

OPTIONAL_DATASETS = ["email", "http"]


# ============================================================
# INITIALIZATION
# ============================================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = load_ground_truth()

INSIDER_IDS = {
    record["insider"]
    for record in SCENARIOS
}


# ============================================================
# TELEMETRY AVAILABILITY
# ============================================================

def check_telemetry_availability():
    """
    Check whether telemetry source files exist locally.
    """

    availability = {}

    for dataset in AVAILABLE_DATASETS + OPTIONAL_DATASETS:

        file_path = DATA_DIR / f"{dataset}.csv"

        availability[dataset] = {
            "available": file_path.is_file(),
            "path": str(file_path) if file_path.is_file() else None,
        }

    return availability


# ============================================================
# EVENT PROCESSING
# ============================================================

def process_dataset(dataset):
    """
    Process a raw telemetry dataset in chunks.

    Counts events for known insiders and preserves exact
    timestamps for events inside incident windows or on
    incident calendar dates.

    Returns:
        summary_counts
        event_records
        invalid_timestamp_counts
    """

    file_path = DATA_DIR / f"{dataset}.csv"

    if not file_path.is_file():
        return None, [], {}

    print(f"\nProcessing {dataset.upper()} telemetry...")

    # Identify available columns.
    header = pd.read_csv(file_path, nrows=0)

    available_columns = list(header.columns)

    required_columns = ["date", "user"]

    missing_columns = [
        column
        for column in required_columns
        if column not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset}.csv is missing required columns: "
            f"{missing_columns}"
        )

    optional_columns = [
        column
        for column in ["id", "pc", "activity"]
        if column in available_columns
    ]

    use_columns = required_columns + optional_columns

    summary_counts = {}

    for scenario in SCENARIOS:

        summary_counts[scenario["scenario_id"]] = {
            "inside_window_events": 0,
            "outside_window_events": 0,
            "same_day_outside_events": 0,
            "invalid_timestamps": 0,
        }

    event_records = []

    invalid_timestamp_counts = {
        scenario["insider"]: 0
        for scenario in SCENARIOS
    }

    reader = pd.read_csv(
        file_path,
        usecols=use_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        chunk["user"] = chunk["user"].astype(str).str.strip()

        # Only retain the five configured insider identities.
        chunk = chunk[
            chunk["user"].isin(INSIDER_IDS)
        ].copy()

        if chunk.empty:
            continue

        chunk["timestamp"] = pd.to_datetime(
            chunk["date"],
            format=TIMESTAMP_FORMAT,
            errors="coerce",
        )

        invalid_mask = chunk["timestamp"].isna()

        invalid_rows = chunk[invalid_mask]

        for _, row in invalid_rows.iterrows():

            employee = row["user"]

            invalid_timestamp_counts[employee] += 1

            scenario = next(
                item for item in SCENARIOS
                if item["insider"] == employee
            )

            summary_counts[
                scenario["scenario_id"]
            ]["invalid_timestamps"] += 1

        chunk = chunk[
            chunk["timestamp"].notna()
        ].copy()

        if chunk.empty:
            continue

        for scenario in SCENARIOS:

            scenario_id = scenario["scenario_id"]
            employee = scenario["insider"]

            employee_events = chunk[
                chunk["user"] == employee
            ]

            if employee_events.empty:
                continue

            start = scenario["start"]
            end = scenario["end"]

            inside_mask = (
                (employee_events["timestamp"] >= start)
                &
                (employee_events["timestamp"] <= end)
            )

            incident_start_date = start.date()
            incident_end_date = end.date()

            same_day_mask = employee_events[
                "timestamp"
            ].dt.date.isin(
                [incident_start_date, incident_end_date]
            )

            same_day_outside_mask = (
                same_day_mask & ~inside_mask
            )

            inside_count = int(inside_mask.sum())

            outside_count = int(
                len(employee_events) - inside_count
            )

            same_day_outside_count = int(
                same_day_outside_mask.sum()
            )

            summary_counts[scenario_id][
                "inside_window_events"
            ] += inside_count

            summary_counts[scenario_id][
                "outside_window_events"
            ] += outside_count

            summary_counts[scenario_id][
                "same_day_outside_events"
            ] += same_day_outside_count

            # Preserve exact event details for events inside
            # the incident window or on its boundary dates.
            relevant_mask = (
                inside_mask | same_day_outside_mask
            )

            relevant_events = employee_events[
                relevant_mask
            ].copy()

            for _, row in relevant_events.iterrows():

                inside = bool(
                    start <= row["timestamp"] <= end
                )

                record = {
                    "scenario_id": scenario_id,
                    "scenario": scenario["scenario"],
                    "user": employee,
                    "dataset": dataset,
                    "timestamp": row["timestamp"],
                    "incident_start": start,
                    "incident_end": end,
                    "window_relation": (
                        "IN_WINDOW"
                        if inside
                        else "SAME_DAY_OUTSIDE_WINDOW"
                    ),
                }

                for column in optional_columns:
                    record[column] = row[column]

                event_records.append(record)

        if chunk_number % 5 == 0:
            print(
                f"  Processed {chunk_number} chunks..."
            )

    print(f"  Finished {dataset.upper()}.")

    return (
        summary_counts,
        event_records,
        invalid_timestamp_counts,
    )


# ============================================================
# SCENARIO-LEVEL SUMMARY
# ============================================================

def build_scenario_summary(
    availability,
    dataset_results,
):
    """
    Build one summary row per scenario and telemetry source.
    """

    records = []

    for scenario in SCENARIOS:

        for dataset in (
            AVAILABLE_DATASETS + OPTIONAL_DATASETS
        ):

            required = (
                dataset in scenario["required_telemetry"]
            )

            known = (
                dataset in scenario["known_telemetry"]
            )

            source_available = availability[
                dataset
            ]["available"]

            if not source_available:
                source_status = "UNAVAILABLE"

            elif dataset in dataset_results:
                source_status = "AVAILABLE"

            else:
                source_status = "NOT_PROCESSED"

            inside_count = None
            outside_count = None
            same_day_outside_count = None
            invalid_count = None

            if (
                source_available
                and dataset in dataset_results
            ):

                counts = dataset_results[
                    dataset
                ][scenario["scenario_id"]]

                inside_count = counts[
                    "inside_window_events"
                ]

                outside_count = counts[
                    "outside_window_events"
                ]

                same_day_outside_count = counts[
                    "same_day_outside_events"
                ]

                invalid_count = counts[
                    "invalid_timestamps"
                ]

            records.append({
                "scenario_id": scenario["scenario_id"],
                "scenario": scenario["scenario"],
                "user": scenario["insider"],
                "dataset": dataset,
                "required_telemetry": required,
                "known_telemetry": known,
                "source_available": source_available,
                "source_status": source_status,
                "inside_window_events": inside_count,
                "outside_window_events": outside_count,
                "same_day_outside_events": same_day_outside_count,
                "invalid_timestamps": invalid_count,
                "incident_start": scenario["start"],
                "incident_end": scenario["end"],
            })

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("AEGIS - CERT INCIDENT-LEVEL EVENT AUDIT")
    print("=" * 75)

    availability = check_telemetry_availability()

    print("\nTELEMETRY SOURCE AVAILABILITY")
    print("-" * 75)

    for dataset, details in availability.items():

        status = (
            "AVAILABLE"
            if details["available"]
            else "UNAVAILABLE"
        )

        print(f"{dataset.upper():10} {status}")

    dataset_results = {}
    all_event_records = []

    for dataset in AVAILABLE_DATASETS:

        result = process_dataset(dataset)

        if result[0] is None:
            print(
                f"\nWARNING: {dataset}.csv is missing."
            )
            continue

        counts, events, invalid_counts = result

        dataset_results[dataset] = counts

        all_event_records.extend(events)

        total_inside = sum(
            item["inside_window_events"]
            for item in counts.values()
        )

        total_outside = sum(
            item["outside_window_events"]
            for item in counts.values()
        )

        print(
            f"  Events inside incident windows: "
            f"{total_inside:,}"
        )

        print(
            f"  Events outside incident windows: "
            f"{total_outside:,}"
        )

    # Build scenario-level report.
    summary_df = build_scenario_summary(
        availability,
        dataset_results,
    )

    summary_path = (
        REPORT_DIR / "incident_telemetry_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # Build event-level report.
    if all_event_records:

        event_df = pd.DataFrame(
            all_event_records
        )

        event_df = event_df.sort_values(
            ["scenario_id", "timestamp", "dataset"]
        )

    else:

        event_df = pd.DataFrame()

    event_path = (
        REPORT_DIR / "incident_event_details.csv"
    )

    event_df.to_csv(
        event_path,
        index=False,
    )

    # Print scenario summary.
    print("\nSCENARIO INCIDENT-WINDOW SUMMARY")
    print("=" * 75)

    for scenario in SCENARIOS:

        scenario_id = scenario["scenario_id"]

        print(
            f"\n{scenario['scenario']} "
            f"({scenario['insider']})"
        )

        print(
            f"Incident: {scenario['start']} "
            f"-> {scenario['end']}"
        )

        for dataset in AVAILABLE_DATASETS:

            if dataset not in dataset_results:
                continue

            counts = dataset_results[
                dataset
            ][scenario_id]

            print(
                f"  {dataset.upper():8} "
                f"Inside: "
                f"{counts['inside_window_events']:>6,} | "
                f"Outside: "
                f"{counts['outside_window_events']:>8,} | "
                f"Same-day outside: "
                f"{counts['same_day_outside_events']:>6,}"
            )

    print("\nREPORTS")
    print("=" * 75)

    print(f"Scenario summary: {summary_path}")
    print(f"Event details:    {event_path}")

    print("\nAudit completed successfully.")


if __name__ == "__main__":
    main()
    