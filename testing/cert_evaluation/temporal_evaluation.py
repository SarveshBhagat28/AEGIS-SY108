import pandas as pd

from pathlib import Path

from testing.cert_evaluation.ground_truth import load_ground_truth


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

REPORT_DIR = BASE_DIR / "testing" / "reports"

CHUNK_SIZE = 200_000
LOOKBACK_DAYS = 30
FOLLOWUP_DAYS = 30

TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"

AVAILABLE_DATASETS = ["logon", "file", "device"]
ALL_DATASETS = ["logon", "file", "device", "email", "http"]

OPTIONAL_COLUMNS = [
    "id",
    "pc",
    "activity",
    "filename",
    "file_tree",
    "to_removable_media",
    "from_removable_media",
]


# ============================================================
# INITIALIZATION
# ============================================================

SCENARIOS = load_ground_truth()

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEMPORAL CLASSIFICATION
# ============================================================

def classify_period(timestamp, scenario):

    start = scenario["start"]
    end = scenario["end"]

    pre_start = start - pd.Timedelta(days=LOOKBACK_DAYS)
    post_end = end + pd.Timedelta(days=FOLLOWUP_DAYS)

    if pre_start <= timestamp < start:
        return "PRE_INCIDENT"

    if start <= timestamp <= end:
        return "INCIDENT"

    if end < timestamp <= post_end:
        return "POST_INCIDENT"

    return None


# ============================================================
# SOURCE AVAILABILITY
# ============================================================

def check_source_availability():

    availability = {}

    for dataset in ALL_DATASETS:

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

    file_path = DATA_DIR / f"{dataset}.csv"

    if not file_path.is_file():
        return None

    print(f"\nProcessing {dataset.upper()}...")

    header = pd.read_csv(file_path, nrows=0)

    available_columns = list(header.columns)

    required_columns = ["date", "user"]

    missing = [
        column
        for column in required_columns
        if column not in available_columns
    ]

    if missing:
        raise ValueError(
            f"{dataset}.csv missing columns: {missing}"
        )

    optional_columns = [
        column
        for column in OPTIONAL_COLUMNS
        if column in available_columns
    ]

    use_columns = required_columns + optional_columns

    # Create per-scenario temporal boundaries.
    boundaries = {}

    for scenario in SCENARIOS:

        boundaries[scenario["insider"]] = {
            "scenario": scenario,
            "pre_start": (
                scenario["start"]
                - pd.Timedelta(days=LOOKBACK_DAYS)
            ),
            "post_end": (
                scenario["end"]
                + pd.Timedelta(days=FOLLOWUP_DAYS)
            ),
        }

    event_records = []

    reader = pd.read_csv(
        file_path,
        usecols=use_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    total_chunks = 0
    invalid_timestamps = 0
    matched_events = 0

    for chunk in reader:

        total_chunks += 1

        chunk["user"] = (
            chunk["user"]
            .astype(str)
            .str.strip()
        )

        # Retain only the five known insider identities.
        chunk = chunk[
            chunk["user"].isin(boundaries)
        ].copy()

        if chunk.empty:
            continue

        chunk["timestamp"] = pd.to_datetime(
            chunk["date"],
            format=TIMESTAMP_FORMAT,
            errors="coerce",
        )

        invalid_timestamps += int(
            chunk["timestamp"].isna().sum()
        )

        chunk = chunk[
            chunk["timestamp"].notna()
        ].copy()

        if chunk.empty:
            continue

        for employee, details in boundaries.items():

            employee_events = chunk[
                chunk["user"] == employee
            ]

            if employee_events.empty:
                continue

            scenario = details["scenario"]

            relevant = employee_events[
                (
                    employee_events["timestamp"]
                    >= details["pre_start"]
                )
                &
                (
                    employee_events["timestamp"]
                    <= details["post_end"]
                )
            ]

            if relevant.empty:
                continue

            for _, row in relevant.iterrows():

                period = classify_period(
                    row["timestamp"],
                    scenario,
                )

                if period is None:
                    continue

                record = {
                    "scenario_id": scenario["scenario_id"],
                    "scenario": scenario["scenario"],
                    "user": employee,
                    "dataset": dataset,
                    "event_timestamp": row["timestamp"],
                    "event_id": (
                        row["id"]
                        if "id" in optional_columns
                        else None
                    ),
                    "incident_start": scenario["start"],
                    "incident_end": scenario["end"],
                    "period": period,
                }

                # Preserve all available source-specific fields.
                for column in optional_columns:

                    if column == "id":
                        continue

                    record[column] = row[column]

                event_records.append(record)

            matched_events += len(relevant)

        if total_chunks % 5 == 0:
            print(
                f"  Processed {total_chunks} chunks | "
                f"Matched events: {matched_events:,}"
            )

    print(
        f"  Completed {dataset.upper()} | "
        f"Chunks: {total_chunks:,} | "
        f"Events retained: {matched_events:,} | "
        f"Invalid timestamps for known insiders: "
        f"{invalid_timestamps:,}"
    )

    return {
        "events": event_records,
        "invalid_timestamps": invalid_timestamps,
        "chunks_processed": total_chunks,
    }


# ============================================================
# TEMPORAL SUMMARY
# ============================================================

def build_temporal_summary(
    event_df,
    availability,
    processing_results,
):

    records = []

    periods = [
        "PRE_INCIDENT",
        "INCIDENT",
        "POST_INCIDENT",
    ]

    for scenario in SCENARIOS:

        for dataset in ALL_DATASETS:

            source_available = availability[
                dataset
            ]["available"]

            for period in periods:

                event_count = None

                if source_available:

                    if event_df.empty:
                        event_count = 0

                    else:

                        selected = event_df[
                            (
                                event_df["scenario_id"]
                                == scenario["scenario_id"]
                            )
                            &
                            (
                                event_df["dataset"]
                                == dataset
                            )
                            &
                            (
                                event_df["period"]
                                == period
                            )
                        ]

                        event_count = len(selected)

                records.append({
                    "scenario_id": scenario["scenario_id"],
                    "scenario": scenario["scenario"],
                    "user": scenario["insider"],
                    "dataset": dataset,
                    "period": period,
                    "source_file_available": source_available,
                    "observed_event_count": event_count,
                    "incident_start": scenario["start"],
                    "incident_end": scenario["end"],
                })

    return pd.DataFrame(records)


# ============================================================
# VALIDATION
# ============================================================

def validate_event_output(event_df):

    if event_df.empty:
        print("\nWARNING: No temporal events were exported.")
        return

    required_columns = [
        "scenario_id",
        "user",
        "dataset",
        "event_timestamp",
        "incident_start",
        "incident_end",
        "period",
    ]

    missing = [
        column
        for column in required_columns
        if column not in event_df.columns
    ]

    if missing:
        raise ValueError(
            f"Event output missing columns: {missing}"
        )

    allowed_periods = {
        "PRE_INCIDENT",
        "INCIDENT",
        "POST_INCIDENT",
    }

    invalid_periods = set(
        event_df["period"].dropna()
    ) - allowed_periods

    if invalid_periods:
        raise ValueError(
            f"Unexpected temporal periods: {invalid_periods}"
        )

    if event_df.duplicated(
        subset=["dataset", "event_id"]
    ).any():

        raise ValueError(
            "Duplicate dataset/event ID records detected."
        )

    for _, row in event_df.iterrows():

        timestamp = row["event_timestamp"]

        start = row["incident_start"]
        end = row["incident_end"]

        if row["period"] == "INCIDENT":
            valid = start <= timestamp <= end

        elif row["period"] == "PRE_INCIDENT":
            valid = (
                start - pd.Timedelta(days=LOOKBACK_DAYS)
                <= timestamp < start
            )

        else:
            valid = (
                end < timestamp
                <= end + pd.Timedelta(days=FOLLOWUP_DAYS)
            )

        if not valid:
            raise ValueError(
                "Temporal boundary validation failed: "
                f"{row['user']} {timestamp} "
                f"{row['period']}"
            )

    print("\nEvent-level temporal validation: PASS")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("AEGIS - CERT TEMPORAL EVALUATION")
    print("=" * 75)

    print(f"Pre-incident window: {LOOKBACK_DAYS} days")
    print(f"Post-incident window: {FOLLOWUP_DAYS} days")

    availability = check_source_availability()

    print("\nSOURCE AVAILABILITY")
    print("-" * 75)

    for dataset, details in availability.items():

        status = (
            "AVAILABLE"
            if details["available"]
            else "UNAVAILABLE"
        )

        print(f"{dataset.upper():10} {status}")

    all_events = []
    processing_results = {}

    for dataset in AVAILABLE_DATASETS:

        result = process_dataset(dataset)

        if result is None:
            print(
                f"WARNING: {dataset}.csv is unavailable."
            )
            continue

        processing_results[dataset] = result

        all_events.extend(result["events"])

    if all_events:

        event_df = pd.DataFrame(all_events)

        event_df = event_df.sort_values(
            [
                "scenario_id",
                "event_timestamp",
                "dataset",
            ]
        ).reset_index(drop=True)

    else:

        event_df = pd.DataFrame()

    validate_event_output(event_df)

    summary_df = build_temporal_summary(
        event_df,
        availability,
        processing_results,
    )

    event_path = (
        REPORT_DIR / "temporal_event_details.csv"
    )

    summary_path = (
        REPORT_DIR / "temporal_evaluation_summary.csv"
    )

    event_df.to_csv(
        event_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    print("\nTEMPORAL EVENT COUNTS")
    print("=" * 75)

    if not event_df.empty:

        grouped = (
            event_df
            .groupby(
                ["scenario", "user", "dataset", "period"]
            )
            .size()
            .reset_index(name="event_count")
        )

        print(
            grouped.to_string(index=False)
        )

    print("\nUNAVAILABLE REQUIRED TELEMETRY")
    print("=" * 75)

    for scenario in SCENARIOS:

        missing = [
            dataset
            for dataset in scenario["required_telemetry"]
            if not availability[dataset]["available"]
        ]

        if missing:

            print(
                f"{scenario['scenario']} "
                f"({scenario['insider']}): "
                f"{', '.join(missing)}"
            )

        else:

            print(
                f"{scenario['scenario']} "
                f"({scenario['insider']}): "
                f"All required source files available"
            )

    print("\nREPORTS")
    print("=" * 75)

    print(f"Event details: {event_path}")
    print(f"Summary:       {summary_path}")

    print(
        f"\nTotal temporal event records: "
        f"{len(event_df):,}"
    )

    print("Temporal evaluation preparation completed.")


if __name__ == "__main__":
    main()
    