import pandas as pd
import numpy as np

from pathlib import Path

from testing.cert_evaluation.ground_truth import load_ground_truth


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = BASE_DIR / "testing" / "reports"

EVENT_FILE = REPORT_DIR / "temporal_event_details.csv"
COMPARISON_FILE = REPORT_DIR / "temporal_behavior_comparison.csv"

OUTPUT_FILE = REPORT_DIR / "temporal_quality_audit.csv"
SUMMARY_FILE = REPORT_DIR / "temporal_quality_audit_summary.csv"

LOOKBACK_DAYS = 30
FOLLOWUP_DAYS = 30

DATASETS = ["logon", "file", "device"]

PERIODS = [
    "PRE_INCIDENT",
    "INCIDENT",
    "POST_INCIDENT",
]

SMALL_SAMPLE_THRESHOLD = 10


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs():

    for path in [EVENT_FILE, COMPARISON_FILE]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    events = pd.read_csv(
        EVENT_FILE,
        low_memory=False,
    )

    comparison = pd.read_csv(
        COMPARISON_FILE,
        low_memory=False,
    )

    events["scenario_id"] = (
        events["scenario_id"].astype(str).str.strip()
    )

    comparison["scenario_id"] = (
        comparison["scenario_id"].astype(str).str.strip()
    )

    required_event_columns = [
        "scenario_id",
        "user",
        "dataset",
        "period",
        "event_timestamp",
        "incident_start",
        "incident_end",
    ]

    required_comparison_columns = [
        "scenario_id",
        "user",
        "dataset",
        "period",
        "total_events",
        "active_days",
        "window_duration_days",
        "events_per_day",
        "events_per_active_day",
        "active_day_percentage",
    ]

    missing_events = [
        col for col in required_event_columns
        if col not in events.columns
    ]

    missing_comparison = [
        col for col in required_comparison_columns
        if col not in comparison.columns
    ]

    if missing_events or missing_comparison:
        raise ValueError(
            f"Missing event columns: {missing_events}\n"
            f"Missing comparison columns: {missing_comparison}\n"
            f"Event columns found: {events.columns.tolist()}\n"
            f"Comparison columns found: {comparison.columns.tolist()}"
        )

    for column in [
        "event_timestamp",
        "incident_start",
        "incident_end",
    ]:

        events[column] = pd.to_datetime(
            events[column],
            errors="coerce",
        )

    if events[
        ["event_timestamp", "incident_start", "incident_end"]
    ].isna().any().any():

        raise ValueError(
            "Invalid timestamps in temporal event report."
        )

    return events, comparison

# ============================================================
# TELEMETRY AVAILABILITY
# ============================================================

def get_availability():

    availability = {}

    for scenario in load_ground_truth():

        scenario_id = str(scenario["scenario_id"])

        availability[scenario_id] = {}

        for dataset in [
            "logon",
            "file",
            "device",
            "email",
            "http",
        ]:

            path = BASE_DIR / "data" / f"{dataset}.csv"

            availability[scenario_id][dataset] = path.is_file()

    return availability


# ============================================================
# WINDOW CALCULATION
# ============================================================

def get_window_bounds(scenario, period):

    start = pd.Timestamp(scenario["start"])
    end = pd.Timestamp(scenario["end"])

    if period == "PRE_INCIDENT":

        return (
            start - pd.Timedelta(days=LOOKBACK_DAYS),
            start,
        )

    if period == "INCIDENT":

        return start, end

    if period == "POST_INCIDENT":

        return (
            end,
            end + pd.Timedelta(days=FOLLOWUP_DAYS),
        )

    raise ValueError(f"Unknown period: {period}")


def get_window_duration(start, end):

    return (
        end - start
    ).total_seconds() / 86400


def get_calendar_dates(start, end, period):

    if period == "PRE_INCIDENT":

        last_included = (
            end - pd.Timedelta(nanoseconds=1)
        )

        return (
            last_included.normalize()
            - start.normalize()
        ).days + 1

    if period == "INCIDENT":

        return (
            end.normalize()
            - start.normalize()
        ).days + 1

    if period == "POST_INCIDENT":

        first_included = (
            start + pd.Timedelta(nanoseconds=1)
        )

        return (
            end.normalize()
            - first_included.normalize()
        ).days + 1

    raise ValueError(f"Unknown period: {period}")


# ============================================================
# EVENT COUNT AUDIT
# ============================================================

def audit_event_counts(events, comparison, availability):

    records = []

    for _, row in comparison.iterrows():

        scenario_id = str(row["scenario_id"])
        dataset = row["dataset"]
        period = row["period"]

        source_available = availability[
            scenario_id
        ][dataset]

        selected = events[
            (
                events["scenario_id"] == scenario_id
            )
            &
            (
                events["dataset"] == dataset
            )
            &
            (
                events["period"] == period
            )
        ]

        actual_count = len(selected)

        reported_count = row["total_events"]

        if not source_available:

            status = (
                "UNAVAILABLE_SOURCE"
                if pd.isna(reported_count)
                else "ERROR_UNAVAILABLE_SOURCE_HAS_COUNT"
            )

        elif pd.isna(reported_count):

            status = "ERROR_AVAILABLE_SOURCE_MISSING_COUNT"

        elif int(reported_count) != actual_count:

            status = "COUNT_MISMATCH"

        else:

            status = "PASS"

        records.append({
            "scenario_id": scenario_id,
            "user": row["user"],
            "dataset": dataset,
            "period": period,
            "source_available": source_available,
            "event_file_count": actual_count,
            "comparison_report_count": reported_count,
            "count_audit_status": status,
        })

    return pd.DataFrame(records)


# ============================================================
# TEMPORAL BOUNDARY AUDIT
# ============================================================

def audit_temporal_boundaries(events):

    records = []

    for _, row in events.iterrows():

        timestamp = row["event_timestamp"]
        start = row["incident_start"]
        end = row["incident_end"]

        if row["period"] == "PRE_INCIDENT":

            valid = (
                start - pd.Timedelta(days=LOOKBACK_DAYS)
                <= timestamp < start
            )

        elif row["period"] == "INCIDENT":

            valid = start <= timestamp <= end

        elif row["period"] == "POST_INCIDENT":

            valid = (
                end < timestamp
                <= end + pd.Timedelta(days=FOLLOWUP_DAYS)
            )

        else:

            valid = False

        if not valid:

            records.append({
                "scenario_id": row["scenario_id"],
                "user": row["user"],
                "dataset": row["dataset"],
                "event_timestamp": timestamp,
                "period": row["period"],
                "boundary_status": "FAIL",
            })

    return pd.DataFrame(records)


# ============================================================
# ACTIVE-DAY AUDIT
# ============================================================

def audit_active_days(events, comparison):

    records = []

    for _, row in comparison.iterrows():

        selected = events[
            (
                events["scenario_id"]
                == row["scenario_id"]
            )
            &
            (
                events["dataset"]
                == row["dataset"]
            )
            &
            (
                events["period"]
                == row["period"]
            )
        ]

        if not row["source_available"]:

            status = (
                "UNAVAILABLE_SOURCE"
                if pd.isna(row["active_days"])
                else "ERROR_UNAVAILABLE_SOURCE_HAS_ACTIVE_DAYS"
            )

            actual_active_days = np.nan

        else:

            actual_active_days = (
                selected["event_timestamp"]
                .dt.normalize()
                .nunique()
            )

            status = (
                "PASS"
                if actual_active_days == row["active_days"]
                else "ACTIVE_DAY_MISMATCH"
            )

        records.append({
            "scenario_id": row["scenario_id"],
            "user": row["user"],
            "dataset": row["dataset"],
            "period": row["period"],
            "reported_active_days": row["active_days"],
            "actual_active_days": actual_active_days,
            "active_day_status": status,
        })

    return pd.DataFrame(records)


# ============================================================
# DURATION AND RATE AUDIT
# ============================================================

def audit_durations_and_rates(comparison):

    records = []

    scenarios = load_ground_truth()

    scenario_lookup = {
        str(scenario["scenario_id"]): scenario
        for scenario in scenarios
    }

    for _, row in comparison.iterrows():

        scenario = scenario_lookup[
            str(row["scenario_id"])
        ]

        start, end = get_window_bounds(
            scenario,
            row["period"],
        )

        expected_duration = get_window_duration(
            start,
            end,
        )

        expected_calendar_dates = get_calendar_dates(
            start,
            end,
            row["period"],
        )

        duration_match = np.isclose(
            expected_duration,
            row["window_duration_days"],
            atol=0.0001,
        )

        if row["source_available"]:

            expected_rate = (
                row["total_events"] / expected_duration
            )

            rate_match = np.isclose(
                expected_rate,
                row["events_per_day"],
                atol=0.0001,
            )

            if row["active_days"] > 0:

                expected_active_rate = (
                    row["total_events"] / row["active_days"]
                )

                active_rate_match = np.isclose(
                    expected_active_rate,
                    row["events_per_active_day"],
                    atol=0.0001,
                )

            else:

                active_rate_match = pd.isna(
                    row["events_per_active_day"]
                )

            expected_active_percentage = (
                row["active_days"]
                / expected_calendar_dates
            ) * 100

            percentage_match = np.isclose(
                expected_active_percentage,
                row["active_day_percentage"],
                atol=0.01,
            )

        else:

            rate_match = True
            active_rate_match = True
            percentage_match = True

        records.append({
            "scenario_id": row["scenario_id"],
            "user": row["user"],
            "dataset": row["dataset"],
            "period": row["period"],
            "expected_duration_days": expected_duration,
            "reported_duration_days": row["window_duration_days"],
            "duration_status": (
                "PASS" if duration_match else "FAIL"
            ),
            "expected_calendar_dates": expected_calendar_dates,
            "expected_events_per_day": (
                row["total_events"] / expected_duration
                if row["source_available"]
                else np.nan
            ),
            "rate_status": (
                "PASS" if rate_match else "FAIL"
            ),
            "active_rate_status": (
                "PASS" if active_rate_match else "FAIL"
            ),
            "active_percentage_status": (
                "PASS" if percentage_match else "FAIL"
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# INTERPRETATION WARNINGS
# ============================================================

def build_interpretation_warnings(
    comparison,
    availability,
):

    records = []

    for _, row in comparison.iterrows():

        scenario_id = str(row["scenario_id"])
        dataset = row["dataset"]
        period = row["period"]

        if not availability[scenario_id][dataset]:

            warning = "SOURCE_UNAVAILABLE"

        elif (
            period == "INCIDENT"
            and row["window_duration_days"] < 1
        ):

            warning = "VERY_SHORT_INCIDENT_WINDOW"

        elif row["total_events"] == 0:

            warning = "ZERO_OBSERVED_EVENTS_NOT_PROOF_OF_INACTIVITY"

        elif row["total_events"] < SMALL_SAMPLE_THRESHOLD:

            warning = "SMALL_EVENT_SAMPLE"

        else:

            warning = "NO_AUTOMATIC_WARNING"

        records.append({
            "scenario_id": scenario_id,
            "user": row["user"],
            "dataset": dataset,
            "period": period,
            "event_count": row["total_events"],
            "interpretation_warning": warning,
        })

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("AEGIS - TEMPORAL COMPARISON QUALITY AUDIT")
    print("=" * 75)

    events, comparison = load_inputs()

    availability = get_availability()

    print(f"\nTemporal event records: {len(events):,}")
    print(f"Comparison rows: {len(comparison):,}")

    count_audit = audit_event_counts(
        events,
        comparison,
        availability,
    )

    boundary_audit = audit_temporal_boundaries(
        events
    )

    active_day_audit = audit_active_days(
        events,
        comparison,
    )

    duration_audit = audit_durations_and_rates(
        comparison
    )

    warnings = build_interpretation_warnings(
        comparison,
        availability,
    )

    # Combine all row-level audit results.
    audit = (
        count_audit
        .merge(
            active_day_audit,
            on=[
                "scenario_id",
                "user",
                "dataset",
                "period",
            ],
        )
        .merge(
            duration_audit,
            on=[
                "scenario_id",
                "user",
                "dataset",
                "period",
            ],
        )
        .merge(
            warnings,
            on=[
                "scenario_id",
                "user",
                "dataset",
                "period",
            ],
        )
    )

    # --------------------------------------------------------
    # SAVE DETAILED AUDIT
    # --------------------------------------------------------

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    boundary_audit.to_csv(
        REPORT_DIR / "temporal_boundary_failures.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_records = []

    audit_columns = [
        "count_audit_status",
        "active_day_status",
        "duration_status",
        "rate_status",
        "active_rate_status",
        "active_percentage_status",
    ]

    for column in audit_columns:

        counts = audit[column].value_counts(
            dropna=False
        )

        for status, count in counts.items():

            summary_records.append({
                "audit_component": column,
                "status": status,
                "record_count": count,
            })

    summary_df = pd.DataFrame(
        summary_records
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\nAUDIT RESULTS")
    print("=" * 75)

    for column in audit_columns:

        print(f"\n{column}:")

        print(
            audit[column]
            .value_counts(dropna=False)
            .to_string()
        )

    print("\nTEMPORAL BOUNDARY FAILURES")

    print(
        f"Invalid event records: {len(boundary_audit)}"
    )

    print("\nSCENARIO 5 INCIDENT REVIEW")
    
    scenario5 = audit[
        (audit["scenario_id"] == "5")
        & (audit["period"] == "INCIDENT")
    ]
    
    print(
        scenario5[
            [
                "dataset",
                "source_available",
                "event_file_count",
                "comparison_report_count",
                "expected_duration_days",
                "interpretation_warning",
            ]
        ].to_string(index=False)
    )

    print("\nSMALL SAMPLE AND INTERPRETATION WARNINGS")

    print(
        warnings[
            warnings["interpretation_warning"]
            != "NO_AUTOMATIC_WARNING"
        ].to_string(index=False)
    )

    print("\nREPORTS")
    print("=" * 75)

    print(OUTPUT_FILE)
    print(SUMMARY_FILE)
    print(
        REPORT_DIR / "temporal_boundary_failures.csv"
    )

    print("\nTemporal quality audit completed.")


if __name__ == "__main__":
    main()