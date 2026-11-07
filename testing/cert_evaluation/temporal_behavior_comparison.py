import pandas as pd
import numpy as np

from pathlib import Path

from testing.cert_evaluation.ground_truth import load_ground_truth


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REPORT_DIR = BASE_DIR / "testing" / "reports"

INPUT_FILE = REPORT_DIR / "temporal_event_details.csv"

OUTPUT_FILE = REPORT_DIR / "temporal_behavior_comparison.csv"

SUMMARY_FILE = (
    REPORT_DIR / "temporal_behavior_comparison_summary.csv"
)

LOOKBACK_DAYS = 30
FOLLOWUP_DAYS = 30

PERIODS = [
    "PRE_INCIDENT",
    "INCIDENT",
    "POST_INCIDENT",
]

DATASETS = [
    "logon",
    "file",
    "device",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_temporal_events():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )
# Normalize scenario IDs to strings for ground-truth matching.
    df["scenario_id"] = df["scenario_id"].astype(str).str.strip()
    required_columns = [
        "scenario_id",
        "scenario",
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
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["event_timestamp"] = pd.to_datetime(
        df["event_timestamp"],
        errors="coerce",
    )

    df["incident_start"] = pd.to_datetime(
        df["incident_start"],
        errors="coerce",
    )

    df["incident_end"] = pd.to_datetime(
        df["incident_end"],
        errors="coerce",
    )

    if df[
        [
            "event_timestamp",
            "incident_start",
            "incident_end",
        ]
    ].isna().any().any():

        raise ValueError(
            "Invalid timestamps found in temporal event file."
        )

    return df


# ============================================================
# LOAD GROUND TRUTH AND SOURCE AVAILABILITY
# ============================================================

def load_source_availability():

    scenarios = load_ground_truth()

    availability = {}

    for scenario in scenarios:

        scenario_id = scenario["scenario_id"]

        availability[scenario_id] = {}

        for dataset in ["logon", "file", "device", "email", "http"]:

            source_path = BASE_DIR / "data" / f"{dataset}.csv"

            availability[scenario_id][dataset] = (
                source_path.exists()
            )

    return availability


# ============================================================
# WINDOW DURATION
# ============================================================

def calculate_window_days(
    incident_start,
    incident_end,
    period,
):

    if period == "PRE_INCIDENT":

        start = (
            incident_start
            - pd.Timedelta(days=LOOKBACK_DAYS)
        )

        end = incident_start

    elif period == "INCIDENT":

        start = incident_start
        end = incident_end

    elif period == "POST_INCIDENT":

        start = incident_end

        end = (
            incident_end
            + pd.Timedelta(days=FOLLOWUP_DAYS)
        )

    else:
        raise ValueError(
            f"Unknown period: {period}"
        )

    duration = (
        end - start
    ).total_seconds() / 86400

    if duration <= 0:
        raise ValueError(
            f"Invalid duration for period {period}"
        )

    return duration


# ============================================================
# GENERATE COMPARISON
# ============================================================

def generate_comparison(
    event_df,
    availability,
):

    records = []

    scenarios = load_ground_truth()

    for scenario in scenarios:

        scenario_id = scenario["scenario_id"]
        employee = scenario["insider"]

        incident_start = scenario["start"]
        incident_end = scenario["end"]

        for dataset in DATASETS:

            source_available = availability[
                scenario_id
            ][dataset]

            employee_events = event_df[
                (
                    event_df["scenario_id"]
                    == scenario_id
                )
                &
                (
                    event_df["dataset"]
                    == dataset
                )
            ]

            for period in PERIODS:

                duration_days = calculate_window_days(
                    incident_start,
                    incident_end,
                    period,
                )

                if not source_available:

                    records.append({
                        "scenario_id": scenario_id,
                        "scenario": scenario["scenario"],
                        "user": employee,
                        "dataset": dataset,
                        "period": period,
                        "source_available": False,
                        "window_duration_days": duration_days,
                        "total_events": np.nan,
                        "active_days": np.nan,
                        "events_per_day": np.nan,
                        "events_per_active_day": np.nan,
                        "active_day_percentage": np.nan,
                    })

                    continue

                period_events = employee_events[
                    employee_events["period"] == period
                ].copy()

                total_events = len(period_events)

                active_days = (
                    period_events["event_timestamp"]
                    .dt.date
                    .nunique()
                )

                events_per_day = (
                    total_events / duration_days
                )

                events_per_active_day = (
                    total_events / active_days
                    if active_days > 0
                    else np.nan
                )

                                # Count calendar dates touched by the temporal window.
                if period == "PRE_INCIDENT":
                    window_start = (
                        incident_start
                        - pd.Timedelta(days=LOOKBACK_DAYS)
                    )
                    window_end = (
                        incident_start
                        - pd.Timedelta(seconds=1)
                    )

                elif period == "INCIDENT":
                    window_start = incident_start
                    window_end = incident_end

                else:
                    window_start = (
                        incident_end
                        + pd.Timedelta(seconds=1)
                    )
                    window_end = (
                        incident_end
                        + pd.Timedelta(days=FOLLOWUP_DAYS)
                    )

                calendar_days = (
                        (pd.Timestamp(window_end).normalize()
                             - pd.Timestamp(window_start).normalize()).days
                                + 1
                        )

                active_day_percentage = (
                    active_days / calendar_days
                ) * 100

                records.append({
                    "scenario_id": scenario_id,
                    "scenario": scenario["scenario"],
                    "user": employee,
                    "dataset": dataset,
                    "period": period,
                    "source_available": True,
                    "window_duration_days": round(
                        duration_days, 4
                    ),
                    "total_events": total_events,
                    "active_days": active_days,
                    "events_per_day": round(
                        events_per_day, 4
                    ),
                    "events_per_active_day": (
                        round(events_per_active_day, 4)
                        if not np.isnan(events_per_active_day)
                        else np.nan
                    ),
                    "active_day_percentage": round(
                        active_day_percentage, 4
                    ),
                })

    return pd.DataFrame(records)


# ============================================================
# CALCULATE PERIOD-TO-PERIOD CHANGES
# ============================================================

def calculate_rate_changes(comparison_df):

    records = []

    group_columns = [
        "scenario_id",
        "scenario",
        "user",
        "dataset",
    ]

    for keys, group in comparison_df.groupby(
        group_columns,
        dropna=False,
    ):

        scenario_id, scenario, user, dataset = keys

        indexed = group.set_index("period")

        pre = indexed.loc["PRE_INCIDENT"]
        incident = indexed.loc["INCIDENT"]
        post = indexed.loc["POST_INCIDENT"]

        def percentage_change(before, after):

            if pd.isna(before) or pd.isna(after):
                return np.nan

            if before == 0:

                if after == 0:
                    return 0.0

                return np.nan

            return (
                (after - before) / before
            ) * 100

        pre_to_incident = percentage_change(
            pre["events_per_day"],
            incident["events_per_day"],
        )

        incident_to_post = percentage_change(
            incident["events_per_day"],
            post["events_per_day"],
        )

        pre_to_post = percentage_change(
            pre["events_per_day"],
            post["events_per_day"],
        )

        records.append({
            "scenario_id": scenario_id,
            "scenario": scenario,
            "user": user,
            "dataset": dataset,
            "pre_to_incident_change_pct": (
                round(pre_to_incident, 2)
                if not pd.isna(pre_to_incident)
                else np.nan
            ),
            "incident_to_post_change_pct": (
                round(incident_to_post, 2)
                if not pd.isna(incident_to_post)
                else np.nan
            ),
            "pre_to_post_change_pct": (
                round(pre_to_post, 2)
                if not pd.isna(pre_to_post)
                else np.nan
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# VALIDATION
# ============================================================

def validate_comparison(
    event_df,
    comparison_df,
):

    expected_rows = (
        len(load_ground_truth())
        * len(DATASETS)
        * len(PERIODS)
    )

    if len(comparison_df) != expected_rows:

        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"got {len(comparison_df)}"
        )

    available = comparison_df[
        comparison_df["source_available"]
    ]

    if available[
        [
            "total_events",
            "active_days",
            "events_per_day",
            "active_day_percentage",
        ]
    ].isna().any().any():

        raise ValueError(
            "Available source contains missing metrics."
        )

    if (
        comparison_df["total_events"].dropna() < 0
    ).any():

        raise ValueError(
            "Negative event count detected."
        )

    if (
        comparison_df["active_days"].dropna() < 0
    ).any():

        raise ValueError(
            "Negative active-day count detected."
        )

    if not (
        comparison_df["active_days"].dropna()
        <= comparison_df["window_duration_days"].max()
    ).all():

        raise ValueError(
            "Active days exceed maximum window duration."
        )

    # Check that exported event counts match comparison counts.
    for _, row in available.iterrows():

        actual = event_df[
            (
                event_df["scenario_id"]
                == row["scenario_id"]
            )
            &
            (
                event_df["dataset"]
                == row["dataset"]
            )
            &
            (
                event_df["period"]
                == row["period"]
            )
        ]

        if len(actual) != row["total_events"]:

            raise ValueError(
                "Event count mismatch detected."
            )

    print("\nComparison validation: PASS")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("AEGIS - TEMPORAL BEHAVIOUR COMPARISON")
    print("=" * 75)

    print("\nLoading temporal event records...")

    event_df = load_temporal_events()

    print(
        f"Loaded {len(event_df):,} event records."
    )

    print("\nChecking telemetry availability...")

    availability = load_source_availability()

    comparison_df = generate_comparison(
        event_df,
        availability,
    )

    changes_df = calculate_rate_changes(
        comparison_df
    )

    validate_comparison(
        event_df,
        comparison_df,
    )

    comparison_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    changes_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("\nTEMPORAL BEHAVIOUR COMPARISON")
    print("=" * 75)

    print(
        comparison_df[
            [
                "scenario",
                "user",
                "dataset",
                "period",
                "window_duration_days",
                "total_events",
                "active_days",
                "events_per_day",
                "events_per_active_day",
                "active_day_percentage",
            ]
        ].to_string(index=False)
    )

    print("\nPERIOD-TO-PERIOD RATE CHANGES")
    print("=" * 75)

    print(
        changes_df.to_string(index=False)
    )

    print("\nOUTPUT FILES")
    print("=" * 75)

    print(OUTPUT_FILE)
    print(SUMMARY_FILE)

    print("\nTemporal behaviour comparison completed.")


if __name__ == "__main__":
    main()