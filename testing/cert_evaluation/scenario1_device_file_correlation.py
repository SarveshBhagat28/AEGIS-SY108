
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE_EVENTS_PATH = (
    PROJECT_ROOT
    / "testing"
    / "reports"
    / "scenario1_acm2278_file_events.csv"
)

DEVICE_EVENTS_PATH = (
    PROJECT_ROOT
    / "testing"
    / "reports"
    / "incident_event_details.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "testing" / "reports"

EMPLOYEE = "ACM2278"
TIME_WINDOWS_MINUTES = [1, 5, 10, 30]

INCIDENT_START = pd.Timestamp("2010-08-18 21:47:42")
INCIDENT_END = pd.Timestamp("2010-08-24 03:48:51")


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    if not FILE_EVENTS_PATH.exists():
        raise FileNotFoundError(FILE_EVENTS_PATH)

    if not DEVICE_EVENTS_PATH.exists():
        raise FileNotFoundError(DEVICE_EVENTS_PATH)

    files = pd.read_csv(FILE_EVENTS_PATH, low_memory=False)
    devices = pd.read_csv(DEVICE_EVENTS_PATH, low_memory=False)

    required_file_columns = {
        "user", "event_timestamp", "pc", "activity"
    }

    required_device_columns = {
        "user", "timestamp", "pc", "activity"
    }

    if not required_file_columns.issubset(files.columns):
        raise ValueError(
            f"File dataset missing columns: "
            f"{required_file_columns - set(files.columns)}"
        )

    if not required_device_columns.issubset(devices.columns):
        raise ValueError(
            f"Device dataset missing columns: "
            f"{required_device_columns - set(devices.columns)}"
        )

    files["event_timestamp"] = pd.to_datetime(
        files["event_timestamp"], errors="coerce"
    )

    devices["timestamp"] = pd.to_datetime(
        devices["timestamp"], errors="coerce"
    )

    files = files[
        (files["user"] == EMPLOYEE)
        & files["event_timestamp"].notna()
    ].copy()

    devices = devices[
        (devices["user"] == EMPLOYEE)
        & devices["timestamp"].notna()
        & devices["activity"].isin(["Connect", "Disconnect"])
    ].copy()

    return files, devices


# ============================================================
# PERIOD CLASSIFICATION
# ============================================================

def classify_period(timestamp):
    if INCIDENT_START <= timestamp <= INCIDENT_END:
        return "INCIDENT"

    if timestamp < INCIDENT_START:
        return "PRE"

    return "POST"


# ============================================================
# DEVICE-FILE CORRELATION
# ============================================================

def correlate_events(files, devices):

    if devices.empty:
        raise ValueError(
            "No ACM2278 device Connect/Disconnect events found."
        )

    devices = devices.sort_values("timestamp").reset_index(drop=True)

    results = []

    for _, file_event in files.iterrows():

        file_time = file_event["event_timestamp"]
        file_pc = file_event["pc"]

        same_pc_devices = devices[
            devices["pc"] == file_pc
        ].copy()

        if same_pc_devices.empty:
            nearest_device = None
            nearest_time = pd.NaT
            nearest_activity = "NO_MATCHING_PC"
            difference_minutes = None

        else:
            same_pc_devices["time_difference_seconds"] = (
                same_pc_devices["timestamp"] - file_time
            ).dt.total_seconds().abs()

            nearest_device = same_pc_devices.loc[
                same_pc_devices["time_difference_seconds"].idxmin()
            ]

            nearest_time = nearest_device["timestamp"]
            nearest_activity = nearest_device["activity"]

            difference_minutes = (
                nearest_device["time_difference_seconds"] / 60
            )

        row = {
            "user": EMPLOYEE,
            "pc": file_pc,
            "file_timestamp": file_time,
            "period": classify_period(file_time),
            "file_activity": file_event["activity"],
            "nearest_device_timestamp": nearest_time,
            "nearest_device_activity": nearest_activity,
            "time_difference_minutes": difference_minutes,
        }

        for window in TIME_WINDOWS_MINUTES:
            row[f"within_{window}_minutes"] = (
                difference_minutes is not None
                and difference_minutes <= window
            )

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# REPORTING
# ============================================================

def generate_reports(results):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_path = (
        OUTPUT_DIR / "scenario1_device_file_correlation_details.csv"
    )

    summary_path = (
        OUTPUT_DIR / "scenario1_device_file_correlation_summary.csv"
    )

    results.to_csv(detail_path, index=False)

    summary_rows = []

    for period in ["PRE", "INCIDENT", "POST"]:

        period_data = results[
            results["period"] == period
        ]

        for activity in sorted(results["file_activity"].dropna().unique()):

            activity_data = period_data[
                period_data["file_activity"] == activity
            ]

            row = {
                "period": period,
                "file_activity": activity,
                "total_file_events": len(activity_data),
            }

            for window in TIME_WINDOWS_MINUTES:
                column = f"within_{window}_minutes"

                row[f"events_within_{window}_minutes"] = int(
                    activity_data[column].sum()
                )

                row[f"percentage_within_{window}_minutes"] = (
                    round(
                        activity_data[column].mean() * 100, 2
                    )
                    if len(activity_data) > 0
                    else 0.0
                )

            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)

    print("\n========== SCENARIO 1 DEVICE-FILE CORRELATION ==========")

    print(f"Employee: {EMPLOYEE}")
    print(f"File events analysed: {len(results)}")

    print("\nDevice events available:")
    print(results["nearest_device_activity"].value_counts().to_string())

    print("\nPeriod-wise results:")

    for period in ["PRE", "INCIDENT", "POST"]:

        subset = results[results["period"] == period]

        print(f"\n{period}")
        print(f"File events: {len(subset)}")

        for window in TIME_WINDOWS_MINUTES:
            count = int(subset[f"within_{window}_minutes"].sum())

            percentage = (
                subset[f"within_{window}_minutes"].mean() * 100
                if len(subset) > 0
                else 0
            )

            print(
                f"Within {window:2} minutes: "
                f"{count}/{len(subset)} ({percentage:.2f}%)"
            )

    print("\nActivity-wise summary:")
    print(summary.to_string(index=False))

    print("\nGenerated files:")
    print(detail_path)
    print(summary_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading ACM2278 event data...")

    files, devices = load_data()

    print(f"File events loaded: {len(files)}")
    print(f"Device events loaded: {len(devices)}")

    results = correlate_events(files, devices)

    generate_reports(results)


if __name__ == "__main__":
    main()
