
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
TARGET_PC = "PC-8431"

INCIDENT_START = pd.Timestamp("2010-08-18 21:47:42")
INCIDENT_END = pd.Timestamp("2010-08-24 03:48:51")

POST_DISCONNECT_MINUTES = 5
PRE_CONNECT_MINUTES = 5


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():

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
            f"Missing file columns: "
            f"{required_file_columns - set(files.columns)}"
        )

    if not required_device_columns.issubset(devices.columns):
        raise ValueError(
            f"Missing device columns: "
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
        & (files["pc"] == TARGET_PC)
        & files["event_timestamp"].notna()
    ].copy()

    devices = devices[
        (devices["user"] == EMPLOYEE)
        & (devices["pc"] == TARGET_PC)
        & devices["activity"].isin(["Connect", "Disconnect"])
        & devices["timestamp"].notna()
    ].copy()

    files = files.sort_values("event_timestamp").reset_index(drop=True)
    devices = devices.sort_values("timestamp").reset_index(drop=True)

    if devices.empty:
        raise ValueError("No device events found for the target employee and PC.")

    return files, devices


# ============================================================
# BUILD DEVICE CONNECTION INTERVALS
# ============================================================

def build_connection_intervals(devices):

    intervals = []
    connected = False
    connection_start = None

    for _, event in devices.iterrows():

        timestamp = event["timestamp"]
        activity = event["activity"]

        if activity == "Connect":

            if connected:
                print(
                    f"WARNING: Duplicate Connect at {timestamp}; "
                    "closing previous interval at this timestamp."
                )

                intervals.append({
                    "connect_time": connection_start,
                    "disconnect_time": timestamp,
                    "interval_status": "CLOSED_BY_NEXT_CONNECT"
                })

            connection_start = timestamp
            connected = True

        elif activity == "Disconnect":

            if connected:
                intervals.append({
                    "connect_time": connection_start,
                    "disconnect_time": timestamp,
                    "interval_status": "CLOSED"
                })

                connection_start = None
                connected = False

            else:
                print(
                    f"WARNING: Disconnect without preceding Connect: "
                    f"{timestamp}"
                )

    if connected:
        intervals.append({
            "connect_time": connection_start,
            "disconnect_time": pd.NaT,
            "interval_status": "OPEN_AT_END_OF_DATA"
        })

    return pd.DataFrame(intervals)


# ============================================================
# CLASSIFY FILE EVENTS
# ============================================================

def classify_period(timestamp):

    if INCIDENT_START <= timestamp <= INCIDENT_END:
        return "INCIDENT"

    if timestamp < INCIDENT_START:
        return "PRE"

    return "POST"


def classify_file_event(timestamp, intervals):

    for _, interval in intervals.iterrows():

        connect_time = interval["connect_time"]
        disconnect_time = interval["disconnect_time"]

        # File operation occurred while device was connected.
        if timestamp >= connect_time and (
            pd.isna(disconnect_time) or timestamp <= disconnect_time
        ):
            return "DURING_CONNECTION", 0.0

        # File operation occurred shortly after disconnection.
        if pd.notna(disconnect_time):

            after_disconnect = (
                timestamp - disconnect_time
            ).total_seconds() / 60

            if 0 < after_disconnect <= POST_DISCONNECT_MINUTES:
                return "AFTER_DISCONNECT", after_disconnect

        # File operation occurred shortly before connection.
        before_connect = (
            connect_time - timestamp
        ).total_seconds() / 60

        if 0 < before_connect <= PRE_CONNECT_MINUTES:
            return "BEFORE_CONNECT", before_connect

    return "OUTSIDE_CONNECTION_WINDOW", None


# ============================================================
# ANALYSIS
# ============================================================

def analyse(files, intervals):

    results = []

    for _, event in files.iterrows():

        timestamp = event["event_timestamp"]

        classification, difference = classify_file_event(
            timestamp, intervals
        )

        results.append({
            "user": EMPLOYEE,
            "pc": event["pc"],
            "file_timestamp": timestamp,
            "period": classify_period(timestamp),
            "file_activity": event["activity"],
            "connection_classification": classification,
            "minutes_from_connection_boundary": difference
        })

    return pd.DataFrame(results)


# ============================================================
# REPORTS
# ============================================================

def generate_reports(intervals, results):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    intervals_path = (
        OUTPUT_DIR / "scenario1_device_connection_intervals.csv"
    )

    details_path = (
        OUTPUT_DIR / "scenario1_connection_state_details.csv"
    )

    summary_path = (
        OUTPUT_DIR / "scenario1_connection_state_summary.csv"
    )

    intervals.to_csv(intervals_path, index=False)
    results.to_csv(details_path, index=False)

    summary = (
        results.groupby(
            ["period", "connection_classification", "file_activity"],
            dropna=False
        )
        .size()
        .reset_index(name="file_event_count")
    )

    summary.to_csv(summary_path, index=False)

    print("\n========== DEVICE CONNECTION INTERVALS ==========")

    print(f"Total intervals: {len(intervals)}")

    if not intervals.empty:
        print(intervals.to_string(index=False))

    print("\n========== FILE EVENT CLASSIFICATION ==========")

    print(
        results.groupby(
            ["period", "connection_classification"]
        ).size().to_string()
    )

    print("\n========== ACTIVITY BREAKDOWN ==========")

    print(summary.to_string(index=False))

    print("\n========== INCIDENT ANALYSIS ==========")

    incident = results[results["period"] == "INCIDENT"]

    total = len(incident)

    during = (
        incident["connection_classification"] == "DURING_CONNECTION"
    ).sum()

    after = (
        incident["connection_classification"] == "AFTER_DISCONNECT"
    ).sum()

    before = (
        incident["connection_classification"] == "BEFORE_CONNECT"
    ).sum()

    outside = (
        incident["connection_classification"] == "OUTSIDE_CONNECTION_WINDOW"
    ).sum()

    print(f"Incident file events: {total}")
    print(f"During connection: {during}")
    print(f"Within 5 minutes after disconnect: {after}")
    print(f"Within 5 minutes before connect: {before}")
    print(f"Outside connection windows: {outside}")

    if total:
        print(f"During connection percentage: {during / total * 100:.2f}%")

    print("\nGenerated reports:")
    print(intervals_path)
    print(details_path)
    print(summary_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading ACM2278 data...")

    files, devices = load_data()

    print(f"File events: {len(files)}")
    print(f"Device events: {len(devices)}")

    intervals = build_connection_intervals(devices)

    results = analyse(files, intervals)

    generate_reports(intervals, results)


if __name__ == "__main__":
    main()
