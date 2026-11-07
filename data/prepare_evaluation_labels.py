import pandas as pd
from pathlib import Path

# ==========================================================
# AEGIS - GROUND TRUTH EVALUATION LABEL PREPARATION
# ==========================================================

BASE_DIR = Path(r"D:\Programs\AEGIS 2\data")

DAILY_FILE = BASE_DIR / "daily_activity_summary.csv"

OUTPUT_DAILY = BASE_DIR / "evaluation_daily_labels.csv"
OUTPUT_SUMMARY = BASE_DIR / "evaluation_employee_summary.csv"

# Known insider incidents from CERT ground truth.
# Scenario 4 start date is corrected using the first evidence
# timestamp because the original insiders.csv value is malformed.

INCIDENTS = [
    {
        "scenario": 1,
        "user": "ACM2278",
        "start": "2010-08-18",
        "end": "2010-08-24"
    },
    {
        "scenario": 2,
        "user": "CMP2946",
        "start": "2011-02-07",
        "end": "2011-03-04"
    },
    {
        "scenario": 3,
        "user": "PLJ1771",
        "start": "2010-08-12",
        "end": "2010-08-12"
    },
    {
        "scenario": 4,
        "user": "CDE1846",
        "start": "2011-02-21",
        "end": "2011-04-25"
    },
    {
        "scenario": 5,
        "user": "MBG3183",
        "start": "2010-10-12",
        "end": "2010-10-12"
    }
]

print("=" * 70)
print("AEGIS - EVALUATION LABEL PREPARATION")
print("=" * 70)

# ----------------------------------------------------------
# 1. LOAD DAILY ACTIVITY DATA
# ----------------------------------------------------------

print("\n[1] Loading daily activity dataset...")

df = pd.read_csv(DAILY_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

if df["date"].isna().any():
    raise ValueError("Invalid dates found in daily activity dataset.")

if df.duplicated(["user", "date"]).any():
    raise ValueError("Duplicate user-date records found.")

print("Rows:", len(df))
print("Employees:", df["user"].nunique())

# ----------------------------------------------------------
# 2. DETERMINE EMPLOYEE-LEVEL TELEMETRY COVERAGE
# ----------------------------------------------------------

print("\n[2] Determining telemetry coverage...")

coverage_columns = [
    "logon_coverage",
    "file_coverage",
    "device_coverage"
]

for column in coverage_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

coverage = (
    df.groupby("user")[coverage_columns]
    .max()
    .reset_index()
)

# ----------------------------------------------------------
# 3. GENERATE CALENDAR-DAY LABELS
# ----------------------------------------------------------

print("\n[3] Generating incident evaluation dates...")

label_frames = []

for incident in INCIDENTS:

    dates = pd.date_range(
        start=incident["start"],
        end=incident["end"],
        freq="D"
    )

    incident_df = pd.DataFrame({
        "user": incident["user"],
        "date": dates,
        "scenario": incident["scenario"],
        "known_insider": 1
    })

    incident_df["incident_start"] = pd.Timestamp(
        incident["start"]
    )

    incident_df["incident_end"] = pd.Timestamp(
        incident["end"]
    )

    label_frames.append(incident_df)

labels = pd.concat(label_frames, ignore_index=True)

# ----------------------------------------------------------
# 4. MERGE WITH DAILY ACTIVITY
# ----------------------------------------------------------

print("\n[4] Matching incident dates with recorded activity...")

labels = labels.merge(
    df,
    on=["user", "date"],
    how="left",
    indicator=True
)

labels["daily_activity_record_present"] = (
    labels["_merge"] == "both"
).astype(int)

labels.drop(columns=["_merge"], inplace=True)

# Attach employee-level coverage even on dates without
# a corresponding daily activity record.

labels = labels.merge(
    coverage,
    on="user",
    how="left",
    suffixes=("", "_employee")
)

for column in coverage_columns:
    employee_column = f"{column}_employee"

    if employee_column in labels.columns:
        labels[column] = labels[employee_column]
        labels.drop(columns=[employee_column], inplace=True)

# ----------------------------------------------------------
# 5. VALIDATE RESULTS
# ----------------------------------------------------------

print("\n[5] Validating evaluation labels...")

expected_users = {item["user"] for item in INCIDENTS}
actual_users = set(labels["user"].unique())

if expected_users != actual_users:
    raise ValueError(
        f"Insider mismatch. Missing: {expected_users - actual_users}"
    )

if labels.duplicated(["user", "date"]).any():
    raise ValueError("Duplicate evaluation dates detected.")

if labels["known_insider"].isna().any():
    raise ValueError("Missing ground-truth labels.")

# ----------------------------------------------------------
# 6. EMPLOYEE-LEVEL SUMMARY
# ----------------------------------------------------------

summary = labels.groupby(
    ["scenario", "user"],
    as_index=False
).agg(
    incident_start=("incident_start", "first"),
    incident_end=("incident_end", "first"),
    calendar_days=("date", "count"),
    recorded_activity_days=(
        "daily_activity_record_present", "sum"
    ),
    logon_coverage=("logon_coverage", "max"),
    file_coverage=("file_coverage", "max"),
    device_coverage=("device_coverage", "max")
)

summary["days_without_activity_record"] = (
    summary["calendar_days"] -
    summary["recorded_activity_days"]
)

# ----------------------------------------------------------
# 7. SAVE OUTPUTS
# ----------------------------------------------------------

labels.to_csv(OUTPUT_DAILY, index=False)
summary.to_csv(OUTPUT_SUMMARY, index=False)

print("\n" + "=" * 70)
print("EVALUATION LABEL SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

print("\nTotal evaluation dates:", len(labels))

print(
    "Dates with recorded activity:",
    labels["daily_activity_record_present"].sum()
)

print(
    "Dates without activity records:",
    (labels["daily_activity_record_present"] == 0).sum()
)

print("\nTelemetry coverage:")

print(
    summary[
        ["user", "logon_coverage", "file_coverage", "device_coverage"]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("EVALUATION LABEL PREPARATION COMPLETED")
print("=" * 70)

print("\nSaved files:")
print(OUTPUT_DAILY)
print(OUTPUT_SUMMARY)