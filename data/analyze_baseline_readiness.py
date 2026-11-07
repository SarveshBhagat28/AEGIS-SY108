import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path(__file__).resolve().parent

INPUT_FILE = DATA_DIR / "daily_activity_summary.csv"
OUTPUT_FILE = DATA_DIR / "baseline_readiness_report.csv"

print("=" * 70)
print("AEGIS - EMPLOYEE BASELINE READINESS ANALYSIS")
print("=" * 70)

# =========================================================
# 1. LOAD DATA
# =========================================================

print("\nLoading daily activity dataset...")

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

df = df.sort_values(["user", "date"]).reset_index(drop=True)

print(f"Records: {len(df):,}")
print(f"Employees: {df['user'].nunique():,}")


# =========================================================
# 2. CREATE DAILY ACTIVITY INDICATORS
# =========================================================

df["logon_active"] = (
    df["logon_events"] + df["logoff_events"] > 0
)

df["file_active"] = df["file_events"] > 0

df["device_active"] = df["device_events"] > 0

df["removable_active"] = (
    df["removable_media_events"] > 0
)


# =========================================================
# 3. EMPLOYEE-LEVEL BASELINE READINESS
# =========================================================

print("\nCalculating employee-level statistics...")

employee = df.groupby("user").agg(

    first_activity_date=("date", "min"),
    last_activity_date=("date", "max"),

    total_observed_days=("date", "nunique"),

    logon_active_days=("logon_active", "sum"),
    file_active_days=("file_active", "sum"),
    device_active_days=("device_active", "sum"),
    removable_active_days=("removable_active", "sum"),

    total_logon_events=("logon_events", "sum"),
    total_file_events=("file_events", "sum"),
    total_device_events=("device_events", "sum"),
    total_removable_events=("removable_media_events", "sum"),

    average_logon_events=("logon_events", "mean"),
    average_file_events=("file_events", "mean"),
    average_device_events=("device_events", "mean"),

    maximum_logon_events=("logon_events", "max"),
    maximum_file_events=("file_events", "max"),
    maximum_device_events=("device_events", "max"),

    unique_pcs_days=("unique_pcs", "sum"),

    logon_coverage=("logon_coverage", "max"),
    file_coverage=("file_coverage", "max"),
    device_coverage=("device_coverage", "max")

).reset_index()


# =========================================================
# 4. CALCULATE OBSERVATION PERIOD
# =========================================================

employee["observation_span_days"] = (
    employee["last_activity_date"]
    - employee["first_activity_date"]
).dt.days + 1


# =========================================================
# 5. CALCULATE ACTIVITY DENSITY
# =========================================================

employee["logon_activity_density"] = (
    employee["logon_active_days"]
    / employee["observation_span_days"]
)

employee["file_activity_density"] = (
    employee["file_active_days"]
    / employee["observation_span_days"]
)

employee["device_activity_density"] = (
    employee["device_active_days"]
    / employee["observation_span_days"]
)

employee["removable_activity_density"] = (
    employee["removable_active_days"]
    / employee["observation_span_days"]
)


# =========================================================
# 6. REMOVABLE MEDIA BEHAVIOUR
# =========================================================

employee["removable_event_ratio"] = np.where(
    employee["total_file_events"] > 0,
    employee["total_removable_events"]
    / employee["total_file_events"],
    np.nan
)


# =========================================================
# 7. PRINT DISTRIBUTION REPORT
# =========================================================

def distribution_report(column, title):

    print(f"\n{title}")
    print("-" * 60)

    print(
        employee[column].describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        ).round(3).to_string()
    )


distribution_report(
    "logon_active_days",
    "LOGON ACTIVE DAYS"
)

distribution_report(
    "file_active_days",
    "FILE ACTIVE DAYS"
)

distribution_report(
    "device_active_days",
    "DEVICE ACTIVE DAYS"
)

distribution_report(
    "removable_active_days",
    "REMOVABLE MEDIA ACTIVE DAYS"
)

distribution_report(
    "observation_span_days",
    "OBSERVATION SPAN"
)


# =========================================================
# 8. TELEMETRY COVERAGE REPORT
# =========================================================

print("\n" + "=" * 70)
print("TELEMETRY COVERAGE")
print("=" * 70)

for source in ["logon", "file", "device"]:

    column = f"{source}_coverage"

    covered = employee[column].sum()
    missing = len(employee) - covered

    print(f"\n{source.upper()}")
    print(f"Covered employees: {covered:,}")
    print(f"Employees without records: {missing:,}")


# =========================================================
# 9. REMOVABLE MEDIA REPORT
# =========================================================

print("\n" + "=" * 70)
print("REMOVABLE MEDIA ANALYSIS")
print("=" * 70)

print(
    "Employees with removable-media activity:",
    (employee["total_removable_events"] > 0).sum()
)

print(
    "Employees with no recorded removable-media activity:",
    (employee["total_removable_events"] == 0).sum()
)

print(
    "Total removable-media events:",
    f"{employee['total_removable_events'].sum():,}"
)

print(
    "Employees with removable activity on multiple days:",
    (employee["removable_active_days"] > 1).sum()
)


# =========================================================
# 10. SAVE REPORT
# =========================================================

employee.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("BASELINE READINESS ANALYSIS COMPLETED")
print("=" * 70)

print(f"Report saved to: {OUTPUT_FILE}")

print("\nFirst 10 employees:")
print(employee.head(10).to_string(index=False))