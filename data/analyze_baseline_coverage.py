import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

INPUT_FILE = DATA_DIR / "daily_activity_summary.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

print("=" * 70)
print("AEGIS - TELEMETRY COVERAGE AND BASELINE INVESTIGATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. EMPLOYEE-LEVEL STATISTICS
# ---------------------------------------------------------

employee = df.groupby("user").agg(
    first_date=("date", "min"),
    last_date=("date", "max"),
    active_recorded_days=("date", "nunique"),

    logon_active_days=("logon_events", lambda x: (x > 0).sum()),
    file_active_days=("file_events", lambda x: (x > 0).sum()),
    device_active_days=("device_events", lambda x: (x > 0).sum()),
    removable_active_days=(
        "removable_media_events",
        lambda x: (x > 0).sum()
    ),

    total_file_events=("file_events", "sum"),
    total_device_events=("device_events", "sum"),
    total_removable_events=("removable_media_events", "sum"),

    file_coverage=("file_coverage", "max"),
    device_coverage=("device_coverage", "max")

).reset_index()

employee["observation_span_days"] = (
    employee["last_date"] - employee["first_date"]
).dt.days + 1


# ---------------------------------------------------------
# 2. COVERAGE COUNTS
# ---------------------------------------------------------

print("\n1. TELEMETRY COVERAGE")

for source in ["file", "device"]:

    covered = employee[employee[f"{source}_coverage"]]

    print(f"\n{source.upper()}")
    print("Employees with records:", len(covered))
    print(
        "Employees without records:",
        len(employee) - len(covered)
    )


# ---------------------------------------------------------
# 3. CONDITIONAL ACTIVITY DISTRIBUTIONS
# ---------------------------------------------------------

metrics = {
    "logon": ("logon_active_days", employee),
    "file": (
        "file_active_days",
        employee[employee["file_coverage"]]
    ),
    "device": (
        "device_active_days",
        employee[employee["device_coverage"]]
    ),
    "removable_media": (
        "removable_active_days",
        employee[employee["file_coverage"]]
    )
}

print("\n2. ACTIVITY DISTRIBUTIONS")
print("Statistics are calculated within each relevant population.")

for name, (column, population) in metrics.items():

    print(f"\n{name.upper()}")
    print("-" * 55)

    if population.empty:
        print("No covered employees.")
        continue

    print(
        population[column].describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        ).round(2).to_string()
    )


# ---------------------------------------------------------
# 4. REMOVABLE MEDIA AND DEVICE RELATIONSHIP
# ---------------------------------------------------------

print("\n3. REMOVABLE MEDIA / DEVICE RELATIONSHIP")

cross = pd.crosstab(
    employee["device_coverage"],
    employee["total_removable_events"] > 0,
    rownames=["Device coverage"],
    colnames=["Removable activity"]
)

print(cross.to_string())

print(
    "\nEmployees with device records but no removable activity:",
    (
        employee["device_coverage"]
        & (employee["total_removable_events"] == 0)
    ).sum()
)

print(
    "Employees with removable activity but no device records:",
    (
        ~employee["device_coverage"]
        & (employee["total_removable_events"] > 0)
    ).sum()
)


# ---------------------------------------------------------
# 5. FILE ACTIVITY DENSITY
# ---------------------------------------------------------

covered_file = employee[employee["file_coverage"]].copy()

covered_file["file_activity_density"] = (
    covered_file["file_active_days"]
    / covered_file["observation_span_days"]
)

print("\n4. FILE ACTIVITY DENSITY - COVERED EMPLOYEES")

print(
    covered_file["file_activity_density"]
    .describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    .round(4)
    .to_string()
)


# ---------------------------------------------------------
# 6. SAVE CORRECTED REPORT
# ---------------------------------------------------------

employee.to_csv(
    DATA_DIR / "baseline_coverage_report.csv",
    index=False
)

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETED")
print("=" * 70)

print("Saved: baseline_coverage_report.csv")