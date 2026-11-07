import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.baseline_engine import (
    prepare_daily_data,
    build_employee_baseline,
)

DATA_FILE = PROJECT_ROOT / "data" / "daily_activity_features.csv"


# ---------------------------------------------------------
# MAIN VALIDATION
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("AEGIS - PHASE 2 STATISTICAL VALIDATION")
    print("=" * 65)

    # 1. LOAD DATASET
    print("\n[1] Loading dataset...")

    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found: {DATA_FILE}")
        return

    raw_data = pd.read_csv(DATA_FILE, low_memory=False)
    daily_data = prepare_daily_data(raw_data)

    print(f"Records: {len(daily_data):,}")
    print(f"Employees: {daily_data['user'].nunique():,}")
    print(
        f"Date range: {daily_data['calendar_date'].min().date()} "
        f"to {daily_data['calendar_date'].max().date()}"
    )

    # 2. DATA QUALITY
    print("\n[2] Data quality checks...")

    duplicate_count = daily_data.duplicated(
        ["user", "calendar_date"]
    ).sum()

    missing_users = daily_data["user"].isna().sum()
    missing_dates = daily_data["calendar_date"].isna().sum()
    login_active_days = daily_data["has_logon_activity"].sum()

    print(f"Duplicate employee-days: {duplicate_count:,}")
    print(f"Missing employee IDs: {missing_users:,}")
    print(f"Missing dates: {missing_dates:,}")
    print(f"Login-active days: {login_active_days:,}")

    # 3. EMPLOYEE BASELINE AVAILABILITY
    print("\n[3] Employee baseline availability...")

    target_date = daily_data["calendar_date"].max()
    employees = daily_data["user"].dropna().unique()

    results = []

    for index, employee in enumerate(employees, start=1):

        baseline = build_employee_baseline(
            daily_data,
            employee,
            target_date
        )

        frequency = baseline["login_frequency"]
        timing = baseline["login_timing"]
        computer = baseline["computer_usage"]

        results.append({
            "employee": employee,

            "historical_days": baseline[
                "historical_employee_days"
            ],

            "frequency_available": frequency[
                "baseline_available"
            ],

            "timing_available": timing[
                "baseline_available"
            ],

            "computer_available": computer[
                "baseline_available"
            ],

            "frequency_observations": frequency[
                "observation_count"
            ],

            "timing_observations": timing[
                "observation_count"
            ],

            "computer_observations": computer[
                "observation_count"
            ],
        })

        if index % 500 == 0:
            print(
                f"Processed {index:,}/{len(employees):,} employees"
            )

    results_df = pd.DataFrame(results)

    # 4. BASELINE AVAILABILITY SUMMARY
    print("\n[4] Baseline availability summary...")

    total = len(results_df)

    baseline_columns = [
        "frequency_available",
        "timing_available",
        "computer_available"
    ]

    for column in baseline_columns:

        available = int(results_df[column].sum())
        unavailable = total - available

        percentage = (
            available / total * 100
            if total else 0
        )

        print(f"\n{column}")
        print(f"Available: {available:,}")
        print(f"Unavailable: {unavailable:,}")
        print(f"Availability: {percentage:.2f}%")

    # 5. HISTORICAL OBSERVATION DISTRIBUTION
    print("\n[5] Historical observation distribution...")

    print(
        results_df["historical_days"].describe().to_string()
    )

    # 6. OBSERVATION COUNTS
    print("\n[6] Baseline observation counts...")

    for column in [
        "frequency_observations",
        "timing_observations",
        "computer_observations"
    ]:
        print(f"\n{column}")
        print(results_df[column].describe().to_string())

    # 7. EXPORT RESULTS
    print("\n[7] Exporting results...")

    output_directory = (
        PROJECT_ROOT / "testing" / "reports"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_directory /
        "phase2_baseline_availability.csv"
    )

    results_df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")

    # FINAL SUMMARY
    print("\n" + "=" * 65)
    print("PHASE 2 VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Employees evaluated: {total:,}")
    print(f"Duplicate employee-days: {duplicate_count:,}")
    print(f"Missing employee IDs: {missing_users:,}")
    print(f"Missing dates: {missing_dates:,}")

    print("\nStatistical availability audit completed.")
    print("This is not a threat-detection performance evaluation.")

    print("=" * 65)


if __name__ == "__main__":
    main()