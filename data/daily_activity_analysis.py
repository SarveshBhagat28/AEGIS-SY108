
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def analyse_daily_activity(filename, usecols):

    file_path = BASE_DIR / filename

    print("\n" + "=" * 75)
    print(f"DAILY ACTIVITY ANALYSIS: {filename}")
    print("=" * 75)

    daily_counts = {}

    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=100000
    ):

        chunk["date"] = pd.to_datetime(
            chunk["date"],
            format="%m/%d/%Y %H:%M:%S",
            errors="coerce"
        )

        chunk = chunk.dropna(subset=["date"])

        chunk["day"] = chunk["date"].dt.date

        grouped = (
            chunk.groupby(["user", "day"])
            .size()
        )

        for (user, day), count in grouped.items():
            key = (user, day)
            daily_counts[key] = daily_counts.get(key, 0) + count

    daily_df = pd.Series(daily_counts, name="events").reset_index()

    daily_df.columns = ["user", "day", "events"]

    employee_summary = daily_df.groupby("user")["events"].agg(
        active_days="count",
        total_events="sum",
        average_daily_events="mean",
        median_daily_events="median",
        maximum_daily_events="max"
    )

    print("\nEmployee Daily Activity Statistics:")
    print(employee_summary.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ))

    print("\nTop 10 Employees by Maximum Daily Activity:")
    print(
        employee_summary.nlargest(
            10, "maximum_daily_events"
        )
    )

    print("\nEmployees with Fewest Active Days:")
    print(
        employee_summary.nsmallest(
            10, "active_days"
        )
    )

    output_file = BASE_DIR / (
        filename.replace(".csv", "_daily_summary.csv")
    )

    employee_summary.to_csv(output_file)

    print(f"\nSaved: {output_file.name}")

    return employee_summary


analyse_daily_activity(
    "logon.csv",
    ["user", "date"]
)

analyse_daily_activity(
    "file.csv",
    ["user", "date"]
)

analyse_daily_activity(
    "device.csv",
    ["user", "date"]
)
