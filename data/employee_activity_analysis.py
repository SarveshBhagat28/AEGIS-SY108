import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def analyse_activity(filename, user_column):

    file_path = BASE_DIR / filename

    print("\n" + "=" * 70)
    print(f"EMPLOYEE ACTIVITY ANALYSIS: {filename}")
    print("=" * 70)

    activity_counts = {}

    for chunk in pd.read_csv(
        file_path,
        usecols=[user_column],
        chunksize=100000
    ):

        counts = chunk[user_column].value_counts()

        for user, count in counts.items():
            activity_counts[user] = (
                activity_counts.get(user, 0) + count
            )

    activity = pd.Series(activity_counts, dtype="int64")

    print("\nTotal Employees:", len(activity))
    print("Total Events:", activity.sum())

    print("\nActivity Statistics:")
    print(activity.describe(percentiles=[
        0.25, 0.50, 0.75, 0.90, 0.95, 0.99
    ]))

    print("\nTop 10 Most Active Employees:")
    print(activity.nlargest(10))

    print("\nBottom 10 Least Active Employees:")
    print(activity.nsmallest(10))

    return activity


logon_activity = analyse_activity(
    "logon.csv", "user"
)

file_activity = analyse_activity(
    "file.csv", "user"
)

device_activity = analyse_activity(
    "device.csv", "user"
)

# Save results for future analysis

results = pd.DataFrame({
    "logon_events": logon_activity,
    "file_events": file_activity,
    "device_events": device_activity
})

results.index.name = "user"

results.to_csv(BASE_DIR / "employee_activity_summary.csv")

print("\nSaved employee_activity_summary.csv")