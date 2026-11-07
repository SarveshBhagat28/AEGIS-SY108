import pandas as pd
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent

datasets = {
    "logon.csv": "user",
    "file.csv": "user",
    "device.csv": "user",
    "psychometric.csv": "user_id",
    "users.csv": "user_id"
}

for filename, user_column in datasets.items():

    file_path = BASE_DIR / filename

    print("\n" + "=" * 65)
    print(f"ANALYSING: {filename}")
    print("=" * 65)

    total_rows = 0
    unique_users = set()
    missing_values = Counter()
    activities = Counter()

    min_date = None
    max_date = None

    # Read large datasets in smaller chunks
    for chunk in pd.read_csv(
        file_path,
        chunksize=100000,
        usecols=lambda col: col != "content"
    ):

        total_rows += len(chunk)

        # Unique employees
        if user_column in chunk.columns:
            unique_users.update(
                chunk[user_column].dropna().unique()
            )

        # Missing values
        missing_values.update(
            chunk.isnull().sum().to_dict()
        )

        # Activity distribution
        if "activity" in chunk.columns:
            activities.update(
                chunk["activity"].dropna().value_counts().to_dict()
            )

        # Date range
        if "date" in chunk.columns:
            dates = pd.to_datetime(
                chunk["date"],
                errors="coerce"
            )

            if dates.notna().any():

                chunk_min = dates.min()
                chunk_max = dates.max()

                min_date = (
                    chunk_min if min_date is None
                    else min(min_date, chunk_min)
                )

                max_date = (
                    chunk_max if max_date is None
                    else max(max_date, chunk_max)
                )

    print(f"Total Records: {total_rows:,}")
    print(f"Unique Employees: {len(unique_users):,}")

    print(f"Date Range: {min_date} to {max_date}")

    print("\nActivity Distribution:")
    for activity, count in activities.most_common():
        print(f"{activity}: {count:,}")

    print("\nMissing Values:")

    for column, count in missing_values.items():
        if count > 0:
            print(f"{column}: {count:,}")

    print("\nAnalysis Completed!")