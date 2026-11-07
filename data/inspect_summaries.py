import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

files = [
    "employee_activity_summary.csv",
    "daily_activity_summary.csv",
    "device_daily_summary.csv",
    "logon_daily_summary.csv",
    "file_daily_summary.csv"
]

for filename in files:

    path = BASE_DIR / filename

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())