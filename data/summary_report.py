import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent

files = [
    "employee_activity_summary.csv",
    "logon_daily_summary.csv",
    "device_daily_summary.csv",
    "file_daily_summary.csv",
    "daily_activity_summary.csv"
]

for name in files:
    path = base / name

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    if not path.exists():
        print("NOT FOUND")
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values:", df.isnull().sum().sum())
    print(df.head(3).to_string(index=False))
