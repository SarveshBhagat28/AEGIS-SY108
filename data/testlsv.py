import pandas as pd
from pathlib import Path

# Automatically locate the script directory
BASE_DIR = Path(__file__).resolve().parent

files = [
    "logon.csv",
    "file.csv",
    "device.csv",
    "psychometric.csv",
    "users.csv"
]

for filename in files:

    file_path = BASE_DIR / filename

    try:
        df = pd.read_csv(file_path, nrows=10)

        print(f"\n{'='*60}")
        print(f"FILE: {filename}")
        print(f"{'='*60}")

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nFirst 10 Rows:")
        print(df.to_string(index=False))

        print(f"\nShape (loaded rows): {df.shape}")

    except FileNotFoundError:
        print(f"\n❌ File not found: {filename}")

    except Exception as e:
        print(f"\n❌ Error reading {filename}: {e}")