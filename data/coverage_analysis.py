import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

datasets = {
    "logon.csv": "user",
    "file.csv": "user",
    "device.csv": "user",
    "psychometric.csv": "user_id",
    "users.csv": "user_id"
}

employee_sets = {}

# Extract unique employee IDs
for filename, column in datasets.items():

    file_path = BASE_DIR / filename

    print(f"Processing {filename}...")

    unique_ids = set()

    for chunk in pd.read_csv(
        file_path,
        usecols=[column],
        chunksize=100000
    ):

        unique_ids.update(
            chunk[column].dropna().unique()
        )

    employee_sets[filename] = unique_ids

# Coverage summary
print("\n" + "=" * 65)
print("EMPLOYEE COVERAGE ANALYSIS")
print("=" * 65)

for filename, employees in employee_sets.items():

    print(f"{filename}: {len(employees):,} employees")

# Intersection analysis
logon = employee_sets["logon.csv"]
files = employee_sets["file.csv"]
devices = employee_sets["device.csv"]
psychometric = employee_sets["psychometric.csv"]
users = employee_sets["users.csv"]

print("\n" + "=" * 65)
print("DATASET OVERLAP")
print("=" * 65)

print("Logon + File:", len(logon & files))

print("Logon + Device:", len(logon & devices))

print("Logon + File + Device:",
      len(logon & files & devices))

print("Logon + Psychometric:",
      len(logon & psychometric))

print("Logon + Users:",
      len(logon & users))

print("Present in all five datasets:",
      len(logon & files & devices & psychometric & users))

print("\n" + "=" * 65)
print("EMPLOYEES WITHOUT DEVICE RECORDS")
print("=" * 65)

without_device = logon - devices

print("Employees without device records:",
      len(without_device))

print("Sample IDs:", list(without_device)[:20])

print("\nAnalysis completed!")