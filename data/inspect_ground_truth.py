import pandas as pd
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ANSWER_DIR = BASE_DIR / "answers"

# --------------------------------------------------
# 1. Inspect insiders.csv
# --------------------------------------------------

print("\n" + "=" * 80)
print("INSIDER INCIDENT REFERENCE")
print("=" * 80)

insiders = pd.read_csv(ANSWER_DIR / "insiders.csv")

# Convert dataset version into numeric form
insiders["dataset"] = pd.to_numeric(
    insiders["dataset"],
    errors="coerce"
)

# Select only version 6.2
insiders_62 = insiders[
    insiders["dataset"] == 6.2
].copy()

print("\nDataset 6.2 records:")
print(insiders_62.to_string(index=False))

print("\nTotal dataset 6.2 incidents:", len(insiders_62))

print("\nUnique insider employees:")
print(insiders_62["user"].nunique())

# --------------------------------------------------
# 2. Inspect raw answer CSV structures
# --------------------------------------------------

print("\n" + "=" * 80)
print("RAW ANSWER FILE INSPECTION")
print("=" * 80)

for file in sorted(ANSWER_DIR.glob("r6.2-*.csv")):

    print("\n" + "-" * 80)
    print("FILE:", file.name)

    row_lengths = {}

    with open(
        file,
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.reader(f)

        for row_number, row in enumerate(reader, start=1):

            length = len(row)

            row_lengths[length] = (
                row_lengths.get(length, 0) + 1
            )

            if row_number <= 3:
                print(f"\nRow {row_number}:")
                for index, value in enumerate(row):
                    print(f"  Column {index}: {value[:150]}")

    print("\nNumber of rows with each field count:")
    print(row_lengths)