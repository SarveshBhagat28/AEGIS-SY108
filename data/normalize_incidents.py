import csv
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ANSWERS_DIR = BASE_DIR / "answers"

EXPECTED_FIELDS = {
    "logon": 6,
    "device": 7,
    "file": 11,
    "email": 14,
    "http": 9
}

events = []
errors = []

for scenario in range(1, 6):

    path = ANSWERS_DIR / f"r6.2-{scenario}.csv"

    with path.open("r", encoding="utf-8-sig", newline="") as f:

        reader = csv.reader(f)

        for line_number, row in enumerate(reader, start=1):

            if not row:
                continue

            event_type = row[0].strip()

            if event_type not in EXPECTED_FIELDS:
                errors.append((scenario, line_number, "Unknown event type"))
                continue

            if len(row) != EXPECTED_FIELDS[event_type]:
                errors.append((scenario, line_number, "Incorrect field count"))
                continue

            try:
                timestamp = pd.to_datetime(
                    row[2],
                    format="%m/%d/%Y %H:%M:%S",
                    errors="raise"
                )
            except ValueError:
                errors.append((scenario, line_number, "Invalid timestamp"))
                continue

            events.append({
                "scenario": scenario,
                "event_id": row[1],
                "timestamp": timestamp,
                "event_type": event_type,
                "user": row[3],
                "pc": row[4],
                "raw_fields": json.dumps(row, ensure_ascii=False)
            })

if errors:
    print("Errors detected:")
    for error in errors:
        print(error)

    raise ValueError(
        f"Normalization stopped: {len(errors)} invalid records."
    )

df = pd.DataFrame(events)

df = df.sort_values(
    ["timestamp", "scenario", "event_id"]
).reset_index(drop=True)

output_path = BASE_DIR / "answers" / "normalized_incidents.csv"

df.to_csv(output_path, index=False)

print("\nNormalization completed.")
print("Total events:", len(df))
print("\nEvents by scenario:")
print(df.groupby("scenario").size())

print("\nEvents by type:")
print(df["event_type"].value_counts())

print("\nUnique users:", df["user"].nunique())
print("Unique PCs:", df["pc"].nunique())

print("\nSaved to:", output_path)