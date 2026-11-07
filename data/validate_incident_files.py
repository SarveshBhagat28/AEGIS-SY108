import csv
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent
ANSWERS_DIR = BASE_DIR / "answers"

EXPECTED_FIELDS = {
    "logon": 6,
    "device": 7,
    "file": 11,
    "email": 14,
    "http": 9
}

for scenario in range(1, 6):

    path = ANSWERS_DIR / f"r6.2-{scenario}.csv"

    counts = Counter()
    invalid = []
    timestamps = []
    users = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as f:

        reader = csv.reader(f)

        for line_number, row in enumerate(reader, start=1):

            if not row:
                continue

            event_type = row[0].strip()
            counts[event_type] += 1

            expected = EXPECTED_FIELDS.get(event_type)

            if expected is None:
                invalid.append((line_number, "Unknown event type", row))
                continue

            if len(row) != expected:
                invalid.append((
                    line_number,
                    f"Expected {expected} fields, got {len(row)}",
                    row
                ))
                continue

            try:
                from datetime import datetime
                timestamp = datetime.strptime(
                    row[2], "%m/%d/%Y %H:%M:%S"
                )
                timestamps.append(timestamp)
            except ValueError:
                invalid.append((
                    line_number,
                    "Invalid timestamp",
                    row
                ))

            users[row[3]] += 1

    print(f"\n{'='*55}")
    print(f"SCENARIO {scenario}")
    print('='*55)

    print("Event counts:", dict(counts))
    print("Total records:", sum(counts.values()))
    print("Unique users:", len(users))
    print("Users:", dict(users))

    if timestamps:
        print("First timestamp:", min(timestamps))
        print("Last timestamp:", max(timestamps))

    print("Invalid records:", len(invalid))

    for item in invalid[:5]:
        print("INVALID:", item)

