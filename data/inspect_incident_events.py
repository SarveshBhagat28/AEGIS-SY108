import csv
from pathlib import Path
from collections import Counter

base = Path(__file__).resolve().parent
answers = base / "answers"

for scenario in range(1, 6):
    path = answers / f"r6.2-{scenario}.csv"

    print(f"\n{'='*60}")
    print(f"SCENARIO {scenario}: {path.name}")
    print('='*60)

    counts = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            counts[row[0].strip()] += 1

    print("Event-type counts:", counts)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            if i >= 5:
                break
            print(f"Record {i+1} ({len(row)} fields):")
            print(row)
