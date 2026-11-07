
import pandas as pd
from detection.removable_media_baseline import build_removable_baseline

df = pd.read_csv("data/unified_activity_features.csv")
df["calendar_date"] = pd.to_datetime(df["calendar_date"])

employee_data = df[df["user"] == "CMP2946"].copy()

history = employee_data[
    (employee_data["calendar_date"] >= "2011-01-08")
    & (employee_data["calendar_date"] < "2011-02-07")
]

baseline = build_removable_baseline(history)

print("FROZEN BASELINE")
print(baseline)

incident_days = employee_data[
    (employee_data["calendar_date"] >= "2011-02-07")
    & (employee_data["calendar_date"] < "2011-03-05")
    & (employee_data["file_day_covered"] == True)
].sort_values("calendar_date")

print("\nINCIDENT RESULTS")

for _, row in incident_days.iterrows():
    findings = []

    for metric in ["to_removable_events", "from_removable_events"]:
        profile = baseline[metric]
        value = row[metric]

        if profile["available"] and value > profile["threshold"]:
            findings.append(metric)

    status = "ANOMALOUS" if findings else "NORMAL"

    print(
        row["calendar_date"].date(),
        "Outbound:", row["to_removable_events"],
        "Inbound:", row["from_removable_events"],
        "Status:", status,
        "Findings:", findings
    )
