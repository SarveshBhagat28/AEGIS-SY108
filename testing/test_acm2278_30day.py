
import pandas as pd

df = pd.read_csv("data/unified_activity_features.csv")
df["calendar_date"] = pd.to_datetime(df["calendar_date"])

data = df[
    (df["user"] == "ACM2278")
    & (df["calendar_date"] >= "2010-07-19")
    & (df["calendar_date"] <= "2010-08-24")
    & (df["file_day_covered"] == True)
].copy()

data["removable_total"] = (
    data["to_removable_events"] + data["from_removable_events"]
)

data["file_modification_total"] = (
    data["file_copies"] + data["file_writes"]
)

historical = data[data["calendar_date"] < "2010-08-18"]
incident = data[data["calendar_date"] >= "2010-08-18"]

metrics = [
    "removable_total",
    "file_modification_total",
    "file_deletes"
]

print("COVERED DAYS")
print("Historical:", len(historical))
print("Incident:", len(incident))

thresholds = historical[metrics].quantile(0.95)

print("\nHISTORICAL 95TH-PERCENTILE THRESHOLDS")
print(thresholds)

print("\nINCIDENT SUMMARY")
print(incident[metrics].agg(["mean", "median", "max"]).round(2))

print("\nINCIDENT DAYS EXCEEDING THRESHOLDS")

for metric in metrics:
    matches = incident[incident[metric] > thresholds[metric]]
    print(f"\n{metric}: {len(matches)}/{len(incident)}")
    print(matches[["calendar_date", metric]].to_string(index=False))

combined = incident[
    (incident["removable_total"] > thresholds["removable_total"])
    & (
        incident["file_modification_total"]
        > thresholds["file_modification_total"]
    )
]

print("\nDAYS EXCEEDING BOTH THRESHOLDS")
print(combined[
    ["calendar_date", "removable_total", "file_modification_total"]
].to_string(index=False))
