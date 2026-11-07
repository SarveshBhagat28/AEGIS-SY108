
import pandas as pd

df = pd.read_csv("data/unified_activity_features.csv")
df["calendar_date"] = pd.to_datetime(df["calendar_date"])

data = df[
    (df["user"] == "CMP2946")
    & (df["calendar_date"] >= "2011-01-01")
    & (df["calendar_date"] < "2011-03-05")
    & (df["file_day_covered"] == True)
    & (df["device_day_covered"] == True)
].copy()

data["period"] = data["calendar_date"].apply(
    lambda d: "Historical"
    if d < pd.Timestamp("2011-02-07")
    else "Incident"
)

data["removable_total"] = (
    data["to_removable_events"]
    + data["from_removable_events"]
)

data["file_modification_total"] = (
    data["file_copies"]
    + data["file_writes"]
)

metrics = [
    "removable_total",
    "file_modification_total",
    "file_deletes"
]

historical = data[data["period"] == "Historical"]
incident = data[data["period"] == "Incident"]

thresholds = historical[metrics].quantile(0.95)

print("HISTORICAL 95TH-PERCENTILE THRESHOLDS")
print(thresholds)

print("\nINCIDENT DAYS EXCEEDING HISTORICAL THRESHOLDS")

for metric in metrics:
    threshold = thresholds[metric]
    matches = incident[incident[metric] > threshold]

    print(f"\n{metric}")
    print("Threshold:", threshold)
    print("Days exceeding:", len(matches), "/", len(incident))

    print(
        matches[["calendar_date", metric]]
        .to_string(index=False)
    )

print("\nDAYS EXCEEDING BOTH REMOVABLE AND FILE MODIFICATION THRESHOLDS")

combined = incident[
    (incident["removable_total"] > thresholds["removable_total"])
    & (
        incident["file_modification_total"]
        > thresholds["file_modification_total"]
    )
]

print(combined[
    ["calendar_date", "removable_total", "file_modification_total"]
].to_string(index=False))

